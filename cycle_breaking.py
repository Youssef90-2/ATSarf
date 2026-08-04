"""
cycle_breaking.py
=================
IDEA (Break cycle, paper §2): a real isnad graph is a DAG — narration flows
one way in time, from the source forward. So a CYCLE (A narrated from B, ...,
back to A) means the merge step wrongly fused two different people into one
node. Find the bad merge and SPLIT it.

Old source (verified): breakManageableCycles() (graph.h:1136) +
LoopBreakingVisitor (graph.h:529) + reMergeNodes (graph.cpp:115).

    3 DFS passes with an ESCALATING threshold:
        threshold_i = equality_threshold + i * delta      (i = 1, 2, 3)
    On a cycle, every person ON the cycle is DISSOLVED into its GroupNodes
    and the groups are re-merged pairwise at the stricter threshold. Groups
    that no longer clear the bar come out as separate persons.

WHY DISSOLVE-AND-REMERGE, AND NOT "PEEL OFF ONE OCCURRENCE":
A GroupNode is atomic (identical key), so dissolving a person yields exactly
the pieces the evidence can distinguish. Re-merging at a stricter bar is a
decision the metric can justify, and the partition can only get finer — which
is what guarantees the loop terminates. Peeling occurrences off one at a time
(the previous implementation) could not undo a bad fusion and had no progress
guarantee.

WHAT WE NEVER DO: delete an edge. The previous version fell back to
`_break_weakest_edge`, removing a real عن relation from the data whenever a
split failed. The old system never does that — R is the evidence, and cycles
are resolved by re-partitioning nodes, not by discarding narrations. Cycles
that survive all three passes are REPORTED, not hidden.

Rationale for escalation: break the cycles caused by the WEAKEST merges
first, then re-check with progressively stricter standards. This conservatism
is exactly why the paper's recall dropped to 63% (§6) — a documented weak
point and our opportunity to improve.
"""

from equality import narrator_distance


class CycleBreaker:

    def __init__(self, persons, chain_nodes, base_threshold=0.1, delta=0.2,
                 rebuild_edges=None, equality_radius=3):
        self.persons = persons              # {id: GraphNarratorNode}
        self.chain_nodes = chain_nodes      # {id: ChainNarratorNode}
        self.base_threshold = base_threshold
        self.delta = delta
        self.equality_radius = equality_radius
        # callback that recomputes parents/children from the current
        # occurrence -> person mapping. Cycle detection needs an up-to-date
        # topology after every split, otherwise the same cycle is re-found
        # forever (and the split appears to have done nothing).
        self._rebuild_edges = rebuild_edges or (lambda: None)
        self._next_id = max(persons.keys()) + 1 if persons else 0
        self.stats = {"dissolved": 0, "splits": 0, "unresolvable": 0}

    # ------------------------------------------------------- cycle detection
    def _find_one_cycle(self):
        """
        Iterative DFS over the 'parents' edges. Returns the list of person ids
        forming a cycle, or None if the graph is acyclic.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {pid: WHITE for pid in self.persons}
        parent_of = {}

        for start in self.persons:
            if color[start] != WHITE:
                continue
            stack = [(start, iter(self.persons[start].parents))]
            color[start] = GRAY
            while stack:
                node, it = stack[-1]
                advanced = False
                for nxt in it:
                    if nxt not in self.persons:
                        continue
                    if color[nxt] == GRAY:
                        cycle = [nxt]
                        cur = node
                        while cur != nxt and cur in parent_of:
                            cycle.append(cur)
                            cur = parent_of[cur]
                        cycle.append(nxt)
                        return cycle
                    if color[nxt] == WHITE:
                        color[nxt] = GRAY
                        parent_of[nxt] = node
                        stack.append((nxt, iter(self.persons[nxt].parents)))
                        advanced = True
                        break
                if not advanced:
                    color[node] = BLACK
                    stack.pop()
        return None

    # ------------------------------------------------------- the Split
    def _dissolve(self, person_id, threshold) -> bool:
        """
        Port of reMergeNodes (graph.cpp:115-210).

        Take the person apart into its GroupNodes, then re-merge pairwise at
        `threshold`. Returns True if the person actually came apart into more
        than one node (i.e. real progress was made).
        """
        person = self.persons.get(person_id)
        if person is None:
            return False
        groups = list(person.groups)
        if len(groups) < 2:
            # a single-group person is indivisible -> "unable to resolve"
            # (old: `if (n->isChainNode()) return;`)
            self.stats["unresolvable"] += 1
            return False

        self.stats["dissolved"] += 1

        # --- detach: the person is dismantled, its groups become free ---
        self._detach_edges(person_id)
        del self.persons[person_id]
        for g in groups:
            g.person_id = None

        # --- re-merge pairwise at the stricter threshold (old double loop) ---
        from graph_nodes import GraphNarratorNode
        owner = {}                      # group index -> new person id
        acc = {}                        # person id -> [lowest, highest, chains]

        def new_person():
            pid = self._next_id
            self._next_id += 1
            self.persons[pid] = GraphNarratorNode(pid)
            acc[pid] = [10 ** 9, -10 ** 9, set()]
            return pid

        def can_join(pid, group):
            """
            DELIBERATE DEVIATION from reMergeNodes (graph.cpp:155): the old
            code re-merged on the score alone, skipping the mustMerge vetoes it
            had enforced during the build — so a split could quietly re-create
            the same-chain and over-span fusions the build had just refused.
            We re-apply them, and against the ACCUMULATED person rather than
            pairwise, because fusion is transitive: j+k and k+m each passing a
            pairwise span check does not mean j..m fits inside the radius.
            """
            lo, hi, chains = acc[pid]
            if chains & group.chain_ids:
                return False
            return (max(hi, group.highest) - min(lo, group.lowest)
                    <= self.equality_radius)

        def add_to(pid, idx, group):
            self.persons[pid].add_group(group)
            a = acc[pid]
            a[0] = min(a[0], group.lowest)
            a[1] = max(a[1], group.highest)
            a[2] |= group.chain_ids
            owner[idx] = pid

        for j in range(len(groups)):
            for k in range(j + 1, len(groups)):
                pj, pk = owner.get(j), owner.get(k)
                if pj is not None and pj == pk:
                    continue                        # already together
                if narrator_distance(groups[j].canonical,
                                     groups[k].canonical) <= threshold:
                    continue

                if pj is None and pk is None:
                    pid = new_person()
                    add_to(pid, j, groups[j])
                    if not can_join(pid, groups[k]):
                        continue                    # j stands alone for now
                    add_to(pid, k, groups[k])
                elif pk is None:
                    if can_join(pj, groups[k]):
                        add_to(pj, k, groups[k])
                elif pj is None:
                    if can_join(pk, groups[j]):
                        add_to(pk, j, groups[j])
                else:
                    # fuse two provisional persons, if the union still fits
                    lo, hi, ch = acc[pk]
                    lo2, hi2, ch2 = acc[pj]
                    if (ch & ch2) or (max(hi, hi2) - min(lo, lo2)
                                      > self.equality_radius):
                        continue
                    for g in list(self.persons[pk].groups):
                        self.persons[pj].add_group(g)
                    acc[pj] = [min(lo, lo2), max(hi, hi2), ch | ch2]
                    del self.persons[pk]
                    del acc[pk]
                    for idx, p in list(owner.items()):
                        if p == pk:
                            owner[idx] = pj

        # groups that matched nothing become their own person (old
        # `unmatchedGroups` loop, graph.cpp:182-194)
        for idx, g in enumerate(groups):
            if idx not in owner:
                add_to(new_person(), idx, g)

        # --- point every occurrence at its new owner ---
        for idx, g in enumerate(groups):
            pid = owner[idx]
            g.person_id = pid
            for oid in g.occurrence_ids:
                self.chain_nodes[oid].group_id = pid

        n_new = len(set(owner.values()))
        if n_new > 1:
            self.stats["splits"] += 1
        return n_new > 1

    def _may_fuse(self, g1, g2) -> bool:
        """The two mustMerge vetoes (graph.h:900-924), at group level."""
        if g1.chain_ids & g2.chain_ids:
            return False                    # same sanad -> different people
        span = max(g1.highest, g2.highest) - min(g1.lowest, g2.lowest)
        return span <= self.equality_radius

    def _detach_edges(self, person_id):
        """Unhook a person from its neighbours (edges are rebuilt after)."""
        person = self.persons[person_id]
        for other in list(person.parents):
            if other in self.persons:
                self.persons[other].children.discard(person_id)
        for other in list(person.children):
            if other in self.persons:
                self.persons[other].parents.discard(person_id)
        person.parents.clear()
        person.children.clear()

    # ------------------------------------------------------- main
    def break_cycles(self):
        """
        Three escalating-threshold passes. Returns the number of persons
        actually split.

        Structure follows the old LoopBreakingVisitor exactly: ONE DFS over
        the whole graph per threshold, dissolving on every back-edge it meets
        (`detectedCycle` -> reMergeNodes on the node plus its parent stack,
        graph.h:550-565), then move to the next threshold. It does NOT loop
        until acyclic — the old system made three passes and accepted whatever
        survived, which is why the paper reports a graph that is only mostly a
        DAG. Termination is therefore trivial, and `remaining_cycles()`
        reports honestly what is left.
        """
        resolved = 0
        for i in range(1, 4):
            threshold = self.base_threshold + i * self.delta
            tried = set()
            for cycle in self._find_all_cycles():
                for pid in cycle:
                    if pid in tried or pid not in self.persons:
                        continue
                    tried.add(pid)
                    if self._dissolve(pid, threshold):
                        resolved += 1
            self._rebuild_edges()
        return resolved

    def _find_all_cycles(self):
        """
        One DFS over 'parents'; yield the cycle behind every back-edge found.
        Collected up front so the caller can mutate the graph while iterating.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {pid: WHITE for pid in self.persons}
        parent_of = {}
        cycles = []

        for start in self.persons:
            if color[start] != WHITE:
                continue
            stack = [(start, iter(self.persons[start].parents))]
            color[start] = GRAY
            while stack:
                node, it = stack[-1]
                advanced = False
                for nxt in it:
                    if nxt not in self.persons:
                        continue
                    if color[nxt] == GRAY:
                        cycle, cur = [nxt], node
                        while cur != nxt and cur in parent_of:
                            cycle.append(cur)
                            cur = parent_of[cur]
                        cycles.append(cycle)
                    elif color[nxt] == WHITE:
                        color[nxt] = GRAY
                        parent_of[nxt] = node
                        stack.append((nxt, iter(self.persons[nxt].parents)))
                        advanced = True
                        break
                if not advanced:
                    color[node] = BLACK
                    stack.pop()
        return cycles

    def remaining_cycles(self) -> int:
        """How many back-edges survive (0 == the graph is a DAG)."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {pid: WHITE for pid in self.persons}
        count = 0
        for start in self.persons:
            if color[start] != WHITE:
                continue
            stack = [(start, iter(self.persons[start].children))]
            color[start] = GRAY
            while stack:
                node, it = stack[-1]
                advanced = False
                for nxt in it:
                    if nxt not in self.persons:
                        continue
                    if color[nxt] == GRAY:
                        count += 1
                    elif color[nxt] == WHITE:
                        color[nxt] = GRAY
                        stack.append((nxt, iter(self.persons[nxt].children)))
                        advanced = True
                        break
                if not advanced:
                    color[node] = BLACK
                    stack.pop()
        return count


# ---------------------------------------------------------------------------
# Quick demo:  python cycle_breaking.py   (tiny graph with a real cycle)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from graph_nodes import GraphNarratorNode, ChainNarratorNode, GroupNode
    from models import Narrator, ConnectorType

    def make(name):
        n = Narrator()
        pos = 0
        for w in name.split():
            if w == "بن":
                n.add_connector(w, pos, pos + 2, ConnectorType.IBN)
            else:
                n.add_name(w, pos, pos + len(w))
            pos += len(w) + 1
        return n

    # Three persons in a cycle 0 -> 1 -> 2 -> 0. Person 1 holds TWO groups
    # (two different people wrongly fused) so it is the one that can split.
    persons, chain_nodes, gid = {}, {}, 0
    plan = [["محمد بن يعقوب"],
            ["علي بن ابراهيم", "احمد بن خالد"],      # the bad merge
            ["احمد بن محمد"]]
    for pid, names in enumerate(plan):
        node = GraphNarratorNode(pid)
        for nm in names:
            cn = ChainNarratorNode(gid, make(nm), chain_id=gid, position=pid)
            chain_nodes[gid] = cn
            node.add_group(GroupNode(gid, nm, cn))
            gid += 1
        persons[pid] = node

    persons[0].parents = {1}; persons[1].children = {0}
    persons[1].parents = {2}; persons[2].children = {1}
    persons[2].parents = {0}; persons[0].children = {2}

    breaker = CycleBreaker(persons, chain_nodes)
    print("cycle before :", breaker._find_one_cycle())
    resolved = breaker.break_cycles()
    print("cycles resolved:", resolved, "| stats:", breaker.stats)
    print("persons after:", len(persons), "(was 3; the fused node split)")
    print("back-edges after:", breaker.remaining_cycles(), "(0 = DAG)")
