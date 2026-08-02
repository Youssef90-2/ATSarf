"""
cycle_breaking.py
=================
IDEA (Break cycle, paper §2): a real isnad graph is a DAG (directed,
acyclic) — narration flows one way in time, from the source forward.
So if the graph has a CYCLE (A narrated from B, ..., back to A), it means
the merge step wrongly fused two different people into one node. We must
find the bad merge and SPLIT it.

Old source (verified): breakManageableCycles() (graph.h:1136) +
LoopBreakingVisitor (graph.h:529). The exact strategy:

    3 DFS passes with an ESCALATING threshold:
        threshold_i = equality_threshold + i * delta   (i = 1, 2, 3)
    Each pass:
        - DFS the graph; when a back-edge closes a cycle, look at the nodes
          on the cycle and re-check their internal merges at the (stricter)
          threshold. Occurrences that no longer meet the stricter bar get
          SPLIT off into their own node.
    Rationale for escalation: first break cycles caused by the WEAKEST
    merges (low-confidence fusions), then re-check with progressively
    stricter standards. This conservatism is exactly why the paper's
    recall dropped to 63% (they say so) — a documented weak point and our
    opportunity to improve.

--------------------------------------------------------------------------
Our simplified realization:
  - detect cycles with an iterative DFS over parents edges
  - on a cycle, pick the WEAKEST merge on it (the person node whose
    occurrences least agree) and split its least-agreeing occurrence out
  - repeat over 3 escalating thresholds
Same idea, readable shape. Operates on the {id: GraphNarratorNode} dict.
--------------------------------------------------------------------------
"""

from equality import narrator_distance
from name_model import canonize_string


class CycleBreaker:

    def __init__(self, persons, chain_nodes, base_threshold=0.1, delta=0.2):
        self.persons = persons              # {id: GraphNarratorNode}
        self.chain_nodes = chain_nodes      # {id: ChainNarratorNode}
        self.base_threshold = base_threshold
        self.delta = delta
        self._next_id = max(persons.keys()) + 1 if persons else 0

    # ------------------------------------------------------- cycle detection
    def _find_one_cycle(self):
        """
        Iterative DFS over the 'parents' edges. Returns a list of person ids
        forming a cycle, or None if the graph is acyclic.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {pid: WHITE for pid in self.persons}
        parent_of = {}

        for start in self.persons:
            if color[start] != WHITE:
                continue
            stack = [(start, iter(self.persons[start].parents))]
            while stack:
                node, it = stack[-1]
                color[node] = GRAY
                advanced = False
                for nxt in it:
                    if nxt not in self.persons:
                        continue
                    if color[nxt] == GRAY:
                        # back-edge -> reconstruct the cycle
                        cycle = [nxt]
                        cur = node
                        while cur != nxt and cur in parent_of:
                            cycle.append(cur)
                            cur = parent_of[cur]
                        cycle.append(nxt)
                        return cycle
                    if color[nxt] == WHITE:
                        parent_of[nxt] = node
                        stack.append((nxt, iter(self.persons[nxt].parents)))
                        advanced = True
                        break
                if not advanced:
                    color[node] = BLACK
                    stack.pop()
        return None

    # ------------------------------------------------------- merge strength
    def _weakest_person_on_cycle(self, cycle, threshold):
        """
        Among the merged persons on the cycle, find the one whose
        occurrences agree LEAST (below the stricter threshold) — the most
        likely bad merge. Returns (person_id, weak_occurrence_id) or None.
        """
        worst = None
        worst_score = threshold
        for pid in cycle:
            person = self.persons.get(pid)
            if person is None or person.occurrence_count < 2:
                continue
            # score every occurrence against the person's primary name
            primary = canonize_string(person.primary_name)
            for occ_id in person.occurrences:
                occ = self.chain_nodes[occ_id]
                if occ.canonical.is_relative or occ.canonical.is_rasoul:
                    continue
                score = narrator_distance(occ.canonical, primary)
                if score < worst_score:
                    worst_score = score
                    worst = (pid, occ_id)
        return worst

    # ------------------------------------------------------- split
    def _split_off(self, person_id, occurrence_id):
        """
        Move one occurrence out of a person into a brand-new person node.

        CRITICAL (bug found on real kafi1 data): the first version did NOT
        touch the edges, so the cycle survived the split and was re-detected
        forever — 'cycles_broken' counted the SAME few cycles thousands of
        times (2,851 for 4,288 persons). Splitting must also detach the
        edges that the moved occurrence contributed, so here we clear the
        old person's edges and let narrator_graph rebuild them from the
        (updated) occurrence -> person mapping.
        """
        from graph_nodes import GraphNarratorNode
        person = self.persons[person_id]
        if occurrence_id not in person.occurrences:
            return False
        if person.occurrence_count < 2:
            return False                      # never empty a person

        person.occurrences.remove(occurrence_id)

        new_person = GraphNarratorNode(self._next_id)
        occ = self.chain_nodes[occurrence_id]
        new_person.add_occurrence(occ)
        occ.group_id = new_person.id
        self.persons[new_person.id] = new_person
        self._next_id += 1

        # rebuild names of the old person from its remaining occurrences
        person.names = []
        for oid in person.occurrences:
            nm = self.chain_nodes[oid].name
            if nm not in person.names:
                person.names.append(nm)

        # detach edges of BOTH nodes; narrator_graph._rebuild_edges will
        # recreate the correct ones from the chains afterwards.
        self._detach(person_id)
        return True

    def _detach(self, person_id):
        """Remove all edges touching a person (both directions)."""
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
    def break_cycles(self, max_iterations=5000):
        """
        Run the 3 escalating-threshold passes. Returns number of cycles
        actually resolved.

        Guard against the spin bug: we remember the cycles we have already
        seen; if the same cycle comes back, we force an edge break instead
        of trying another split.
        """
        resolved = 0
        for i in range(1, 4):                       # 3 passes
            threshold = self.base_threshold + i * self.delta
            seen = set()
            guard = 0
            while guard < max_iterations:
                guard += 1
                cycle = self._find_one_cycle()
                if cycle is None:
                    break                            # acyclic at this level
                signature = frozenset(cycle)
                repeat = signature in seen
                seen.add(signature)

                did_split = False
                if not repeat:
                    weak = self._weakest_person_on_cycle(cycle, threshold)
                    if weak is not None:
                        did_split = self._split_off(*weak)

                if not did_split:
                    # forced: remove a real edge on the cycle so the DAG
                    # property is restored (and progress is guaranteed)
                    if not self._break_weakest_edge(cycle):
                        break                        # cannot progress
                resolved += 1
        return resolved

    def _break_weakest_edge(self, cycle):
        """
        Remove ONE edge that actually exists along the cycle.

        The old version blindly discarded (cycle[0] -> cycle[1]); since the
        DFS returns the cycle in reverse order that edge often did not
        exist, so nothing changed and the cycle was found again. Now we scan
        the cycle for a real edge and remove it.
        """
        n = len(cycle)
        for i in range(n - 1):
            a, b = cycle[i], cycle[i + 1]
            if a not in self.persons or b not in self.persons:
                continue
            # 'a' is a parent of 'b' in the traversal direction we used
            if a in self.persons[b].parents:
                self.persons[b].parents.discard(a)
                self.persons[a].children.discard(b)
                return True
            if b in self.persons[a].parents:
                self.persons[a].parents.discard(b)
                self.persons[b].children.discard(a)
                return True
        return False


# ---------------------------------------------------------------------------
# Quick demo:  python cycle_breaking.py   (build a tiny graph with a cycle)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from graph_nodes import GraphNarratorNode, ChainNarratorNode
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

    # 3 persons in a cycle: 0 -> 1 -> 2 -> 0  (impossible in real isnad)
    persons, chain_nodes = {}, {}
    names = ["محمد بن يعقوب", "علي بن ابراهيم", "احمد بن محمد"]
    for i, nm in enumerate(names):
        cn = ChainNarratorNode(i, make(nm), chain_id=0, position=i)
        chain_nodes[i] = cn
        p = GraphNarratorNode(i)
        p.add_occurrence(cn)
        persons[i] = p
    # wire a cycle
    persons[0].parents = {1}; persons[1].children = {0}
    persons[1].parents = {2}; persons[2].children = {1}
    persons[2].parents = {0}; persons[0].children = {2}

    breaker = CycleBreaker(persons, chain_nodes)
    print("cycle before breaking:", breaker._find_one_cycle())
    splits = breaker.break_cycles()
    print("splits/edge-breaks done:", splits)
    print("cycle after breaking:", breaker._find_one_cycle(), "(None = DAG ✓)")
