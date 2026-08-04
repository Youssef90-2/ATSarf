"""
narrator_graph.py
=================
IDEA: the orchestrator — ties the whole graph phase together.
Port of the old NarratorGraph class + its init() sequence (graph.h:1355).

The exact old order (verified):
    buildGraph  ->  correctTopNodesList  ->  breakManageableCycles
                ->  computeRanks         ->  built = true

So this file:
    1. builds the graph (graph_build.GraphBuilder)
    2. breaks cycles (cycle_breaking.CycleBreaker)  [if enabled]
    3. rebuilds edges (a split may have changed occurrence->person mapping)
    4. computes ranks (generation = distance from the Prophet/Imam)
    5. saves / loads the whole graph as JSON (replaces the old .por binary;
       becomes the state layer for the web system and the input for Part 3)

RANKS (old computeRanks, graph.h:1125): the rasoul node is generation 0;
everyone who narrated from the Prophet is 1; their students 2; etc. A node's
rank = shortest distance (in "narrated from" edges) toward a rasoul node.
The old code ran two BFS passes to stabilize ranks; we do a single BFS from
all rasoul nodes, which gives the same shortest-distance result.
"""

import json
from collections import deque

from graph_build import GraphBuilder, GraphParams
from cycle_breaking import CycleBreaker


class NarratorGraph:

    def __init__(self, params: GraphParams = None, break_cycles: bool = True,
                 delta: float = 0.2):
        self.params = params or GraphParams()
        self.break_cycles = break_cycles
        self.delta = delta
        self.persons = {}            # id -> GraphNarratorNode
        self.chain_nodes = {}        # id -> ChainNarratorNode
        self._builder = None

    # ------------------------------------------------------------- build
    def build(self, chains):
        """Full pipeline: chains -> finished graph (self.persons)."""
        # 1. merge pass
        self._builder = GraphBuilder(self.params)
        self.persons = self._builder.build(chains)
        self.chain_nodes = self._builder.chain_nodes

        # 2. edges must exist BEFORE cycle detection can run
        self._rebuild_edges(chains)

        # 3. break cycles (a real isnad graph must be a DAG).
        #    The breaker rebuilds edges after every split through this
        #    callback — previously the rebuild happened only ONCE, after the
        #    whole loop, which silently undid every change the breaker made
        #    and left the graph exactly as cyclic as it started.
        cycles_broken = 0
        self._remaining_cycles = 0
        if self.break_cycles:
            breaker = CycleBreaker(self.persons, self.chain_nodes,
                                   base_threshold=self.params.equality_threshold,
                                   delta=self.delta,
                                   equality_radius=self.params.equality_radius,
                                   rebuild_edges=lambda: self._rebuild_edges(chains))
            cycles_broken = breaker.break_cycles()
            self._remaining_cycles = breaker.remaining_cycles()
            self._breaker_stats = breaker.stats

        # 4. ranks (generation from the Prophet/Imam)
        self._compute_ranks()

        self._cycles_broken = cycles_broken
        return self.persons

    # --------------------------------------------------------- edges (rebuild)
    def _rebuild_edges(self, chains):
        """
        Recompute parents/children from scratch using the (possibly updated)
        occurrence->person mapping. Safer than patching after splits.
        """
        for p in self.persons.values():
            p.parents.clear()
            p.children.clear()

        # map (chain_id, position) -> person id via occurrences
        pos_to_person = {}
        for pid, person in self.persons.items():
            for occ_id in person.occurrences:
                occ = self.chain_nodes[occ_id]
                pos_to_person[(occ.chain_id, occ.position)] = pid

        for chain_id, chain in enumerate(chains):
            n = len(chain.narrators)
            for i in range(n - 1):
                pa = pos_to_person.get((chain_id, i))
                pb = pos_to_person.get((chain_id, i + 1))
                if pa is None or pb is None or pa == pb:
                    continue
                self.persons[pa].parents.add(pb)
                self.persons[pb].children.add(pa)

    # --------------------------------------------------------- ranks
    def _compute_ranks(self):
        """
        Generation = distance from the SOURCE of narration.
        rank 0 = the source (the Prophet / an Imam), rank 1 = whoever heard
        from them, and so on. `parents` are teachers (earlier), `children`
        are students (later), so rank increases along every edge.

        LONGEST path, not shortest: rank(n) = 1 + max(rank of its parents).
        The property that matters is MONOTONICITY — every edge must go from a
        lower rank to a higher one, or "generation" means nothing and the
        teacher/student direction cannot be read off it. A shortest-path BFS
        (the previous implementation) breaks that: a node reachable by both a
        1-hop and a 4-hop route takes rank 1, so the 4-hop edge runs backwards.
        Its -1 fallback then used min(parent_ranks), making it worse.

        Relation to the old code: RankCorrectorNodeVisitor (graph.h:498) is
        also a longest-path relaxation — `if (rank1 >= rank2) rank2 = rank1+1`
        — but run as exactly TWO BFS passes, which under-converges on a graph
        deeper than two levels. We run a proper topological relaxation
        instead. (In the old system rank only drove graphviz layout, so the
        under-convergence never mattered there; here it feeds the biography
        stage, so it does.)

        Cycles: nodes inside a surviving cycle have no topological position.
        They get a bounded relaxation afterwards — capped, because a cycle
        would otherwise raise its own ranks forever.
        """
        persons = self.persons
        for p in persons.values():
            p.rank = 0

        # Kahn over parent->child edges; in-degree = number of parents
        indegree = {pid: sum(1 for q in p.parents if q in persons)
                    for pid, p in persons.items()}
        queue = deque(pid for pid, d in indegree.items() if d == 0)

        while queue:
            pid = queue.popleft()
            rank = persons[pid].rank
            for child in persons[pid].children:
                if child not in persons:
                    continue
                if persons[child].rank < rank + 1:
                    persons[child].rank = rank + 1      # longest path
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        # whatever is left sits on a cycle: relax a bounded number of times
        leftover = [pid for pid, d in indegree.items() if d > 0]
        for _ in range(min(len(leftover), 10)):
            changed = False
            for pid in leftover:
                parent_ranks = [persons[q].rank for q in persons[pid].parents
                                if q in persons]
                if parent_ranks and persons[pid].rank < max(parent_ranks) + 1:
                    persons[pid].rank = max(parent_ranks) + 1
                    changed = True
            if not changed:
                break
        self._cyclic_persons = len(leftover)

    def rank_violations(self) -> int:
        """
        Edges where rank does not increase (rank(child) <= rank(parent)).
        Zero on the acyclic part; a non-zero count is exactly the number of
        edges trapped in surviving cycles.
        """
        bad = 0
        for p in self.persons.values():
            for child in p.children:
                c = self.persons.get(child)
                if c is not None and c.rank <= p.rank:
                    bad += 1
        return bad

    # --------------------------------------------------------- stats
    def stats(self):
        s = self._builder.stats() if self._builder else {}
        s["cycles_broken"] = getattr(self, "_cycles_broken", 0)
        s["remaining_cycles"] = getattr(self, "_remaining_cycles", 0)
        s["cyclic_persons"] = getattr(self, "_cyclic_persons", 0)
        s["rank_violations"] = self.rank_violations()
        s.update(getattr(self, "_breaker_stats", {}))
        ranked = [p for p in self.persons.values() if p.rank >= 0]
        s["ranked_persons"] = len(ranked)
        s["max_rank"] = max((p.rank for p in ranked), default=-1)
        # most-connected narrators (highest number of students)
        top = sorted(self.persons.values(),
                     key=lambda p: len(p.children), reverse=True)[:5]
        s["top_connected"] = [(p.primary_name, len(p.children)) for p in top]
        return s

    # --------------------------------------------------------- save / load
    def save(self, path):
        data = {
            "params": {
                "equality_threshold": self.params.equality_threshold,
                "equality_radius": self.params.equality_radius,
                "qualifiers_mode": self.params.qualifiers_mode,
                "break_cycles": self.break_cycles,
            },
            "persons": [p.to_dict() for p in self.persons.values()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    @staticmethod
    def load(path):
        """Load a saved graph (for Part 3 / the web layer). Returns dict."""
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # --------------------------------------------------------- graphviz
    def to_dot(self, path, max_nodes=60):
        """
        Export a small subgraph to Graphviz .dot for visualization
        (old displayGraph used `dot -Tsvg`). Renders the most-connected
        nodes so the picture stays readable.
        """
        top = sorted(self.persons.values(),
                     key=lambda p: len(p.children) + len(p.parents),
                     reverse=True)[:max_nodes]
        keep = {p.id for p in top}
        lines = ["digraph narrators {", '  rankdir=TB; node [shape=box];']
        for p in top:
            label = p.primary_name.replace('"', "")
            lines.append(f'  n{p.id} [label="{label}"];')
        for p in top:
            for child in p.children:
                if child in keep:
                    lines.append(f"  n{p.id} -> n{child};")
        lines.append("}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Quick demo:  python narrator_graph.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from models import Chain, Narrator, NarratorConnector, ConnectorType

    def narrator(parts, rasoul=False):
        n = Narrator(); pos = 0
        for text, kind in parts:
            if kind == "name":
                n.add_name(text, pos, pos + len(text))
            else:
                n.add_connector(text, pos, pos + len(text), kind)
            pos += len(text) + 1
        n.is_rasoul = rasoul
        return n

    def chain(narrs):
        c = Chain()
        for i, n in enumerate(narrs):
            if i:
                c.add(NarratorConnector("عن", 0, 2))
            c.add(n)
        return c

    c1 = chain([
        narrator([("محمد", "name"), ("بن", ConnectorType.IBN), ("يعقوب", "name")]),
        narrator([("علي", "name"), ("بن", ConnectorType.IBN), ("ابراهيم", "name")]),
        narrator([("ابي", ConnectorType.AB), ("جعفر", "name")], rasoul=True),
    ])
    c2 = chain([
        narrator([("محمد", "name"), ("بن", ConnectorType.IBN), ("يعقوب", "name"),
                  ("الكليني", ConnectorType.POSSESSIVE)]),
        narrator([("احمد", "name"), ("بن", ConnectorType.IBN), ("محمد", "name")]),
        narrator([("ابي", ConnectorType.AB), ("جعفر", "name")], rasoul=True),
    ])

    graph = NarratorGraph()
    graph.build([c1, c2])

    print("stats:")
    for k, v in graph.stats().items():
        print(f"  {k}: {v}")

    print("\npersons with ranks (generation from rasoul):")
    for p in sorted(graph.persons.values(), key=lambda x: x.rank):
        print(f"  rank {p.rank}: {p.primary_name} (seen {p.occurrence_count}x)")

    graph.save("demo_graph.json")
    print("\nsaved -> demo_graph.json")
