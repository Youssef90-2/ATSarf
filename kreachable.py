"""
kreachable.py
=============
The k-reachable graph algorithm — paper §2 ("Graph algorithms"):

    "k-reachable: takes two nodes n1 and n2 and returns whether n1 is
     reachable from n2 using at most k edges."

This is the primitive the paper's biography BOUNDARY detection is built on
(§4): walking the narrators named in a rijal entry, two CONSECUTIVE narrators
that are k-reachable in the hadith graph belong to the same entry; when they
are NOT k-reachable, ANGE "decides that it crossed a biography boundary."

Old code (narratordetector.cpp:284-285) implemented the neighbourhood step
with BFS at k=1 in BOTH directions:
        node->BFS_traverse(cont, 1,  1);   // k=1, toward children (students)
        node->BFS_traverse(cont, 1, -1);   // k=1, toward parents  (teachers)
so the operational k was 1 (immediate teachers/students). We keep k
configurable (default 2, so a shared teacher/student one hop away also
counts) but reproduce the same directed, bounded search.

This module is deliberately tiny and dependency-free: it works on the plain
persons dict loaded from a <book>.graph.json ({id: {parents:[], children:[]}}),
so both the boundary evaluator and any future feature can reuse it.
"""

from collections import deque


class ReachabilityIndex:
    """
    Wraps a graph's person nodes and answers k-reachability queries.

    persons: dict {person_id: {"parents": [...], "children": [...], ...}}
    Edges are treated as UNDIRECTED for reachability by default (a teacher and
    a student are both "reachable"), matching the old code that traversed both
    directions; set undirected=False to follow child links only.
    """

    def __init__(self, persons: dict, undirected: bool = True):
        self.persons = persons
        self.undirected = undirected

    # ---- neighbours of a node (one hop) ----
    def _neighbours(self, pid):
        p = self.persons.get(pid)
        if not p:
            return ()
        if self.undirected:
            return tuple(p.get("parents", [])) + tuple(p.get("children", []))
        return tuple(p.get("children", []))

    # ---- the paper's primitive: is n1 reachable from n2 within k edges? ----
    def is_reachable(self, n1: int, n2: int, k: int = 2) -> bool:
        """
        True iff n1 can be reached from n2 using AT MOST k edges (BFS bounded
        by depth k). k=1 -> direct teacher/student only (old code's setting).
        """
        if n1 == n2:
            return True
        if n1 not in self.persons or n2 not in self.persons:
            return False
        visited = {n2}
        frontier = deque([(n2, 0)])
        while frontier:
            node, depth = frontier.popleft()
            if depth >= k:
                continue
            for nb in self._neighbours(node):
                if nb == n1:
                    return True
                if nb not in visited:
                    visited.add(nb)
                    frontier.append((nb, depth + 1))
        return False

    # ---- convenience: are ANY of two candidate sets k-reachable? ----
    def any_reachable(self, ids1, ids2, k: int = 2) -> bool:
        """
        True if some person in ids1 is k-reachable from some person in ids2.
        Used when a biography name matched several graph persons (ambiguity):
        the two consecutive narrators are "connected" if ANY of their
        candidate nodes are k-reachable.
        """
        for a in ids1:
            for b in ids2:
                if self.is_reachable(a, b, k):
                    return True
        return False

    # ---- reachable set: everything within k hops of a node (the cluster) ----
    def reachable_within(self, source: int, k: int = 2) -> set:
        """
        All person ids reachable from `source` within k edges (the local
        cluster the paper inspects to pick a biography's subject n_c).
        """
        if source not in self.persons:
            return set()
        seen = {source}
        frontier = deque([(source, 0)])
        while frontier:
            node, depth = frontier.popleft()
            if depth >= k:
                continue
            for nb in self._neighbours(node):
                if nb not in seen:
                    seen.add(nb)
                    frontier.append((nb, depth + 1))
        seen.discard(source)
        return seen


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # tiny graph:  0 -> 1 -> 2 -> 3   (child links), plus 0 -> 4
    persons = {
        0: {"parents": [], "children": [1, 4]},
        1: {"parents": [0], "children": [2]},
        2: {"parents": [1], "children": [3]},
        3: {"parents": [2], "children": []},
        4: {"parents": [0], "children": []},
    }
    r = ReachabilityIndex(persons)
    assert r.is_reachable(1, 0, k=1) is True      # direct child
    assert r.is_reachable(2, 0, k=1) is False     # two hops, k=1 too small
    assert r.is_reachable(2, 0, k=2) is True      # two hops, k=2 ok
    assert r.is_reachable(3, 0, k=2) is False     # three hops
    assert r.is_reachable(3, 0, k=3) is True
    assert r.any_reachable([2, 9], [0], k=2) is True
    assert r.reachable_within(0, k=1) == {1, 4}
    assert r.reachable_within(0, k=2) == {1, 4, 2}
    print("kreachable.py self-test passed")
