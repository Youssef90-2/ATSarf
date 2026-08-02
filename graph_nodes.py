"""
graph_nodes.py
==============
IDEA: the building blocks of the narrator graph (old graph_nodes.h).

After Part 1 we have chains of Narrator objects. To build a graph where
each REAL person is one node, we need three node types (old system's
hierarchy, simplified):

    ChainNarratorNode  — one narrator OCCURRENCE in one chain.
                         (محمد بن يعقوب as it appears in chain #5)
                         Many occurrences of the same person exist.

    GraphNarratorNode  — one UNIQUE person after merging.
                         Holds edges (who they narrated from / to) and a
                         `rank` = generation (distance from Prophet/Imam).

    GroupNode          — groups equivalent GraphNarratorNodes together;
                         this is what the hash indexes and what merging
                         operates on.

Paper mapping (§2 "Merge/Split"): merge removes two nodes and adds one with
their combined edges; split reverses it. Edges = the "narrated from"
relation R in the graph G = <N, R>.

--------------------------------------------------------------------------
Simplification vs the old C++: we use plain Python objects with sets of
neighbor ids instead of pointer webs — easier to read, serialize (JSON),
and reason about. Same information, cleaner shape.
--------------------------------------------------------------------------
"""

from name_model import canonize


# ===========================================================================
# 1. ChainNarratorNode — one occurrence in one chain
# ===========================================================================

class ChainNarratorNode:
    """A single narrator as it appears at one position in one chain."""

    __slots__ = ("id", "narrator", "canonical", "chain_id", "position",
                 "group_id")

    def __init__(self, node_id, narrator, chain_id, position):
        self.id = node_id
        self.narrator = narrator                 # Part-1 Narrator object
        self.canonical = canonize(narrator)      # cached CanonicalName
        self.chain_id = chain_id                 # which chain
        self.position = position                 # index within the chain
        self.group_id = None                     # set when merged into a group

    @property
    def name(self):
        return self.narrator.full_name

    def __repr__(self):
        return f"ChainNode(#{self.id} '{self.name}' chain={self.chain_id})"


# ===========================================================================
# 2. GraphNarratorNode — one unique person (a graph node)
# ===========================================================================

class GraphNarratorNode:
    """One real person. Edges point to other GraphNarratorNodes by id."""

    def __init__(self, node_id):
        self.id = node_id
        self.occurrences = []          # ChainNarratorNode ids merged here
        self.names = []                # distinct name strings seen
        self.parents = set()           # narrated FROM these (teachers/earlier)
        self.children = set()          # narrated TO these (students/later)
        self.rank = -1                 # generation; distance from the Prophet
        self.is_rasoul = False
        # biography back-links (filled in Part 3 — cross-doc)
        self.biography_refs = []

    def add_occurrence(self, chain_node: ChainNarratorNode):
        self.occurrences.append(chain_node.id)
        if chain_node.name not in self.names:
            self.names.append(chain_node.name)
        if chain_node.narrator.is_rasoul:
            self.is_rasoul = True

    @property
    def primary_name(self):
        """The longest name seen — usually the most complete form."""
        return max(self.names, key=len) if self.names else ""

    @property
    def occurrence_count(self):
        return len(self.occurrences)

    def to_dict(self):
        return {
            "id": self.id,
            "primary_name": self.primary_name,
            "names": self.names,
            "rank": self.rank,
            "is_rasoul": self.is_rasoul,
            "occurrences": self.occurrence_count,
            "parents": sorted(self.parents),
            "children": sorted(self.children),
            "biography_refs": self.biography_refs,
        }

    def __repr__(self):
        return (f"GraphNode(#{self.id} '{self.primary_name}' "
                f"rank={self.rank} occ={self.occurrence_count})")


# ===========================================================================
# 3. GroupNode — a merge group (what the hash stores)
# ===========================================================================

class GroupNode:
    """
    Groups equivalent GraphNarratorNodes. In the simple design each group
    holds exactly one GraphNarratorNode; the type exists so the merge/split
    vocabulary matches the paper and the old code (and so Part 3's hash can
    reference stable group ids).
    """

    def __init__(self, group_id, graph_node: GraphNarratorNode):
        self.id = group_id
        self.graph_node = graph_node
        self.keys = []                 # hash keys this group is indexed under

    @property
    def canonical_name(self):
        return self.graph_node.primary_name

    def __repr__(self):
        return f"Group(#{self.id} -> {self.graph_node!r})"


# ---------------------------------------------------------------------------
# Quick demo:  python graph_nodes.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from models import Narrator, ConnectorType

    # build two occurrences of the same person from two "chains"
    def make(name_parts):
        n = Narrator()
        pos = 0
        for text, kind in name_parts:
            if kind == "name":
                n.add_name(text, pos, pos + len(text))
            else:
                n.add_connector(text, pos, pos + len(text), kind)
            pos += len(text) + 1
        return n

    occ1 = ChainNarratorNode(1, make([("محمد", "name"),
                                      ("بن", ConnectorType.IBN),
                                      ("يعقوب", "name")]), chain_id=0, position=0)
    occ2 = ChainNarratorNode(2, make([("محمد", "name"),
                                      ("بن", ConnectorType.IBN),
                                      ("يعقوب", "name"),
                                      ("الكليني", ConnectorType.POSSESSIVE)]),
                             chain_id=5, position=0)

    print("two occurrences:")
    print("  ", occ1)
    print("  ", occ2)

    # merge them into one graph node
    person = GraphNarratorNode(100)
    person.add_occurrence(occ1)
    person.add_occurrence(occ2)
    person.children.add(101)          # narrated to node 101
    person.rank = 3

    print("\nmerged into one person:")
    print("  ", person)
    print("   names seen :", person.names)
    print("   primary    :", person.primary_name)

    group = GroupNode(100, person)
    print("\ngroup:", group)
    print("\nJSON:")
    import json
    print(json.dumps(person.to_dict(), ensure_ascii=False, indent=2))
