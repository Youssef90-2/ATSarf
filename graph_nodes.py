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

    GroupNode          — all occurrences sharing an IDENTICAL canonical key.
                         ATOMIC: its members are literally the same name, so
                         a group is never split. This is what the hash
                         indexes and what merge/split operate on.

    GraphNarratorNode  — one UNIQUE person = a set of GroupNodes fused by the
                         fuzzy equality metric. Holds edges (who they narrated
                         from / to) and a `rank` = generation.

Paper mapping (§2 "Merge/Split"): merge removes two nodes and adds one with
their combined edges; split reverses it. Edges = the "narrated from"
relation R in the graph G = <N, R>.

--------------------------------------------------------------------------
WHY THE MIDDLE LEVEL EXISTS (old graph_nodes.h, restored here):
Split is defined as  m(n1..nk) -> m1(n1..ni), m2(ni+1..nk).  It needs
indivisible pieces to redistribute. GroupNodes are those pieces: because all
their members share one exact key, no evidence can ever separate them, so
LoopBreakingVisitor::reMergeNodes (graph.cpp:115) can dissolve a person into
its groups and re-merge them at a stricter threshold — always terminating,
always making a decision it can justify.

Collapsing this to occurrence->person (the previous design) made split
degenerate into "peel off one occurrence at a time", which could not undo a
bad fusion and had no progress guarantee.
--------------------------------------------------------------------------
Simplification vs the old C++: plain Python objects with sets of neighbour
ids instead of pointer webs — easier to read, serialize (JSON), reason about.
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
# 2. GroupNode — occurrences sharing one exact key (the atomic merge unit)
# ===========================================================================

class GroupNode:
    """
    All occurrences whose canonical key is IDENTICAL. Indivisible: since the
    members are the same name, nothing can justify separating them.

    `person_id` is which GraphNarratorNode currently owns this group; split
    changes that ownership, it never touches the group's contents.
    """

    __slots__ = ("id", "key", "canonical", "occurrence_ids", "person_id",
                 "name", "is_rasoul", "lowest", "highest", "chain_ids")

    def __init__(self, group_id, key, chain_node: ChainNarratorNode):
        self.id = group_id
        self.key = key
        self.canonical = chain_node.canonical
        self.name = chain_node.name
        self.occurrence_ids = [chain_node.id]
        self.person_id = None
        self.is_rasoul = bool(chain_node.narrator.is_rasoul)
        # position span + chains, kept incrementally for the mustMerge guards
        self.lowest = chain_node.position
        self.highest = chain_node.position
        self.chain_ids = {chain_node.chain_id}

    def add(self, chain_node: ChainNarratorNode):
        self.occurrence_ids.append(chain_node.id)
        self.lowest = min(self.lowest, chain_node.position)
        self.highest = max(self.highest, chain_node.position)
        self.chain_ids.add(chain_node.chain_id)
        if chain_node.narrator.is_rasoul:
            self.is_rasoul = True

    @property
    def size(self):
        return len(self.occurrence_ids)

    def to_dict(self):
        """
        Serialize the group WITH its CanonicalName.

        Why the canonical form and not just the name string: the biography
        stage matches a Narrator (carrying the FSM's real IBN/AB/POSSESSIVE
        tags) against graph persons. If the graph side is reconstructed by
        re-parsing a name string, the two sides can canonize differently and
        a true match is silently missed. Storing the structure keeps the
        comparison symmetric.
        """
        c = self.canonical
        return {
            "name": self.name,
            "key": self.key,
            "size": self.size,
            "levels": [[[it.text, it.kind] for it in lv] for lv in c.levels],
            "qualifiers": list(c.qualifiers),
            "is_rasoul": c.is_rasoul,
            "is_relative": c.is_relative,
        }

    def __repr__(self):
        return f"Group(#{self.id} '{self.name}' x{self.size})"


# ===========================================================================
# 3. GraphNarratorNode — one unique person (a graph node)
# ===========================================================================

class GraphNarratorNode:
    """
    One real person = a set of GroupNodes fused by the equality metric.
    Edges point to other GraphNarratorNodes by id.

    `occurrences` / `names` / `is_rasoul` are DERIVED from the groups, so a
    split only has to move groups around — the person's view stays consistent
    automatically.
    """

    def __init__(self, node_id):
        self.id = node_id
        self.groups = []               # list[GroupNode]
        self.parents = set()           # narrated FROM these (teachers/earlier)
        self.children = set()          # narrated TO these (students/later)
        self.rank = -1                 # generation; distance from the Prophet
        # biography back-links (filled in Part 3 — cross-doc)
        self.biography_refs = []

    # ---------------------------------------------------------- composition
    def add_group(self, group: GroupNode):
        if group not in self.groups:
            self.groups.append(group)
        group.person_id = self.id

    def remove_group(self, group: GroupNode):
        if group in self.groups:
            self.groups.remove(group)

    def add_occurrence(self, chain_node: ChainNarratorNode):
        """
        Convenience for callers that have no key handy (demos, tests): wraps
        the occurrence in a singleton group keyed by its own name.
        """
        for g in self.groups:
            if g.name == chain_node.name:
                g.add(chain_node)
                return
        self.add_group(GroupNode(-1, chain_node.name, chain_node))

    # -------------------------------------------------------------- derived
    @property
    def occurrences(self):
        return [oid for g in self.groups for oid in g.occurrence_ids]

    @property
    def occurrence_count(self):
        return sum(g.size for g in self.groups)

    @property
    def names(self):
        """Distinct name forms, in the order the groups were fused."""
        seen, out = set(), []
        for g in self.groups:
            if g.name not in seen:
                seen.add(g.name)
                out.append(g.name)
        return out

    @property
    def is_rasoul(self):
        return any(g.is_rasoul for g in self.groups)

    @property
    def lowest(self):
        return min((g.lowest for g in self.groups), default=0)

    @property
    def highest(self):
        return max((g.highest for g in self.groups), default=0)

    @property
    def chain_ids(self):
        s = set()
        for g in self.groups:
            s |= g.chain_ids
        return s

    @property
    def primary_name(self):
        """
        The name form seen MOST OFTEN (old: the group's key). Using the
        LONGEST form instead — the previous behaviour — labelled a node with
        its rarest variant, so an over-merged node ended up named after a
        one-off spelling and was impossible to recognise in the output.
        """
        if not self.groups:
            return ""
        best = max(self.groups, key=lambda g: (g.size, len(g.name)))
        return best.name

    def to_dict(self):
        return {
            "id": self.id,
            "primary_name": self.primary_name,
            "names": self.names,
            "rank": self.rank,
            "is_rasoul": self.is_rasoul,
            "occurrences": self.occurrence_count,
            "groups": [g.to_dict() for g in self.groups],
            "parents": sorted(self.parents),
            "children": sorted(self.children),
            "biography_refs": self.biography_refs,
        }

    def __repr__(self):
        return (f"GraphNode(#{self.id} '{self.primary_name}' "
                f"rank={self.rank} occ={self.occurrence_count} "
                f"groups={len(self.groups)})")


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

    print("\ngroups inside this person (atomic, one per exact name form):")
    for g in person.groups:
        print("   ", g)
    print("\nJSON:")
    import json
    print(json.dumps(person.to_dict(), ensure_ascii=False, indent=2))
