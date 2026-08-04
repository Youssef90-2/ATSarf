"""
graph_build.py
==============
IDEA (buildGraph, paper §3.2): turn the list of chains into a graph where
each REAL person is ONE node, with edges = "narrated from".

This is the merge engine — the heart of Part 2. Old source (verified):
buildGraph() (graph.h:1081) + transform2ChainNodes() (graph.h:852).

THE ALGORITHM (as in the old code):
  1. transform: wrap every narrator of every chain as a ChainNarratorNode,
     and remember the chain order (narrator i narrated from narrator i+1).
  2. for each occurrence, decide which existing person it is:
        - look up CANDIDATES via the hash (blocking, name_hash.py)
        - among candidates within the GENERATION RADIUS r, score with the
          equality metric (equality.py); best score >= threshold -> MERGE
          into that existing person; else -> create a NEW person.
  3. add the "narrated from" edges between consecutive narrators, now that
     each occurrence maps to a person node.

KEY PARAMETERS (old hadithParameters, all tunable):
    equality_threshold = 0.1   min score to merge (old default)
    equality_radius    = 3     max generation gap between merge candidates
                               (a Companion can't merge with a 5th-gen
                                narrator even if names overlap)

OUTPUT: a dict of GraphNarratorNode by id + the ChainNarratorNode table.
Ranks and cycle-breaking are applied later by narrator_graph.py.
"""

from dataclasses import dataclass

from graph_nodes import ChainNarratorNode, GraphNarratorNode, GroupNode
from name_hash import NameHash, primary_key
from equality import narrator_distance


@dataclass
class GraphParams:
    equality_threshold: float = 0.1
    equality_radius: int = 3
    qualifiers_mode: str = "off"     # "off" = old behavior, "on" = improvement
    metric_mode: str = "structural"  # "structural" = old; "hybrid" = +embeddings
    context_check: bool = False      # equalHelper-style child check (opt-in;
                                     # conservative — can split prolific narrators)


class GraphBuilder:

    def __init__(self, params: GraphParams = None):
        self.params = params or GraphParams()
        self.chain_nodes = {}        # id -> ChainNarratorNode
        self.persons = {}            # id -> GraphNarratorNode
        self.groups = {}             # id -> GroupNode  (atomic merge units)
        self._groups_by_key = {}     # exact key -> list[GroupNode]
        self.hash = NameHash()
        self._next_chain_id = 0
        self._next_person_id = 0
        self._next_group_id = 0
        # name-embedding model (AraBERT), built lazily only in hybrid mode
        self._embedder = None
        if self.params.metric_mode == "hybrid":
            try:
                from name_embeddings import NameEmbeddings
                self._embedder = NameEmbeddings(enabled=True)
            except Exception as e:
                print(f"[graph] embeddings unavailable, using structural: {e}")
                self._embedder = None
        # (chain_id, position) -> chain_node id, to reach a node's neighbours
        # in its sanad (the next narrator = its "child", old equalHelper).
        self._pos_index = {}
        # per-person caches so the two mustMerge guards are O(1) instead of
        # O(occurrences) -- a 400-occurrence node would otherwise be rescanned
        # on every candidate test.
        self._person_chains = {}     # pid -> set of chain_ids it appears in
        self._person_span = {}       # pid -> [lowest position, highest position]

    # ------------------------------------------------------------ step 1
    def _transform_chains(self, chains):
        """
        Wrap every narrator occurrence as a ChainNarratorNode.
        Returns list of chains, each a list of ChainNarratorNode ids in order.
        """
        chain_sequences = []
        for chain_id, chain in enumerate(chains):
            sequence = []
            for position, narrator in enumerate(chain.narrators):
                node = ChainNarratorNode(self._next_chain_id, narrator,
                                         chain_id, position)
                self.chain_nodes[node.id] = node
                self._pos_index[(chain_id, position)] = node.id
                sequence.append(node.id)
                self._next_chain_id += 1
            chain_sequences.append(sequence)
        return chain_sequences

    # ------------------------------------------------------------ step 2
    def _find_or_create_person(self, chain_node: ChainNarratorNode) -> int:
        """
        Decide which person this occurrence is. Returns a person id.

        TWO STAGES, exactly like the old buildGraph (graph.h:1093-1118):

          A. EXACT key  (old BuildAction + performActionToExactCorrespondingNodes)
             Same written name -> join that GroupNode. No scoring needed.
          B. FUZZY      (old MergeAction + performActionToAllCorrespondingNodes)
             Otherwise open a new GroupNode and fuse it into the best-scoring
             existing person, if any clears the threshold and the guards.
        """
        c = chain_node.canonical
        key = primary_key(c)

        # relative narrators (ابيه) never merge by name — always their own
        # person for now; Part 3 / graph structure resolves them later.
        if c.is_relative:
            return self._new_person_for(self._new_group(key, chain_node))

        # rasoul: all rasoul occurrences are the SAME single node
        if c.is_rasoul:
            for pid, person in self.persons.items():
                if person.is_rasoul:
                    self._join_group(pid, key, chain_node)
                    return pid
            return self._new_person_for(self._new_group(key, chain_node))

        # ---- STAGE A: an existing group with the identical key ----
        for group in self._groups_by_key.get(key, ()):
            pid = group.person_id
            person = self.persons.get(pid)
            if person is not None and not self._merge_allowed(chain_node, pid):
                continue
            group.add(chain_node)
            chain_node.group_id = pid
            self._note(pid, chain_node)
            return pid

        # ---- STAGE B: fuzzy match against existing persons ----
        group = self._new_group(key, chain_node)
        best_id, best_score = None, self.params.equality_threshold
        for pid, _key_score in self.hash.find_candidates(c).items():
            person = self.persons.get(pid)
            if person is None:
                continue
            if not self._merge_allowed(chain_node, pid):
                continue
            # compare against the person's fullest known form
            score = self._score_against_person(c, person)
            if score <= best_score:
                continue
            # CONTEXT CHECK (old equalHelper, graph.h:1329): a name match is
            # not enough — the occurrence's neighbour in its sanad (the next
            # narrator = its "child") must be compatible with the person's
            # other occurrences' children. This blocks a SHORT ambiguous name
            # ("احمد بن محمد") from bridging two different people ("...بن خالد"
            # vs "...بن عيسا") that the name metric alone would fuse.
            if self.params.context_check and \
                    not self._context_compatible(chain_node, person):
                continue
            best_id, best_score = pid, score

        if best_id is not None:
            self._attach_group(best_id, group)
            return best_id
        return self._new_person_for(group)

    def _score_against_person(self, canonical, person: GraphNarratorNode):
        """
        Best equality score of `canonical` vs any GROUP of the person.
        Compares cached CanonicalNames — the old code re-derived a canonical
        form from the name STRING on every comparison, which both threw away
        the FSM's real connector tags and cost an O(n) re-parse per pair.
        """
        best = 0.0
        for group in person.groups:
            s = narrator_distance(canonical, group.canonical,
                                  self.params.qualifiers_mode,
                                  self.params.metric_mode,
                                  self._embedder)
            best = max(best, s)
        return best

    # ------------------------------------------------------- merge guards
    def _merge_allowed(self, chain_node, person_id: int) -> bool:
        """
        Both mustMerge vetoes (graph.h:900-924), applied at PERSON level as
        the old code did (it read the GraphNarratorNode's lowest/highest and
        its per-chain slot, not the individual group's).
        """
        return (not self._same_chain_conflict(chain_node, person_id)
                and self._within_radius(chain_node, person_id))

    def _same_chain_conflict(self, chain_node, person_id: int) -> bool:
        """True if `person_id` already has an occurrence in this sanad."""
        return chain_node.chain_id in self._person_chains.get(person_id, ())

    def _within_radius(self, chain_node, person_id: int) -> bool:
        """
        Generation guard (paper's r), as the old mustMerge computed it
        (graph.h:914-924):

            least   = min(new_position, group.getLowestIndex())
            highest = max(new_position, group.getHighestIndex())
            return highest - least <= equality_radius

        The test is on the SPAN of the whole group once the new occurrence
        joins -- not on the distance to the nearest existing occurrence. The
        previous any-occurrence version let a group ratchet across generations
        one hop at a time (0 merges with 3, 3 with 6, 6 with 9 ...), which is
        how a single node came to span 12 positions of an isnad.
        """
        span = self._person_span.get(person_id)
        if span is None:
            return True
        lowest = min(span[0], chain_node.position)
        highest = max(span[1], chain_node.position)
        return highest - lowest <= self.params.equality_radius

    def _child_of(self, node):
        """The next narrator in the same sanad (old getChild), or None."""
        nid = self._pos_index.get((node.chain_id, node.position + 1))
        return self.chain_nodes.get(nid) if nid is not None else None

    def _context_compatible(self, chain_node, person: GraphNarratorNode) -> bool:
        """
        Old equalHelper (graph.h:1329): two occurrences may be the same person
        only if their CHILDREN (next narrator in the sanad) are compatible.

        We apply this ONLY to SHORT / AMBIGUOUS names (<= 2 levels, e.g.
        "احمد بن محمد" with no grandfather). A fuller name (3+ levels, or a
        nisba) is specific enough to trust by itself — and a prolific narrator
        like الكليني legitimately has many different children, so we must NOT
        block those. This targets exactly the bridging case (a bare name
        fusing "...بن خالد" and "...بن عيسا") without hurting real merges.
        """
        c = chain_node.canonical
        # a name is "ambiguous" if it has <=2 levels AND no distinguishing nisba
        n_levels = len([lv for lv in c.levels if lv])
        if n_levels > 2 or c.qualifiers:
            return True                      # specific enough -> trust it

        new_child = self._child_of(chain_node)
        if new_child is None:
            return True                      # new occ ends the chain -> ok
        new_cc = new_child.canonical
        if new_cc.is_relative or new_cc.is_rasoul:
            return True                      # ابيه / الإمام: uninformative

        existing_children = []
        for occ_id in person.occurrences:
            occ = self.chain_nodes[occ_id]
            ch = self._child_of(occ)
            if ch is not None:
                cc = ch.canonical
                if not (cc.is_relative or cc.is_rasoul):
                    existing_children.append(cc)

        if not existing_children:
            return True                      # nothing informative to conflict

        # compatible if the new child matches ANY existing child (not a hard
        # reject). If it conflicts with ALL of them, the merge would bridge
        # two different sub-chains -> refuse.
        for cc in existing_children:
            if narrator_distance(new_cc, cc, self.params.qualifiers_mode) > 0.0:
                return True
        return False

    # ---- group / person creation and attachment ----
    def _new_group(self, key, chain_node) -> GroupNode:
        gid = self._next_group_id
        self._next_group_id += 1
        group = GroupNode(gid, key, chain_node)
        self.groups[gid] = group
        self._groups_by_key.setdefault(key, []).append(group)
        return group

    def _join_group(self, person_id, key, chain_node):
        """Add an occurrence to this person's group with `key`, or open one."""
        person = self.persons[person_id]
        for group in person.groups:
            if group.key == key:
                group.add(chain_node)
                chain_node.group_id = person_id
                self._note(person_id, chain_node)
                return
        self._attach_group(person_id, self._new_group(key, chain_node))

    def _new_person_for(self, group: GroupNode) -> int:
        pid = self._next_person_id
        self._next_person_id += 1
        self.persons[pid] = GraphNarratorNode(pid)
        self._person_chains[pid] = set()
        self._person_span[pid] = [group.lowest, group.highest]
        self._attach_group(pid, group)
        return pid

    def _attach_group(self, person_id, group: GroupNode):
        person = self.persons[person_id]
        person.add_group(group)
        for oid in group.occurrence_ids:
            self.chain_nodes[oid].group_id = person_id
        self._person_chains[person_id] |= group.chain_ids
        span = self._person_span[person_id]
        span[0] = min(span[0], group.lowest)
        span[1] = max(span[1], group.highest)
        # index this name form in the hash so later occurrences can find it
        if not group.canonical.is_relative:
            self.hash.add(person_id, group.canonical)

    def _note(self, person_id, chain_node):
        """Keep the guard caches current after an in-place group addition."""
        self._person_chains[person_id].add(chain_node.chain_id)
        span = self._person_span[person_id]
        span[0] = min(span[0], chain_node.position)
        span[1] = max(span[1], chain_node.position)

    # ------------------------------------------------------------ step 3
    def _add_edges(self, chain_sequences):
        """
        Add "narrated from" edges. In a sanad, narrator i heard from
        narrator i+1 (the chain goes from latest transmitter back toward
        the source). So: person(i).parents += person(i+1);
                          person(i+1).children += person(i).
        """
        for sequence in chain_sequences:
            for a, b in zip(sequence, sequence[1:]):
                pa = self.chain_nodes[a].group_id
                pb = self.chain_nodes[b].group_id
                if pa is None or pb is None or pa == pb:
                    continue
                self.persons[pa].parents.add(pb)
                self.persons[pb].children.add(pa)

    # ------------------------------------------------------------ run
    def build(self, chains):
        """chains (list of Part-1 Chain objects) -> {person_id: GraphNarratorNode}."""
        chain_sequences = self._transform_chains(chains)

        # merge pass: assign every occurrence to a person
        for chain_id, sequence in enumerate(chain_sequences):
            for node_id in sequence:
                node = self.chain_nodes[node_id]
                if node.group_id is None:
                    self._find_or_create_person(node)

        # edges
        self._add_edges(chain_sequences)
        return self.persons

    # ---- stats for sanity checking ----
    def stats(self):
        edges = sum(len(p.children) for p in self.persons.values())
        merged = sum(1 for p in self.persons.values()
                     if p.occurrence_count > 1)
        return {
            "chain_occurrences": len(self.chain_nodes),
            "groups": len(self.groups),
            "unique_persons": len(self.persons),
            "merged_persons": merged,
            "edges": edges,
            "hash_keys": len(self.hash),
        }


# ---------------------------------------------------------------------------
# Quick demo:  python graph_build.py    (uses two hand-made chains)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from models import Chain, Narrator, NarratorConnector, ConnectorType

    def narrator(parts, rasoul=False):
        n = Narrator()
        pos = 0
        for text, kind in parts:
            if kind == "name":
                n.add_name(text, pos, pos + len(text))
            else:
                n.add_connector(text, pos, pos + len(text), kind)
            pos += len(text) + 1
        n.is_rasoul = rasoul
        return n

    def chain(narrators):
        c = Chain()
        for i, n in enumerate(narrators):
            if i:
                c.add(NarratorConnector("عن", 0, 2))
            c.add(n)
        return c

    # chain 1: محمد بن يعقوب  عن  علي بن ابراهيم  عن  ابي جعفر (rasoul)
    c1 = chain([
        narrator([("محمد", "name"), ("بن", ConnectorType.IBN), ("يعقوب", "name")]),
        narrator([("علي", "name"), ("بن", ConnectorType.IBN), ("ابراهيم", "name")]),
        narrator([("ابي", ConnectorType.AB), ("جعفر", "name")], rasoul=True),
    ])
    # chain 2: محمد بن يعقوب الكليني  عن  احمد بن محمد  عن  ابي جعفر (rasoul)
    c2 = chain([
        narrator([("محمد", "name"), ("بن", ConnectorType.IBN), ("يعقوب", "name"),
                  ("الكليني", ConnectorType.POSSESSIVE)]),
        narrator([("احمد", "name"), ("بن", ConnectorType.IBN), ("محمد", "name")]),
        narrator([("ابي", ConnectorType.AB), ("جعفر", "name")], rasoul=True),
    ])

    builder = GraphBuilder()
    persons = builder.build([c1, c2])

    print("stats:", builder.stats())
    print("\npersons in the graph:")
    for pid, p in persons.items():
        pj = [persons[x].primary_name for x in p.parents]
        print(f"  #{pid} {p.primary_name}  (seen {p.occurrence_count}x)")
        if pj:
            print(f"       narrated from: {pj}")
    print("\nNote: 'محمد بن يعقوب' and 'محمد بن يعقوب الكليني' should be ONE "
          "person seen 2x; the shared rasoul 'ابي جعفر' should be ONE node.")
