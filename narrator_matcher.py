"""
narrator_matcher.py
===================
IDEA (paper §4; old isRealNarrator + RealNarratorAction,
biographyGraphUtilities.cpp:4 / .h:6):
Given a narrator detected in a BIOGRAPHY book, find which person(s) in the
hadith narrator graph it refers to.

    bool isRealNarrator(NarratorGraph *graph, Narrator *n, NarratorNodeList &list) {
        RealNarratorAction v(list);
        v.resetFound();
        graph->performActionToAllCorrespondingNodes(n, v);   // multi-key hash
        return v.isFound();
    }

    void RealNarratorAction::action(const QString &, GroupNode *node, double similarity) {
        if (similarity > hadithParameters.equality_threshold) {
            found = true;
            ... keep the HIGHEST similarity per node ...
        }
    }

So it is not a yes/no test: it returns a GRADED, de-duplicated list of
(person, similarity), and "is a real narrator" simply means that list is
non-empty. A3 needs the graded list (to disambiguate between several
candidates); the biography FSM needs only the boolean (to renew its
tolerance budget).

--------------------------------------------------------------------------
THRESHOLD: read from the graph's own saved params, not a local default.
Matching biography names at 0.1 against a graph that was built at 0.5 means
the two stages disagree about who is the same person — the biography side
would fuse people the graph deliberately kept apart.
--------------------------------------------------------------------------
"""

import json
from dataclasses import dataclass

from equality import narrator_distance
from name_hash import NameHash
from name_model import canonize, canonical_from_dict, canonize_string


# ===========================================================================
# 1. One graded match (old Biography::MatchingNode, narrator_abstraction.h:374)
# ===========================================================================

@dataclass
class MatchingNode:
    person_id: int
    similarity: float
    name: str = ""

    def __lt__(self, other):
        # old operator< sorts DESCENDING by similarity
        return self.similarity > other.similarity

    def to_dict(self):
        return {"person_id": self.person_id,
                "similarity": round(self.similarity, 3),
                "name": self.name}


# ===========================================================================
# 2. The graph index
# ===========================================================================

class GraphIndex:
    """
    Wraps a saved narrator graph and answers "is this a real narrator?".

    Rebuilds the SAME multi-key hash the graph builder used, over every name
    form of every person (old: every GroupNode inserted into NarratorHash).
    """

    def __init__(self, data: dict):
        self.persons = {p["id"]: p for p in data.get("persons", [])}
        params = data.get("params", {})
        # D7: use the graph's own threshold, never a local default
        self.threshold = float(params.get("equality_threshold", 0.5))

        self.hash = NameHash()
        self._canon = {}          # person_id -> [CanonicalName, ...]
        self._exact = 0
        self._rebuilt = 0

        for pid, person in self.persons.items():
            for canonical in self._canonicals_of(person):
                self._canon.setdefault(pid, []).append(canonical)
                self.hash.add(pid, canonical)

    def _canonicals_of(self, person):
        """
        Prefer the serialized CanonicalName of each group (exact). Fall back
        to re-parsing the name string only for graphs written before the
        canonical form was serialized.
        """
        groups = person.get("groups")
        if isinstance(groups, list) and groups and isinstance(groups[0], dict):
            self._exact += 1
            return [canonical_from_dict(g) for g in groups]
        self._rebuilt += 1
        names = person.get("names") or [person.get("primary_name", "")]
        return [canonize_string(n) for n in names if n]

    # ------------------------------------------------- the port of the action
    def match(self, narrator_or_canonical, threshold=None):
        """
        Graded matches for a narrator, best first. `narrator_or_canonical`
        may be a Part-1 Narrator (preferred — carries the FSM's connector
        tags) or an already-built CanonicalName.
        """
        c = (narrator_or_canonical
             if hasattr(narrator_or_canonical, "levels")
             else canonize(narrator_or_canonical))
        if c.is_relative:
            # identity unknown until the graph structure resolves it
            # (old equalNew returns 0 for these)
            return []

        limit = self.threshold if threshold is None else threshold
        best = {}
        for pid in self.hash.find_candidates(c):
            for other in self._canon.get(pid, ()):
                similarity = narrator_distance(c, other)
                if similarity > limit and similarity > best.get(pid, 0.0):
                    best[pid] = similarity

        out = [MatchingNode(pid, s,
                            self.persons[pid].get("primary_name", ""))
               for pid, s in best.items()]
        out.sort()                      # descending, via MatchingNode.__lt__
        return out

    def is_real_narrator(self, narrator, threshold=None) -> bool:
        """Old isRealNarrator: true iff at least one node matches."""
        return bool(self.match(narrator, threshold))

    # ------------------------------------------------------- FSM hook
    def confirm(self, narrator):
        """
        The `confirm` callback for BiographyFSM. Returns the graded match
        list; the FSM treats a non-empty list as a confirmation and renews
        its tolerance budget, and stores the list for A3.
        """
        return self.match(narrator)

    # ------------------------------------------------------- loading
    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def stats(self):
        return {
            "persons": len(self.persons),
            "hash_keys": len(self.hash),
            "threshold": self.threshold,
            "persons_exact_canonical": self._exact,
            "persons_reparsed_from_string": self._rebuilt,
        }


# ---------------------------------------------------------------------------
# Quick demo:  python narrator_matcher.py [graph.json]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from models import Narrator, ConnectorType

    def narrator(name):
        n = Narrator()
        pos = 0
        for w in name.split():
            if w in ("بن", "ابن"):
                n.add_connector(w, pos, pos + len(w), ConnectorType.IBN)
            elif w in ("ابو", "ابي", "ابا"):
                n.add_connector(w, pos, pos + len(w), ConnectorType.AB)
            else:
                n.add_name(w, pos, pos + len(w))
            pos += len(w) + 1
        return n

    if len(sys.argv) > 1:
        index = GraphIndex.load(sys.argv[1])
    else:
        # a tiny in-memory graph, so the demo runs with no data files
        from graph_nodes import GraphNarratorNode, ChainNarratorNode, GroupNode
        persons = []
        for pid, nm in enumerate(["محمد بن يعقوب الكليني", "علي بن ابراهيم",
                                  "احمد بن محمد بن عيسا"]):
            cn = ChainNarratorNode(pid, narrator(nm), chain_id=0, position=pid)
            node = GraphNarratorNode(pid)
            node.add_group(GroupNode(pid, nm, cn))
            persons.append(node.to_dict())
        index = GraphIndex({"params": {"equality_threshold": 0.5},
                            "persons": persons})

    print("index:", index.stats())
    print()
    for probe in ["محمد بن يعقوب", "احمد بن محمد", "علي بن ابراهيم",
                  "خالد بن سعد"]:
        matches = index.match(narrator(probe))
        verdict = "REAL narrator" if matches else "not in the graph"
        print(f"  {probe:<22} -> {verdict}")
        for m in matches[:3]:
            print(f"        #{m.person_id}  {m.similarity:.2f}  {m.name}")
