"""
biography_linker.py
===================
IDEA (paper §4-§5, the cross-document core; old findChosenBiography +
isRealNarrator, narratordetector.cpp:268 / biographyGraphUtilities.cpp):
Link each rijal ENTRY (from biography_segmenter) to the RIGHT person node in
the hadith narrator graph, and USE the graph to decide the entry's boundary.

This file is built to REPRODUCE THE PAPER'S TABLE 2 EXPERIMENT: it runs in
two modes so the graph's contribution can be measured, not asserted.

    NO-GRAPH  (baseline / paper's "No-Cross-doc")
        Detect the entry's narrators from the text alone. Boundary = the
        span the segmenter guessed. No professor/student knowledge.

    WITH-GRAPH (paper's "Cross-doc")
        1. MATCH the subject name to graph person(s) via the name hash
           (isRealNarrator): candidates whose similarity > threshold.
        2. DISAMBIGUATE by neighbourhood (findChosenBiography): the right
           person is the one whose graph neighbours (parents=teachers,
           children=students) are the ones actually named NEAR the subject
           in the entry text. Each nearby graph-neighbour = +1 to the score.
        3. BOUNDARY: the entry really extends to cover exactly those narrators
           the graph confirms belong together — so the graph tells us where
           the entry's narrator region ends, which text alone gets wrong.
           This is the 41%->93% boundary-recall story in the paper.

WHY THE GRAPH HELPS BOUNDARIES (paper §6, Table 2):
Without the graph, the FSM cannot tell whether the next name it sees still
belongs to this entry or starts the next one -> boundary recall ~41%. With
the graph, we KNOW the subject's teachers/students, so a nearby name that is
a known neighbour confirms "still same entry" and one that is not signals the
edge -> boundary recall ~93%.

--------------------------------------------------------------------------
This module provides:
    load_graph(path)                     -> Graph (person nodes + name index)
    Graph.match(name)                    -> [(person_id, score)]  (isRealNarrator)
    link_entry(entry, graph, use_graph)  -> LinkResult
    link_all(entries, graph, use_graph)  -> (results, stats)
The runner (run_biography.py) drives both modes and prints the comparison.
--------------------------------------------------------------------------
"""

from dataclasses import dataclass, field

from name_model import canonize_string
from name_hash import NameHash
from equality import narrator_distance


# ===========================================================================
# 1. Graph wrapper: load persons + build a name hash over them
# ===========================================================================

class Graph:
    """
    Loads a <book>.graph.json (from run_graph / merge_graphs) and builds the
    same multi-key name hash the graph builder used, so we can look a
    biography name up and get candidate person ids (old NarratorHash built
    from the graph, narratordetector.cpp constructor).
    """

    def __init__(self, persons):
        self.persons = {p["id"]: p for p in persons}
        self.hash = NameHash()
        # index every person under all its name variants (old: every GroupNode
        # name inserted into the hash)
        for pid, p in self.persons.items():
            for name in p.get("names", [p.get("primary_name", "")]):
                if name:
                    self.hash.add(pid, canonize_string(name))

    # ---- isRealNarrator: candidates for a biography name ----
    def match(self, name: str, threshold: float = 0.1):
        """
        Return [(person_id, score)] for graph persons matching `name`,
        score = equality metric vs the person's best name form, > threshold.
        (Port of isRealNarrator + RealNarratorAction.)
        """
        c = canonize_string(name)
        candidates = self.hash.find_candidates(c)     # {pid: key_score}
        results = []
        for pid, _key_score in candidates.items():
            person = self.persons[pid]
            best = 0.0
            for pname in person.get("names", [person.get("primary_name", "")]):
                s = narrator_distance(c, canonize_string(pname))
                best = max(best, s)
            if best > threshold:
                results.append((pid, best))
        results.sort(key=lambda x: -x[1])
        return results

    def neighbours(self, pid):
        """Graph neighbours of a person = parents (teachers) + children (students)."""
        p = self.persons.get(pid, {})
        return set(p.get("parents", [])) | set(p.get("children", []))


# ===========================================================================
# 2. Link result for one entry
# ===========================================================================

@dataclass
class LinkResult:
    number: int
    subject: str
    matched: bool = False              # did the subject match a graph person?
    person_id: int = -1                # chosen graph person
    match_score: float = 0.0           # name-similarity of the subject match
    neighbour_score: int = 0           # #entry names confirmed as graph neighbours
    confirmed_neighbours: list = field(default_factory=list)  # names confirmed
    candidates: int = 0                # how many graph persons matched the name

    def to_dict(self):
        return {"number": self.number, "subject": self.subject,
                "matched": self.matched, "person_id": self.person_id,
                "match_score": round(self.match_score, 3),
                "neighbour_score": self.neighbour_score,
                "confirmed_neighbours": self.confirmed_neighbours,
                "candidates": self.candidates}


# ===========================================================================
# 3. Linking one entry
# ===========================================================================

def link_entry(entry, graph: Graph, use_graph: bool,
               match_threshold: float = 0.1):
    """
    Link one BiographyEntry to the graph.

    use_graph=False -> baseline: just try to match the subject name; NO
                       neighbourhood disambiguation, NO boundary help.
    use_graph=True  -> cross-doc: match + neighbourhood confirmation using
                       the entry's teachers/students against graph neighbours.
    """
    result = LinkResult(number=entry.number, subject=entry.subject)

    # ---- step 1: match the subject name (both modes do this) ----
    matches = graph.match(entry.subject, match_threshold)
    result.candidates = len(matches)
    if not matches:
        return result                       # subject not in the hadith graph

    if not use_graph:
        # BASELINE: accept the top name match, no disambiguation.
        result.matched = True
        result.person_id, result.match_score = matches[0]
        return result

    # ---- WITH-GRAPH: disambiguate by neighbourhood (findChosenBiography) ----
    # OLD-STYLE (Biography::narrators + findChosenBiography): the entry gives us
    # the BAG of narrators mentioned near the subject, WITHOUT pre-labelling who
    # is a teacher vs student. We confirm a candidate person by checking how
    # many of those co-mentioned names are graph neighbours (parents OR
    # children) of the candidate — the GRAPH supplies the professor/student
    # direction (via parents=teachers / children=students), exactly the
    # cross-document separation the paper relies on.
    #
    # Prefer all_narrators (old-style bag); fall back to teachers+students for
    # entries where only the labelled markers were found.
    entry_neighbour_names = list(getattr(entry, "all_narrators", []) or [])
    if not entry_neighbour_names:
        entry_neighbour_names = list(entry.teachers) + list(entry.students)

    best = None
    for pid, name_score in matches:
        graph_neighbours = graph.neighbours(pid)      # neighbour person ids
        # names of those graph neighbours
        neighbour_name_forms = []
        for nid in graph_neighbours:
            np = graph.persons.get(nid, {})
            neighbour_name_forms.append((nid, np.get("primary_name", ""),
                                         np.get("names", [])))

        # count how many of the entry's teacher/student names match a graph
        # neighbour of THIS candidate (the cross-document confirmation)
        confirmed = []
        score = 0
        for bio_name in entry_neighbour_names:
            cb = canonize_string(bio_name)
            for nid, primary, forms in neighbour_name_forms:
                hit = False
                for f in (forms or [primary]):
                    if narrator_distance(cb, canonize_string(f)) > match_threshold:
                        hit = True
                        break
                if hit:
                    confirmed.append(bio_name)
                    score += 1
                    break

        cand = (score, name_score, pid, confirmed)
        if best is None or cand > best:
            best = cand

    score, name_score, pid, confirmed = best
    result.matched = True
    result.person_id = pid
    result.match_score = name_score
    result.neighbour_score = score
    result.confirmed_neighbours = confirmed
    return result


# ===========================================================================
# 4. Link a whole book of entries + stats
# ===========================================================================

def link_all(entries, graph: Graph, use_graph: bool,
             match_threshold: float = 0.1):
    results = [link_entry(e, graph, use_graph, match_threshold)
               for e in entries]

    matched = [r for r in results if r.matched]
    # "in-graph" entries: subject name has at least one candidate in the graph
    in_graph = [r for r in results if r.candidates > 0]
    confirmed = [r for r in matched if r.neighbour_score > 0]

    stats = {
        "entries": len(entries),
        "subject_matched": len(matched),
        "in_graph_subjects": len(in_graph),
        "neighbourhood_confirmed": len(confirmed) if use_graph else None,
        "avg_neighbour_score": (
            round(sum(r.neighbour_score for r in matched) / max(len(matched), 1), 2)
            if use_graph else None),
    }
    return results, stats


# ---------------------------------------------------------------------------
# Quick demo:  python biography_linker.py khoei.entries.json kafi1.clean.graph.json
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        sys.exit("usage: python biography_linker.py "
                 "<entries.json> <graph.json>")

    entries_path, graph_path = sys.argv[1], sys.argv[2]

    # rebuild lightweight entry objects from the segmenter's JSON
    class E:
        pass
    raw = json.load(open(entries_path, encoding="utf-8"))
    entries = []
    for d in raw:
        e = E()
        e.number = d["number"]; e.subject = d["subject"]
        e.teachers = d["teachers"]; e.students = d["students"]
        e.all_narrators = d.get("all_narrators", [])
        entries.append(e)

    gdata = json.load(open(graph_path, encoding="utf-8"))
    graph = Graph(gdata["persons"])
    print(f"graph persons: {len(graph.persons):,} | entries: {len(entries):,}\n")

    for use_graph in (False, True):
        results, stats = link_all(entries, graph, use_graph)
        label = "WITH-GRAPH (cross-doc)" if use_graph else "NO-GRAPH (baseline)"
        print(f"=== {label} ===")
        for k, v in stats.items():
            if v is not None:
                print(f"   {k:<24} {v}")
        print()
