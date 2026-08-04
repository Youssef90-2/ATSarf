"""
partial_annotation.py
=====================
Paper §4.1 — annotating a PARTIAL narrator graph. The paper's second headline
result: 80% recall / 89% precision (§6).

    "In most cases, a scholar is interested in only annotating a partial
     narrator graph Gp that he or she extracted using ANGE from a selected set
     of hadith of interest, e.g. the set may contain narrations related in
     topic. For each narrator n in Gp, ANGE computes Ns and Np, the children
     and parents of n in Gp. We compute Nb the intersection of {n} ∪ Ns ∪ Np
     and lb, and sort Nb by the order of appearance in the biography books.
     ANGE considers the clusters in Nb that contain n and that happen in the
     same text locality as the candidate biographies of n and ranks them in
     terms of the number of matches."

--------------------------------------------------------------------------
HOW THIS DIFFERS FROM FULL-BOOK SEGMENTATION

Both live in the same C++ class, distinguished by two overrides:

                        modifyNodes()              checkBiography()
    BiographySegmenter  every narrator in the book  keep top-1, REJECT spans
                                                    overlapping an emitted one
    NarratorDetector    the nodes the scholar
    (the BASE class,    picked (an input dialog,    keep ALL top-3, no overlap
     narratordetector.  narratordetector.cpp:346)   rejection (:375 returns
     cpp:16)                                        true unconditionally)

So partial annotation is not a different algorithm — it is the same scoring
with a different question. Segmentation asks "where does each entry end?" and
must therefore partition the book. Annotation asks "which entries might
describe THIS narrator?" and should return a RANKED SHORTLIST, because the
scholar is the one who decides.

That is also why the paper measures it differently: §6 says "the accuracy of
biography boundary detection is not well defined in this task since the
partial graph annotation method reports several biographies ranked with a
similarity metric."

--------------------------------------------------------------------------
WHY THE NEIGHBOURS COME FROM Gp, NOT THE FULL GRAPH

This is the point of the method. In the full graph a prolific narrator has
hundreds of neighbours, so "a neighbour appears nearby" is weak evidence. In a
graph built from ten topically-related hadiths, a narrator has a handful of
neighbours and their co-occurrence in a biography entry is strong evidence.
The restriction is what makes the ranking sharp.
"""

from dataclasses import dataclass, field

from bio_fsm import BiographyFSM, BioParams
from biography_detector import BiographyDetector, BiographyParams
from graph_build import GraphParams
from narrator_graph import NarratorGraph
from narrator_matcher import GraphIndex


# ===========================================================================
# 1. Selecting the hadiths and building Gp
# ===========================================================================

def select_hadiths(hadiths, numbers=None, contains=None, limit=10):
    """
    Pick the "set of hadith of interest". The paper's example is topical
    relatedness, which is a retrieval question, not an ANGE question — so this
    offers the two selectors a scholar actually has:

        numbers   explicit hadith numbers
        contains  a substring of the matn (a crude topic filter)

    `limit` defaults to 10, the paper's "sets of at most ten hadith documents".
    """
    chosen = hadiths
    if numbers:
        wanted = set(numbers)
        chosen = [h for h in chosen if h.get("number") in wanted]
    if contains:
        chosen = [h for h in chosen
                  if contains in h.get("matn", {}).get("text", "")]
    return chosen[:limit] if limit else chosen


def build_partial_graph(chains, params: GraphParams = None):
    """
    Gp — a narrator graph over the selected hadiths only. Same builder as the
    full graph; the only difference is how few chains go in.
    """
    graph = NarratorGraph(params or GraphParams(equality_threshold=0.5))
    graph.build(chains)
    return graph


def graph_to_index(graph) -> GraphIndex:
    """Wrap a freshly built NarratorGraph for matching, without a round trip."""
    return GraphIndex({
        "params": {"equality_threshold": graph.params.equality_threshold},
        "persons": [p.to_dict() for p in graph.persons.values()],
    })


# ===========================================================================
# 2. The annotation of one node
# ===========================================================================

@dataclass
class Annotation:
    """The candidate biographies for one narrator of Gp, best first."""
    person_id: int
    name: str
    neighbours_in_gp: list = field(default_factory=list)
    candidates: list = field(default_factory=list)   # [DetectedBiography]

    @property
    def annotated(self) -> bool:
        return bool(self.candidates)

    @property
    def best(self):
        return self.candidates[0] if self.candidates else None

    def to_dict(self):
        return {
            "person_id": self.person_id,
            "name": self.name,
            "neighbours_in_gp": list(self.neighbours_in_gp),
            "candidates": [
                {"rank": i + 1, "score": c.score,
                 "start": c.start, "end": c.end,
                 "confirmed_neighbours": list(c.confirmed_neighbours)}
                for i, c in enumerate(self.candidates)],
        }


# ===========================================================================
# 3. The annotator
# ===========================================================================

class PartialAnnotator:
    """
    Annotate every node of a partial graph with its candidate biographies.

        gp   = build_partial_graph(selected_chains)
        ann  = PartialAnnotator(gp)
        out  = ann.annotate(biography_tokens)
    """

    def __init__(self, partial_graph, bio_params: BioParams = None,
                 det_params: BiographyParams = None):
        self.graph = partial_graph
        self.index = graph_to_index(partial_graph)
        self.bio_params = bio_params or BioParams()
        # `max_candidates` is the paper's ranked shortlist; the old code fixes
        # it at 3 (MAX_SIZE, narratordetector.cpp:159).
        self.det_params = det_params or BiographyParams(max_candidates=3)
        self.mentions = []
        self.detector = None

    def annotate(self, tokens, person_ids=None):
        """
        tokens      : the biography book, analysed (lb comes from these)
        person_ids  : which nodes of Gp to annotate; default = all of them
        """
        fsm = BiographyFSM(self.bio_params, confirm=self.index.confirm)
        self.mentions = fsm.run(tokens)
        self.detector = BiographyDetector(self.mentions, self.index,
                                          self.det_params)

        targets = (list(person_ids) if person_ids is not None
                   else list(self.graph.persons))

        out = []
        for pid in targets:
            person = self.graph.persons.get(pid)
            if person is None:
                continue
            neighbours = sorted(person.parents | person.children)
            out.append(Annotation(
                person_id=pid,
                name=person.primary_name,
                neighbours_in_gp=[self.graph.persons[n].primary_name
                                  for n in neighbours
                                  if n in self.graph.persons],
                candidates=self.detector.candidates_for(pid)))
        # most confidently annotated first
        out.sort(key=lambda a: -(a.best.score if a.best else -1))
        return out

    def stats(self, annotations):
        annotated = [a for a in annotations if a.annotated]
        return {
            "nodes_in_gp": len(self.graph.persons),
            "nodes_requested": len(annotations),
            "nodes_annotated": len(annotated),
            "annotation_rate": (round(len(annotated) / len(annotations), 3)
                                if annotations else 0.0),
            "mentions_in_biography": len(self.mentions),
            "confirmed_mentions": sum(1 for m in self.mentions if m.confirmed),
            "avg_candidates": (round(sum(len(a.candidates) for a in annotated)
                                     / len(annotated), 2) if annotated else 0),
            "avg_best_score": (round(sum(a.best.score for a in annotated)
                                     / len(annotated), 2) if annotated else 0),
        }


# ===========================================================================
# 4. Scoring, when gold exists
# ===========================================================================

def score_annotations(annotations, gold_map, at_k=1):
    """
    Evaluate against {person_id: (start, end)} — the entry a human says
    describes that narrator.

    A hit means a candidate's span OVERLAPS the gold entry, at rank <= at_k.
    `at_k=1` is the strict reading; the paper reports a ranked shortlist, so
    at_k=3 is the honest companion number and both should be given.

        recall     annotated correctly / gold entries available
        precision  annotated correctly / nodes we annotated at all
    """
    from spans import overlaps

    available = sum(1 for a in annotations if a.person_id in gold_map)
    attempted = correct = 0
    for a in annotations:
        if not a.annotated:
            continue
        attempted += 1
        target = gold_map.get(a.person_id)
        if target is None:
            continue
        for cand in a.candidates[:at_k]:
            if overlaps(cand.start, cand.end, target[0], target[1]):
                correct += 1
                break

    def ratio(x, y):
        return round(x / y, 4) if y else 0.0

    return {"at_k": at_k, "gold_available": available, "attempted": attempted,
            "correct": correct,
            "recall": ratio(correct, available),
            "precision": ratio(correct, attempted)}


def format_report(stats, scored=None):
    lines = ["PARTIAL GRAPH ANNOTATION (paper §4.1)", "=" * 58]
    for k, v in stats.items():
        lines.append(f"  {k:<26}{v:>16}")
    if scored:
        lines.append("-" * 58)
        lines.append(f"{'':<26}{'recall':>10}{'precision':>12}")
        for s in (scored if isinstance(scored, list) else [scored]):
            lines.append(f"  {'at rank <= ' + str(s['at_k']):<24}"
                         f"{s['recall']:>10}{s['precision']:>12}")
        lines.append("  (paper reports 0.80 recall / 0.89 precision)")
    lines.append("=" * 58)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick demo:  python partial_annotation.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(__doc__.strip().splitlines()[1])
    print()
    print("Worked example:  py -3.11 tests/test_a8.py")
    print("Runner:          py -3.11 run_partial.py <hadiths.json> <rijal.txt>")
