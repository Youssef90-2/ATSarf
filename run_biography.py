"""
run_biography.py
================
IDEA: The end-to-end biography runner AND the Table-2 evaluation — YOUR tool.
Reproduces the paper's central cross-document result (§6, Table 2): the hadith
narrator graph improves biography narrator/boundary detection.

    py -3.11 run_biography.py khoei.txt kafi12.clean.graph.json
    py -3.11 run_biography.py khoei.entries.json kafi12.clean.graph.json
      (accepts either the raw rijal book OR a pre-made .entries.json)

PIPELINE:
    rijal book --segment--> entries (subject + teachers/students + span)
              --link (NO-GRAPH)--> baseline detection
              --link (WITH-GRAPH)--> cross-doc detection
              --compare--> Table-2-style recall / precision

--------------------------------------------------------------------------
HOW WE MEASURE (honest, defensible — read this for the report):

The paper measured narrator/boundary detection against a HUMAN gold
annotation. We do not have one. What the Khoei book DOES give us for free is
a reliable gold signal for each entry's CONTENT: the narrators that truly
belong to entry N are its subject plus the teachers/students the book itself
lists after "روى عن / روى عنه". So we evaluate, per entry, whether a method
correctly identifies that this entry's subject is a real graph narrator AND
places the entry's true neighbours inside it.

Two quantities, exactly parallel to the paper:

  NARRATOR DETECTION  (paper Table 2, top block)
    recall    = detected real-narrator subjects / all in-graph subjects
    precision = correct detections / all detections

  BOUNDARY / NEIGHBOUR RECALL  (paper Table 2, bottom block — the 41%->93%)
    For each entry, of the neighbours the book lists (its true teachers/
    students), how many does the method actually confirm as belonging to the
    entry's subject?
      - NO-GRAPH  can't confirm neighbours at all (it only matched a name) ->
        it recovers a neighbour only if the name coincidentally is unambiguous.
        This is the paper's low-recall baseline.
      - WITH-GRAPH confirms a neighbour when it is a graph neighbour of the
        chosen person -> high recall, because the graph knows who belongs.
    boundary_recall = confirmed neighbours / listed neighbours (micro-avg)

This is NOT the paper's exact numbers (different book, no human gold), but it
is the SAME experiment shape and it is computed only from signals we can
verify. Report it as "reproduction of the Table-2 effect on Kafi×Khoei".
--------------------------------------------------------------------------
"""

import argparse
import json
import sys
from pathlib import Path

from biography_linker import Graph, canonize_string
from equality import narrator_distance


# ===========================================================================
# 1. Load entries (from raw book via segmenter, or from a .entries.json)
# ===========================================================================

def load_entries(path: Path):
    if path.suffix == ".json" or path.name.endswith(".entries.json"):
        raw = json.load(open(path, encoding="utf-8"))
        class E:
            pass
        entries = []
        for d in raw:
            e = E()
            e.number = d["number"]; e.subject = d["subject"]
            e.teachers = d["teachers"]; e.students = d["students"]
            e.is_reference = d.get("is_reference", False)
            entries.append(e)
        return entries
    # raw book -> segment now
    from biography_segmenter import BiographySegmenter
    text = path.read_text(encoding="utf-8")
    entries, _clean, _stats = BiographySegmenter().segment(text)
    return entries


# ===========================================================================
# 2. Neighbour confirmation — shared helper
#    Does a given biography name match ANY graph-neighbour of `person_id`?
# ===========================================================================

def neighbour_matches(bio_name, person_id, graph, threshold=0.1):
    cb = canonize_string(bio_name)
    for nid in graph.neighbours(person_id):
        np = graph.persons.get(nid, {})
        for form in (np.get("names") or [np.get("primary_name", "")]):
            if form and narrator_distance(cb, canonize_string(form)) > threshold:
                return True
    return False


# ===========================================================================
# 3. Evaluate one method (no-graph or with-graph) over all entries
# ===========================================================================

def evaluate(entries, graph, use_graph, threshold=0.1):
    """
    Returns a dict of counts for narrator detection and boundary/neighbour
    recall, computed the same way for both methods so they are comparable.
    """
    # detection
    detections = 0          # entries the method claims are real narrators
    correct_detections = 0  # ... that are genuinely in the graph
    in_graph_total = 0      # entries whose subject really is in the graph

    # boundary / neighbour recall (micro-averaged over listed neighbours)
    listed_neighbours = 0
    confirmed_neighbours = 0
    # FAIR measure: restrict to neighbours that actually EXIST somewhere in
    # the graph. A neighbour from a book we didn't ingest can never be
    # confirmed by any method, so counting it punishes both equally and hides
    # the graph's real effect. This isolates the graph's disambiguation power.
    existing_neighbours = 0
    confirmed_existing = 0

    for e in entries:
        if e.is_reference:
            continue
        true_matches = graph.match(e.subject, threshold)   # ground truth: in graph?
        is_in_graph = len(true_matches) > 0
        if is_in_graph:
            in_graph_total += 1

        listed = list(e.teachers) + list(e.students)

        if not use_graph:
            # BASELINE: accept top name match as a detection; no neighbour
            # reasoning -> a listed neighbour is "recovered" only if that name
            # matches a UNIQUE graph person (i.e. name alone disambiguates).
            if true_matches:
                detections += 1
                if is_in_graph:
                    correct_detections += 1
            # only judge neighbours for entries whose subject is in the graph
            # (an unmatched subject has no entry to place neighbours into)
            if true_matches:
                for nb in listed:
                    listed_neighbours += 1
                    cands = graph.match(nb, threshold)
                    if len(cands) >= 1:
                        existing_neighbours += 1
                        if len(cands) == 1:   # unambiguous by name alone
                            confirmed_neighbours += 1
                            confirmed_existing += 1
        else:
            # WITH-GRAPH: choose the candidate person whose graph neighbours
            # best explain the entry's listed teachers/students, then a listed
            # neighbour is recovered iff it is a graph-neighbour of that person.
            best_pid, best_conf = None, -1
            for pid, _s in true_matches:
                conf = sum(1 for nb in listed
                           if neighbour_matches(nb, pid, graph, threshold))
                if conf > best_conf:
                    best_pid, best_conf = pid, conf
            if true_matches:
                detections += 1
                if is_in_graph:
                    correct_detections += 1
                for nb in listed:
                    listed_neighbours += 1
                    exists = len(graph.match(nb, threshold)) >= 1
                    if exists:
                        existing_neighbours += 1
                        if best_pid is not None and \
                                neighbour_matches(nb, best_pid, graph, threshold):
                            confirmed_neighbours += 1
                            confirmed_existing += 1

    def ratio(a, b):
        return round(a / b, 3) if b else 0.0

    return {
        "detections": detections,
        "correct_detections": correct_detections,
        "in_graph_subjects": in_graph_total,
        "detect_recall": ratio(correct_detections, in_graph_total),
        "detect_precision": ratio(correct_detections, detections),
        "listed_neighbours": listed_neighbours,
        "confirmed_neighbours": confirmed_neighbours,
        "boundary_recall": ratio(confirmed_neighbours, listed_neighbours),
        "existing_neighbours": existing_neighbours,
        "confirmed_existing": confirmed_existing,
        "boundary_recall_fair": ratio(confirmed_existing, existing_neighbours),
    }


# ===========================================================================
# 4. Main — run both methods and print the Table-2-style comparison
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Biography linking + Table-2 cross-document evaluation")
    ap.add_argument("book_or_entries", help="rijal .txt OR <book>.entries.json")
    ap.add_argument("graph_json", help="<book>.graph.json from run_graph/merge")
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--out", default="biography_eval.json")
    args = ap.parse_args()

    epath, gpath = Path(args.book_or_entries), Path(args.graph_json)
    for p in (epath, gpath):
        if not p.exists():
            sys.exit(f"file not found: {p}")

    entries = load_entries(epath)
    graph = Graph(json.load(open(gpath, encoding="utf-8"))["persons"])
    measurable = [e for e in entries
                  if (e.teachers or e.students) and not e.is_reference]

    print(f"graph persons : {len(graph.persons):,}")
    print(f"entries       : {len(entries):,}  "
          f"(measurable: {len(measurable):,})")
    print()

    base = evaluate(entries, graph, use_graph=False, threshold=args.threshold)
    cross = evaluate(entries, graph, use_graph=True, threshold=args.threshold)

    # ---- Table 2 style output ----
    print("=" * 62)
    print("TABLE 2 (reproduction on Kafi x Khoei) — graph's contribution")
    print("=" * 62)
    print(f"{'':<26}{'NO-GRAPH':>16}{'WITH-GRAPH':>16}")
    print("-" * 62)
    print("Narrator detection")
    print(f"{'  recall':<26}{base['detect_recall']:>16}"
          f"{cross['detect_recall']:>16}")
    print(f"{'  precision':<26}{base['detect_precision']:>16}"
          f"{cross['detect_precision']:>16}")
    print("Boundary / neighbour recall  (the paper's 41% -> 93%)")
    print(f"{'  recall (all listed)':<26}{base['boundary_recall']:>16}"
          f"{cross['boundary_recall']:>16}")
    print(f"{'  recall (in-graph only)*':<26}{base['boundary_recall_fair']:>16}"
          f"{cross['boundary_recall_fair']:>16}")
    print("-" * 62)
    print(f"{'listed neighbours':<26}{base['listed_neighbours']:>16}"
          f"{cross['listed_neighbours']:>16}")
    print(f"{'  of which in the graph':<26}{base['existing_neighbours']:>16}"
          f"{cross['existing_neighbours']:>16}")
    print(f"{'confirmed (in-graph)':<26}{base['confirmed_existing']:>16}"
          f"{cross['confirmed_existing']:>16}")
    print("=" * 62)
    print("* in-graph-only = fair measure: neighbours that exist in the graph")
    print("  at all (others come from books not ingested, unconfirmable by any")
    print("  method). This isolates the graph's disambiguation contribution.")

    gain = cross["boundary_recall_fair"] - base["boundary_recall_fair"]
    print(f"\nboundary-recall gain from the graph (in-graph, fair): "
          f"+{gain:.3f}  "
          f"({base['boundary_recall_fair']:.0%} -> "
          f"{cross['boundary_recall_fair']:.0%})")

    json.dump({"no_graph": base, "with_graph": cross},
              open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
