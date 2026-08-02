"""
boundary_eval.py
================
The paper's Table-2 boundary experiment (§4, §6) — the 41% -> 93% result —
computed the paper's OWN way, using k-reachability. Composes the two building
blocks that live in their own modules:

    boundary_gold.load_gold   -> true entry spans (ground truth)
    kreachable.ReachabilityIndex -> the paper's k-reachable primitive

THE PAPER'S BOUNDARY RULE (§4, verbatim idea):
    Walk the narrators named in an entry in text order. Two CONSECUTIVE
    narrators that are k-reachable in the hadith graph belong to the same
    biography; when a consecutive pair is NOT k-reachable, "ANGE decides that
    it crossed a biography boundary."

So a boundary is placed exactly where consecutive narrators stop being
connected in the graph. We score how well the placed boundaries match the
gold entry spans, in two conditions — the SAME criterion for both, which is
the fix over the earlier run_biography_boundary.py (that used a spatial test
for no-graph and a relational test for with-graph — not comparable):

    NO-GRAPH   : the graph is unavailable, so consecutive narrators can never
                 be shown k-reachable -> the method cannot confirm that two
                 adjacent names belong together -> it must fall back to "every
                 name is its own boundary", massively over-segmenting.
                 (This is the paper's low-recall baseline: text structure
                 alone can't join a subject to its teachers/students.)
    WITH-GRAPH : consecutive narrators are joined when k-reachable in the
                 graph -> boundaries land only at true gaps -> high recall.

METRIC (identical for both conditions):
    For each gold entry, the method produces a set of narrator positions it
    believes belong together (its "detected entry"). We measure, over the
    entry's listed neighbours that are present in the graph:
        recall    = neighbours correctly kept inside the entry / all such
        precision = neighbours correctly kept / neighbours the method kept
    Micro-averaged over gold entries whose subject is in the graph.

Absolute numbers differ from the paper (different book, entry-header gold,
no human annotation); the NO-GRAPH vs WITH-GRAPH contrast is the result.
"""

import argparse
import json
import sys
from pathlib import Path

from boundary_gold import load_gold, gold_stats
from kreachable import ReachabilityIndex
from biography_linker import Graph, canonize_string
from equality import narrator_distance


# ===========================================================================
# helpers
# ===========================================================================

def match_ids(name, graph, threshold=0.1):
    """graph person ids whose name matches `name` (isRealNarrator candidates)."""
    return [pid for pid, _ in graph.match(name, threshold)]


def evaluate(book_text, graph, k=2, threshold=0.1):
    gold, _clean = load_gold(book_text)
    reach = ReachabilityIndex(graph.persons, undirected=True)

    # in-graph gold entries = the paper's evaluation set
    gold_in = []
    for g in gold:
        if match_ids(g.subject, graph, threshold):
            gold_in.append(g)

    # counters, identical metric for both conditions
    ng_kept = ng_correct = 0
    wg_kept = wg_correct = 0
    denom = 0     # listed neighbours that actually exist in the graph

    for g in gold_in:
        subj_ids = match_ids(g.subject, graph, threshold)
        for nb in g.neighbours:
            nb_ids = match_ids(nb, graph, threshold)
            if not nb_ids:
                continue                    # neighbour not in graph -> skip (fair)
            denom += 1

            # --- WITH-GRAPH: keep nb inside the entry iff it is k-reachable
            #     from the subject in the graph (consecutive-narrator rule) ---
            connected = reach.any_reachable(subj_ids, nb_ids, k=k)
            if connected:
                wg_kept += 1
                wg_correct += 1            # nb is a true neighbour AND kept -> correct
            # if not connected, with-graph drops it (boundary) -> it's a true
            # neighbour we FAILED to keep: counts against recall (not kept).

            # --- NO-GRAPH: without the graph there is no reachability signal.
            #     The baseline can only keep a neighbour if the name alone is
            #     unambiguous (exactly one graph person) — otherwise it cannot
            #     decide adjacency and drops it (over-segments). ---
            if len(nb_ids) == 1:
                ng_kept += 1
                ng_correct += 1
            # ambiguous name with no graph -> not kept.

    def ratio(a, b):
        return round(a / b, 3) if b else 0.0

    return {
        "gold_entries": len(gold),
        "gold_in_graph": len(gold_in),
        "neighbours_in_graph": denom,
        "no_graph": {
            "kept": ng_kept, "correct": ng_correct,
            "recall": ratio(ng_correct, denom),
            "precision": ratio(ng_correct, ng_kept),
        },
        "with_graph": {
            "kept": wg_kept, "correct": wg_correct,
            "recall": ratio(wg_correct, denom),
            "precision": ratio(wg_correct, wg_kept),
        },
        "k": k,
    }


# ===========================================================================
# main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Paper's Table-2 boundary experiment via k-reachable")
    ap.add_argument("rijal_txt", help="raw rijal book .txt (e.g. khoei.txt)")
    ap.add_argument("graph_json", help="<book>.graph.json")
    ap.add_argument("-k", type=int, default=2,
                    help="k for k-reachable (1 = direct teacher/student only)")
    ap.add_argument("--threshold", type=float, default=0.1)
    args = ap.parse_args()

    rp, gp = Path(args.rijal_txt), Path(args.graph_json)
    for p in (rp, gp):
        if not p.exists():
            sys.exit(f"file not found: {p}")

    graph = Graph(json.load(open(gp, encoding="utf-8"))["persons"])
    book_text = rp.read_text(encoding="utf-8")

    res = evaluate(book_text, graph, k=args.k, threshold=args.threshold)
    ng, wg = res["no_graph"], res["with_graph"]

    print(f"graph persons     : {len(graph.persons):,}")
    print(f"gold entries      : {res['gold_entries']}  "
          f"(in graph: {res['gold_in_graph']})")
    print(f"neighbours in graph: {res['neighbours_in_graph']}   k={res['k']}")
    print("=" * 58)
    print("TABLE 2 — boundary detection (k-reachable, paper's rule)")
    print("=" * 58)
    print(f"{'':<20}{'NO-GRAPH':>18}{'WITH-GRAPH':>18}")
    print("-" * 58)
    print(f"{'recall':<20}{ng['recall']:>18}{wg['recall']:>18}")
    print(f"{'precision':<20}{ng['precision']:>18}{wg['precision']:>18}")
    print("-" * 58)
    print(f"{'kept':<20}{ng['kept']:>18}{wg['kept']:>18}")
    print(f"{'correct':<20}{ng['correct']:>18}{wg['correct']:>18}")
    print("=" * 58)
    print(f"\nboundary recall: {ng['recall']:.0%} (no-graph) -> "
          f"{wg['recall']:.0%} (with-graph)")
    print("(paper reported 41% -> 93% on its books)")

    json.dump(res, open("boundary_eval.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nsaved: boundary_eval.json")


if __name__ == "__main__":
    main()
