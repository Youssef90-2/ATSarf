"""
run_equality.py
===============
Paper §5 — does the hierarchical name metric actually beat plain edit
distance? Port of `narrator_equality_comparision` (narratorEqualityComparision.cpp).

    py -3.11 run_equality.py khoei.equal.json
    py -3.11 run_equality.py legacy.equal --qt
    py -3.11 run_equality.py --bootstrap kafi1.clean.graph.json --n 150

This is the ONLY evaluation that isolates the distance metric. Everything
else (Tables 1-3) measures it entangled with segmentation and graph building,
so a metric change shows up there only faintly. Here it is the whole signal —
and each item is a yes/no judgement on a pair of names, not a span, so a
useful gold set costs an evening rather than a week.

--------------------------------------------------------------------------
WHAT IS COMPARED

  eNarrator     normalized edit distance > 0.75          the paper's baseline
                (narratorEqualityComparision.cpp:47)
  structural    our port of equalNew, qualifiers OFF     the faithful metric
  + qualifiers  structural with nisba logic ON           OUR implementation of
                                                         paper §3.2, which the
                                                         released system never
                                                         shipped (getdistance()
                                                         is dead code)

The third column is the point. §3.2 describes conflicting nisba (العراقي vs
المصري -> reject) and reinforcing nisba (العراقي + الكوفي, Kufa being in Iraq
-> boost). `getdistance()` implements it and is never called; `equal()` routes
to `equalNew()`, whose own comment reads "till now possessives not even
compared". So this column measures a claim the paper makes and the original
never tested.

--------------------------------------------------------------------------
METRIC. Recall and precision over the POSITIVE class, exactly as the old code
computes them:

    recall     pairs correctly called same-person / pairs annotated same
    precision  pairs correctly called same-person / pairs called same

Note this says nothing directly about the negative class, though precision
punishes false positives. The old code also printed the pairs where the
annotation disagreed with BOTH methods; those are reproduced under
`--disagreements`, because they are where the metric actually needs work.
"""

import argparse
import json
import sys
from pathlib import Path

from equality import edit_distance_score, narrator_distance
from gold import (EqualityGold, GoldError, import_qt_equality,
                  load_equality_for_scoring)
from name_model import canonize_string


# ===========================================================================
# 1. The three predictors
# ===========================================================================

def predict_edit(a, b, threshold=0.75):
    """The paper's eNarrator baseline (narratorEqualityComparision.cpp:47)."""
    return edit_distance_score(a, b) > threshold


def predict_structural(a, b, threshold, qualifiers="off"):
    return narrator_distance(canonize_string(a), canonize_string(b),
                             qualifiers_mode=qualifiers) > threshold


# ===========================================================================
# 2. Evaluation
# ===========================================================================

def evaluate(pairs, predict):
    """pairs: [(a, b, is_same)] -> counts + recall/precision."""
    annotated_same = called_same = correct = 0
    false_positives = []
    false_negatives = []
    for a, b, truth in pairs:
        said = predict(a, b)
        if truth:
            annotated_same += 1
        if said:
            called_same += 1
        if truth and said:
            correct += 1
        elif truth and not said:
            false_negatives.append((a, b))
        elif said and not truth:
            false_positives.append((a, b))

    def ratio(x, y):
        return round(x / y, 4) if y else 0.0

    recall = ratio(correct, annotated_same)
    precision = ratio(correct, called_same)
    f = (round(2 * recall * precision / (recall + precision), 4)
         if (recall + precision) else 0.0)
    return {"annotated_same": annotated_same, "called_same": called_same,
            "correct": correct, "recall": recall, "precision": precision,
            "f_score": f,
            "false_positives": false_positives,
            "false_negatives": false_negatives}


def sweep(pairs, qualifiers="off", steps=None):
    """Recall/precision of the structural metric across thresholds."""
    steps = steps or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    out = []
    for t in steps:
        r = evaluate(pairs, lambda a, b, t=t: predict_structural(
            a, b, t, qualifiers))
        out.append((t, r["recall"], r["precision"], r["f_score"]))
    return out


# ===========================================================================
# 3. Bootstrap — produce the pairs a human should judge
# ===========================================================================

def bootstrap_pairs(graph_path, n=150, bands=6):
    """
    Emit candidate pairs for annotation, drawn from a built graph.

    Which pairs are worth a human's time? The ones the HASH already considers
    plausible — those are exactly the decisions the metric has to make. Random
    name pairs are almost all trivially different and teach nothing.

    Sampled evenly across score bands so the set is informative near the
    decision boundary instead of piling up at 1.0.
    """
    from narrator_matcher import GraphIndex

    index = GraphIndex.load(graph_path)
    seen, scored = set(), []
    for pid, canons in index._canon.items():
        for canonical in canons:
            for other_pid in index.hash.find_candidates(canonical):
                if other_pid == pid:
                    continue
                key = (min(pid, other_pid), max(pid, other_pid))
                if key in seen:
                    continue
                seen.add(key)
                a = index.persons[pid].get("primary_name", "")
                b = index.persons[other_pid].get("primary_name", "")
                if not a or not b or a == b:
                    continue
                score = narrator_distance(canonize_string(a),
                                          canonize_string(b))
                scored.append((score, a, b))

    scored.sort(key=lambda x: -x[0])
    per_band, out = max(1, n // bands), []
    for i in range(bands):
        lo, hi = i / bands, (i + 1) / bands
        band = [(s, a, b) for s, a, b in scored if lo <= s < hi]
        step = max(1, len(band) // per_band)
        out.extend(band[::step][:per_band])
    return [(a, b, None) for _s, a, b in out[:n]]


# ===========================================================================
# 4. Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Narrator-equality metric comparison (paper §5)")
    ap.add_argument("pairs_file", nargs="?",
                    help="labelled pairs JSON, or a legacy .equal with --qt")
    ap.add_argument("--qt", action="store_true",
                    help="pairs_file is an original Qt .equal binary")
    ap.add_argument("--bootstrap", metavar="GRAPH_JSON",
                    help="emit candidate pairs to annotate, then stop")
    ap.add_argument("--n", type=int, default=150,
                    help="how many pairs to emit when bootstrapping")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="structural-metric threshold (graph default is 0.5)")
    ap.add_argument("--edit-threshold", type=float, default=0.75,
                    help="eNarrator baseline threshold (paper's value)")
    ap.add_argument("--sweep", action="store_true",
                    help="show recall/precision across thresholds")
    ap.add_argument("--disagreements", type=int, default=0,
                    help="print N pairs where the metric was wrong")
    ap.add_argument("--out", default="equality_eval.json")
    args = ap.parse_args()

    # ---- bootstrap mode ----
    if args.bootstrap:
        pairs = bootstrap_pairs(args.bootstrap, n=args.n)
        seed = EqualityGold(source=Path(args.bootstrap).name, pairs=pairs)
        path = Path(args.bootstrap).with_suffix("").with_suffix(
            ".equal.seed.json")
        seed.save(path)
        print(f"{len(pairs)} candidate pairs written to:\n    {path}")
        print("\nLabel each pair's \"equal\" as true or false, set")
        print('"reviewed": true, then run:')
        print(f"    py -3.11 run_equality.py {path.name}")
        print("\nPairs were drawn from names the HASH already considers "
              "plausible,\nsampled across score bands — random pairs are "
              "trivially different\nand would teach the evaluation nothing.")
        return 0

    if not args.pairs_file:
        sys.exit("give a pairs file, or --bootstrap <graph.json>")

    # ---- load ----
    path = Path(args.pairs_file)
    if not path.exists():
        sys.exit(f"file not found: {path}")
    if args.qt:
        gold = import_qt_equality(path)
    else:
        try:
            gold = load_equality_for_scoring(path)
        except GoldError as exc:
            sys.exit(f"\n{exc}")

    pairs = gold.labelled()
    print(f"pairs         : {gold.stats()}")
    print(f"thresholds    : structural {args.threshold}, "
          f"edit {args.edit_threshold}")
    print()

    conditions = [
        ("eNarrator (edit)", lambda a, b: predict_edit(a, b,
                                                       args.edit_threshold)),
        ("structural", lambda a, b: predict_structural(a, b, args.threshold,
                                                       "off")),
        ("+ qualifiers", lambda a, b: predict_structural(a, b, args.threshold,
                                                         "on")),
    ]
    results = {name: evaluate(pairs, fn) for name, fn in conditions}

    print("=" * 66)
    print("NARRATOR EQUALITY — metric comparison (paper §5)")
    print("=" * 66)
    print(f"{'':<20}{'recall':>12}{'precision':>13}{'F':>10}{'called':>10}")
    print("-" * 66)
    for name, _fn in conditions:
        r = results[name]
        print(f"{name:<20}{r['recall']:>12}{r['precision']:>13}"
              f"{r['f_score']:>10}{r['called_same']:>10}")
    print("=" * 66)
    print("'+ qualifiers' implements paper §3.2 (conflicting/reinforcing")
    print("nisba), which the released system never shipped — getdistance()")
    print("is dead code. That column is ours, not a reproduction.")

    if args.sweep:
        print()
        print("threshold sweep (structural, qualifiers off)")
        print(f"  {'t':>6}{'recall':>10}{'precision':>12}{'F':>9}")
        for t, r, p, f in sweep(pairs):
            print(f"  {t:>6}{r:>10}{p:>12}{f:>9}")

    if args.disagreements:
        r = results["structural"]
        print()
        print(f"false NEGATIVES (annotated same, metric said different) "
              f"— up to {args.disagreements}:")
        for a, b in r["false_negatives"][:args.disagreements]:
            print(f"   {a}   ||   {b}")
        print(f"false POSITIVES (annotated different, metric said same) "
              f"— up to {args.disagreements}:")
        for a, b in r["false_positives"][:args.disagreements]:
            print(f"   {a}   ||   {b}")

    Path(args.out).write_text(json.dumps(
        {"gold": gold.stats(),
         "thresholds": {"structural": args.threshold,
                        "edit": args.edit_threshold},
         "results": {k: {kk: vv for kk, vv in v.items()
                         if kk not in ("false_positives", "false_negatives")}
                     for k, v in results.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
