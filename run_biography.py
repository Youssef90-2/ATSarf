"""
run_biography.py
================
The paper's cross-document experiment, end to end (paper §4, §6 Table 2).

    py -3.11 run_biography.py khoei.txt kafi1.clean.graph.json --gold khoei.bio.gold.json

WHAT IT DOES
    rijal book --normalize--> clean text --engine--> tokens
              --BiographyFSM--> flat narrator mentions   (the old narratorList)
              --BiographyDetector--> graph-derived entry spans
              --agreement.score--> segmentation / detection / boundary

run TWICE, with the SAME automaton, differing in one argument:

    NO-GRAPH    BiographyFSM(confirm=None)
    WITH-GRAPH  BiographyFSM(confirm=GraphIndex.confirm)

That single switch is the paper's independent variable. Everything else —
the FSM, its parameters, the detector, the scorer — is identical, which is
what makes the two columns comparable.

--------------------------------------------------------------------------
WHAT THIS FILE REPLACES, and why it had to be replaced.

The previous version compared two DIFFERENT methods and scored them against
gold produced by the very regex that made the predictions:

  * `boundary_gold.py` built the "truth" by calling BiographySegmenter —
    the same segmenter under test. Gold == prediction, so every number it
    produced was meaningless. Deleted.
  * the no-graph baseline was `if len(candidates) == 1: correct`, a
    name-uniqueness heuristic that is not the paper's baseline and cannot
    win. The paper's baseline is the same FSM with the graph switched off,
    which is what `confirm=None` now gives.
  * the metric was a set-membership ratio over listed neighbours. The
    paper's boundary metric is at WORD granularity (agreement.py).

--------------------------------------------------------------------------
NO GOLD? Then no statistics — by design. Following the old system
(AbstractTwoLevelAgreement.cpp:178), a missing gold file causes the run to
WRITE A SEED from its own output, tell you to correct it, and stop. Scoring
against uncorrected output is the trap described above.
"""

import argparse
import json
import sys
from pathlib import Path

from agreement import TwoLevelItem, score, score_one_level, format_report
from bio_fsm import BiographyFSM, BioParams
from biography_detector import BiographyDetector, BiographyParams
from gold import (GoldError, bootstrap_from_biographies, load_for_scoring,
                  validate)
from narrator_matcher import GraphIndex
from normalization import normalize


# ===========================================================================
# 1. Tokens
# ===========================================================================

def analyze_text(clean_text, use_wojood=True, lexicon_only=False):
    """
    Produce TokenInfo for the whole book.

    `lexicon_only` skips CAMeL and Wojood and uses the closed-class lexicons
    plus the phrase matcher alone. It exists so the pipeline can be exercised
    on a machine without the models installed — it is NOT a valid setting for
    reported results, and the runner says so loudly.
    """
    if not lexicon_only:
        try:
            from engine import ArabicEngine
            engine = ArabicEngine(use_wojood=use_wojood)
            return engine.analyze_cached(clean_text)
        except ImportError as exc:
            sys.exit(
                f"morphology layer unavailable ({exc}).\n"
                "Install camel_tools, or pass --lexicon-only for a structural "
                "smoke test (results from that mode are NOT reportable).")

    from engine import (TokenInfo, tokenize_with_positions, ArabicEngine,
                        PUNCTUATION_CHARS)
    shell = ArabicEngine.__new__(ArabicEngine)          # no model loading
    tokens = [TokenInfo(word=w, start=s, end=e)
              for w, s, e in tokenize_with_positions(clean_text)]
    word_idx = []
    for i, t in enumerate(tokens):
        if t.word in PUNCTUATION_CHARS:
            t.is_punct = True
        elif t.word.isdigit():
            t.is_number = True
        else:
            word_idx.append(i)
    for i in word_idx:
        shell._apply_lexicon_flags(tokens[i])
        if tokens[i].is_nrc or tokens[i].is_nmc or tokens[i].is_rasoul:
            tokens[i].is_name = False
    shell._apply_phrase_flags(clean_text, tokens)
    shell._apply_context_name_rule(tokens)
    return tokens


# ===========================================================================
# 2. One condition
# ===========================================================================

def run_condition(tokens, graph_index, bio_params, det_params):
    """
    graph_index=None -> the paper's no-graph baseline.
    Returns (mentions, detected, fsm, detector).
    """
    confirm = graph_index.confirm if graph_index is not None else None
    fsm = BiographyFSM(bio_params, confirm=confirm)
    mentions = fsm.run(tokens)
    detector = BiographyDetector(mentions, graph_index, det_params) \
        if graph_index is not None else None
    detected = detector.detect() if detector else []
    return mentions, detected, fsm, detector


def as_predictions(detected, mentions):
    """Detected entries -> TwoLevelItem, the shape the scorer consumes."""
    return [TwoLevelItem(main=(b.start, b.end),
                         names=[(mentions[i].start, mentions[i].end)
                                for i in b.mention_indices])
            for b in detected]


def narrator_spans(mentions, confirmed_only=False):
    """
    The flat list of narrator name spans — what TABLE 2 scores.

    This must NOT be routed through the detected entries. Table 2 is
    "Narrator detection in biographies": its two column groups are whether a
    narrator was found at all, and whether the span OF HIS NAME is right. It
    says nothing about entry boundaries, which are Table 3. Scoring it through
    `detected` would collapse the no-graph column to zero, because that
    condition derives no entries by construction.

    `confirmed_only` reproduces the original's filter. addNarrators
    (narratordetector.cpp:186-205) appends a narrator to `narratorList` only
    when its graph match list is non-empty:

        if (size > 0) { narratorList.append((*biography)[i]); ... }

    So in the cross-document condition the original DISCARDS every narrator the
    graph does not recognise. That single test is what produces both halves of
    the Table 2 trade-off: detection recall falls (narrators are thrown away)
    while boundary recall rises (a narrator whose span is wrong canonizes onto
    nobody, so it is exactly what gets thrown away). The graph is a filter, not
    a corrector — it never adjusts a span.

    Without this flag the two columns would be numerically identical, since the
    automaton produces the same spans either way.
    """
    return [(m.start, m.end) for m in mentions
            if not confirmed_only or m.confirmed]


def gold_narrator_spans(gold_items):
    """Every gold narrator span, flattened out of its entry."""
    return sorted(n for item in gold_items for n in item.names)


def restrict_to_detected(gold_spans, pred_spans):
    """
    Keep only the gold narrators that some prediction overlaps.

    THIS IS THE PAPER'S DENOMINATOR, not a convenience. §6 says ANGE
    "detected 40% of the boundaries OF THE EXTRACTED NARRATORS" — the boundary
    column is conditioned on detection, and the original computes it that way
    by construction: in overLapMainFinished a gold name that matches nothing is
    skipped with `k++` and never appended to `tagWords`, so it never reaches
    countCorrect (AbstractTwoLevelAgreement.cpp:123-131).

    Our scorer deliberately diverges: it counts unmatched gold names with zero
    overlap, because leaving them out inflates recall. That correction is right
    for a standalone metric, but it makes the paper's headline movement
    impossible to reproduce — discarding a narrator would then always LOWER
    boundary recall, whereas in the original it raises it by removing the
    badly-bounded names from the denominator entirely.

    So we report both: "all gold" (our strict reading) and "detected only"
    (the original's, and the one Table 2's numbers actually describe).
    """
    return [g for g in gold_spans
            if any(g[0] <= p[1] and p[0] <= g[1] for p in pred_spans)]


def describe(label, mentions, detected, fsm):
    runs = len(getattr(fsm, "candidates", []))
    return {
        "condition": label,
        "mentions": len(mentions),
        "confirmed_mentions": sum(1 for m in mentions if m.confirmed),
        "narrator_runs": runs,
        "runs_ended_by_budget": fsm.stats.get("runs_ended_by_budget", 0),
        "entries_detected": len(detected),
        "avg_entry_chars": (round(sum(b.end - b.start for b in detected)
                                  / len(detected)) if detected else 0),
    }


# ===========================================================================
# 3. Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Cross-document biography experiment (paper Table 2)")
    ap.add_argument("rijal_txt", help="raw rijal book, e.g. khoei.txt")
    ap.add_argument("graph_json", help="<book>.graph.json from run_graph.py")
    ap.add_argument("--gold", default=None,
                    help="reviewed gold JSON; omit to write a seed and stop")
    ap.add_argument("--k", type=int, default=1,
                    help="k-reachable radius (1 = the old system's hardcoded value)")
    ap.add_argument("--near", type=int, default=100,
                    help="areNear window in characters (old bio_nrc_max)")
    ap.add_argument("--bio-threshold", type=float, default=2.0,
                    help="min confirmed neighbours for an entry (old bio_threshold)")
    ap.add_argument("--sequential", action="store_true",
                    help="enable sequential disambiguation (OUR addition, "
                         "absent from the original)")
    ap.add_argument("--lexicon-only", action="store_true",
                    help="skip CAMeL/Wojood — smoke test only, not reportable")
    ap.add_argument("--no-wojood", action="store_true")
    ap.add_argument("--out", default="biography_eval.json")
    args = ap.parse_args()

    rijal, gpath = Path(args.rijal_txt), Path(args.graph_json)
    for p in (rijal, gpath):
        if not p.exists():
            sys.exit(f"file not found: {p}")

    clean_text, _index_map = normalize(rijal.read_text(encoding="utf-8"))
    index = GraphIndex.load(gpath)

    if index.stats()["persons_reparsed_from_string"]:
        print("! this graph predates canonical-form serialization; regenerate "
              "it with run_graph.py for exact matching\n")

    print(f"book          : {rijal.name}  ({len(clean_text):,} clean chars)")
    print(f"graph         : {gpath.name}  "
          f"({index.stats()['persons']:,} persons, "
          f"threshold {index.threshold})")
    if args.lexicon_only:
        print("mode          : LEXICON-ONLY — smoke test, results not reportable")
    print()

    tokens = analyze_text(clean_text, use_wojood=not args.no_wojood,
                          lexicon_only=args.lexicon_only)
    print(f"tokens        : {len(tokens):,}\n")

    bio_params = BioParams()
    det_params = BiographyParams(near_max_chars=args.near,
                                 threshold=args.bio_threshold, k=args.k,
                                 sequential_disambiguation=args.sequential)

    ng_m, ng_d, ng_f, _ = run_condition(tokens, None, bio_params, det_params)
    wg_m, wg_d, wg_f, wg_det = run_condition(tokens, index, bio_params,
                                             det_params)

    ng_desc, wg_desc = (describe("no-graph", ng_m, ng_d, ng_f),
                        describe("with-graph", wg_m, wg_d, wg_f))

    print("=" * 64)
    print("PIPELINE — same automaton, one switch")
    print("=" * 64)
    print(f"{'':<28}{'NO-GRAPH':>17}{'WITH-GRAPH':>17}")
    print("-" * 64)
    for key, label in (("mentions", "narrator mentions"),
                       ("confirmed_mentions", "  confirmed by graph"),
                       ("narrator_runs", "narrator runs"),
                       ("runs_ended_by_budget", "  ended by budget"),
                       ("entries_detected", "entries detected"),
                       ("avg_entry_chars", "  avg entry chars")):
        print(f"{label:<28}{ng_desc[key]:>17}{wg_desc[key]:>17}")
    print("=" * 64)
    if wg_det:
        print("disambiguation:", wg_det.disambiguation_stats)
    print()

    # ---------------------------------------------------------------- gold
    if not args.gold:
        seed_path = rijal.with_suffix("").with_suffix(".bio.gold.seed.json")
        bootstrap_from_biographies(wg_d, wg_m, clean_text,
                                   source=rijal.name).save(seed_path)
        print("NO GOLD GIVEN -> no statistics produced.")
        print(f"A seed was written from this run's own output:\n    {seed_path}")
        print("Correct the spans, set \"reviewed\": true, then re-run with")
        print(f"    --gold {seed_path.name}")
        print("\nScoring against an uncorrected seed would compare the system")
        print("to itself — which is exactly what the deleted boundary_gold.py did.")
        return 0

    try:
        gold = load_for_scoring(args.gold, clean_text)
    except GoldError as exc:
        sys.exit(f"\n{exc}")

    problems = validate(gold, clean_text)
    if problems:
        print(f"! {len(problems)} problem(s) in the gold file:")
        for p in problems[:5]:
            print("   -", p)
        print()

    gold_items = gold.to_items()

    # ------------------------------------------------------------- TABLE 2
    # NARRATOR level. Scored on the flat mention list, never through the
    # detected entries — see narrator_spans() for why. The cross column
    # applies the original's `if (size > 0)` filter.
    gold_names = gold_narrator_spans(gold_items)
    ng_spans = narrator_spans(ng_m)
    wg_spans = narrator_spans(wg_m, confirmed_only=True)

    t2_ng = score_one_level(gold_names, ng_spans, clean_text)
    t2_wg = score_one_level(gold_names, wg_spans, clean_text)

    # the paper's denominator: boundaries OF THE EXTRACTED NARRATORS
    t2_ng_det = score_one_level(restrict_to_detected(gold_names, ng_spans),
                                ng_spans, clean_text)
    t2_wg_det = score_one_level(restrict_to_detected(gold_names, wg_spans),
                                wg_spans, clean_text)

    print("=" * 64)
    print("TABLE 2 — narrator detection in biographies")
    print("=" * 64)
    print(f"{'':<28}{'NO-CROSS':>17}{'CROSS-DOC':>17}")
    print("-" * 64)
    print(f"{'  narrators scored':<28}{len(ng_spans):>17,}{len(wg_spans):>17,}")
    for metric in ("recall", "precision"):
        print(f"{'  detection ' + metric:<28}"
              f"{t2_ng['detection'][metric]:>17}"
              f"{t2_wg['detection'][metric]:>17}")
    print("-" * 64)
    print("  boundary, all gold narrators        (our strict reading)")
    for metric in ("recall", "precision"):
        print(f"{'    min ' + metric:<28}"
              f"{t2_ng['boundary_min'][metric]:>17}"
              f"{t2_wg['boundary_min'][metric]:>17}")
    print("-" * 64)
    print("  boundary, detected narrators only   (the paper's denominator)")
    for metric in ("recall", "precision"):
        print(f"{'    min ' + metric:<28}"
              f"{t2_ng_det['boundary_min'][metric]:>17}"
              f"{t2_wg_det['boundary_min'][metric]:>17}")
    print("=" * 64)
    print("the paper reports detection 0.94 -> 0.65, boundary 0.41 -> 0.93.")
    print("Its boundary column is conditioned on detection, so compare it")
    print("against the SECOND block; see restrict_to_detected().")
    print()

    # ------------------------------------------------------------- TABLE 3
    # ENTRY level. Here the no-cross column is legitimately empty: without the
    # graph no entry boundary can be derived at all.
    ng_score = score(gold_items, as_predictions(ng_d, ng_m), clean_text)
    wg_score = score(gold_items, as_predictions(wg_d, wg_m), clean_text)

    print("=" * 64)
    print("TABLE 3 — biography entry detection")
    print("=" * 64)
    print(f"{'':<26}{'NO-CROSS':>18}{'CROSS-DOC':>18}")
    print("-" * 64)
    print(f"{'  entries detected':<26}{len(ng_d):>18,}{len(wg_d):>18,}")
    for key, label in (("segmentation", "entry segmentation"),
                       ("boundary_max", "entry boundary (max)")):
        for metric in ("recall", "precision"):
            print(f"{'  ' + label + ' ' + metric:<26}"
                  f"{ng_score[key][metric]:>18}{wg_score[key][metric]:>18}")
    print("=" * 64)
    if not ng_d:
        print("no-cross detects zero entries by construction: entry boundaries")
        print("are derived from the graph, so with the graph off there are none.")
    print()
    print(format_report(wg_score, "WITH-GRAPH, full two-level report"))

    Path(args.out).write_text(json.dumps(
        {"pipeline": {"no_graph": ng_desc, "with_graph": wg_desc},
         "table2_narrator": {
             "all_gold": {"no_cross": t2_ng, "cross_doc": t2_wg},
             "detected_only": {"no_cross": t2_ng_det,
                               "cross_doc": t2_wg_det}},
         "table3_entry": {"no_cross": ng_score, "cross_doc": wg_score},
         "gold": gold.stats()}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\nsaved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
