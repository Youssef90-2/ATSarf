"""
quality_filter.py
=================
IDEA: A post-segmentation quality gate. Part 1 (run_hadith.py) sometimes
turns a stretch of MATN into a fake sanad: three ordinary words that happen
to look like names (فرعون / الله / رجلا) pass the FSM's narr_min=3 and get
emitted as a "hadith". On kafi1 this produced 436 hadiths with number=-1
(the segmenter could not attach a real "N -" number to them) — and ~187 of
those are pure matn noise that later inflates the narrator graph
("الله ايمانهم بولايتنا" appeared as a narrator 370x after merging).

This file is a SEPARATE, non-destructive step. It does NOT touch the FSM or
the segmenter (both tested and wired into other things). It reads a
<book>.hadiths.json, scores each hadith, and writes:

    <book>.clean.hadiths.json   accepted hadiths (feed this to the graph)
    <book>.rejected.json        what was dropped + why (for your review)

run_graph.py / merge_graphs.py then run on the .clean.hadiths.json exactly
as before — same format, fewer fakes.

--------------------------------------------------------------------------
THE RULE (validated on real kafi1 data — see the numbers below):

A hadith is a REAL sanad if ANY of these holds:
    (1) some narrator name contains بن / ابن        -> "فلان بن فلان"
    (2) the sanad ends at a rasoul/imam             -> honorific present
        (عليه السلام / رسول الله / النبي ...) or a narrator is_rasoul
    (3) a narration connector (عن / حدثنا ...) sits between narrators

Signals come from the system's OWN lexicons (lexicons.is_ibn_word,
is_rasoul_word, is_nrc_word) — not a hand-written list — so the filter
stays consistent with how the rest of the pipeline classifies words.

MEASURED ON kafi1 (1,740 hadiths):
    - false-drop on the 1,304 numbered (known-good) hadiths : 0  (0.00%)
    - of the 436 number=-1 hadiths: 187 dropped as matn, 249 kept
      (real sanads whose "N -" number was just too far away)
So the filter removes noise without discarding a single verified hadith.

WHY WE KEEP number=-1 hadiths that pass the rule: a missing number is a
NUMBERING problem (segmenter._number_for only looks back 30 chars), not
proof the sanad is fake. Dropping them would lose ~249 real chains.
--------------------------------------------------------------------------
"""

import argparse
import json
import sys
from pathlib import Path

import lexicons


# ===========================================================================
# 1. The quality test for one hadith dict (same shape as hadiths.json)
# ===========================================================================

def sanad_signals(hadith: dict):
    """
    Inspect one hadith dict; return the dict of boolean signals found.
    Works purely on the saved JSON (name strings + is_rasoul flag), so it
    needs no re-analysis.
    """
    narrators = hadith.get("sanad", {}).get("narrators", [])
    joined = " ".join(n.get("name", "") for n in narrators)
    words = joined.split()

    has_ibn = any(lexicons.is_ibn_word(w) for w in words)
    # `is_rasoul` now marks only references to the Prophet HIMSELF, so an
    # honorific ending (ابي عبد الله عليه السلام) no longer sets it. Detect
    # the honorific as a PHRASE — not word by word, because 'الله' alone is a
    # component of ordinary names (عبد الله) and would match everything.
    has_rasoul = (any(n.get("is_rasoul") for n in narrators)
                  or any(lexicons.is_rasoul_word(w) for w in words)
                  or any(p in joined for p in lexicons.HONORIFIC_PHRASES))
    # narration word appearing inside a narrator name (عن/حدثنا that the FSM
    # folded into a name rather than storing as a connector)
    has_nrc = any(lexicons.is_nrc_word(w) for w in words)

    return {
        "has_ibn": has_ibn,
        "has_rasoul": has_rasoul,
        "has_nrc": has_nrc,
        "narrator_count": len(narrators),
        "numbered": hadith.get("number", -1) != -1,
    }


def is_real_sanad(hadith: dict) -> bool:
    """
    Accept the hadith as a real sanad if it shows any real-sanad structure.
    Numbered hadiths are always kept (they carried a real 'N -' marker).
    """
    if hadith.get("number", -1) != -1:
        return True                       # verified by its book number
    s = sanad_signals(hadith)
    return s["has_ibn"] or s["has_rasoul"] or s["has_nrc"]


def rejection_reason(hadith: dict) -> str:
    """A short human-readable reason a hadith was dropped (for review)."""
    s = sanad_signals(hadith)
    return (f"no بن / no honorific / no narration word; "
            f"{s['narrator_count']} narrators, number=-1")


# ===========================================================================
# 2. Filter a whole book
# ===========================================================================

def filter_hadiths(hadiths: list):
    """Return (accepted, rejected) lists of hadith dicts."""
    accepted, rejected = [], []
    for h in hadiths:
        if is_real_sanad(h):
            accepted.append(h)
        else:
            rejected.append({
                "hadith": h,
                "reason": rejection_reason(h),
                "sanad_preview": " | ".join(
                    n.get("name", "") for n in
                    h.get("sanad", {}).get("narrators", []))[:120],
            })
    return accepted, rejected


# ===========================================================================
# 3. Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Drop matn-noise 'hadiths' before graph building")
    parser.add_argument("hadiths_json",
                        help="path to <book>.hadiths.json from run_hadith.py")
    parser.add_argument("--out", default=None,
                        help="clean output path "
                             "(default: <book>.clean.hadiths.json)")
    parser.add_argument("--show", type=int, default=10,
                        help="how many rejected samples to print")
    args = parser.parse_args()

    path = Path(args.hadiths_json)
    if not path.exists():
        sys.exit(f"file not found: {path}")

    with open(path, encoding="utf-8") as f:
        hadiths = json.load(f)

    accepted, rejected = filter_hadiths(hadiths)

    # outputs
    base = path.name.replace(".hadiths.json", "")
    out_path = Path(args.out) if args.out else \
        path.with_name(base + ".clean.hadiths.json")
    rej_path = path.with_name(base + ".rejected.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(accepted, f, ensure_ascii=False, indent=2)
    with open(rej_path, "w", encoding="utf-8") as f:
        json.dump(rejected, f, ensure_ascii=False, indent=2)

    # report
    total = len(hadiths)
    kept, dropped = len(accepted), len(rejected)
    numbered_kept = sum(1 for h in accepted if h.get("number", -1) != -1)
    unnumbered_kept = kept - numbered_kept
    print(f"book: {path.name}")
    print("-" * 46)
    print(f"  total hadiths        {total:,}")
    print(f"  kept                 {kept:,}  "
          f"({numbered_kept:,} numbered + {unnumbered_kept:,} unnumbered)")
    print(f"  dropped as matn      {dropped:,}  "
          f"({100*dropped/total:.1f}%)")
    print("-" * 46)

    if rejected and args.show:
        print(f"\nsample of dropped 'hadiths' (first {args.show}):")
        for r in rejected[:args.show]:
            print("  ", r["sanad_preview"])

    print("\noutputs:")
    print("  ", out_path, "  <-- feed THIS to run_graph.py / merge_graphs.py")
    print("  ", rej_path, "  (review what was removed)")


if __name__ == "__main__":
    main()
