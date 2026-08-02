"""
run_hadith.py
=============
IDEA: The end-to-end runner — YOUR testing tool.

    python run_hadith.py kafi1.txt
    python run_hadith.py musnad_ahmad.txt --no-wojood
    python run_hadith.py kafi1.txt --narr-min 4

Runs the whole pipeline (normalize -> engine -> FSM -> segmenter) on a
book file and writes, next to it:

    <book>.hadiths.json    every hadith: number, kitab/bab, sanad
                           (narrators with positions), matn
    <book>.chains.txt      human-readable dump — same idea as the old
                           system's hadith_chains.txt (hadithRefactored.cpp)
    + statistics printed to the console (your sanity check on any book)

All FSM parameters are command-line flags (the old system exposed them
in the GUI the same way — no recompiling to experiment).
"""

import argparse
import json
import sys
import time
from pathlib import Path

from engine import ArabicEngine
from fsm import FsmParams
from segmenter import HadithSegmenter, SegmenterConfig


def write_json(hadiths, out_path: Path):
    data = []
    for h in hadiths:
        d = h.to_dict()
        d["kitab"] = getattr(h, "kitab", "")
        d["bab"] = getattr(h, "bab", "")
        data.append(d)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")


def write_chains_txt(hadiths, out_path: Path):
    """Readable dump — open it in VS Code/Notepad (Arabic renders fine)."""
    lines = []
    for h in hadiths:
        lines.append("=" * 60)
        lines.append(f"hadith #{h.number}   "
                     f"[{getattr(h, 'kitab', '')} / {getattr(h, 'bab', '')}]")
        lines.append(f"narrators: {h.chain.narrator_count}")
        for i, n in enumerate(h.chain.narrators, 1):
            tag = "  << RASOUL" if n.is_rasoul else (
                  "  << relative" if n.is_relative else "")
            lines.append(f"  {i}. {n.full_name}{tag}")
        lines.append(f"matn: {h.matn_text[:300]}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Hadith segmentation runner")
    parser.add_argument("book", help="path to the book .txt file (UTF-8)")
    parser.add_argument("--narr-min", type=int, default=3,
                        help="min narrators per sanad (default 3)")
    parser.add_argument("--nmc-max", type=int, default=3)
    parser.add_argument("--nrc-max", type=int, default=5)
    parser.add_argument("--no-wojood", action="store_true",
                        help="disable Wojood NER (CAMeL+context only)")
    parser.add_argument("--camel-mode", choices=["mle", "bert"],
                        default="mle")
    parser.add_argument("--start-marker", default="",
                        help="regex; skip text before it (front matter). "
                             "Example for kafi: \"كتاب العقل\"")
    args = parser.parse_args()

    book_path = Path(args.book)
    if not book_path.exists():
        sys.exit(f"file not found: {book_path}")

    book_text = book_path.read_text(encoding="utf-8")
    print(f"book: {book_path.name}  ({len(book_text):,} chars)")

    print("loading engine...")
    engine = ArabicEngine(camel_mode=args.camel_mode,
                          use_wojood=not args.no_wojood)
    print("  wojood:", engine.wojood.available,
          "| camel mode:", args.camel_mode)

    config = SegmenterConfig(fsm=FsmParams(nmc_max=args.nmc_max,
                                           nrc_max=args.nrc_max,
                                           narr_min=args.narr_min),
                             start_marker=args.start_marker)
    segmenter = HadithSegmenter(engine, config)

    print("segmenting (first run on a big book takes a few minutes; "
          "results are cached)...")
    t0 = time.time()
    hadiths, stats, clean_text, index_map = segmenter.segment(book_text)
    elapsed = time.time() - t0

    # ---- outputs ----
    json_path = book_path.with_suffix(".hadiths.json")
    txt_path = book_path.with_suffix(".chains.txt")
    write_json(hadiths, json_path)
    write_chains_txt(hadiths, txt_path)

    # ---- console report ----
    print(f"\ndone in {elapsed:.1f}s")
    print("-" * 40)
    for key, value in stats.items():
        print(f"  {key:<24} {value:,}")
    if hadiths:
        avg = stats["total_narrators"] / len(hadiths)
        print(f"  {'avg narrators/hadith':<24} {avg:.1f}")
    print("-" * 40)
    print("outputs:")
    print("  ", json_path)
    print("  ", txt_path)

    # small preview
    print("\npreview (first 3 hadiths):")
    for h in hadiths[:3]:
        names = "  <-  ".join(n.full_name for n in h.chain.narrators)
        print(f"  #{h.number}: {names}")


if __name__ == "__main__":
    main()
