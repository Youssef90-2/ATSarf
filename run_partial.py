"""
run_partial.py
==============
Paper §4.1 — annotate a PARTIAL narrator graph with candidate biographies.

    py -3.11 run_partial.py kafi1.clean.hadiths.json khoei.txt --contains العقل
    py -3.11 run_partial.py kafi1.clean.hadiths.json khoei.txt --numbers 1 2 3 5

Pick a small set of topically-related hadiths, build Gp from just those, then
for every narrator in Gp report a RANKED shortlist of the biography entries
that might describe them.

This is a different question from `run_biography.py`. That one segments a
whole rijal book into entries and must therefore partition it. This one asks,
for a handful of narrators a scholar cares about, "which entries describe
them?" — and answers with a shortlist, because the scholar decides. The paper
is explicit that boundary accuracy is not well defined here for exactly that
reason (§6).
"""

import argparse
import json
import sys
from pathlib import Path

from bio_fsm import BioParams
from biography_detector import BiographyParams
from graph_build import GraphParams
from models import Chain, NarratorConnector
from normalization import normalize
from partial_annotation import (PartialAnnotator, build_partial_graph,
                                format_report, select_hadiths)
from run_biography import analyze_text
from run_graph import narrator_from_dict


def chains_from_hadiths(hadiths):
    chains = []
    for h in hadiths:
        c = Chain()
        for i, nd in enumerate(h["sanad"]["narrators"]):
            if i:
                c.add(NarratorConnector("عن", 0, 0))
            c.add(narrator_from_dict(nd))
        chains.append(c)
    return chains


def main():
    ap = argparse.ArgumentParser(description="Partial graph annotation (§4.1)")
    ap.add_argument("hadiths_json", help="<book>.clean.hadiths.json")
    ap.add_argument("rijal_txt", help="raw rijal book, e.g. khoei.txt")
    ap.add_argument("--numbers", type=int, nargs="*", default=None,
                    help="hadith numbers to include")
    ap.add_argument("--contains", default=None,
                    help="include hadiths whose matn contains this text")
    ap.add_argument("--limit", type=int, default=10,
                    help="max hadiths in Gp (the paper's 'at most ten')")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="graph equality threshold")
    ap.add_argument("--near", type=int, default=100)
    ap.add_argument("--bio-threshold", type=float, default=2.0)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--lexicon-only", action="store_true",
                    help="skip CAMeL/Wojood — smoke test, not reportable")
    ap.add_argument("--out", default="partial_annotation.json")
    args = ap.parse_args()

    hpath, rpath = Path(args.hadiths_json), Path(args.rijal_txt)
    for p in (hpath, rpath):
        if not p.exists():
            sys.exit(f"file not found: {p}")

    if not args.numbers and not args.contains:
        sys.exit("give --numbers or --contains: §4.1 annotates a SELECTED set "
                 "of hadiths, not a whole book (use run_biography.py for that)")

    hadiths = json.loads(hpath.read_text(encoding="utf-8"))
    selected = select_hadiths(hadiths, numbers=args.numbers,
                              contains=args.contains, limit=args.limit)
    if not selected:
        sys.exit("no hadith matched the selection")

    gp = build_partial_graph(chains_from_hadiths(selected),
                             GraphParams(equality_threshold=args.threshold))

    print(f"selected hadiths : {len(selected)} "
          f"({[h.get('number') for h in selected]})")
    print(f"Gp               : {len(gp.persons)} narrators, "
          f"{sum(len(p.children) for p in gp.persons.values())} edges")
    if args.lexicon_only:
        print("mode             : LEXICON-ONLY — not reportable")
    print()

    clean_text, _ = normalize(rpath.read_text(encoding="utf-8"))
    tokens = analyze_text(clean_text, lexicon_only=args.lexicon_only)

    annotator = PartialAnnotator(
        gp, BioParams(),
        BiographyParams(near_max_chars=args.near, threshold=args.bio_threshold,
                        k=args.k, max_candidates=3))
    annotations = annotator.annotate(tokens)
    stats = annotator.stats(annotations)

    print(format_report(stats))
    print()
    for a in annotations:
        if not a.annotated:
            continue
        print(f"{a.name}")
        print(f"   neighbours in Gp: {a.neighbours_in_gp}")
        for i, c in enumerate(a.candidates, 1):
            snippet = clean_text[c.start:min(c.end + 1, c.start + 70)]
            print(f"   #{i}  score {c.score}  [{c.start}-{c.end}]  {snippet}...")
        print()

    unannotated = [a.name for a in annotations if not a.annotated]
    if unannotated:
        print(f"not annotated ({len(unannotated)}): "
              f"{', '.join(unannotated[:8])}"
              f"{' ...' if len(unannotated) > 8 else ''}")
        print("(a narrator with no biography entry in this book, or one whose "
              "Gp neighbours never co-occur near it)")

    Path(args.out).write_text(json.dumps(
        {"selected": [h.get("number") for h in selected],
         "stats": stats,
         "annotations": [a.to_dict() for a in annotations]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
