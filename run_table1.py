"""
run_table1.py
=============
Paper §6, Table 1 — narrator graph accuracy.

    py -3.11 run_table1.py kafi1.clean.hadiths.json --bootstrap
    py -3.11 run_table1.py kafi1.clean.hadiths.json --gold kafi1.clean.merge.gold.json

WHAT TABLE 1 MEASURES, and why it needs its own runner and its own gold.

Every other table in this project scores SPANS: did we mark the right stretch
of text. Table 1 scores MERGE DECISIONS. The same person appears across many
chains under different written names —

    hadith 5,  position 2 :  محمد بن يعقوب
    hadith 12, position 0 :  محمد بن يعقوب الكليني
    hadith 40, position 1 :  الكليني

— and the graph's job is to fuse those occurrences into ONE node. Table 1 asks
whether it did:

    under-merge (left one out)      -> recall falls
    over-merge  (fused the wrong person) -> precision falls

So the gold is not offsets. It is a cluster label per occurrence, keyed
"chain#position", which is exactly what `labels_from_graph` produces for the
prediction side and what a human can address without knowing internal ids.
`MergeGold` in gold.py holds it.

--------------------------------------------------------------------------
THE TWO ROWS (paper §6)

    detected  only occurrences present in BOTH mappings — the merge logic in
              isolation, excluding narrators the segmenter never found
    all       every gold occurrence; one we failed to extract becomes a
              cluster of one, so its merges are all missed

The paper reports 0.673/0.824 for "detected" and 0.632/0.771 for "all", and
attributes the 63% recall to "the conservative cycle breaking transformation
that splits merged narrator nodes even if they were equivalent" — i.e. to
under-merging. Compare our `--no-break-cycles` run to test that claim directly.

--------------------------------------------------------------------------
THE SAFEGUARD

`--bootstrap` seeds the labels from the graph's OWN clustering and writes
reviewed=false. `load_merge_for_scoring` refuses such a file, because scoring
a partition against itself returns a perfect 1.0 that means nothing — the same
trap the deleted boundary_gold.py fell into. Correct the clusters, set
"reviewed": true, then score.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from gold import (GoldError, MergeGold, bootstrap_merge_labels,
                  load_merge_for_scoring)
from graph_agreement import format_report, labels_from_graph, score_table1
from graph_build import GraphParams
from narrator_graph import NarratorGraph
from run_graph import load_chains


# ===========================================================================
# 1. Build the graph and read off both the labels and the display names
# ===========================================================================

def build(chains, args):
    params = GraphParams(equality_threshold=args.threshold,
                         equality_radius=args.radius,
                         qualifiers_mode=args.qualifiers)
    graph = NarratorGraph(params, break_cycles=not args.no_break_cycles,
                          delta=args.delta)
    graph.build(chains)
    return graph


def occurrence_names(graph):
    """
    "chain#position" -> the written name at that occurrence.

    The labels alone are unreadable to an annotator ("h12#0" says nothing), so
    the seed carries this alongside them as a read-only reference.
    """
    names = {}
    for node in graph.chain_nodes.values():
        key = f"{node.chain_id}#{node.position}"
        name = getattr(node, "name", None)
        if name is None:
            narrator = getattr(node, "narrator", None)
            name = getattr(narrator, "full_name", "") if narrator else ""
        names[key] = name
    return names


# ===========================================================================
# 2. Bootstrap — emit a partition for a human to correct
# ===========================================================================

def write_seed(graph, pred, names, out_path, source, limit_chains):
    if limit_chains:
        keep = {k for k in pred
                if int(k.split("#")[0].lstrip("c")) < limit_chains} \
            if all("#" in k for k in pred) else set(pred)
        pred = {k: v for k, v in pred.items() if k in keep}

    seed = bootstrap_merge_labels(pred, source=source)
    payload = seed.to_dict()
    # read-only reference so the annotator can see WHAT each key is; the
    # loader ignores any key it does not know.
    payload["_occurrence_names"] = {k: names.get(k, "") for k in seed.labels}
    payload["_how_to_review"] = (
        "Each entry in `labels` maps an occurrence to a cluster id. Two "
        "occurrences are 'the same person' iff they carry the SAME value. "
        "Split a wrong merge by giving one of them a new value; join a missed "
        "merge by making the values equal. The values themselves are opaque — "
        "only the grouping is compared. Then set reviewed: true."
    )
    Path(out_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return seed


def print_clusters(seed, names, limit=15):
    clusters = seed.clusters()
    ranked = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    print(f"\nlargest {min(limit, len(ranked))} clusters the graph proposes "
          f"(these are what you check first):\n")
    for label, keys in ranked[:limit]:
        forms = []
        for k in keys:
            n = names.get(k, "")
            if n and n not in forms:
                forms.append(n)
        print(f"  cluster {label}  ({len(keys)} occurrences)")
        print(f"      forms: {' | '.join(forms[:6])}"
              + (" …" if len(forms) > 6 else ""))


# ===========================================================================
# 3. Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Table 1 — narrator graph accuracy (merge decisions)")
    ap.add_argument("hadiths_json", help="<book>.hadiths.json from run_hadith.py")
    ap.add_argument("--gold", default=None,
                    help="reviewed MergeGold JSON; omit and pass --bootstrap "
                         "to write a seed")
    ap.add_argument("--bootstrap", action="store_true",
                    help="write a seed partition from the graph's own merges")
    ap.add_argument("--sample-chains", type=int, default=0,
                    help="bootstrap only the first N chains, so the annotation "
                         "job is a few evenings rather than a month")
    # graph parameters — defaults are the old system's (hadithCommon.h:50-52)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--qualifiers", choices=["off", "on"], default="off")
    ap.add_argument("--no-break-cycles", action="store_true",
                    help="skip cycle breaking — the paper blames it for the "
                         "63%% recall, so this tests that claim directly")
    ap.add_argument("--out", default=None, help="where to write the JSON result")
    args = ap.parse_args()

    src = Path(args.hadiths_json)
    if not src.exists():
        sys.exit(f"file not found: {src}")

    chains, _numbers = load_chains(src)
    print(f"book      : {src.name}  ({len(chains):,} chains)")
    graph = build(chains, args)

    pred = labels_from_graph(graph.persons, graph.chain_nodes)
    names = occurrence_names(graph)
    print(f"graph     : {len(graph.persons):,} persons, "
          f"{len(pred):,} occurrences  "
          f"(threshold {args.threshold}, radius {args.radius}, "
          f"break_cycles {not args.no_break_cycles})")

    # ----------------------------------------------------------- bootstrap
    if args.bootstrap or not args.gold:
        stem = src.name.replace(".hadiths.json", "")
        seed_path = src.with_name(f"{stem}.merge.gold.seed.json")
        seed = write_seed(graph, pred, names, seed_path, stem,
                          args.sample_chains)
        print(f"\nseed      : {json.dumps(seed.stats(), ensure_ascii=False)}")
        print_clusters(seed, names)
        print(f"\nNO GOLD GIVEN -> no statistics produced.")
        print(f"A seed partition was written from the graph's own merges:")
        print(f"    {seed_path}")
        print("Correct the clusters, set \"reviewed\": true, then re-run with")
        print(f"    --gold {seed_path.name}")
        print("\nScoring a partition against itself returns 1.0 and means "
              "nothing —\nwhich is exactly what the deleted boundary_gold.py did.")
        return 0

    # --------------------------------------------------------------- score
    try:
        gold = load_merge_for_scoring(args.gold)
    except GoldError as exc:
        sys.exit(f"\n{exc}")

    print(f"gold      : {json.dumps(gold.stats(), ensure_ascii=False)}")

    unknown = set(gold.labels) - set(pred)
    if unknown:
        print(f"\n! {len(unknown)} gold occurrence(s) the graph never produced "
              f"— they count as singletons in the 'all' row, e.g. "
              f"{sorted(unknown)[:3]}")

    table1 = score_table1(gold.labels, pred)
    print()
    print(format_report(table1))

    out = args.out or str(src.with_name(src.name.replace(".hadiths.json", "")
                                        + ".table1.json"))
    Path(out).write_text(json.dumps(
        {"book": src.name,
         "params": {"threshold": args.threshold, "radius": args.radius,
                    "delta": args.delta, "qualifiers": args.qualifiers,
                    "break_cycles": not args.no_break_cycles},
         "gold": gold.stats(), "table1": table1},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
