"""
run_graph.py
============
IDEA: the end-to-end runner for Part 2 — YOUR tool.

    py -3.11 run_graph.py kafi1.hadiths.json
    py -3.11 run_graph.py kafi1.hadiths.json --threshold 0.2 --radius 4
    py -3.11 run_graph.py kafi1.hadiths.json --qualifiers on
    py -3.11 run_graph.py kafi1.hadiths.json --no-break-cycles

Takes the hadith JSON produced by Part 1 (run_hadith.py), rebuilds the
chains, builds the narrator graph, and writes next to it:

    <book>.graph.json    the full graph (persons, names, edges, ranks)
    <book>.graph.dot     a small Graphviz subgraph for visualisation
    + statistics printed to the console

Old equivalent: test_GraphFunctionalities() (graph.h:2009) which built the
graph after segmentation and serialised it to a .por file.

All parameters are CLI flags (the old system exposed them in the GUI).
"""

import argparse
import json
import sys
import time
from pathlib import Path

from models import (Chain, Narrator, NameConnector, NamePrim,
                    NarratorConnector, ConnectorType)
from graph_build import GraphParams
from narrator_graph import NarratorGraph


# ===========================================================================
# 1. Rebuild Part-1 objects from the saved JSON
#    (Part 1 saved narrators as dicts; we restore Narrator/Chain objects)
# ===========================================================================

def narrator_from_dict(d) -> Narrator:
    n = Narrator()
    for part in d.get("parts", []):
        if "kind" in part:                       # a connector
            kind = ConnectorType(part["kind"]) if part["kind"] in \
                [c.value for c in ConnectorType] else ConnectorType.OTHER
            n.parts.append(NameConnector(part["text"], part["start"],
                                         part["end"], kind))
        else:                                    # a name piece
            n.parts.append(NamePrim(part["text"], part["start"],
                                    part["end"], part.get("learned", False)))
    n.is_rasoul = d.get("is_rasoul", False)
    n.is_relative = d.get("is_relative", False)
    return n


def load_chains(json_path: Path):
    """Read <book>.hadiths.json -> list of Chain objects (+ hadith numbers)."""
    with open(json_path, encoding="utf-8") as f:
        hadiths = json.load(f)

    chains, numbers = [], []
    for h in hadiths:
        chain = Chain()
        for i, nd in enumerate(h["sanad"]["narrators"]):
            if i:
                chain.add(NarratorConnector("عن", 0, 0))
            chain.add(narrator_from_dict(nd))
        chains.append(chain)
        numbers.append(h.get("number", -1))
    return chains, numbers


# ===========================================================================
# 2. Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build the narrator graph from Part-1 hadith output")
    parser.add_argument("hadiths_json",
                        help="path to <book>.hadiths.json from run_hadith.py")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="merge threshold (old equality_threshold=0.1)")
    parser.add_argument("--radius", type=int, default=3,
                        help="generation radius (old equality_radius=3)")
    parser.add_argument("--delta", type=float, default=0.2,
                        help="cycle-breaking threshold step")
    parser.add_argument("--qualifiers", choices=["off", "on"], default="off",
                        help="off = old behaviour; on = use nisba (our extra)")
    parser.add_argument("--no-break-cycles", action="store_true",
                        help="skip cycle breaking (for comparison)")
    parser.add_argument("--dot-nodes", type=int, default=60,
                        help="how many nodes to export to .dot")
    parser.add_argument("--metric", choices=["structural", "hybrid"],
                        default="structural",
                        help="structural = old metric; hybrid = +AraBERT "
                             "embeddings boost (our extra)")
    parser.add_argument("--context-check", action="store_true",
                        help="equalHelper-style child check (opt-in; can split "
                             "prolific narrators — use for comparison)")
    args = parser.parse_args()

    path = Path(args.hadiths_json)
    if not path.exists():
        sys.exit(f"file not found: {path}")

    print(f"loading chains from {path.name} ...")
    chains, numbers = load_chains(path)
    total_occurrences = sum(c.narrator_count for c in chains)
    print(f"  chains: {len(chains):,} | narrator occurrences: "
          f"{total_occurrences:,}")

    params = GraphParams(equality_threshold=args.threshold,
                         equality_radius=args.radius,
                         qualifiers_mode=args.qualifiers,
                         metric_mode=args.metric,
                         context_check=args.context_check)
    graph = NarratorGraph(params,
                          break_cycles=not args.no_break_cycles,
                          delta=args.delta)

    print("building graph (merging narrators)...")
    t0 = time.time()
    graph.build(chains)
    elapsed = time.time() - t0

    # ---- outputs ----
    base = path.name.replace(".hadiths.json", "")
    graph_path = path.with_name(base + ".graph.json")
    dot_path = path.with_name(base + ".graph.dot")
    graph.save(graph_path)
    graph.to_dot(dot_path, max_nodes=args.dot_nodes)

    # ---- report ----
    stats = graph.stats()
    print(f"\ndone in {elapsed:.1f}s")
    print("-" * 46)
    for key in ("chain_occurrences", "unique_persons", "merged_persons",
                "edges", "hash_keys", "cycles_broken", "ranked_persons",
                "max_rank"):
        if key in stats:
            print(f"  {key:<22} {stats[key]:,}")
    if stats.get("unique_persons"):
        ratio = stats["chain_occurrences"] / stats["unique_persons"]
        print(f"  {'occurrences/person':<22} {ratio:.1f}")
    print("-" * 46)

    print("\ntop connected narrators (most students):")
    for name, count in stats.get("top_connected", []):
        print(f"  {count:>4} students   {name}")

    # most-repeated persons: a good sanity signal (the famous narrators
    # should be the most frequent ones)
    frequent = sorted(graph.persons.values(),
                      key=lambda p: p.occurrence_count, reverse=True)[:10]
    print("\nmost frequent narrators (occurrences):")
    for p in frequent:
        print(f"  {p.occurrence_count:>5}x  rank={p.rank:<3} {p.primary_name}")

    print("\noutputs:")
    print("  ", graph_path)
    print("  ", dot_path)
    print("\n(render the graph:  dot -Tsvg "
          f"{dot_path.name} -o {base}.graph.svg)")


if __name__ == "__main__":
    main()
