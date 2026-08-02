"""
merge_graphs.py
===============
IDEA (mergeGraphs / mergeWith, old code graph.cpp:243 + graph.h:1627):
Combine several books into ONE narrator graph, so a narrator who appears in
Kafi vol.1 AND vol.3 becomes a SINGLE person node with all their edges.

    py -3.11 merge_graphs.py kafi1.hadiths.json kafi2.hadiths.json ...
    py -3.11 merge_graphs.py kafi*.hadiths.json -o kafi_all.graph.json
    py -3.11 merge_graphs.py kafi*.hadiths.json --threshold 0.2 --qualifiers on

--------------------------------------------------------------------------
WHY THIS IS THE SIMPLE + OPTIMIZED WAY (and faithful to paper + old code):

The old system merged two ALREADY-BUILT graphs (mergeWith): it appended
graph2's chains, then re-ran the SAME matching over graph2's hash against
graph1's hash (mergeGraphs, graph.h:1031), then called init() again to
re-break cycles and re-rank. The key fact from that code: a merge is NOT a
plain append — it re-runs the exact narrator-matching (hash blocking +
equal() + generation radius) across the books, otherwise the same person
in two books stays two nodes.

We get the identical result more cheaply: instead of building N graphs and
stitching them, we CONCATENATE the chains of all books and run GraphBuilder
ONCE. The merge pass then sees every occurrence from every book in one hash,
so cross-book identical narrators merge naturally — same logic, same
matching, no second hash to reconcile. Cycle-breaking and ranks run once on
the unified graph, exactly like the old init() after mergeWith.

    old:  build(A); build(B); mergeWith  -> re-match B vs A -> init()
    here: chains(A)+chains(B) -> build() once  (re-match is the build itself)

Paper mapping (§6): "We considered sets of ... hadith documents ... and
extracted their narrator graphs ... more hadith books -> more complete
narrator graphs." A unified multi-book graph is that "more complete" graph.
--------------------------------------------------------------------------
OUTPUT: one <name>.graph.json (same shape as run_graph.py) + a .dot.
Also records, per person, WHICH books each occurrence came from, so the web
layer / graph-comparison feature can ask "who is shared between vol.1 and
vol.3?" — information the old .por file did not keep explicitly.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from models import Chain, NarratorConnector
from graph_build import GraphParams
from narrator_graph import NarratorGraph

# reuse Part-2's JSON->objects loader so the two runners stay in sync
from run_graph import narrator_from_dict


# ===========================================================================
# 1. Load one book's chains, tagging each chain with its source book
# ===========================================================================

def load_book_chains(json_path: Path, book_id: int, book_name: str):
    """
    Read one <book>.hadiths.json -> (chains, meta).

    meta[i] = {"book_id", "book", "hadith"} for chain i, so after the graph
    is built we can report which book(s) every person came from.
    """
    with open(json_path, encoding="utf-8") as f:
        hadiths = json.load(f)

    chains, meta = [], []
    for h in hadiths:
        chain = Chain()
        narrators = h["sanad"]["narrators"]
        for i, nd in enumerate(narrators):
            if i:
                chain.add(NarratorConnector("عن", 0, 0))
            chain.add(narrator_from_dict(nd))
        chains.append(chain)
        meta.append({"book_id": book_id,
                     "book": book_name,
                     "hadith": h.get("number", -1)})
    return chains, meta


# ===========================================================================
# 2. Merge = concatenate chains from all books, build ONE graph
# ===========================================================================

def merge_books(json_paths, params, break_cycles=True, delta=0.2):
    """
    json_paths -> (NarratorGraph, chain_meta, per_book_counts).

    All chains from all books share one chain-id space; GraphBuilder's merge
    pass runs once over the lot, so identical narrators across books merge.
    """
    all_chains = []
    chain_meta = []
    per_book = []

    for book_id, path in enumerate(json_paths):
        book_name = Path(path).name.replace(".hadiths.json", "")
        chains, meta = load_book_chains(Path(path), book_id, book_name)
        occ = sum(c.narrator_count for c in chains)
        per_book.append((book_name, len(chains), occ))
        all_chains.extend(chains)
        chain_meta.extend(meta)
        print(f"  loaded {book_name:<22} {len(chains):>6,} chains  "
              f"{occ:>7,} occurrences")

    print(f"\n  TOTAL {'':<22} {len(all_chains):>6,} chains")

    graph = NarratorGraph(params, break_cycles=break_cycles, delta=delta)
    graph.build(all_chains)
    return graph, chain_meta, per_book


# ===========================================================================
# 3. Annotate each person with the books it appears in (cross-book value)
# ===========================================================================

def attach_book_provenance(graph, chain_meta):
    """
    For every person, collect the set of books its occurrences came from.
    An occurrence knows its chain_id; chain_meta[chain_id] knows the book.
    Adds person.books (sorted list of book names) usable by the web layer.
    """
    for person in graph.persons.values():
        books = set()
        for occ_id in person.occurrences:
            occ = graph.chain_nodes[occ_id]
            m = chain_meta[occ.chain_id]
            books.add(m["book"])
        person.books = sorted(books)
    return graph


def save_with_provenance(graph, path):
    """Same JSON shape as NarratorGraph.save, plus per-person 'books'."""
    persons = []
    for p in graph.persons.values():
        d = p.to_dict()
        d["books"] = getattr(p, "books", [])
        persons.append(d)
    data = {
        "params": {
            "equality_threshold": graph.params.equality_threshold,
            "equality_radius": graph.params.equality_radius,
            "qualifiers_mode": graph.params.qualifiers_mode,
            "break_cycles": graph.break_cycles,
            "merged_books": True,
        },
        "persons": persons,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


# ===========================================================================
# 4. Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge several books into one unified narrator graph")
    parser.add_argument("hadiths_json", nargs="+",
                        help="two or more <book>.hadiths.json files")
    parser.add_argument("-o", "--out", default=None,
                        help="output graph path (default: merged.graph.json)")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="merge threshold (old equality_threshold=0.1)")
    parser.add_argument("--radius", type=int, default=3,
                        help="generation radius (old equality_radius=3)")
    parser.add_argument("--delta", type=float, default=0.2,
                        help="cycle-breaking threshold step")
    parser.add_argument("--qualifiers", choices=["off", "on"], default="off",
                        help="off = old behaviour; on = use nisba (our extra)")
    parser.add_argument("--no-break-cycles", action="store_true")
    parser.add_argument("--dot-nodes", type=int, default=60)
    args = parser.parse_args()

    paths = [Path(p) for p in args.hadiths_json]
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit("file(s) not found: " + ", ".join(str(m) for m in missing))
    if len(paths) < 2:
        print("note: only one book given — this just builds its graph "
              "(same as run_graph.py).")

    params = GraphParams(equality_threshold=args.threshold,
                         equality_radius=args.radius,
                         qualifiers_mode=args.qualifiers)

    print(f"merging {len(paths)} book(s):")
    t0 = time.time()
    graph, chain_meta, per_book = merge_books(
        paths, params,
        break_cycles=not args.no_break_cycles, delta=args.delta)
    attach_book_provenance(graph, chain_meta)
    elapsed = time.time() - t0

    # ---- outputs ----
    out_path = Path(args.out) if args.out else \
        paths[0].with_name("merged.graph.json")
    dot_path = out_path.with_suffix(".dot")
    save_with_provenance(graph, out_path)
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

    # cross-book signal: how many persons appear in more than one book
    multi = [p for p in graph.persons.values()
             if len(getattr(p, "books", [])) > 1]
    print(f"\npersons appearing in >1 book: {len(multi):,}")
    for p in sorted(multi, key=lambda p: p.occurrence_count,
                    reverse=True)[:10]:
        print(f"  {p.occurrence_count:>5}x  {'/'.join(p.books):<20} "
              f"{p.primary_name}")

    print("\noutputs:")
    print("  ", out_path)
    print("  ", dot_path)


if __name__ == "__main__":
    main()
