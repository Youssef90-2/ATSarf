"""
check_graph.py — Phase-2 regression harness.

Rebuilds the narrator graph from <book>.clean.hadiths.json and reports the
five PASS CONDITIONS that need no gold annotation:

    1. same-chain merges          must be 0   (old mustMerge veto, graph.h:900)
    2. groups exceeding span r    must be 0   (old span radius, graph.h:914)
    3. back-edges (cycles)        must be 0   (DAG property, paper §2)
    4. largest node occurrences   sanity      (over-merge indicator)
    5. rank-0 count / imam rank   sanity      (old computeRanks, graph.h:498)

usage:  python check_graph.py [book.clean.hadiths.json] [threshold]
"""
import json
import sys
import pathlib
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from run_graph import narrator_from_dict           # noqa: E402
from models import Chain, NarratorConnector        # noqa: E402
from graph_build import GraphParams                # noqa: E402
from narrator_graph import NarratorGraph           # noqa: E402

BOOK = sys.argv[1] if len(sys.argv) > 1 else \
    r"c:\Users\Youssef\Desktop\ATSarf-FYP\ATSarf\kafi1.clean.hadiths.json"
THRESHOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
RADIUS = 3


def load_chains(path):
    chains = []
    for h in json.load(open(path, encoding="utf-8")):
        c = Chain()
        for i, nd in enumerate(h["sanad"]["narrators"]):
            if i:
                c.add(NarratorConnector("عن", 0, 0))
            c.add(narrator_from_dict(nd))
        chains.append(c)
    return chains


def back_edges(persons):
    """DFS over children edges; count back-edges (cycle evidence)."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {pid: WHITE for pid in persons}
    found = []
    for start in persons:
        if color[start] != WHITE:
            continue
        stack = [(start, iter(persons[start].children))]
        color[start] = GRAY
        while stack:
            node, it = stack[-1]
            advanced = False
            for nxt in it:
                if nxt not in persons:
                    continue
                if color[nxt] == GRAY:
                    found.append((node, nxt))
                elif color[nxt] == WHITE:
                    color[nxt] = GRAY
                    stack.append((nxt, iter(persons[nxt].children)))
                    advanced = True
                    break
            if not advanced:
                color[node] = BLACK
                stack.pop()
    return found


def main():
    chains = load_chains(BOOK)
    graph = NarratorGraph(GraphParams(equality_threshold=THRESHOLD,
                                      equality_radius=RADIUS))
    persons = graph.build(chains)
    cn = graph.chain_nodes

    # 1. same-chain merges
    same_chain = 0
    for p in persons.values():
        seen = set()
        for oid in p.occurrences:
            ch = cn[oid].chain_id
            if ch in seen:
                same_chain += 1
            seen.add(ch)

    # 2. span violations. Rasoul is exempt: the old system indexes every
    #    rasoul occurrence under one key (narratorHash.h:284) precisely
    #    because the Prophet/Imam legitimately appears at any depth.
    span_bad = 0
    for p in persons.values():
        if p.is_rasoul:
            continue
        pos = [cn[o].position for o in p.occurrences]
        if pos and max(pos) - min(pos) > RADIUS:
            span_bad += 1

    # 3. cycles
    be = back_edges(persons)

    # 4. over-merge
    top = sorted(persons.values(), key=lambda p: p.occurrence_count,
                 reverse=True)[:5]

    # 5. ranks
    rank0 = sum(1 for p in persons.values() if p.rank == 0)
    imam = [p for p in persons.values() if p.primary_name.strip() ==
            "ابي عبد الله"]

    print(f"book      : {Path(BOOK).name}")
    print(f"threshold : {THRESHOLD}   radius: {RADIUS}")
    print("=" * 62)
    print(f"{'occurrences':<28}{len(cn):>12,}")
    print(f"{'persons':<28}{len(persons):>12,}")
    print(f"{'merge ratio':<28}{len(cn)/max(len(persons),1):>12.2f} : 1")
    print("-" * 62)
    tick = lambda ok: "PASS" if ok else "FAIL"          # noqa: E731
    print(f"{'1. same-chain merges':<28}{same_chain:>12,}   {tick(same_chain==0)}")
    print(f"{'2. span > radius':<28}{span_bad:>12,}   {tick(span_bad==0)}")
    print(f"{'3. back-edges (cycles)':<28}{len(be):>12,}   {tick(len(be)==0)}")
    print("-" * 62)
    print("4. largest nodes:")
    for p in top:
        print(f"     {p.occurrence_count:>5} occ / {len(p.names):>3} names  "
              f"{p.primary_name[:44]}")
    print("-" * 62)
    print(f"5. rank-0 nodes            {rank0:>12,}")
    for p in imam:
        print(f"   'ابي عبد الله' rank={p.rank}  occ={p.occurrence_count}"
              f"  children={len(p.children)}")
    print("=" * 62)


if __name__ == "__main__":
    main()
