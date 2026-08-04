# tests — regression suite for the ANGE port

Run everything:

```
py -3.11 tests/run_all.py
```

**None of these require CAMeL.** Token flags come from `lexicons.py` plus the
phrase matcher, or are synthesized by hand. That is deliberate: the FSM and
graph logic must stay verifiable without the morphology layer installed, so a
regression is caught immediately rather than at the next full book run.

| file | covers |
|---|---|
| `test_c3.py` | honorific / junk narrators — `عليه السلام` never becomes a narrator; the Prophet himself still does; `عبد الله` untouched |
| `test_a1.py` | biography-mode FSM — each of the five documented differences from hadith mode, plus the tolerance-budget mechanics |
| `test_a2.py` | graph lookup (`isRealNarrator`) end-to-end: build a graph, round-trip it through JSON, confirm narrators against it, verify the feedback loop |
| `check_graph.py` | **harness, not a test.** Rebuilds the narrator graph from a book and reports the pass conditions |

## The graph harness

```
py -3.11 tests/check_graph.py kafi1.clean.hadiths.json 0.5
```

Reports five conditions that need **no gold annotation**, because each has an
intrinsic correctness criterion:

1. **same-chain merges → 0** — a person cannot appear twice in one sanad
   (old `mustMerge`, `graph.h:900`)
2. **span > radius → 0** — a merged group must fit inside `equality_radius`
   generations (`graph.h:914`). Rasoul is exempt: the old system indexes every
   rasoul occurrence under one key regardless of depth.
3. **back-edges → 0** — the isnad graph must be a DAG (paper §2)
4. **largest nodes** — over-merge indicator
5. **rank-0 count / Imam rank** — generation sanity

Conditions 1 and 2 pass. Condition 3 does not, and that is **expected**: the
surviving cycles run almost entirely through single-group persons — narrators
whose occurrences share one exact written name. A `GroupNode` is atomic, so an
over-fusion landing inside one group is permanent. The causes are upstream
(matn leakage) and inherent name ambiguity (`ابي بصير`, `يونس`), not the cycle
breaker. See the roadmap notes.

## Writing a new test

Build tokens without CAMeL — copy the `analyze()` helper from `test_c3.py`
(lexicons + `_apply_phrase_flags`) for real Arabic sentences, or the `stream()`
helper from `test_a1.py` to force an exact flag combination by hand. The second
is better for state-machine edge cases, because it lets you construct inputs the
morphology layer produces only rarely.
