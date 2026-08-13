# ANGE Port — Change Notes

Everything changed while porting **ANGE** (Zaraket & Makhlouta, *Arabic
Cross-Document NLP for the Hadith and Biography Literature*, FLAIRS-25 2012)
from the original C++/Qt system to this Python system.

Reference implementation: `../atmine-master/ATSarf/src/case/`
Specification: `paper (2).pdf`

Every claim below was checked against the C++ source or measured on real
data. Line references point at the original.

---

## Contents

1. [Findings about the original](#1-findings-about-the-original)
2. [Phase 2 — the narrator graph](#2-phase-2--the-narrator-graph)
3. [C3 — narrator hygiene](#3-c3--narrator-hygiene)
4. [Phase 3 — the biography stage](#4-phase-3--the-biography-stage)
5. [Phase 1 — measurement](#5-phase-1--measurement)
6. [Integration](#6-integration)
7. [Deliberate deviations](#7-deliberate-deviations)
8. [New / deleted files](#8-new--deleted-files)
9. [Test suite](#9-test-suite)
10. [What remains](#10-what-remains)

---

## 1. Findings about the original

These are contributions in their own right — each is verifiable from the
source and each is something this port implements, corrects, or documents.

### 1.1 Three claims the paper makes that the released system never implemented

| # | Claim | Reality |
|---|---|---|
| 1 | The hierarchical distance metric with **conflicting / reinforcing nisba** (§3.2 — الكوفي reinforces العراقي, المصري conflicts) | `getdistance()` (`narrator_abstraction.cpp:399`) contains it but is **never called**. `equal()` routes to `equalNew()`, a plain level-match ratio whose own comment says *"till now possessives not even compared"*. |
| 2 | **k-reachable** with a tunable k (§2, §4) | `bio_max_reachability` is declared, defaulted and wired to a GUI text box (`biographies.h:132`) but **never read**. `findChosenBiography` hardcodes `BFS_traverse(cont, 1, ±1)`, so k was always 1. |
| 3 | Sequential ambiguity resolution — *"ANGE only considers matches of nᵢ within k steps of the matches of nᵢ₋₁"* (§4) | **Absent entirely.** Only the first of the two stated measures (hash-score thresholding, via `LocateInGraphAction`) exists. |

### 1.2 Two implementation defects

**`updateGroups` discards the better candidate** (`narratordetector.cpp:158`).
Trace it on a descending-sorted list: a set scoring *higher* than an existing
one sets `add = true`; the `else if (add)` branch is then never reached;
nothing is inserted and the better set is **silently dropped**. Only sets
scoring worse than everything already present get appended.

**`getBoundingParagraph` / `isInsideBound` are dead code**
(`narratordetector.cpp:213-266`) — defined, never called. So the paragraph
delimiters that `additionalCheck` collects never bound anything.

### 1.3 The gold annotations were never published

`atmine-master/.gitignore` lines 1–5:

```gitignore
*.por
*.narr
*.names
*.tags
```

Deliberately excluded from version control. Confirmed absent from the working
tree and from `ATSarf.rar` (scanned its headers). **Nobody can reproduce
Tables 1–3 from this repository**, including the authors.

What the drafts do say about the method:

- `tex/hadith.ijcnlp.tex:1007` — *"manual verification of the correctness of
  10% of the narrations"*
- `tex/hadith.short.tex:606` — *"marking the beginning and end of a
  representative sample"*
- `tex/hadith.ijcnlp.tex:1010` — an unwritten TODO: *"(discuss annotation and
  verification process and mention smg about inter-annotator agreement)"*
- FLAIRS §6 admits the consequence: *"a conservative decision by the manual
  annotator"* skewed Table 1, and *"Further studies should accommodate several
  manual annotators and report on inter-annotation agreement."*

**Implication:** building your own gold is the first deliverable, not a gap.
Annotating an overlap slice twice and reporting agreement answers the paper's
own stated limitation.

---

## 2. Phase 2 — the narrator graph

### Measured effect

`kafi1.clean.hadiths.json`, threshold 0.5, radius 3
(harness: `tests/check_graph.py`)

| | before | after |
|---|---|---|
| persons | 1,634 | 1,714 |
| merge ratio | 5.70 : 1 | 5.24 : 1 |
| same-chain merges | 76 | **0** |
| groups exceeding radius | 99 | **0** |
| back-edges (cycles) | 771 | 843 † |

† Not a regression — see [§2.9](#29-why-cycles-remain).

### 2.1 `canonize` dropped kunya connectors — B8

`name_model.py` claimed the old code ignored ابو/ام for levelling. It does
not: `isFamilyConnector()` returns true for `IBN | AB | OM | FAMILY_OTHER`
(`narrator_abstraction.h:152`), and `preProcessForEquality` appends family
connectors into the level. `getKey` then writes them as the literals
`"AB"` / `"OM"` (`narratorHash.h:78-95`), which collapses the case variants
ابو/ابي/ابا onto one token.

Measured consequence before the fix:

```
ابو جعفر      vs جعفر      -> 1.000    (should be 0)
ابو عبد الله  vs عبد الله  -> 1.000    (should be 0)
ام سلمه       vs سلمه      -> 1.000    (should be 0)
```

**Fixed** by introducing `LevelItem(text, kind)` so a level slot knows whether
it holds a name or a connector — required because `equalNew`
(`narrator_abstraction.cpp:636`) skips a slot pair **only when both sides are
connectors** (letting ابو/ابي vary) and hard-rejects a connector facing a name.

### 2.2 `canonize` stripped all empty levels — B9

The code removed *every* empty level though the docstring said "trailing".
A leading empty level is meaningful: it means the first name is unknown
(`ابن ابي عمير`), which `getKey` encodes as a leading `-` and
`isFirstNameAmbiguous` (`narratorHash.h:127`) keys off.

Consequences: `ابن ابي عمير` scored **1.000** against `عمير`, and
`level_one_empty` in `name_hash.py` could never be true — killing the entire
skip-first key family. Now only trailing empties are dropped;
`ابن ابي عمير` keys as `-AB عمير`.

### 2.3 Same-chain merge veto — B1

`mustMerge` (`graph.h:900-912`) vetoes a merge when the candidate group
already holds an occurrence from the same chain — a person cannot appear
twice in one sanad. Absent here, so narrators merged with their own
neighbours, producing self-loops and spurious cycles. **76 violations
measured → 0.**

### 2.4 Generation radius used the wrong test — B2

Old (`graph.h:914-924`) tests the **span** of the whole group once the new
occurrence joins:

```cpp
least   = min(c_index, g->getLowestIndex());
highest = max(c_index, g->getHighestIndex());
return highest - least <= equality_radius;
```

The Python returned true if **any single** occurrence was within *r*, which
lets a group ratchet across generations one hop at a time (0 merges with 3,
3 with 6, 6 with 9…). One node had reached span 12. **99 violations → 0.**

Both guards are now O(1) via per-person caches rather than rescanning
400-occurrence nodes.

### 2.5 The three-level node hierarchy was collapsed to two — B7

Original: `ChainNarratorNode` → **`GroupNode`** (occurrences with an identical
key) → `GraphNarratorNode` (fuzzily merged set of groups). The Python had only
occurrence → person, and `GroupNode` was a vestigial stub.

This matters because Split is defined as `m(n1..nk) → m1(n1..ni), m2(ni+1..nk)`
— it needs **indivisible pieces** to redistribute. A `GroupNode` is that piece:
its members share one exact key, so no evidence can separate them.

Restored in `graph_nodes.py`. `GraphNarratorNode.occurrences` / `.names` /
`.is_rasoul` are now derived from its groups, so a split only moves groups
around. The build is two-stage like `buildGraph` (`graph.h:1093-1118`):
exact-key join (old `BuildAction`) then fuzzy fuse (old `MergeAction`).

### 2.6 Cycle breaking was inert, and destroyed data when it acted — B4, B5, B6

Two separate problems:

**Inert.** `narrator_graph.build()` called `break_cycles()` and then
`_rebuild_edges(chains)`, which regenerates every edge from the chains —
undoing every deletion the breaker made. Measured: **771 back-edges survived**
a run that reported breaking cycles.

**Destructive.** When a split failed, `_break_weakest_edge` deleted a real
narration edge. The old system **never** removes a عن relation
(`graph.cpp:115`); it only splits nodes. And `resolved += 1` counted forced
edge-deletions as "cycles broken", so the statistic did not mean what it said.

**Rewritten** as the real `reMergeNodes`: dissolve a person into its groups,
re-merge pairwise at the stricter threshold, one DFS pass per threshold (the
`LoopBreakingVisitor` shape). `_break_weakest_edge` deleted. Edges rebuild via
a callback after each pass, so splits take effect. Reports `dissolved`,
`splits`, `unresolvable`, `remaining_cycles` separately.

### 2.7 `primary_name` was the longest form — B11

`max(names, key=len)` labelled an over-merged node with its **rarest**
variant. The largest node in the graph read
`احمد بن محمد بن عبد الله بن مروان الانباري` — a one-off spelling — making the
output unreadable. Now the most frequent form: `احمد بن محمد`.

### 2.8 Rank was a different algorithm — B10

Old `RankCorrectorNodeVisitor` (`graph.h:498`) is a longest-path relaxation
(`if (rank1 >= rank2) rank2 = rank1 + 1`) over two BFS passes. The Python did
a **shortest-path** BFS from all parentless nodes and its docstring asserted
equivalence.

The property that matters is **monotonicity**: every edge must go from a lower
rank to a higher one, or "generation" is meaningless. Shortest-path breaks it
— a node reachable by both a 1-hop and a 4-hop route takes rank 1, so the
4-hop edge runs backwards. The `-1` fallback used `min(parent_ranks)`, making
it worse.

Replaced with a proper topological (Kahn) longest-path relaxation, plus
`rank_violations()` as an invariant check. Note the old system used rank only
for graphviz layout, so its under-convergence never mattered there; here it
feeds the biography stage.

### 2.9 Why cycles remain

**651 of 658 remaining cycles consist entirely of single-group persons.** A
`GroupNode` is atomic, so an over-fusion landing inside one group is
permanent — no split algorithm can undo it.

Persons on the most cycles: `ابي بصير` (1229), `الله` (786),
`ابن فضال` (503), `عبد الله` (491), `يونس` (463). Two causes, neither a
cycle-breaking bug:

1. **Upstream leakage** — `الله` is not a narrator (see [§3](#3-c3--narrator-hygiene)).
2. **Inherent name ambiguity** — `ابي بصير`, `يونس`, `ابن فضال` really are
   several people sharing one written name. The only defence is at *build*
   time.

This is why the paper reports **63% recall in Table 1**. It should be
reported as a limitation of the method, not patched silently.

### 2.10 A bug found while restoring `GroupNode`

`_find_or_create_person` returned the rasoul person's id **without attaching
the occurrence**. About **1,115 rasoul occurrences and every edge into them
were silently dropped** from the graph. Edge count went 3,651 → 3,962 after
the fix.

### 2.11 Smaller items

- **B12** — comparisons re-parsed name *strings* via `canonize_string` on every
  pair, discarding the FSM's real connector tags and costing an O(n) re-parse.
  Now compares cached `CanonicalName`s.
- **B14** — `_within_radius` ended with `return len(person.occurrences) == 0`,
  always false. Removed with the rewrite.
- **B13** — **verified NOT a port.** `equalHelper` (`graph.h:1329`) is used
  only to *compare two graphs* (`graph.h:1956-1975`), never in the merge path.
  The old merge had exactly two guards, both now implemented. Left **off**;
  if enabled it must be labelled an improvement, not fidelity.
- **F1** — `normalization.py` had a module-level `print` outside the
  `__main__` guard; every import printed `True` and corrupted piped output.
- **F2** — dead `differed` variable in `equality.py`.
- **F3** — the `GroupNode` stub, now real.

---

## 3. C3 — narrator hygiene

### The evidence

The old system kept **two separate lexicon files**:

| `src/case/stop_words` → sets `isRasoul`, ends the sanad | `src/case/phrases` → compound units, **not** narrators |
|---|---|
| رسول الله، الرسول، النبي، ص، صلى الله عليه وآله | **عليه السلام**، رضي الله عنه، عليهم السلام |

`lexicons.py` merged them, promoting `عليه السلام` from *modifier* to
*sanad-terminating narrator*.

### Measured on kafi1

```
total narrator slots        8,986
flagged is_rasoul           1,116   (all at the LAST position — structurally right)
STANDALONE honorific-only   1,031   = 11.5% of every narrator slot
```

The bare honorific became the **largest node in the graph** (1,116
occurrences, 143 children) and a major cycle generator. Alongside it:
`ابيه` ×236 (relative narrators), `عده من اصحابنا` ×182 (a group placeholder),
`الله` ×45 (pure matn leakage).

### The fix

Running the pipeline **without CAMeL** (lexicons + phrases only) produced the
*correct* single narrator — so the FSM was never the problem; a
CAMeL-dependent flag was. The invariant is therefore enforced rather than one
code path patched:

- `lexicons.py` — restored the distinction: `is_prophet_token()` (the person)
  vs `is_honorific_modifier()` (decoration).
- `fsm.py` — `_mark_rasoul` consolidated; the logic had been **duplicated in
  `_in_stop` and `_in_nrc`**, which is why a first attempt fixed only one of
  three paths. Three rules now: already building the Prophet's node → extend
  it; a Prophet token → **close** the open narrator and start his; a pure
  modifier → contribute **nothing**.
- `_absorb_honorific_narrators()` in `_end_chain` as a safety net.
- `quality_filter.py` — detects the honorific as a **phrase**, since
  `is_rasoul` no longer fires. Word-level matching would hit `الله` inside
  `عبد الله` and match everything.

### Two decisions, both corrected by testing

1. **The modifier is not appended to the name.** Appending changes the exact
   hash key, so `ابي عبد الله عليه السلام` and `ابي عبد الله` would land in
   different buckets and never be compared — trading one bug for a subtler one.
2. **The modifier does not set `is_rasoul`.** That flag collapses every
   flagged narrator into **one** graph node, which would fuse Imam al-Sadiq
   with Imam al-Baqir.

### Also fixed here

A relative narrator was swallowing the following name:
`روى عنه حرب بن الحسين` became one narrator called `عنه حرب بن الحسين`. This
affects the hadith path too.

### 3.5 C1 — gating the weak `noun_prop` category

**The root cause of matn leakage.** `pos == "noun_prop"` was accepted as a
name **anywhere**, so any proper-noun-ish word in the matn became a narrator,
three of them in a row passed `narr_min`, and a fake sanad was emitted. The
`MATN_CUES` and `NAME_BREAKING_POS` heuristics in `fsm.py` exist to clean up
after that — they treat the symptom.

The old system does not trust this category on its own. `bit_NOUN_PROP` is
deliberately kept **out** of `bits_NAME` (`hadithCommon.cpp:117-133`) and
appended only while `tryToLearnNames` is set (`hadithCommon.h:259`), which
happens in exactly four positions:

| # | context | source |
|---|---|---|
| 1 | the current word is a family connector | `cpp:1160` |
| 2 | just after a narration word, ≤1 NRC deep, not mere punctuation | `cpp:1172` |
| 3 | the **previous** word was a narration word | `cpp:1178` |
| 4 | the word looks like a nisba (ال…ي) in NRC/NAME context | `cpp:1227` |

plus two hard filters in `analyze()`: the word must carry **no suffix**
(`h:267`) and its stem must be **at least 3 characters** (`h:269`).

Implemented as `TokenInfo.is_name_candidate` (weak evidence) plus
`_promote_name_candidates()`, which promotes to `is_name` only in those
contexts. Promoted names are tagged `camel-learned` in `name_sources`, so the
provenance is visible in the output. `has_enclitic` comes from CAMeL's `enc0`
and stands in for the suffix test.

`ArabicEngine(strict_names=False)` restores the loose behaviour, so the two
can be compared directly rather than argued about.

---

## 4. Phase 3 — the biography stage

Not implemented at all before this work. It is where the paper's headline
claim lives.

### 4.1 `bio_fsm.py` — the biography automaton (A1)

`BiographyFSM(HadithFSM)` — a **subclass**, not an `if bio_mode` flag threaded
through every handler (which is what the original did, and why `getNextState`
is 500 lines). Five overrides, each tied to a `structures->hadith` branch:

| difference | old source |
|---|---|
| numbers are not boundaries | `hadithCommon.cpp:600` |
| `nmc_max` 3→1, `nrc_max` 5→100 | `hadithCommon.cpp:612` |
| NMC overflow → NRC instead of ending the run | `hadithCommon.cpp:907` |
| a rasoul word does **not** end the run | `hadithCommon.cpp:1085` |
| a name may open a run anywhere | `hadithCommon.cpp:639` |

Output is a flat, text-ordered `list[NarratorMention]` with char offsets —
the old `narratorList`. Also collects paragraph delimiters from number+dash
(`narratordetector.cpp:378`), which the original collects and never uses.

### 4.2 The feedback loop — the actual cross-document mechanism (A2)

`bio_nrcCount` is a **word** counter (`hadithCommon.cpp:264-268`) reset to
zero **only when a completed narrator is confirmed against the hadith graph**
(`hadithCommon.cpp:184-186`):

```cpp
if (biography->addNarrator(narrator))  currentData.bio_nrcCount = 0;
```

So biography mode tolerates up to 100 words **since the last graph-confirmed
narrator**. Every confirmation renews the budget, which is how one narrator
run spans an entire entry. That single line is the cross-document
reconciliation, and it explains the paper's otherwise odd numbers: with the
graph, detection recall **drops** 0.94 → 0.65 (only confirmable names survive)
while boundary recall **rises** 0.41 → 0.93 (the survivors have correct spans).

`narrator_matcher.py` supplies it. `GraphIndex` is the port of
`isRealNarrator` + `RealNarratorAction` (`biographyGraphUtilities.cpp:4`):
a **graded, de-duplicated** `[MatchingNode(person_id, similarity)]` sorted
best-first, where "is a real narrator" means that list is non-empty — the old
signature fills the list and returns the bool in one call.

Two fidelity fixes came out of it:

- **Symmetry.** Matching compares a biography `Narrator` (carrying the FSM's
  real connector tags) against graph persons. A graph loaded from JSON held
  only name *strings*, so it was re-canonized through the lossy regex path —
  and if the two sides canonize differently a true match is missed with no
  error. `GroupNode.to_dict()` now serializes the `CanonicalName` and
  `canonical_from_dict()` restores it exactly. `GraphIndex.stats()` reports
  `persons_exact_canonical` vs `persons_reparsed_from_string`.
- **D7.** The threshold now comes from the graph's own saved params.
  Biography names were matched at 0.1 against a graph built at 0.5.

### 4.3 `biography_detector.py` — graph-derived boundaries (A3, A6, A10)

Port of `findChosenBiography` + `checkBiography`
(`narratordetector.cpp:268` / `:665`). Per graph node:

- **centers** = every position in the mention list where its name occurs
- **neighbours** = k-reachable (**A6** — routed through `kreachable.py`, which
  existed but was never in the production path; `k=1` reproduces the old
  hardcoded `BFS(1,+1)` + `BFS(1,-1)`)
- **score** = distinct neighbours with an occurrence within `near_max_chars`
  **characters** (old `areNear`)
- **span = [min start, max end]** over the winning set — **this is the
  boundary**
- overlapping spans rejected best-first

**A10** came free: subjects whose key starts with `-` (`isFirstNameAmbiguous`)
are skipped, which only became possible because §2.2 restored the leading `-`.

### 4.4 Sequential disambiguation (A7) — an addition, not a port

Paper §4's second ambiguity measure, [verified absent](#11-three-claims-the-paper-makes-that-the-released-system-never-implemented)
from the original. Implemented as `sequential_disambiguation`, **off by
default**, so the faithful baseline stays clean and the gain is separately
measurable via `disambiguation_stats`.

At a genuine entry boundary nothing is reachable (the previous narrator
belongs to the previous entry), so the unfiltered set is kept rather than
deleting the narrator — the constraint disambiguates *within* a run.

---

## 5. Phase 1 — measurement

### 5.1 `spans.py` + `agreement.py` — the two-level scorer (A4)

Port of `AbstractTwoLevelAgreement::calculateStatisticsHelper`
(`AbstractTwoLevelAgreement.cpp:195`). Three levels:

- **segmentation** — the main spans
- **detection** — names, greedy **one-to-one**. `OneLevelAgreement::equalNames`
  is literally `overLaps` (`OneLevelAgreement.cpp:42`) — no string comparison
  at all.
- **boundary** — at **word** granularity, in `min` (per matched pair) and
  `max` (per merged overlap group) variants.

"Boundary recall 0.41" therefore means *41% of gold narrator **words** fell
inside a predicted span* — not a set-membership ratio.

The formulas are kept verbatim:

```
recall    = (common/correct) * correct        # == common
precision = (common/detected) * correct
denominator += correct
```

The `* correct` is a **weight**: it cancels in recall (true micro-average) and
does not in precision (recall-weighted macro). Simplifying either changes the
numbers.

What the metric is for, on the demo case:

```
detection       0.6667   ← two of three names "found"
boundary (min)  0.4444   ← but only 4 of 9 gold WORDS covered
```

Detection cannot see a bad span, because overlap is enough. The boundary rows
expose it.

### 5.2 `qdatastream.py` + `gold.py` — the gold layer (A5)

Since the gold arrives from Dr Fadi, the priority was the code that **reads**
it.

`qdatastream.py` decodes the original Qt binaries, with the layout taken off
the C++ rather than guessed:

| file | format |
|---|---|
| `.names` / `.narr` / biography `.tags` | flat `QList<QPair<int,int>>` |
| hadith `.tags` | two-level: main span + name spans + the annotator's `Chain` |
| `.equal` | `QMap<QPair<QString,QString>, bool>` |

One detail that would otherwise corrupt everything after the first record:
inside the chain, `NameConnectorPrim` stores its offsets **twice** — as
`qint64` in its own `serialize`, then again as `qint32` in the enclosing
`Narrator` loop (`narrator_abstraction.cpp:79`, `:146`). The parser is
**strict**: an unexpected tag raises with the byte position rather than
producing plausible garbage. `.tags` is overloaded between the two formats, so
`read_tags()` sniffs which it is.

`gold.py` is the canonical store everything scores against, whatever the
source. `GoldItem` carries optional **`persons` labels**, because Table 1
scores merge decisions and needs cluster labels rather than offsets.

**The safeguard is restored.** The old system, given no gold, wrote the
current system output as a seed, printed *"Correct it before use"*, and
**returned false so no statistics were produced**
(`AbstractTwoLevelAgreement.cpp:178`). A bootstrapped seed here is written
`reviewed: false` and `load_for_scoring()` refuses it. Plus a `text_sha256`
guard — gold made against a different text is an **error**, not a warning.

This is exactly the trap the deleted `boundary_gold.py` fell into: it built
"truth" by calling the same segmenter under test, so gold == prediction.

### 5.3 `partial_annotation.py` — paper §4.1 (A8)

The paper's **second headline result**: 80% recall / 89% precision (§6).

A scholar selects a few topically-related hadiths, builds a partial graph
**Gp** from just those, and asks — for each narrator in Gp — which biography
entries might describe them.

The key structural point is that the original has *both* modes in one class,
distinguished by two overrides:

| | `modifyNodes()` | `checkBiography()` |
|---|---|---|
| `BiographySegmenter` | every narrator in the book | keep top-1, **reject** overlapping spans |
| `NarratorDetector` (the base class) | the nodes the scholar picked (`:346`) | keep **all top-3**, no overlap rejection (`:375` returns true unconditionally) |

So partial annotation is not a different algorithm — it is the same scoring
answering a different question. Segmentation must partition the book;
annotation returns a **ranked shortlist** because the scholar decides. That is
why §6 says *"the accuracy of biography boundary detection is not well defined
in this task since the partial graph annotation method reports several
biographies ranked with a similarity metric."*

**Why Gp and not the full graph** — this is the point of the method. In the
full graph a prolific narrator has hundreds of neighbours, so "a neighbour
appears nearby" is weak evidence. In a graph built from ten related hadiths
a narrator has a handful, and their co-occurrence in one entry is strong
evidence. The restriction is what makes the ranking sharp. `tests/test_a8.py`
asserts this directly.

`score_annotations(..., at_k=)` evaluates a hit as *a candidate's span
overlaps the gold entry at rank ≤ k*. `at_k=1` is the strict reading; since
the paper reports a shortlist, `at_k=3` is the honest companion and both
should be given.

Runner: `run_partial.py --contains العقل` or `--numbers 1 2 3`.

### 5.4 `run_equality.py` — paper §5

Port of `narrator_equality_comparision`
(`narratorEqualityComparision.cpp`). The **only** evaluation that isolates the
distance metric — everywhere else it is entangled with segmentation and graph
building, so a metric change barely registers. Here it is the whole signal, and
each item is a yes/no on a pair of names rather than a span, so a useful gold
set costs an evening.

Three conditions:

| | |
|---|---|
| `eNarrator (edit)` | normalized edit distance > 0.75 — the paper's baseline (`:47`) |
| `structural` | our port of `equalNew`, qualifiers off — the faithful metric |
| `+ qualifiers` | with the nisba logic **on** — OUR implementation of §3.2, which the released system never shipped |

The third column is the point: §3.2 describes conflicting nisba
(العراقي vs المصري) and reinforcing nisba (العراقي + الكوفي), `getdistance()`
implements it, and nothing calls it. That column measures a claim the paper
makes and the original never tested.

`--bootstrap <graph.json>` emits the pairs a human should judge, drawn from
names the **hash already considers plausible** and sampled across score bands.
Random name pairs are trivially different and teach the evaluation nothing;
these are the decisions the metric actually has to make:

```
احمد                          || احمد بن علي بن محمد بن عبد الله
محمد بن السكين                || محمد بن علي بن معمر
موسا بن محمد العجلي           || موسا بن محمد بن اسماعيل بن عبيد
```

`--sweep` shows recall/precision across thresholds; `--disagreements` prints
the false positives and negatives, which is where the metric needs work.

### 5.5 `graph_agreement.py` — Table 1

Port of `compareGlobalGraphs` (`hadithInterAnnotatorAgreement.cpp:210`). This
scores **merge decisions**, not spans, which `agreement.py` cannot do. Per
occurrence:

```
common     = |gold_cluster ∩ pred_cluster| - 1
recall     = common / (|gold_cluster| - 1)     ... 1 if a singleton
precision  = common / (|pred_cluster| - 1)     ... 1 if a singleton
```

averaged over occurrences. **This is B-Cubed with the element itself
excluded** — worth naming in the report, since citing a recognised metric is
stronger than defending a bespoke formula.

Two rows, as in the paper: `detected` (occurrences present in both) and `all`
(every gold occurrence; anything unextracted becomes a singleton).
`labels_from_graph()` keys occurrences `"chain#position"` so gold and
prediction match up without sharing internal ids.

---

## 6. Integration

`run_biography.py` rewritten as the real experiment:

```
rijal book --normalize--> tokens --BiographyFSM--> mentions
           --BiographyDetector--> graph-derived spans --agreement.score--> Table 2
```

run **twice**, differing in exactly one argument:

```python
NO-GRAPH    BiographyFSM(confirm=None)
WITH-GRAPH  BiographyFSM(confirm=GraphIndex.confirm)
```

That single switch is the independent variable; the FSM, its parameters, the
detector and the scorer are identical, which is what makes the two columns
comparable.

With no `--gold` it writes a seed from its own output, says to correct it, and
**produces no statistics** — the old `readAnnotations` behaviour.

### Measured end-to-end

`khoei.txt` × the kafi1 graph, `--lexicon-only`:

| | NO-GRAPH | WITH-GRAPH |
|---|---|---|
| narrator mentions | 5,048 | 5,441 |
| confirmed by graph | 0 | 2,870 |
| **narrator runs** | **4,165** | **1,980** |
| runs killed by budget | 4,852 | 505 |
| entries detected | 0 | 33 |

The graph does not find *more* narrators — **it stops the runs from dying.**
That is the paper's mechanism, visible on real data.

(Entry count is low because `--lexicon-only` degrades name detection, the
graph covers kafi1 only, and most Khoei narrators are simply not in it. That
mode is loudly marked not-reportable.)

---

## 7. Deliberate deviations

Every one is documented in the code at the point it occurs.

| # | Deviation | Reason |
|---|---|---|
| 1 | Cycle breaking re-applies the `mustMerge` vetoes during re-merge | The original (`graph.cpp:155`) re-merges on score alone, so a split could quietly re-create the same-chain and over-span fusions the build had just refused. Checked against the **accumulated** person, since fusion is transitive. |
| 2 | Rank uses a proper topological longest-path relaxation | The original runs exactly two BFS passes, which under-converges past depth 2. It only drove graphviz layout there; here it feeds the biography stage. |
| 3 | `updateGroups` implements the evident **intent** (top-N by score) | The original discards the better set — see [§1.2](#12-two-implementation-defects). |
| 4 | Scorer computes each quantity **once** | The original runs the word sweep inside both passes, appending to the min *and* max lists each time. Its own comment (`cpp:30`) concedes the max path is unreliable. |
| 5 | Scorer counts unmatched **tails** | Both original sweeps are `while (k < gold && h < pred)`, so gold names past the last prediction never reach the denominator — words nobody predicted were not counted as missed, inflating recall. |
| 6 | Sequential disambiguation (A7) added, **off by default** | Not in the original at all. Kept opt-in so the faithful baseline is clean and the gain is measurable. |
| 7 | `context_check` (B13) left **off** | Not a port. Enabling it must be labelled an improvement. |

### The additions, and how to switch them off (C5)

Three heuristics in `fsm.py` have no counterpart in the original. Each is now
a parameter, and `tests/test_c5.py` verifies each one actually changes
behaviour rather than being a dead flag:

| `FsmParams` field | what it adds |
|---|---|
| `matn_cues` | `قال:` can close a sanad |
| `pos_hard_stop` | a verb/pronoun/particle breaks a name |
| `one_chain_per_hadith` | after the Imam, no new sanad until the next number |

`FsmParams.faithful()` turns all three off; `run_hadith.py --faithful` does
the same from the command line, and `--loose-names` reverts C1's gating.

**The distinction that matters:** `--faithful` disables the port's
*additions*. It does **not** disable its *fidelity fixes* — C1's name gating,
C3's honorific handling, the graph guards — because turning those off would
make the port **less** faithful, not more. `test_c5.py` asserts this directly:
`عليه السلام` is not a narrator in either mode.

**Left alone deliberately:** the scorer's precision is weighted by `correct`,
so a spurious prediction overlapping no gold name has weight zero and cannot
move precision. That asymmetry is inherent to the paper's formula. Precision
here means *of the gold words we claimed, how many were right*.

---

## 8. New / deleted files

### New

| file | purpose |
|---|---|
| `bio_fsm.py` | biography-mode automaton (A1) |
| `narrator_matcher.py` | `isRealNarrator` / graph lookup (A2) |
| `biography_detector.py` | `findChosenBiography` / boundaries (A3, A6, A7, A10) |
| `spans.py` | interval + word-counting primitives |
| `agreement.py` | two-level scorer (A4) — Tables 2 & 3 |
| `graph_agreement.py` | merge-decision scorer — Table 1 |
| `gold.py` | canonical gold store + safeguard (A5) |
| `partial_annotation.py` | paper §4.1 partial graph annotation (A8) |
| `run_partial.py` | runner for §4.1 |
| `run_equality.py` | paper §5 — metric vs Levenshtein |
| `qdatastream.py` | reader for the original Qt binaries (A5) |
| `tests/` | 165-test regression suite + graph harness |
| `PORT_NOTES.md` | this document |

### Deleted

| file | reason |
|---|---|
| `boundary_gold.py` | built "gold" by calling the same segmenter under test — gold == prediction, so every number derived from it was meaningless |
| `boundary_eval.py` | strawman no-graph baseline (`len(candidates) == 1`) plus the wrong metric shape; superseded by `run_biography.py` + `agreement.py` |
| `biography_segmenter.py` | a SECOND biography method — see §6.1 |
| `biography_linker.py` | ditto |

### 6.1 One biography method, not two

The web app used to run its own biography pipeline: `biography_segmenter.py`
cut the rijal book on an `N - name :` regex, and `biography_linker.py` scored
entries against the graph. Both are gone.

**Why.** They were not the paper's method, and said so
(`biography_segmenter.py:28-37`: *"WHY WE DON'T JUST RUN THE HADITH FSM
HERE… the segmenter stays deliberately simple"*). A regex over entry headers
works on al-Khoei because that book is typeset with numbered headers; it is
not §4, it does not produce `lb`, and its boundaries come from typography
rather than from the graph. A number obtainable only from the UI is not a
number that can go in the report.

Three defects that fell out with them, each a symptom of the split:

1. `_clean_name` stripped the `عن` prefix **before** the `و` prefix, so
   `وعن علي بن محمد القمي` kept its `عن` — the regexes ran in the wrong order
   and nothing re-checked.
2. `/link-biography` tested reachability from **every** subject candidate
   rather than the disambiguated one, so a single hub node among the
   candidates (one had 28 parents + 22 children) made almost any name look
   reachable. Ambiguity inflated the score instead of being resolved by it.
   `biography_linker.link_entry` did disambiguate correctly — the endpoint
   simply never called it.
3. Prose leaked in as narrator names (`من اهل بلخ`,
   `قيل انه كان يقول بالتفويض`), because a header regex has no morphology
   behind it.

**Now.** `app.py` drives the same modules `run_biography.py` drives:

| step | module | paper |
|---|---|---|
| `/segment-biography` | `bio_fsm.BiographyFSM(confirm=None)` | §4 — builds `lb`, the no-graph baseline |
| `/link-biography` | `narrator_matcher.GraphIndex` + `biography_detector.BiographyDetector` | §4 — isRealNarrator, k-reachable boundaries |

The endpoint runs **both conditions**, differing in one argument
(`confirm=None` vs `confirm=GraphIndex.confirm`), as the runner does.

Two fixes came with it. `SESSION["graphs"]` now stores `params`, so
`GraphIndex` matches at the threshold the graph was **built** at rather than a
hardcoded `0.1` (D7). And because the session graph carries serialized
`groups`, matching runs on exact canonical forms —
`persons_reparsed_from_string: 0`, measured.

Measured on a 60k-char kafi slice (288 persons) × a 40k-char khoei slice:

| | no-graph | with-graph |
|---|---|---|
| mentions | 88 | 90 |
| confirmed by graph | 0 | 20 |
| **runs killed by budget** | **302** | **213** |
| entries detected | 0 | 1 |

The graph does not find more narrators — it stops the runs from dying. Same
mechanism as §6, visible on a slice.

**Still not Table 2.** The endpoint reports pipeline behaviour, not
recall/precision, and returns `"reportable": false`. Table 2 needs a reviewed
gold file via `run_biography.py --gold`.

### Substantially rewritten

`cycle_breaking.py` · `run_biography.py` · `graph_nodes.py` ·
`graph_build.py` (merge logic) · `narrator_graph.py` (ranks, edge ordering) ·
`name_model.py` (canonize) · `fsm.py` (rasoul handling, `_on_token` hook)

---

## 9. Test suite

```
py -3.11 tests/run_all.py
```

**165 tests, all green.** None require CAMeL — token flags come from
`lexicons.py` plus the phrase matcher, or are synthesized by hand. That is
deliberate: the FSM and graph logic must stay verifiable without the
morphology layer, so a regression is caught immediately rather than at the
next full book run.

| file | covers |
|---|---|
| `test_c1.py` | gating the weak noun_prop category |
| `test_c5.py` | the ablation switches |
| `test_c3.py` | honorific / junk narrators |
| `test_a1.py` | biography FSM — each documented difference + budget mechanics |
| `test_a2.py` | graph lookup end-to-end, including a JSON round-trip |
| `test_a3.py` | graph-derived boundaries + sequential disambiguation |
| `test_a4.py` | the scorer — primitives hand-checked, then aggregation |
| `test_a5.py` | Qt reader, the safeguard, validation |
| `test_a8.py` | partial graph annotation (§4.1) |
| `test_equality_eval.py` | metric vs edit distance (§5) |
| `test_table1.py` | merge-decision scoring |
| `check_graph.py` | **harness**: rebuilds a graph and reports the pass conditions |

The harness reports five conditions needing no gold, because each has an
intrinsic correctness criterion. Conditions 1 and 2 pass; condition 3
(back-edges) does not, for the reason in [§2.9](#29-why-cycles-remain).

---

## 10. What remains

### For full parity with the old system

- **Annotation UI** — plus C6 (`segmentNarrators`), which is precisely the
  "parse a hand-selected span into narrators" utility such a UI needs
  (`hadithChainGraph.cpp:81`)

### Quality

- **C2** — nisba detection is `startswith("ال") and endswith("ي")`.
- **C4** — `_trim_trailing_connectors` does not decrement the narrator count
  as `removeLastSpuriousNarrators` did.
- **A9** iterative threshold lowering · **A11** narrator gazetteer + nisba
  lexicon (which would subsume C2, and would let C1's gating rely on a real
  gazetteer rather than CAMeL's `noun_prop` alone)
- **E1–E4** — web app: wrong connector class, hardcoded params, a web/CLI
  filter mismatch, in-memory session state.
  **E5 is closed** — `/link-biography` no longer reports strawman numbers; it
  drives the paper's modules and declares `reportable: false`. See §6.1.

### Data

Gold annotations, from Dr Fadi or annotated locally. When requesting his
files, also ask for **the exact `.txt` files he annotated against** — the
`text_sha256` guard will reject a mismatch — and any `.por` graph files,
which would give Table 1 directly.

### Housekeeping

`kafi1.clean.graph.json` is **stale**: the `groups` field changed shape when
canonical forms were added. Regenerate it; `GraphIndex.stats()` reports
`persons_reparsed_from_string > 0` if an old file slips through.
