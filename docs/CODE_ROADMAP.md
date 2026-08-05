# Code Roadmap — how the system actually runs

Which file does what, in what order, and what each one hands to the next.
Split into the **hadith side** and the **biography side**, with the narrator
graph as the bridge between them.

Dependency arrows below were extracted from the imports, not from memory.

---

## 0. The shape of it

```
   HADITH BOOKS                                      RIJAL BOOK
   kafi1..8.txt                                      khoei.txt
        |                                                 |
        |  run_hadith.py                                  |  run_biography.py
        v                                                 v
   <book>.hadiths.json                              tokens (same engine)
        |                                                 |
        |  quality_filter.py                              |
        v                                                 |
   <book>.clean.hadiths.json                              |
        |                                                 |
        |  run_graph.py  /  merge_graphs.py               |
        v                                                 |
   <book>.graph.json  ------------- THE BRIDGE ---------->|
                                                          v
                                                  detected entries
                                                          |
                                                          v
                                                  agreement.py -> tables
```

The graph is the only thing crossing from one side to the other. Everything
the cross-document method claims comes down to that one arrow.

---

## 1. Shared foundation

Used by both sides. These are the leaves — nothing in the project depends on
anything outside them.

| file | in | out | notes |
|---|---|---|---|
| `normalization.py` | raw text | `(clean_text, index_map)` | strips diacritics, unifies alef/hamza/ة, collapses spaces. `index_map[i]` = position of clean char *i* in the original — this is how any span can be shown back to a human |
| `lexicons.py` | — | word sets + predicates | narration connectors, name connectors, honorifics, relative narrators, compound narrators. Consulted on **surface form and lemma** |
| `models.py` | — | dataclasses | `NamePrim`, `NameConnector`, `Narrator`, `Chain`, `Hadith` |
| `wojood.py` | words | person flags | standalone NER runner |
| `engine.py` | clean text | `list[TokenInfo]` | **the whole feature layer.** CAMeL (lemma/POS) + Wojood (PERS) + lexicons, merged into one flag set per token |

**`engine.py` is the seam.** Nothing below it imports CAMeL or Wojood. That's
why the tests run without the models installed.

---

## 2. HADITH SIDE

### 2.1 Flow

```
run_hadith.py
    |
    +-- normalization.normalize()          raw -> clean + index_map
    +-- engine.ArabicEngine.analyze_cached()   clean -> tokens  [CACHED]
    |
    +-- segmenter.HadithSegmenter.segment()
    |       |
    |       +-- fsm.HadithFSM.run(tokens)   -> [ChainCandidate]
    |       +-- filters by narr_min
    |       +-- links matn spans, attaches kitab/bab
    |       v
    |   [Hadith]
    |
    +-- writes <book>.hadiths.json + <book>.chains.txt
```

### 2.2 File by file

| file | role |
|---|---|
| **`run_hadith.py`** | entry point. Flags: `--narr-min`, `--nmc-max`, `--nrc-max`, `--faithful`, `--loose-names`, `--no-wojood` |
| **`fsm.py`** | **the automaton.** 5 states (TEXT / NAME / NMC / NRC / STOP). Decides where a sanad starts and ends. Port of `getNextState` |
| **`segmenter.py`** | book-level driver. Runs the FSM over the whole text, applies `narr_min`, links each matn to its sanad, finds hadith numbers and kitab/bab headings |
| **`quality_filter.py`** | separate pass. Drops narrations with no sanad structure at all (no بن, no honorific ending, no narration connector) |

### 2.3 What to know about `fsm.py`

Three things live here that aren't obvious from the state table:

- **Two tolerance counters** — `nmc_count` (junk words inside a name) and
  `nrc_count` (words between narrators). Overflow ends the chain attempt.
- **Honorific handling** (`_mark_rasoul`) — an honorific is **never** a
  narrator. A Prophet reference closes the open name and opens his node; a
  bare `عليه السلام` contributes nothing. This logic was duplicated in three
  places and is now consolidated in one method.
- **Three switchable additions** — `matn_cues`, `pos_hard_stop`,
  `one_chain_per_hadith`. Not in the original. `FsmParams.faithful()` turns
  all three off.

### 2.4 Commands

```bash
python run_hadith.py kafi1.txt
python quality_filter.py kafi1.hadiths.json
```

---

## 3. THE BRIDGE — the narrator graph

### 3.1 Flow

```
run_graph.py
    |
    +-- rebuilds Chain objects from the JSON
    |
    +-- narrator_graph.NarratorGraph.build(chains)
            |
            +-- graph_build.GraphBuilder.build()
            |       for every narrator occurrence:
            |         name_model.canonize()      -> levels + qualifiers
            |         name_hash.primary_key()    -> exact key
            |         STAGE A: identical key?    -> join that GroupNode
            |         STAGE B: name_hash.find_candidates()
            |                  equality.narrator_distance()
            |                  + 2 vetoes (same-chain, span radius)
            |
            +-- _rebuild_edges()
            +-- cycle_breaking.CycleBreaker.break_cycles()
            |       dissolve person -> groups, re-merge stricter, 3 passes
            +-- _compute_ranks()   topological longest path
            v
        <book>.graph.json
```

### 3.2 File by file

| file | role |
|---|---|
| **`run_graph.py`** | entry point |
| **`name_model.py`** | `canonize()` — splits a name into **levels** at each بن, qualifiers separately. `LevelItem(text, kind)` so a slot knows if it's a name or a connector |
| **`equality.py`** | `narrator_distance()` — compares level by level. Both connectors → skip; mismatch → **hard reject (0)**; score = agreeing levels / max levels |
| **`name_hash.py`** | multi-key hash. Indexes each name under every sub-name that could refer to the same person. Turns O(n²) into near-O(n) |
| **`graph_nodes.py`** | the **three** node types — `ChainNarratorNode` (occurrence) → `GroupNode` (identical key, **atomic**) → `GraphNarratorNode` (a person) |
| **`graph_build.py`** | the merge engine. Two-stage, plus both `mustMerge` vetoes |
| **`cycle_breaking.py`** | Split. Dissolve → re-merge at stricter threshold. **Never deletes an edge** |
| **`narrator_graph.py`** | orchestration, ranks, save/load, DOT export |
| **`merge_graphs.py`** | combine graphs across volumes |
| **`kreachable.py`** | bounded reachability. Used by the biography side |

### 3.3 The two vetoes (easy to lose, hard to notice)

```python
# 1. a person cannot appear twice in ONE sanad
if chain_node.chain_id in person.chain_ids: refuse

# 2. the whole group must fit inside equality_radius generations
low  = min(new_position, group.lowest)
high = max(new_position, group.highest)
if high - low > equality_radius: refuse
```

Veto 2 tests the **span of the group**, not the distance to the nearest
member. Testing the nearest lets a group ratchet across generations one hop
at a time.

### 3.4 Commands

```bash
python run_graph.py kafi1.clean.hadiths.json
python tests/check_graph.py kafi1.clean.hadiths.json 0.5   # invariants
```

---

## 4. BIOGRAPHY SIDE

### 4.1 Flow

```
run_biography.py
    |
    +-- normalization.normalize()
    +-- engine.analyze()                    same feature layer as hadith
    +-- narrator_matcher.GraphIndex.load()  <-- THE BRIDGE ARRIVES HERE
    |
    +-- run TWICE, differing in ONE argument:
    |
    |     bio_fsm.BiographyFSM(confirm=None)              NO-GRAPH
    |     bio_fsm.BiographyFSM(confirm=index.confirm)     WITH-GRAPH
    |             |
    |             +-- per word: words_since_confirm += 1
    |             +-- on a completed narrator:
    |                    matches = confirm(narrator)
    |                    if matches: words_since_confirm = 0   <<< THE LOOP
    |             v
    |         [NarratorMention]   flat, text-ordered, with offsets
    |
    +-- biography_detector.BiographyDetector.detect()
    |       per graph person:
    |         centers    = its positions in the mention list
    |         neighbours = kreachable.reachable_within(pid, k)
    |         score      = distinct neighbours within near_max_chars
    |         span       = [min start, max end]      <<< THE BOUNDARY
    |         reject spans overlapping an accepted one
    |       v
    |   [DetectedBiography]
    |
    +-- agreement.score()  (needs gold — see §6)
```

### 4.2 File by file

| file | role |
|---|---|
| **`run_biography.py`** | entry point. Runs both conditions and scores them |
| **`bio_fsm.py`** | `BiographyFSM(HadithFSM)` — **subclass**, 5 overrides. Holds the tolerance budget |
| **`narrator_matcher.py`** | `GraphIndex` — port of `isRealNarrator`. Returns a **graded** `[MatchingNode(person_id, similarity)]`, best first |
| **`biography_detector.py`** | port of `findChosenBiography` + `checkBiography`. Derives boundaries from graph structure |
| **`partial_annotation.py`** | paper §4.1 — a *different question*: given a small topical graph, which entries describe each narrator? Returns a ranked shortlist |
| **`run_partial.py`** | entry point for §4.1 |

### 4.3 The five bio-mode overrides

| what changes | old source |
|---|---|
| numbers are not boundaries | `cpp:600` |
| `nmc_max` 3→1, `nrc_max` 5→100 | `cpp:612` |
| NMC overflow → NRC, doesn't end the run | `cpp:907` |
| a rasoul word does **not** end the run | `cpp:1085` |
| a name may open a run anywhere | `cpp:639` |

### 4.4 The feedback loop — read this one twice

`bio_nrc_max = 100` looks absurd until you see that the counter it consumes is
**reset on every graph confirmation**:

```
word ... word ... word              budget draining
narrator completed
  -> narrator_matcher.confirm()
  -> matched a graph person?
       YES -> budget = 0, the run continues
       NO  -> budget keeps draining; at 100 the run dies
```

So it's not "100 words allowed" — it's **"100 words since the last narrator
the graph could confirm"**. That is the entire cross-document mechanism.
Remove the graph and the automaton still runs; the budget is simply never
renewed, and entries fragment.

Measured on khoei × kafi1: narrator **runs** 4,165 → 1,980, runs killed by
budget 4,852 → 505, while the mention count barely moves. The graph doesn't
find more narrators — it stops the runs dying.

### 4.5 Commands

```bash
python run_biography.py khoei.txt kafi1.clean.graph.json --lexicon-only
python run_biography.py khoei.txt kafi1.clean.graph.json --gold khoei.gold.json
python run_partial.py kafi1.clean.hadiths.json khoei.txt --contains العقل
```

---

## 5. Evaluation (details deferred until gold exists)

| file | scores |
|---|---|
| `spans.py` | interval + word-counting primitives |
| `agreement.py` | segmentation / detection / boundary — **Tables 2 & 3** |
| `graph_agreement.py` | merge decisions (B-Cubed) — **Table 1** |
| `gold.py` | the annotation store + the refuse-unreviewed safeguard |
| `qdatastream.py` | reads Dr Fadi's original Qt binaries |
| `run_equality.py` | metric vs edit distance — **§5** |

All of it works; it just has nothing to score yet.

---

## 6. Legacy — still present, superseded

| file | status |
|---|---|
| `biography_segmenter.py` | the **regex** entry-header segmenter. Superseded by `bio_fsm.py`. Keep only as a gold-bootstrapping helper |
| `biography_linker.py` | superseded by `narrator_matcher.py` + `biography_detector.py` |
| `app.py` | **still imports both of the above** — the web layer was never migrated. Its `/link-biography` endpoint reports numbers from the deleted strawman experiment |

`app.py` is the one place the old path can still be reached. Worth fixing
before the web demo is shown to anyone.

---

## 7. Artifacts on disk

| file | produced by | consumed by |
|---|---|---|
| `<book>.hadiths.json` | `run_hadith` | `quality_filter` |
| `<book>.clean.hadiths.json` | `quality_filter` | `run_graph`, `run_partial` |
| `<book>.chains.txt` | `run_hadith` | humans |
| `<book>.graph.json` | `run_graph`, `merge_graphs` | `run_biography`, `run_equality` |
| `.engine_cache/*.json` | `engine.analyze_cached` | itself — delete to force re-analysis |
| `<book>.bio.gold.seed.json` | `run_biography` with no `--gold` | you, to correct |

**Stale right now:** `kafi1.clean.graph.json` predates the `groups` schema
change, and `kafi1.clean.hadiths.json` predates the honorific fix. Both need
regenerating with CAMeL. `GraphIndex.stats()` reports
`persons_reparsed_from_string > 0` when an old graph file is loaded.

---

## 8. Reading order, if you're coming back cold

1. `models.py` — the vocabulary
2. `fsm.py` — the automaton, hadith side
3. `name_model.py` + `equality.py` — how two names are compared
4. `graph_build.py` — how people are formed
5. `bio_fsm.py` §"THE TOLERANCE BUDGET" — the cross-document mechanism
6. `biography_detector.py` — how a boundary is derived

Every module docstring cites the C++ file and line it came from. When
something looks arbitrary, that citation is usually the answer.
