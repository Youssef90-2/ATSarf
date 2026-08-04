"""
bio_fsm.py
==========
IDEA (paper §4, old NarratorDetector::lookup, narratordetector.cpp:424):
Run the SAME automaton over a rijal (biography) book instead of a hadith
book, and produce a flat, text-ordered list of every narrator mention with
its character offsets — `narratorList` in the old code.

That list is the input to the cross-document step (find_chosen_biography):
no entry headers, no "روى عن" markers, no book-specific regex. The biography
boundaries are supposed to come from the GRAPH, not from typography.

--------------------------------------------------------------------------
WHAT BIOGRAPHY MODE CHANGES (all verified against getNextState, the
`structures->hadith` branches in hadithCommon.cpp):

  1. numbers are NOT boundaries              (hadithCommon.cpp:600)
     A hadith book numbers its narrations; a rijal book numbers its entries
     and also cites volume/page numbers mid-sentence, so a number cannot end
     a narrator run.
  2. nmc_max 3 -> 1, nrc_max 5 -> 100        (hadithCommon.cpp:612-613)
  3. NMC overflow does not end the run       (hadithCommon.cpp:907-914)
     hadith: -> TEXT + return false.  biography: -> NRC, keep going.
  4. a rasoul word does not end the run      (hadithCommon.cpp:1085-1092)
     hadith: -> TEXT + return false.  biography: -> NRC, keep going.
     (A biography mentions the Prophet in passing; the entry continues.)
  5. a name may start a run anywhere         (hadithCommon.cpp:639)
     hadith requires a preceding full stop; biography prose does not.

--------------------------------------------------------------------------
THE TOLERANCE BUDGET — the heart of the cross-document method.

In hadith mode the tolerance counter counts NRC words. In biography mode the
old code uses a DIFFERENT counter, `bio_nrcCount`, which:

    * increments on EVERY processed word          (hadithCommon.cpp:264-268)
    * resets to 0 whenever a completed narrator is CONFIRMED against the
      hadith narrator graph                       (hadithCommon.cpp:184-186)
          if (biography->addNarrator(narrator))  currentData.bio_nrcCount = 0;
      where addNarrator -> Biography::isRealNarrator -> a hash lookup in the
      graph above equality_threshold.
    * is what the NRC overflow test reads         (hadithCommon.cpp:1031)

So the automaton tolerates up to `nrc_max` (100) words SINCE THE LAST
GRAPH-CONFIRMED NARRATOR. Every confirmation renews the budget, which is how
one narrator run can span an entire biography entry.

That feedback is the cross-document reconciliation. It is also why the
paper's numbers move the way they do (Table 2): WITH the graph, narrator
detection recall drops 0.94 -> 0.65 because only confirmable names survive,
while boundary recall rises 0.41 -> 0.93 because the ones that survive have
correct spans.

`confirm` is the hook. Leave it None and you get the paper's NO-GRAPH
baseline (nothing ever confirms, the budget only ever drains). Pass the graph
lookup and you get the cross-document condition. Same automaton, one switch —
which is what makes the two columns comparable, unlike a spatial-vs-relational
comparison of two different methods.
"""

from dataclasses import dataclass, field

import lexicons
from fsm import HadithFSM, FsmParams, State
from models import Narrator


# ===========================================================================
# 1. Parameters — old hadithParameters bio_* fields (hadithCommon.h:34-37)
# ===========================================================================

@dataclass
class BioParams(FsmParams):
    nmc_max: int = 1        # bio_nmc_max — much stricter than hadith's 3
    nrc_max: int = 100      # bio_nrc_max — words tolerated since last confirm
    narr_min: int = 4       # bio_narr_min (see note in run(): not applied)


# ===========================================================================
# 2. A detected narrator mention, in text order
# ===========================================================================

@dataclass
class NarratorMention:
    """One narrator occurrence in the biography book (old narratorList entry)."""
    narrator: Narrator
    start: int
    end: int
    confirmed: bool = False      # matched a node in the hadith graph
    # graded (person_id, similarity) matches — old Biography::nodeGroups.
    # A3 disambiguates between these; the FSM only needs "is it non-empty".
    matches: list = field(default_factory=list)

    @property
    def name(self):
        return self.narrator.full_name

    def to_dict(self):
        return {"name": self.name, "start": self.start, "end": self.end,
                "confirmed": self.confirmed,
                "matches": [m.to_dict() if hasattr(m, "to_dict") else m
                            for m in self.matches]}


# ===========================================================================
# 3. The biography automaton
# ===========================================================================

class BiographyFSM(HadithFSM):
    """
    Subclass rather than an `if bio_mode` flag threaded through every state
    handler (which is what the old C++ did, and why getNextState is 500 lines).
    The hadith path is left byte-for-byte as validated; biography mode
    overrides only the five behaviours that genuinely differ, plus the
    confirmation hook.
    """

    def __init__(self, params: BioParams = None, confirm=None):
        """
        confirm: callable(Narrator) -> bool
            True if this narrator matches a person in the hadith narrator
            graph (old Biography::isRealNarrator). None -> never confirms,
            which reproduces the paper's no-graph baseline exactly.
        """
        super().__init__(params or BioParams())
        self._confirm = confirm
        self.mentions = []                 # flat, text-ordered (narratorList)
        self.paragraph_delimiters = []     # old additionalCheck positions
        self.stats = {"words": 0, "narrators": 0, "confirmed": 0,
                      "budget_resets": 0, "runs_ended_by_budget": 0}

    # ------------------------------------------------------------ lifecycle
    def _reset_all(self):
        super()._reset_all()
        # words seen since the last graph-confirmed narrator (old bio_nrcCount)
        self.words_since_confirm = 0
        self.mentions = []
        self.paragraph_delimiters = []
        # In biography prose a run may start anywhere, so the hadith rule
        # "only right after a full stop" is permanently satisfied, and the
        # "one sanad per hadith" lock never applies.
        self.at_sentence_start = True
        self.chain_done_this_hadith = False

    # --------------------------------------------------- narrator completion
    def _close_narrator(self):
        """
        Old addNarrator (hadithCommon.cpp:176-193): a completed narrator is
        offered to the graph; a hit RENEWS the tolerance budget.
        """
        narrator = self.current_narrator
        if narrator is not None and not narrator.is_empty():
            self.chain.add(narrator)
            self.stats["narrators"] += 1
            # `confirm` may return the graded match list (GraphIndex.confirm)
            # or a plain bool (a stub). Old isRealNarrator does both at once:
            # it fills the list AND returns whether it is non-empty.
            result = self._confirm(narrator) if self._confirm else None
            matches = list(result) if isinstance(result, (list, tuple)) else []
            confirmed = bool(matches) if matches else bool(result)
            if confirmed:
                self.words_since_confirm = 0          # <-- the feedback loop
                self.stats["confirmed"] += 1
                self.stats["budget_resets"] += 1
            self.mentions.append(NarratorMention(
                narrator=narrator, start=narrator.start, end=narrator.end,
                confirmed=confirmed, matches=matches))
        self.current_narrator = None

    # ---------------------------------------------------------------- run
    def run(self, tokens, hadith_mode: bool = False):
        """
        Process the whole book. Returns the flat, text-ordered list of
        NarratorMention — the old `narratorList`, which is what
        find_chosen_biography consumes.

        `hadith_mode` is forced False: in a rijal book numbers are entry
        headers and citations, never narrator-run boundaries
        (hadithCommon.cpp:600).

        NOTE on bio_narr_min: the old lookup() harvests narrators on every
        chain end with no minimum — the `bio_narr_min` branch in result()
        (hadithCommon.cpp:1120-1122) is unreachable, guarded by
        `currentChain->hadith &&`. We keep the parameter for the record and
        likewise do not filter.
        """
        super().run(tokens, hadith_mode=False)
        # collected AFTER the run: super().run() calls _reset_all(), which
        # would clear the list if we filled it first.
        self._collect_paragraph_delimiters(tokens)
        return self.mentions

    def _collect_paragraph_delimiters(self, tokens):
        """
        Old additionalCheck (narratordetector.cpp:378): a number immediately
        followed by a dash marks an entry boundary. Recorded, not acted on —
        find_chosen_biography uses it to bound a candidate span
        (getBoundingParagraph, narratordetector.cpp:213).
        """
        self.paragraph_delimiters = []
        for i, tok in enumerate(tokens[:-1]):
            if tok.is_number and tokens[i + 1].word == "-":
                nxt = tokens[i + 2] if i + 2 < len(tokens) else tokens[i + 1]
                self.paragraph_delimiters.append(nxt.start)

    # ------------------------------------------------- the tolerance budget
    def _on_token(self, token):
        """Every processed word drains the budget (hadithCommon.cpp:264-268)."""
        if not (token.is_punct or token.is_number):
            self.words_since_confirm += 1
            self.stats["words"] += 1

    def _budget_exhausted(self) -> bool:
        return self.words_since_confirm >= self.params.nrc_max

    # ======================================================== state handlers
    # Only the five behaviours that differ from hadith mode.

    def _in_text(self, token, is_rasoul):
        """
        DIFFERENCE 5 (hadithCommon.cpp:639): a name may open a run anywhere in
        biography prose; the hadith rule requires a preceding full stop.
        `at_sentence_start` is pinned True in _reset_all, so the inherited
        handler already behaves this way — but pin it again here because
        run() clears it after each token.
        """
        self.at_sentence_start = True
        super()._in_text(token, is_rasoul)

    def _in_nmc(self, token, is_rasoul):
        """
        DIFFERENCE 3 (hadithCommon.cpp:907-914): on NMC overflow the hadith
        path ends the run; biography falls through to NRC and keeps going.
        """
        p = self.params
        if (not is_rasoul and not token.is_nrc and not token.is_name
                and self.nmc_count + 1 > p.nmc_max
                and not (self.nmc_valid and not self.nmc_second_chance_used)):
            self._close_narrator()
            self.state = State.NRC
            self.nmc_count = 0
            return
        super()._in_nmc(token, is_rasoul)

    def _in_nrc(self, token, is_rasoul):
        """
        DIFFERENCE (hadithCommon.cpp:1031): the overflow test reads the
        word budget since the last confirmation, not the NRC-word count.
        """
        if (not token.is_name and not token.is_nmc and not token.is_relative
                and not is_rasoul and self._budget_exhausted()):
            self.stats["runs_ended_by_budget"] += 1
            self._end_chain(self.last_word_end, "budget")
            return
        super()._in_nrc(token, is_rasoul)

    def _in_stop(self, token, is_rasoul):
        """
        DIFFERENCE 4 (hadithCommon.cpp:1085-1092): a rasoul word ends a
        hadith sanad but NOT a biography run — a rijal entry mentions the
        Prophet in passing and then carries on listing narrators.
        """
        if token.is_rasoul:
            self._mark_rasoul(token)
            return
        self._close_narrator()
        self.state = State.NRC
        self.nrc_count = 0


# ---------------------------------------------------------------------------
# Quick demo:  python bio_fsm.py
# Uses lexicon-only token flags so it runs without CAMeL installed.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from normalization import normalize
    from engine import (TokenInfo, tokenize_with_positions, ArabicEngine,
                        PUNCTUATION_CHARS)

    _eng = ArabicEngine.__new__(ArabicEngine)

    def analyze(clean):
        toks = [TokenInfo(word=w, start=s, end=e)
                for w, s, e in tokenize_with_positions(clean)]
        idx = []
        for i, t in enumerate(toks):
            if t.word in PUNCTUATION_CHARS:
                t.is_punct = True
            elif t.word.isdigit():
                t.is_number = True
            else:
                idx.append(i)
        for i in idx:
            _eng._apply_lexicon_flags(toks[i])
            if toks[i].is_nrc or toks[i].is_nmc or toks[i].is_rasoul:
                toks[i].is_name = False
        _eng._apply_phrase_flags(clean, toks)
        _eng._apply_context_name_rule(toks)
        return toks

    # A real Khoei entry shape: header, subject, teachers, students, sources.
    text = ("356 - ابراهيم الشيباني: روى عن ابي الجارود، وروى عنه حرب بن "
            "الحسين. التهذيب الجزء 4 الحديث 987. "
            "357 - ابراهيم بن ابي البلاد: روى عن ابي عبد الله عليه السلام، "
            "وروى عنه محمد بن يحيا وعلي بن ابراهيم.")
    clean, _ = normalize(text)
    tokens = analyze(clean)

    print("=== NO-GRAPH (paper's baseline: nothing ever confirms) ===")
    fsm = BiographyFSM()
    for m in fsm.run(tokens):
        print(f"  [{m.start:>4}-{m.end:<4}] {m.name}")
    print("  stats:", fsm.stats)
    print("  paragraph delimiters:", fsm.paragraph_delimiters)

    print()
    print("=== WITH a stub graph (confirms anything containing 'بن') ===")
    fsm2 = BiographyFSM(confirm=lambda n: "بن" in n.full_name.split())
    for m in fsm2.run(tokens):
        mark = "  <-- confirmed, budget renewed" if m.confirmed else ""
        print(f"  [{m.start:>4}-{m.end:<4}] {m.name}{mark}")
    print("  stats:", fsm2.stats)
