"""
fsm.py
======
IDEA: The finite state machine — the BRAIN of segmentation (paper §3.1, Fig.3).

Faithful port of the old system's getNextState() (hadithCommon.cpp:587,
~500 lines of C++) — every behavioral detail preserved, rewritten for
readability. It walks the text word by word (TokenInfo flags from
engine.py) and decides:

    "a sanad may be starting"  /  "still collecting this narrator's name"
    "narrator finished, next one coming"  /  "sanad done, matn begins"
    "false alarm — this was normal text"

THE 5 STATES (old enum, hadithCommon.h:70):
    TEXT      wandering in normal text, waiting for a sanad to start
    NAME      inside a narrator's name        (جعفر، محمد...)
    NMC       after a name connector          (بن، ابو، ام...)
    NRC       after a narration connector     (عن، حدثنا، قال...)
    STOP_WORD reached the Prophet/Imam        (رسول الله، عليه السلام)

TRANSITION TABLE (as implemented below — matches the old switch/case):

  state   sees NAME    sees NMC     sees NRC     sees RASOUL  sees other
  ------- ------------ ------------ ------------ ------------ -------------
  TEXT    ->NAME (a)   ->NMC (b)    ->NRC        ->STOP (c)   stay
  NAME    stay(multi)  ->NMC        ->NRC (d)    ->STOP       (tolerated->NMC)
  NMC     ->NAME       stay (e)     ->NRC (d)    ->STOP       stay, count++ (e)
  NRC     ->NAME (new  ->NMC (f)    stay,        ->STOP       stay, count++ (g)
             narrator)               count++ (g)
  STOP    (exit: sanad complete -> TEXT, signal chain_end)    extend if RASOUL

  (a) only right after a sentence boundary (old previousPunctuationInfo.fullstop)
  (b) only family connectors (عن ابيه starting a chain), same boundary rule
  (c) unless preceded by عبد/بن — then رسول is part of a NAME! (old
      familyConnectorOr3abid check: "عبد الرسول" is a person's name)
  (d) عن fast-track: after عن a NAME is expected immediately (old _3an flag)
  (e) tolerance: up to nmc_max non-name words inside a name; if the name had
      a valid family connector, ONE extra chance (old nmcValid second chance)
  (f) family NMC after NRC = new narrator starting with kinship: عن ابيه
  (g) tolerance: up to nrc_max words between narrators, then give up -> TEXT
      (we drifted into the matn)

CHAIN-END SIGNALS (old `return false`):
  * leaving STOP_WORD state (the clean end: matn definitely starts)
  * ending punctuation (fullstop / hadith number in hadith mode)
  * nrc_max overflow (drifted into matn)
  * nmc_max overflow without the second chance
The segmenter then accepts the chain iff narrator_count >= narr_min.

All parameters tunable (old hadithParameters):
    nmc_max=3, nrc_max=5, narr_min=3  (bio_* variants come in Phase D)
"""

from dataclasses import dataclass, field
from enum import Enum

import lexicons
from models import (Chain, Narrator, NarratorConnector, ConnectorType)


# ===========================================================================
# 1. Parameters — old hadithParameters (hadithCommon.h:28-54), all tunable
# ===========================================================================

@dataclass
class FsmParams:
    nmc_max: int = 3      # max non-name words tolerated inside a name
    nrc_max: int = 5      # max words between narrators before giving up
    narr_min: int = 3     # min narrators for a valid sanad (used by segmenter)

    # ---- heuristics THIS PORT ADDED; the old FSM had none of them ----------
    # Kept switchable so a reproduction claim can present both columns rather
    # than quietly shipping three undocumented rules. Note these are
    # ADDITIONS, not the port's fidelity FIXES (C1's name gating, C3's
    # honorific handling, the graph guards) — those are always on, because
    # turning them off would make the port LESS faithful, not more.
    matn_cues: bool = True          # 'قال:' can close a sanad
    pos_hard_stop: bool = True      # a verb/pronoun/particle breaks a name
    one_chain_per_hadith: bool = True   # after the Imam, no new sanad until
                                        # the next hadith number

    @classmethod
    def faithful(cls, **kw):
        """
        Exactly the old system's automaton: none of the three additions.
        Expect more matn leakage — that is the point of having the comparison.
        """
        return cls(matn_cues=False, pos_hard_stop=False,
                   one_chain_per_hadith=False, **kw)


class State(Enum):
    TEXT = "TEXT"
    NAME = "NAME"
    NMC = "NMC"
    NRC = "NRC"
    STOP = "STOP_WORD"


# What the FSM emits when a chain attempt ends
@dataclass
class ChainCandidate:
    chain: Chain
    sanad_start: int          # clean-text offset where the sanad started
    sanad_end: int            # clean-text offset where it ended
    ended_by: str             # "rasoul" | "punctuation" | "overflow" | "eof"

    @property
    def narrator_count(self):
        return self.chain.narrator_count


# ===========================================================================
# 2. The FSM
# ===========================================================================

class HadithFSM:
    """
    Feed it the token list of a (normalized) text; collect ChainCandidates.

        fsm = HadithFSM(FsmParams())
        candidates = fsm.run(tokens)
    """

    # punctuation that means "sentence ended here" (old ending_punc =
    # fullstop; hadith numbers also force it — handled in run())
    ENDING_PUNCT = {".", "؟", "?", "!", "؛", ";"}
    SOFT_PUNCT = {"،", ",", ":", "-", "(", ")"}

    # Words that introduce the matn when followed by ':' — قال: / قالت: /
    # سالته: ... After the last narrator, hadith books write "قال:" and then
    # the content. The OLD SYSTEM HAD NO SUCH RULE (it closed a sanad only
    # on a rasoul word, ending punctuation, or tolerance overflow) — which is
    # why it swallowed matn text containing names, e.g.
    #   "...محمد بن عيسى قال: كنت انا وابن فضال جلوسا اذ اقبل يونس..."
    # produced a fake narrator "انا وابن فضال جلوسا اذ اقبل يونس" (93x!).
    # This is our documented addition.
    MATN_CUES = {"قال", "قالت", "قالوا", "يقول", "قلت", "سالته", "سالت",
                 "كتبت", "سمعته"}

    # CAMeL POS tags that can NEVER be part of a person's name.
    # Discovered on real kafi1 data: the inherited tolerance counters let
    # the FSM swallow matn words into a name, producing fake narrators like
    #   'عبد الله بن جندب انه كتب اليه الرضا'   (310 occurrences!)
    #   'احمد بن الخضيب الا محمد بن الفرج يساله الخروج'
    # A verb/pronoun/particle is grammatically impossible inside an Arabic
    # personal name, so seeing one means the name ended. The OLD SYSTEM had
    # no such rule (it only had the word-count tolerance), which is why it
    # produced the same kind of noise. Our addition, and fully general:
    # the judgement comes from CAMeL's POS tag, not from a word list.
    NAME_BREAKING_POS = {"verb", "verb_pseudo", "pron", "pron_dem",
                         "pron_rel", "pron_interrog", "part", "part_neg",
                         "part_verb", "part_interrog", "conj_sub",
                         "prep", "abbrev", "punc", "interj"}

    def __init__(self, params: FsmParams = None):
        self.params = params or FsmParams()
        self._reset_all()

    # ------------------------------------------------------------ lifecycle
    def _reset_all(self):
        self.state = State.TEXT
        self.candidates = []
        self._reset_chain()
        # old previousPunctuationInfo.fullstop — a chain may only START
        # right after a sentence boundary; text beginning counts as one.
        self.at_sentence_start = True
        # after a sanad completes at the Imam (STOP), the matn of the SAME
        # hadith often mentions the Imam again with عن ("يبلغه عن ابي عبد الله")
        # — that must NOT start a new sanad. We block new chains until the next
        # hadith number resets this flag. (Fixes the #220 "بمصر زنديق" leak.)
        self.chain_done_this_hadith = False

    def _reset_chain(self):
        self.chain = Chain()
        self.current_narrator = None
        self.chain_start = -1
        self.nmc_count = 0
        self.nrc_count = 0
        self.nmc_valid = False        # old nmcValid (second-chance flag)
        self.nmc_second_chance_used = False
        self.last_word_end = -1

    # ------------------------------------------------------- chain building
    # (these replace the old fillStructure(): they build models.py objects)

    def _ensure_narrator(self):
        if self.current_narrator is None:
            self.current_narrator = Narrator()

    def _close_narrator(self):
        if self.current_narrator and not self.current_narrator.is_empty():
            self.chain.add(self.current_narrator)
        self.current_narrator = None

    def _add_name(self, token, learned=False):
        self._ensure_narrator()
        self.current_narrator.add_name(token.word, token.start, token.end,
                                       learned or bool(
                                           token.name_sources == ["context"]))

    def _connector_kind(self, token):
        if token.is_ibn:
            return ConnectorType.IBN
        if token.is_ab:
            return ConnectorType.AB
        if token.is_om:
            return ConnectorType.OM
        return ConnectorType.FAMILY if token.is_nmc else ConnectorType.OTHER

    def _add_name_connector(self, token):
        self._ensure_narrator()
        self.current_narrator.add_connector(token.word, token.start,
                                            token.end,
                                            self._connector_kind(token))

    def _add_narrator_connector(self, token):
        self._close_narrator()
        self.chain.add(NarratorConnector(token.word, token.start, token.end))

    def _mark_rasoul(self, token):
        """
        Attach an honorific / rasoul word.

        AN HONORIFIC IS NEVER A NARRATOR OF ITS OWN. If no narrator is open,
        the word belongs to the one just closed (عن ابي عبد الله عليه السلام),
        not to a fresh node. Without this, 'عليه السلام' became a standalone
        narrator 1,031 times in kafi1 — 11.5% of every narrator slot, and the
        largest node in the whole graph.
        """
        # Already building the Prophet's node -> keep extending it, so
        # 'رسول' + 'الله' + 'صلا الله عليه واله' stay one narrator.
        if self.current_narrator is not None and self.current_narrator.is_rasoul:
            self.current_narrator.add_name(token.word, token.start, token.end)
            return

        if lexicons.is_prophet_token(token.word):
            # The Prophet is a PERSON, not a qualifier: close whatever name is
            # open (عن ابي هريره عن رسول الله -> two narrators) and open his.
            if self.current_narrator is not None:
                self._close_narrator()
            self._ensure_narrator()
            self.current_narrator.add_name(token.word, token.start, token.end)
            self.current_narrator.is_rasoul = True
            return

        # A pure modifier (عليه السلام / رضي الله عنه) contributes NOTHING to
        # the narrator: it is decoration on a name, not part of it. Two
        # reasons it must not be appended:
        #   * identity — 'ابي عبد الله عليه السلام' and 'ابي عبد الله' are the
        #     same person, but the extra words change the exact hash key, so
        #     they would land in different buckets and never be compared;
        #   * is_rasoul — flagging the narrator collapses every flagged node
        #     into ONE graph node, fusing Imam al-Sadiq with Imam al-Baqir.
        # The state machine still moves to STOP, so the sanad still ends here.
        return

    @staticmethod
    def _is_honorific_only(narrator) -> bool:
        """
        Is this "narrator" nothing but an honorific? (عليه السلام / رضي الله
        عنه / a bare الله). A reference to the Prophet himself (رسول الله /
        النبي) does NOT count — that is a real terminal narrator, which is why
        the old system kept those two vocabularies in separate files
        (src/case/stop_words vs src/case/phrases).
        """
        words = [p.text for p in narrator.parts if p.text]
        if not words:
            return False
        if any(lexicons.is_prophet_token(w) for w in words):
            return False
        return all(lexicons.is_honorific_modifier(w) for w in words)

    def _absorb_honorific_narrators(self):
        """
        Safety net for the invariant: whatever state produced it, a narrator
        made only of honorific words is folded into the narrator before it
        (or dropped if it is the first). Catches the cases the state handlers
        cannot see — the flag combination that opens a fresh narrator on an
        honorific depends on the morphology layer, so we enforce the result
        rather than every path into it.
        """
        kept = [item for item in self.chain.items
                if not (isinstance(item, Narrator)
                        and self._is_honorific_only(item))]
        self.chain.items = kept

    def _end_chain(self, end_pos: int, reason: str):
        """Old `return false` — the chain attempt is over."""
        self._close_narrator()
        self._absorb_honorific_narrators()
        self._trim_trailing_connectors()       # old removeLastSpuriousNarrators
        if self.chain.narrator_count > 0:
            self.candidates.append(ChainCandidate(
                chain=self.chain,
                sanad_start=self.chain_start,
                sanad_end=end_pos,
                ended_by=reason))
        self._reset_chain()
        self.state = State.TEXT

    def _trim_trailing_connectors(self):
        """Drop dangling connectors/empty narrators at the chain's tail."""
        while self.chain.items and (
                isinstance(self.chain.items[-1], NarratorConnector)
                or (isinstance(self.chain.items[-1], Narrator)
                    and self.chain.items[-1].is_empty())):
            self.chain.items.pop()

    # ---------------------------------------------------------------- run
    def run(self, tokens, hadith_mode: bool = True):
        """
        Process the whole token stream. Returns list[ChainCandidate].
        hadith_mode: numbers act as hard sentence boundaries (the "N - "
        numbering in hadith books — old: number => fullstop+newline).
        """
        self._reset_all()
        # index for cheap lookahead (used by the matn-cue rule)
        self._token_index = {id(t): i for i, t in enumerate(tokens)}

        for token in tokens:
            # per-token hook; biography mode uses it to drain the tolerance
            # budget (old bio_nrcCount, hadithCommon.cpp:264-268). No-op here.
            self._on_token(token)

            # ---------- punctuation & numbers: boundary bookkeeping -------
            if token.is_punct or token.is_number:
                is_hard = (token.word in self.ENDING_PUNCT
                           or (token.is_number and hadith_mode))
                if is_hard:
                    if self.state != State.TEXT:
                        # ending punctuation closes any open chain
                        # (old: ending_punc -> TEXT_S + return false)
                        self._end_chain(self.last_word_end, "punctuation")
                    self.at_sentence_start = True
                    # a hadith number begins a NEW hadith -> allow a sanad again
                    if token.is_number and hadith_mode:
                        self.chain_done_this_hadith = False
                elif self.state in (State.NAME, State.NMC):
                    # soft punctuation (،) right after a name acts as an
                    # implicit narrator boundary (old <punc2>/<punc3>:
                    # punctuation becomes a zero-length narrator connector)
                    self._close_narrator()
                    self.state = State.NRC
                    self.nrc_count = 0
                continue

            # ---------- the word's type (old WordType priority) ------------
            word_end = token.end

            # DETAIL (c): rasoul word preceded by عبد or بن is a NAME part:
            # "عبد الرسول" is a person! (old familyConnectorOr3abid check)
            is_rasoul_here = token.is_rasoul and not self._prev_is_abd_or_ibn()

            # ---------- matn cue: 'قال:' ends the sanad (our addition) -----
            # Only when we have already collected narrators AND the colon is
            # NOT followed by a narration word or a name.
            # Why the extra test (found on real kafi1 data): sanads often
            # contain a mid-chain 'قال:' that CONTINUES the chain, e.g.
            #   'اخبرنا ابو جعفر محمد بن يعقوب قال: حدثني عده من اصحابنا عن...'
            # Closing there would truncate a perfectly good sanad. But
            #   '...محمد بن عيسى قال: كنت انا وابن فضال جلوسا...'
            # is a real matn start. The difference is exactly what follows
            # the colon: a narration word / name (chain continues) versus
            # ordinary words (content begins).
            if (self.params.matn_cues
                    and self.state in (State.NAME, State.NMC, State.NRC)
                    and token.word in self.MATN_CUES
                    and self._is_matn_start(tokens, token)):
                self._end_chain(token.start, "matn_cue")
                self.at_sentence_start = False
                self.last_word_end = token.end
                self._last_token = token
                continue

            # ---------- POS hard-stop (our addition) -----------------------
            # A verb / pronoun / particle cannot be inside a personal name.
            # Exempt everything the chain legitimately uses: narration words
            # (عن، حدثنا — themselves verbs/preps), name connectors, names,
            # rasoul words and relative narrators.
            # ALSO exempt VERB-LOOKING NAMES: many Arabic given names are
            # morphologically verbs (يحيى، صالح، سعيد...). CAMeL tags يحيا as
            # a verb, and blindly cutting there truncated 'محمد بن يحيا' to
            # 'محمد بن'. So a verb directly after a name connector (بن/ابو)
            # is treated as a NAME, not a break.
            if (self.params.pos_hard_stop
                    and self.state in (State.NAME, State.NMC)
                    and token.pos in self.NAME_BREAKING_POS
                    and not (token.is_nrc or token.is_nmc or token.is_name
                             or token.is_rasoul or token.is_relative)
                    and not self._after_name_connector()):
                self._close_narrator()
                self.state = State.NRC
                self.nrc_count = 1
                self.nmc_count = 0
                self.last_word_end = token.end
                self._last_token = token
                self.at_sentence_start = False
                continue

            handler = {State.TEXT: self._in_text,
                       State.NAME: self._in_name,
                       State.NMC: self._in_nmc,
                       State.NRC: self._in_nrc,
                       State.STOP: self._in_stop}[self.state]

            # DETAIL (parallel narrators): a word carrying a waw prefix
            # (واحمد / وابن) while we are inside a name means a NEW narrator
            # starts here — 'سهل بن زياد واسحاق بن محمد' is TWO narrators.
            # The old system injected a virtual NRC at the waw position
            # (hadithCommon.cpp:1373-1387); we close the narrator instead.
            # The waw itself is not counted in the tolerance counters
            # (old: `if (!stateInfo.isWaw)`, hadithCommon.cpp:763).
            if token.has_waw and self.state in (State.NAME, State.NMC):
                self._close_narrator()
                self.state = State.NRC
                self.nrc_count = 0
                self.nmc_count = 0
                handler = self._in_nrc

            handler(token, is_rasoul_here)

            self.last_word_end = word_end
            self._last_token = token
            if not (token.is_punct or token.is_number):
                self.at_sentence_start = False

        # end of text: close whatever is open (old end-of-loop flush)
        if self.state != State.TEXT:
            self._end_chain(self.last_word_end, "eof")
        return self.candidates

    def _on_token(self, token):
        """Hook called once per token before dispatch. Overridden by bio mode."""

    def _after_name_connector(self) -> bool:
        """
        Was the previous token a name connector (بن / ابو / ام / family)?
        If so, the current word is the expected name after it — even if its
        POS looks like a verb (يحيا، صالح، سعيد are real given names).
        """
        prev = self._last_token
        return bool(prev is not None and prev.is_nmc)

    def _is_matn_start(self, tokens, token) -> bool:
        """
        Is this cue the real start of the matn?
        Requires: cue + ':' , and the word AFTER the colon is neither a
        narration word (حدثني/عن/اخبرنا) nor a name — in those cases the
        sanad continues and we must not close it.
        """
        idx = self._token_index.get(id(token))
        if idx is None or idx + 1 >= len(tokens):
            return False
        if tokens[idx + 1].word != ":":
            return False
        # look at the first real word after the colon
        j = idx + 2
        while j < len(tokens) and (tokens[j].is_punct or tokens[j].is_number):
            j += 1
        if j >= len(tokens):
            return True                       # nothing after -> matn/end
        nxt = tokens[j]
        chain_continues = (nxt.is_nrc or nxt.is_name or nxt.is_nmc
                           or nxt.is_relative)
        return not chain_continues

    _last_token = None

    def _prev_is_abd_or_ibn(self):
        prev = self._last_token
        if prev is None:
            return False
        return prev.is_ibn or prev.word in ("عبد", "وعبد")

    # ======================================================== state handlers
    # Each mirrors one `case` of the old switch (hadithCommon.cpp:624-1095).

    def _in_text(self, token, is_rasoul):
        p = self.params

        # after a sanad already completed at the Imam in THIS hadith, do not
        # start a new one from matn text (e.g. "يبلغه عن ابي عبد الله") — wait
        # for the next hadith number. (#220 fix)
        if self.params.one_chain_per_hadith and self.chain_done_this_hadith:
            return

        if token.is_name:
            # DETAIL (a): in hadith mode a chain only starts at a sentence
            # start (old: requires previousPunctuationInfo.fullstop)
            if not self.at_sentence_start:
                return
            self._reset_chain()
            self.chain_start = token.start
            self._add_name(token)
            self.state = State.NAME

        elif token.is_nrc:
            # NRC can start a chain anywhere (حدثنا at hadith beginning)
            self._reset_chain()
            self.chain_start = token.start
            self._add_narrator_connector(token)
            self.state = State.NRC
            self.nrc_count = 1

        elif token.is_nmc and (self._is_family(token) or token.is_ab):
            # DETAIL (b): connector chain start at a sentence boundary —
            # family (عن ابيه...) or kunya (ابو جعفر قال...)
            if not self.at_sentence_start:
                return
            self._reset_chain()
            self.chain_start = token.start
            self._add_name_connector(token)
            self.state = State.NMC
            self.nmc_count = 1
            self.nmc_valid = True
        # else: stay in TEXT

    def _in_name(self, token, is_rasoul):
        if is_rasoul:
            # "عليه السلام" after a name (أبي جعفر عليه السلام) is an honorific
            # that belongs to the SAME narrator — attach it, keep the narrator
            # OPEN so _in_stop can append further honorific words (عليه+السلام),
            # then _end_chain closes it cleanly.
            self._mark_rasoul(token)
            self.state = State.STOP
            return

        if token.is_nmc:
            self._add_name_connector(token)
            self.state = State.NMC
            self.nmc_count = 1
            # old: nmcValid = isFamilyConnectorOrPossessivePlace
            self.nmc_valid = self._is_family(token) or token.is_ibn \
                or token.is_ab or token.is_om
            self.nmc_second_chance_used = False

        elif token.is_nrc:
            # narrator finished; NRC bridges to the next one
            # DETAIL: bare و is not counted (old isWaw) — handled naturally
            # since our tokens keep وعن attached; a lone و is not NRC.
            self._add_narrator_connector(token)
            self.state = State.NRC
            self.nrc_count = 1
            if self._is_3an(token):
                # DETAIL (d): عن fast-track — a NAME is expected right away
                self.state = State.NRC   # stays NRC; next NAME opens narrator
        else:
            if token.is_name:
                # A relative narrator (عن ابيه / وعنه) is a COMPLETE one-word
                # narrator whose identity the graph resolves later. It must
                # not swallow the name that follows it — 'روى عنه حرب بن
                # الحسين' is the connector عنه plus the narrator حرب بن
                # الحسين, not one narrator called 'عنه حرب بن الحسين'.
                if self.current_narrator is not None \
                        and self.current_narrator.is_relative:
                    self._close_narrator()
                self._add_name(token)              # multi-word name: محمد + يعقوب
            else:
                # unknown word inside a name -> NMC-style tolerance
                self._add_name_connector(token)
                self.state = State.NMC
                self.nmc_count = 1
                self.nmc_valid = False

    def _in_nmc(self, token, is_rasoul):
        p = self.params

        if is_rasoul:
            # honorific attaches to the current narrator and keeps it open
            # (see _in_name); _end_chain will close it after STOP.
            self._mark_rasoul(token)
            self.state = State.STOP
            return

        if token.is_nrc:
            self.nmc_count = 0
            self._add_narrator_connector(token)
            self.state = State.NRC
            self.nrc_count = 1

        elif token.is_name:
            self.nmc_count = 0
            self._add_name(token)
            self.state = State.NAME

        else:
            # tolerance counter (old: nmcCount++ up to nmc_max, with ONE
            # second chance if the name had a valid family connector)
            self.nmc_count += 1
            if self.nmc_count > p.nmc_max:
                if self.nmc_valid and not self.nmc_second_chance_used:
                    self.nmc_count = 0                    # old second chance
                    self.nmc_second_chance_used = True
                    self.nmc_valid = False
                else:
                    self._end_chain(self.last_word_end, "overflow")
                    return
            self._add_name_connector(token)

    def _in_nrc(self, token, is_rasoul):
        p = self.params

        if is_rasoul:
            # rasoul right after عن with no name between — _mark_rasoul knows
            # whether to open the Prophet's node or attach a modifier to the
            # narrator already in the chain.
            self._mark_rasoul(token)
            self.state = State.STOP
            return

        if token.is_name:
            # a NEW narrator begins
            self.nrc_count = 1
            self._add_name(token)
            self.state = State.NAME

        elif token.is_relative:
            # DETAIL (old abyi+suffix, hadithCommon.cpp:103-106):
            # عن ابيه / عن جده — a RELATIVE narrator: a complete one-word
            # narrator whose identity is resolved later (graph phase).
            self.nrc_count = 1
            self._close_narrator()
            self._ensure_narrator()
            self.current_narrator.add_name(token.word, token.start, token.end)
            self.current_narrator.is_relative = True
            self.state = State.NAME

        elif token.is_nmc:
            # DETAIL (f): a name connector after NRC = a NEW narrator is
            # starting with it:
            #   عن ابيه   (family)          عن ابي جعفر (kunya: AB)
            #   حدثنا ابو جعفر (kunya)     عن ابن عباس (IBN-first name)
            self.nrc_count = 1
            self._add_name_connector(token)
            self.state = State.NMC
            self.nmc_count = 1
            self.nmc_valid = True
            self.nmc_second_chance_used = False

        elif token.is_nrc:
            self.nrc_count += 1
            self._add_narrator_connector(token)
            if self.nrc_count >= p.nrc_max:
                self._end_chain(self.last_word_end, "overflow")

        else:
            # DETAIL (g): plain words between narrators — tolerated up to
            # nrc_max, then we've drifted into the matn -> give up
            self.nrc_count += 1
            if self.nrc_count >= p.nrc_max:
                self._end_chain(self.last_word_end, "overflow")

    def _in_stop(self, token, is_rasoul):
        if token.is_rasoul:
            # extend the honorific: عليه + السلام, صلى الله عليه وسلم...
            # _mark_rasoul routes to the right target and decides is_rasoul.
            self._mark_rasoul(token)
        else:
            # THE clean sanad-complete signal (old STOP_WORD_S exit,
            # hadith mode: return false). The matn starts at this token.
            self._end_chain(token.start, "rasoul")
            # sanad ended at the Imam: block any further sanad in THIS hadith
            # until the next number (blocks matn "عن ابي عبد الله" re-triggers).
            if self.params.one_chain_per_hadith:
                self.chain_done_this_hadith = True

    # ---------------------------------------------------------- small utils
    @staticmethod
    def _is_family(token):
        return token.is_nmc and not (token.is_ibn or token.is_ab
                                     or token.is_om) or token.is_relative

    @staticmethod
    def _is_3an(token):
        return token.word in ("عن", "وعن") or token.lemma == "عن"


# ---------------------------------------------------------------------------
# Quick demo:  python fsm.py    (uses engine -> needs CAMeL installed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from normalization import normalize
    from engine import ArabicEngine

    original = ("1 - أخبرنا أبو جعفر محمد بن يعقوب قال: حدثني عدة من "
                "أصحابنا عن أحمد بن محمد عن الحسن بن محبوب عن أبي جعفر "
                "عليه السلام قال: لما خلق الله العقل استنطقه ثم قال له أقبل")
    clean_text, _ = normalize(original)

    print("Loading engine...")
    engine = ArabicEngine(camel_mode="mle", use_wojood=False)
    tokens = engine.analyze(clean_text)

    fsm = HadithFSM(FsmParams())
    candidates = fsm.run(tokens)

    print(f"\nchains found: {len(candidates)}\n")
    for c in candidates:
        print(f"ended_by={c.ended_by}  narrators={c.narrator_count}  "
              f"span={c.sanad_start}->{c.sanad_end}")
        for n in c.chain.narrators:
            tag = " (RASOUL)" if n.is_rasoul else ""
            print("   -", n.full_name, tag)
        matn_preview = clean_text[c.sanad_end:c.sanad_end + 50]
        print("   matn starts:", matn_preview, "...")
