"""
engine.py
=========
IDEA: The analysis engine — answers "what is this word?" for every word.

This file replaces the ENTIRE old Sarf + MySQL stack. In the old system
(src/case/hadithCommon.h:147), the `hadith_stemmer` raised boolean flags
per word (name, nrc, nmc, ibn, possessive, stopword...) by looking up a
MySQL lexicon. The FSM consumed ONLY those flags.

Here, the same flags are produced by three modern sources, merged:

    1. CAMeL Tools  -> lemma + POS per word        (replaces Sarf morphology)
    2. Wojood NER   -> person-name spans (PERS)    (replaces name lexicon)
    3. lexicons.py  -> closed-class words           (NRC/NMC/rasoul lists)

Paper reference (Figure 3): this is the <word, cats> feature vector that
feeds the finite state machine.

--------------------------------------------------------------------------
KEY DESIGN: hybrid name detection (defensive against domain shift)
    Wojood is trained on modern Arabic; hadith text is classical.
    So `is_name` never depends on one source alone:

        is_name = Wojood says PERS
                  OR CAMeL says noun_prop
                  OR context: word squeezed between name connectors
                     (the old system's `tryToLearnNames` idea)
--------------------------------------------------------------------------
USAGE:
    engine = ArabicEngine()            # loads CAMeL (+ Wojood if available)
    tokens = engine.analyze(clean_text)
    for t in tokens: print(t.word, t.is_name, t.is_nrc)
"""

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict

from normalization import normalize_word
import lexicons


# ===========================================================================
# 1. TokenInfo — the feature vector (paper Fig.3 / old hadith_stemmer flags)
# ===========================================================================

@dataclass
class TokenInfo:
    """Everything the FSM needs to know about one word."""

    # --- identity & position (offsets are into the CLEAN text) ---
    word: str
    start: int
    end: int                      # exclusive: clean_text[start:end] == word

    # --- morphology (from CAMeL) ---
    lemma: str = ""               # normalized lemma, e.g. حدثنا -> حدث
    pos: str = ""                 # CAMeL POS tag, e.g. noun_prop / verb / prep

    # --- the flags (same set as old hadith_stemmer, hadithCommon.h:147) ---
    is_name: bool = False         # part of a person name
    is_nrc: bool = False          # narration word: حدثنا / اخبرنا / عن / قال
    is_nmc: bool = False          # name connector: بن / ابو / ام / عمه ...
    is_ibn: bool = False          # specifically the (son) connector: بن/ابن
    is_ab: bool = False           # specifically (father): ابو/ابي/ابا
    is_om: bool = False           # specifically (mother): ام
    is_rasoul: bool = False       # rasoul / honorific word (ends the sanad)
    is_relative: bool = False     # relative narrator: ابيه / جده / عنه
    is_punct: bool = False        # punctuation token: : ، . ( ) -
    is_number: bool = False       # digits (hadith numbering "1 - ")

    # waw conjunction attached to the front of the word: واحمد / وابن
    # (old system: hadith_stemmer.has_waw via prefix analysis,
    #  hadithCommon.h:148 — used to split PARALLEL narrators)
    has_waw: bool = False

    # WEAK name evidence: CAMeL called it noun_prop and nothing else agrees.
    # Kept separate from is_name because the old system did NOT trust this
    # category on its own — see _promote_name_candidates.
    is_name_candidate: bool = False

    # the word carries a pronominal enclitic (كتابه، عنهم). The old analyze()
    # required `Suffix->info.finish - Suffix->info.start < 0`, i.e. NO suffix,
    # before accepting a word as a name (hadithCommon.h:267).
    has_enclitic: bool = False

    # --- provenance: which source(s) said it's a name (for debugging/report)
    name_sources: list = field(default_factory=list)  # e.g. ["wojood","camel"]


# ===========================================================================
# 2. Position-aware tokenizer
#    (simple whitespace/punct split that KEEPS start/end offsets — CAMeL's
#     own tokenizer drops positions, and the whole pipeline lives on offsets)
# ===========================================================================

PUNCTUATION_CHARS = set(":،,.؛;؟?!()-*\u2013\u2014/")


def tokenize_with_positions(text: str):
    """
    Split clean text into tokens, keeping (word, start, end) for each.
    Punctuation characters become single-char tokens (the FSM uses them
    as boundary hints, exactly like the old previousPunctuationInfo).
    """
    tokens = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in PUNCTUATION_CHARS:
            tokens.append((ch, i, i + 1))
            i += 1
            continue
        # a word: consume until space/punct
        start = i
        while i < n and not text[i].isspace() and text[i] not in PUNCTUATION_CHARS:
            i += 1
        tokens.append((text[start:i], start, i))
    return tokens


# ===========================================================================
# 3. CAMeL layer — lemma + POS  (replaces Sarf morphology)
# ===========================================================================

class CamelLayer:
    """
    Wraps the CAMeL disambiguator. `mode` is a tunable parameter, in the
    spirit of the old system where everything was a parameter:
        mode="mle"  -> fast, good accuracy (default)
        mode="bert" -> slower, more accurate (needs GPU ideally)
    """

    def __init__(self, mode: str = "mle"):
        self.mode = mode
        if mode == "bert":
            from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
            self.disambiguator = BERTUnfactoredDisambiguator.pretrained()
        else:
            from camel_tools.disambig.mle import MLEDisambiguator
            self.disambiguator = MLEDisambiguator.pretrained()

    def analyze_words(self, words: list):
        """
        Returns a list of (lemma, pos, has_enclitic) aligned with `words`.
        Lemmas are normalized (CAMeL returns them WITH diacritics,
        e.g. أَخْبَر — we normalize to اخبر so lexicons can match).

        `has_enclitic` stands in for the old suffix test: a word carrying a
        pronominal enclitic is not a bare proper name (hadithCommon.h:267
        required no suffix at all before accepting one).
        """
        results = []
        disambiguated = self.disambiguator.disambiguate(words)
        for d in disambiguated:
            if d.analyses:
                analysis = d.analyses[0].analysis
                lemma = normalize_word(analysis.get("lex", ""))
                pos = analysis.get("pos", "")
                enc = str(analysis.get("enc0", "") or "")
                has_enclitic = bool(enc) and enc not in ("0", "na", "-")
            else:
                lemma, pos, has_enclitic = "", "", False
            results.append((lemma, pos, has_enclitic))
        return results


# ===========================================================================
# 4. Wojood layer — person-name spans  (replaces the old name lexicon)
#    Loaded lazily; if the model isn't installed/downloadable the engine
#    still works (hybrid rule covers it) — graceful degradation.
# ===========================================================================

class WojoodLayer:
    """
    Person-name detection via the OFFICIAL Wojood model (SinaLab, Birzeit).
    Uses our standalone runner (wojood.py) — the official arabiner package
    is uninstallable (setup.py bug) and the HF repo is in a non-standard
    format; see the documentation at the top of wojood.py.
    Word-level API: input words -> aligned boolean flags.
    """

    def __init__(self, enabled: bool = True):
        self.ner = None
        self.available = False
        if not enabled:
            return
        try:
            from wojood import WojoodNER
            self.ner = WojoodNER()
            self.available = True
        except Exception as error:
            # No internet / files missing -> engine still works without it.
            print(f"[engine] Wojood not available ({type(error).__name__}: "
                  f"{error}) -> running in CAMeL+context mode.")

    def person_flags(self, words):
        """words -> aligned list of booleans (True = part of person name)."""
        if not self.available:
            return [False] * len(words)
        return self.ner.person_flags(words)


# ===========================================================================
# 5. The engine — merges everything into TokenInfo flags
# ===========================================================================

class ArabicEngine:

    def __init__(self, camel_mode: str = "mle", use_wojood: bool = True,
                 cache_dir: str = ".engine_cache", strict_names: bool = True):
        """
        strict_names=True  (default) reproduces the old gating: a bare CAMeL
                           `noun_prop` is accepted as a name only where a name
                           is expected. See _promote_name_candidates.
        strict_names=False the previous behaviour — accept noun_prop anywhere.
                           Kept so the two can be compared directly (C5).
        """
        self.camel = CamelLayer(mode=camel_mode)
        self.wojood = WojoodLayer(enabled=use_wojood)
        self.strict_names = strict_names
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def _looks_like_name(stripped: str) -> bool:
        """
        After removing a leading و, does the rest look like a name or a
        name/narration connector? Used to confirm the و is a conjunction.
        """
        return (lexicons.is_ibn_word(stripped)
                or lexicons.is_ab_word(stripped)
                or lexicons.is_nrc_word(stripped)
                or lexicons.is_relative_narrator(stripped))

    # ---------------------------------------------------------------- flags
    def _apply_lexicon_flags(self, token: TokenInfo):
        """
        Closed-class flags from lexicons.py (exact & accurate).
        DETAIL (from old system's prefix analysis): words with the waw
        conjunction — وعن / وحدثنا — must still match. The surface word
        fails the list check, but the CAMeL LEMMA strips the prefix
        (lemma of وعن is عن), so we check both word AND lemma.
        """
        w, lemma = token.word, token.lemma
        token.is_nrc = lexicons.is_nrc_word(w) or (lemma and lexicons.is_nrc_word(lemma))
        token.is_ibn = lexicons.is_ibn_word(w) or (lemma and lexicons.is_ibn_word(lemma))
        token.is_ab = lexicons.is_ab_word(w) or (lemma and lexicons.is_ab_word(lemma))
        token.is_om = lexicons.is_om_word(w)
        token.is_nmc = token.is_ibn or token.is_ab or token.is_om \
            or lexicons.is_nmc_word(w)
        token.is_rasoul = lexicons.is_rasoul_word(w)
        token.is_relative = lexicons.is_relative_narrator(w) \
            or (lemma and lexicons.is_relative_narrator(lemma))

    # ---- C1: gate the weak noun_prop category on context ------------------
    MIN_NAME_CHARS = 3          # old: removeDiacritics(stem).count() < 3 -> skip

    @staticmethod
    def _looks_like_nisba(word: str) -> bool:
        """starts with ال and ends in ي — old startsWithAL(n) && last == ya2."""
        return word.startswith("ال") and word.endswith("ي") and len(word) > 3

    def _promote_name_candidates(self, tokens: list):
        """
        Decide which weak `noun_prop` candidates really are names.

        THE PROBLEM THIS FIXES. `pos == "noun_prop"` was accepted anywhere,
        so any capitalised-looking noun in the matn became a narrator. That is
        the root cause of the matn leakage that `MATN_CUES` and
        `NAME_BREAKING_POS` in fsm.py patch downstream — those two heuristics
        exist because names were being invented in the first place.

        THE OLD RULE. `bit_NOUN_PROP` is deliberately NOT in `bits_NAME`
        (hadithCommon.cpp:117-133). It is appended only while
        `tryToLearnNames` is set (hadithCommon.h:259), which happens in
        exactly four places:

          1. the current word is a family connector      (cpp:1160, familyNMC)
          2. we are just after a narration word, at most one NRC deep and not
             merely punctuation                          (cpp:1172, nrcLearning)
          3. the PREVIOUS word was a narration word      (cpp:1178)
          4. the word looks like a nisba (ال...ي) while we are in an NRC or
             NAME context                                (cpp:1227)

        plus two hard filters in analyze(): the word must carry NO suffix
        (h:267) and its stem must be at least 3 characters (h:269).

        Translated to the token stream — which is all this layer has, since it
        runs before the FSM — that is: a name connector or narration word on
        one side, or a connector coming next, or a nisba in narration context.
        """
        n = len(tokens)

        def neighbour(i, step):
            j = i + step
            while 0 <= j < n and (tokens[j].is_punct or tokens[j].is_number):
                j += step
            return tokens[j] if 0 <= j < n else None

        for i, token in enumerate(tokens):
            if not token.is_name_candidate or token.is_name:
                continue
            if len(token.word) < self.MIN_NAME_CHARS:
                continue                      # old: stem shorter than 3
            if token.has_enclitic:
                continue                      # old: must have no suffix

            prev, nxt = neighbour(i, -1), neighbour(i, +1)
            expected = (
                (prev is not None and (prev.is_nmc or prev.is_nrc))   # 1, 2, 3
                or (nxt is not None and nxt.is_nmc)                   # ... بن X
                or (self._looks_like_nisba(token.word)                # 4
                    and prev is not None and (prev.is_nrc or prev.is_name))
            )
            if expected:
                token.is_name = True
                token.name_sources.append("camel-learned")

    def _apply_context_name_rule(self, tokens: list):
        """
        Old system's `tryToLearnNames` idea: a word between two name
        connectors must be a name:  ... بن  X  بن ...  ->  X is a name.
        Also: the word right AFTER ابو/ابي (kunya) is a name: ابو جعفر.
        """
        for i, token in enumerate(tokens):
            if token.is_name or token.is_nmc or token.is_nrc \
               or token.is_punct or token.is_number or token.is_rasoul:
                continue
            prev_token = tokens[i - 1] if i > 0 else None
            next_token = tokens[i + 1] if i + 1 < len(tokens) else None
            squeezed = (prev_token and prev_token.is_ibn
                        and next_token and next_token.is_ibn)
            after_kunya = prev_token and prev_token.is_ab
            before_ibn = next_token and next_token.is_ibn
            if squeezed or after_kunya or before_ibn:
                token.is_name = True
                token.name_sources.append("context")

    # -------------------------------------------------------------- analyze
    def analyze(self, clean_text: str) -> list:
        """
        Full analysis of (already normalized) text -> list[TokenInfo].
        Steps: tokenize -> CAMeL -> Wojood -> lexicons -> hybrid merge.
        """
        # 1) tokenize with positions
        raw_tokens = tokenize_with_positions(clean_text)
        tokens = [TokenInfo(word=w, start=s, end=e) for (w, s, e) in raw_tokens]

        # mark punctuation / numbers immediately (no morphology needed)
        word_indexes = []
        for idx, token in enumerate(tokens):
            if token.word in PUNCTUATION_CHARS:
                token.is_punct = True
            elif token.word.isdigit():
                token.is_number = True
            else:
                word_indexes.append(idx)

        # 2) CAMeL lemma+POS (only for real words)
        words = [tokens[i].word for i in word_indexes]
        if words:
            for idx, (lemma, pos, enc) in zip(word_indexes,
                                              self.camel.analyze_words(words)):
                tokens[idx].lemma = lemma
                tokens[idx].pos = pos
                tokens[idx].has_enclitic = enc
                if pos == "noun_prop":
                    # WEAK evidence only. The old system kept this category
                    # (bit_NOUN_PROP) OUT of bits_NAME and admitted it solely
                    # when `tryToLearnNames` was on — i.e. in a position where
                    # a name was already expected. See
                    # _promote_name_candidates for the four contexts.
                    if self.strict_names:
                        tokens[idx].is_name_candidate = True
                    else:
                        tokens[idx].is_name = True
                        tokens[idx].name_sources.append("camel")

        # 2b) waw-prefix detection (old hadith_stemmer.has_waw).
        #     The old system got this from Sarf's prefix analysis; we get it
        #     from CAMeL: if the word starts with و but its LEMMA does not,
        #     the و is a prefix (conjunction), not part of the name.
        #     Critical for PARALLEL narrators: 'سهل بن زياد واسحاق بن محمد'
        #     must become two narrators, not one long name.
        for idx in word_indexes:
            token = tokens[idx]
            w, lemma = token.word, token.lemma
            if len(w) > 2 and w.startswith("و"):
                stripped = w[1:]
                lemma_lacks_waw = bool(lemma) and not lemma.startswith("و")
                if lemma_lacks_waw or self._looks_like_name(stripped):
                    token.has_waw = True
                    # strip the conjunction from the token text and shift its
                    # start, so the NAME stored in the graph is 'اسحاق', not
                    # 'واسحاق' (otherwise the same person would never merge).
                    token.word = stripped
                    token.start += 1

        # 3) Wojood PERS flags -> word-aligned booleans from the official
        #    model (only runs if Wojood is available)
        if self.wojood.available and word_indexes:
            pers_flags = self.wojood.person_flags(
                [tokens[i].word for i in word_indexes])
            for idx, is_pers in zip(word_indexes, pers_flags):
                if is_pers:
                    tokens[idx].is_name = True
                    if "wojood" not in tokens[idx].name_sources:
                        tokens[idx].name_sources.append("wojood")

        # 4) lexicon flags — closed-class lists BEAT the models:
        #    a narration/connector word is never treated as a name.
        for idx in word_indexes:
            token = tokens[idx]
            self._apply_lexicon_flags(token)
            if token.is_nrc or token.is_nmc or token.is_rasoul:
                token.is_name = False
                token.is_name_candidate = False
                token.name_sources.clear()

        # 5) multi-word phrases — token-level flags can't see them.
        #    عليه السلام (honorific -> ends sanad), عده من اصحابنا
        #    (compound narrator -> IS a narrator), ابي عبد الله (imam kunya).
        self._apply_phrase_flags(clean_text, tokens)

        # 6) C1: promote weak noun_prop candidates only where a name is
        #    expected (the old tryToLearnNames gate). Runs before the
        #    zero-evidence context rule so the two do not double-fire.
        if self.strict_names:
            self._promote_name_candidates(tokens)

        # 7) context rule for words with NO morphological evidence at all
        self._apply_context_name_rule(tokens)

        return tokens

    def _apply_phrase_flags(self, clean_text: str, tokens: list):
        """Find known multi-word phrases and flag their tokens."""

        def mark_ranges(phrases, flag_setter):
            for (p_start, p_end, _) in lexicons.find_phrases_in(clean_text,
                                                                phrases):
                for token in tokens:
                    if token.start < p_end and token.end > p_start:
                        flag_setter(token)

        def set_rasoul(token):
            token.is_rasoul = True
            token.is_name = False          # honorific words are not names
            token.name_sources.clear()

        def set_compound_name(token):
            token.is_name = True
            if "compound" not in token.name_sources:
                token.name_sources.append("compound")
            token.is_nrc = False           # 'من' inside the phrase is not NRC

        mark_ranges(lexicons.HONORIFIC_PHRASES | lexicons.RASOUL_WORDS,
                    set_rasoul)
        mark_ranges(lexicons.COMPOUND_NARRATORS | lexicons.IMAM_KUNYA_ENDINGS,
                    set_compound_name)

    # -------------------------------------------------------------- caching
    def analyze_cached(self, clean_text: str) -> list:
        """
        Same as analyze(), but caches results per text (md5 key) — the old
        system did the same with its binary trie caches. First run of a
        full book is slow; every run after is instant.
        """
        key = hashlib.md5(clean_text.encode("utf-8")).hexdigest()
        cache_path = os.path.join(self.cache_dir, key + ".json")

        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                return [TokenInfo(**d) for d in json.load(f)]

        tokens = self.analyze(clean_text)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump([asdict(t) for t in tokens], f, ensure_ascii=False)
        return tokens


# ---------------------------------------------------------------------------
# Quick demo:  python engine.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from normalization import normalize

    # A real sentence from kafi1.txt (first hadith)
    original = ("أخبرنا أبو جعفر محمد بن يعقوب قال: حدثني عدة من أصحابنا "
                "عن أبي جعفر عليه السلام قال: لما خلق الله العقل")
    clean_text, index_map = normalize(original)

    print("Loading engine...")
    engine = ArabicEngine(camel_mode="mle", use_wojood=True)
    print("Wojood available:", engine.wojood.available, "\n")

    tokens = engine.analyze(clean_text)

    header = f"{'word':<12} {'lemma':<10} {'pos':<10} flags"
    print(header)
    print("-" * 50)
    for t in tokens:
        flags = []
        if t.is_name:     flags.append("NAME(" + "+".join(t.name_sources) + ")")
        if t.is_nrc:      flags.append("NRC")
        if t.is_ibn:      flags.append("IBN")
        elif t.is_nmc:    flags.append("NMC")
        if t.is_rasoul:   flags.append("RASOUL")
        if t.is_relative: flags.append("REL")
        if t.is_punct:    flags.append("punct")
        if t.is_number:   flags.append("num")
        print(f"{t.word:<12} {t.lemma:<10} {t.pos:<10} {' '.join(flags)}")
