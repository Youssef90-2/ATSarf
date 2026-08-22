import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from normalization import normalize_word
import lexicons


# 1. TokenInfo — the feature vector 
@dataclass
class TokenInfo:

    word: str      #l kelme 
    start: int      #wen balachit 
    end: int         #wen kholset            

    #  from CAMeL
    lemma: str = ""               # normalized lemma,  حدثنا -> حدث
    pos: str = ""                 # CAMeL POS tag,  noun_prop / verb / prep

    # --- the flags 
    is_name: bool = False         # part of a person name
    is_nrc: bool = False          # narration word: حدثنا / اخبرنا / عن / قال
    is_nmc: bool = False          # name connector: بن / ابو / ام / عمه ...
    is_ibn: bool = False          #  the (son) connector: بن/ابن
    is_ab: bool = False           #  (father): ابو/ابي/ابا
    is_om: bool = False           #  (mother): ام
    is_rasoul: bool = False       # rasoul / honorific word (ends the sanad)
    is_relative: bool = False     #  narrator: ابيه / جده / عنه
    is_punct: bool = False        # punctuation token: : ، . ( ) -
    is_number: bool = False       # digits (hadith numbering "1 -)

    has_waw: bool = False  # وأحمد ya3ne waw ma3 l esem jeye 

    is_name_candidate: bool = False #mmkn tkun ba3den name (CAMeL called it noun_prop and nothing else agrees)

   
    has_enclitic: bool = False  #fiha damir  كتابه

    #  which source(s) said it's a name (for debugging/report)
    name_sources: list = field(default_factory=list)  # ["wojood","camel"]


#  lezem ykoun m3e l offsets la kel word , fa ma fine e3temd 3al tokenizer la Camel 
# la2ano ma bya3tine l offset , w aham shi l offset 3ende ba3den lal pipline bl FSM

PUNCTUATION_CHARS = set(":،,.؛;؟?!()-*\u2013\u2014/")

def tokenize_with_positions(text: str):

   # Split clean text into tokens, keeping (word, start, end) for each.
   # token + position
   
    tokens = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue          # if space ---> continue
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
     #example of output ([("أخبرنا", 0, 6), ("أبو", 7, 10), ("جعفر", 11, 15), ...])
                #  (word, start, end)



# 3. CAMeL layer — lemma + POS  (replaces Sarf morphology)

class CamelLayer:
    
    def __init__(self, mode: str = "mle"):
        self.mode = mode
        if mode == "bert":
            from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
            self.disambiguator = BERTUnfactoredDisambiguator.pretrained() #good accuracy
        else:
            from camel_tools.disambig.mle import MLEDisambiguator
            self.disambiguator = MLEDisambiguator.pretrained() 
            # slower, more accurate



    def analyze_words(self, words: list):
      
        results = []
        disambiguated = self.disambiguator.disambiguate(words)
        for d in disambiguated:
            if d.analyses:
                analysis = d.analyses[0].analysis #camel return lemma with diacritics,fa lezm e3mal normalization warha,kermel l matching ma3 lexixon.py
                lemma = normalize_word(analysis.get("lex", ""))
                pos = analysis.get("pos", "")
                enc = str(analysis.get("enc0", "") or "")
                has_enclitic = bool(enc) and enc not in ("0", "na", "-")
            else:
                lemma, pos, has_enclitic = "", "", False
            results.append((lemma, pos, has_enclitic))
        return results  

        # output ha ykun result of tuples (lemma, pos, has_enclitic)،


# 4. Wojood layer — person-name spans  (replaces the old name lexicon)

class WojoodLayer:
    
    #Person-name detection via  Wojood model (SinaLab, Birzeit).
    #Uses our standalone runner (wojood.py) 
    
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
        #words ->  (True = part of person name)
        if not self.available:
            return [False] * len(words)
        return self.ner.person_flags(words)




# 5. The engine — merges everything into TokenInfo flags


class ArabicEngine:

    def __init__(self, camel_mode: str = "mle", use_wojood: bool = True,
                 cache_dir: str = ".engine_cache", strict_names: bool = True):
        
        #strict_names=True  (default) reproduces the old gating: a bare CAMeL
                        #   `noun_prop` is accepted as a name only where a name
                         #  is expected. See _promote_name_candidates.
        self.camel = CamelLayer(mode=camel_mode)
        self.wojood = WojoodLayer(enabled=use_wojood)
        self.strict_names = strict_names
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def _looks_like_name(stripped: str) -> bool:
        #Hyde l function kermel l WAW , lama nshila mne3mal test lal kelme li waraha
        return (lexicons.is_ibn_word(stripped)
                or lexicons.is_ab_word(stripped)
                or lexicons.is_nrc_word(stripped)
                or lexicons.is_relative_narrator(stripped))

    #  flags
    def _apply_lexicon_flags(self, token: TokenInfo):
        
        #Closed-class flags from lexicons.py (exact & accurate).
        
        
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

     
    MIN_NAME_CHARS = 3          

    @staticmethod
    def _looks_like_nisba(word: str) -> bool:
        """starts with ال and ends in ي — old startsWithAL(n) && last == ya2."""
        return word.startswith("ال") and word.endswith("ي") and len(word) > 3

    def _promote_name_candidates(self, tokens: list):
       #hay important
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
            #ya3ne mnshuf 4 cases la nkarrer iza hayda condidate rah na3mlo promotion
            if expected:
                token.is_name = True
                token.name_sources.append("camel-learned")



    def _apply_context_name_rule(self, tokens: list):
        #hon 3am nshuf iza name w hye ma 3enda sarf mn camel 
       
        for i, token in enumerate(tokens):
            if token.is_name or token.is_nmc or token.is_nrc \
               or token.is_punct or token.is_number or token.is_rasoul:
                continue #hol token ma3rouf shu henne , fa continue

            prev_token = tokens[i - 1] if i > 0 else None
            next_token = tokens[i + 1] if i + 1 < len(tokens) else None #hon dghre 3m bishuf jirano

            squeezed = (prev_token and prev_token.is_ibn
                        and next_token and next_token.is_ibn) # بن X بن 
            after_kunya = prev_token and prev_token.is_ab   #أبو جعفر
            before_ibn = next_token and next_token.is_ibn
            if squeezed or after_kunya or before_ibn:
                token.is_name = True
                token.name_sources.append("context")

    #  analyze
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
                word_indexes.append(idx)    #l word index li byje sarf li elon ma camel

        # 2) CAMeL lemma+POS 
        words = [tokens[i].word for i in word_indexes]
        if words:
            for idx, (lemma, pos, enc) in zip(word_indexes,
                                              self.camel.analyze_words(words)):
                tokens[idx].lemma = lemma
                tokens[idx].pos = pos
                tokens[idx].has_enclitic = enc
                if pos == "noun_prop":
                    if self.strict_names:
                        tokens[idx].is_name_candidate = True
                    else:
                        tokens[idx].is_name = True
                        tokens[idx].name_sources.append("camel")

        # waw-prefix detection (old hadith_stemmer.has_waw).
        
        #    'سهل بن زياد واسحاق بن محمد'
        
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
        #hon l kalimet l mrakabe

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
