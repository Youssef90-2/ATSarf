"""
segmenter.py
============
IDEA: Segment a FULL BOOK into hadiths (paper §3.1 — "ANGE first segments
a hadith book into narrations, where each narration has a matn and a
sanad"). Port of the old segmentHelper() loop (hadithRefactored.cpp:26).

The FSM (fsm.py) finds individual chain candidates. This file runs the
whole pipeline over a complete book and turns candidates into accepted
Hadith objects:

    book text --normalize--> clean text --engine--> tokens
              --FSM--> chain candidates
              --[narr_min filter]--> accepted sanads
              --[matn linking]--> Hadith(number, sanad, matn, book, bab)

IDEAS INSIDE (source of each):
  1. narr_min filter (old system, hadithParameters.narr_min=3):
     chains with too few narrators are false positives — this alone
     rejects almost everything in the editor's muqaddima (front matter).
  2. Matn linking (old system: sanadEnd -> newHadithStart): the matn of
     hadith N spans from its sanad end to the start of hadith N+1.
  3. Hadith numbering "N - " (from the real kafi1 data): used both as a
     boundary hint (already inside the FSM: numbers = hard boundary) and
     to extract the hadith's actual number in the book.
  4. Book/chapter headers "(كتاب ...)" / "باب ..." (from the real data):
     recorded as metadata so every hadith knows its kitab/bab.
  All patterns are CONFIGURABLE (SegmenterConfig) — not welded in — so a
  differently-formatted book (musnad_ahmad...) needs config, not code.
"""

import re
from dataclasses import dataclass, field

from normalization import normalize
from engine import ArabicEngine
from fsm import HadithFSM, FsmParams
from models import Hadith


# ===========================================================================
# 1. Configuration — data-format specifics live HERE, not in logic
# ===========================================================================

@dataclass
class SegmenterConfig:
    # FSM parameters (old hadithParameters)
    fsm: FsmParams = field(default_factory=FsmParams)

    # hadith numbering pattern in the clean text, e.g. "12 - "
    # (kafi/istibsar style; adjust for other books if they differ)
    number_pattern: str = r"(\d+)\s*-"

    # book / chapter header patterns (searched in the clean text)
    kitab_pattern: str = r"كتاب\s+\S+(?:\s+\S+){0,3}"
    bab_pattern: str = r"باب\s+\S+(?:\s+\S+){0,4}"

    # maximum matn length in characters (a matn that runs until the next
    # sanad far away is suspicious; we cap it for sanity)
    matn_max_chars: int = 4000

    # optional: skip everything before this marker (regex) — used to jump
    # over the editor's muqaddima/front matter. Example for kafi:
    #     start_marker = r"كتاب العقل"
    # Empty string = process the whole file.
    start_marker: str = ""


# ===========================================================================
# 2. The segmenter
# ===========================================================================

class HadithSegmenter:

    def __init__(self, engine: ArabicEngine, config: SegmenterConfig = None):
        self.engine = engine
        self.config = config or SegmenterConfig()

    # ------------------------------------------------------------- helpers
    def _find_numbers(self, clean_text: str):
        """All 'N - ' markers: list of (position, number), sorted."""
        return [(m.start(), int(m.group(1)))
                for m in re.finditer(self.config.number_pattern, clean_text)]

    def _find_headers(self, clean_text: str, pattern: str):
        """All header positions: list of (position, header_text)."""
        return [(m.start(), m.group(0).strip())
                for m in re.finditer(pattern, clean_text)]

    @staticmethod
    def _last_before(items, position, default=""):
        """From (pos, value) list, the value of the last item before position."""
        value = default
        for pos, v in items:
            if pos > position:
                break
            value = v
        return value

    def _number_for(self, numbers, sanad_start):
        """
        The hadith number = the LAST 'N - ' marker right before the sanad
        (old system used numbers as newHadithStart hints the same way).
        Returns -1 if none found nearby (within 30 chars).
        """
        best = -1
        for pos, num in numbers:
            if pos > sanad_start:
                break
            if sanad_start - pos <= 30:
                best = num
        return best

    # ----------------------------------------------------------- main run
    def segment(self, book_text: str, hadith_mode: bool = True):
        """
        Full pipeline on a raw book string.
        Returns (hadiths, stats_dict, clean_text, index_map).
        """
        cfg = self.config

        # 1) normalize (keeps index_map back to the original text)
        clean_text, index_map = normalize(book_text)

        # 1b) skip front matter if a start marker is configured
        if cfg.start_marker:
            m = re.search(cfg.start_marker, clean_text)
            if m:
                clean_text = clean_text[m.start():]
                index_map = index_map[m.start():]

        # 2) analyze (cached: second run of the same book is instant)
        tokens = self.engine.analyze_cached(clean_text)

        # 3) FSM over the whole token stream
        fsm = HadithFSM(cfg.fsm)
        candidates = fsm.run(tokens, hadith_mode=hadith_mode)

        # 4) accept only chains with enough narrators (old narr_min check —
        #    this is what silently discards the muqaddima's false chains)
        accepted = [c for c in candidates
                    if c.narrator_count >= cfg.fsm.narr_min]

        # 5) metadata: numbering and headers
        numbers = self._find_numbers(clean_text)
        kitabs = self._find_headers(clean_text, cfg.kitab_pattern)
        babs = self._find_headers(clean_text, cfg.bab_pattern)

        # 6) build Hadith objects with matn spans
        hadiths = []
        for i, cand in enumerate(accepted):
            matn_start = cand.sanad_end
            # matn ends at the next accepted sanad's start (old:
            # newHadithStart), or at the next number marker — whichever
            # comes first; capped by matn_max_chars.
            next_sanad = (accepted[i + 1].sanad_start
                          if i + 1 < len(accepted) else len(clean_text))
            next_number = next((pos for pos, _ in numbers
                                if pos > matn_start), len(clean_text))
            matn_end = min(next_sanad, next_number,
                           matn_start + cfg.matn_max_chars)

            hadith = Hadith(
                number=self._number_for(numbers, cand.sanad_start),
                chain=cand.chain,
                matn_start=matn_start,
                matn_end=matn_end,
                matn_text=clean_text[matn_start:matn_end].strip())
            # attach location metadata (kitab/bab) dynamically
            hadith.kitab = self._last_before(kitabs, cand.sanad_start)
            hadith.bab = self._last_before(babs, cand.sanad_start)
            hadiths.append(hadith)

        # 7) stats (for your sanity checks on any new book)
        stats = {
            "text_chars": len(clean_text),
            "tokens": len(tokens),
            "chain_candidates": len(candidates),
            "accepted_hadiths": len(hadiths),
            "rejected_short_chains": len(candidates) - len(accepted),
            "numbered_markers_found": len(numbers),
            "total_narrators": sum(h.chain.narrator_count for h in hadiths),
        }
        return hadiths, stats, clean_text, index_map


# ---------------------------------------------------------------------------
# Quick demo:  python segmenter.py   (small built-in text, no file needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "1 - أخبرنا أبو جعفر محمد بن يعقوب قال: حدثني عدة من أصحابنا عن "
        "أحمد بن محمد عن الحسن بن محبوب عن أبي جعفر عليه السلام قال: لما "
        "خلق الله العقل استنطقه ثم قال له أقبل فأقبل. "
        "2 - علي بن إبراهيم عن أبيه عن حماد بن عيسى عن إبراهيم بن عمر عن "
        "أبي عبد الله عليه السلام قال: الحجة على الخلق هو العقل."
    )

    print("Loading engine...")
    engine = ArabicEngine(camel_mode="mle", use_wojood=False)
    segmenter = HadithSegmenter(engine)

    hadiths, stats, clean_text, _ = segmenter.segment(sample)

    print("\nstats:", stats, "\n")
    for h in hadiths:
        print(f"hadith #{h.number}  narrators={h.chain.narrator_count}")
        for n in h.chain.narrators:
            tag = "  (RASOUL)" if n.is_rasoul else ""
            print("   -", n.full_name, tag)
        print("   matn:", h.matn_text[:70], "...\n")
