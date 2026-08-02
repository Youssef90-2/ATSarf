"""
models.py
=========
IDEA: The data structures — "what does a hadith look like in memory?"

Direct port of the old system's narrator_abstraction.h class hierarchy,
simplified into flat Python dataclasses (no deep inheritance needed).

The paper (§1, Figure 1a) defines the structure:
    A hadith = sanad (chain of narrators) + matn (the content).
    A narrator's name = name pieces connected by name connectors:
        قتيبة بن سعيد  ->  NamePrim(قتيبة) + Connector(بن) + NamePrim(سعيد)

Old system mapping (narrator_abstraction.h):
    NamePrim              -> NamePrim         (+ learned_name flag, line 101)
    NameConnectorPrim     -> NameConnector    (+ Type enum, line 124-130)
    Narrator              -> Narrator         (+ is_rasoul flag, line 227)
    NarratorConnectorPrim -> NarratorConnector
    Chain                 -> Chain
    (new) Hadith          -> groups chain + matn + hadith number

--------------------------------------------------------------------------
KEY DESIGN (same philosophy as the old system):
    Structures store POSITIONS (start/end offsets into the text),
    not copies of the text. The old NamePrim kept m_start/m_end into
    the original QString. We do the same — text lives once, structures
    point into it. `index_map` from normalization.py converts clean-text
    positions back to original-text positions when needed.
--------------------------------------------------------------------------
"""

from dataclasses import dataclass, field
from enum import Enum


# ===========================================================================
# 1. Connector types — old NameConnectorPrim::Type enum (line 124)
#    {POS, IBN, AB, OM, FAMILY_OTHER, OTHER}
# ===========================================================================

class ConnectorType(Enum):
    IBN = "ibn"                  # بن / ابن       (son of)  -> parenthood!
    AB = "ab"                    # ابو / ابي      (father of / kunya start)
    OM = "om"                    # ام             (mother of)
    FAMILY = "family"            # عمه / جده / خاله ...
    POSSESSIVE = "possessive"    # nisba: الكوفي / العراقي (old POS type)
    OTHER = "other"


# ===========================================================================
# 2. Name-level pieces (inside ONE narrator's name)
# ===========================================================================

@dataclass
class NamePrim:
    """
    One piece of a name: جعفر / محمد / يعقوب ...
    Old: NamePrim (narrator_abstraction.h:99).
    """
    text: str                    # the word itself (from clean text)
    start: int                   # offset into clean text
    end: int
    learned_name: bool = False   # True if detected by context rule only
                                 # (old learnedName flag, line 101)

    def to_dict(self):
        return {"text": self.text, "start": self.start, "end": self.end,
                "learned": self.learned_name}


@dataclass
class NameConnector:
    """
    A connector INSIDE a name: بن / ابو / ام ...
    Old: NameConnectorPrim (narrator_abstraction.h:124).
    """
    text: str
    start: int
    end: int
    kind: ConnectorType = ConnectorType.OTHER

    # convenience checks (same API style as the old isIbn()/isAB()/isOM())
    @property
    def is_ibn(self): return self.kind == ConnectorType.IBN
    @property
    def is_ab(self): return self.kind == ConnectorType.AB
    @property
    def is_om(self): return self.kind == ConnectorType.OM

    def to_dict(self):
        return {"text": self.text, "start": self.start, "end": self.end,
                "kind": self.kind.value}


# ===========================================================================
# 3. Narrator — one person in the chain
# ===========================================================================

@dataclass
class Narrator:
    """
    One narrator = ordered list of name pieces and connectors.
    قتيبة بن سعيد -> parts = [NamePrim(قتيبة), NameConnector(بن), NamePrim(سعيد)]

    Old: Narrator (narrator_abstraction.h:216) with m_narrator list
    and the isRasoul flag (line 227).
    """
    parts: list = field(default_factory=list)   # NamePrim | NameConnector
    is_rasoul: bool = False       # this "narrator" is the Prophet/Imam
    is_relative: bool = False     # عن ابيه / وعنه — resolved later (graph)

    # ---- position helpers (old getStart/getEnd, lines 240-259) ----
    @property
    def start(self):
        return self.parts[0].start if self.parts else -1

    @property
    def end(self):
        return self.parts[-1].end if self.parts else -1

    @property
    def full_name(self) -> str:
        """The complete name as one string: 'قتيبه بن سعيد'."""
        return " ".join(p.text for p in self.parts)

    @property
    def name_pieces(self) -> list:
        """Only the NamePrim texts (no connectors): ['قتيبه','سعيد']."""
        return [p.text for p in self.parts if isinstance(p, NamePrim)]

    def add_name(self, text, start, end, learned=False):
        self.parts.append(NamePrim(text, start, end, learned))

    def add_connector(self, text, start, end, kind):
        self.parts.append(NameConnector(text, start, end, kind))

    def is_empty(self):
        return len(self.parts) == 0

    def to_dict(self):
        return {"name": self.full_name,
                "start": self.start, "end": self.end,
                "is_rasoul": self.is_rasoul,
                "is_relative": self.is_relative,
                "parts": [p.to_dict() for p in self.parts]}


@dataclass
class NarratorConnector:
    """
    The narration word BETWEEN two narrators: عن / حدثنا / قال ...
    Old: NarratorConnectorPrim (narrator_abstraction.h:182).
    """
    text: str
    start: int
    end: int

    def to_dict(self):
        return {"text": self.text, "start": self.start, "end": self.end}


# ===========================================================================
# 4. Chain — the sanad
# ===========================================================================

@dataclass
class Chain:
    """
    The full sanad: alternating narrators and connectors, in text order.
    Old: Chain (narrator_abstraction.h:302) with m_chain list.
    """
    items: list = field(default_factory=list)   # Narrator | NarratorConnector

    @property
    def narrators(self) -> list:
        return [x for x in self.items if isinstance(x, Narrator)]

    @property
    def narrator_count(self) -> int:
        return len(self.narrators)

    @property
    def start(self):
        return self.items[0].start if self.items else -1

    @property
    def end(self):
        return self.items[-1].end if self.items else -1

    def add(self, item):
        self.items.append(item)

    def to_dict(self):
        return {"narrator_count": self.narrator_count,
                "start": self.start, "end": self.end,
                "narrators": [n.to_dict() for n in self.narrators]}


# ===========================================================================
# 5. Hadith — the final unit: number + sanad + matn
#    (the paper's Figure 1a structure; the old system printed sanadEnd /
#     newHadithStart positions — we make it an explicit object)
# ===========================================================================

@dataclass
class Hadith:
    number: int                   # hadith number in the book ("1 - ...")
    chain: Chain                  # the sanad
    matn_start: int               # matn = clean_text[matn_start:matn_end]
    matn_end: int
    matn_text: str = ""           # filled by the segmenter for convenience

    def to_dict(self):
        return {"number": self.number,
                "sanad": self.chain.to_dict(),
                "matn": {"start": self.matn_start, "end": self.matn_end,
                         "text": self.matn_text}}


# ---------------------------------------------------------------------------
# Quick demo:  python models.py
# Builds the first kafi1 sanad by hand to show the structures.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    # "اخبرنا ابو جعفر محمد بن يعقوب قال ..."
    #  0......8...12....18....24.27......34
    chain = Chain()

    chain.add(NarratorConnector("اخبرنا", 0, 6))

    narrator1 = Narrator()
    narrator1.add_connector("ابو", 7, 10, ConnectorType.AB)
    narrator1.add_name("جعفر", 11, 15)
    narrator1.add_name("محمد", 16, 20)
    narrator1.add_connector("بن", 21, 23, ConnectorType.IBN)
    narrator1.add_name("يعقوب", 24, 29)
    chain.add(narrator1)

    chain.add(NarratorConnector("عن", 35, 37))

    narrator2 = Narrator()
    narrator2.add_connector("ابي", 38, 41, ConnectorType.AB)
    narrator2.add_name("جعفر", 42, 46)
    narrator2.is_rasoul = True        # the Imam — sanad ends here
    chain.add(narrator2)

    hadith = Hadith(number=1, chain=chain, matn_start=60, matn_end=95,
                    matn_text="لما خلق الله العقل ...")

    print("narrator 1 full name :", narrator1.full_name)
    print("narrator 1 pieces    :", narrator1.name_pieces)
    print("chain narrator count :", chain.narrator_count)
    print("chain span           :", chain.start, "->", chain.end)
    print("\nJSON export:")
    print(json.dumps(hadith.to_dict(), ensure_ascii=False, indent=2)[:600])
