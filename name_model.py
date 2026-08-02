"""
name_model.py
=============
IDEA (canonize, paper §3.2 "canonize(n)"):
Turn a narrator's raw name into a COMPARABLE structure, so two narrators
can be checked for "same person?".

A narrator name has three kinds of pieces:
    - name words:      محمد ، جعفر ، يعقوب
    - parenthood (بن): separates generations (son-of)
    - qualifiers:      nisba (الكوفي) ، kunya (ابو زرعه) ، family (عمه)

The key insight (paper): split the name into LEVELS at each بن —
    عماره بن القعقاع بن شبرمه
        level 0 = [عماره]        (the person)
        level 1 = [القعقاع]      (his father)
        level 2 = [شبرمه]        (his grandfather)
Two narrators are the same only if their levels agree.

Old source (verified): Narrator::preProcessForEquality()
(narrator_abstraction.h:265). Exact rules from that code:
    - a name word  -> append to the CURRENT level
    - بن / ابن     -> start a NEW level (j++)
    - possessive (nisba الكوفي) -> goes to the SEPARATE qualifiers list
    - family connector (عمه/جده) -> stays in the current level
    - anything else (ابو/ابي kunya OTHER) -> ignored for leveling

--------------------------------------------------------------------------
INPUT:  a Narrator (from Part 1's models.py) OR a plain name string.
OUTPUT: a CanonicalName with .levels (list of lists) and .qualifiers.
--------------------------------------------------------------------------
"""

from dataclasses import dataclass, field

from normalization import normalize_word
from models import ConnectorType
import lexicons


# alef normalization for the first letter (old code: alefs.contains -> alef)
_ALEFS = {"ا", "أ", "إ", "آ", "ى"}


def _norm_first_letter(word: str) -> str:
    """Normalize a leading alef variant to bare alef (old equality rule)."""
    if word and word[0] in _ALEFS:
        return "ا" + word[1:]
    return word


@dataclass
class CanonicalName:
    """A narrator name split for comparison."""
    levels: list = field(default_factory=list)   # list[list[str]] name words
    qualifiers: list = field(default_factory=list)  # nisba/family qualifiers
    is_rasoul: bool = False
    is_relative: bool = False
    raw: str = ""

    @property
    def num_levels(self):
        return len(self.levels)

    def __repr__(self):
        lv = " | ".join("+".join(l) for l in self.levels)
        q = (" {" + ",".join(self.qualifiers) + "}") if self.qualifiers else ""
        tag = " <rasoul>" if self.is_rasoul else (
              " <relative>" if self.is_relative else "")
        return f"CanonicalName([{lv}]{q}{tag})"


def canonize(narrator) -> CanonicalName:
    """
    Build a CanonicalName from a Part-1 Narrator object.

    Follows preProcessForEquality exactly:
        - start with one empty level
        - name word  -> current level
        - IBN (بن)   -> open a new level
        - possessive/nisba -> qualifiers
        - family (عمه) -> current level
    """
    result = CanonicalName(levels=[[]], raw=narrator.full_name)
    result.is_rasoul = narrator.is_rasoul
    result.is_relative = narrator.is_relative

    # our Part-1 Narrator stores .parts = [NamePrim | NameConnector...]
    for part in narrator.parts:
        text = normalize_word(part.text)
        if not text:
            continue

        # NamePrim has no 'kind' attribute; NameConnector does
        is_connector = hasattr(part, "kind")

        if not is_connector:
            # a name word -> current (last) level
            result.levels[-1].append(_norm_first_letter(text))
        else:
            if part.is_ibn:                     # بن / ابن -> new level
                result.levels.append([])
            elif part.kind == ConnectorType.POSSESSIVE or _is_possessive(text):
                result.qualifiers.append(text)  # nisba -> qualifiers
            elif _is_family(part, text):        # عمه/جده -> current level
                result.levels[-1].append(_norm_first_letter(text))
            # kunya (ابو/ابي) and others: ignored for leveling (old code)

    # drop empty trailing level (if name ended on a بن)
    result.levels = [lv for lv in result.levels if lv] or [[]]
    result.qualifiers.sort()                    # old: qSort(possessives)
    return result


def canonize_string(name: str) -> CanonicalName:
    """
    Convenience: canonize a plain string like 'محمد بن يعقوب الكليني'.
    Useful for testing and for biography names (Part 3).

    Note: nisba words (الكوفي...) are detected here and marked as
    possessive connectors so canonize() routes them to qualifiers,
    matching how the FSM would have tagged them in a real chain.
    """
    from models import Narrator, ConnectorType
    n = Narrator()
    pos = 0
    for word in name.split():
        w = normalize_word(word)
        if not w:
            continue
        if lexicons.is_ibn_word(w):
            n.add_connector(w, pos, pos + len(w), ConnectorType.IBN)
        elif lexicons.is_ab_word(w):
            n.add_connector(w, pos, pos + len(w), ConnectorType.AB)
        elif lexicons.is_om_word(w):
            n.add_connector(w, pos, pos + len(w), ConnectorType.OM)
        elif _is_possessive(w):
            n.add_connector(w, pos, pos + len(w), ConnectorType.POSSESSIVE)
        elif lexicons.is_nmc_word(w):
            n.add_connector(w, pos, pos + len(w), ConnectorType.FAMILY)
        else:
            n.add_name(w, pos, pos + len(w))
        pos += len(word) + 1
    return canonize(n)


# ---- qualifier detection (nisba: الكوفي / العراقي / المصري ...) -----------
def _is_possessive(text: str) -> bool:
    """
    A nisba/qualifier: starts with ال and ends with ي (adjective of origin),
    e.g. الكوفي، العراقي، المصري، الاشعري. Simple, extendable rule.
    """
    return text.startswith("ال") and text.endswith("ي") and len(text) > 3


def _is_family(part, text: str) -> bool:
    """family connector like عمه / جده / خاله (not بن, not kunya)."""
    return text in lexicons.FAMILY_WORDS


# ---------------------------------------------------------------------------
# Quick demo:  python name_model.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    examples = [
        "عماره بن القعقاع بن شبرمه",
        "محمد بن يعقوب الكليني",
        "محمد بن يعقوب",
        "علي بن ابراهيم بن هاشم",
        "احمد بن محمد بن عيسا الاشعري القمي",
        "ابو عبد الله",
    ]
    for name in examples:
        c = canonize_string(name)
        print(f"{name}")
        print(f"    -> {c}")
        print()
