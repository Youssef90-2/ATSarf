"""
gold.py
=======
The gold-annotation store: one canonical format that everything scores
against, plus importers for the original Qt binaries.

WHY A CANONICAL FORMAT AT ALL. Gold can arrive three ways — imported from
Dr Fadi's `.tags`/`.names`/`.narr` files, produced by our annotation tool, or
bootstrapped from system output and then corrected by hand. The scorer must
not care which. It sees `GoldSet` and nothing else.

--------------------------------------------------------------------------
THE SAFEGUARD, and why it exists.

The old system, when no gold file was present, wrote the CURRENT SYSTEM
OUTPUT as a seed file, printed "Correct it before use", and RETURNED FALSE so
that no statistics were produced from it (AbstractTwoLevelAgreement.cpp:178).
Both halves matter. This project's `boundary_gold.py` is that same pattern
with both halves removed: it generates gold from the segmenter and scores
against it in the same breath, so gold == prediction and every number it
reports is meaningless.

So a bootstrapped GoldSet is written with `reviewed: false`, and
`load_for_scoring()` REFUSES it. You cannot accidentally score against
un-reviewed output; you have to open the file and set the flag, which is the
moment you actually look at the annotations.

--------------------------------------------------------------------------
THE TEXT FINGERPRINT. Offsets are meaningless against a different text. Every
GoldSet stores the sha256 of the clean text it was made against, and loading
it for a different text is an error, not a warning. This catches the failure
mode where a normalization change silently shifts every span by a few
characters and the numbers merely get quietly worse.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from agreement import TwoLevelItem


FORMAT_VERSION = 1


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ===========================================================================
# 1. The data model
# ===========================================================================

@dataclass
class GoldItem:
    """
    One annotated unit: a hadith, or a rijal entry.

    main    (start, end) of the unit itself      -> segmentation + boundary
    names   narrator name spans inside it        -> detection + boundary
    persons optional person label per name, same length as `names`.
            Table 1 scores MERGE DECISIONS, not spans, so it needs to know
            which occurrences are the same human being — a label, not an
            offset. `None` for an unlabelled name.
    """
    main: tuple
    names: list = field(default_factory=list)
    persons: list = field(default_factory=list)
    note: str = ""

    def to_item(self) -> TwoLevelItem:
        return TwoLevelItem(main=tuple(self.main),
                            names=[tuple(s) for s in self.names])

    def to_dict(self):
        d = {"main": list(self.main), "names": [list(s) for s in self.names]}
        if any(p is not None for p in self.persons):
            d["persons"] = list(self.persons)
        if self.note:
            d["note"] = self.note
        return d

    @staticmethod
    def from_dict(d):
        names = [tuple(s) for s in d.get("names", [])]
        persons = list(d.get("persons", [])) or [None] * len(names)
        return GoldItem(main=tuple(d["main"]), names=names,
                        persons=persons, note=d.get("note", ""))


@dataclass
class GoldSet:
    kind: str                       # "hadith" | "biography"
    source: str                     # which book this annotates
    text_sha256: str
    items: list = field(default_factory=list)
    reviewed: bool = False          # THE safeguard — see the module docstring
    annotator: str = ""
    format_version: int = FORMAT_VERSION

    # ---------------------------------------------------------------- io
    def to_dict(self):
        return {"format_version": self.format_version, "kind": self.kind,
                "source": self.source, "text_sha256": self.text_sha256,
                "reviewed": self.reviewed, "annotator": self.annotator,
                "items": [i.to_dict() for i in self.items]}

    def save(self, path):
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=1),
            encoding="utf-8")
        return path

    @staticmethod
    def load(path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return GoldSet(kind=d.get("kind", "hadith"),
                       source=d.get("source", ""),
                       text_sha256=d.get("text_sha256", ""),
                       items=[GoldItem.from_dict(x) for x in d.get("items", [])],
                       reviewed=bool(d.get("reviewed")),
                       annotator=d.get("annotator", ""),
                       format_version=d.get("format_version", 0))

    # ------------------------------------------------------------ scoring
    def to_items(self):
        return [i.to_item() for i in self.items]

    def stats(self):
        names = sum(len(i.names) for i in self.items)
        labelled = sum(1 for i in self.items
                       for p in i.persons if p is not None)
        return {"kind": self.kind, "source": self.source,
                "reviewed": self.reviewed, "annotator": self.annotator,
                "units": len(self.items), "names": names,
                "person_labels": labelled,
                "avg_names_per_unit": (round(names / len(self.items), 2)
                                       if self.items else 0)}


class GoldError(Exception):
    pass


def load_for_scoring(path, text):
    """
    Load a GoldSet and refuse it if it cannot legitimately be scored against.

    Two refusals, both deliberate:
      * un-reviewed  — it is still raw system output; scoring against it
                       compares the system to itself
      * wrong text   — the offsets index a different string
    """
    gold = GoldSet.load(path)
    if not gold.reviewed:
        raise GoldError(
            f"{Path(path).name} is still a BOOTSTRAP seed (reviewed=false). "
            "It holds system output, not gold — scoring against it would "
            "compare the system to itself. Correct the spans, then set "
            '"reviewed": true.')
    actual = text_fingerprint(text)
    if gold.text_sha256 and gold.text_sha256 != actual:
        raise GoldError(
            f"{Path(path).name} was annotated against a different text "
            f"(gold {gold.text_sha256[:12]}..., given {actual[:12]}...). "
            "Its character offsets do not apply here.")
    return gold


# ===========================================================================
# 2. Bootstrap — seed a gold file from system output
# ===========================================================================

def bootstrap_from_hadiths(hadiths, text, source="", annotator=""):
    """
    Seed hadith gold from run_hadith.py output (a list of hadith dicts).
    Written with reviewed=false; a human then fixes the spans.
    """
    items = []
    for h in hadiths:
        sanad = h.get("sanad", {})
        narrators = sanad.get("narrators", [])
        if not narrators:
            continue
        start = sanad.get("start", narrators[0].get("start", 0))
        end = h.get("matn", {}).get("end", sanad.get("end", 0))
        items.append(GoldItem(
            main=(start, end),
            names=[(n["start"], n["end"]) for n in narrators
                   if n.get("start") is not None],
            persons=[None] * len(narrators)))
    return GoldSet(kind="hadith", source=source, text_sha256=text_fingerprint(text),
                   items=items, reviewed=False, annotator=annotator)


def bootstrap_from_biographies(detected, mentions, text, source="",
                               annotator=""):
    """
    Seed biography gold from BiographyDetector output. `detected` are the
    DetectedBiography spans; `mentions` the flat narrator list they index.
    """
    items = []
    for bio in detected:
        items.append(GoldItem(
            main=(bio.start, bio.end),
            names=[(mentions[i].start, mentions[i].end)
                   for i in bio.mention_indices],
            persons=[None] * len(bio.mention_indices),
            note=bio.subject_name))
    return GoldSet(kind="biography", source=source,
                   text_sha256=text_fingerprint(text), items=items,
                   reviewed=False, annotator=annotator)


# ===========================================================================
# 3. Importers for the original Qt files
# ===========================================================================

def import_qt_two_level(path, text, source="", annotator="ATSarf-original"):
    """Hadith gold from an original two-level `.tags`."""
    from qdatastream import read_two_level
    items = [GoldItem(main=tuple(x["main"]),
                      names=[tuple(s) for s in x["names"]],
                      persons=[None] * len(x["names"]))
             for x in read_two_level(path)]
    return GoldSet(kind="hadith", source=source or Path(path).stem,
                   text_sha256=text_fingerprint(text), items=items,
                   reviewed=True, annotator=annotator)


def import_qt_spans(path, text, kind="biography", source="",
                    annotator="ATSarf-original"):
    """
    Gold from a flat `.tags` / `.names` / `.narr`. These carry spans with no
    enclosing unit, so each becomes its own single-name item — which is what
    the one-level scorer expects.
    """
    from qdatastream import read_span_list
    items = [GoldItem(main=tuple(s), names=[tuple(s)], persons=[None])
             for s in read_span_list(path)]
    return GoldSet(kind=kind, source=source or Path(path).stem,
                   text_sha256=text_fingerprint(text), items=items,
                   reviewed=True, annotator=annotator)


# ===========================================================================
# 4. Validation — catch a broken annotation before it silently skews a table
# ===========================================================================

def validate(gold: GoldSet, text: str):
    """Returns a list of human-readable problems; empty means clean."""
    problems = []
    n = len(text)
    for idx, item in enumerate(gold.items):
        s, e = item.main
        if not (0 <= s <= e < n):
            problems.append(f"item {idx}: main span ({s},{e}) outside text")
        for j, (ns, ne) in enumerate(item.names):
            if not (0 <= ns <= ne < n):
                problems.append(f"item {idx} name {j}: span ({ns},{ne}) "
                                "outside text")
            elif not (s <= ns and ne <= e):
                problems.append(f"item {idx} name {j}: '{text[ns:ne+1][:20]}' "
                                "falls outside its own unit")
        if item.persons and len(item.persons) != len(item.names):
            problems.append(f"item {idx}: {len(item.persons)} person labels "
                            f"for {len(item.names)} names")
        ordered = sorted(item.names)
        for a, b in zip(ordered, ordered[1:]):
            if a[1] >= b[0]:
                problems.append(f"item {idx}: name spans {a} and {b} overlap")
    spans = sorted((i.main for i in gold.items))
    for a, b in zip(spans, spans[1:]):
        if a[1] >= b[0]:
            problems.append(f"units {a} and {b} overlap")
    return problems


# ---------------------------------------------------------------------------
# Quick demo:  python gold.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import tempfile
    sys.stdout.reconfigure(encoding="utf-8")

    TEXT = "محمد بن يحيا عن احمد بن محمد عن الحسن بن محبوب عن ابي جعفر"
    seed = GoldSet(kind="hadith", source="demo",
                   text_sha256=text_fingerprint(TEXT),
                   items=[GoldItem(main=(0, len(TEXT) - 1),
                                   names=[(0, 11), (16, 27), (32, 45)],
                                   persons=[None, None, None])])
    path = Path(tempfile.gettempdir()) / "demo.gold.json"
    seed.save(path)
    print("seed written:", seed.stats())

    print()
    try:
        load_for_scoring(path, TEXT)
    except GoldError as e:
        print("REFUSED (as designed):")
        print("   ", str(e)[:96], "...")

    d = json.loads(path.read_text(encoding="utf-8"))
    d["reviewed"] = True
    path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    ok = load_for_scoring(path, TEXT)
    print()
    print("after review:", ok.stats())

    print()
    try:
        load_for_scoring(path, TEXT + " زيادة")
    except GoldError as e:
        print("wrong text REFUSED:")
        print("   ", str(e)[:96], "...")

    print()
    print("validation on a deliberately broken set:")
    broken = GoldSet(kind="hadith", source="x", text_sha256=text_fingerprint(TEXT),
                     items=[GoldItem(main=(0, 20), names=[(0, 11), (5, 15)]),
                            GoldItem(main=(10, 40), names=[(900, 950)])])
    for p in validate(broken, TEXT):
        print("   -", p)
    path.unlink(missing_ok=True)
