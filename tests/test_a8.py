"""
test_a8.py — paper §4.1, partial narrator graph annotation.

Checks the thing the method actually claims: that restricting the neighbour
evidence to a SMALL topically-selected graph produces a sharp, ranked
shortlist of candidate biographies per narrator.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from models import Chain, Narrator, NarratorConnector, ConnectorType  # noqa: E402
from graph_build import GraphParams                        # noqa: E402
from bio_fsm import BioParams                              # noqa: E402
from biography_detector import BiographyParams             # noqa: E402
from partial_annotation import (build_partial_graph,       # noqa: E402
                                PartialAnnotator, select_hadiths,
                                score_annotations)
from engine import TokenInfo                               # noqa: E402

_score = {"pass": 0, "fail": 0}


def check(label, got, want):
    if got == want:
        _score["pass"] += 1
        print(f"  PASS  {label}")
    else:
        _score["fail"] += 1
        print(f"  FAIL  {label}\n        got  {got}\n        want {want}")


def narrator(name):
    n, pos = Narrator(), 0
    for w in name.split():
        if w in ("بن", "ابن"):
            n.add_connector(w, pos, pos + len(w), ConnectorType.IBN)
        else:
            n.add_name(w, pos, pos + len(w))
        pos += len(w) + 1
    return n


def chain(names):
    c = Chain()
    for i, nm in enumerate(names):
        if i:
            c.add(NarratorConnector("عن", 0, 0))
        c.add(narrator(nm))
    return c


NAME = ("is_name",); NRC = ("is_nrc",); IBN = ("is_nmc", "is_ibn")
PUNCT = ("is_punct",)


def stream(spec):
    toks, pos = [], 0
    for word, *flags in spec:
        t = TokenInfo(word=word, start=pos, end=pos + len(word))
        for f in flags:
            setattr(t, f, True)
        toks.append(t)
        pos += len(word) + 1
    return toks


def person_tokens(spec, waw=False):
    out = []
    for i, w in enumerate(spec.split()):
        flags = list(IBN if w in ("بن", "ابن") else NAME)
        if i == 0 and waw:
            flags.append("has_waw")
        out.append((w, *flags))
    return out


print("A8. selecting the hadiths of interest")

hadiths = [{"number": n, "matn": {"text": "باب العقل" if n % 2 else "باب الايمان"}}
           for n in range(1, 21)]
check("select by explicit numbers",
      [h["number"] for h in select_hadiths(hadiths, numbers=[3, 7, 11])],
      [3, 7, 11])
check("select by matn content, capped at the paper's 10",
      len(select_hadiths(hadiths, contains="العقل")), 10)

print()
print("A8. building Gp and annotating")

# Gp: two narrators, each with its own distinct pair of neighbours.
gp_chains = [
    chain(["حرب بن الحسين", "ابراهيم الشيباني"]),
    chain(["حرب بن الحسين", "زياد بن مروان"]),
    chain(["سعد بن عمار", "خالد بن سعيد"]),
    chain(["سعد بن عمار", "عمرو بن جميع"]),
]
gp = build_partial_graph(gp_chains, GraphParams(equality_threshold=0.5))

# a rijal book holding an entry for each, plus an unrelated entry
bio = (person_tokens("حرب بن الحسين") + [("روا", *NRC), ("عن", *NRC)]
       + person_tokens("ابراهيم الشيباني")
       + person_tokens("زياد بن مروان", waw=True) + [(".", *PUNCT)]
       + person_tokens("طلحه بن عبيد") + [("روا", *NRC), ("عن", *NRC)]
       + person_tokens("قيس بن سعد") + [(".", *PUNCT)]
       + person_tokens("سعد بن عمار") + [("روا", *NRC), ("عن", *NRC)]
       + person_tokens("خالد بن سعيد")
       + person_tokens("عمرو بن جميع", waw=True) + [(".", *PUNCT)])

annotator = PartialAnnotator(
    gp, BioParams(narr_min=1),
    BiographyParams(near_max_chars=60, threshold=2, k=1, max_candidates=3))
annotations = annotator.annotate(stream(bio))
stats = annotator.stats(annotations)

by_name = {a.name: a for a in annotations}
check("every node of Gp is reported on", len(annotations), len(gp.persons))
check("the two subjects were annotated",
      (by_name["حرب بن الحسين"].annotated, by_name["سعد بن عمار"].annotated),
      (True, True))
check("candidates are RANKED, not a single answer (paper §6)",
      all(len(a.candidates) <= 3 for a in annotations), True)
check("each annotation names its Gp neighbours",
      sorted(by_name["حرب بن الحسين"].neighbours_in_gp),
      sorted(["ابراهيم الشيباني", "زياد بن مروان"]))

a = by_name["حرب بن الحسين"].best
check("the best candidate is scored by confirmed Gp neighbours", a.score, 2)
check("its span covers the entry, not the whole book",
      a.end - a.start < 60, True)

# the unrelated entry must NOT be picked for either subject
for subject in ("حرب بن الحسين", "سعد بن عمار"):
    best = by_name[subject].best
    check(f"'{subject}' is not annotated with the unrelated entry",
          "طلحه بن عبيد" in (best.confirmed_neighbours if best else []), False)

print()
print("A8. why Gp and not the full graph")

# Add many unrelated chains: in a FULL graph حرب would have many neighbours,
# and co-occurrence would be weak evidence. Gp keeps the evidence sharp.
noise = [chain(["حرب بن الحسين", f"راو {i}"]) for i in range(20)]
full = build_partial_graph(gp_chains + noise, GraphParams(equality_threshold=0.5))
full_person = next(p for p in full.persons.values()
                   if p.primary_name == "حرب بن الحسين")
gp_person = next(p for p in gp.persons.values()
                 if p.primary_name == "حرب بن الحسين")
check("the same narrator has far more neighbours in the full graph",
      (len(gp_person.parents | gp_person.children)
       < len(full_person.parents | full_person.children)), True)

print()
print("A8. scoring against gold")

gold = {by_name["حرب بن الحسين"].person_id:
        (by_name["حرب بن الحسين"].best.start,
         by_name["حرب بن الحسين"].best.end)}
s1 = score_annotations(annotations, gold, at_k=1)
check("a correct top-1 annotation scores recall 1.0", s1["recall"], 1.0)

wrong = {by_name["حرب بن الحسين"].person_id: (9000, 9100)}
s_bad = score_annotations(annotations, wrong, at_k=1)
check("a wrong gold span scores 0", s_bad["recall"], 0.0)
check("at_k widens the shortlist, never narrows it",
      score_annotations(annotations, gold, at_k=3)["recall"] >= s1["recall"],
      True)

print()
for a in annotations:
    if a.annotated:
        print(f"  {a.name:<20} -> {len(a.candidates)} candidate(s), "
              f"best score {a.best.score} @ [{a.best.start}-{a.best.end}]")
print()
print("  stats:", stats)

print()
print(f"{_score['pass']} passed, {_score['fail']} failed")
sys.exit(1 if _score["fail"] else 0)
