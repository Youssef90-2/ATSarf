"""
test_a3.py — Phase 3 / A3: find_chosen_biography (graph-derived boundaries).

Builds a real narrator graph, runs the biography FSM over rijal-style prose
in both conditions, and checks that the biography SPAN comes from the graph.
No CAMeL needed.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from models import Chain, Narrator, NarratorConnector, ConnectorType  # noqa: E402
from graph_build import GraphParams                       # noqa: E402
from narrator_graph import NarratorGraph                  # noqa: E402
from narrator_matcher import GraphIndex                   # noqa: E402
from bio_fsm import BiographyFSM, BioParams               # noqa: E402
from biography_detector import (BiographyDetector,        # noqa: E402
                                BiographyParams)
from engine import TokenInfo                              # noqa: E402

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
        elif w in ("ابو", "ابي", "ابا"):
            n.add_connector(w, pos, pos + len(w), ConnectorType.AB)
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


def person(spec, waw=False):
    """
    'محمد بن يحيا' -> token spec fragment.
    waw=True marks the first token as carrying a واو conjunction, which is
    how engine.py emits 'ومحمد' (it strips the و and sets has_waw) and what
    separates parallel narrators.
    """
    out = []
    for i, w in enumerate(spec.split()):
        flags = list(IBN if w in ("بن", "ابن") else NAME)
        if i == 0 and waw:
            flags.append("has_waw")
        out.append((w, *flags))
    return out


# ---------------------------------------------------------------- the graph
# Subject A (حرب بن الحسين) has two teachers; subject B (سعد بن عمار) has two.
# The two entries share NO narrators, so the graph can separate them.
chains = [
    chain(["حرب بن الحسين", "ابراهيم الشيباني"]),
    chain(["حرب بن الحسين", "زياد بن مروان"]),
    chain(["سعد بن عمار", "خالد بن سعيد"]),
    chain(["سعد بن عمار", "عمرو بن جميع"]),
]
graph = NarratorGraph(GraphParams(equality_threshold=0.5))
graph.build(chains)
index = GraphIndex({"params": {"equality_threshold": 0.5},
                    "persons": [p.to_dict() for p in graph.persons.values()]})

# ------------------------------------------------------- two rijal entries
# entry 1: حرب بن الحسين + its two graph neighbours
# entry 2: سعد بن عمار  + its two graph neighbours
spec = (person("حرب بن الحسين") + [("روا", *NRC), ("عن", *NRC)]
        + person("ابراهيم الشيباني")
        + person("زياد بن مروان", waw=True) + [(".", *PUNCT)]
        + person("سعد بن عمار") + [("روا", *NRC), ("عن", *NRC)]
        + person("خالد بن سعيد")
        + person("عمرو بن جميع", waw=True) + [(".", *PUNCT)])

fsm = BiographyFSM(BioParams(narr_min=1), confirm=index.confirm)
mentions = fsm.run(stream(spec))

print("A3. find_chosen_biography")
print()
check("every mention matched a graph person",
      all(m.matches for m in mentions), True)

params = BiographyParams(near_max_chars=60, threshold=2, k=1)
detector = BiographyDetector(mentions, index, params)
found = detector.detect()

check("two entries detected", len(found), 2)
check("each entry confirmed 2 neighbours",
      sorted(b.score for b in found), [2, 2])
check("spans do not overlap",
      found[0].end < found[1].start, True)

# THE point: the span is derived from the graph, and covers exactly the
# subject plus its graph-confirmed neighbours.
first = found[0]
covered = [mentions[i].name for i in first.mention_indices]
check("entry 1 covers its subject and both its graph neighbours",
      sorted(covered),
      sorted(["حرب بن الحسين", "ابراهيم الشيباني", "زياد بن مروان"]))
check("entry 1 span starts at its subject",
      first.start, mentions[first.mention_indices[0]].start)

print()
for b in found:
    print(f"  [{b.start:>3}-{b.end:<3}] score={b.score}  {b.subject_name}")
    print(f"        confirmed: {b.confirmed_neighbours}")

print()
print("A3. the graph is what makes the boundary")

# Without the graph, mentions carry no matches -> no position index ->
# no neighbours can be confirmed -> no biography span can be derived.
fsm_ng = BiographyFSM(BioParams(narr_min=1))
mentions_ng = fsm_ng.run(stream(spec))
found_ng = BiographyDetector(mentions_ng, index, params).detect()
check("NO-GRAPH: no entry boundary can be derived", len(found_ng), 0)
check("WITH-GRAPH: boundaries derived", len(found) > 0, True)

print()
print("A3. guards")

# an ambiguous first name (key starts with '-') is not a biography subject
from name_hash import primary_key                          # noqa: E402
from name_model import canonize                            # noqa: E402
check("isFirstNameAmbiguous guard is live (B9 restored the leading '-')",
      primary_key(canonize(narrator("ابن ابي عمير"))).startswith("-"), True)

# overlap rejection
a = found[0]
clash = type(a)(person_id=-1, subject_name="x", start=a.start + 1,
                end=a.end - 1, score=99)
check("overlapping spans are detected", clash.overlaps(a), True)

print()
print("A7. sequential disambiguation (OUR improvement — absent from the old code)")

# Two DIFFERENT people share a 2-level prefix: 'خالد بن سعيد بن عمرو' is a
# neighbour of سعد بن عمار, 'خالد بن سعيد بن زيد' belongs to another entry.
# A rijal entry writing only 'خالد بن سعيد' matches BOTH (2 levels / 3 = 0.67),
# which is exactly the ambiguity paper §4 says to resolve with the graph.
chains2 = [
    chain(["سعد بن عمار", "خالد بن سعيد بن عمرو"]),
    chain(["طلحه بن عبيد", "خالد بن سعيد بن زيد"]),
]
graph2 = NarratorGraph(GraphParams(equality_threshold=0.5))
graph2.build(chains2)
index2 = GraphIndex({"params": {"equality_threshold": 0.5},
                     "persons": [p.to_dict() for p in graph2.persons.values()]})

amb = (person("سعد بن عمار") + [("روا", *NRC), ("عن", *NRC)]
       + person("خالد بن سعيد"))
fsm2 = BiographyFSM(BioParams(narr_min=1), confirm=index2.confirm)
ms2 = fsm2.run(stream(amb))
n_cands = len(ms2[-1].matches)
check("the bare name really is ambiguous (>1 graph person)", n_cands > 1, True)

off = BiographyDetector(ms2, index2, BiographyParams(
    near_max_chars=60, threshold=1, k=1, sequential_disambiguation=False))
on = BiographyDetector(ms2, index2, BiographyParams(
    near_max_chars=60, threshold=1, k=1, sequential_disambiguation=True))

check("OFF (faithful baseline): every candidate kept",
      len(off._resolved[-1]), n_cands)
check("ON: candidates narrowed to those reachable from the previous narrator",
      len(on._resolved[-1]) < n_cands, True)
check("ON: the surviving candidate is the graph neighbour",
      on._resolved[-1][0].person_id in on.reach.reachable_within(
          ms2[0].matches[0].person_id, k=1), True)
check("ON: narrowing was recorded", on.disambiguation_stats["narrowed"], 1)

print()
print(f"{_score['pass']} passed, {_score['fail']} failed")
sys.exit(1 if _score["fail"] else 0)
