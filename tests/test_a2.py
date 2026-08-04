"""
test_a2.py — Phase 3 / A2: the graph lookup wired into the biography FSM.

End-to-end with a REAL (small) narrator graph: build it from synthetic
chains, save/load it as JSON, then run the biography FSM over rijal-style
prose in both conditions and check the feedback loop fires on real matches.
No CAMeL needed.
"""
import json
import sys
import pathlib
import tempfile
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from models import Chain, Narrator, NarratorConnector, ConnectorType  # noqa: E402
from graph_build import GraphParams                     # noqa: E402
from narrator_graph import NarratorGraph                # noqa: E402
from narrator_matcher import GraphIndex                 # noqa: E402
from bio_fsm import BiographyFSM, BioParams             # noqa: E402
from engine import TokenInfo                            # noqa: E402

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


# ---------------------------------------------------------------- the graph
chains = [
    chain(["محمد بن يحيا", "احمد بن محمد بن عيسا", "ابي عبد الله"]),
    chain(["محمد بن يحيا", "علي بن ابراهيم", "ابي عبد الله"]),
    chain(["حرب بن الحسين", "ابراهيم الشيباني", "ابي الجارود"]),
]
graph = NarratorGraph(GraphParams(equality_threshold=0.5))
graph.build(chains)

tmp = Path(tempfile.gettempdir()) / "a2_demo_graph.json"
graph.save(tmp)
index = GraphIndex.load(tmp)

print("A2. graph lookup (isRealNarrator)")
print()
check("graph round-trips through JSON with exact canonical forms",
      (index.stats()["persons_reparsed_from_string"],
       index.stats()["persons_exact_canonical"] > 0),
      (0, True))
check("threshold read from the graph's own params, not a local default",
      index.threshold, 0.5)

check("a narrator in the graph is REAL",
      index.is_real_narrator(narrator("محمد بن يحيا")), True)
check("a narrator absent from the graph is NOT real",
      index.is_real_narrator(narrator("زياد بن مروان")), False)
check("a shorter form still matches (احمد بن محمد -> ...بن عيسا)",
      index.is_real_narrator(narrator("احمد بن محمد")), True)
check("a conflicting father does NOT match (...بن خالد vs ...بن عيسا)",
      index.is_real_narrator(narrator("احمد بن محمد بن خالد")), False)

m = index.match(narrator("محمد بن يحيا"))
check("matches are graded and sorted best-first",
      (len(m) >= 1, m[0].similarity == max(x.similarity for x in m)),
      (True, True))

print()
print("A2. the feedback loop on a real graph")

NAME = ("is_name",); NRC = ("is_nrc",); IBN = ("is_nmc", "is_ibn")
AB = ("is_nmc", "is_ab")


def stream(spec):
    toks, pos = [], 0
    for word, *flags in spec:
        t = TokenInfo(word=word, start=pos, end=pos + len(word))
        for f in flags:
            setattr(t, f, True)
        toks.append(t)
        pos += len(word) + 1
    return toks


# A rijal entry: a subject IN the graph, filler prose, then another narrator
# in the graph. With a low budget, only graph confirmations keep them together.
spec = [("حرب", *NAME), ("بن", *IBN), ("الحسين", *NAME),
        ("روا",), ("عن",) ,
        ("محمد", *NAME), ("بن", *IBN), ("يحيا", *NAME),
        ("حشو1",), ("حشو2",), ("حشو3",), ("حشو4",),
        ("عن", *NRC), ("علي", *NAME), ("بن", *IBN), ("ابراهيم", *NAME)]

no_graph = BiographyFSM(BioParams(nrc_max=6, narr_min=1))
no_graph.run(stream(spec))
with_graph = BiographyFSM(BioParams(nrc_max=6, narr_min=1),
                          confirm=index.confirm)
mentions = with_graph.run(stream(spec))

check("NO-GRAPH: budget drains, entry fragments",
      (no_graph.stats["confirmed"], len(no_graph.candidates) > 1), (0, True))
check("WITH-GRAPH: real graph hits renew the budget",
      with_graph.stats["confirmed"] > 0, True)
check("WITH-GRAPH keeps the entry as one run",
      len(with_graph.candidates), 1)
check("confirmed mentions carry their graded matches",
      all(m.matches for m in mentions if m.confirmed), True)

print()
for m in mentions:
    tag = ("  <-- " + ", ".join(f"#{x.person_id}({x.similarity:.2f})"
                                for x in m.matches[:2])) if m.confirmed else ""
    print(f"  [{m.start:>3}-{m.end:<3}] {m.name}{tag}")

tmp.unlink(missing_ok=True)
print()
print(f"{_score['pass']} passed, {_score['fail']} failed")
sys.exit(1 if _score["fail"] else 0)
