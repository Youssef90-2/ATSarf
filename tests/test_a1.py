"""
test_a1.py — Phase 3 / A1: biography-mode FSM.

Verifies each documented difference from hadith mode against the old
getNextState branches, plus the tolerance-budget mechanics A2 hooks into.
Synthetic token streams -> no CAMeL needed.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from engine import TokenInfo                              # noqa: E402
from fsm import HadithFSM, FsmParams                      # noqa: E402
from bio_fsm import BiographyFSM, BioParams               # noqa: E402

_score = {"pass": 0, "fail": 0}


def check(label, got, want):
    if got == want:
        _score["pass"] += 1
        print(f"  PASS  {label}")
    else:
        _score["fail"] += 1
        print(f"  FAIL  {label}\n        got  {got}\n        want {want}")


def stream(spec):
    """spec: list of (word, *flags) -> fresh tokens with real offsets."""
    toks, pos = [], 0
    for word, *flags in spec:
        t = TokenInfo(word=word, start=pos, end=pos + len(word))
        for f in flags:
            setattr(t, f, True)
        toks.append(t)
        pos += len(word) + 1
    return toks


NAME = ("is_name",)
NRC = ("is_nrc",)
IBN = ("is_nmc", "is_ibn")
RAS = ("is_rasoul",)
PUNCT = ("is_punct",)
NUM = ("is_number",)


def names_of(mentions):
    return [m.name for m in mentions]


def hadith_names(spec):
    f = HadithFSM(FsmParams(narr_min=1))
    return [n.full_name for c in f.run(stream(spec)) for n in c.chain.narrators]


print("A1. biography mode vs hadith mode")
print()

# ---- DIFFERENCE 5: a name may open a run anywhere (hadithCommon.cpp:639) ---
# no NRC anywhere: in hadith mode only a full stop may open a run, so a name
# sitting mid-sentence is ignored entirely.
spec = [("كلام",), ("كلام2",), ("محمد", *NAME), ("بن", *IBN), ("يحيا", *NAME)]
check("hadith: name mid-sentence does NOT open a run", hadith_names(spec), [])
check("biography: name mid-sentence DOES open a run",
      names_of(BiographyFSM(BioParams(narr_min=1)).run(stream(spec))),
      ["محمد بن يحيا"])

# ---- DIFFERENCE 1: numbers are not boundaries (hadithCommon.cpp:600) ------
spec = [("محمد", *NAME), ("بن", *IBN), ("يحيا", *NAME), ("4", *NUM),
        ("عن", *NRC), ("احمد", *NAME), ("بن", *IBN), ("خالد", *NAME)]
check("biography: a number does not end the run",
      names_of(BiographyFSM(BioParams(narr_min=1)).run(stream(spec))),
      ["محمد بن يحيا", "احمد بن خالد"])

# ---- DIFFERENCE 4: rasoul does not end the run (hadithCommon.cpp:1085) ----
spec = [("محمد", *NAME), ("بن", *IBN), ("يحيا", *NAME),
        ("عن", *NRC), ("النبي", *RAS), ("عن", *NRC),
        ("احمد", *NAME), ("بن", *IBN), ("خالد", *NAME)]
check("hadith: rasoul ENDS the sanad, later names are cut off",
      hadith_names(spec), ["محمد بن يحيا", "النبي"])
check("biography: rasoul does NOT end the run, later names survive",
      names_of(BiographyFSM(BioParams(narr_min=1)).run(stream(spec))),
      ["محمد بن يحيا", "النبي", "احمد بن خالد"])

# ---- DIFFERENCE 3: NMC overflow does not end the run (cpp:907-914) --------
spec = [("محمد", *NAME), ("بن", *IBN), ("يحيا", *NAME),
        ("حشو1",), ("حشو2",), ("حشو3",),
        ("عن", *NRC), ("احمد", *NAME), ("بن", *IBN), ("خالد", *NAME)]
check("biography: NMC overflow keeps the run alive",
      names_of(BiographyFSM(BioParams(narr_min=1)).run(stream(spec)))[-1],
      "احمد بن خالد")

# ---- paragraph delimiters (old additionalCheck, narratordetector.cpp:378) -
spec = [("356", *NUM), ("-", *PUNCT), ("ابراهيم", *NAME),
        ("357", *NUM), ("-", *PUNCT), ("محمد", *NAME)]
f = BiographyFSM(BioParams(narr_min=1))
f.run(stream(spec))
check("number+dash recorded as a paragraph delimiter",
      len(f.paragraph_delimiters), 2)

print()
print("A1/A2. the tolerance budget — the cross-document feedback loop")

spec = [("محمد", *NAME), ("بن", *IBN), ("يحيا", *NAME),
        ("عن", *NRC),
        ("علي", *NAME), ("بن", *IBN), ("حسن", *NAME),
        ("حشو1",), ("حشو2",), ("حشو3",), ("حشو4",),
        ("عن", *NRC), ("احمد", *NAME), ("بن", *IBN), ("خالد", *NAME)]

no_graph = BiographyFSM(BioParams(nrc_max=5, narr_min=1))
ng = names_of(no_graph.run(stream(spec)))
with_graph = BiographyFSM(BioParams(nrc_max=5, narr_min=1),
                          confirm=lambda n: "بن" in n.full_name.split())
wg = names_of(with_graph.run(stream(spec)))

check("NO-GRAPH: budget drains and the run dies mid-entry",
      no_graph.stats["runs_ended_by_budget"], 1)
check("WITH-GRAPH: confirmations renew the budget, run never dies",
      with_graph.stats["runs_ended_by_budget"], 0)

# THE point of the whole method. Both conditions FIND the same narrators —
# the flat mention list is identical. What the graph changes is whether they
# stay in ONE run, which is what A3 reads as one biography's span.
check("both conditions detect the same narrators", (ng == wg, ng), (True, ng))
check("NO-GRAPH fragments the entry into 2 runs", len(no_graph.candidates), 2)
check("WITH-GRAPH keeps the entry as 1 run", len(with_graph.candidates), 1)
check("one budget reset per confirmed narrator",
      with_graph.stats["budget_resets"], with_graph.stats["confirmed"])

f = BiographyFSM(BioParams(nrc_max=100, narr_min=1))
f.run(stream([("محمد", *NAME), ("بن", *IBN), ("يحيا", *NAME),
              ("و",), ("ثم",)]))
check("budget counts every word (5 words, no punctuation)", f.stats["words"], 5)

print()
print(f"{_score['pass']} passed, {_score['fail']} failed")
sys.exit(1 if _score["fail"] else 0)
