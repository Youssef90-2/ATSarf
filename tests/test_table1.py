"""
test_table1.py — the merge-decision scorer (paper Table 1).

Checks the metric responds in the right DIRECTION to each kind of error, that
the two rows of Table 1 differ where they should, and that it reads a real
built graph.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from graph_agreement import (score_merges, score_table1,        # noqa: E402
                             labels_from_graph)
from models import Chain, Narrator, NarratorConnector, ConnectorType  # noqa: E402
from graph_build import GraphParams                             # noqa: E402
from narrator_graph import NarratorGraph                        # noqa: E402

_score = {"pass": 0, "fail": 0}


def check(label, got, want):
    if got == want:
        _score["pass"] += 1
        print(f"  PASS  {label}")
    else:
        _score["fail"] += 1
        print(f"  FAIL  {label}\n        got  {got}\n        want {want}")


GOLD = {"a": "P1", "b": "P1", "c": "P1", "d": "P2", "e": "P2", "f": "P3"}


def rp(pred, restrict=True):
    s = score_merges(GOLD, pred, restrict)
    return (round(s.recall, 4), round(s.precision, 4))


print("Table 1. merge-decision scoring")

check("identical partition, different label values -> perfect",
      rp({"a": 1, "b": 1, "c": 1, "d": 2, "e": 2, "f": 3}), (1.0, 1.0))

check("UNDER-merge costs recall, not precision",
      rp({"a": 1, "b": 1, "c": 9, "d": 2, "e": 2, "f": 3}), (0.6667, 1.0))

check("OVER-merge costs precision, not recall",
      rp({"a": 1, "b": 1, "c": 1, "d": 1, "e": 1, "f": 3}), (1.0, 0.5))

check("everything in one node: perfect recall, poor precision",
      rp({k: 1 for k in GOLD})[0], 1.0)

check("everything split apart: zero recall for merged people",
      rp({k: i for i, k in enumerate(GOLD)})[0], round(1 / 6, 4))

check("a singleton is trivially perfect (nothing to merge)",
      score_merges({"f": "P3"}, {"f": 99}).recall, 1.0)

print()
print("Table 1. the 'detected' vs 'all' rows")

partial = {"a": 1, "b": 1, "d": 2, "e": 2, "f": 3}     # 'c' never extracted
t1 = score_table1(GOLD, partial)
check("'detected' judges only what we produced", t1["detected"]["recall"], 1.0)
check("'all' penalises the narrator we never extracted",
      t1["all"]["recall"], 0.6667)
check("'detected' counts fewer occurrences",
      (t1["detected"]["occurrences"], t1["all"]["occurrences"]), (5, 6))

print()
print("Table 1. reading a real graph")


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


# the same person written two ways in two chains -> must be ONE node
chains = [chain(["محمد بن يعقوب", "علي بن ابراهيم"]),
          chain(["محمد بن يعقوب", "احمد بن محمد"])]
graph = NarratorGraph(GraphParams(equality_threshold=0.5))
graph.build(chains)
pred = labels_from_graph(graph.persons, graph.chain_nodes)

check("labels are keyed 'chain#position' (addressable by an annotator)",
      sorted(pred), ["0#0", "0#1", "1#0", "1#1"])
check("the repeated narrator got ONE label in both chains",
      pred["0#0"], pred["1#0"])

# gold agrees the two occurrences of محمد بن يعقوب are one person
gold_real = {"0#0": "kulayni", "1#0": "kulayni",
             "0#1": "ali", "1#1": "ahmad"}
check("scoring a correctly-built graph gives 1.0",
      rp_real := (round(score_merges(gold_real, pred).recall, 4),
                  round(score_merges(gold_real, pred).precision, 4)),
      (1.0, 1.0))

print()
print(f"{_score['pass']} passed, {_score['fail']} failed")
sys.exit(1 if _score["fail"] else 0)
