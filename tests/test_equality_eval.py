"""
test_equality_eval.py — paper §5, the narrator-equality metric comparison.

Checks that the evaluation reproduces the paper's argument on pairs whose
answer is not in dispute: edit distance is fooled by names that LOOK alike,
the structural metric is not.
"""
import json
import sys
import pathlib
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from gold import (EqualityGold, GoldError,                    # noqa: E402
                  load_equality_for_scoring)
from run_equality import (evaluate, predict_edit,             # noqa: E402
                          predict_structural, sweep)

_score = {"pass": 0, "fail": 0}
TMP = pathlib.Path(tempfile.gettempdir())


def check(label, got, want):
    if got == want:
        _score["pass"] += 1
        print(f"  PASS  {label}")
    else:
        _score["fail"] += 1
        print(f"  FAIL  {label}\n        got  {got}\n        want {want}")


def raises(label, fn, exc):
    try:
        fn()
    except exc:
        _score["pass"] += 1
        print(f"  PASS  {label}")
        return
    _score["fail"] += 1
    print(f"  FAIL  {label} (did not raise)")


# Pairs whose answer is uncontroversial. The "different father" cases are the
# paper's whole point: the names differ by one word, so edit distance sees
# them as near-identical while the structural metric hard-rejects.
PAIRS = [
    ("محمد بن يعقوب", "محمد بن يعقوب الكليني", True),    # nisba added
    ("علي بن ابراهيم", "علي بن ابراهيم بن هاشم", True),   # grandfather added
    ("عماره بن القعقاع", "عماره بن القعقاع بن شبرمه", True),
    ("محمد بن يعقوب", "محمد بن الحسن", False),            # different father
    ("احمد بن محمد", "احمد بن علي", False),               # different father
    ("خالد بن سعيد", "خالد بن سعد", False),               # one letter apart
]

print("§5. the two metrics on undisputed pairs")

edit = evaluate(PAIRS, lambda a, b: predict_edit(a, b, 0.75))
struct = evaluate(PAIRS, lambda a, b: predict_structural(a, b, 0.5, "off"))

check("structural rejects every different-father pair",
      struct["false_positives"], [])
check("edit distance is fooled by at least one of them",
      len(edit["false_positives"]) > 0, True)
check("so structural precision beats edit precision",
      struct["precision"] > edit["precision"], True)

for a, b, truth in PAIRS:
    if not truth:
        check(f"structural: '{a}' != '{b}'",
              predict_structural(a, b, 0.5, "off"), False)

print()
print("§5. the metric's own weakness is visible too")

# A shorter form of a longer name scores 2/3 and is accepted at 0.5 — this is
# the ambiguity that drives the paper's 63% Table-1 recall, and the evaluation
# should SHOW it rather than hide it.
check("a 2-level prefix matches a 3-level name at threshold 0.5",
      predict_structural("احمد بن محمد", "احمد بن محمد بن عيسا", 0.5, "off"),
      True)
check("...and is rejected at a stricter threshold",
      predict_structural("احمد بن محمد", "احمد بن محمد بن عيسا", 0.7, "off"),
      False)

print()
print("§5. threshold sweep")

rows = sweep(PAIRS)
check("sweep covers the configured thresholds", len(rows), 9)
recalls = [r for _t, r, _p, _f in rows]
check("recall is monotonically non-increasing as the threshold rises",
      all(a >= b for a, b in zip(recalls, recalls[1:])), True)

print()
print("§5. gold handling")

seed = EqualityGold(source="demo", pairs=[(a, b, None) for a, b, _ in PAIRS])
path = TMP / "demo.equal.seed.json"
seed.save(path)
check("a bootstrap seed carries no judgements", seed.stats()["labelled"], 0)
raises("an unlabelled seed is REFUSED for scoring",
       lambda: load_equality_for_scoring(path), GoldError)

labelled = EqualityGold(source="demo", reviewed=True, pairs=PAIRS)
labelled.save(path)
loaded = load_equality_for_scoring(path)
check("a reviewed, labelled file loads", loaded.stats()["labelled"], 6)
check("counts split correctly",
      (loaded.stats()["same_person"], loaded.stats()["different"]), (3, 3))

d = json.loads(path.read_text(encoding="utf-8"))
d["pairs"] = [{"a": p["a"], "b": p["b"], "equal": None} for p in d["pairs"]]
path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
raises("reviewed but with nothing labelled is also refused",
       lambda: load_equality_for_scoring(path), GoldError)

path.unlink(missing_ok=True)

print()
print(f"  edit       recall {edit['recall']}  precision {edit['precision']}")
print(f"  structural recall {struct['recall']}  precision {struct['precision']}")

print()
print(f"{_score['pass']} passed, {_score['fail']} failed")
sys.exit(1 if _score["fail"] else 0)
