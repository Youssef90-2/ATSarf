"""
test_a4.py — Phase 1 / A4: the two-level agreement scorer.

Layered: the interval/word primitives are checked against hand-counted
values, then the aggregation logic is checked against numbers derived from
those (already verified) primitives. No CAMeL, no gold file needed.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from spans import (overlaps, before, after, count_words,          # noqa: E402
                   count_words_spans, common_words)
from agreement import (score, score_one_level, common_names,      # noqa: E402
                       TwoLevelItem)

_score = {"pass": 0, "fail": 0}


def check(label, got, want):
    if got == want:
        _score["pass"] += 1
        print(f"  PASS  {label}")
    else:
        _score["fail"] += 1
        print(f"  FAIL  {label}\n        got  {got}\n        want {want}")


# ---------------------------------------------------------------- primitives
print("A4. interval primitives (text_handling.h:407-449)")
check("overlapping intervals", overlaps(0, 10, 5, 15), True)
check("touching at a point overlaps", overlaps(0, 10, 10, 20), True)
check("disjoint intervals", overlaps(0, 10, 11, 20), False)
check("before", before(0, 5, 10, 20), True)
check("after", after(10, 20, 0, 5), True)
check("before is not symmetric", before(10, 20, 0, 5), False)

TEXT = "محمد بن يحيا عن احمد بن محمد عن الحسن بن محبوب عن ابي جعفر"
#       0    5  8    13 16   21 24   29 32    38 41    47 50  54

print()
print("A4. word counting (text_handling.h:459)")
check("a 3-word name", count_words(TEXT, 0, 11), 3)
check("a 1-word name", count_words(TEXT, 0, 3), 1)
check("degenerate span counts 0 (start >= end, as in the original)",
      count_words(TEXT, 5, 5), 0)
check("span across the whole text", count_words(TEXT, 0, len(TEXT) - 1), 14)
check("common words = words in the intersection",
      common_words(TEXT, (0, 27), (16, 45)), count_words(TEXT, 16, 27))

# ------------------------------------------------------------ common_names
print()
print("A4. greedy one-to-one name matching (AbstractTwoLevelAgreement.h:86)")
gold2 = [(0, 11), (16, 27)]
one_pred_covering_both = [(0, 27)]
common, all_common = common_names(gold2, one_pred_covering_both)
check("one prediction can only claim ONE gold name (1:1)", common, 1)
check("...but both gold names DO overlap it (all_common)", all_common, 2)

common, all_common = common_names(gold2, [(0, 11), (16, 27)])
check("exact match: both claimed", (common, all_common), (2, 2))
check("no overlap at all", common_names(gold2, [(50, 57)]), (0, 0))

# --------------------------------------------------------------- the scorer
print()
print("A4. scoring")

WHOLE = (0, len(TEXT) - 1)
GOLD_NAMES = [(0, 11), (16, 27), (32, 45)]        # 3 names, 9 words

perfect = score([TwoLevelItem(WHOLE, GOLD_NAMES)],
                [TwoLevelItem(WHOLE, GOLD_NAMES)], TEXT)
check("perfect prediction: detection 1.0",
      (perfect["detection"]["recall"], perfect["detection"]["precision"]),
      (1.0, 1.0))
check("perfect prediction: boundary 1.0",
      (perfect["boundary_min"]["recall"], perfect["boundary_max"]["recall"]),
      (1.0, 1.0))
check("denominator is every gold word",
      perfect["boundary_min"]["units"], count_words_spans(TEXT, GOLD_NAMES))

missed = score([TwoLevelItem(WHOLE, GOLD_NAMES)],
               [TwoLevelItem(WHOLE, [])], TEXT)
check("nothing predicted: recall 0, denominator still full",
      (missed["detection"]["recall"],
       missed["boundary_min"]["units"]),
      (0.0, count_words_spans(TEXT, GOLD_NAMES)))

# the headline case: a truncated name still counts as DETECTED, and only the
# boundary rows reveal that its span is wrong.
truncated = score([TwoLevelItem(WHOLE, GOLD_NAMES)],
                  [TwoLevelItem(WHOLE, [(0, 11), (16, 20)])], TEXT)
check("truncation: detection sees 2 of 3 names",
      truncated["detection"]["recall"], round(2 / 3, 4))
check("truncation: boundary is much lower, exposing the bad span",
      truncated["boundary_min"]["recall"] < truncated["detection"]["recall"],
      True)
check("truncation: boundary recall = covered gold words / all gold words",
      truncated["boundary_min"]["recall"], round(4 / 9, 4))

# ------------------------------------------------- the two divergences fixed
print()
print("A4. corrections to the original")

check("unmatched TAIL gold names still reach the denominator",
      truncated["boundary_min"]["units"], 9)

# One gold name split by two predictions. This is precisely what the two
# boundary variants exist to distinguish, and is NOT double counting:
#   min  scores each (gold, predicted) PAIR alone, so the gold name takes
#        part in two pairs and contributes its words twice;
#   max  merges the mutually-overlapping run into one group, so it counts once.
dup = score([TwoLevelItem(WHOLE, [(0, 11)])],
            [TwoLevelItem(WHOLE, [(0, 5), (6, 11)])], TEXT)
check("max-boundaries counts a split gold name ONCE",
      dup["boundary_max"]["units"], count_words(TEXT, 0, 11))
check("min-boundaries counts it once PER PAIR (by definition)",
      dup["boundary_min"]["units"], 2 * count_words(TEXT, 0, 11))
check("...and max sees the split name as fully covered",
      dup["boundary_max"]["recall"], 1.0)

# ---------------------------------------------------------- min vs max
print()
print("A4. min-boundaries vs max-boundaries")
merged = score([TwoLevelItem(WHOLE, gold2)],
               [TwoLevelItem(WHOLE, one_pred_covering_both)], TEXT)
check("one prediction spanning two gold names: max recall is 1.0",
      merged["boundary_max"]["recall"], 1.0)
check("...but min precision is worse (each pair judged alone)",
      merged["boundary_min"]["precision"] < merged["boundary_max"]["precision"],
      True)

# --------------------------------------------------------------- one level
print()
print("A4. one-level scoring (old OneLevelAgreement, for entry spans)")
one = score_one_level([(0, 11), (16, 27)], [(0, 11)], TEXT)
check("one-level: half the spans found",
      one["detection"]["recall"], 0.5)

print()
print(f"{_score['pass']} passed, {_score['fail']} failed")
sys.exit(1 if _score["fail"] else 0)
