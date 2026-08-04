"""
agreement.py
============
The paper's evaluation metric — the two-level agreement scorer.
Port of AbstractTwoLevelAgreement::calculateStatisticsHelper
(AbstractTwoLevelAgreement.cpp:195) and OneLevelAgreement.

THE TWO LEVELS
    outer  a "main" span      — a hadith, or a biography entry
    inner  the NAMES inside it — the narrators

and the three numbers the paper reports:

    SEGMENTATION  did we find the main spans at all
    DETECTION     did we find the right narrator names inside them
                  (a predicted name counts as correct if it OVERLAPS a gold
                   name — OneLevelAgreement::equalNames is literally
                   `overLaps`, no string comparison at all), matched greedily
                   one-to-one
    BOUNDARY      at WORD granularity: of all the words belonging to gold
                  narrator names, how many fell inside a predicted name span

"Boundary recall 0.41" therefore means 41% of gold narrator WORDS were
covered — not a set-membership ratio. Reported twice, as the old code does:
    min-boundaries  per matched (gold, predicted) PAIR
    max-boundaries  per merged overlap GROUP

--------------------------------------------------------------------------
THE FORMULAS look redundant and must be kept anyway (cpp:36-37, 60-61):

    recall    = (common/correct) * correct        # == common
    precision = (common/detected) * correct
    denominator += correct

The `* correct` is a WEIGHT. It cancels in recall, giving a true micro-average
(sum of commons / sum of corrects), and does not cancel in precision, giving a
recall-weighted macro-average. Simplifying either one changes the numbers.

--------------------------------------------------------------------------
TWO DELIBERATE DIVERGENCES, both of which make the numbers mean what the
paper says they mean. The metric DEFINITIONS are unchanged.

 1. No double counting. The original runs the whole word-level sweep inside
    BOTH of its passes — once per overlapping (gold, predicted) pair and
    again per merged group (cpp:249 and cpp:263 both call
    overLapMainFinished, which appends to the min AND max lists every time).
    The author's own comment at cpp:30 concedes the max-boundary path is
    unreliable ("not working correctly on maximal boundaries"). We compute
    each quantity exactly once.

 2. Unmatched tails count. Both of the original's sweeps are
    `while (k < gold && h < predicted)`, and the post-loop flush only empties
    what was already collected. So gold names past the last predicted one
    never reach the denominator — words nobody predicted are not counted as
    missed, which inflates recall. We add them with zero overlap.

One asymmetry is INHERENT to the paper's formula and is left alone: every
contribution is weighted by `correct`, so a spurious prediction overlapping
no gold name has weight zero and cannot move precision. Precision here means
"of the gold words we claimed, how many were right".
--------------------------------------------------------------------------
USAGE
    from agreement import score, TwoLevelItem
    gold = [TwoLevelItem((0, 120), [(0, 12), (17, 30)]), ...]
    pred = [TwoLevelItem((0, 118), [(0, 12), (17, 28)]), ...]
    result = score(gold, pred, clean_text)
"""

from dataclasses import dataclass, field

from spans import (overlaps, before, after, count_words_spans,
                   common_words_spans)


# ===========================================================================
# 1. Input shape
# ===========================================================================

@dataclass
class TwoLevelItem:
    """One annotated unit: a main span plus the name spans inside it."""
    main: tuple                              # (start, end)
    names: list = field(default_factory=list)   # [(start, end), ...]

    @property
    def start(self):
        return self.main[0]

    @property
    def end(self):
        return self.main[1]


# ===========================================================================
# 2. Greedy one-to-one name matching (AbstractTwoLevelAgreement.h:86)
# ===========================================================================

def common_names(list1, list2, equal=None):
    """
    Returns (common, all_common).

      common      one-to-one matches: each gold name may claim at most one
                  predicted name and vice versa (the visitedTags sets)
      all_common  how many gold names have AT LEAST ONE overlapping
                  predicted name, with no exclusivity

    `equal` defaults to plain interval overlap, which is exactly what
    OneLevelAgreement::equalNames does (OneLevelAgreement.cpp:42).
    """
    if equal is None:
        def equal(a, b):
            return overlaps(a[0], a[1], b[0], b[1])

    used2 = set()
    common = 0
    all_common = 0

    for i, a in enumerate(list1):
        found = False
        counted_all = False
        for j, b in enumerate(list2):
            if not equal(a, b):
                continue
            if not counted_all:
                counted_all = True
                all_common += 1
            if j not in used2 and not found:
                found = True
                used2.add(j)
        if found:
            common += 1

    return common, all_common


# ===========================================================================
# 3. The weighted accumulator (cpp:36-37, 60-61)
# ===========================================================================

class _Weighted:
    """
    Accumulates (common, correct, detected) triples and produces
        recall    = sum(common) / sum(correct)
        precision = sum(common/detected * correct) / sum(correct)
    """

    def __init__(self):
        self.recall_num = 0.0
        self.precision_num = 0.0
        self.denominator = 0

    def add(self, common, correct, detected):
        if correct <= 0:
            return
        self.recall_num += common
        self.precision_num += (common / detected * correct) if detected else 0.0
        self.denominator += correct

    def result(self):
        if not self.denominator:
            return {"recall": 0.0, "precision": 0.0, "f_score": 0.0,
                    "units": 0}
        recall = self.recall_num / self.denominator
        precision = self.precision_num / self.denominator
        f = (2 * recall * precision / (recall + precision)
             if (recall + precision) else 0.0)
        return {"recall": round(recall, 4), "precision": round(precision, 4),
                "f_score": round(f, 4), "units": self.denominator}


# ===========================================================================
# 4. The word-level sweep (overLapMainFinished's inner loop, cpp:71-149)
# ===========================================================================

def _word_level(text, gold_names, pred_names, min_acc, max_acc):
    """
    Walk the two sorted name lists together. Every overlapping pair feeds the
    MIN-boundary accumulator on its own; a maximal run of mutually
    overlapping names feeds the MAX-boundary accumulator as one group.
    """
    k = h = 0
    seen_k, seen_h = set(), set()
    group_gold, group_pred = [], []

    def flush():
        if group_gold or group_pred:
            common = common_words_spans(text, group_gold, group_pred)
            correct = count_words_spans(text, group_gold)
            detected = count_words_spans(text, group_pred)
            max_acc.add(common, correct, detected)
            group_gold.clear()
            group_pred.clear()

    while k < len(gold_names) and h < len(pred_names):
        g, p = gold_names[k], pred_names[h]
        if overlaps(g[0], g[1], p[0], p[1]) and g[0] != p[1]:
            if k not in seen_k:
                seen_k.add(k)
                group_gold.append(g)
            if h not in seen_h:
                seen_h.add(h)
                group_pred.append(p)

            # min-boundary: this pair alone
            min_acc.add(common_words_spans(text, [g], [p]),
                        count_words_spans(text, [g]),
                        count_words_spans(text, [p]))

            advanced = 0
            if g[1] <= p[1]:
                k += 1
                advanced += 1
            if p[1] <= g[1]:
                h += 1
                advanced += 1
            if advanced == 2:            # both moved on -> the group closed
                flush()
        elif before(g[0], g[1], p[0], p[1]):
            flush()
            k += 1
        else:
            flush()
            h += 1

    flush()

    # Gold names past the last predicted one (and vice versa). The original's
    # loop is `while (k < gold && h < pred)` and its post-loop flush only
    # empties what was already collected, so an unmatched TAIL never reaches
    # the denominator — which inflates boundary recall, because words nobody
    # predicted are simply not counted. We count them, with zero overlap.
    # keyed off what was actually SEEN, not off k: the loop can exit with k
    # still pointing at a name it already processed, which would double count.
    for idx, g in enumerate(gold_names):
        if idx in seen_k:
            continue
        correct = count_words_spans(text, [g])
        min_acc.add(0, correct, 0)
        max_acc.add(0, correct, 0)
    # Note the asymmetry, which is inherent to the paper's formula rather than
    # a choice of ours: every contribution is WEIGHTED BY `correct`, so a
    # spurious prediction overlapping no gold name has weight zero and cannot
    # move precision. Precision here means "of the gold words we claimed, how
    # many did we get right", not "how much of what we output was right".


# ===========================================================================
# 5. The scorer
# ===========================================================================

def score(gold_items, pred_items, text, equal=None):
    """
    gold_items / pred_items : list[TwoLevelItem]
    text                    : the clean text the offsets index into

    Returns segmentation / detection / boundary_min / boundary_max blocks.
    """
    gold = sorted(gold_items, key=lambda x: x.main)
    pred = sorted(pred_items, key=lambda x: x.main)

    detection = _Weighted()
    boundary_min = _Weighted()
    boundary_max = _Weighted()

    matched_gold, matched_pred = set(), set()

    i = j = 0
    group_gold_names, group_pred_names = [], []

    def flush_names():
        if group_gold_names or group_pred_names:
            common, _all = common_names(group_gold_names, group_pred_names,
                                        equal)
            detection.add(common, len(group_gold_names),
                          len(group_pred_names))
            _word_level(text, group_gold_names, group_pred_names,
                        boundary_min, boundary_max)
            group_gold_names.clear()
            group_pred_names.clear()

    while i < len(gold) and j < len(pred):
        g, p = gold[i], pred[j]
        if overlaps(g.start, g.end, p.start, p.end) and g.start != p.end:
            if i not in matched_gold:
                matched_gold.add(i)
                group_gold_names.extend(g.names)
            if j not in matched_pred:
                matched_pred.add(j)
                group_pred_names.extend(p.names)

            advanced = 0
            if g.end <= p.end:
                i += 1
                advanced += 1
            if p.end <= g.end:
                j += 1
                advanced += 1
            if advanced == 2:
                flush_names()
        elif before(g.start, g.end, p.start, p.end):
            flush_names()
            i += 1
        else:
            flush_names()
            j += 1

    flush_names()

    # Same tail correction at the outer level: gold items never reached by the
    # sweep still owe their names to the detection and boundary denominators.
    for g in gold[i:]:
        if g.names:
            detection.add(0, len(g.names), 0)
            _word_level(text, g.names, [], boundary_min, boundary_max)

    return {
        "segmentation": {
            "recall": round(len(matched_gold) / len(gold), 4) if gold else 0.0,
            "precision": (round(len(matched_pred) / len(pred), 4)
                          if pred else 0.0),
            "gold": len(gold), "predicted": len(pred),
            "matched_gold": len(matched_gold),
            "matched_predicted": len(matched_pred),
        },
        "detection": detection.result(),
        "boundary_min": boundary_min.result(),
        "boundary_max": boundary_max.result(),
    }


# ===========================================================================
# 6. One-level convenience (old OneLevelAgreement)
# ===========================================================================

def score_one_level(gold_spans, pred_spans, text):
    """
    Score a flat list of spans (no main/inner structure) — the old
    OneLevelAgreement, which wraps everything in a single main span covering
    the whole text. Used for biography-entry spans (Table 3).
    """
    whole = (0, max(len(text) - 1, 0))
    return score([TwoLevelItem(whole, list(gold_spans))],
                 [TwoLevelItem(whole, list(pred_spans))], text)


def format_report(result, title="results"):
    """A compact table, in the shape the paper prints."""
    lines = [f"{title}", "=" * 58,
             f"{'':<18}{'recall':>10}{'precision':>12}{'F':>10}{'units':>8}"]
    seg = result["segmentation"]
    lines.append(f"{'segmentation':<18}{seg['recall']:>10}"
                 f"{seg['precision']:>12}{'':>10}{seg['gold']:>8}")
    for key, label in (("detection", "detection"),
                       ("boundary_min", "boundary (min)"),
                       ("boundary_max", "boundary (max)")):
        b = result[key]
        lines.append(f"{label:<18}{b['recall']:>10}{b['precision']:>12}"
                     f"{b['f_score']:>10}{b['units']:>8}")
    lines.append("=" * 58)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick demo:  python agreement.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    text = "محمد بن يحيا عن احمد بن محمد عن الحسن بن محبوب عن ابي جعفر"
    # gold: three narrators; prediction: the middle one truncated, the last missed
    gold = [TwoLevelItem((0, len(text) - 1),
                         [(0, 11), (16, 27), (32, 45)])]
    pred = [TwoLevelItem((0, len(text) - 1),
                         [(0, 11), (16, 20)])]

    print("text:", text)
    print("gold names:", [text[s:e + 1] for s, e in gold[0].names])
    print("pred names:", [text[s:e + 1] for s, e in pred[0].names])
    print()
    print(format_report(score(gold, pred, text), "demo"))
    print()
    print("detection counts a name correct if it merely OVERLAPS a gold name,")
    print("so the truncated 'احمد بن' still counts as detected — the boundary")
    print("rows are what expose that its span is wrong.")
