"""
test_table2.py — the narrator-level scoring path (paper Table 2).

Table 2 is "Narrator detection in biographies": its columns are whether a
narrator was found, and whether the span OF HIS NAME is right. It is NOT about
biography entry boundaries — those are Table 3. Scoring it through the detected
entries collapses the no-cross column to zero, because that condition derives
no entries at all by construction.

Two behaviours are pinned here:

  1. the cross-document column applies the original's filter. addNarrators
     (narratordetector.cpp:193) appends a narrator to narratorList only when
     its graph match list is non-empty, so an unrecognised narrator is
     discarded outright. Without that filter both columns are identical,
     since the automaton produces the same spans either way.

  2. the boundary column is conditioned on detection. §6 says ANGE "detected
     40% of the boundaries OF THE EXTRACTED NARRATORS", and the original
     computes it that way: a gold name matching nothing is skipped with k++
     and never reaches countCorrect. Our scorer deliberately counts those
     unmatched names with zero overlap, which is the stricter reading but
     makes the paper's movement impossible to reproduce — discarding a
     narrator would then always LOWER boundary recall. Both denominators are
     therefore reported, and this file asserts they move in OPPOSITE
     directions, which is the whole reason to report both.

No CAMeL needed: spans are synthesized.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from agreement import TwoLevelItem, score_one_level          # noqa: E402
from run_biography import (narrator_spans, gold_narrator_spans,  # noqa: E402
                           restrict_to_detected)

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    PASS, FAIL = PASS + ok, FAIL + (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        print(f"        got  {got}")
        print(f"        want {want}")


class Mention:
    """Enough of a NarratorMention for the scoring path."""

    def __init__(self, start, end, confirmed):
        self.start, self.end, self.confirmed = start, end, confirmed


# 40 three-letter words separated by spaces: word i occupies [4i, 4i+3)
TEXT = " ".join(f"w{i:02d}" for i in range(40))


def span(first_word, last_word):
    return (4 * first_word, 4 * last_word + 3)


# four gold narrators, three words each
G = [span(0, 2), span(5, 7), span(12, 14), span(17, 19)]


print("A. narrator_spans — the flat list Table 2 scores")
mentions = [Mention(*G[0], False), Mention(*G[1], True),
            Mention(*G[2], False), Mention(*G[3], True)]
check("unfiltered keeps every mention", narrator_spans(mentions), G)
check("confirmed_only reproduces addNarrators' `if (size > 0)`",
      narrator_spans(mentions, confirmed_only=True), [G[1], G[3]])
check("a run with nothing confirmed yields nothing under the filter",
      narrator_spans([Mention(*G[0], False)], confirmed_only=True), [])

print("\nB. gold_narrator_spans flattens names out of their entries")
gold_items = [TwoLevelItem(main=span(0, 8), names=[G[1], G[0]]),
              TwoLevelItem(main=span(11, 20), names=[G[3], G[2]])]
check("flattened and sorted", gold_narrator_spans(gold_items), G)
check("no entries -> no names", gold_narrator_spans([]), [])

print("\nC. detection: the filter costs recall (the paper's 0.94 -> 0.65)")
gold_names = gold_narrator_spans(gold_items)
ng = score_one_level(gold_names, narrator_spans(mentions), TEXT)
wg = score_one_level(gold_names,
                     narrator_spans(mentions, confirmed_only=True), TEXT)
check("no-cross finds all four", ng["detection"]["recall"], 1.0)
check("cross keeps two of four", wg["detection"]["recall"], 0.5)
check("but precision does not fall", wg["detection"]["precision"], 1.0)

print("\nD. the two boundary denominators move in OPPOSITE directions")
# unconfirmed mentions are truncated to one word of three; confirmed are exact
bad = [Mention(*span(0, 0), False), Mention(*G[1], True),
       Mention(*span(12, 12), False), Mention(*G[3], True)]
ng_s = narrator_spans(bad)
wg_s = narrator_spans(bad, confirmed_only=True)

a_ng = score_one_level(gold_names, ng_s, TEXT)
a_wg = score_one_level(gold_names, wg_s, TEXT)
d_ng = score_one_level(restrict_to_detected(gold_names, ng_s), ng_s, TEXT)
d_wg = score_one_level(restrict_to_detected(gold_names, wg_s), wg_s, TEXT)

print(f"        all gold      : {a_ng['boundary_min']['recall']} -> "
      f"{a_wg['boundary_min']['recall']}")
print(f"        detected only : {d_ng['boundary_min']['recall']} -> "
      f"{d_wg['boundary_min']['recall']}")
check("all-gold denominator: discarding narrators LOWERS boundary recall",
      a_wg["boundary_min"]["recall"] < a_ng["boundary_min"]["recall"], True)
check("detected-only denominator: it RAISES it (the paper's 0.41 -> 0.93)",
      d_wg["boundary_min"]["recall"] > d_ng["boundary_min"]["recall"], True)
check("under the paper's denominator the exact spans score perfectly",
      d_wg["boundary_min"]["recall"], 1.0)

print("\nE. restrict_to_detected")
check("keeps only gold overlapped by some prediction",
      restrict_to_detected(gold_names, wg_s), [G[1], G[3]])
check("empty predictions keep nothing",
      restrict_to_detected(gold_names, []), [])
check("a partial overlap still counts as detected",
      restrict_to_detected([G[0]], [span(0, 0)]), [G[0]])

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
