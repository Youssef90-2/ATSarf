"""
test_c3.py — C3 regression tests (honorific / junk narrators).

Runs the REAL FSM. Tokens are built from lexicons + phrase flags only, so the
tests need no CAMeL install. Two groups:

  A. end-to-end on real kafi sentences (the normal path)
  B. synthetic token streams that FORCE the standalone-honorific situation
     (an honorific arriving with no narrator open) — this is the case the
     morphology layer was producing on the full book.
"""
import sys
import pathlib
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from normalization import normalize                        # noqa: E402
from engine import (TokenInfo, tokenize_with_positions,    # noqa: E402
                    ArabicEngine, PUNCTUATION_CHARS)
from fsm import HadithFSM, FsmParams                       # noqa: E402

_eng = ArabicEngine.__new__(ArabicEngine)      # no CAMeL needed


def analyze(clean):
    toks = [TokenInfo(word=w, start=s, end=e)
            for w, s, e in tokenize_with_positions(clean)]
    idx = []
    for i, t in enumerate(toks):
        if t.word in PUNCTUATION_CHARS:
            t.is_punct = True
        elif t.word.isdigit():
            t.is_number = True
        else:
            idx.append(i)
    for i in idx:
        _eng._apply_lexicon_flags(toks[i])
        if toks[i].is_nrc or toks[i].is_nmc or toks[i].is_rasoul:
            toks[i].is_name = False
    _eng._apply_phrase_flags(clean, toks)
    _eng._apply_context_name_rule(toks)
    return toks


def run(text):
    clean, _ = normalize(text)
    cands = HadithFSM(FsmParams()).run(analyze(clean))
    return [[(n.full_name, n.is_rasoul) for n in c.chain.narrators]
            for c in cands]


PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    ok = got == want
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        print(f"        got  {got}")
        print(f"        want {want}")


print("A. real sentences")
r = run("3 - احمد بن ادريس عن محمد بن عبد الجبار عن بعض اصحابنا "
        "عن ابي عبد الله عليه السلام قال: قلت له ما العقل")
check("honorific dropped from the name, no standalone node",
      r[0], [("احمد بن ادريس", False), ("محمد بن عبد الجبار", False),
             ("بعض اصحابنا", False), ("ابي عبد الله", False)])

r = run("5 - محمد بن يحيا عن احمد بن محمد عن ابي جعفر عليه السلام قال: كذا")
check("different Imams stay distinct (is_rasoul NOT set)",
      r[0][-1], ("ابي جعفر", False))

r = run("7 - قتيبه بن سعيد عن جرير عن ابي هريره عن رسول الله "
        "صلا الله عليه واله قال: كذا")
check("the Prophet HIMSELF stays a narrator, flagged rasoul",
      r[0][-1], ("رسول الله صلا الله عليه واله", True))

print()
print("B. synthetic: honorific arrives with NO narrator open")


def synth(words, flags):
    """Build tokens by hand so we can force any flag combination."""
    toks, pos = [], 0
    for w, f in zip(words, flags):
        t = TokenInfo(word=w, start=pos, end=pos + len(w))
        for k in f:
            setattr(t, k, True)
        toks.append(t)
        pos += len(w) + 1
    return toks


# محمد بن يحيا  عن  <honorific with no name after عن>
toks = synth(
    ["محمد", "بن", "يحيا", "عن", "عليه", "السلام", "قال"],
    [["is_name"], ["is_nmc", "is_ibn"], ["is_name"], ["is_nrc"],
     ["is_rasoul"], ["is_rasoul"], ["is_nrc"]])
out = [(n.full_name, n.is_rasoul)
       for n in HadithFSM(FsmParams(narr_min=1)).run(toks)[0].chain.narrators]
check("no narrator open -> honorific dropped, no new node",
      out, [("محمد بن يحيا", False)])

# honorific as the very FIRST thing -> nothing to attach to -> dropped
toks = synth(["عليه", "السلام", "محمد", "بن", "يحيا", "عن", "احمد"],
             [["is_rasoul"], ["is_rasoul"], ["is_name"],
              ["is_nmc", "is_ibn"], ["is_name"], ["is_nrc"], ["is_name"]])
res = HadithFSM(FsmParams(narr_min=1)).run(toks)
names = [n.full_name for c in res for n in c.chain.narrators]
check("leading honorific with no predecessor is dropped",
      [n for n in names if n.strip() == "عليه السلام"], [])

# a bare الله must not survive as a narrator
toks = synth(["محمد", "بن", "يحيا", "عن", "الله", "عن", "احمد"],
             [["is_name"], ["is_nmc", "is_ibn"], ["is_name"], ["is_nrc"],
              ["is_name"], ["is_nrc"], ["is_name"]])
res = HadithFSM(FsmParams(narr_min=1)).run(toks)
names = [n.full_name.strip() for c in res for n in c.chain.narrators]
check("a bare 'الله' is not a narrator", [n for n in names if n == "الله"], [])

# عبد الله must NOT be absorbed (عبد is not an honorific token)
toks = synth(["عبد", "الله", "عن", "احمد", "بن", "محمد"],
             [["is_name"], ["is_name"], ["is_nrc"], ["is_name"],
              ["is_nmc", "is_ibn"], ["is_name"]])
res = HadithFSM(FsmParams(narr_min=1)).run(toks)
names = [n.full_name.strip() for c in res for n in c.chain.narrators]
check("'عبد الله' is a real narrator, untouched", names[0], "عبد الله")

print()
print(f"{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
