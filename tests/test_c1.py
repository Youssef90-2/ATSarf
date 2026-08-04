"""
test_c1.py — gating the weak `noun_prop` category on context.

The old system keeps bit_NOUN_PROP OUT of bits_NAME and admits it only while
`tryToLearnNames` is set — in four specific positions where a name is already
expected. This checks the same gate, plus the two hard filters (no suffix,
stem >= 3 chars), and that the loose behaviour is still reachable for the
ablation.

CAMeL is not needed: the promotion pass is driven by flags we set by hand.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from engine import ArabicEngine, TokenInfo                    # noqa: E402

_score = {"pass": 0, "fail": 0}


def check(label, got, want):
    if got == want:
        _score["pass"] += 1
        print(f"  PASS  {label}")
    else:
        _score["fail"] += 1
        print(f"  FAIL  {label}\n        got  {got}\n        want {want}")


def engine(strict=True):
    """An engine shell with no models loaded."""
    e = ArabicEngine.__new__(ArabicEngine)
    e.strict_names = strict
    return e


def toks(spec):
    """spec: [(word, *flags)] -> TokenInfo list with offsets."""
    out, pos = [], 0
    for word, *flags in spec:
        t = TokenInfo(word=word, start=pos, end=pos + len(word))
        for f in flags:
            setattr(t, f, True)
        out.append(t)
        pos += len(word) + 1
    return out


CAND = ("is_name_candidate",)
NRC = ("is_nrc",)
IBN = ("is_nmc", "is_ibn")
NAME = ("is_name",)
PUNCT = ("is_punct",)


def promoted(spec, strict=True):
    tokens = toks(spec)
    if strict:
        engine(True)._promote_name_candidates(tokens)
    return [t.word for t in tokens if t.is_name]


print("C1. the four contexts where a bare noun_prop IS a name")

check("after a name connector (بن X)",
      promoted([("محمد", *NAME), ("بن", *IBN), ("يعقوب", *CAND)]),
      ["محمد", "يعقوب"])

check("after a narration word (عن X)",
      promoted([("عن", *NRC), ("جعفر", *CAND)]), ["جعفر"])

check("before a name connector (X بن)",
      promoted([("سهل", *CAND), ("بن", *IBN), ("زياد", *NAME)]),
      ["سهل", "زياد"])

check("a nisba in narration context (عن الكوفي)",
      promoted([("عن", *NRC), ("الكوفي", *CAND)]), ["الكوفي"])

print()
print("C1. and where it is NOT")

check("a bare noun_prop in running matn is NOT a name",
      promoted([("لما",), ("خلق",), ("العقل", *CAND), ("استنطقه",)]), [])

check("...even between two ordinary words",
      promoted([("كان",), ("فرعون", *CAND), ("يقول",)]), [])

check("punctuation does not count as a name-expecting neighbour",
      promoted([(".", *PUNCT), ("موسا", *CAND), (".", *PUNCT)]), [])

print()
print("C1. the two hard filters from the old analyze()")

check("a stem shorter than 3 chars is rejected (h:269)",
      promoted([("عن", *NRC), ("ال", *CAND)]), [])

t = toks([("عن", *NRC), ("كتابهم", *CAND)])
t[1].has_enclitic = True
engine(True)._promote_name_candidates(t)
check("a word carrying an enclitic is rejected (h:267)",
      [x.word for x in t if x.is_name], [])

print()
print("C1. provenance and the ablation")

t = toks([("عن", *NRC), ("جعفر", *CAND)])
engine(True)._promote_name_candidates(t)
check("a promoted name records how it was learned",
      t[1].name_sources, ["camel-learned"])

# strict=False is the old loose behaviour: noun_prop alone is a name. The
# candidate flag is never set in that mode (analyze sets is_name directly),
# so the promotion pass simply has nothing to do.
check("with strict_names off the promotion pass is skipped",
      promoted([("لما",), ("العقل", *CAND)], strict=False), [])

print()
print("C1. the matn case this was built for")

# 'الله' and 'العقل' are noun_prop-ish in running text. Under the old loose
# rule both became narrators; the FSM's narr_min=3 then emitted a fake sanad.
matn = [("لما",), ("خلق", ), ("الله", *CAND), ("العقل", *CAND),
        ("استنطقه",), ("ثم",), ("قال",)]
check("no fake narrators survive in matn prose", promoted(matn), [])

# but a real sanad is untouched
sanad = [("عن", *NRC), ("احمد", *CAND), ("بن", *IBN), ("محمد", *CAND),
         ("عن", *NRC), ("الحسن", *CAND), ("بن", *IBN), ("محبوب", *CAND)]
check("a real sanad keeps every narrator name",
      promoted(sanad), ["احمد", "محمد", "الحسن", "محبوب"])

print()
print(f"{_score['pass']} passed, {_score['fail']} failed")
sys.exit(1 if _score["fail"] else 0)
