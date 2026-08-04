"""
test_c5.py — the ablation switches.

This port added three heuristics the old FSM did not have, and one fidelity
fix (C1) that changes name detection. A reproduction claim has to be able to
show both columns rather than quietly shipping undocumented rules, so each is
switchable and each switch is verified to actually change behaviour.

IMPORTANT DISTINCTION, asserted below: `FsmParams.faithful()` turns off the
ADDITIONS only. The fidelity FIXES (C1 gating, C3 honorific handling, the
graph guards) stay on in both modes, because disabling them would make the
port less faithful, not more.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from engine import TokenInfo                                   # noqa: E402
from fsm import HadithFSM, FsmParams                           # noqa: E402

_score = {"pass": 0, "fail": 0}


def check(label, got, want):
    if got == want:
        _score["pass"] += 1
        print(f"  PASS  {label}")
    else:
        _score["fail"] += 1
        print(f"  FAIL  {label}\n        got  {got}\n        want {want}")


def stream(spec):
    out, pos = [], 0
    for word, *flags in spec:
        t = TokenInfo(word=word, start=pos, end=pos + len(word))
        for f in flags:
            if f.startswith("pos="):
                t.pos = f.split("=", 1)[1]
            else:
                setattr(t, f, True)
        out.append(t)
        pos += len(word) + 1
    return out


NAME = ("is_name",); NRC = ("is_nrc",); IBN = ("is_nmc", "is_ibn")
RAS = ("is_rasoul",); PUNCT = ("is_punct",); NUM = ("is_number",)


def names(spec, params):
    return [n.full_name
            for c in HadithFSM(params).run(stream(spec))
            for n in c.chain.narrators]


print("C5. the defaults are all ON")
d = FsmParams()
check("matn_cues", d.matn_cues, True)
check("pos_hard_stop", d.pos_hard_stop, True)
check("one_chain_per_hadith", d.one_chain_per_hadith, True)

f = FsmParams.faithful()
check("faithful() turns all three off",
      (f.matn_cues, f.pos_hard_stop, f.one_chain_per_hadith),
      (False, False, False))
check("faithful() still accepts the tunables",
      FsmParams.faithful(narr_min=1).narr_min, 1)

print()
print("C5. matn_cues actually changes behaviour")

# '...محمد بن عيسا قال: كنت انا' — with the cue ON the sanad closes at قال:،
# with it OFF the matn words are swallowed into the chain.
spec = [("محمد", *NAME), ("بن", *IBN), ("عيسا", *NAME),
        ("قال",), (":", *PUNCT),
        ("كنت",), ("انا",), ("جالسا",)]
on = names(spec, FsmParams(narr_min=1))
off = names(spec, FsmParams(narr_min=1, matn_cues=False))
check("cue ON keeps the sanad clean", on, ["محمد بن عيسا"])
check("cue OFF swallows matn words", len(" ".join(off)) > len(" ".join(on)),
      True)

print()
print("C5. pos_hard_stop actually changes behaviour")

# a verb inside a name: ON it breaks the name, OFF it is tolerated as an NMC
spec = [("عبد", *NAME), ("الله", *NAME), ("بن", *IBN), ("جندب", *NAME),
        ("كتب", "pos=verb"), ("اليه",), ("الرضا",)]
on = names(spec, FsmParams(narr_min=1))
off = names(spec, FsmParams(narr_min=1, pos_hard_stop=False))
check("hard-stop ON cuts the name at the verb",
      all("كتب" not in n for n in on), True)
check("hard-stop OFF lets the verb into a name",
      any("كتب" in n for n in off), True)

print()
print("C5. one_chain_per_hadith actually changes behaviour")

# after the sanad ends at the Imam, matn text mentioning عن ابي عبد الله
# must not open a second sanad — unless the switch is off.
spec = [("محمد", *NAME), ("بن", *IBN), ("يحيا", *NAME),
        ("عن", *NRC), ("النبي", *RAS), ("قال",),
        ("يبلغه",), ("عن", *NRC), ("جعفر", *NAME), ("بن", *IBN),
        ("محمد", *NAME)]
on = HadithFSM(FsmParams(narr_min=1)).run(stream(spec))
off = HadithFSM(FsmParams(narr_min=1,
                          one_chain_per_hadith=False)).run(stream(spec))
check("ON: the matn mention does not open a second sanad", len(on), 1)
check("OFF: it does", len(off) > 1, True)

print()
print("C5. fidelity FIXES stay on in faithful mode")

# C3: an honorific must never become a narrator, in EITHER mode.
spec = [("محمد", *NAME), ("بن", *IBN), ("يحيا", *NAME),
        ("عن", *NRC), ("عليه", *RAS), ("السلام", *RAS), ("قال",)]
for label, params in (("default", FsmParams(narr_min=1)),
                      ("faithful", FsmParams.faithful(narr_min=1))):
    got = names(spec, params)
    check(f"{label}: 'عليه السلام' is not a narrator",
          [n for n in got if n.strip() == "عليه السلام"], [])

print()
print(f"{_score['pass']} passed, {_score['fail']} failed")
sys.exit(1 if _score["fail"] else 0)
