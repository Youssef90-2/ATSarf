"""
test_a5.py — Phase 1 / A5: the gold store and the Qt importers.

Covers the three things that make gold trustworthy:
  * the Qt reader decodes the documented layout, and is STRICT on anything else
  * un-reviewed and wrong-text gold is REFUSED, not silently scored
  * validation catches broken spans before they skew a table
"""
import json
import struct
import sys
import pathlib
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from qdatastream import (QDataStreamReader, QDataStreamError,   # noqa: E402
                         read_span_list, read_two_level, read_tags,
                         read_equality_map)
from gold import (GoldSet, GoldItem, GoldError, load_for_scoring,  # noqa: E402
                  text_fingerprint, validate, bootstrap_from_hadiths,
                  import_qt_spans, import_qt_two_level)
from agreement import score                                    # noqa: E402

_score = {"pass": 0, "fail": 0}
TMP = pathlib.Path(tempfile.gettempdir())
TEXT = "محمد بن يحيا عن احمد بن محمد عن الحسن بن محبوب عن ابي جعفر"


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
    except Exception as other:               # noqa: BLE001
        _score["fail"] += 1
        print(f"  FAIL  {label}\n        raised {type(other).__name__}")
        return
    _score["fail"] += 1
    print(f"  FAIL  {label}\n        did not raise")


# --------------------------------------------------------------- encoders
def u32(v):
    return struct.pack(">I", v)


def i32(v):
    return struct.pack(">i", v)


def i64(v):
    return struct.pack(">q", v)


def qstr(s):
    b = s.encode("utf-16-be")
    return u32(len(b)) + b


def write(name, data):
    p = TMP / name
    p.write_bytes(data)
    return p


print("A5. Qt QDataStream reader")

flat = write("t.names", u32(2) + i32(0) + i32(11) + i32(16) + i32(27))
check("flat span list (.names / .narr / biography .tags)",
      read_span_list(flat), [(0, 11), (16, 27)])

CHAIN = (bytes([ord("A")]) + i32(1)
         + bytes([ord("a")]) + bytes([0]) + i32(2)
         + bytes([ord("N")]) + bytes([0]) + i32(0) + i32(3)
         + bytes([ord("c")]) + i64(5) + i64(6) + bytes([1]) + i32(5) + i32(6))
two = write("t.tags", i32(1) + i32(0) + i32(60)
            + u32(2) + i32(0) + i32(11) + i32(16) + i32(27) + CHAIN)
items = read_two_level(two)
check("two-level: main span", items[0]["main"], (0, 60))
check("two-level: name spans", items[0]["names"], [(0, 11), (16, 27)])
check("two-level: annotator's chain recovered", items[0]["chain"], [(0, 6)])

check("read_tags sniffs the flat form", read_tags(flat)["kind"], "flat")
check("read_tags sniffs the two-level form", read_tags(two)["kind"],
      "two_level")

eq = write("t.equal", u32(1) + qstr("محمد بن يعقوب")
           + qstr("محمد بن يعقوب الكليني") + bytes([1]))
check("equality map (.equal)",
      read_equality_map(eq), {("محمد بن يعقوب", "محمد بن يعقوب الكليني"): True})

# strictness — a wrong byte must raise with a position, never guess
bad = write("bad.tags", i32(1) + i32(0) + i32(5) + u32(0)
            + bytes([ord("Z")]) + i32(0))
raises("a corrupt type tag raises rather than guessing",
       lambda: read_two_level(bad), QDataStreamError)
raises("truncated stream raises",
       lambda: read_span_list(write("short.names", u32(5) + i32(0))),
       QDataStreamError)

print()
print("A5. the safeguard (what boundary_gold.py removed)")

seed = GoldSet(kind="hadith", source="demo", text_sha256=text_fingerprint(TEXT),
               items=[GoldItem(main=(0, len(TEXT) - 1),
                               names=[(0, 11), (16, 27)])])
gpath = TMP / "demo.gold.json"
seed.save(gpath)

check("a bootstrapped seed is not reviewed", seed.reviewed, False)
raises("scoring against an UN-REVIEWED seed is refused",
       lambda: load_for_scoring(gpath, TEXT), GoldError)

d = json.loads(gpath.read_text(encoding="utf-8"))
d["reviewed"] = True
gpath.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
check("after review it loads", load_for_scoring(gpath, TEXT).reviewed, True)

raises("gold made against a DIFFERENT text is refused",
       lambda: load_for_scoring(gpath, TEXT + " زيادة"), GoldError)

print()
print("A5. bootstrap from system output")

hadiths = [{"sanad": {"start": 0, "end": 27,
                      "narrators": [{"start": 0, "end": 11},
                                    {"start": 16, "end": 27}]},
            "matn": {"end": len(TEXT) - 1}}]
boot = bootstrap_from_hadiths(hadiths, TEXT, source="demo")
check("bootstrap captures the narrator spans",
      boot.items[0].names, [(0, 11), (16, 27)])
check("bootstrap is marked un-reviewed", boot.reviewed, False)

print()
print("A5. validation")

broken = GoldSet(kind="hadith", source="x", text_sha256=text_fingerprint(TEXT),
                 items=[GoldItem(main=(0, 20), names=[(0, 11), (5, 15)]),
                        GoldItem(main=(10, 40), names=[(900, 950)])])
problems = validate(broken, TEXT)
check("overlapping name spans caught",
      any("overlap" in p and "name spans" in p for p in problems), True)
check("out-of-text span caught",
      any("outside text" in p for p in problems), True)
check("overlapping units caught",
      any(p.startswith("units") for p in problems), True)
check("a clean set validates clean",
      validate(GoldSet(kind="hadith", source="x",
                       text_sha256=text_fingerprint(TEXT),
                       items=[GoldItem(main=(0, 27),
                                       names=[(0, 11), (16, 27)])]), TEXT),
      [])

print()
print("A5. end-to-end: import Qt gold -> score against it")

imported = import_qt_spans(flat, TEXT, kind="biography", source="t")
check("imported Qt gold is reviewed (a human made it)", imported.reviewed, True)
result = score(imported.to_items(), imported.to_items(), TEXT)
check("gold scored against itself is perfect",
      (result["detection"]["recall"], result["boundary_min"]["recall"]),
      (1.0, 1.0))

two_gold = import_qt_two_level(two, TEXT, source="t")
check("two-level import produces scorable items",
      len(two_gold.to_items()[0].names), 2)

for p in (flat, two, eq, bad, gpath, TMP / "short.names"):
    pathlib.Path(p).unlink(missing_ok=True)

print()
print(f"{_score['pass']} passed, {_score['fail']} failed")
sys.exit(1 if _score["fail"] else 0)
