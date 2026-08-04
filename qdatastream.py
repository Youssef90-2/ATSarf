"""
qdatastream.py
==============
Reader for Qt `QDataStream` binaries, so the original ATSarf annotation files
can be imported directly.

The old system stored every gold artifact this way (and `.gitignore`d all of
them, which is why none shipped):

    <book>.names   QList<QPair<int,int>>            name spans
    <book>.narr    QList<QPair<int,int>>            narrator spans
    <book>.tags    QList<QPair<int,int>>            biography spans (FLAT form)
    <book>.tags    two-level form (see below)       hadith gold
    <book>.equal   QMap<QPair<QString,QString>,bool>  narrator-pair judgements

Note `.tags` is OVERLOADED — biographies write a flat span list
(narratordetector.cpp:743) while hadith writes the two-level structure
(AbstractTwoLevelAgreement.cpp:185). `read_tags` sniffs which one it is.

--------------------------------------------------------------------------
ENCODING (QDataStream, big-endian, stable across versions for these types):
    qint8/quint8   1 byte
    bool           1 byte
    qint32         4 bytes, signed
    qint64         8 bytes, signed
    QString        quint32 byte-length (0xFFFFFFFF = null) + UTF-16BE
    QList<T>       quint32 count + count x T
    QPair<A,B>     A then B
    QMap<K,V>      quint32 count + count x (K, V)

THE TWO-LEVEL .tags LAYOUT, read off the C++ rather than guessed:
    qint32 count                                 (SelectionList::readFromStream)
    count x TwoLevelSelection:                   (twoLevelTaggerSelection.h:80)
        QPair<int,int>            main span
        QList<QPair<int,int>>     name spans
        Chain                     the annotator's chain graph:
            qint8 'A', qint32 size, size x ChainPrim
            ChainPrim 'a' = Narrator:
                bool isRasoul, qint32 n, n x (NarratorPrim + qint32 start + qint32 end)
                    NarratorPrim 'N' = NamePrim:          bool learnedName
                    NarratorPrim 'c' = NameConnectorPrim: qint64 start, qint64 end, quint8 type
            ChainPrim 'p' = NarratorConnectorPrim: qint64 start, qint64 end

    (NameConnectorPrim really does store its offsets TWICE — once as qint64 in
     its own serialize, once as qint32 in the enclosing Narrator loop. That is
     the format, not a misreading: narrator_abstraction.cpp:79 and :146.)

Parsing is STRICT. An unexpected type tag raises with the byte offset rather
than silently producing plausible-looking garbage — a gold file decoded
slightly wrong would poison every number downstream and look fine.
"""

import struct
from pathlib import Path


class QDataStreamError(Exception):
    pass


# ===========================================================================
# 1. Primitive reader
# ===========================================================================

class QDataStreamReader:
    """Sequential big-endian reader over a bytes buffer."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    # ---- low level ----
    def _take(self, n):
        if self.pos + n > len(self.data):
            raise QDataStreamError(
                f"unexpected end of stream at byte {self.pos} "
                f"(wanted {n} more, {len(self.data) - self.pos} left)")
        chunk = self.data[self.pos:self.pos + n]
        self.pos += n
        return chunk

    def at_end(self):
        return self.pos >= len(self.data)

    # ---- scalars ----
    def int8(self):
        return struct.unpack(">b", self._take(1))[0]

    def uint8(self):
        return struct.unpack(">B", self._take(1))[0]

    def bool_(self):
        return self.uint8() != 0

    def int32(self):
        return struct.unpack(">i", self._take(4))[0]

    def uint32(self):
        return struct.unpack(">I", self._take(4))[0]

    def int64(self):
        return struct.unpack(">q", self._take(8))[0]

    def qstring(self):
        length = self.uint32()
        if length == 0xFFFFFFFF:            # Qt's null string
            return None
        if length % 2:
            raise QDataStreamError(
                f"QString byte length {length} is odd at {self.pos}")
        return self._take(length).decode("utf-16-be")

    # ---- containers ----
    def qlist(self, item):
        return [item() for _ in range(self.uint32())]

    def qpair(self, first, second):
        return (first(), second())

    def qmap(self, key, value):
        return {key(): value() for _ in range(self.uint32())}


# ===========================================================================
# 2. ATSarf structures
# ===========================================================================

_CHAIN = ord("A")
_NARRATOR = ord("a")
_NARR_CONNECTOR = ord("p")
_NAME_PRIM = ord("N")
_NAME_CONNECTOR = ord("c")


def _read_span(r):
    return (r.int32(), r.int32())


def _read_chain(r):
    """
    Consume one serialized Chain and return its narrator spans.
    We keep only what the scorer can use; the rest is validated and skipped.
    """
    tag = r.int8()
    if tag != _CHAIN:
        raise QDataStreamError(
            f"expected Chain tag 'A' at byte {r.pos - 1}, got {tag!r}")
    narrators = []
    for _ in range(r.int32()):
        prim = r.int8()
        if prim == _NARRATOR:
            r.bool_()                                  # isRasoul
            parts = []
            for _ in range(r.int32()):
                kind = r.int8()
                if kind == _NAME_PRIM:
                    r.bool_()                          # learnedName
                elif kind == _NAME_CONNECTOR:
                    r.int64(); r.int64(); r.uint8()    # start, end, type
                else:
                    raise QDataStreamError(
                        f"unknown NarratorPrim tag {kind!r} at byte {r.pos - 1}"
                        " — the file layout does not match the expected one")
                parts.append((r.int32(), r.int32()))   # start, end
            if parts:
                narrators.append((parts[0][0], parts[-1][1]))
        elif prim == _NARR_CONNECTOR:
            r.int64(); r.int64()
        else:
            raise QDataStreamError(
                f"unknown ChainPrim tag {prim!r} at byte {r.pos - 1}")
    return narrators


# ===========================================================================
# 3. Public loaders
# ===========================================================================

def read_span_list(path):
    """
    A flat QList<QPair<int,int>> — the `.names`, `.narr`, and biography
    `.tags` files. Returns [(start, end), ...].
    """
    r = QDataStreamReader(Path(path).read_bytes())
    spans = r.qlist(lambda: _read_span(r))
    if not r.at_end():
        raise QDataStreamError(
            f"{Path(path).name}: {len(r.data) - r.pos} trailing bytes — this "
            "is probably a two-level .tags file; use read_two_level()")
    return spans


def read_two_level(path):
    """
    The hadith `.tags` form. Returns
        [{"main": (s, e), "names": [(s, e), ...], "chain": [(s, e), ...]}]
    `chain` is the annotator's own narrator segmentation, kept for Table 1.
    """
    r = QDataStreamReader(Path(path).read_bytes())
    items = []
    for _ in range(r.int32()):
        main = _read_span(r)
        names = r.qlist(lambda: _read_span(r))
        chain = _read_chain(r)
        items.append({"main": main, "names": names, "chain": chain})
    return items


def read_tags(path):
    """
    `.tags` is overloaded. Try the flat form, fall back to two-level, and if
    neither parses cleanly say so rather than returning half a file.
    """
    try:
        return {"kind": "flat", "spans": read_span_list(path)}
    except QDataStreamError:
        pass
    try:
        return {"kind": "two_level", "items": read_two_level(path)}
    except QDataStreamError as exc:
        raise QDataStreamError(
            f"{Path(path).name} parsed as neither a flat span list nor a "
            f"two-level annotation: {exc}") from exc


def read_equality_map(path):
    """
    `.equal` — QMap<QPair<QString,QString>, bool>. Returns {(n1, n2): bool},
    the human judgement of whether two narrator names are the same person.
    Feeds the metric-vs-Levenshtein comparison (paper §5).
    """
    r = QDataStreamReader(Path(path).read_bytes())
    return r.qmap(lambda: (r.qstring(), r.qstring()), r.bool_)


# ---------------------------------------------------------------------------
# Self-test: round-trip against bytes we encode ourselves to the documented
# layout. Without a real .tags file to hand this is the only honest check —
# it verifies the READER against the SPEC, not against Qt.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    def u32(v):
        return struct.pack(">I", v)

    def i32(v):
        return struct.pack(">i", v)

    def i64(v):
        return struct.pack(">q", v)

    def qstr(s):
        b = s.encode("utf-16-be")
        return u32(len(b)) + b

    # flat span list
    flat = u32(2) + i32(0) + i32(11) + i32(16) + i32(27)
    r = QDataStreamReader(flat)
    assert r.qlist(lambda: _read_span(r)) == [(0, 11), (16, 27)]
    print("flat span list        OK")

    # two-level: one item, main + 2 names + a chain of one narrator
    chain = (bytes([_CHAIN]) + i32(1)
             + bytes([_NARRATOR]) + bytes([0]) + i32(2)
             + bytes([_NAME_PRIM]) + bytes([0]) + i32(0) + i32(3)
             + bytes([_NAME_CONNECTOR]) + i64(5) + i64(6) + bytes([1])
             + i32(5) + i32(6))
    two = (i32(1) + i32(0) + i32(60)
           + u32(2) + i32(0) + i32(11) + i32(16) + i32(27)
           + chain)
    items = read_two_level_bytes = None
    r = QDataStreamReader(two)
    got = []
    for _ in range(r.int32()):
        main = _read_span(r)
        names = r.qlist(lambda: _read_span(r))
        got.append({"main": main, "names": names, "chain": _read_chain(r)})
    assert got[0]["main"] == (0, 60), got
    assert got[0]["names"] == [(0, 11), (16, 27)], got
    assert got[0]["chain"] == [(0, 6)], got
    print("two-level annotation  OK")

    # equality map
    eq = u32(1) + qstr("محمد بن يعقوب") + qstr("محمد بن يعقوب الكليني") + bytes([1])
    r = QDataStreamReader(eq)
    m = r.qmap(lambda: (r.qstring(), r.qstring()), r.bool_)
    assert list(m.values()) == [True] and len(m) == 1
    print("equality map          OK")

    # strictness: a corrupted tag must raise, not guess
    bad = i32(1) + i32(0) + i32(5) + u32(0) + bytes([ord("Z")]) + i32(0)
    try:
        r = QDataStreamReader(bad)
        r.int32(); _read_span(r); r.qlist(lambda: _read_span(r)); _read_chain(r)
        raise AssertionError("should have raised")
    except QDataStreamError as e:
        print("strict on bad input   OK  ->", str(e)[:52], "...")

    print("\nqdatastream.py self-test passed")
