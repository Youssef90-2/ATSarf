"""
biography_segmenter.py
======================
IDEA (paper §5.2-5.3, old NarratorDetector::lookup, narratordetector.cpp:424):
Cut a rijal (biography) book into one ENTRY per narrator. Each entry names a
subject narrator and lists the narrators he/she "روى عن" (teachers) and
"روى عنه" (students) — exactly the professors/students relation the paper
uses for cross-document disambiguation (§4, §5.5).

Old system: same FSM as hadith but in BIOGRAPHY MODE
(HadithData(text, false, ...)) with the bio_* parameters and
paragraphDelimiters tracking "number + dash" as hard entry boundaries
(additionalCheck, narratordetector.cpp:378). We reproduce the ideas, adapted
to the Khoei book's real structure that we verified:

    356 - إبراهيم الشيباني: روى عن أبي الجارود، وروى عنه حرب بن الحسين. التهذيب...
    ^^^   ^^^^^^^^^^^^^^^^   ^^^^^^ teachers    ^^^^^^^^ students     ^^ source
    num   subject name       (روى عن ...)        (روى عنه ...)

WHAT WE EXTRACT PER ENTRY (this is what the linker/Table-2 evaluation needs):
    number        the sequential entry number (356, 357, ...)
    subject       the narrator the entry is about ("إبراهيم الشيباني")
    teachers[]    names after "روى عن"   (the subject's professors)
    students[]    names after "روى عنه"  (the subject's students)
    start, end    char offsets of the entry in the clean text (its boundary)

--------------------------------------------------------------------------
WHY WE DON'T JUST RUN THE HADITH FSM HERE:
The paper's own Table 2 shows that WITHOUT the hadith graph, narrator
*boundary* recall is only 41% — the FSM alone cannot reliably tell where one
entry's narrators end and the next begin in biography prose. In THIS book the
"N - name:" entry header is an explicit, reliable boundary (unlike free hadith
text), so we use it as the paragraph delimiter (old paragraphDelimiters idea)
and read teachers/students from the "روى عن / روى عنه" markers directly. The
harder graph-assisted boundary detection is exactly what the linker adds on
top (the 41%->93% story), so the segmenter stays deliberately simple and
transparent, and the *graph's contribution* is measured separately.

NOTE ON NORMALization: after normalize(), ى->ا, so "روى" becomes "روا".
All markers below are matched in NORMALIZED text.
--------------------------------------------------------------------------
"""

import re
from dataclasses import dataclass, field

from normalization import normalize
import lexicons


# ===========================================================================
# 1. Entry data structure
# ===========================================================================

@dataclass
class BiographyEntry:
    number: int                       # sequential entry number in the book
    subject: str                      # the narrator this entry is about
    teachers: list = field(default_factory=list)   # روى عن ...  (professors)
    students: list = field(default_factory=list)   # روى عنه ... (students)
    all_narrators: list = field(default_factory=list)  # OLD-STYLE: every
                                      # narrator name in the entry, in order of
                                      # appearance, WITHOUT pre-labelling
                                      # teacher/student — the graph decides
                                      # who is a professor (older generation)
                                      # vs student (newer) via rank. This is
                                      # how the old code (Biography::narrators)
                                      # worked; it lets the graph's generation
                                      # ordering do the disambiguation (the
                                      # cross-document step), so the graph's
                                      # contribution shows up more strongly.
    start: int = -1                   # char offset in clean text
    end: int = -1
    is_reference: bool = False        # "X: = Y" cross-reference, no body
    raw: str = ""                     # the raw entry text (for review)

    def to_dict(self):
        return {"number": self.number, "subject": self.subject,
                "teachers": self.teachers, "students": self.students,
                "all_narrators": self.all_narrators,
                "start": self.start, "end": self.end,
                "is_reference": self.is_reference}


# ===========================================================================
# 2. Markers (matched in NORMALIZED text: روى -> روا)
# ===========================================================================

# student marker "روى عنه" — MUST be tried before teacher, since "روا عن" is
# a prefix of "روا عنه".
STUDENT_MARKER = re.compile(r"و?روا عنه")
# teacher marker "روى عن" but NOT "روى عنه"
TEACHER_MARKER = re.compile(r"و?روا عن(?!ه)")

# an entry header: "N - " at a paragraph-ish position, followed by a name then
# a ":" — this is the reliable boundary in the Khoei book.
ENTRY_HEADER = re.compile(r"(?:^|\s)(\d+)\s*-\s*([^:،.]{2,60}?)\s*:")

# a cross-reference body "= اسم آخر" (X is the same as Y, no real biography)
REF_MARKER = re.compile(r":\s*=")

# where the source citation starts (ends the teacher/student region).
# Anything from here on is bibliographic, not a narrator name.
SOURCE_CUES = ["التهذيب", "الكافي", "الاستبصار", "الفقيه", "رجال الشيخ",
               "رجال الكشي", "رجال النجاشي", "الفهرست", "كامل الزيارات",
               "الروضه", "المستدرك", "الخصال", "اقول", "الجزء", "الحديث",
               "باب ", "الكتاب", "قال:", "قال ", "عده من اصحابنا",
               "رجل من اصحابنا", "رجل عن", "مرسلا", "الوسائل", "الوافي",
               "العلل", "المرءا", "ترجمه", "المهنا", "قد تقدم", "ياتي",
               "الطبعه", "النسخه", "نسخه"]

# markers that also END a name region (a new روى-clause starts)
CLAUSE_BREAK = re.compile(r"و?روا عن")


# ===========================================================================
# 3. Helpers to pull names out of a "روى عن X، وY، وZ" region
# ===========================================================================

def _split_names(region: str) -> list:
    """
    A teacher/student region is a comma/واو-separated list of names:
        'معاويه بن عمار، وابن ابي عمير، وصفوان'
    Split it into individual narrator name strings. The region ENDS at:
      - a source citation (التهذيب، الجزء، الحديث...), or
      - the next روى-clause (وروى عنه ... / وروى عن ...), or
      - a full stop.
    """
    # 1) cut at the first source citation or full stop
    cut = len(region)
    dot = region.find(".")
    if dot != -1:
        cut = min(cut, dot)
    for cue in SOURCE_CUES:
        i = region.find(cue)
        if i != -1:
            cut = min(cut, i)
    # 2) cut at the next روى-clause (so a teacher region doesn't swallow the
    #    following "وروى عنه ..." student clause, and vice-versa)
    nxt = CLAUSE_BREAK.search(region)
    if nxt:
        cut = min(cut, nxt.start())
    region = region[:cut]

    # split on Arabic comma and on leading واو before a name
    parts = re.split(r"[،,]|\sو(?=\S)", region)
    names = []
    for p in parts:
        name = _clean_name(p)
        if name:
            names.append(name)
    return names


def _clean_name(p: str) -> str:
    """
    Validate/clean one candidate narrator name. Returns "" if it doesn't look
    like a person name (leftover prose, a lone connector, too long, etc.).
    """
    p = p.strip(" .،,:؛()").strip()
    if not p:
        return ""
    # a leftover "عن ..." / "عنه ..." fragment: drop the leading particle
    p = re.sub(r"^عنه?\s+", "", p).strip()
    # drop a leading واو conjunction stuck to the name (ومحمد -> محمد)
    if p.startswith("و") and len(p) > 3:
        p = p[1:].strip()
    # any digit means a source citation leaked in (الجزء 4، الحديث 987)
    if any(ch.isdigit() for ch in p):
        return ""
    words = p.split()
    # too long to be a name -> it's a sentence fragment
    if len(words) > 6:
        return ""
    # leftover bibliographic/prose tokens that survived the region cut
    LEFTOVER = {"الكشي", "النجاشي", "الشيخ", "في", "ترجمه", "باب",
                "الجزء", "الحديث", "الكتاب", "رجال", "قال", "اقول",
                "المستدرك", "الطبعه", "النسخه", "المهنا", "عمده",
                "التهذيب", "الاستبصار", "الكافي", "الفقيه", "الفهرست",
                "التعليقه", "الوجيزه", "الخلاصه", "المجمع", "معجم"}
    if any(w in LEFTOVER for w in words):
        return ""
    # honorifics like "عليه السلام" attached to a name are fine, but a bare
    # honorific / connector alone is not a name
    if lexicons.is_rasoul_word(p) or lexicons.is_nmc_word(p) \
            or lexicons.is_nrc_word(p):
        return ""
    # a single word that is a stop-ish token (عن/عنه/رجل) is not a name
    if len(words) == 1 and (words[0] in {"عن", "عنه", "رجل", "غيره",
                                         "مثله", "جميعا", "بعضهم"}):
        return ""
    # strip a trailing honorific for a cleaner key, but keep the name
    return p


def _extract_teachers_students(body: str):
    """
    From an entry body, extract teacher list (روى عن ...) and student list
    (روى عنه ...). An entry can have several of each (multiple روى-clauses).
    """
    teachers, students = [], []

    # students first (marker is a superset of the teacher marker)
    for m in STUDENT_MARKER.finditer(body):
        region = body[m.end():]
        students.extend(_split_names(region[:120]))

    # teachers: matches of "روا عن" that are NOT "روا عنه"
    for m in TEACHER_MARKER.finditer(body):
        region = body[m.end():]
        teachers.extend(_split_names(region[:120]))

    # de-duplicate while keeping order
    def dedup(xs):
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return dedup(teachers), dedup(students)


def _extract_all_narrators(body: str):
    """
    OLD-STYLE (Biography::narrators, narrator_abstraction.h:360): collect EVERY
    narrator name in the entry body, in order, WITHOUT labelling teacher vs
    student. The old code gathered all names an entry mentions and let the
    narrator graph's generation ranks decide who is a professor (older) and
    who is a student (newer). We reproduce that: split the body on the
    "روى/عن/عنه/و" connectors and keep every plausible name.

    This is what makes the graph's contribution visible: the segmenter only
    provides an unordered bag of co-mentioned narrators; the graph supplies
    the teacher/student direction (the cross-document disambiguation).
    """
    names = []
    # break the body into candidate name spans on connectors and punctuation
    # (روا عن / روا عنه / و / ، / . ), then clean each into a name or drop it.
    parts = re.split(r"(?:و?روا عنه?|[،.:؛]|\bو\b)", body)
    for p in parts:
        nm = _clean_name(p.strip())
        if nm and len(nm) > 2:
            names.append(nm)
    # de-duplicate, keep order
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


# ===========================================================================
# 4. The segmenter
# ===========================================================================

@dataclass
class BioSegmenterConfig:
    # skip everything before the first sequential entry (front matter/muqaddima)
    # the Khoei entries become sequential around "آدم"/"أبان"; we detect the
    # first long run of consecutive numbers instead of hard-coding a marker.
    min_sequential_run: int = 5       # need N, N+1, ... this many in a row
    # an entry with < this many chars of body and no روى is treated as a
    # bare cross-reference
    min_body_chars: int = 15


class BiographySegmenter:

    def __init__(self, config: BioSegmenterConfig = None):
        self.config = config or BioSegmenterConfig()

    def segment(self, book_text: str):
        """
        Raw rijal book -> (entries, clean_text, stats).
        """
        clean_text, index_map = normalize(book_text)

        # 1) find every "N - name :" header with its position
        headers = []
        for m in ENTRY_HEADER.finditer(clean_text):
            num = int(m.group(1))
            subject = m.group(2).strip()
            headers.append((m.start(), m.end(), num, subject))

        # 2) keep only the region where numbers run sequentially (the real
        #    entries), dropping muqaddima list-items ("1 - ", "2 - " scattered).
        entry_headers = self._keep_sequential(headers)

        # 3) build entries: body = text from this header to the next entry
        entries = []
        for i, (h_start, h_end, num, subject) in enumerate(entry_headers):
            body_start = h_end
            body_end = (entry_headers[i + 1][0]
                        if i + 1 < len(entry_headers) else len(clean_text))
            body = clean_text[body_start:body_end]

            entry = BiographyEntry(number=num, subject=subject.strip(),
                                   start=h_start, end=body_end,
                                   raw=clean_text[h_start:body_end][:400])

            # cross-reference "= ..." with no real body?
            head_and_body = clean_text[h_end - 1:body_end]
            if REF_MARKER.search(clean_text[h_start:h_end + 5]) and \
               "روا عن" not in body:
                entry.is_reference = True
            else:
                entry.teachers, entry.students = \
                    _extract_teachers_students(body)
                # OLD-STYLE: also collect every narrator in the entry, letting
                # the graph decide teacher/student later (cross-document step).
                entry.all_narrators = _extract_all_narrators(body)

            entries.append(entry)

        stats = {
            "total_headers_found": len(headers),
            "sequential_entries": len(entry_headers),
            "entries_with_teachers": sum(1 for e in entries if e.teachers),
            "entries_with_students": sum(1 for e in entries if e.students),
            "entries_with_any_narrator": sum(1 for e in entries
                                             if e.all_narrators),
            "cross_references": sum(1 for e in entries if e.is_reference),
            "clean_chars": len(clean_text),
        }
        return entries, clean_text, stats

    def _keep_sequential(self, headers):
        """
        From all "N - name:" headers, keep the longest region where the
        numbers increase (the real narrator entries), not the muqaddima's
        scattered "1 - ", "2 - " list items that restart repeatedly.

        Rule: walk the headers; a header joins the current run if its number
        is >= previous kept number (entries only go up). A big DROP (num
        smaller than the run's max by a lot) starts a new candidate run.
        We return the single longest such run.
        """
        if not headers:
            return []

        runs = []
        current = [headers[0]]
        for h in headers[1:]:
            prev_num = current[-1][2]
            if h[2] >= prev_num:            # numbers keep increasing -> same run
                current.append(h)
            else:
                runs.append(current)
                current = [h]
        runs.append(current)

        # the real entries are the longest increasing run
        longest = max(runs, key=len)
        # only accept if it is genuinely a run (guards tiny books)
        if len(longest) < self.config.min_sequential_run:
            return headers            # fall back: treat all as entries
        return longest


# ---------------------------------------------------------------------------
# Quick demo:  python biography_segmenter.py khoei.txt
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path

    if len(sys.argv) < 2:
        sys.exit("usage: python biography_segmenter.py <rijal_book.txt>")

    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")

    seg = BiographySegmenter()
    entries, clean_text, stats = seg.segment(text)

    print(f"book: {path.name}")
    print("-" * 50)
    for k, v in stats.items():
        print(f"  {k:<24} {v:,}")
    print("-" * 50)

    # preview: first entries that actually have teachers/students
    rich = [e for e in entries if e.teachers or e.students]
    print(f"\nfirst {min(6, len(rich))} entries with teachers/students:\n")
    for e in rich[:6]:
        print(f"#{e.number}  {e.subject}")
        print(f"    teachers (روى عن) : {e.teachers}")
        print(f"    students (روى عنه): {e.students}")
        print()

    out = path.with_suffix(".entries.json")
    out.write_text(json.dumps([e.to_dict() for e in entries],
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out)
