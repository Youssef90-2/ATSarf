"""
boundary_gold.py
================
The GOLD boundaries for the Table-2 experiment.

The paper scored narrator/boundary detection against a human annotation we
don't have. The Khoei rijal book gives us a reliable substitute: every entry
starts with a sequential "N - name:" header, so the true span of entry N runs
from its header to the next entry's header. We extract those spans and treat
them as ground truth — used ONLY to SCORE a method, never fed into detection.

This is a thin wrapper over biography_segmenter (which already finds the
"N -" headers and the entry's listed teachers/students). Splitting it into
its own module keeps the three ideas separate:
    boundary_gold.py  -> the truth        (this file)
    kreachable.py     -> the graph tool   (paper's k-reachable primitive)
    boundary_eval.py  -> the experiment   (uses both)

A GoldEntry is what the evaluator scores against:
    number     entry number
    subject    the narrator the entry is about
    start,end  char span in the clean text (the true boundary)
    teachers,students  names the book lists (used to locate graph neighbours)
"""

from dataclasses import dataclass, field

from biography_segmenter import BiographySegmenter, BioSegmenterConfig


@dataclass
class GoldEntry:
    number: int
    subject: str
    start: int
    end: int
    teachers: list = field(default_factory=list)
    students: list = field(default_factory=list)

    @property
    def neighbours(self):
        """All narrators the book links to this subject (teachers+students)."""
        return list(self.teachers) + list(self.students)


def load_gold(book_text: str, config: BioSegmenterConfig = None):
    """
    Raw rijal book text -> (gold_entries, clean_text).

    gold_entries: list[GoldEntry] with true spans, in text order. Cross-
    reference stubs ("X: = Y") are dropped — they have no real boundary.
    """
    seg = BiographySegmenter(config)
    entries, clean_text, _stats = seg.segment(book_text)

    gold = []
    for e in entries:
        if e.is_reference:
            continue
        gold.append(GoldEntry(
            number=e.number, subject=e.subject,
            start=e.start, end=e.end,
            teachers=list(e.teachers), students=list(e.students),
        ))
    return gold, clean_text


def gold_stats(gold):
    return {
        "gold_entries": len(gold),
        "with_neighbours": sum(1 for g in gold if g.neighbours),
        "total_neighbours": sum(len(g.neighbours) for g in gold),
        "avg_span_chars": (round(sum(g.end - g.start for g in gold) / len(gold))
                           if gold else 0),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: python boundary_gold.py <rijal_book.txt>")
    text = open(sys.argv[1], encoding="utf-8").read()
    gold, _clean = load_gold(text)
    stats = gold_stats(gold)
    print("gold boundaries from '%s':" % sys.argv[1])
    for k, v in stats.items():
        print(f"  {k:<20} {v:,}")
    print("\nfirst 5 gold spans:")
    for g in gold[:5]:
        print(f"  #{g.number} [{g.start}-{g.end}] {g.subject}  "
              f"(neighbours: {len(g.neighbours)})")
