"""
spans.py
========
Interval and word-counting primitives for the evaluation scorer.
Direct port of src/util/text_handling.h:407-537.

Everything downstream (agreement.py) is built from these five functions, so
they are kept separate and exhaustively tested — an off-by-one here would
silently shift every number in Tables 1-3.
"""

# characters that separate words (the old isDelimiter, letters.h)
DELIMITERS = set(" \t\r\n ") | set(":،,.؛;؟?!()[]{}-*–—/\"'")


# ===========================================================================
# 1. Interval predicates (text_handling.h:407-449)
# ===========================================================================

def overlaps(start1, end1, start2, end2) -> bool:
    """Do [start1,end1] and [start2,end2] share at least one position?"""
    return (start2 <= start1 <= end2) or (start1 <= start2 <= end1)


def after(start1, end1, start2, end2) -> bool:
    """Is the first interval after the second? (old: start1 >= end2)"""
    return start1 >= end2


def before(start1, end1, start2, end2) -> bool:
    return after(start2, end2, start1, end1)


# ===========================================================================
# 2. Word counting (text_handling.h:459-537)
# ===========================================================================

def _next_word_start(text, pos):
    """
    From `pos` (inside a word), skip to the end of that word and return the
    start of the next one. None if there is no next word.
    Old: next_positon(text, getLastLetter_IN_currentWord(text, pos)).
    """
    n = len(text)
    while pos < n and text[pos] not in DELIMITERS:      # to end of this word
        pos += 1
    while pos < n and text[pos] in DELIMITERS:          # over the separators
        pos += 1
    return pos if pos < n else None


def count_words(text, start, end) -> int:
    """
    Number of words in text[start..end].

    Faithful to the old countWords: `if (start >= end) return 0`, then
    count = 1 and walk forward while the next word STARTS before `end`.
    A degenerate one-character span therefore counts 0 words, which is the
    original's behaviour and matters for spans built from bad offsets.
    """
    if start is None or end is None or start >= end:
        return 0
    count = 1
    pos = start
    while True:
        pos = _next_word_start(text, pos)
        if pos is None or pos >= end:
            return count
        count += 1


def count_words_spans(text, spans) -> int:
    """Total words over a list of (start, end) spans."""
    return sum(count_words(text, s, e) for s, e in spans)


def common_words(text, span1, span2) -> int:
    """Words inside the INTERSECTION of two spans (text_handling.h:497)."""
    start = max(span1[0], span2[0])
    end = min(span1[1], span2[1])
    return count_words(text, start, end)


def common_words_spans(text, spans1, spans2) -> int:
    """
    Words common to two span LISTS.

    Same all-pairs sum as the original, which carries this caveat in a
    comment (text_handling.h:521): it assumes the lists are sane, i.e. no
    span in one list straddles two spans of the other. Overlapping input
    would double-count.
    """
    return sum(common_words(text, a, b) for a in spans1 for b in spans2)
