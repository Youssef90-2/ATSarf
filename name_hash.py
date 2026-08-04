"""
name_hash.py
============
IDEA (multi-key hash, paper §3.2 + Figure 4):
Comparing every narrator against every other is O(n²) — with ~90,000
narrators that's billions of comparisons. Impossible.

Solution: index each narrator under SEVERAL keys — one for every sub-name
that could refer to the same person. Then to find candidates for a
narrator, we just look up its keys in the hash (O(1) each) and only run
the real equality test on the few narrators that share a key.

Example (Figure 4) — عماره بن القعقاع الكوفي generates keys like:
    عماره-القعقاع:الكوفي   (full,        score 7/7)
    عماره-القعقاع          (no nisba,    score 3/7)
    القعقاع الكوفي         (skip first,  score 3/7)
    القعقاع                (minimal,     score 1/7)
    ...
Each key carries a RELEVANCE SCORE (value/total) = how complete this
sub-name is. A full-name match is trusted more than a bare-surname match.

Old source (verified): NarratorHash (narratorHash.h)
    - getKey(): builds a canonical key string; levels joined by "-",
      words in a level by " ", AB/OM literals for kunya, nisba after ":",
      missing leading levels become "-".
    - generateAllPossibilities(): enumerates keys for
        level = full .. minimal,  {with/without nisba} x {skip first or not}
      with the exact value/total scoring from the code.

--------------------------------------------------------------------------
This file provides:
    generate_keys(canonical) -> list of (key_string, score) for a narrator
    NameHash                 -> dict-based index: add(id, canonical) /
                                find_candidates(canonical) -> {id: score}
--------------------------------------------------------------------------
"""

from name_model import canonize, CanonicalName


# ===========================================================================
# 1. Build ONE key string for a given number of levels + options
#    (port of getKey: narratorHash.h:66)
# ===========================================================================

def _build_key(levels, qualifiers, size, with_nisba, skip_first) -> str:
    """
    levels     : list[list[str]]  (from CanonicalName)
    size       : how many leading levels to include
    with_nisba : append the qualifiers after ":"
    skip_first : drop level 0 (the person's own first name)
    Missing/leading skipped levels appear as "-", exactly like the old code.
    """
    parts = []
    for i in range(size):
        if skip_first and i == 0:
            level_str = ""                 # skipped -> becomes "-"
        elif i < len(levels):
            # key_token writes "AB"/"OM" for kunya connectors, the plain text
            # otherwise — exactly getKey() (narratorHash.h:78-95)
            level_str = " ".join(it.key_token for it in levels[i])
        else:
            level_str = ""
        parts.append(level_str)

    key = "-".join(parts)

    if with_nisba and qualifiers:
        key += ":" + " ".join(qualifiers)
    return key


# ===========================================================================
# 2. Enumerate ALL keys for a narrator, each with its relevance score
#    (port of generateAllPossibilities: narratorHash.h:259)
# ===========================================================================

def primary_key(c: CanonicalName) -> str:
    """
    The EXACT key of a narrator — all levels, with nisba, nothing skipped.
    Old: Narrator::getKey() (narrator_abstraction.cpp:680), which is
    getKey(names, possessives, names.size(), true, false).

    Two occurrences sharing this key are the same written name, so they form
    one atomic GroupNode.
    """
    if c.is_relative:
        return "<REL>"
    if c.is_rasoul:
        return "<RASOUL>"
    return _build_key(c.levels, c.qualifiers, len(c.levels), True, False)


def generate_keys(c: CanonicalName):
    """
    Return list of (key_string, score) where score in (0,1].
    Relative narrators and rasoul are handled specially (single key).
    """
    if c.is_relative:
        return []                          # unknown identity -> no keys
    if c.is_rasoul:
        return [("<RASOUL>", 1.0)]

    levels = c.levels
    quals = c.qualifiers
    level_size = len(levels)
    if level_size == 0:
        return []

    has_nisba = len(quals) > 0
    level_one_empty = len(levels[0]) == 0

    # 'total' = the maximum possible score (a fully-specified name),
    # replicating the old code's arithmetic so scores are comparable.
    total = ((level_size - 1) * 2 if has_nisba else level_size - 1)
    if not level_one_empty:
        total += (level_size * 2 if has_nisba else level_size)
    total = max(total, 1)                  # guard against divide-by-zero

    min_level = 2 if level_one_empty else 1
    keys = {}

    def add(key, value):
        score = value / total
        # keep the HIGHEST score if the same key arises twice
        if key not in keys or score > keys[key]:
            keys[key] = score

    for level in range(level_size, min_level - 1, -1):
        base = (level - min_level + 1) * (2 if not level_one_empty else 1) \
               + (min_level - 2)

        add(_build_key(levels, quals, level, False, False), base)
        if has_nisba:
            add(_build_key(levels, quals, level, True, False), base * 2)

        if level > 1 and not level_one_empty:
            v = level - 1
            add(_build_key(levels, quals, level, False, True), v)
            if has_nisba:
                add(_build_key(levels, quals, level, True, True), v * 2)

    return list(keys.items())


# ===========================================================================
# 3. The hash index — add narrators, find candidates
# ===========================================================================

class NameHash:
    """
    Maps key_string -> list of (narrator_id, score).
    add(id, canonical)          : index a narrator under all its keys
    find_candidates(canonical)  : {narrator_id: best_score} of narrators
                                  sharing at least one key
    """

    def __init__(self):
        self.table = {}                    # key -> list[(id, score)]

    def add(self, narrator_id, canonical: CanonicalName):
        for key, score in generate_keys(canonical):
            self.table.setdefault(key, []).append((narrator_id, score))

    def find_candidates(self, canonical: CanonicalName) -> dict:
        """
        Return {narrator_id: best_score} for every narrator that shares a
        key with `canonical`. The score is the max over shared keys — a
        proxy for "how strongly do these two look like the same person".
        """
        candidates = {}
        for key, _score in generate_keys(canonical):
            for (nid, stored_score) in self.table.get(key, []):
                # combined confidence = min of the two sub-name scores
                # (both sides must be reasonably complete to trust it)
                conf = min(_score, stored_score)
                if nid not in candidates or conf > candidates[nid]:
                    candidates[nid] = conf
        return candidates

    def __len__(self):
        return len(self.table)


# ---------------------------------------------------------------------------
# Quick demo:  python name_hash.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from name_model import canonize_string

    print("=== keys generated for one narrator ===")
    c = canonize_string("عماره بن القعقاع الكوفي")
    for key, score in sorted(generate_keys(c), key=lambda x: -x[1]):
        print(f"   score {score:.2f}   key: {key}")

    print("\n=== blocking demo: who are candidates for 'محمد بن يعقوب'? ===")
    hash_index = NameHash()
    corpus = [
        "محمد بن يعقوب الكليني",     # same person, +nisba
        "محمد بن يعقوب",             # same person
        "محمد بن الحسن",             # different (father الحسن)
        "علي بن ابراهيم",            # totally different
        "احمد بن محمد بن عيسا",      # different
    ]
    for i, name in enumerate(corpus):
        hash_index.add(i, canonize_string(name))

    query = canonize_string("محمد بن يعقوب")
    candidates = hash_index.find_candidates(query)
    print("query: محمد بن يعقوب")
    for nid, score in sorted(candidates.items(), key=lambda x: -x[1]):
        print(f"   candidate [{corpus[nid]}]  shared-key score {score:.2f}")
    print("\nNote: only names SHARING a key show up — 'علي بن ابراهيم' is "
          "never even compared. That's the O(n²) -> near-O(n) win.")
