"""
equality.py
===========
IDEA (distance, paper §3.2 "distance(n1,n2)"):
Given two canonized narrators, return a score in [0,1]:
    1.0 = definitely the same person
    0.0 = definitely different
Used by the graph builder to decide whether to MERGE two narrator nodes.

Old source (verified): equalNew() / preProcessForEquality tail
(narrator_abstraction.cpp:602 and .h:265). Exact algorithm:

    1. relative narrator (ابيه) -> 0   (identity unknown until graph resolves)
    2. rasoul -> 1 iff both are rasoul, else 0
    3. identical full strings -> 1
    4. canonize both (name_model.py) into levels
    5. compare level by level, up to the shorter one:
         - for each name word pair at the same position:
              if they differ (after alef-norm) -> return 0   (HARD reject)
         - if both levels equal (or both empty) -> this level counts
    6. score = matching_levels / max(num_levels)

TWO DOCUMENTED FACTS (from reading the old code — important for the report):
  (a) score is levelsEqual / levelMax, NOT the paper's 1 - count/(...).
  (b) the code comment says: "till now possessives not even compared".
      The paper describes conflicting/reinforcing nisba logic
      (الكوفي vs المصري); it is NOT implemented in the old system.
      -> qualifiers_mode="off" reproduces the old behavior exactly.
      -> qualifiers_mode="on" is OUR improvement (paper §3.2 fully),
         benchmarkable against the baseline.

Also provided: edit_distance_score() — the "eNarrator" related-work
baseline (Levenshtein, paper's comparison target, old threshold 0.75).
"""

from name_model import canonize, CanonicalName


# ===========================================================================
# 1. The main metric (equalNew port)
# ===========================================================================

def narrator_distance(c1: CanonicalName, c2: CanonicalName,
                      qualifiers_mode: str = "off",
                      metric_mode: str = "structural",
                      embedder=None) -> float:
    """
    Score two CanonicalNames in [0,1]. qualifiers_mode:
        "off" -> exact old-system behavior (ignore nisba)  [default/baseline]
        "on"  -> also use qualifiers (our improvement, see _qualifier_factor)
    metric_mode:
        "structural" -> hierarchical metric only (old system) [default]
        "hybrid"     -> structural, then BOOST with name embeddings
                        (AraBERT) when an embedder is supplied. Embeddings
                        NEVER override a hard reject (score 0 stays 0) — they
                        only nudge an already-plausible match. This is OUR
                        addition (not in the paper/old code).
    embedder: a NameEmbeddings instance (or None). Only used in "hybrid".
    """
    # 1. relative narrator: identity not yet known
    if c1.is_relative or c2.is_relative:
        return 0.0

    # 2. rasoul: same only if both are the Prophet/Imam
    if c1.is_rasoul or c2.is_rasoul:
        return 1.0 if (c1.is_rasoul and c2.is_rasoul) else 0.0

    # 3. exact same raw name -> identical
    if c1.raw and c1.raw == c2.raw:
        return 1.0

    # 4-6. level-by-level comparison
    levels1, levels2 = c1.levels, c2.levels
    level_min = min(len(levels1), len(levels2))
    level_max = max(len(levels1), len(levels2))
    if level_max == 0:
        return 0.0

    levels_equal = 0
    for i in range(level_min):
        words1, words2 = levels1[i], levels2[i]
        name_min = min(len(words1), len(words2))

        # compare overlapping name words at this level
        differed = False
        compared_any = False
        for j in range(name_min):
            compared_any = True
            if words1[j] != words2[j]:           # already alef-normalized
                return 0.0                       # HARD reject (paper's ∞)
        # both empty at this level counts as equal (old code)
        if len(words1) == 0 and len(words2) == 0:
            compared_any = True

        if compared_any and not differed:
            levels_equal += 1

    score = levels_equal / level_max

    # OUR improvement (optional): let qualifiers adjust the score
    if qualifiers_mode == "on":
        score *= _qualifier_factor(c1.qualifiers, c2.qualifiers)

    # OUR improvement (optional): embedding BOOST. Only when the structural
    # metric already sees a plausible (non-zero) match. A hard reject above
    # already returned 0, so embeddings can never fuse different people.
    if metric_mode == "hybrid" and embedder is not None and score > 0.0:
        score = _embedding_boost(score, c1.raw, c2.raw, embedder)

    return score


def _embedding_boost(structural_score: float, name1: str, name2: str,
                     embedder) -> float:
    """
    Blend the structural score with embedding similarity. Conservative:
      - embeddings only RAISE the score (spelling variants of the same name
        push it up), never lower it below the structural value;
      - capped at 1.0.
    If embeddings are unavailable, returns the structural score unchanged.
    """
    sim = embedder.similarity(name1, name2)
    if sim is None:                       # offline / unavailable
        return structural_score
    # weighted blend: keep structural as the floor, let a high embedding
    # similarity add up to a small bonus (0.3 weight).
    boosted = structural_score + 0.3 * sim * (1.0 - structural_score)
    return max(structural_score, min(1.0, boosted))


def narrators_equal(c1, c2, threshold: float = 0.1,
                    qualifiers_mode: str = "off") -> bool:
    """Boolean merge decision (old equality_threshold = 0.1)."""
    return narrator_distance(c1, c2, qualifiers_mode) > threshold


# ===========================================================================
# 2. Qualifier logic — OUR improvement (paper §3.2, unimplemented in old code)
#    conflicting nisba (الكوفي vs المصري) -> strong penalty
#    reinforcing nisba (shared)           -> small boost
# ===========================================================================

def _qualifier_factor(q1: list, q2: list) -> float:
    """
    Returns a multiplier in (0,1]. If both have qualifiers and they fully
    conflict (no overlap), penalize; if they share, keep full score.
    Conservative: absence of qualifiers on either side = no change (1.0).
    """
    if not q1 or not q2:
        return 1.0                     # can't compare -> don't penalize
    shared = set(q1) & set(q2)
    if shared:
        return 1.0                     # reinforcing -> full score
    return 0.5                         # conflicting -> halve the score


# ===========================================================================
# 3. Baseline: edit-distance ("eNarrator", the related-work comparison)
#    paper argues their metric beats plain Levenshtein; old threshold 0.75
# ===========================================================================

def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edit_distance_score(name1: str, name2: str) -> float:
    """
    Normalized similarity in [0,1] (1 = identical). This is the BASELINE
    metric the paper compares against; used in the evaluation ablation.
    """
    if not name1 and not name2:
        return 1.0
    dist = _levenshtein(name1, name2)
    return 1.0 - dist / max(len(name1), len(name2))


# ---------------------------------------------------------------------------
# Quick demo:  python equality.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from name_model import canonize_string

    pairs = [
        # (name A, name B, should they be the same person?)
        ("محمد بن يعقوب", "محمد بن يعقوب الكليني", "YES (nisba added)"),
        ("محمد بن يعقوب", "محمد بن يعقوب", "YES (identical)"),
        ("علي بن ابراهيم", "علي بن ابراهيم بن هاشم", "YES (grandfather added)"),
        ("محمد بن يعقوب", "محمد بن الحسن", "NO (different father)"),
        ("احمد بن محمد", "احمد بن علي", "NO (different father)"),
        ("عماره بن القعقاع", "عماره بن القعقاع بن شبرمه", "YES"),
    ]

    print(f"{'name A':<22}{'name B':<28}{'our':>6}{'edit':>7}  verdict")
    print("-" * 78)
    for a, b, expected in pairs:
        ca, cb = canonize_string(a), canonize_string(b)
        our = narrator_distance(ca, cb)
        edit = edit_distance_score(a, b)
        merge = "MERGE" if our > 0.1 else "keep separate"
        print(f"{a:<22}{b:<28}{our:>6.2f}{edit:>7.2f}  {merge:<14} [{expected}]")

    print("\nNote: 'our' hard-rejects (0.00) when a name level conflicts, "
          "while 'edit' can give a misleadingly high score to different "
          "people with similar spelling — this is the paper's whole point.")
