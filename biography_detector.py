"""
biography_detector.py
=====================
IDEA (paper §4; old findChosenBiography + checkBiography,
narratordetector.cpp:268 / :665): decide WHERE each biography entry starts
and ends, using the hadith narrator graph rather than the book's typography.

THE METHOD, per graph node n:

    centers      = every position in the mention list where n's name occurs
    neighbours   = the k=1 graph neighbours of n
                   (old: BFS_traverse(cont,1,+1) then (cont,1,-1)
                    -> children = students, parents = teachers)
    for each center occurrence:
        score = how many DISTINCT neighbours have an occurrence textually
                NEAR that center (areNear = char-offset gap < bio_nrc_max)
    keep the best-scoring sets above bio_threshold
    THE BIOGRAPHY SPAN = [min start, max end] over the winning set

That span is the boundary. It is derived from the graph — the entry extends
exactly as far as narrators the graph confirms belong together — which is the
41% -> 93% boundary-recall claim in Table 2.

--------------------------------------------------------------------------
TWO FINDINGS ABOUT THE ORIGINAL, both deliberate deviations here:

1. `updateGroups` (narratordetector.cpp:158) does not do what it intends.
   Trace it on a descending-sorted list: a set scoring HIGHER than an
   existing one sets `add = true`, the `else if (add)` branch is then never
   reached, nothing is inserted, and the better set is silently DISCARDED.
   Only sets scoring worse than everything already present are appended.
   We implement the evident intent: keep the top-N by score.

2. `getBoundingParagraph` / `isInsideBound` (narratordetector.cpp:213-266)
   are DEAD CODE — defined, never called. So the paragraph delimiters that
   `additionalCheck` collects never bound anything. We keep collecting them
   (they are useful for evaluation) but do not invent a use for them.
--------------------------------------------------------------------------
"""

from bisect import bisect_left
from dataclasses import dataclass, field

from kreachable import ReachabilityIndex
from name_hash import primary_key


# ===========================================================================
# 1. Parameters — old hadithParameters bio_* fields
# ===========================================================================

@dataclass
class BiographyParams:
    near_max_chars: int = 100   # bio_nrc_max, used as the areNear window
    threshold: float = 5.0      # bio_threshold — min neighbours confirmed
    max_candidates: int = 3     # old MAX_SIZE in updateGroups
    k: int = 1                  # k-reachable radius; the old code hardcodes 1
    skip_ambiguous_first_name: bool = True   # old isFirstNameAmbiguous

    # OUR IMPROVEMENT (paper §4, NOT in the released system — see
    # _disambiguate_sequentially). Off by default so the faithful baseline
    # stays clean and the gain is separately measurable.
    sequential_disambiguation: bool = False


# ===========================================================================
# 2. A detected biography
# ===========================================================================

@dataclass
class DetectedBiography:
    person_id: int
    subject_name: str
    start: int                  # char span in the clean text = THE BOUNDARY
    end: int
    score: int                  # neighbours the graph confirmed nearby
    mention_indices: list = field(default_factory=list)
    confirmed_neighbours: list = field(default_factory=list)

    def overlaps(self, other) -> bool:
        return self.start <= other.end and other.start <= self.end

    def to_dict(self):
        return {"person_id": self.person_id, "subject": self.subject_name,
                "start": self.start, "end": self.end, "score": self.score,
                "mentions": list(self.mention_indices),
                "confirmed_neighbours": list(self.confirmed_neighbours)}


# ===========================================================================
# 3. The detector
# ===========================================================================

class BiographyDetector:
    """
    mentions : list[NarratorMention] from BiographyFSM (the old narratorList).
               Each carries .matches — the graded graph persons it may be.
    graph    : a narrator_matcher.GraphIndex
    """

    def __init__(self, mentions, graph, params: BiographyParams = None):
        self.mentions = mentions
        self.graph = graph
        self.params = params or BiographyParams()
        self.reach = ReachabilityIndex(graph.persons, undirected=True)
        self.disambiguation_stats = {"ambiguous": 0, "narrowed": 0,
                                     "no_survivor": 0}
        self._resolved = self._resolve_matches()
        self._positions = self._build_position_index()

    # --------------------------------------- sequential disambiguation (A7)
    def _resolve_matches(self):
        """
        Decide, per mention, WHICH graph persons it may refer to.

        Paper §4 states two ambiguity measures:
            "First ANGE uses a threshold against the hash score to limit the
             matches. Also, ANGE only considers matches of n_i that are
             within k steps of the matches of narrator n_{i-1} preceding
             n_i in lb."

        Only the FIRST is in the released system (LocateInGraphAction keeps
        the highest-similarity nodes). We verified the second is absent:
        narratordetector.cpp has no sequential/previous-narrator constraint
        anywhere. It joins `getdistance`'s nisba logic and the unread
        `bio_max_reachability` as a paper claim the implementation never
        made — so this is OUR contribution, not a port, and it is off by
        default.

        The idea: consecutive narrators in a rijal entry are connected in the
        hadith graph, so a name with several candidates should keep only the
        candidates reachable from the previous narrator's. At a real entry
        boundary NOTHING is reachable (the previous narrator belongs to the
        previous entry) — there we keep the unfiltered set, because the
        constraint is meant to disambiguate within a run, not to delete
        narrators at boundaries.
        """
        resolved = [list(m.matches) for m in self.mentions]
        if not self.params.sequential_disambiguation:
            return resolved

        previous_ids = []
        for i, matches in enumerate(resolved):
            if not matches:
                continue
            if len(matches) > 1 and previous_ids:
                self.disambiguation_stats["ambiguous"] += 1
                kept = [m for m in matches
                        if self.reach.any_reachable([m.person_id],
                                                    previous_ids,
                                                    k=self.params.k)]
                if kept:
                    if len(kept) < len(matches):
                        self.disambiguation_stats["narrowed"] += 1
                    resolved[i] = kept
                else:
                    # entry boundary: keep everything rather than lose the
                    # narrator entirely
                    self.disambiguation_stats["no_survivor"] += 1
            previous_ids = [m.person_id for m in resolved[i]]
        return resolved

    # ------------------------------------------------- the position index
    def _build_position_index(self):
        """
        person_id -> sorted list of mention indices that matched that person.
        Old getSortedIndexList (narratordetector.cpp:76), which walked the
        hash to find every biography position a graph node occurs at.
        """
        index = {}
        for i, matches in enumerate(self._resolved):
            for match in matches:
                index.setdefault(match.person_id, []).append(i)
        for positions in index.values():
            positions.sort()
        return index

    # ------------------------------------------------- areNear
    def _are_near(self, i: int, j: int) -> bool:
        """
        Old areNear (narratordetector.cpp:207): the two mentions sit within
        bio_nrc_max CHARACTERS of each other, measured end-to-start in either
        direction.
        """
        a, b = self.mentions[i], self.mentions[j]
        limit = self.params.near_max_chars
        return (abs(a.end - b.start) < limit) or (abs(a.start - b.end) < limit)

    # ------------------------------------------------- neighbours
    def _neighbours_of(self, person_id):
        """
        k-reachable neighbours. The old code hardcodes k=1 in both directions
        (narratordetector.cpp:284-285) — its `bio_max_reachability` parameter
        was wired to the GUI but never read. We route through the k-reachable
        primitive so k is a real parameter; k=1 reproduces the original.
        """
        return self.reach.reachable_within(person_id, k=self.params.k)

    # ------------------------------------------------- findChosenBiography
    def candidates_for(self, person_id):
        """
        Port of findChosenBiography. Returns up to `max_candidates` scored
        DetectedBiography objects for this graph person, best first.
        """
        centers = self._positions.get(person_id, ())
        if not centers:
            return []
        neighbours = self._neighbours_of(person_id)
        if not neighbours:
            return []

        results = []
        for center in centers:
            chosen = {center}
            confirmed = []
            score = 0

            for neighbour_id in neighbours:
                positions = self._positions.get(neighbour_id)
                if not positions:
                    continue
                # the old code inspects the two CLOSEST occurrences of this
                # neighbour: the one at/after the center, then the one before
                # (qLowerBound, then itr--). First hit that is near and not
                # already claimed wins.
                at = bisect_left(positions, center)
                for cand in (positions[at] if at < len(positions) else None,
                             positions[at - 1] if at > 0 else None):
                    if cand is None or cand in chosen:
                        continue
                    if self._are_near(center, cand):
                        chosen.add(cand)
                        confirmed.append(self.mentions[cand].name)
                        score += 1
                        break

            if score >= self.params.threshold:
                indices = sorted(chosen)
                results.append(DetectedBiography(
                    person_id=person_id,
                    subject_name=self.mentions[center].name,
                    start=min(self.mentions[i].start for i in indices),
                    end=max(self.mentions[i].end for i in indices),
                    score=score,
                    mention_indices=indices,
                    confirmed_neighbours=confirmed))

        # keep the top-N by score. See the module docstring: the original
        # updateGroups discards the BETTER set instead of the worse one.
        results.sort(key=lambda b: -b.score)
        return results[: self.params.max_candidates]

    # ------------------------------------------------- subjects to consider
    def subject_ids(self):
        """
        Which graph persons are worth testing as biography subjects.

        Old modifyNodes (narratordetector.cpp:633) walks the narrator list,
        de-duplicates by key, and SKIPS keys whose first name is unknown
        (isFirstNameAmbiguous — the key begins with '-', e.g. 'ابن ابي عمير').
        Such a name cannot identify the subject of an entry.

        (The old code also skipped every key alphabetically after a hardcoded
        constant, `biographyTagMax` — a leftover from a partial annotation
        run, not a method. Not reproduced.)
        """
        seen, out = set(), []
        for i, mention in enumerate(self.mentions):
            if self.params.skip_ambiguous_first_name:
                if primary_key(_canon_of(mention)).startswith("-"):
                    continue
            for match in self._resolved[i]:
                if match.person_id not in seen:
                    seen.add(match.person_id)
                    out.append(match.person_id)
        return out

    # ------------------------------------------------- checkBiography
    def detect(self):
        """
        Full-book detection. Scores every candidate subject, then accepts
        them best-first, rejecting any whose span overlaps one already
        accepted (old checkBiography, narratordetector.cpp:665 — which also
        keeps only the top-1 set per subject).
        """
        all_candidates = []
        for person_id in self.subject_ids():
            best = self.candidates_for(person_id)
            if best:
                all_candidates.append(best[0])      # old: keeps list[0] only

        all_candidates.sort(key=lambda b: (-b.score, b.start))

        accepted = []
        for cand in all_candidates:
            if any(cand.overlaps(a) for a in accepted):
                continue
            accepted.append(cand)
        accepted.sort(key=lambda b: b.start)
        return accepted

    def stats(self, detected):
        return {
            "mentions": len(self.mentions),
            "mentions_matched": sum(1 for m in self.mentions if m.matches),
            "mentions_ambiguous": sum(1 for r in self._resolved if len(r) > 1),
            "sequential_disambiguation": self.params.sequential_disambiguation,
            **{f"disamb_{k}": v
               for k, v in self.disambiguation_stats.items()},
            "candidate_subjects": len(self.subject_ids()),
            "biographies_detected": len(detected),
            "avg_score": (round(sum(b.score for b in detected) / len(detected), 2)
                          if detected else 0),
            "avg_span_chars": (round(sum(b.end - b.start for b in detected)
                                     / len(detected)) if detected else 0),
        }


def _canon_of(mention):
    """CanonicalName of a mention's narrator (cached on the object)."""
    cached = getattr(mention, "_canon", None)
    if cached is None:
        from name_model import canonize
        cached = canonize(mention.narrator)
        try:
            mention._canon = cached
        except AttributeError:
            pass
    return cached


# ---------------------------------------------------------------------------
# Quick demo:  python biography_detector.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(__doc__.strip().splitlines()[1])
    print()
    print("Run the test suite for a worked example:")
    print("    py -3.11 tests/test_a3.py")
