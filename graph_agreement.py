"""
graph_agreement.py
==================
Table 1 — narrator-graph accuracy.

This scores something different from agreement.py. That file scores SPANS:
did we mark the right stretch of text. This one scores MERGE DECISIONS: given
the same narrator occurrences, did we group them into the same people.

Port of HadithInterAnnotatorAgreement::compareGlobalGraphs
(hadithInterAnnotatorAgreement.cpp:210), which is the paper's own wording:

    "For each narrator node n, we computed the ratio of the correctly merged
     nodes against the number of nodes with which n should be merged. The
     precision metric denotes the average of the ratio of correctly merged
     nodes against the number of merged nodes for each narrator node."

Per occurrence o:

    gold_cluster      the occurrences the annotator says are the same person
    pred_cluster      the occurrences our graph merged with o
    common            |gold_cluster ∩ pred_cluster| - 1
    recall(o)         common / (|gold_cluster| - 1)    ... 1 if o is a singleton
    precision(o)      common / (|pred_cluster| - 1)    ... 1 if o is a singleton

averaged over occurrences. The `- 1` excludes o itself, which is what the C++
does (`getNumChainNodes(...) - 1`).

This is B-Cubed with the element itself excluded — worth naming in the report,
because it makes the metric recognisable and lets you cite its properties
instead of defending a bespoke formula.

--------------------------------------------------------------------------
THE `detected` VS `all` ROWS OF TABLE 1

    detected  only occurrences present in BOTH mappings — the accuracy of the
              merge logic in isolation, excluding narrators the segmenter
              never found
    all       every gold occurrence; a narrator we failed to extract counts
              as a cluster of one, so its merges are all missed

The paper reports 0.673/0.824 for "detected" and 0.632/0.771 for "all".
--------------------------------------------------------------------------
INPUT is two mappings from the SAME occurrence keys to cluster labels:

    gold = {"h12#0": "ahmad_isa", "h12#1": "muhammad_yahya", ...}
    pred = {"h12#0": 41,          "h12#1": 7,               ...}

Label VALUES are opaque and need not match between the two — only the
partitions they induce are compared.
"""

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class MergeScore:
    recall: float
    precision: float
    f_score: float
    occurrences: int
    gold_clusters: int
    predicted_clusters: int

    def to_dict(self):
        return {"recall": round(self.recall, 4),
                "precision": round(self.precision, 4),
                "f_score": round(self.f_score, 4),
                "occurrences": self.occurrences,
                "gold_clusters": self.gold_clusters,
                "predicted_clusters": self.predicted_clusters}


def _invert(labels):
    """label -> set of occurrence keys carrying it."""
    clusters = defaultdict(set)
    for key, label in labels.items():
        clusters[label].add(key)
    return clusters


def score_merges(gold_labels, pred_labels, restrict_to_detected=True):
    """
    gold_labels / pred_labels : {occurrence_key: cluster_label}

    restrict_to_detected=True  -> Table 1's "detected" row (keys in both)
    restrict_to_detected=False -> Table 1's "all" row: every gold occurrence,
                                  with anything we never extracted treated as
                                  a singleton cluster of its own.
    """
    if restrict_to_detected:
        keys = set(gold_labels) & set(pred_labels)
    else:
        keys = set(gold_labels)

    if not keys:
        return MergeScore(0.0, 0.0, 0.0, 0, 0, 0)

    gold_clusters = _invert({k: gold_labels[k] for k in keys})
    # an occurrence we never produced forms its own singleton, so it can share
    # a cluster with nothing — exactly how a missed narrator should score
    pred_view = {k: pred_labels.get(k, ("__missing__", k)) for k in keys}
    pred_clusters = _invert(pred_view)

    recall_sum = precision_sum = 0.0
    for key in keys:
        gold_c = gold_clusters[gold_labels[key]]
        pred_c = pred_clusters[pred_view[key]]
        common = len(gold_c & pred_c) - 1          # exclude the element itself
        correct = len(gold_c) - 1
        detected = len(pred_c) - 1
        # a singleton has nothing to merge, so it is trivially perfect — the
        # C++ appends 1 in that case rather than skipping it
        recall_sum += (common / correct) if correct > 0 else 1.0
        precision_sum += (common / detected) if detected > 0 else 1.0

    n = len(keys)
    recall = recall_sum / n
    precision = precision_sum / n
    f = (2 * recall * precision / (recall + precision)
         if (recall + precision) else 0.0)
    return MergeScore(recall, precision, f, n,
                      len(gold_clusters), len(pred_clusters))


def score_table1(gold_labels, pred_labels):
    """Both rows of Table 1."""
    return {
        "detected": score_merges(gold_labels, pred_labels, True).to_dict(),
        "all": score_merges(gold_labels, pred_labels, False).to_dict(),
    }


# ===========================================================================
# Producing the predicted labels from a built graph
# ===========================================================================

def labels_from_graph(persons, chain_nodes, key_fn=None):
    """
    {occurrence_key: person_id} from a built NarratorGraph.

    The default key is "chain#position", which is stable across runs and is
    what a human annotator can also address ("the 3rd narrator of hadith 12"),
    so gold and prediction can be matched up without sharing internal ids.
    """
    if key_fn is None:
        def key_fn(node):
            return f"{node.chain_id}#{node.position}"

    labels = {}
    for pid, person in persons.items():
        for occ_id in person.occurrences:
            node = chain_nodes.get(occ_id)
            if node is not None:
                labels[key_fn(node)] = pid
    return labels


def format_report(table1, title="TABLE 1 — narrator graph accuracy"):
    lines = [title, "=" * 58,
             f"{'':<12}{'recall':>12}{'precision':>13}{'F':>10}{'occ':>9}"]
    for row in ("detected", "all"):
        r = table1[row]
        lines.append(f"{row:<12}{r['recall']:>12}{r['precision']:>13}"
                     f"{r['f_score']:>10}{r['occurrences']:>9}")
    lines.append("=" * 58)
    lines.append("(paper: detected 0.673/0.824, all 0.632/0.771)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick demo:  python graph_agreement.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    # 6 occurrences. Gold: {a,b,c} are one person, {d,e} another, {f} alone.
    gold = {"a": "P1", "b": "P1", "c": "P1", "d": "P2", "e": "P2", "f": "P3"}

    print("perfect prediction (different label values, same partition):")
    print(format_report(score_table1(
        gold, {"a": 1, "b": 1, "c": 1, "d": 2, "e": 2, "f": 3})))

    print()
    print("under-merged: P1 split into {a,b} and {c}")
    print(format_report(score_table1(
        gold, {"a": 1, "b": 1, "c": 9, "d": 2, "e": 2, "f": 3})))

    print()
    print("over-merged: P1 and P2 fused into one node")
    print(format_report(score_table1(
        gold, {"a": 1, "b": 1, "c": 1, "d": 1, "e": 1, "f": 3})))

    print()
    print("'all' row: 'c' was never extracted -> counts as a missed merge")
    print(format_report(score_table1(
        gold, {"a": 1, "b": 1, "d": 2, "e": 2, "f": 3})))
