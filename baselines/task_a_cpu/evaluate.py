"""Offline directed exact-match diagnostics; NOT official Graph Similarity."""

import argparse
from pathlib import Path

try:
    from .predict import KEYS, digest, read_rows, triple_tuple, write_json
except ImportError:  # Direct script invocation.
    from predict import KEYS, digest, read_rows, triple_tuple, write_json

GROUPS = ("is-a", "instance-of", "other")


def group(predicate):
    return predicate if predicate in GROUPS[:2] else "other"


def score(gold, predicted):
    tp = len(gold & predicted)
    fp, fn = len(predicted - gold), len(gold - predicted)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def evaluate(gold_rows, prediction_rows):
    gold_by_id = {row["id"]: row for row in gold_rows}
    pred_by_id = {row["id"]: row for row in prediction_rows}
    if len(gold_by_id) != len(gold_rows) or len(pred_by_id) != len(prediction_rows):
        raise ValueError("Duplicate document IDs")
    if gold_by_id.keys() != pred_by_id.keys():
        raise ValueError("Gold and prediction IDs must match exactly")
    all_gold, all_pred = set(), set()
    documents = []
    for doc_id, row in gold_by_id.items():
        gold = {triple_tuple(t) for t in row["triples"]}
        pred = {triple_tuple(t) for t in pred_by_id[doc_id]["triples"]}
        all_gold.update((doc_id, *t) for t in gold)
        all_pred.update((doc_id, *t) for t in pred)
        documents.append({
            "id": doc_id, "metrics": score(gold, pred),
            "exact_match": gold == pred,
            "correct": [dict(zip(KEYS, t)) for t in sorted(gold & pred)],
            "false_positive": [dict(zip(KEYS, t)) for t in sorted(pred - gold)],
            "false_negative": [dict(zip(KEYS, t)) for t in sorted(gold - pred)],
        })
    successes = sorted((d for d in documents if d["metrics"]["tp"]),
                       key=lambda d: (-d["metrics"]["f1"], str(d["id"])))[:5]
    failures = sorted((d for d in documents if not d["exact_match"]),
                      key=lambda d: (d["metrics"]["f1"], str(d["id"])))[:5]
    return {
        "official_graph_similarity": None,
        "official_status": "Not run: requires the existing official Task A adapter",
        "diagnostic_definition": "Case-sensitive, whitespace-trimmed directed exact triple match; micro F1",
        "overall": score(all_gold, all_pred),
        "by_relation": {g: score({t for t in all_gold if group(t[2]) == g},
                                 {t for t in all_pred if group(t[2]) == g}) for g in GROUPS},
        "documents": documents,
        "success_definition": "At least one correct triple; highest F1 first (may be partial)",
        "failure_definition": "At least one false positive or false negative; lowest F1 first",
        "success_examples": successes, "failure_examples": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.gold.resolve(), args.predictions.resolve()}:
        parser.error("Report must not overwrite gold or predictions")
    report = evaluate(read_rows(args.gold), read_rows(args.predictions))
    report["gold_sha256"] = digest(args.gold)
    report["predictions_sha256"] = digest(args.predictions)
    write_json(args.output, report)


if __name__ == "__main__":
    main()
