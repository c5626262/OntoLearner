"""Prepare the pinned split and score saved predictions with the existing adapter.

Prediction runs separately via predict.py, with train labels and id/context-only
validation input. This module never constructs predictions from gold.
"""

import argparse
import json
from pathlib import Path
import platform

from research.llms4ol2026.evaluator import OFFICIAL_METRIC_BLOB, load_official_functions, score_triples
from research.llms4ol2026.schema import TRIPLES_KEY, load_training_data, validate_prediction_records

from .evaluate import evaluate
from .predict import KEYS, digest, read_rows, triple_tuple, write_json


def partitions(source, split_dir):
    manifest = json.loads((split_dir / "manifest.json").read_text())
    if digest(source) != manifest["source"]["sha256"]:
        raise ValueError("Source SHA-256 differs from saved split")
    ids = {}
    for name in ("train", "validation"):
        path = split_dir / f"{name}_ids.json"
        if digest(path) != manifest["ids"][f"{name}_sha256"]:
            raise ValueError("Split ID SHA-256 mismatch")
        ids[name] = json.loads(path.read_text())
        if len(ids[name]) != len(set(ids[name])):
            raise ValueError("Duplicate split ID")
    samples = {sample.id: sample for sample in load_training_data(source)}
    if set(ids["train"]) & set(ids["validation"]) or set(ids["train"] + ids["validation"]) != set(samples):
        raise ValueError("Split must be disjoint and cover source exactly")
    return {name: [samples[i] for i in values] for name, values in ids.items()}


def prepare(source, split_dir, output):
    samples = partitions(source, split_dir)
    train = [{"id": s.id, "context": s.context,
              "triples": [dict(zip(KEYS, t)) for t in s.triples]} for s in samples["train"]]
    validation = [s.inference_input() for s in samples["validation"]]
    write_json(output / "train.json", train)
    write_json(output / "validation-contexts.json", [{"id": s.id, "context": s.context} for s in validation])
    write_json(output / "preparation.json", {
        "source_sha256": digest(source), "split_manifest_sha256": digest(split_dir / "manifest.json"),
        "train_sha256": digest(output / "train.json"),
        "validation_contexts_sha256": digest(output / "validation-contexts.json"),
        "train_documents": len(train), "validation_documents": len(validation),
    })


def score_saved(source, split_dir, predictions_path, output):
    samples = partitions(source, split_dir)["validation"]
    rows = read_rows(predictions_path)
    adapter_rows = [{"id": r["id"], TRIPLES_KEY: [triple_tuple(t) for t in r["triples"]]} for r in rows]
    predicted = validate_prediction_records(adapter_rows, (s.id for s in samples))
    normalizer = load_official_functions().normalize_triples
    gold_rows = [{"id": s.id, "triples": [dict(zip(KEYS, t)) for t in normalizer(s.triples)]} for s in samples]
    pred_rows = [{"id": r["id"], "triples": [dict(zip(KEYS, t)) for t in normalizer(predicted[r["id"]])]} for r in rows]
    report = evaluate(gold_rows, pred_rows)
    per_document = [{"id": s.id, **score_triples(s.triples, predicted[s.id], "exact")} for s in samples]
    means = {key: sum(row[key] for row in per_document) / len(per_document)
             for key in ("edge_f1", "neighborhood_similarity", "taxonomy_similarity", "graph_similarity")}
    report.update({
        "official_graph_similarity": means["graph_similarity"],
        "official_status": "Existing pinned adapter, exact mode, no embedding model",
        "official_mean_scores": means, "official_per_document": per_document,
        "aggregation": "Local arithmetic mean over validation documents; official dataset aggregation unconfirmed",
        "diagnostic_definition": "Directed exact triple micro F1 after official lowercase/whitespace normalization",
        "source_sha256": digest(source), "predictions_sha256": digest(predictions_path),
        "split_manifest_sha256": digest(split_dir / "manifest.json"),
        "official_metric_blob": OFFICIAL_METRIC_BLOB, "python": platform.python_version(),
    })
    by_id = {s.id: s for s in samples}
    scores = {r["id"]: r for r in per_document}
    for category in ("success_examples", "failure_examples"):
        for example in report[category]:
            example["context"] = by_id[example["id"]].context
            example["official_scores"] = scores[example["id"]]
    write_json(output / "official-predictions.json", adapter_rows)
    write_json(output / "evaluation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("research/llms4ol2026/splits/seed42"))
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing outputs are never overwritten")
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; use a new directory")
    if args.command == "prepare":
        prepare(args.source, args.split, args.output)
    else:
        if args.predictions is None:
            parser.error("evaluate requires --predictions")
        report = score_saved(args.source, args.split, args.predictions, args.output)
        print(json.dumps({"official_mean_scores": report["official_mean_scores"], "by_relation": report["by_relation"],
                          "successes": len(report["success_examples"]), "failures": len(report["failure_examples"])}))


if __name__ == "__main__":
    main()
