"""CLI for pinned train statistics and model-free official evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_training
from .evaluator import score_triples
from .schema import load_training_data, validate_prediction_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--train", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--train", required=True, type=Path)
    evaluate.add_argument("--predictions", required=True, type=Path)
    evaluate.add_argument("--mode", choices=["exact", "fuzzy"], default="exact")
    evaluate.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    samples = load_training_data(args.train)
    if args.command == "analyze":
        report = analyze_training(samples)
    else:
        predicted = validate_prediction_records(json.loads(args.predictions.read_text()), (s.id for s in samples))
        per_document = [{"id": sample.id, **score_triples(sample.triples, predicted[sample.id], args.mode)}
                        for sample in samples]
        report = {"mode": args.mode, "aggregation": "local per-document arithmetic mean; official dataset aggregation unconfirmed",
                  "prediction_source": str(args.predictions), "documents": len(samples),
                  "mean_scores": {key: sum(row[key] for row in per_document) / len(per_document) if per_document else 0
                                  for key in ["edge_f1", "neighborhood_similarity", "taxonomy_similarity", "graph_similarity"]},
                  "per_document": per_document}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "documents": len(samples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
