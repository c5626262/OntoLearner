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
    split_command = commands.add_parser("split", help="grouped train/validation holdout from pinned public train")
    split_command.add_argument("--train", required=True, type=Path)
    split_command.add_argument("--output", required=True, type=Path)
    split_command.add_argument("--seed", type=int, default=42)
    split_command.add_argument("--validation-fraction", type=float, default=0.20)
    split_command.add_argument("--candidates", type=int, default=64)
    split_command.add_argument("--near-threshold", type=float, default=0.80)
    args = parser.parse_args()
    samples = load_training_data(args.train)
    if args.command == "split":
        from .splitting import SIMILARITY_FLOOR, create_split, find_similar_documents, save_split
        pairs = find_similar_documents(samples, min(SIMILARITY_FLOOR, args.near_threshold))
        split = create_split(samples, args.seed, args.validation_fraction, args.candidates, args.near_threshold, pairs)
        manifest = save_split(samples, split, args.output, args.train, pairs)
        print(json.dumps({"output": str(args.output), **manifest["counts"]}, ensure_ascii=False))
        return
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
