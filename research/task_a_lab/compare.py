"""Paired official-exact comparison of frozen predictions (evaluation only)."""

import argparse
import json
from pathlib import Path

from baselines.task_a_cpu.experiment import partitions
from baselines.task_a_cpu.predict import digest, read_rows, triple_tuple, write_json
from research.llms4ol2026.evaluator import load_official_functions, score_triples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite comparison")
    samples = partitions(args.source, Path("research/llms4ol2026/splits/seed42"))["validation"]
    predictions = [{r["id"]: [triple_tuple(t) for t in r["triples"]] for r in read_rows(path)}
                   for path in (args.before, args.after)]
    if any(set(p) != {s.id for s in samples} for p in predictions):
        parser.error("Prediction IDs differ from validation")
    normalizer = load_official_functions().normalize_triples
    rows = []
    for sample in samples:
        left, right = (p[sample.id] for p in predictions)
        a, b = (score_triples(sample.triples, p)["graph_similarity"] for p in (left, right))
        gold = set(normalizer(sample.triples))
        correct_a, correct_b = (set(normalizer(p)) & gold for p in (left, right))
        rows.append({"id": sample.id, "before": a, "after": b, "delta": b - a,
                     "lost_correct": len(correct_a - correct_b), "gained_correct": len(correct_b - correct_a)})
    report = {"before_sha256": digest(args.before), "after_sha256": digest(args.after),
              "comparison": "paired document GS exact; epsilon 1e-12; no significance claim",
              "improved_documents": sum(r["delta"] > 1e-12 for r in rows),
              "worsened_documents": sum(r["delta"] < -1e-12 for r in rows),
              "tied_documents": sum(abs(r["delta"]) <= 1e-12 for r in rows),
              "lost_correct": sum(r["lost_correct"] for r in rows),
              "gained_correct": sum(r["gained_correct"] for r in rows),
              "mean_delta": sum(r["delta"] for r in rows) / len(rows), "documents": rows}
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "documents"}))


if __name__ == "__main__":
    main()
