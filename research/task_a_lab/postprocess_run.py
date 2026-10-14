"""Apply A/B/C to one frozen prediction file; no model or gold access."""

import argparse
import json
from pathlib import Path

from baselines.task_a_cpu.predict import digest, read_rows, write_json
from .contracts import Triple
from .postprocess import Processing
from .run_llm import input_documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="id/context only")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("A", "B", "C"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing output")
    documents = input_documents(read_rows(args.input))
    rows = read_rows(args.predictions)
    predicted = {r["id"]: tuple(Triple(**t) for t in r["triples"]) for r in rows}
    if set(predicted) != {d.id for d in documents}:
        parser.error("Input and prediction IDs must match exactly")
    processor = Processing(args.mode)
    output = []
    args.output.mkdir(parents=True)
    merges = 0
    with (args.output / "lineage.jsonl").open("x", encoding="utf-8") as handle:
        for document in documents:
            result = processor.process(document, predicted[document.id])
            merges += len(result["merge_groups"])
            handle.write(json.dumps({"id": document.id, **result}, ensure_ascii=False) + "\n")
            output.append({"id": document.id, "triples": result["triples"]})
    write_json(args.output / "predictions.json", output)
    write_json(args.output / "manifest.json", {
        "mode": args.mode, "input_sha256": digest(args.input), "raw_predictions_sha256": digest(args.predictions),
        "output_sha256": digest(args.output / "predictions.json"),
        "implementation_sha256": digest(Path(__file__).with_name("postprocess.py")),
        "merge_groups": merges, "context_threshold": processor.threshold,
        "scope": "within document", "semantic_model_used": False,
    })


if __name__ == "__main__":
    main()
