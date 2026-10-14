"""Run an injectable LLM extractor on strictly id/context-only JSON inputs."""

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import subprocess

from baselines.task_a_cpu.predict import digest, read_rows, write_json
from research.llms4ol2026.schema import InferenceDocument
from .contracts import Triple
from .llm import Extractor, MockBackend, TrainingExample, TrainingPool
from .postprocess import Processing


def input_documents(rows):
    ids, documents = set(), []
    for row in rows:
        if set(row) != {"id", "context"}:
            raise ValueError("Inference records must contain exactly id/context; labels are forbidden")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in ids:
            raise ValueError("Unique nonempty string IDs are required")
        if not isinstance(row["context"], str) or not row["context"].strip():
            raise ValueError("Context must be nonempty text")
        ids.add(row["id"])
        documents.append(InferenceDocument(row["id"], row["context"]))
    return documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("mock", "local"), default="mock")
    parser.add_argument("--mock-responses", type=Path, help="JSON array of response strings, one per document")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--examples", type=int, default=0)
    parser.add_argument("--train", type=Path, help="Prepared train file only; never full public source")
    parser.add_argument("--postprocess", choices=("A", "B", "C"), default="A")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing run")
    documents = input_documents(read_rows(args.input))
    if args.examples < 0:
        parser.error("Example count must be nonnegative")
    pool = None
    if args.examples:
        if args.train is None:
            parser.error("Few-shot mode requires --train")
        examples = [TrainingExample(r["id"], r["context"], tuple(Triple(**t) for t in r["triples"]))
                    for r in read_rows(args.train)]
        pool = TrainingPool(examples, (d.id for d in documents))
    if args.backend == "mock":
        if args.mock_responses is None:
            parser.error("Mock mode requires --mock-responses")
        responses = json.loads(args.mock_responses.read_text())
        if not isinstance(responses, list) or len(responses) != len(documents) or not all(isinstance(r, str) for r in responses):
            parser.error("Mock responses must be one string per input document")
        backend = MockBackend(responses)
    else:
        if args.model_path is None:
            parser.error("Local mode requires --model-path")
        from .local_backend import LocalTransformersBackend
        backend = LocalTransformersBackend(args.model_path, args.device, args.max_input_tokens, args.max_new_tokens)
    extractor = Extractor(backend, pool, args.examples, Processing(args.postprocess))
    # JSONL is appended per document so completed responses survive interruption.
    args.output.mkdir(parents=True)
    records = []
    with (args.output / "records.jsonl").open("x", encoding="utf-8") as handle:
        for document in documents:
            record = extractor.extract(document)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
    write_json(args.output / "raw-predictions.json", [{"id": r["id"], "triples": r["raw_triples"]} for r in records])
    write_json(args.output / "normalized-predictions.json", [{"id": r["id"], "triples": r["normalized_triples"]} for r in records])
    write_json(args.output / "predictions.json", [
        {"id": r["id"], "triples": r.get("postprocessed", {}).get("triples", [])} for r in records])
    write_json(args.output / "manifest.json", {
        "backend": args.backend, "mock_only": args.backend == "mock", "model_path": str(args.model_path),
        "device": args.device, "examples": args.examples, "postprocess": args.postprocess,
        "max_input_tokens": args.max_input_tokens, "max_new_tokens": args.max_new_tokens,
        "input_sha256": digest(args.input), "train_sha256": digest(args.train) if args.examples else None,
        "mock_responses_sha256": digest(args.mock_responses) if args.backend == "mock" else None,
        "code_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(), "statuses": dict(Counter(r["status"] for r in records)),
        "generation": "greedy; no sampling; no automatic retries or silent truncation",
        "evaluation": "not performed by this command; mock outputs are not model performance",
    })


if __name__ == "__main__":
    main()
