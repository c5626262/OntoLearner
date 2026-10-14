"""Evaluation-only error candidates and annotation surface coverage."""

from collections import defaultdict
import difflib
import json
from pathlib import Path
import re

from baselines.task_a_cpu.experiment import partitions
from baselines.task_a_cpu.predict import read_rows, triple_tuple, write_json
from research.llms4ol2026.evaluator import load_official_functions
from .analysis import diagnose, surface


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite diagnostics")
    samples = partitions(args.source, Path("research/llms4ol2026/splits/seed42"))["validation"]
    predictions = {r["id"]: r["triples"] for r in read_rows(args.predictions)}
    if set(predictions) != {s.id for s in samples}:
        parser.error("Prediction IDs differ from validation")
    normalized = load_official_functions().normalize_triples
    examples = defaultdict(list)
    coverage = defaultdict(int)
    for sample in samples:
        gold = set(normalized(sample.triples))
        pred = set(normalized([triple_tuple(t) for t in predictions[sample.id]]))
        context = " " + surface(sample.context) + " "
        for s, _, o in gold:
            mentioned = [" " + surface(label) + " " in context for label in (s, o)]
            coverage[f"{sum(mentioned)}_endpoints_literal_token_match"] += 1
        for triple in sorted(pred - gold):
            category = next(iter(diagnose(gold, {triple})))
            if len(examples[category]) >= 3:
                continue
            s, p, o = triple
            if category == "reversed_endpoints":
                matched = [(o, p, s)]
            elif category == "wrong_predicate_same_endpoints":
                matched = sorted(t for t in gold if (t[0], t[2]) == (s, o))
            elif category == "surface_variant_candidate":
                matched = sorted(t for t in gold if t[1] == p and surface(s) == surface(t[0]) and surface(o) == surface(t[2]))
            elif category == "near_spelling_candidate":
                matched = sorted(t for t in gold if t[1] == p and min(difflib.SequenceMatcher(None, s, t[0]).ratio(),
                                                                   difflib.SequenceMatcher(None, o, t[2]).ratio()) >= 0.8)
            else:
                matched = []
            sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", sample.context)
                         if s in part.lower() or o in part.lower()]
            examples[category].append({"id": sample.id, "predicted": triple,
                                       "gold_candidates": matched, "context_excerpt": " ".join(sentences)[:800]})
    report = {"examples": dict(examples), "gold_surface_coverage": dict(coverage),
              "note": "Evaluation-only candidate diagnoses; near spelling is not confirmed synonymy. No predictions were changed."}
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
