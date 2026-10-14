"""Separate CPU ablations; train-only fitting and context-only prediction."""

import argparse
from pathlib import Path
import re
import time

from baselines.task_a_cpu.predict import Baseline, KEYS, digest, predict_rows, read_rows, tokens, write_json

# Fixed English surface patterns, not inferred from validation labels.
ENTITY = r"[\w][\w\s()'’/\-]{0,180}?"
CUES = (
    (r"(?:is|are) (?:an? )?instance of", "instance-of"),
    (r"(?:is|are|represents?|constitutes?) (?:an? |the )?(?:subclass|type|kind|form|member) of", "is-a"),
    (r"(?:is|are) (?:explicitly )?(?:classified|categorized|recognized|defined) as", "is-a"),
    (r"(?:is|are) (?:an? )?part of", "part_of"),
    (r"(?:is|are) located in", "located in"),
)
STOP = {"a", "an", "the", "it", "this", "that", "these", "those", "they", "which", "there", "both"}


def clean_phrase(value):
    value = " ".join(value.strip(" ,:;").split())
    value = re.sub(r"^(?:(?:additionally|furthermore|similarly|conversely|finally|specifically)\s*,?\s*)", "", value, flags=re.I)
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.I)
    if not 1 <= len(tokens(value)) <= 14 or value.casefold() in STOP:
        return None
    return value


def surface_rules(context):
    """Extract finite clauses, with conservative boundaries and no gold access."""
    for sentence in re.split(r"[.!?;\n]+", context):
        # Commas delimit discourse prefixes and nonrestrictive clauses.
        for clause in re.split(r",|\b(?:which|that|while|whereas)\b", sentence):
            clause = clause.strip()
            for cue, predicate in CUES:
                match = re.fullmatch(rf"({ENTITY})\s+{cue}\s+({ENTITY})", clause, re.I)
                if match:
                    left, right = (clean_phrase(v) for v in match.groups())
                    if left and right and left.casefold() != right.casefold():
                        yield left, predicate, right


class ImprovedCPU(Baseline):
    """Preserve the original baseline; change prediction strategy only here."""

    def __init__(self, train, variant="rules_only"):
        if variant not in {"rules_only", "surface", "conservative"}:
            raise ValueError("Unknown CPU variant")
        super().__init__(train)
        self.variant = variant

    def predict(self, context):
        if not isinstance(context, str):
            raise ValueError("context must be text")
        result = set(self.mention_rules(context))
        if self.variant == "conservative":
            # Suppress generic stopword concepts and nested-label false matches.
            result = {t for t in result if all(tokens(t[i]) and t[i].casefold() not in STOP for i in (0, 2))}
            result = {t for t in result if not any(
                t != u and t[1] == u[1]
                and (t[0] == u[0] and t[2] != u[2] and " " + t[2].casefold() in u[2].casefold()
                     or t[2] == u[2] and t[0] != u[0] and " " + t[0].casefold() in u[0].casefold())
                for u in result)}
        elif self.variant == "surface":
            result.update(surface_rules(context))
        return [dict(zip(KEYS, t)) for t in sorted(result)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--variant", choices=("rules_only", "conservative", "surface"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite experiment output")
    train, inputs = read_rows(args.train), read_rows(args.input)
    if not train or {r["id"] for r in train} & {r["id"] for r in inputs}:
        parser.error("Nonempty train and disjoint input IDs required")
    started = time.perf_counter()
    predictions = predict_rows(ImprovedCPU(train, args.variant), inputs)
    write_json(args.output / "predictions.json", predictions)
    write_json(args.output / "manifest.json", {
        "variant": args.variant, "train_sha256": digest(args.train), "input_sha256": digest(args.input),
        "predictions_sha256": digest(args.output / "predictions.json"),
        "implementation_sha256": digest(__file__), "baseline_sha256": digest(Path(__file__).parents[2] / "baselines/task_a_cpu/predict.py"),
        "documents": len(inputs), "fit_and_predict_seconds": time.perf_counter() - started,
        "seed": None, "deterministic": True, "device": "CPU", "fitting_scope": "train only",
    })


if __name__ == "__main__":
    main()
