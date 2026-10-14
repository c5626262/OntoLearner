"""Evaluation-only diagnostics; never imported by extraction implementations."""

from collections import Counter
import difflib
import json
from pathlib import Path
import unicodedata
import re

from baselines.task_a_cpu.evaluate import group, score
from baselines.task_a_cpu.predict import read_rows, triple_tuple
from research.llms4ol2026.evaluator import load_official_functions, score_triples
from research.llms4ol2026.schema import TRIPLES_KEY, validate_prediction_records


def surface(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def diagnose(gold, predicted):
    """Exclusive FP categories in priority order; NOT corrected metric scores.

    Matches are diagnostic candidates, not one-to-one alignments or synonyms.
    """
    counts = Counter()
    for s, p, o in sorted(predicted - gold):
        if (o, p, s) in gold:
            category = "reversed_endpoints"
        elif any((s, o) == (gs, go) for gs, _, go in gold):
            category = "wrong_predicate_same_endpoints"
        elif any(p == gp and surface(s) == surface(gs) and surface(o) == surface(go)
                 for gs, gp, go in gold):
            category = "surface_variant_candidate"
        elif any(p == gp and min(difflib.SequenceMatcher(None, s, gs).ratio(),
                                 difflib.SequenceMatcher(None, o, go).ratio()) >= 0.8
                 for gs, gp, go in gold):
            category = "near_spelling_candidate"
        else:
            category = "other_endpoint_or_unsupported_relation"
        counts[category] += 1
    return dict(counts)


def analyze(samples, prediction_rows, train_rows=()):
    predictions = validate_prediction_records(
        [{"id": r["id"], TRIPLES_KEY: [triple_tuple(t) for t in r["triples"]]} for r in prediction_rows],
        (s.id for s in samples),
    )
    normalize = load_official_functions().normalize_triples
    train_nodes = {part for r in train_rows for t in normalize([triple_tuple(t) for t in r["triples"]])
                   for part in (t[0], t[2])}
    documents, gold_all, pred_all = [], set(), set()
    diagnostics = Counter()
    raw_counts = Counter()
    for sample in samples:
        gold, pred = set(normalize(sample.triples)), set(normalize(predictions[sample.id]))
        gold_all.update((sample.id, *t) for t in gold)
        pred_all.update((sample.id, *t) for t in pred)
        diagnostics.update(diagnose(gold, pred))
        raw_counts.update(group(p) for _, p, _ in predictions[sample.id])
        metrics = score(gold, pred)
        documents.append({
            "id": sample.id, "characters": len(sample.context), "words": len(sample.context.split()),
            "raw_gold": len(sample.triples), "gold": len(gold), "predicted": len(pred),
            **metrics, "official": score_triples(sample.triples, predictions[sample.id]),
            "correct": sorted(gold & pred), "fp_examples": sorted(pred - gold)[:2],
            "fn_examples": sorted(gold - pred)[:2],
            "gold_endpoints_unseen_in_train": sum(s not in train_nodes or o not in train_nodes for s, _, o in gold),
            "title": sample.context.splitlines()[0].strip(),
        })

    def aggregate(rows):
        tp, fp, fn = (sum(d[k] for d in rows) for k in ("tp", "fp", "fn"))
        return {"documents": len(rows), "tp": tp, "fp": fp, "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else 0,
                "recall": tp / (tp + fn) if tp + fn else 0,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                "official_mean": {key: sum(d["official"][key] for d in rows) / len(rows) if rows else 0
                                  for key in ("edge_f1", "neighborhood_similarity", "taxonomy_similarity", "graph_similarity")}}

    successes = sorted((d for d in documents if d["tp"]), key=lambda d: (-d["f1"], d["id"]))[:10]
    # Show both empty-output and nonempty-error cases, not only the first 10 IDs.
    empty = sorted((d for d in documents if not d["predicted"] and d["fn"]), key=lambda d: d["id"])[:5]
    nonempty = sorted((d for d in documents if d["predicted"] and (d["fp"] or d["fn"])),
                      key=lambda d: (d["f1"], -d["fp"], d["id"]))[:5]
    return {
        "definition": "Official exact, per-document arithmetic mean; relation F1 is normalized directed micro F1",
        "overall": aggregate(documents),
        "raw_prediction_triples": sum(len(r["triples"]) for r in prediction_rows),
        "raw_predictions_by_relation": dict(raw_counts),
        "raw_gold_triples": sum(len(s.triples) for s in samples),
        "normalized_prediction_triples": len(pred_all), "normalized_gold_triples": len(gold_all),
        "empty_prediction_documents": sum(not d["predicted"] for d in documents),
        "documents_with_correct_triples": sum(d["tp"] > 0 for d in documents),
        "exact_graph_documents": sum(d["fp"] == d["fn"] == 0 for d in documents),
        "gold_triples_with_unseen_endpoint": sum(d["gold_endpoints_unseen_in_train"] for d in documents),
        "by_relation": {g: score({t for t in gold_all if group(t[2]) == g},
                                 {t for t in pred_all if group(t[2]) == g}) for g in ("is-a", "instance-of", "other")},
        "exclusive_fp_diagnostics": dict(diagnostics),
        "diagnostic_note": "Candidate categories, not semantic judgments or one-to-one correction scores",
        "by_characters": {f"{lo}-{hi - 1}": aggregate([d for d in documents if lo <= d["characters"] < hi])
                          for lo, hi in ((0, 1000), (1000, 2000), (2000, 4000), (4000, 1000000))},
        "by_gold_triples": {f"{lo}-{hi - 1}": aggregate([d for d in documents if lo <= d["gold"] < hi])
                            for lo, hi in ((0, 6), (6, 11), (11, 21), (21, 1000000))},
        "success_definition": "TP>0, highest F1 then ID; partial successes, not perfect graphs",
        "failure_definition": "5 empty predictions by ID and 5 nonempty errors by F1, descending FP, ID",
        "success_examples": successes, "failure_examples": empty + nonempty,
    }


def main():
    import argparse
    from baselines.task_a_cpu.experiment import partitions
    from baselines.task_a_cpu.predict import digest, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("research/llms4ol2026/splits/seed42"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing analysis")
    report = analyze(partitions(args.source, args.split)["validation"], read_rows(args.predictions), read_rows(args.train))
    report["hashes"] = {"source": digest(args.source), "split": digest(args.split / "manifest.json"),
                        "predictions": digest(args.predictions), "train": digest(args.train)}
    write_json(args.output, report)
    print(json.dumps({k: report[k] for k in ("overall", "empty_prediction_documents", "exclusive_fp_diagnostics")}))


if __name__ == "__main__":
    main()
