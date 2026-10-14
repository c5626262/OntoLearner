"""Seeded, fact/document-grouped splitting of public Flagship training data.

Reference annotations are used only for split construction and diagnostics.
Inference consumers receive id/context through inference_documents().
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .schema import OFFICIAL_TRAIN_BLOB, InferenceDocument, SchemaError, TrainingSample, git_blob_sha

ALGORITHM = "seeded_group_subset_search_v1"
SIMILARITY_FLOOR = 0.70


def normalize(value: str) -> str:
    return " ".join(value.lower().split())


def normalized_triples(sample: TrainingSample) -> set[tuple[str, str, str]]:
    return {tuple(normalize(part) for part in triple) for triple in sample.triples}


def find_similar_documents(samples, threshold: float = SIMILARITY_FLOOR, batch_size: int = 128):
    """Word/bigram TF-IDF cosine audit; no pretrained/semantic model is used."""
    if not 0 < threshold <= 1 or batch_size < 1:
        raise ValueError("similarity threshold must be in (0, 1] and batch_size must be positive")
    samples = sorted(samples, key=lambda sample: sample.id)
    if len(samples) < 2:
        return ()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, dtype=np.float64)
    try:
        matrix = vectorizer.fit_transform([sample.context for sample in samples])
    except ValueError as error:
        raise SchemaError("TF-IDF document audit has no usable word vocabulary") from error
    pairs = []
    for start in range(0, len(samples), batch_size):
        similarities = (matrix[start:start + batch_size] @ matrix.T).toarray()
        for offset, row in enumerate(similarities):
            index = start + offset
            for other in np.flatnonzero(row[index + 1:] >= threshold) + index + 1:
                pairs.append((samples[index].id, samples[other].id, min(1.0, float(row[other]))))
    return tuple(sorted(pairs))


def build_groups(samples, similarity_pairs=(), near_threshold: float = 0.80):
    samples = tuple(sorted(samples, key=lambda sample: sample.id))
    if len({sample.id for sample in samples}) != len(samples):
        raise SchemaError("duplicate document ID in split input")
    graph = nx.Graph()
    graph.add_nodes_from(sample.id for sample in samples)
    facts, contexts = defaultdict(list), defaultdict(list)
    for sample in samples:
        contexts[normalize(sample.context)].append(sample.id)
        for triple in sorted(normalized_triples(sample)):
            facts[triple].append(sample.id)
    for index in (facts, contexts):
        for ids in index.values():
            for sample_id in ids[1:]:
                graph.add_edge(ids[0], sample_id)
    near_pairs = 0
    for left, right, similarity in similarity_pairs:
        if left not in graph or right not in graph:
            raise SchemaError("similarity pair refers to an unknown document ID")
        if similarity >= near_threshold:
            graph.add_edge(left, right)
            near_pairs += 1
    groups = tuple(sorted(tuple(sorted(component)) for component in nx.connected_components(graph)))
    return groups, {
        "shared_normalized_triple_types": sum(len(ids) > 1 for ids in facts.values()),
        "duplicate_normalized_context_groups": sum(len(ids) > 1 for ids in contexts.values()),
        "near_document_pairs_grouped": near_pairs,
        "groups": len(groups), "largest_group_documents": max(map(len, groups), default=0),
    }


def _subset_of_groups(groups, target: int, rng: random.Random):
    """Find the closest feasible document count; group membership is atomic."""
    total = sum(map(len, groups))
    order = list(range(len(groups)))
    rng.shuffle(order)
    reachable = 1
    parent = {}
    for index in order:
        size = len(groups[index])
        new = (reachable << size) & ~reachable
        while new:
            bit = new & -new
            count = bit.bit_length() - 1
            parent[count] = (count - size, index)
            new ^= bit
        reachable |= reachable << size
        if reachable & (1 << target):
            break
    feasible = [count for count in range(1, total) if reachable & (1 << count)]
    if not feasible:
        raise SchemaError("all documents form one leakage group; a nonempty holdout cannot be made safely")
    count = min(feasible, key=lambda value: (abs(value - target), value))
    selected = set()
    while count:
        count, index = parent[count]
        selected.add(index)
    return selected


@dataclass(frozen=True)
class Split:
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...]
    config: dict
    grouping: dict


def create_split(samples, seed: int = 42, validation_fraction: float = 0.20,
                 candidates: int = 64, near_threshold: float = 0.80, similarity_pairs=None) -> Split:
    if not 0 < validation_fraction < 1 or candidates < 1 or not 0 < near_threshold <= 1:
        raise ValueError("invalid split fraction, candidates, or near_threshold")
    samples = tuple(sorted(samples, key=lambda sample: sample.id))
    if len(samples) < 2:
        raise SchemaError("at least two training documents are required")
    if similarity_pairs is None:
        similarity_pairs = find_similar_documents(samples, min(SIMILARITY_FLOOR, near_threshold))
    groups, grouping = build_groups(samples, similarity_pairs, near_threshold)
    target = min(len(samples) - 1, math.ceil(len(samples) * validation_fraction))
    by_id = {sample.id: sample for sample in samples}
    counts = [Counter(predicate for sample_id in group for _, predicate, _ in by_id[sample_id].triples)
              for group in groups]
    totals = sum(counts, Counter())
    labels = sorted(totals)
    target_counts = {label: totals[label] * validation_fraction for label in labels}
    rng = random.Random(seed)
    best = None
    for candidate in range(candidates):
        selected = _subset_of_groups(groups, target, rng)
        validation = tuple(sorted(sample_id for index in selected for sample_id in groups[index]))
        val_counts = sum((counts[index] for index in sorted(selected)), Counter())
        # Weighted count discrepancy approximates relation stratification; it
        # does not guarantee coverage for single-document/atomic rare labels.
        loss = sum((val_counts[label] - target_counts[label]) ** 2 / max(totals[label], 1) for label in labels)
        key = (abs(len(validation) - target), loss, validation)
        if best is None or key < best[0]:
            best = (key, validation, candidate)
    validation = best[1]
    train = tuple(sorted(set(by_id) - set(validation)))
    config = {
        "algorithm": ALGORITHM, "seed": seed, "validation_fraction_requested": validation_fraction,
        "rounding": "ceil(N * validation_fraction), capped at N-1; nearest feasible atomic-group count",
        "target_validation_documents": target, "candidates": candidates,
        "selected_candidate_index": best[2], "relation_count_discrepancy": best[0][1],
        "candidate_objective": "min(count error, sum((validation label count - requested fraction * full label count)^2 / max(full label count,1)), sorted validation IDs)",
        "near_document_grouping_cosine_threshold": near_threshold,
        "text_normalization": "lowercase and collapse whitespace",
        "fact_normalization": "lowercase and collapse whitespace on each triple component",
        "grouping": "connected components of shared normalized facts, duplicate normalized contexts, and near documents",
        "similarity": {"method": "word/bigram TF-IDF cosine", "ngram_range": [1, 2],
                       "lowercase": True, "token_pattern": TfidfVectorizer().token_pattern, "norm": "l2",
                       "use_idf": True, "smooth_idf": True, "sublinear_tf": False, "dtype": "float64",
                       "scope": "all public train for partition construction only; future model/retrieval fitting must use split train only"},
    }
    return Split(train, validation, groups, config, grouping)


def inference_documents(samples, split: Split, partition: str) -> tuple[InferenceDocument, ...]:
    if partition not in {"train", "validation"}:
        raise ValueError("partition must be train or validation")
    ids = split.train_ids if partition == "train" else split.validation_ids
    by_id = {sample.id: sample for sample in samples}
    return tuple(by_id[sample_id].inference_input() for sample_id in ids)


def relation_comparison(samples, split: Split) -> dict:
    by_id = {sample.id: sample for sample in samples}
    def counts(ids):
        triples, documents = Counter(), Counter()
        for sample_id in ids:
            labels = [predicate for _, predicate, _ in by_id[sample_id].triples]
            triples.update(labels)
            documents.update(set(labels))
        return triples, documents
    train, train_docs = counts(split.train_ids)
    val, val_docs = counts(split.validation_ids)
    totals = train + val
    train_total, val_total = sum(train.values()) or 1, sum(val.values()) or 1
    rows = [{"label": label, "full_triples": totals[label], "train_triples": train[label],
             "validation_triples": val[label], "full_documents": train_docs[label] + val_docs[label],
             "train_documents": train_docs[label], "validation_documents": val_docs[label],
             "train_triple_fraction": train[label] / train_total,
             "validation_triple_fraction": val[label] / val_total}
            for label in sorted(totals, key=lambda label: (-totals[label], label))]
    categories = {}
    for name, allowed in [("is-a", {"is-a"}), ("instance-of", {"instance-of"}),
                          ("other", set(totals) - {"is-a", "instance-of"})]:
        a, b = sum(train[label] for label in allowed), sum(val[label] for label in allowed)
        categories[name] = {"train_triples": a, "validation_triples": b,
                            "train_fraction": a / train_total, "validation_fraction": b / val_total}
    return {"labels": rows, "categories": categories,
            "rare_definition": "present in at most 5 documents of the public full train",
            "rare_labels_missing_validation": [row["label"] for row in rows if row["full_documents"] <= 5 and row["validation_documents"] == 0],
            "all_labels_missing_validation": [row["label"] for row in rows if row["validation_documents"] == 0],
            "all_labels_missing_train": [row["label"] for row in rows if row["train_documents"] == 0]}


def leakage_audit(samples, split: Split, similarity_pairs) -> dict:
    by_id = {sample.id: sample for sample in samples}
    def information(ids):
        facts, nodes, contexts = set(), set(), set()
        for sample_id in ids:
            sample = by_id[sample_id]
            facts.update(normalized_triples(sample))
            nodes.update(normalize(value) for s, _, o in sample.triples for value in (s, o))
            contexts.add(normalize(sample.context))
        return facts, nodes, contexts
    a, b = information(split.train_ids), information(split.validation_ids)
    val_ids = set(split.validation_ids)
    crossing = [(left, right, score) for left, right, score in similarity_pairs
                if (left in val_ids) != (right in val_ids)]
    crossing.sort(key=lambda row: (-row[2], row[0], row[1]))
    return {
        "overlapping_ids": len(set(split.train_ids) & val_ids),
        "shared_normalized_triples": len(a[0] & b[0]), "duplicate_normalized_contexts": len(a[2] & b[2]),
        "shared_node_strings": len(a[1] & b[1]),
        "split_type": "document/fact-group holdout, not entity-disjoint or verified domain-disjoint",
        "lexical_similarity_audit_floor": min(SIMILARITY_FLOOR, split.config["near_document_grouping_cosine_threshold"]),
        "all_document_pairs_at_or_above_floor": len(similarity_pairs),
        "cross_partition_pairs_at_or_above_floor": len(crossing),
        "cross_partition_pairs_at_or_above_grouping_threshold": sum(score >= split.config["near_document_grouping_cosine_threshold"] for _, _, score in crossing),
        "highest_detected_cross_partition_cosine": crossing[0][2] if crossing else None,
        "cross_partition_examples": [{"left_id": left, "right_id": right, "cosine": score} for left, right, score in crossing[:10]],
        "limitations": "Lexical audit only. Similarity below the floor and paraphrase/semantic leakage are not ruled out. Shared concepts are reported, not treated as identical facts.",
    }


def save_split(samples, split: Split, output: str | Path, source_path: str | Path, similarity_pairs):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        path = output / name
        if path.exists() and path.read_text() != text:
            raise FileExistsError(f"refusing to overwrite a different split artifact: {path}")
        path.write_text(text)
        return hashlib.sha256(text.encode()).hexdigest()
    train_sha = save("train_ids.json", split.train_ids)
    val_sha = save("validation_ids.json", split.validation_ids)
    comparison = relation_comparison(samples, split)
    audit = leakage_audit(samples, split, similarity_pairs)
    save("relation_distribution.json", comparison)
    save("leakage_audit.json", audit)
    content = Path(source_path).read_bytes()
    source_scope = ("official public train only; no official test used" if git_blob_sha(content) == OFFICIAL_TRAIN_BLOB
                    else "synthetic fixture or unpinned input; not verified official training data")
    provenance = json.loads((Path(__file__).parent / "source_manifest.json").read_text())
    manifest = {"source": {"scope": source_scope,
                            "official_repository": provenance["repository"],
                            "official_commit": provenance["commit"], "official_train_path": provenance["train_path"],
                            "git_blob_sha": git_blob_sha(content), "sha256": hashlib.sha256(content).hexdigest()},
                "config": split.config, "grouping": split.grouping,
                "counts": {"full": len(samples), "train": len(split.train_ids), "validation": len(split.validation_ids)},
                "ids": {"train_file": "train_ids.json", "train_sha256": train_sha,
                        "validation_file": "validation_ids.json", "validation_sha256": val_sha},
                "versions": {"python": platform.python_version(),
                             **{name: importlib.metadata.version(name) for name in ["numpy", "scikit-learn", "networkx"]}},
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "inference_contract": "id/context only via inference_documents(); reference triples are used for splitting and evaluation, never exported as inference inputs"}
    save("manifest.json", manifest)
    return manifest
