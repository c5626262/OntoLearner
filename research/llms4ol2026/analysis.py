"""Descriptive statistics over the pinned public training set only."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
import statistics
import unicodedata

import networkx as nx

from .schema import TrainingSample

ALIAS_RELATIONS = {"equivalent class", "same as", "exact match"}


def distribution(values) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    def percentile(q):
        position = (len(ordered) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    return {
        "count": len(ordered), "min": ordered[0], "mean": statistics.mean(ordered),
        "median": statistics.median(ordered), "p90": percentile(.9),
        "p95": percentile(.95), "max": ordered[-1], "sum": sum(ordered),
    }


def lexical_key(text: str) -> str:
    """A candidate-grouping heuristic; it does not establish identity."""
    return " ".join(re.sub(r"[_-]+", " ", unicodedata.normalize("NFKC", text).casefold()).split())


def analyze_training(samples: tuple[TrainingSample, ...]) -> dict:
    relations = Counter()
    self_relations = Counter()
    relation_documents = Counter()
    global_triples = set()
    unique_total = duplicate_total = self_total = 0
    same_reverse = Counter()
    reverse_examples = []
    alias_examples = []
    alias_nonidentical = Counter()
    variants = defaultdict(Counter)
    variant_docs = defaultdict(set)
    node_counts, component_counts = [], []
    taxonomy_cycles = []
    nontrivial_taxonomy_cycles = []
    without_taxonomy = without_non_taxonomy = 0
    reverse_documents = different_reverse_pairs = any_reverse_documents = any_reverse_pairs = 0
    for sample in samples:
        triples = sample.triples
        unique = set(triples)
        unique_total += len(unique)
        duplicate_total += len(triples) - len(unique)
        global_triples.update(unique)
        relations.update(predicate for _, predicate, _ in triples)
        relation_documents.update({predicate for _, predicate, _ in triples})
        self_total += sum(s == o for s, _, o in triples)
        self_relations.update(p for s, p, o in triples if s == o)
        without_taxonomy += not any(p == "is-a" for _, p, _ in triples)
        without_non_taxonomy += not any(p not in {"is-a", "instance-of"} for _, p, _ in triples)
        graph, taxonomy = nx.Graph(), nx.DiGraph()
        pairs = defaultdict(set)
        sample_reverse = set()
        for s, p, o in unique:
            graph.add_edge(s, o)
            if p == "is-a":
                taxonomy.add_edge(s, o)
            if s != o:
                pairs[(s, o)].add(p)
                if (o, p, s) in unique:
                    sample_reverse.add((min(s, o), p, max(s, o)))
        for s, p, o in triples:
            for node in (s, o):
                key = lexical_key(node)
                variants[key][node] += 1
                variant_docs[(key, node)].add(sample.id)
            if p in ALIAS_RELATIONS and s != o:
                alias_nonidentical[p] += 1
                if len(alias_examples) < 15:
                    alias_examples.append({"id": sample.id, "triple": [s, p, o],
                                           "interpretation": "annotated equivalence candidate, not independently verified synonymy"})
        same_reverse.update(p for _, p, _ in sample_reverse)
        reverse_documents += bool(sample_reverse)
        for s, p, o in sorted(sample_reverse):
            if len(reverse_examples) < 12:
                reverse_examples.append({"id": sample.id, "subject": s, "predicate": p, "object": o})
        any_pairs = set()
        for s, o in pairs:
            if (o, s) in pairs:
                any_pairs.add(tuple(sorted((s, o))))
                if s < o and any(a != b for a in pairs[(s, o)] for b in pairs[(o, s)]):
                    different_reverse_pairs += 1
        any_reverse_documents += bool(any_pairs)
        any_reverse_pairs += len(any_pairs)
        node_counts.append(graph.number_of_nodes())
        component_counts.append(nx.number_connected_components(graph))
        if not nx.is_directed_acyclic_graph(taxonomy):
            taxonomy_cycles.append(sample.id)
        taxonomy.remove_edges_from(list(nx.selfloop_edges(taxonomy)))
        if not nx.is_directed_acyclic_graph(taxonomy):
            nontrivial_taxonomy_cycles.append(sample.id)
    lexical_variants = []
    for key, forms in sorted(variants.items(), key=lambda item: (-sum(item[1].values()), item[0])):
        if len(forms) > 1:
            lexical_variants.append({"candidate_key": key,
                                     "forms": dict(sorted(forms.items())),
                                     "example_ids": {form: sorted(variant_docs[(key, form)])[:3] for form in sorted(forms)}})
    total = sum(relations.values())
    groups = {
        "taxonomy_is_a": relations["is-a"],
        "term_typing_instance_of": relations["instance-of"],
        "other_non_taxonomic_labels": sum(n for p, n in relations.items() if p not in {"is-a", "instance-of"}),
    }
    return {
        "scope": "public training data only; statistics are not model scores",
        "counts": {"documents": len(samples), "unique_ids": len({s.id for s in samples}),
                   "triples_raw": total, "triples_unique_within_documents": unique_total,
                   "duplicates_within_documents": duplicate_total, "global_unique_triples": len(global_triples),
                   "relation_labels": len(relations), "global_node_strings": sum(len(v) for v in variants.values()),
                   "self_loop_triples_raw": self_total,
                   "duplicate_contexts": len(samples)-len({s.context for s in samples})},
        "lengths": {"context_characters": distribution([len(s.context) for s in samples]),
                    "context_whitespace_words": distribution([len(s.context.split()) for s in samples]),
                    "triples_per_document": distribution([len(s.triples) for s in samples]),
                    "nodes_per_document": distribution(node_counts)},
        "relation_groups": {name: {"count": count, "fraction": count / total if total else 0}
                            for name, count in groups.items()},
        "relations": [{"label": p, "triples": n, "fraction": n/total if total else 0,
                       "documents": relation_documents[p]} for p, n in relations.most_common()],
        "structure": {"documents_without_is_a": without_taxonomy,
                      "self_loop_triples_by_predicate": dict(self_relations.most_common()),
                      "documents_without_other_non_taxonomic": without_non_taxonomy,
                      "weak_components_per_document": distribution(component_counts),
                      "connected_documents": sum(n == 1 for n in component_counts),
                      "cyclic_taxonomy_documents": len(taxonomy_cycles), "cyclic_taxonomy_example_ids": taxonomy_cycles[:10],
                      "cyclic_taxonomy_documents_excluding_self_loops": len(nontrivial_taxonomy_cycles),
                      "nontrivial_cycle_example_ids": nontrivial_taxonomy_cycles[:10]},
        "bidirectionality": {"definition": "within each document, unique triples; self loops excluded; each unordered pair counted once",
                             "same_predicate_reverse_pairs": sum(same_reverse.values()),
                             "triples_participating_in_same_predicate_reverse": 2 * sum(same_reverse.values()),
                             "by_predicate": dict(same_reverse.most_common()),
                             "documents_with_same_predicate_reverse": reverse_documents,
                             "different_predicate_reverse_entity_pairs": different_reverse_pairs,
                             "documents_with_any_reverse_entity_pair": any_reverse_documents,
                             "any_reverse_entity_pairs": any_reverse_pairs,
                             "examples": reverse_examples},
        "alias_candidates": {"lexical_variant_groups": len(lexical_variants),
                             "lexical_examples": lexical_variants[:15],
                             "annotated_nonidentical_alias_triples": dict(alias_nonidentical),
                             "annotated_examples": alias_examples,
                             "note": "Candidates only: lexical similarity/equivalence annotations do not authorize automatic node merging."},
    }
