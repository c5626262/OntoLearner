"""Execute pinned official exact/fuzzy functions without loading an LLM.

The vendored source is unmodified. AST selection avoids its module-level
SentenceTransformer construction and disables all semantic/model code.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import networkx as nx

from .schema import SchemaError, git_blob_sha, validate_triples

OFFICIAL_METRIC_BLOB = "6c2971bb0c500fdc34bc52abaa7d309a61d6056d"
OFFICIAL_SOURCE = Path(__file__).parent / "vendor" / "graph_similarity.py"
EXACT_FUNCTIONS = {
    "build_graph", "edge_f1", "get_neighborhood", "neighborhood_similarity",
    "normalize", "normalize_triples", "build_taxonomy_graph", "taxonomy_similarity", "exact_match",
}
FUZZY_FUNCTIONS = {"triple_similarity_fuzzy", "align_triples_fuzzy", "fuzzy_match"}


@lru_cache(maxsize=2)
def load_official_functions(include_fuzzy: bool = False) -> SimpleNamespace:
    content = OFFICIAL_SOURCE.read_bytes()
    if git_blob_sha(content) != OFFICIAL_METRIC_BLOB:
        raise RuntimeError("official evaluator source does not match the pinned Git blob")
    tree = ast.parse(content, filename=str(OFFICIAL_SOURCE))
    names = EXACT_FUNCTIONS | (FUZZY_FUNCTIONS if include_fuzzy else set())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise RuntimeError("official evaluator function contract changed")
    constants = [node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "TAXONOMY_RELATIONS" for target in node.targets)]
    namespace = {"nx": nx, "defaultdict": defaultdict}
    if include_fuzzy:
        import numpy as np
        from rapidfuzz import fuzz
        from scipy.optimize import linear_sum_assignment
        namespace.update(np=np, fuzz=fuzz, linear_sum_assignment=linear_sum_assignment)
    selected = ast.Module(body=constants + functions, type_ignores=[])
    exec(compile(selected, str(OFFICIAL_SOURCE), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def score_triples(gold, predicted, mode: str = "exact") -> dict[str, float]:
    gold = list(validate_triples(list(gold), "gold"))
    predicted = list(validate_triples(list(predicted), "predicted"))
    if mode == "exact":
        return load_official_functions().exact_match(gold, predicted)
    if mode == "fuzzy":
        return load_official_functions(True).fuzzy_match(gold, predicted)
    raise ValueError("only official exact/fuzzy modes are available; semantic mode needs a real embedding model")


def directed_taxonomy_diagnostic(gold, predicted) -> float:
    """Independent diagnostic: compare parent reachability in child->parent DAGs.

    This score is NOT the official Taxonomy Similarity and is not averaged into
    Graph Similarity. Cycles are rejected instead of silently repaired.
    """
    graphs = []
    for triples in (gold, predicted):
        graph = nx.DiGraph()
        for subject, predicate, obj in validate_triples(list(triples)):
            if " ".join(predicate.lower().split()) == "is-a":
                graph.add_edge(subject, obj)
        if not nx.is_directed_acyclic_graph(graph):
            raise SchemaError("directed taxonomy diagnostic requires an acyclic is-a graph")
        graphs.append(graph)
    left, right = graphs
    scores = []
    for node in set(left) | set(right):
        a = nx.descendants(left, node) if node in left else set()
        b = nx.descendants(right, node) if node in right else set()
        if a | b:
            scores.append(len(a & b) / len(a | b))
    return sum(scores) / len(scores) if scores else 0.0


def ontology_graph(triples) -> nx.MultiDiGraph:
    """Retain predicate, direction, and multiple relations on an entity pair."""
    graph = nx.MultiDiGraph()
    for subject, predicate, obj in validate_triples(list(triples)):
        graph.add_edge(subject, obj, key=predicate, label=predicate)
    return graph
