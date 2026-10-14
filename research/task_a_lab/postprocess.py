"""A/B/C postprocessing with per-edge lineage. C is a lexical CPU prototype.

No semantic model, synonym dictionary, relation reversal or relation clustering
is implemented. Candidates are merged only within one document, after matching
lexical form, observed type roles, signed relation roles and local text context.
"""

from collections import defaultdict
import re
from typing import Sequence

from research.llms4ol2026.schema import InferenceDocument
from .contracts import Triple


def normalize(value):
    return " ".join(value.lower().split())


def lexical_key(value):
    return " ".join(value.replace("-", " ").replace("_", " ").split())


def context_features(label, text):
    pattern = re.compile(r"(?<!\w)" + re.escape(label).replace(r"\ ", r"\s+") + r"(?!\w)", re.I)
    features = set()
    for sentence in re.split(r"[.!?;\n]+", text):
        if pattern.search(sentence):
            features.update(re.findall(r"\w+", sentence.lower()))
    return features


def roles(triples):
    kinds, signed = defaultdict(set), defaultdict(set)
    for s, p, o in triples:
        signed[s].add(("OUT", p))
        signed[o].add(("IN", p))
        if p == "is-a":
            kinds[s].add("class")
            kinds[o].add("class")
        elif p == "instance-of":
            kinds[s].add("individual")
            kinds[o].add("class")
    return kinds, signed


class Processing:
    def __init__(self, mode="A", context_threshold=0.6):
        if mode not in {"A", "B", "C"} or not 0 <= context_threshold <= 1:
            raise ValueError("Invalid processing mode/threshold")
        self.mode, self.threshold = mode, context_threshold

    def process(self, document: InferenceDocument, triples: Sequence[Triple]):
        raw = [t.as_tuple() for t in triples]
        normalized = raw if self.mode == "A" else [tuple(normalize(v) for v in t) for t in raw]
        labels = sorted({v for s, _, o in normalized for v in (s, o)})
        mapping = {label: label for label in labels}
        merged = []
        if self.mode == "C":
            kinds, signed = roles(normalized)
            features = {label: context_features(label, document.context) for label in labels}
            linked = {frozenset((s, o)) for s, _, o in normalized}

            def compatible(a, b):
                if lexical_key(a) != lexical_key(b) or not kinds[a] or kinds[a] != kinds[b]:
                    return False
                if kinds[a] == {"class", "individual"} or signed[a] != signed[b]:
                    return False
                if frozenset((a, b)) in linked:
                    return False  # Never create a self-loop by collapsing related endpoints.
                union = features[a] | features[b]
                return bool(union) and len(features[a] & features[b]) / len(union) >= self.threshold

            # Complete-link groups avoid transitive chaining past a failed constraint.
            groups = []
            for label in labels:
                target = next((g for g in groups if all(compatible(label, other) for other in g)), None)
                if target is None:
                    groups.append([label])
                else:
                    target.append(label)
            for members in groups:
                representative = min(members)
                for member in members:
                    mapping[member] = representative
                if len(members) > 1:
                    merged.append({"representative": representative, "members": members,
                                   "rule": "lexical+type+signed-relation+context complete-link",
                                   "context_threshold": self.threshold})
        transformed = [(mapping[s], p, mapping[o]) for s, p, o in normalized]
        edges = []
        if self.mode == "A":
            edges = [{"triple": dict(zip(("subject", "predicate", "object"), t)), "source_indices": [i]}
                     for i, t in enumerate(transformed)]
        else:
            origins = defaultdict(list)
            for i, t in enumerate(transformed):
                origins[t].append(i)
            edges = [{"triple": dict(zip(("subject", "predicate", "object"), t)), "source_indices": indices}
                     for t, indices in sorted(origins.items())]
        return {
            "mode": self.mode, "raw_triples": [t.as_dict() for t in triples],
            "triples": [e["triple"] for e in edges], "edges": edges, "merge_groups": merged,
            "concept_lineage": [{"source": source, "normalized": normal, "target": mapping[normal]}
                                for source, normal in sorted({(raw[i][j], normalized[i][j])
                                                            for i in range(len(raw)) for j in (0, 2)})],
            "relation_lineage": [{"source": source, "target": target}
                                 for source, target in sorted({(raw[i][1], normalized[i][1]) for i in range(len(raw))})],
            "semantic_model_used": False,
        }
