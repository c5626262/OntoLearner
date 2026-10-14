"""Deterministic, standard-library-only Task A retrieval/rule baseline."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

KEYS = ("subject", "predicate", "object")
TOKEN = re.compile(r"\w+", re.UNICODE)
RULES = (
    (r" is an? instance of ", "instance-of"),
    (r" is an? (?:type|kind|subclass) of ", "is-a"),
    (r" is part of ", "part_of"),
    (r" is located in ", "located in"),
)


def tokens(text):
    return TOKEN.findall(text.casefold())


def read_rows(path):
    """Read a JSON array or JSONL, rejecting ambiguous document identifiers."""
    text = Path(path).read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    if not isinstance(rows, list):
        raise ValueError("Expected a JSON array or JSONL")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each document must be an object")
        doc_id = row.get("id")
        if isinstance(doc_id, bool) or not isinstance(doc_id, (str, int)):
            raise ValueError("id must be a string or integer")
        if doc_id in seen:
            raise ValueError(f"Duplicate id: {doc_id!r}")
        seen.add(doc_id)
    return rows


def triple_tuple(row):
    if not isinstance(row, dict) or any(
        not isinstance(row.get(key), str) or not row[key].strip() for key in KEYS
    ):
        raise ValueError("Triples require nonempty subject, predicate, object strings")
    return tuple(row[key].strip() for key in KEYS)


def context_of(row):
    context = row.get("context")
    if not isinstance(context, str):
        raise ValueError("context must be a string")
    return context


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Baseline:
    """Fit TF-IDF on train contexts only; retrieve directed train triples.

    Only transfer triples whose two endpoint token sequences occur in the query.
    Query IDs and any attached labels are never used by predict().
    """

    def __init__(self, train, top_k=3, min_similarity=0.1, rules=True):
        if top_k < 1 or not 0 <= min_similarity <= 1:
            raise ValueError("top_k must be positive and similarity in [0, 1]")
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.rules = rules
        counts = [Counter(tokens(context_of(row))) for row in train]
        self.triples = [sorted({triple_tuple(t) for t in row["triples"]}) for row in train]
        df = Counter(term for count in counts for term in count)
        self.idf = {term: math.log((1 + len(train)) / (1 + freq)) + 1 for term, freq in df.items()}
        self.vectors = [self.vector(count) for count in counts]
        self.postings = defaultdict(list)
        for index, vector in enumerate(self.vectors):
            for term, weight in vector.items():
                self.postings[term].append((index, weight))
        self.lexicon = {}
        classes, instances = set(), set()
        for triples in self.triples:
            for subject, predicate, obj in triples:
                if predicate == "is-a":
                    classes.update((subject, obj))
                elif predicate == "instance-of":
                    instances.add(subject)
                    classes.add(obj)
                for label in (subject, obj):
                    parts = tokens(label)
                    if not parts:
                        continue
                    node = self.lexicon
                    for term in parts:
                        node = node.setdefault(term, {})
                    node.setdefault(None, set()).add(label)
        self.individuals = {tuple(tokens(label)) for label in instances} - {
            tuple(tokens(label)) for label in classes}

    def mention_rules(self, context):
        """Combine train-only entity labels when local text states a relation."""
        for sentence in re.split(r"[.!?;\n]+", context):
            words = list(TOKEN.finditer(sentence))
            mentions = []
            for start in range(len(words)):
                node, longest = self.lexicon, None
                for end in range(start, len(words)):
                    node = node.get(words[end].group().casefold())
                    if node is None:
                        break
                    if None in node:
                        longest = (words[start].start(), words[end].end(), sorted(node[None])[0])
                if longest:
                    mentions.append(longest)
            for left in mentions:
                for right in mentions:
                    if left[1] >= right[0] or left[2] == right[2]:
                        continue
                    between = sentence[left[1]:right[0]].casefold().strip(" ,")
                    if len(between) > 100:
                        continue
                    predicate = None
                    if re.fullmatch(r"(?:is |are )?(?:an? )?instance of(?: an?| the)?", between):
                        predicate = "instance-of"
                    elif re.fullmatch(
                        r"(?:is|are|was|represents?|constitutes?) "
                        r"(?:(?:explicitly |chemically )?(?:classified|defined|recognized|categorized) as )?"
                        r"(?:an? |the )?(?:(?:subclass|type|kind|specialized form|member) (?:of|within) )?"
                        r"(?:an? |the )?", between + " "
                    ):
                        predicate = "instance-of" if tuple(tokens(left[2])) in self.individuals else "is-a"
                    elif between in {"is part of", "is a part of"}:
                        predicate = "part_of"
                    elif between == "is located in":
                        predicate = "located in"
                    if predicate:
                        yield left[2], predicate, right[2]
                    if between in {"such as", "such as the", "including", "including the"}:
                        predicate = "instance-of" if tuple(tokens(right[2])) in self.individuals else "is-a"
                        yield right[2], predicate, left[2]

    def vector(self, counts):
        weighted = {term: freq * self.idf[term] for term, freq in sorted(counts.items()) if term in self.idf}
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        return {term: value / norm for term, value in weighted.items()} if norm else {}

    def predict(self, context):
        if not isinstance(context, str):
            raise ValueError("context must be a string")
        query = self.vector(Counter(tokens(context)))
        scores = defaultdict(float)
        for term, value in query.items():
            for index, weight in self.postings.get(term, ()):
                scores[index] += value * weight
        ranked = sorted(((score, index) for index, score in scores.items()),
                        key=lambda item: (-item[0], item[1]))
        query_tokens = " " + " ".join(tokens(context)) + " "
        result = set()
        for score, index in ranked[:self.top_k]:
            if score <= 0 or score < self.min_similarity:
                continue
            for triple in self.triples[index]:
                endpoints = [tokens(triple[0]), tokens(triple[2])]
                if all(parts and " " + " ".join(parts) + " " in query_tokens for parts in endpoints):
                    result.add(triple)
        if self.rules:
            result.update(self.mention_rules(context))
            # Conservative full-sentence patterns; generic "X is a Y" is ambiguous.
            for sentence in re.split(r"[.!?;\n]+", context):
                sentence = sentence.strip()
                for separator, predicate in RULES:
                    match = re.fullmatch(r"(.+?)" + separator + r"(.+?)", sentence, re.IGNORECASE)
                    if match:
                        subject, obj = (part.strip() for part in match.groups())
                        if all(1 <= len(tokens(part)) <= 8 for part in (subject, obj)):
                            result.add((subject, predicate, obj))
        return [dict(zip(KEYS, triple)) for triple in sorted(result)]


def predict_rows(model, rows):
    # Explicit projection: neither gold triples nor IDs enter the predictor.
    return [{"id": row["id"], "triples": model.predict(context_of(row))} for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="JSON/JSONL with id and context")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.1)
    parser.add_argument("--no-rules", action="store_true")
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".manifest.json")
    if len({p.resolve() for p in (args.train, args.input, args.output, manifest_path)}) != 4:
        parser.error("Train, input, output and manifest paths must be distinct")
    train, inputs = read_rows(args.train), read_rows(args.input)
    if not train:
        parser.error("Train must not be empty")
    overlap = {row["id"] for row in train} & {row["id"] for row in inputs}
    if overlap:
        parser.error("Train/input IDs overlap; check split provenance")
    model = Baseline(train, args.top_k, args.min_similarity, not args.no_rules)
    predictions = predict_rows(model, inputs)
    write_json(args.output, predictions)
    write_json(manifest_path, {
        "algorithm": "train-tfidf-cosine-grounded-triples+train-lexicon-rules-v2",
        "train_sha256": digest(args.train), "input_sha256": digest(args.input),
        "prediction_sha256": digest(args.output), "source_sha256": digest(__file__),
        "train_documents": len(train), "predicted_documents": len(inputs),
        "top_k": args.top_k, "min_similarity": args.min_similarity, "rules": not args.no_rules,
        "device": "cpu", "dependencies": "Python standard library",
    })


if __name__ == "__main__":
    main()
