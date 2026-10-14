"""Validate the observed public training format without changing its labels."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

TRIPLES_KEY = "primitive-ontology-triples"
OFFICIAL_TRAIN_BLOB = "a074265ed063dfc7e34f0f6745d4b0aaca7d0425"


class SchemaError(ValueError):
    """An input does not satisfy the observed Task A JSON contract."""


def _text(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{location}: expected a nonempty string")
    return value


def validate_triples(value: object, location: str = TRIPLES_KEY) -> tuple[tuple[str, str, str], ...]:
    if not isinstance(value, list):
        raise SchemaError(f"{location}: expected a JSON array")
    result = []
    for index, triple in enumerate(value):
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            raise SchemaError(f"{location}[{index}]: expected [subject, predicate, object]")
        result.append(tuple(_text(part, f"{location}[{index}][{j}]") for j, part in enumerate(triple)))
    return tuple(result)


@dataclass(frozen=True)
class InferenceDocument:
    """The extraction interface deliberately contains no reference triples."""

    id: str
    context: str


@dataclass(frozen=True)
class TrainingSample:
    id: str
    context: str
    triples: tuple[tuple[str, str, str], ...]

    def inference_input(self) -> InferenceDocument:
        return InferenceDocument(self.id, self.context)


def git_blob_sha(content: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def validate_training_records(payload: object) -> tuple[TrainingSample, ...]:
    if not isinstance(payload, list):
        raise SchemaError("training data: expected a JSON array of records")
    samples = []
    ids = set()
    for index, record in enumerate(payload):
        location = f"records[{index}]"
        if not isinstance(record, dict) or set(record) != {"id", "context", TRIPLES_KEY}:
            raise SchemaError(f"{location}: expected exactly id, context, {TRIPLES_KEY}")
        sample_id = _text(record["id"], f"{location}.id")
        if sample_id in ids:
            raise SchemaError(f"{location}: duplicate id {sample_id!r}")
        ids.add(sample_id)
        samples.append(TrainingSample(
            sample_id,
            _text(record["context"], f"{location}.context"),
            validate_triples(record[TRIPLES_KEY], f"{location}.{TRIPLES_KEY}"),
        ))
    return tuple(samples)


def load_training_data(path: str | Path, expected_blob_sha: str | None = OFFICIAL_TRAIN_BLOB) -> tuple[TrainingSample, ...]:
    content = Path(path).read_bytes()
    if expected_blob_sha is not None and git_blob_sha(content) != expected_blob_sha:
        raise SchemaError("training file does not match the pinned official public train blob")
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SchemaError(f"invalid training JSON: {error}") from error
    return validate_training_records(payload)


def validate_prediction_records(payload: object, expected_ids: Iterable[str] | None = None) -> dict[str, tuple[tuple[str, str, str], ...]]:
    """Check a provisional id/triples output contract, not a confirmed submission spec.

    Empty predictions are valid. Context and reference annotations cannot be
    passed through as prediction fields. No model predictions are fabricated.
    """
    if not isinstance(payload, list):
        raise SchemaError("predictions: expected a JSON array")
    result = {}
    for index, record in enumerate(payload):
        if not isinstance(record, dict) or set(record) != {"id", TRIPLES_KEY}:
            raise SchemaError(f"predictions[{index}]: expected exactly id and {TRIPLES_KEY}")
        sample_id = _text(record["id"], f"predictions[{index}].id")
        if sample_id in result:
            raise SchemaError(f"predictions[{index}]: duplicate id {sample_id!r}")
        result[sample_id] = validate_triples(record[TRIPLES_KEY], f"predictions[{index}].{TRIPLES_KEY}")
    if expected_ids is not None:
        expected = set(expected_ids)
        if set(result) != expected:
            raise SchemaError(f"prediction IDs differ: missing={sorted(expected-set(result))}, extra={sorted(set(result)-expected)}")
    return result
