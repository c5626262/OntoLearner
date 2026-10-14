"""Offline baseline contract, leakage and directed evaluation regressions."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from baselines.task_a_cpu.evaluate import evaluate
from baselines.task_a_cpu.predict import Baseline, predict_rows, read_rows

ROOT = Path(__file__).resolve().parents[1]


def triple(subject="Cat", predicate="is-a", obj="Animal"):
    return {"subject": subject, "predicate": predicate, "object": obj}


@pytest.fixture
def train():
    return [{"id": "train-1", "context": "Cat Animal mammal", "triples": [triple()]}]


def test_train_retrieval_preserves_direction(train):
    model = Baseline(train, rules=False)
    assert model.predict("Animal Cat") == [triple()]
    assert model.predict("Cat") == []
    assert model.predict("Catfish Animal") == []
    assert model.predict("") == []
    assert model.predict("unrelated words") == []


def test_validation_labels_and_ids_cannot_influence_predictions(train):
    model = Baseline(train)
    clean = [{"id": "v", "context": "Cat Animal"}]
    poisoned = [{"id": "train-1", "context": "Cat Animal", "triples": [triple("LEAK")],
                 "gold": {"subject": "LEAK"}}]
    assert predict_rows(model, clean)[0]["triples"] == predict_rows(model, poisoned)[0]["triples"]
    assert "LEAK" not in json.dumps(predict_rows(model, poisoned))


@pytest.mark.parametrize(("context", "expected"), [
    ("Cat is a type of Animal.", [triple()]),
    ("Milo is an instance of Cat.", [triple("Milo", "instance-of", "Cat")]),
    ("Wheel is part of Car.", [triple("Wheel", "part_of", "Car")]),
    ("Museum is located in Paris.", [triple("Museum", "located in", "Paris")]),
    ("Cat is an Animal.", []),
    ("Cat is not a type of Animal.", []),
])
def test_explicit_rules(context, expected):
    assert Baseline([]).predict(context) == expected


def test_determinism_and_deduplication(train):
    train[0]["triples"] *= 2
    model = Baseline(train)
    assert model.predict("Cat Animal") == model.predict("Cat Animal") == [triple()]


def test_evaluation_direction_relation_and_document_identity():
    gold = [{"id": "a", "triples": [triple(), triple("Milo", "instance-of", "Cat")]},
            {"id": "b", "triples": [triple("Cat", "likes", "Milk")]}]
    pred = [{"id": "a", "triples": [triple("Animal", "is-a", "Cat"),
                                      triple("Milo", "is-a", "Cat")]},
            {"id": "b", "triples": [triple("Cat", "likes", "Milk"), triple()]}]
    report = evaluate(gold, pred)
    assert report["overall"]["tp"] == 1
    assert report["by_relation"]["is-a"]["f1"] == 0
    assert report["by_relation"]["instance-of"]["fn"] == 1
    assert report["by_relation"]["other"]["f1"] == 1
    assert report["official_graph_similarity"] is None


def test_missing_duplicate_or_extra_predictions_rejected():
    with pytest.raises(ValueError):
        evaluate([{"id": "a", "triples": []}], [])
    with pytest.raises(ValueError):
        evaluate([], [{"id": "a", "triples": []}])
    with pytest.raises(ValueError):
        evaluate([], [{"id": "a", "triples": []}] * 2)


def test_examples_are_real_and_capped():
    gold = [{"id": i, "triples": [triple()]} for i in range(12)]
    pred = [{"id": i, "triples": [triple()] if i < 6 else []} for i in range(12)]
    report = evaluate(gold, pred)
    assert len(report["success_examples"]) == len(report["failure_examples"]) == 5
    assert all(d["metrics"]["tp"] == 1 for d in report["success_examples"])
    assert all(d["metrics"]["fn"] == 1 for d in report["failure_examples"])


def test_jsonl_and_duplicate_id_validation(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text('{"id": "x", "context": "猫"}\n', encoding="utf-8")
    assert read_rows(path)[0]["context"] == "猫"
    path.write_text('[{"id": "x"}, {"id": "x"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        read_rows(path)


def test_cli_reproducible_and_protects_inputs(tmp_path, train):
    train_path, inputs = tmp_path / "train.json", tmp_path / "input.json"
    output = tmp_path / "predictions.json"
    train_path.write_text(json.dumps(train), encoding="utf-8")
    inputs.write_text(json.dumps([{"id": "v", "context": "Cat Animal"}]), encoding="utf-8")
    command = [sys.executable, str(ROOT / "baselines/task_a_cpu/predict.py"),
               "--train", str(train_path), "--input", str(inputs), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    first = output.read_bytes(), output.with_suffix(".manifest.json").read_bytes()
    subprocess.run(command, check=True, capture_output=True)
    assert first == (output.read_bytes(), output.with_suffix(".manifest.json").read_bytes())
    assert json.loads(first[0]) == [{"id": "v", "triples": [triple()]}]
    assert subprocess.run(command[:-1] + [str(inputs)], capture_output=True).returncode != 0
    inputs.write_text(json.dumps(train), encoding="utf-8")
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_official_adapter_scores_saved_predictions_only(tmp_path, monkeypatch):
    from baselines.task_a_cpu import experiment
    from research.llms4ol2026.schema import TrainingSample

    sample = TrainingSample("v", "Ignored during evaluation", (("Cat", "is-a", "Animal"),))
    monkeypatch.setattr(experiment, "partitions", lambda *args: {"validation": [sample]})
    monkeypatch.setattr(experiment, "digest", lambda *args: "fixture-hash")
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(json.dumps([{"id": "v", "triples": [triple(" CAT ", "is-a", "animal")]}]))
    report = experiment.score_saved(None, tmp_path, pred_path, tmp_path / "scores")
    assert report["official_graph_similarity"] == 1
    assert report["by_relation"]["is-a"]["f1"] == 1
    pred_path.write_text(json.dumps([{"id": "v", "triples": []}]))
    report = experiment.score_saved(None, tmp_path, pred_path, tmp_path / "empty")
    assert report["official_graph_similarity"] == 0
    assert report["overall"]["tp"] == 0


def test_prepare_projects_only_context_and_training_labels(tmp_path, monkeypatch):
    from baselines.task_a_cpu import experiment
    from research.llms4ol2026.schema import TrainingSample

    monkeypatch.setattr(experiment, "partitions", lambda *args: {
        "train": [TrainingSample("t", "Cat Animal", (("Cat", "is-a", "Animal"),))],
        "validation": [TrainingSample("v", "New context", (("SECRET", "is-a", "GOLD"),))],
    })
    monkeypatch.setattr(experiment, "digest", lambda *args: "fixture-hash")
    output = tmp_path / "prepared"
    experiment.prepare(None, tmp_path, output)
    assert json.loads((output / "validation-contexts.json").read_text()) == [{"id": "v", "context": "New context"}]
    assert "SECRET" not in "".join(p.read_text() for p in output.iterdir())
    assert json.loads((output / "train.json").read_text())[0]["triples"] == [triple()]


def test_partitions_reject_changed_source_or_ids(tmp_path):
    from baselines.task_a_cpu.experiment import partitions
    from baselines.task_a_cpu.predict import digest

    source = tmp_path / "source.json"
    source.write_text("[]")
    manifest = {"source": {"sha256": "wrong"}, "ids": {"train_sha256": "wrong"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Source"):
        partitions(source, tmp_path)
    manifest["source"]["sha256"] = digest(source)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "train_ids.json").write_text("[]")
    with pytest.raises(ValueError, match="ID SHA"):
        partitions(source, tmp_path)


def test_train_lexicon_generates_unseen_directed_relations():
    model = Baseline([{"context": "Cat Animal Dog Pet Milo", "triples": [
        triple(), triple("Dog", "is-a", "Pet"), triple("Milo", "instance-of", "Cat")]}])
    assert list(model.mention_rules("Dog is a type of Animal.")) == [("Dog", "is-a", "Animal")]
    assert list(model.mention_rules("Animal such as Dog.")) == [("Dog", "is-a", "Animal")]
    assert list(model.mention_rules("Milo is a Pet.")) == [("Milo", "instance-of", "Pet")]
    assert list(model.mention_rules("Dog is not an Animal.")) == []
    assert list(model.mention_rules("Unseen is an Animal.")) == []


def test_sparse_retrieval_matches_brute_force(train):
    from collections import Counter
    from baselines.task_a_cpu.predict import tokens

    train += [{"context": "Cat Pet", "triples": [triple("Cat", "is-a", "Pet")]}]
    model = Baseline(train, rules=False, top_k=1)
    for context in ("Cat Animal Pet", "Pet Cat", "Animal", "unknown"):
        query = model.vector(Counter(tokens(context)))
        brute = [sum(value * vector.get(term, 0) for term, value in query.items()) for vector in model.vectors]
        sparse = [0.0] * len(train)
        for term, value in query.items():
            for index, weight in model.postings.get(term, ()):
                sparse[index] += value * weight
        assert sparse == brute
