"""Grouped split checks using only synthetic text and annotations."""

from dataclasses import asdict
import json
import random

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from research.llms4ol2026.schema import SchemaError, TrainingSample
from research.llms4ol2026.splitting import (
    build_groups, create_split, find_similar_documents, inference_documents,
    leakage_audit, relation_comparison, save_split,
)


def samples(n=20):
    return tuple(TrainingSample(f"doc-{index:03}", f"independent document subject{index}",
                                ((f"child{index}", "is-a", f"parent{index}"),)) for index in range(n))


def test_fixed_seed_reproducible_and_input_order_invariant():
    data = samples()
    a = create_split(data, similarity_pairs=())
    b = create_split(tuple(reversed(data)), similarity_pairs=())
    assert a == b
    assert len(a.train_ids) == 16 and len(a.validation_ids) == 4
    assert not set(a.train_ids) & set(a.validation_ids)
    assert set(a.train_ids) | set(a.validation_ids) == {row.id for row in data}
    assert a.config["seed"] == 42
    assert a.config["similarity"]["token_pattern"] == TfidfVectorizer().token_pattern


def test_seed_does_not_change_global_random_state_and_can_change_selection():
    random.seed(100)
    before = random.getstate()
    a = create_split(samples(), seed=42, similarity_pairs=())
    assert random.getstate() == before
    b = create_split(samples(), seed=43, similarity_pairs=())
    assert a.validation_ids != b.validation_ids


def test_ceil_rounding_for_nondivisible_document_count():
    result = create_split(samples(11), similarity_pairs=())
    assert len(result.validation_ids) == 3


def test_shared_facts_and_normalized_contexts_group_transitively():
    data = list(samples(5))
    data[0] = TrainingSample("doc-000", "first passage", (("Child", " IS-A ", "Parent"),))
    data[1] = TrainingSample("doc-001", " Second   passage ", ((" child ", "is-a", "parent"),))
    data[2] = TrainingSample("doc-002", "second passage", (("different", "related", "object"),))
    groups, metadata = build_groups(data)
    assert ("doc-000", "doc-001", "doc-002") in groups
    assert metadata["shared_normalized_triple_types"] == 1
    assert metadata["duplicate_normalized_context_groups"] == 1
    result = create_split(data, similarity_pairs=())
    assert set(groups[0]).issubset(result.train_ids) or set(groups[0]).issubset(result.validation_ids)
    audit = leakage_audit(data, result, ())
    assert audit["shared_normalized_triples"] == 0
    assert audit["duplicate_normalized_contexts"] == 0


def test_similarity_grouping_boundary_and_unknown_ids():
    data = samples(4)
    groups, metadata = build_groups(data, (("doc-000", "doc-001", .8), ("doc-001", "doc-002", .79)))
    assert ("doc-000", "doc-001") in groups
    assert metadata["near_document_pairs_grouped"] == 1
    with pytest.raises(SchemaError, match="unknown"):
        build_groups(data, (("missing", "doc-001", .9),))


def test_similarity_scanner_finds_identical_text_without_using_annotations():
    data = (TrainingSample("a", "alpha beta gamma document", (("a", "r", "b"),)),
            TrainingSample("b", "alpha beta gamma document", (("different", "q", "other"),)),
            TrainingSample("c", "unrelated zebra forest picnic", ()))
    assert find_similar_documents(data, .8)[0][:2] == ("a", "b")
    assert find_similar_documents(data, .8)[0][2] == pytest.approx(1)
    assert find_similar_documents(tuple(reversed(data)), .8) == find_similar_documents(data, .8)


def test_closest_feasible_group_count_is_reported():
    data = tuple(TrainingSample(str(i), f"context group {i // 2}", ((f"child{i // 2}", "is-a", "parent"),)) for i in range(6))
    result = create_split(data, similarity_pairs=())
    assert result.config["target_validation_documents"] == 2
    assert len(result.validation_ids) == 2
    result = create_split(data, validation_fraction=.1, similarity_pairs=())
    assert result.config["target_validation_documents"] == 1
    assert len(result.validation_ids) == 2


def test_one_leakage_group_cannot_be_split():
    data = tuple(TrainingSample(str(i), "same context", (("a", "r", "b"),)) for i in range(4))
    with pytest.raises(SchemaError, match="one leakage group"):
        create_split(data, similarity_pairs=())


@pytest.mark.parametrize("kwargs", [{"validation_fraction": 0}, {"validation_fraction": 1}, {"candidates": 0}, {"near_threshold": 0}])
def test_invalid_split_configuration(kwargs):
    with pytest.raises(ValueError):
        create_split(samples(), similarity_pairs=(), **kwargs)


def test_too_few_or_duplicate_documents_rejected():
    with pytest.raises(SchemaError, match="at least two"):
        create_split(samples(1), similarity_pairs=())
    with pytest.raises(SchemaError, match="duplicate"):
        create_split((samples(1)[0], samples(1)[0]), similarity_pairs=())


def test_saved_ids_manifest_and_no_reference_annotations_in_inference(tmp_path):
    data = samples()
    source = tmp_path / "synthetic-source.json"
    source.write_text(json.dumps([asdict(row) for row in data]))
    split = create_split(data, similarity_pairs=())
    first = save_split(data, split, tmp_path / "saved", source, ())
    before = {path.name: path.read_bytes() for path in (tmp_path / "saved").iterdir()}
    assert save_split(data, split, tmp_path / "saved", source, ()) == first
    assert {path.name: path.read_bytes() for path in (tmp_path / "saved").iterdir()} == before
    assert json.loads((tmp_path / "saved/train_ids.json").read_text()) == list(split.train_ids)
    docs = inference_documents(data, split, "validation")
    assert all(set(asdict(doc)) == {"id", "context"} for doc in docs)
    assert [doc.id for doc in docs] == list(split.validation_ids)
    with pytest.raises(ValueError):
        inference_documents(data, split, "official-test")


def test_different_existing_split_artifact_is_not_overwritten(tmp_path):
    source = tmp_path / "synthetic.json"
    source.write_text("[]")
    output = tmp_path / "out"
    output.mkdir()
    (output / "train_ids.json").write_text("do not overwrite")
    with pytest.raises(FileExistsError):
        save_split(samples(), create_split(samples(), similarity_pairs=()), output, source, ())
    assert (output / "train_ids.json").read_text() == "do not overwrite"


def test_rare_relation_absence_and_aggregate_counts():
    data = list(samples())
    data[0] = TrainingSample(data[0].id, data[0].context, data[0].triples + (("a", "rare relation", "b"),))
    split = create_split(data, similarity_pairs=())
    comparison = relation_comparison(data, split)
    assert sum(row["train_triples"] + row["validation_triples"] for row in comparison["labels"]) == 21
    rare = next(row for row in comparison["labels"] if row["label"] == "rare relation")
    assert rare["full_documents"] == 1
    assert ("rare relation" in comparison["rare_labels_missing_validation"]) == (rare["validation_documents"] == 0)


def test_audit_reports_shared_entities_separately_from_shared_facts():
    data = tuple(TrainingSample(str(i), f"unique context {i}", ((f"child{i}", "is-a", "common parent"),)) for i in range(10))
    split = create_split(data, similarity_pairs=())
    train_id, val_id = split.train_ids[0], split.validation_ids[0]
    audit = leakage_audit(data, split, ((train_id, val_id, .75),))
    assert audit["shared_normalized_triples"] == 0
    assert audit["shared_node_strings"] == 1
    assert audit["cross_partition_pairs_at_or_above_floor"] == 1
    assert audit["cross_partition_pairs_at_or_above_grouping_threshold"] == 0


def test_empty_annotation_sets_have_zero_distributions():
    data = tuple(TrainingSample(str(i), f"unique context {i}", ()) for i in range(5))
    split = create_split(data, similarity_pairs=())
    comparison = relation_comparison(data, split)
    assert comparison["labels"] == []
    assert all(row["validation_fraction"] == 0 for row in comparison["categories"].values())
