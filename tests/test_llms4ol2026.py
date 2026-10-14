"""Synthetic CPU checks for 2026 Flagship preparation; no model predictions."""

import json

import pytest

from research.llms4ol2026.analysis import analyze_training
from research.llms4ol2026.evaluator import (
    directed_taxonomy_diagnostic, load_official_functions, ontology_graph, score_triples,
)
from research.llms4ol2026.schema import (
    SchemaError, git_blob_sha, load_training_data,
    validate_prediction_records, validate_training_records, validate_triples,
)


def record(sample_id="synthetic", triples=None):
    return {"id": sample_id, "context": "Synthetic: a dog is an animal.",
            "primitive-ontology-triples": triples if triples is not None else [["dog", "is-a", "animal"]]}


def test_training_input_separates_reference_from_inference():
    sample, = validate_training_records([record()])
    document = sample.inference_input()
    assert document.id == "synthetic"
    assert not hasattr(document, "triples")
    assert not hasattr(document, "primitive-ontology-triples")


@pytest.mark.parametrize("payload", [{}, [None], [{"id": "x", "context": "c"}],
                                        [record(), record()], [dict(record(), context=" ")],
                                        [dict(record(), id=1)], [dict(record(), secret="unexpected")]])
def test_training_schema_rejects_invalid_records(payload):
    with pytest.raises(SchemaError):
        validate_training_records(payload)


@pytest.mark.parametrize("triples", ["bad", [["s", "p"]], [["s", "p", "o", "x"]],
                                      [["s", "p", 1]], [["s", "", "o"]]])
def test_triple_schema_rejects_invalid_shape_or_type(triples):
    with pytest.raises(SchemaError):
        validate_triples(triples)


def test_empty_predictions_and_id_alignment():
    result = validate_prediction_records([{"id": "x", "primitive-ontology-triples": []}], ["x"])
    assert result == {"x": ()}
    with pytest.raises(SchemaError, match="IDs differ"):
        validate_prediction_records([], ["x"])
    with pytest.raises(SchemaError, match="exactly"):
        validate_prediction_records([record()])


def test_training_file_integrity_is_checked(tmp_path):
    content = json.dumps([record()]).encode()
    path = tmp_path / "synthetic.json"
    path.write_bytes(content)
    assert len(load_training_data(path, git_blob_sha(content))) == 1
    with pytest.raises(SchemaError, match="pinned"):
        load_training_data(path)


@pytest.mark.parametrize("gold,predicted,expected", [
    ([("dog", "is-a", "animal")], [("dog", "is-a", "animal")], (1, 1, 1, 1)),
    ([("dog", "is-a", "animal")], [("animal", "is-a", "dog")], (0, 0, 1, 1/3)),
    ([("sensor", "measures", "temperature")], [("sensor", "measures", "temperature")], (1, 1, 0, 2/3)),
    ([], [], (1, 0, 0, 1/3)),
    ([("dog", "is-a", "animal")], [], (0, 0, 0, 0)),
    ([("a", "is-a", "b")], [("x", "is-a", "y")], (0, 0, 0, 0)),
    ([("dog", "is-a", "animal"), ("cat", "is-a", "animal")],
     [("dog", "is-a", "animal")], (2/3, 1/2, 1/2, 5/9)),
    ([("a", "is-a", "b"), ("b", "is-a", "c")], [("a", "is-a", "c")], (0, 0, 1/3, 1/9)),
])
def test_official_exact_components_against_hand_calculations(gold, predicted, expected):
    scores = score_triples(gold, predicted)
    assert list(scores.values()) == pytest.approx(expected)


def test_official_normalization_and_duplicate_invariance():
    gold = [("Dog", "is-a", "Animal")]
    predicted = [(" DOG ", " IS-A ", " animal ")] * 2
    assert score_triples(gold, predicted)["graph_similarity"] == 1


def test_neighborhood_uses_incoming_and_outgoing_edges():
    functions = load_official_functions()
    assert functions.get_neighborhood([("a", "r", "b")]) == {
        "a": {("OUT", "r", "b")}, "b": {("IN", "r", "a")},
    }


def test_official_taxonomy_combines_ancestors_and_descendants():
    functions = load_official_functions()
    assert functions.taxonomy_similarity([("a", "is-a", "b")], [("b", "is-a", "a")]) == 1
    assert directed_taxonomy_diagnostic([("a", "is-a", "b")], [("b", "is-a", "a")]) == 0


def test_cycles_are_visible_and_diagnostic_rejects_them():
    cyclic = [("a", "is-a", "b"), ("b", "is-a", "a")]
    assert score_triples(cyclic, cyclic)["graph_similarity"] == 1
    with pytest.raises(SchemaError, match="acyclic"):
        directed_taxonomy_diagnostic(cyclic, cyclic)


def test_multiple_predicates_survive_in_new_graph():
    triples = [("a", "r", "b"), ("a", "q", "b")]
    assert ontology_graph(triples).number_of_edges("a", "b") == 2
    # Document a limitation of the official auxiliary graph builder.
    assert load_official_functions().build_graph(triples).number_of_edges("a", "b") == 1
    assert score_triples(triples, triples)["edge_f1"] == 1


def test_fuzzy_alignment_is_evaluator_behavior_not_model_inference():
    gold = [("dog", "is-a", "animal")]
    synthetic_typo = [("dogg", "is-a", "animal")]
    assert score_triples(gold, synthetic_typo)["edge_f1"] == 0
    assert score_triples(gold, synthetic_typo, "fuzzy")["graph_similarity"] == 1
    assert synthetic_typo == [("dogg", "is-a", "animal")]


def test_semantic_mode_is_unavailable_without_real_model():
    with pytest.raises(ValueError, match="real embedding"):
        score_triples([], [], "semantic")


def test_adapter_does_not_import_or_construct_sentence_transformer():
    functions = load_official_functions()
    namespace = functions.exact_match.__globals__
    assert "model" not in namespace
    assert "SentenceTransformer" not in namespace


def test_statistics_with_duplicates_reverse_pairs_aliases_and_no_taxonomy():
    samples = validate_training_records([
        record("one", [["dog", "is-a", "animal"], ["dog", "is-a", "animal"],
                       ["shore", "equivalent class", "shoreline"], ["shoreline", "equivalent class", "shore"]]),
        record("two", [["DOG", "instance-of", "Animal"], ["node", "related", "node"]]),
    ])
    report = analyze_training(samples)
    assert report["counts"]["triples_raw"] == 6
    assert report["counts"]["duplicates_within_documents"] == 1
    assert report["counts"]["self_loop_triples_raw"] == 1
    assert report["relation_groups"]["taxonomy_is_a"]["count"] == 2
    assert report["relation_groups"]["term_typing_instance_of"]["count"] == 1
    assert report["relation_groups"]["other_non_taxonomic_labels"]["count"] == 3
    assert report["bidirectionality"]["same_predicate_reverse_pairs"] == 1
    assert report["structure"]["documents_without_is_a"] == 1
    assert report["alias_candidates"]["lexical_variant_groups"] == 2
    assert report["lengths"]["triples_per_document"]["median"] == 3
    assert report["alias_candidates"]["annotated_examples"][0]["id"] == "one"


def test_reverse_pair_count_does_not_double_count_or_include_self_loops():
    samples = validate_training_records([record(triples=[
        ["a", "r", "b"], ["b", "r", "a"], ["a", "r", "b"], ["a", "r", "a"],
        ["a", "q", "b"], ["b", "s", "a"],
    ])])
    report = analyze_training(samples)
    assert report["bidirectionality"]["same_predicate_reverse_pairs"] == 1
    assert report["bidirectionality"]["different_predicate_reverse_entity_pairs"] == 1
