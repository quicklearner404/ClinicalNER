import pytest

from clinical_ner.postprocess import Entity, decode_bio, split_words


def test_split_words_keeps_hyphenated_drug_names_and_offsets():
    text = "Started 5-fluorouracil (0.5 mg), then L-dopa."
    words = split_words(text)
    assert [w.text for w in words] == ["Started", "5-fluorouracil", "(", "0.5", "mg", ")", ",", "then", "L-dopa", "."]
    assert all(text[w.start : w.end] == w.text for w in words)


def test_split_words_follows_bc5cdr_hyphen_rules():
    text = "lithium-induced nephropathy, alpha-methyldopa, MK-801 and dose-dependent N-methyl-D-aspartate"
    words = split_words(text)
    assert [w.text for w in words] == [
        "lithium", "-", "induced", "nephropathy", ",", "alpha-methyldopa", ",", "MK-801", "and",
        "dose", "-", "dependent", "N-methyl", "-", "D-aspartate",
    ]  # fmt: skip
    assert all(text[w.start : w.end] == w.text for w in words)


def decode(text, labels, scores=None):
    words = split_words(text)
    return decode_bio(text, words, labels, scores or [0.9] * len(words))


def test_decode_multi_word_entity_uses_original_text_span():
    text = "Developed acute  renal failure today"
    entities = decode(text, ["O", "B-Disease", "I-Disease", "I-Disease", "O"])
    assert entities == [Entity("acute  renal failure", "Disease", 10, 30, pytest.approx(0.9))]


def test_decode_separate_entities_and_types():
    text = "cisplatin nephrotoxicity"
    entities = decode(text, ["B-Chemical", "B-Disease"])
    assert [(e.text, e.label) for e in entities] == [("cisplatin", "Chemical"), ("nephrotoxicity", "Disease")]


def test_decode_type_switch_and_orphan_inside_tag_start_new_entities():
    text = "aspirin ulcer bleeding"
    entities = decode(text, ["I-Chemical", "I-Disease", "I-Disease"])
    assert [(e.text, e.label) for e in entities] == [("aspirin", "Chemical"), ("ulcer bleeding", "Disease")]


def test_decode_score_is_mean_of_words():
    text = "renal failure"
    (entity,) = decode(text, ["B-Disease", "I-Disease"], [0.8, 0.6])
    assert entity.score == pytest.approx(0.7)


def test_decode_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        decode_bio("a b", split_words("a b"), ["O"], [1.0])


def test_decode_empty_text():
    assert decode("", []) == []
