"""End-to-end checks against the real exported ONNX model.

Skipped unless the model is present (python scripts/download_model.py).
CI downloads it from the Hugging Face Hub and runs these on every push.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/clinical-ner-onnx"))
MODEL_FILE = os.getenv("MODEL_FILE", "model.int8.onnx")

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / MODEL_FILE).exists(),
    reason=f"model not found at {MODEL_DIR / MODEL_FILE}; run scripts/download_model.py",
)


@pytest.fixture(scope="module")
def model():
    from clinical_ner.model import NerModel

    return NerModel(MODEL_DIR, MODEL_FILE)


def _spans(entities):
    return [(e.text, e.label, e.start, e.end) for e in entities]


def test_readme_example_is_reproduced(model):
    text = "Patient developed acute renal failure after starting cisplatin."

    assert _spans(model.predict(text)) == [
        ("acute renal failure", "Disease", 18, 37),
        ("cisplatin", "Chemical", 53, 62),
    ]


def test_scores_are_probabilities(model):
    entities = model.predict("Patient developed acute renal failure after starting cisplatin.")

    assert entities and all(0.0 <= e.score <= 1.0 for e in entities)


def test_text_without_entities(model):
    assert model.predict("The patient was discharged home on Tuesday.") == []


def test_long_note_is_chunked_and_offsets_stay_exact(model):
    sentence = "Patient developed acute renal failure after starting cisplatin. "
    text = sentence * 120  # well beyond the 512-token window

    entities = model.predict(text)

    assert all(text[e.start : e.end] == e.text for e in entities)
    found = {(e.text, e.label) for e in entities}
    assert ("cisplatin", "Chemical") in found
    assert ("acute renal failure", "Disease") in found
    # Mentions late in the note are still detected after chunking.
    assert max(e.start for e in entities) > len(text) - len(sentence)


def test_api_end_to_end(model):
    from fastapi.testclient import TestClient

    from clinical_ner.api import create_app
    from clinical_ner.config import Settings

    with TestClient(create_app(Settings(), model=model)) as client:
        body = client.post("/v1/extract", json={"text": "Cisplatin can cause nephrotoxicity."}).json()

    assert {e["label"] for e in body["entities"]} <= {"Chemical", "Disease"}
    assert any(e["label"] == "Chemical" and e["text"].lower() == "cisplatin" for e in body["entities"])
