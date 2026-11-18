import pytest
from fastapi.testclient import TestClient

from clinical_ner.api import create_app
from clinical_ner.config import Settings
from clinical_ner.postprocess import Entity


class StubModel:
    name = "stub.onnx"
    version = "test"

    def predict(self, text: str) -> list[Entity]:
        return [
            Entity("cisplatin", "Chemical", 0, 9, 0.99),
            Entity("renal failure", "Disease", 20, 33, 0.7),
            Entity("noise", "Disease", 40, 45, 0.2),
        ]


@pytest.fixture
def client():
    settings = Settings(min_score=0.5, review_threshold=0.85, max_text_chars=100)
    with TestClient(create_app(settings, model=StubModel())) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready").json()
    assert ready["ready"] is True and ready["model"] == {"name": "stub.onnx", "version": "test"}


def test_extract_filters_low_scores_and_flags_review(client):
    body = client.post("/v1/extract", json={"text": "cisplatin caused renal failure"}).json()
    assert [(e["text"], e["needs_review"]) for e in body["entities"]] == [("cisplatin", False), ("renal failure", True)]
    assert body["model"]["version"] == "test"
    assert body["processing_ms"] >= 0


def test_extract_min_score_override(client):
    body = client.post("/v1/extract", json={"text": "x", "min_score": 0.0}).json()
    assert len(body["entities"]) == 3


@pytest.mark.parametrize(
    "payload, status",
    [({"text": ""}, 422), ({}, 422), ({"text": "x", "min_score": 2}, 422), ({"text": "x" * 101}, 413)],
)
def test_extract_rejects_bad_input(client, payload, status):
    assert client.post("/v1/extract", json=payload).status_code == status


def test_not_ready_when_model_fails_to_load(tmp_path):
    settings = Settings(model_dir=str(tmp_path))
    with TestClient(create_app(settings)) as c:
        ready = c.get("/ready")
        assert ready.status_code == 503 and ready.json()["ready"] is False
        assert c.post("/v1/extract", json={"text": "aspirin"}).status_code == 503
        assert c.get("/health").status_code == 200
