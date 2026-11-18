"""Request and response models for the REST API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, description="Free-text clinical or biomedical note.")
    min_score: float | None = Field(
        default=None, ge=0, le=1, description="Drop entities below this confidence. Defaults to the server setting."
    )

    model_config = {
        "json_schema_extra": {"examples": [{"text": "Patient developed acute renal failure after starting cisplatin."}]}
    }


class EntityOut(BaseModel):
    text: str
    label: Literal["Chemical", "Disease"]
    start: int = Field(description="Character offset where the entity starts in the input text.")
    end: int = Field(description="Character offset just after the entity ends.")
    score: float = Field(description="Mean model confidence over the entity's words.")
    needs_review: bool = Field(description="True when the score is below the review threshold.")


class ModelInfo(BaseModel):
    name: str
    version: str


class ExtractResponse(BaseModel):
    entities: list[EntityOut]
    model: ModelInfo
    processing_ms: float


class Health(BaseModel):
    status: Literal["ok"]


class Ready(BaseModel):
    ready: bool
    model: ModelInfo | None = None
    detail: str | None = None
