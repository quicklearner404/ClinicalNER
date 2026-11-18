"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_dir: str = "models/clinical-ner-onnx"
    model_file: str = "model.int8.onnx"
    # Entities scoring below min_score are dropped; below review_threshold they are flagged.
    min_score: float = 0.5
    review_threshold: float = 0.85
    max_text_chars: int = 20_000
    ort_threads: int = 0  # 0 lets ONNX Runtime pick

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_dir=os.getenv("MODEL_DIR", cls.model_dir),
            model_file=os.getenv("MODEL_FILE", cls.model_file),
            min_score=float(os.getenv("MIN_SCORE", cls.min_score)),
            review_threshold=float(os.getenv("REVIEW_THRESHOLD", cls.review_threshold)),
            max_text_chars=int(os.getenv("MAX_TEXT_CHARS", cls.max_text_chars)),
            ort_threads=int(os.getenv("ORT_THREADS", cls.ort_threads)),
        )
