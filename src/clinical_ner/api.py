"""FastAPI service: POST clinical text, get structured drug/chemical and disease mentions back.

Run locally:
    uvicorn clinical_ner.api:app --app-dir src
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from clinical_ner import __version__
from clinical_ner.config import Settings
from clinical_ner.postprocess import Entity
from clinical_ner.schemas import EntityOut, ExtractRequest, ExtractResponse, Health, ModelInfo, Ready

logger = logging.getLogger("clinical_ner")


class Predictor(Protocol):
    name: str
    version: str

    def predict(self, text: str) -> list[Entity]: ...


def create_app(settings: Settings | None = None, model: Predictor | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model, app.state.load_error = model, None
        if model is None:
            try:
                from clinical_ner.model import NerModel

                start = time.perf_counter()
                app.state.model = NerModel(settings.model_dir, settings.model_file, settings.ort_threads)
                logger.info("Loaded %s in %.1fs", settings.model_file, time.perf_counter() - start)
            except Exception as exc:  # keep the process up so /ready can report the problem
                logger.exception("Model failed to load")
                app.state.load_error = str(exc)
        yield

    app = FastAPI(
        title="ClinicalNER",
        version=__version__,
        description="Extracts chemical/drug and disease mentions from clinical text, on CPU.",
        lifespan=lifespan,
    )

    def get_model(request: Request) -> Predictor:
        if request.app.state.model is None:
            raise HTTPException(503, detail="Model is not loaded")
        return request.app.state.model

    @app.get("/health", response_model=Health, tags=["ops"])
    def health() -> Health:
        """Liveness: the process is up."""
        return Health(status="ok")

    @app.get("/ready", response_model=Ready, tags=["ops"])
    def ready(request: Request):
        """Readiness: the model is loaded and requests can be served."""
        loaded = request.app.state.model
        if loaded is None:
            body = Ready(ready=False, detail=request.app.state.load_error or "Model is not loaded")
            return JSONResponse(status_code=503, content=body.model_dump())
        return Ready(ready=True, model=ModelInfo(name=loaded.name, version=loaded.version))

    @app.post("/v1/extract", response_model=ExtractResponse, tags=["ner"])
    def extract(body: ExtractRequest, request: Request) -> ExtractResponse:
        """Returns every chemical/drug and disease mention with character offsets and confidence."""
        if len(body.text) > settings.max_text_chars:
            raise HTTPException(413, detail=f"Text is longer than {settings.max_text_chars} characters")
        predictor = get_model(request)
        min_score = settings.min_score if body.min_score is None else body.min_score

        start = time.perf_counter()
        entities = predictor.predict(body.text)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return ExtractResponse(
            entities=[
                EntityOut(
                    text=e.text,
                    label=e.label,
                    start=e.start,
                    end=e.end,
                    score=round(e.score, 4),
                    needs_review=e.score < settings.review_threshold,
                )
                for e in entities
                if e.score >= min_score
            ],
            model=ModelInfo(name=predictor.name, version=predictor.version),
            processing_ms=round(elapsed_ms, 2),
        )

    return app


app = create_app()
