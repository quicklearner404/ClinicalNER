# ClinicalNER API: ONNX Runtime on CPU, no PyTorch in the image.
# Build after the model is in models/clinical-ner-onnx (python scripts/download_model.py).
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/models/clinical-ner-onnx \
    MODEL_FILE=model.int8.onnx
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ src/
COPY models/clinical-ner-onnx/ models/clinical-ner-onnx/
RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://localhost:8000/ready').status != 200)"
CMD ["uvicorn", "clinical_ner.api:app", "--host", "0.0.0.0", "--port", "8000"]
