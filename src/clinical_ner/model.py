"""ONNX Runtime inference for the fine-tuned NER model.

Only onnxruntime, tokenizers and numpy are needed at serving time (no PyTorch),
which keeps the Docker image small and inference fast on plain CPUs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from clinical_ner.postprocess import Entity, Word, decode_bio, split_words

# RoBERTa allows 512 positions; 2 go to the <s> and </s> special tokens.
MAX_TOKENS = 510


class NerModel:
    def __init__(self, model_dir: str | Path, model_file: str = "model.int8.onnx", threads: int = 0):
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text())
        self.id2label = {int(i): label for i, label in config["id2label"].items()}
        self.version = config.get("model_version", "unknown")
        self.name = model_file

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_dir / model_file), options, providers=["CPUExecutionProvider"])

    def predict(self, text: str) -> list[Entity]:
        words = split_words(text)
        labels: list[str] = []
        scores: list[float] = []
        for chunk in self._chunks(words):
            chunk_labels, chunk_scores = self._predict_words([w.text for w in chunk])
            labels += chunk_labels
            scores += chunk_scores
        return decode_bio(text, words, labels, scores)

    def _chunks(self, words: list[Word]) -> list[list[Word]]:
        """Splits long notes into pieces that fit the model's 512-token window."""
        if not words:
            return []
        chunks: list[list[Word]] = [[]]
        used = 0
        encodings = self.tokenizer.encode_batch([[w.text] for w in words], is_pretokenized=True)
        for word, enc in zip(words, encodings, strict=True):
            size = len(enc.ids) - 2  # without <s> and </s>
            if chunks[-1] and used + size > MAX_TOKENS:
                chunks.append([])
                used = 0
            chunks[-1].append(word)
            used += size
        return chunks

    def _predict_words(self, words: list[str]) -> tuple[list[str], list[float]]:
        enc = self.tokenizer.encode(words, is_pretokenized=True)
        ids = np.array([enc.ids], dtype=np.int64)
        logits = self.session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)

        labels = ["O"] * len(words)
        scores = [0.0] * len(words)
        seen: set[int] = set()
        # The model was trained on the first sub-token of each word, so read it there.
        for position, word_id in enumerate(enc.word_ids):
            if word_id is None or word_id in seen:
                continue
            seen.add(word_id)
            best = int(probs[position].argmax())
            labels[word_id] = self.id2label[best]
            scores[word_id] = float(probs[position, best])
        return labels, scores
