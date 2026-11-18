"""Compare PyTorch, ONNX fp32 and ONNX INT8: accuracy, latency and size, on CPU.

Usage:
    python training/benchmark.py --torch-dir models/clinical-ner-bc5cdr --onnx-dir models/clinical-ner-onnx

Accuracy is entity-level F1 on the full BC5CDR test split. Latency is measured
on single sentences and on "notes" of 10 consecutive test sentences (about 200
words), one request at a time, after a warm-up.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import f1_score
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import load_labels, load_split  # noqa: E402

from clinical_ner.model import NerModel  # noqa: E402


class TorchRunner:
    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
        self.id2label = self.model.config.id2label

    @torch.inference_mode()
    def predict_words(self, words: list[str]) -> list[str]:
        enc = self.tokenizer(words, is_split_into_words=True, return_tensors="pt")
        pred = self.model(**enc).logits[0].argmax(-1).tolist()
        labels, seen = ["O"] * len(words), set()
        for position, word_id in enumerate(enc.word_ids()):
            if word_id is not None and word_id not in seen:
                seen.add(word_id)
                labels[word_id] = self.id2label[pred[position]]
        return labels


class OnnxRunner:
    def __init__(self, model_dir: str, model_file: str):
        self.model = NerModel(model_dir, model_file)

    def predict_words(self, words: list[str]) -> list[str]:
        return self.model._predict_words(words)[0]


def latency_ms(fn, inputs: list[list[str]], warmup: int = 5) -> dict[str, float]:
    for words in inputs[:warmup]:
        fn(words)
    times = []
    for words in inputs:
        start = time.perf_counter()
        fn(words)
        times.append((time.perf_counter() - start) * 1000)
    return {"p50": round(float(np.percentile(times, 50)), 2), "p95": round(float(np.percentile(times, 95)), 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--torch-dir", default="models/clinical-ner-bc5cdr")
    parser.add_argument("--onnx-dir", default="models/clinical-ner-onnx")
    parser.add_argument("--report", default="reports/benchmark.json")
    parser.add_argument("--latency-samples", type=int, default=200)
    args = parser.parse_args()

    id2label = load_labels()
    test = load_split("test")
    gold = [[id2label[t] for t in row["tags"]] for row in test]
    sentences = [row["tokens"] for row in test]
    notes = [sum(sentences[i : i + 10], []) for i in range(0, len(sentences) - 10, 10)]

    weights = Path(args.torch_dir) / "model.safetensors"
    runners = {
        "PyTorch fp32": (TorchRunner(args.torch_dir), weights),
        "ONNX Runtime fp32": (OnnxRunner(args.onnx_dir, "model.onnx"), Path(args.onnx_dir) / "model.onnx"),
        "ONNX Runtime INT8": (
            OnnxRunner(args.onnx_dir, "model.int8.onnx"),
            Path(args.onnx_dir) / "model.int8.onnx",
        ),
    }

    results = {}
    for name, (runner, path) in runners.items():
        pred = [runner.predict_words(words) for words in sentences]
        results[name] = {
            "test_f1": round(f1_score(gold, pred), 4),
            "sentence_ms": latency_ms(runner.predict_words, sentences[: args.latency_samples]),
            "note_ms": latency_ms(runner.predict_words, notes[: args.latency_samples]),
            "size_mb": round(path.stat().st_size / 1e6, 1),
        }
        print(name, results[name], flush=True)

    report = {
        "machine": {
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
        },
        "test_sentences": len(sentences),
        "words_per_note": round(float(np.mean([len(n) for n in notes])), 1),
        "results": results,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
