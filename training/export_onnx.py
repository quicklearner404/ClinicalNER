"""Export the fine-tuned model to ONNX and make a dynamically quantized INT8 copy.

Usage:
    python training/export_onnx.py --model-dir models/clinical-ner-bc5cdr --output-dir models/clinical-ner-onnx

The output folder is everything the API needs: model.onnx, model.int8.onnx,
tokenizer.json and config.json (labels plus a model version string).
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForTokenClassification, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default="models/clinical-ner-bc5cdr")
    parser.add_argument("--output-dir", default="models/clinical-ner-onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).eval()

    sample = tokenizer(["Cisplatin", "nephrotoxicity"], is_split_into_words=True, return_tensors="pt")
    fp32_path = out / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            (sample["input_ids"], sample["attention_mask"]),
            str(fp32_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=args.opset,
            dynamo=False,
        )

    # Weights go to int8; activations are quantised on the fly. Good speed-up on CPU, no calibration data.
    quantize_dynamic(str(fp32_path), str(out / "model.int8.onnx"), weight_type=QuantType.QInt8)

    tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))
    config = {
        "id2label": {str(k): v for k, v in model.config.id2label.items()},
        "model_version": f"bc5cdr-distilroberta-{date.today():%Y%m%d}",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    metrics = Path("reports/metrics.json")
    if metrics.exists():
        shutil.copy(metrics, out / "metrics.json")

    for f in sorted(out.iterdir()):
        print(f"{f.name:<20} {f.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
