"""Download the exported ONNX model from the Hugging Face Hub into models/clinical-ner-onnx.

Usage:
    python scripts/download_model.py [--repo <user>/clinical-ner-bc5cdr-onnx]
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download

DEFAULT_REPO = os.getenv("HF_MODEL_REPO", "quicklearner404/clinical-ner-bc5cdr-onnx")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--output-dir", default="models/clinical-ner-onnx")
    parser.add_argument("--include-fp32", action="store_true", help="Also download the larger fp32 model.onnx")
    args = parser.parse_args()

    patterns = ["model.int8.onnx", "tokenizer.json", "config.json", "metrics.json"]
    if args.include_fp32:
        patterns.append("model.onnx")
    path = snapshot_download(args.repo, local_dir=args.output_dir, allow_patterns=patterns)
    print(f"Model files saved to {path}")


if __name__ == "__main__":
    main()
