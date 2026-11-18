"""Publish models/clinical-ner-onnx to the Hugging Face Hub with a model card built from the measured reports.

Usage (needs a Hugging Face write token, e.g. `hf auth login`):
    python scripts/publish_model.py --repo <user>/clinical-ner-bc5cdr-onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

CARD = """---
license: other
language: en
library_name: onnx
pipeline_tag: token-classification
datasets:
  - tner/bc5cdr
base_model: distilroberta-base
tags:
  - ner
  - biomedical
  - clinical
  - onnx
  - int8
---

# ClinicalNER: chemical/drug and disease NER (ONNX, INT8)

DistilRoBERTa fine-tuned on BC5CDR for `Chemical` and `Disease` mentions, exported to ONNX and dynamically
quantised to INT8 for CPU serving. Built for the [ClinicalNER API]({github}).

## Results (BC5CDR test set, entity-level, seqeval)

{results}

## Files

| File | Use |
|---|---|
| `model.int8.onnx` | Default serving model |
| `model.onnx` | fp32 export |
| `tokenizer.json` | RoBERTa BPE tokenizer for the `tokenizers` library (`add_prefix_space`, pre-split words) |
| `config.json` | `id2label` and model version |
| `metrics.json` | Full training and evaluation report |

## Usage

Inputs are pre-split words (see `split_words` in the GitHub repo); read each word's label from its first
sub-token. The easiest way to use it is the API: `docker compose up` in the GitHub repo.

## Limitations

Trained on PubMed abstracts, not hospital notes; expect lower accuracy on real clinical text. Research prototype,
not for diagnosis or treatment decisions. Check the BioCreative V CDR terms before commercial use.
"""


def results_table(onnx_dir: Path, benchmark: Path) -> str:
    metrics = json.loads((onnx_dir / "metrics.json").read_text())["test"]
    rows = [
        "| Metric | Value |",
        "|---|---|",
        f"| Precision | {metrics['precision']:.3f} |",
        f"| Recall | {metrics['recall']:.3f} |",
        f"| F1 (overall) | {metrics['f1']:.3f} |",
        f"| F1 Chemical | {metrics['chemical_f1']:.3f} |",
        f"| F1 Disease | {metrics['disease_f1']:.3f} |",
    ]
    if benchmark.exists():
        int8 = json.loads(benchmark.read_text())["results"]["ONNX Runtime INT8"]
        rows.append(f"| F1 (INT8 ONNX model) | {int8['test_f1']:.3f} |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="quicklearner404/clinical-ner-bc5cdr-onnx")
    parser.add_argument("--onnx-dir", default="models/clinical-ner-onnx")
    parser.add_argument("--benchmark", default="reports/benchmark.json")
    parser.add_argument("--github", default="https://github.com/quicklearner404/ClinicalNER")
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    card = CARD.format(github=args.github, results=results_table(onnx_dir, Path(args.benchmark)))
    (onnx_dir / "README.md").write_text(card)

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        folder_path=str(onnx_dir),
        allow_patterns=["*.onnx", "tokenizer.json", "config.json", "metrics.json", "README.md"],
        commit_message="Upload ClinicalNER ONNX model",
    )
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
