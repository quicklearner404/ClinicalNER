"""Fine-tune DistilRoBERTa for chemical/drug and disease NER on BC5CDR.

Usage:
    python training/train.py --output-dir models/clinical-ner-bc5cdr

Data: BioCreative V CDR (tner/bc5cdr on the Hugging Face Hub), pre-split into
sentences with word-level BIO tags. Metrics are entity-level (seqeval), so a
prediction only counts if both the span and the type are exactly right.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

DATASET_REPO = "tner/bc5cdr"
SPLITS = {"train": "train.json", "validation": "valid.json", "test": "test.json"}


def load_split(name: str, limit: int | None = None) -> list[dict]:
    path = hf_hub_download(DATASET_REPO, f"dataset/{SPLITS[name]}", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def load_labels() -> dict[int, str]:
    path = hf_hub_download(DATASET_REPO, "dataset/label.json", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        label2id = json.load(f)
    return {i: label for label, i in label2id.items()}


class TokenizedNER(Dataset):
    """Tokenizes pre-split words and labels only the first sub-token of each word.

    Remaining sub-tokens and special tokens get -100 so the loss ignores them;
    inference reads the prediction from the same first sub-token.
    """

    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.items = []
        for row in rows:
            enc = tokenizer(row["tokens"], is_split_into_words=True, truncation=True, max_length=max_length)
            labels, previous = [], None
            for word_id in enc.word_ids():
                if word_id is None or word_id == previous:
                    labels.append(-100)
                else:
                    labels.append(row["tags"][word_id])
                previous = word_id
            enc["labels"] = labels
            self.items.append(dict(enc))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def build_metrics(id2label: dict[int, str]):
    def to_tags(predictions: np.ndarray, label_ids: np.ndarray):
        true, pred = [], []
        for p_row, l_row in zip(predictions, label_ids, strict=True):
            keep = l_row != -100
            true.append([id2label[int(i)] for i in l_row[keep]])
            pred.append([id2label[int(i)] for i in p_row[keep]])
        return true, pred

    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        true, pred = to_tags(np.argmax(logits, axis=-1), label_ids)
        report = classification_report(true, pred, output_dict=True, zero_division=0)
        metrics = {
            "precision": precision_score(true, pred),
            "recall": recall_score(true, pred),
            "f1": f1_score(true, pred),
        }
        for entity in ("Chemical", "Disease"):
            metrics[f"{entity.lower()}_f1"] = report[entity]["f1-score"]
        return metrics

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="distilroberta-base")
    parser.add_argument("--output-dir", default="models/clinical-ner-bc5cdr")
    parser.add_argument("--report", default="reports/metrics.json")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Use only N rows per split (smoke test).")
    args = parser.parse_args()

    set_seed(args.seed)
    id2label = load_labels()
    label2id = {v: k for k, v in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, add_prefix_space=True)
    data = {split: TokenizedNER(load_split(split, args.limit), tokenizer, args.max_length) for split in SPLITS}
    print({split: len(ds) for split, ds in data.items()})

    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir) / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=0.1,  # a float below 1 is a fraction of total steps
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_metrics(id2label),
    )

    start = time.time()
    trainer.train()
    train_minutes = (time.time() - start) / 60

    validation = trainer.evaluate(data["validation"], metric_key_prefix="validation")
    test = trainer.evaluate(data["test"], metric_key_prefix="test")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    report = {
        "dataset": DATASET_REPO,
        "base_model": args.base_model,
        "sizes": {split: len(ds) for split, ds in data.items()},
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "seed": args.seed,
        },
        "train_minutes": round(train_minutes, 1),
        "validation": {k.removeprefix("validation_"): v for k, v in validation.items()},
        "test": {k.removeprefix("test_"): v for k, v in test.items()},
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
