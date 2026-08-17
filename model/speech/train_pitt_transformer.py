from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PITT_DIR = REPO_ROOT / "dataset" / "speech" / "Pitt"
OUTPUT_DIR = REPO_ROOT / "output_performance" / "speech" / "pitt_transformer"

LABELS = {"Control": 0, "Dementia": 1}
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass(frozen=True)
class TrainConfig:
    pitt_dir: str
    output_dir: str
    task: str = "cookie"
    max_vocab: int = 12000
    max_len: int = 256
    min_freq: int = 2
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.2
    batch_size: int = 16
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    test_size: float = 0.2
    val_size: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class PittRecord:
    subject_id: str
    path: str
    task: str
    label: int
    label_name: str
    text: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_chat_utterance(text: str) -> str:
    text = re.sub(r"\x15\d+_\d+\x15", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"&[+=-]?\w+", " ", text)
    text = text.replace("(", " ").replace(")", " ")
    text = re.sub(r"[/@:+!?.,;\"']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text.lower())


def read_participant_text(path: Path) -> str:
    utterances: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if raw_line.startswith("*PAR:"):
            utterance = raw_line.split(":", 1)[1]
            cleaned = normalize_chat_utterance(utterance)
            if cleaned:
                utterances.append(cleaned)
    return " ".join(utterances)


def load_pitt_records(pitt_dir: Path, task: str) -> list[PittRecord]:
    records: list[PittRecord] = []
    for label_name, label in LABELS.items():
        task_dir = pitt_dir / label_name / task
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Missing Pitt task directory: {task_dir}")

        for path in sorted(task_dir.glob("*.cha")):
            text = read_participant_text(path)
            if not text:
                continue
            records.append(
                PittRecord(
                    subject_id=path.stem,
                    path=str(path.relative_to(REPO_ROOT)),
                    task=task,
                    label=label,
                    label_name=label_name,
                    text=text,
                )
            )

    if not records:
        raise ValueError(f"No usable Pitt records found under {pitt_dir} for task={task!r}")
    return records


def build_vocab(texts: list[str], max_vocab: int, min_freq: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text in texts:
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    sorted_tokens = sorted(
        ((token, count) for token, count in counts.items() if count >= min_freq),
        key=lambda item: (-item[1], item[0]),
    )
    for token, _ in sorted_tokens[: max_vocab - len(vocab)]:
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> tuple[list[int], list[int]]:
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize(text)[:max_len]]
    attention = [1] * len(ids)
    pad_count = max_len - len(ids)
    if pad_count > 0:
        ids.extend([vocab[PAD_TOKEN]] * pad_count)
        attention.extend([0] * pad_count)
    return ids, attention


class PittTextDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, records: list[PittRecord], vocab: dict[str, int], max_len: int) -> None:
        self.records = records
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        input_ids, attention_mask = encode_text(record.text, self.vocab, self.max_len)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.bool),
            torch.tensor(record.label, dtype=torch.long),
        )


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SpeechTransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, config: TrainConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embed_dim, padding_idx=0)
        self.position = PositionalEncoding(config.embed_dim, config.max_len, config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, 2),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.position(x)
        padding_mask = ~attention_mask
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(self.norm(pooled))


def split_records(records: list[PittRecord], config: TrainConfig) -> tuple[list[PittRecord], list[PittRecord], list[PittRecord]]:
    labels = np.array([record.label for record in records])
    groups = np.array([record.subject_id.split("-", 1)[0] for record in records])

    train_val_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.seed).split(
            records, labels, groups
        )
    )
    train_val = [records[index] for index in train_val_idx]
    test = [records[index] for index in test_idx]

    train_val_labels = np.array([record.label for record in train_val])
    train_val_groups = np.array([record.subject_id.split("-", 1)[0] for record in train_val])
    train_idx, val_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=config.val_size, random_state=config.seed + 1).split(
            train_val, train_val_labels, train_val_groups
        )
    )
    train = [train_val[index] for index in train_idx]
    val = [train_val[index] for index in val_idx]
    return train, val, test


def class_weights(records: list[PittRecord], device: torch.device) -> torch.Tensor:
    counts = np.bincount([record.label for record in records], minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            total_loss += float(loss.item()) * labels.size(0)
            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())
            all_probs.extend(probs.detach().cpu().tolist())

    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    probs_np = np.array(all_probs)
    mean_loss = total_loss / max(len(labels_np), 1)
    return mean_loss, labels_np, preds_np, probs_np


def evaluate_split(
    model: nn.Module,
    records: list[PittRecord],
    vocab: dict[str, int],
    config: TrainConfig,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    loader = DataLoader(PittTextDataset(records, vocab, config.max_len), batch_size=config.batch_size)
    criterion = nn.CrossEntropyLoss()
    loss, y_true, y_pred, y_prob = run_epoch(model, loader, criterion, device)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true.tolist())) > 1 else None
    metrics = {
        "loss": loss,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": None if auc is None else float(auc),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Control", "Dementia"],
            output_dict=True,
            zero_division=0,
        ),
    }
    predictions = pd.DataFrame(
        {
            "subject_id": [record.subject_id for record in records],
            "path": [record.path for record in records],
            "label": y_true,
            "label_name": [record.label_name for record in records],
            "dementia_probability": y_prob,
            "prediction": y_pred,
            "prediction_name": ["Dementia" if pred == 1 else "Control" for pred in y_pred],
        }
    )
    return metrics, predictions


def train(config: TrainConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_pitt_records(Path(config.pitt_dir), config.task)
    train_records, val_records, test_records = split_records(records, config)
    vocab = build_vocab([record.text for record in train_records], config.max_vocab, config.min_freq)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeechTransformerClassifier(len(vocab), config).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_records, device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_loader = DataLoader(
        PittTextDataset(train_records, vocab, config.max_len),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(PittTextDataset(val_records, vocab, config.max_len), batch_size=config.batch_size)

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_loss, train_y, train_pred, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_y, val_pred, val_prob = run_epoch(model, val_loader, nn.CrossEntropyLoss(), device)
        val_auc = roc_auc_score(val_y, val_prob) if len(set(val_y.tolist())) > 1 else None
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": float(accuracy_score(train_y, train_pred)),
            "val_loss": val_loss,
            "val_accuracy": float(accuracy_score(val_y, val_pred)),
            "val_roc_auc": None if val_auc is None else float(val_auc),
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"train_acc={row['train_accuracy']:.3f} val_loss={val_loss:.4f} "
            f"val_acc={row['val_accuracy']:.3f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    split_metrics: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []
    for split_name, split_records_for_eval in (
        ("train", train_records),
        ("validation", val_records),
        ("test", test_records),
    ):
        metrics, predictions = evaluate_split(model, split_records_for_eval, vocab, config, device)
        split_metrics[split_name] = metrics
        predictions.insert(0, "split", split_name)
        prediction_frames.append(predictions)

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    metadata = {
        "model_version": "pitt_text_transformer_v1",
        "label_mapping": LABELS,
        "config": asdict(config),
        "device": str(device),
        "n_records": len(records),
        "n_train": len(train_records),
        "n_validation": len(val_records),
        "n_test": len(test_records),
        "vocab_size": len(vocab),
        "metrics": split_metrics,
        "history": history,
    }

    predictions_path = output_dir / "pitt_transformer_predictions.csv"
    metrics_path = output_dir / "pitt_transformer_metrics.json"
    vocab_path = output_dir / "pitt_transformer_vocab.json"
    model_path = output_dir / "pitt_transformer.pt"

    all_predictions.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    vocab_path.write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "vocab": vocab,
            "label_mapping": LABELS,
        },
        model_path,
    )

    return {
        "metadata": metadata,
        "predictions_path": predictions_path,
        "metrics_path": metrics_path,
        "vocab_path": vocab_path,
        "model_path": model_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a text transformer on Pitt CHAT transcripts.")
    parser.add_argument("--pitt-dir", type=Path, default=PITT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--task", default="cookie", choices=["cookie", "fluency", "recall", "sentence"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--max-vocab", type=int, default=12000)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        pitt_dir=str(args.pitt_dir),
        output_dir=str(args.output_dir),
        task=args.task,
        max_vocab=args.max_vocab,
        max_len=args.max_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    result = train(config)
    test_metrics = result["metadata"]["metrics"]["test"]
    print("Pitt speech transformer complete")
    print(f"Test accuracy: {test_metrics['accuracy']:.3f}")
    print(f"Test ROC AUC: {test_metrics['roc_auc']}")
    print(f"Predictions CSV: {result['predictions_path']}")
    print(f"Metrics JSON: {result['metrics_path']}")
    print(f"Model checkpoint: {result['model_path']}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
