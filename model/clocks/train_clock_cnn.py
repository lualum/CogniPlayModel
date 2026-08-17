from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RAW_CLOCK_DATA_DIR = REPO_ROOT / "dataset" / "clocks" / "ClockData"
RESIZED_CLOCK_DATA_DIR = REPO_ROOT / "dataset" / "clocks" / "ClockData_256"
CLOCK_DATA_DIR = RESIZED_CLOCK_DATA_DIR if RESIZED_CLOCK_DATA_DIR.is_dir() else RAW_CLOCK_DATA_DIR
NHATS_ROUND_14B_SAS = RAW_CLOCK_DATA_DIR / "NHATS_Round_14B_SP_File.sas7bdat"
OUTPUT_DIR = REPO_ROOT / "output_performance" / "clocks" / "nhats_cnn"

TARGET_COLUMN = "cg14dclkdlnn"
LABELS = {str(score): score for score in range(6)}
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class TrainConfig:
    clock_data_dir: str
    nhats_round_14b_sas: str
    output_dir: str
    target_column: str = TARGET_COLUMN
    image_size: int = 256
    batch_size: int = 64
    epochs: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.25
    num_workers: int = 0
    impairment_threshold: int = 4
    seed: int = 42


@dataclass(frozen=True)
class ClockRecord:
    subject_id: str
    path: str
    split: str
    score: int
    label: int
    label_name: str
    binary_label: int
    binary_label_name: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subject_id_from_path(path: Path) -> str:
    match = re.match(r"(\d+)", path.name)
    if not match:
        raise ValueError(f"Clock image filename does not start with a subject id: {path}")
    return match.group(1).zfill(8)


def read_ground_signal(sas_path: Path, target_column: str) -> pd.DataFrame:
    if not sas_path.is_file():
        raise FileNotFoundError(f"Missing NHATS Round 14B SAS file: {sas_path}")

    data = pd.read_sas(sas_path, format="sas7bdat", encoding="latin1")
    missing_columns = {"spid", target_column} - set(data.columns)
    if missing_columns:
        raise ValueError(f"NHATS Round 14B file is missing columns: {sorted(missing_columns)}")

    labels = data[["spid", target_column]].dropna().copy()
    labels["subject_id"] = labels["spid"].astype(int).astype(str).str.zfill(8)
    labels["score"] = labels[target_column].astype(int)
    labels = labels[labels["score"].between(0, 5)]
    labels = labels[["subject_id", "score"]].drop_duplicates("subject_id")
    if labels.empty:
        raise ValueError(f"No usable 0-5 labels found in NHATS column {target_column!r}")
    return labels


def load_clock_records(clock_data_dir: Path, sas_path: Path, config: TrainConfig) -> list[ClockRecord]:
    labels = read_ground_signal(sas_path, config.target_column).set_index("subject_id")["score"].to_dict()
    records: list[ClockRecord] = []

    for split in ("train", "valid", "test"):
        split_dir = clock_data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing clock image split directory: {split_dir}")

        image_paths = sorted(path for path in split_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        for path in image_paths:
            subject_id = subject_id_from_path(path)
            score = labels.get(subject_id)
            if score is None:
                continue
            binary_label = int(score < config.impairment_threshold)
            records.append(
                ClockRecord(
                    subject_id=subject_id,
                    path=str(path.relative_to(REPO_ROOT)),
                    split="validation" if split == "valid" else split,
                    score=int(score),
                    label=int(score),
                    label_name=str(int(score)),
                    binary_label=binary_label,
                    binary_label_name="Impaired" if binary_label == 1 else "Normal",
                )
            )

    if not records:
        raise ValueError("No clock images could be matched to NHATS Round 14B ground-signal rows.")
    return records


class ClockImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, records: list[ClockRecord], image_size: int) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(REPO_ROOT / record.path) as raw_image:
            image = raw_image.convert("L")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        return tensor, torch.tensor(record.label, dtype=torch.long)


class ClockCNN(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 6),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def class_weights(records: list[ClockRecord], device: torch.device) -> torch.Tensor:
    counts = np.bincount([record.label for record in records], minlength=6).astype(np.float32)
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
    all_probs: list[list[float]] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            probs = torch.softmax(logits, dim=1)
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


def binary_probabilities(score_probabilities: np.ndarray, impairment_threshold: int) -> np.ndarray:
    return score_probabilities[:, :impairment_threshold].sum(axis=1)


def evaluate_split(
    model: nn.Module,
    records: list[ClockRecord],
    config: TrainConfig,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    loader = DataLoader(
        ClockImageDataset(records, config.image_size),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    criterion = nn.CrossEntropyLoss()
    loss, y_true, y_pred, y_prob = run_epoch(model, loader, criterion, device)
    y_binary = np.array([record.binary_label for record in records])
    binary_prob = binary_probabilities(y_prob, config.impairment_threshold)
    binary_pred = (binary_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_binary, binary_prob) if len(set(y_binary.tolist())) > 1 else None

    metrics = {
        "loss": loss,
        "score_accuracy": float(accuracy_score(y_true, y_pred)),
        "score_mean_absolute_error": float(mean_absolute_error(y_true, y_pred)),
        "score_confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(6))).tolist(),
        "score_classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(6)),
            target_names=[str(score) for score in range(6)],
            output_dict=True,
            zero_division=0,
        ),
        "binary_accuracy": float(accuracy_score(y_binary, binary_pred)),
        "binary_roc_auc": None if auc is None else float(auc),
        "binary_confusion_matrix": confusion_matrix(y_binary, binary_pred, labels=[0, 1]).tolist(),
        "binary_classification_report": classification_report(
            y_binary,
            binary_pred,
            labels=[0, 1],
            target_names=["Normal", "Impaired"],
            output_dict=True,
            zero_division=0,
        ),
    }
    predictions = pd.DataFrame(
        {
            "subject_id": [record.subject_id for record in records],
            "path": [record.path for record in records],
            "clock_score": y_true,
            "label": y_binary,
            "label_name": [record.binary_label_name for record in records],
            "impairment_probability": binary_prob,
            "prediction_score": y_pred,
            "prediction": binary_pred,
            "prediction_name": ["Impaired" if pred == 1 else "Normal" for pred in binary_pred],
        }
    )
    for score in range(6):
        predictions[f"score_{score}_probability"] = y_prob[:, score]
    return metrics, predictions


def train(config: TrainConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_clock_records(Path(config.clock_data_dir), Path(config.nhats_round_14b_sas), config)
    records_by_split = {
        split_name: [record for record in records if record.split == split_name]
        for split_name in ("train", "validation", "test")
    }
    missing_splits = [split_name for split_name, split_records in records_by_split.items() if not split_records]
    if missing_splits:
        raise ValueError(f"Clock dataset has no labeled records for splits: {missing_splits}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClockCNN(config.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(records_by_split["train"], device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = DataLoader(
        ClockImageDataset(records_by_split["train"], config.image_size),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    validation_loader = DataLoader(
        ClockImageDataset(records_by_split["validation"], config.image_size),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_loss, train_y, train_pred, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_y, val_pred, val_prob = run_epoch(model, validation_loader, nn.CrossEntropyLoss(), device)
        val_binary = np.array([record.binary_label for record in records_by_split["validation"]])
        val_binary_prob = binary_probabilities(val_prob, config.impairment_threshold)
        val_auc = roc_auc_score(val_binary, val_binary_prob) if len(set(val_binary.tolist())) > 1 else None
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_score_accuracy": float(accuracy_score(train_y, train_pred)),
            "val_loss": val_loss,
            "val_score_accuracy": float(accuracy_score(val_y, val_pred)),
            "val_binary_roc_auc": None if val_auc is None else float(val_auc),
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"train_score_acc={row['train_score_accuracy']:.3f} val_loss={val_loss:.4f} "
            f"val_score_acc={row['val_score_accuracy']:.3f}",
            flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    split_metrics: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []
    for split_name in ("train", "validation", "test"):
        metrics, predictions = evaluate_split(model, records_by_split[split_name], config, device)
        split_metrics[split_name] = metrics
        predictions.insert(0, "split", split_name)
        prediction_frames.append(predictions)

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    legacy_predictions = all_predictions[["label", "impairment_probability"]].rename(
        columns={"label": "True", "impairment_probability": "Prob"}
    )
    metadata = {
        "model_version": "nhats_clock_cnn_v1",
        "label_mapping": LABELS,
        "target": f"NHATS Round 14B {config.target_column} score in [0, 5]",
        "binary_target": f"{config.target_column} < {config.impairment_threshold}",
        "config": asdict(config),
        "device": str(device),
        "n_records": len(records),
        "n_train": len(records_by_split["train"]),
        "n_validation": len(records_by_split["validation"]),
        "n_test": len(records_by_split["test"]),
        "metrics": split_metrics,
        "history": history,
    }

    predictions_path = output_dir / "nhats_clock_cnn_predictions.csv"
    legacy_predictions_path = output_dir / "clock_preds.csv"
    metrics_path = output_dir / "nhats_clock_cnn_metrics.json"
    model_path = output_dir / "nhats_clock_cnn.pt"

    all_predictions.to_csv(predictions_path, index=False)
    legacy_predictions.to_csv(legacy_predictions_path, index=False)
    metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "label_mapping": LABELS,
            "target_column": config.target_column,
        },
        model_path,
    )

    return {
        "metadata": metadata,
        "predictions_path": predictions_path,
        "legacy_predictions_path": legacy_predictions_path,
        "metrics_path": metrics_path,
        "model_path": model_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on NHATS Round 14B clock drawings.")
    parser.add_argument("--clock-data-dir", type=Path, default=CLOCK_DATA_DIR)
    parser.add_argument("--nhats-round-14b-sas", type=Path, default=NHATS_ROUND_14B_SAS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--impairment-threshold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        clock_data_dir=str(args.clock_data_dir),
        nhats_round_14b_sas=str(args.nhats_round_14b_sas),
        output_dir=str(args.output_dir),
        target_column=args.target_column,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        num_workers=args.num_workers,
        impairment_threshold=args.impairment_threshold,
        seed=args.seed,
    )
    result = train(config)
    test_metrics = result["metadata"]["metrics"]["test"]
    print("NHATS clock CNN complete")
    print(f"Test score accuracy: {test_metrics['score_accuracy']:.3f}")
    print(f"Test score MAE: {test_metrics['score_mean_absolute_error']:.3f}")
    print(f"Test binary ROC AUC: {test_metrics['binary_roc_auc']}")
    print(f"Predictions CSV: {result['predictions_path']}")
    print(f"Metrics JSON: {result['metrics_path']}")
    print(f"Model checkpoint: {result['model_path']}")
    print(f"Fusion-compatible CSV: {result['legacy_predictions_path']}")


if __name__ == "__main__":
    main()
