from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HCAP_CSV_ZIP = REPO_ROOT / "dataset" / "games" / "HC22" / "HC22csv.zip"
OUTPUT_DIR = REPO_ROOT / "output_performance" / "games" / "hcap"

LABELS = {"Normal": 0, "Impaired": 1}
TARGET_COLUMN = "R2MMSE_SCORE"

RAW_FEATURE_COLUMNS = [
    "R2HRSNAME_SCORE",
    "R2WORD_TOTAL",
    "R2VERBAL_TOTAL",
    "R2LC_SCORE",
    "R2BC_SCORE",
    "R2WORD_DSCORE",
    "R2BM_IMMSCORE",
    "R2LMB_IMMSCORE",
    "R2WLREC_TOTSCORE",
    "R2CP_SCORE",
    "R2DIG_SCORE",
    "R2CPDEL_SCORE",
    "R2BM_DELSCORE",
    "R2LMB_RECOSCORE",
    "R2NS_SCORE",
    "R2RV_SCORE",
    "R2TMA_SCORE",
    "R2TMA_MIN",
    "R2TMA_SEC",
    "R2TMB_SCORE",
    "R2TMB_MIN",
    "R2TMB_SEC",
]

FEATURE_COLUMNS = [
    "R2HRSNAME_SCORE",
    "R2WORD_TOTAL",
    "R2VERBAL_TOTAL",
    "R2LC_SCORE",
    "R2BC_SCORE",
    "R2WORD_DSCORE",
    "R2BM_IMMSCORE",
    "R2LMB_IMMSCORE",
    "R2WLREC_TOTSCORE",
    "R2CP_SCORE",
    "R2DIG_SCORE",
    "R2CPDEL_SCORE",
    "R2BM_DELSCORE",
    "R2LMB_RECOSCORE",
    "R2NS_SCORE",
    "R2RV_SCORE",
    "R2TMA_SCORE",
    "R2TMA_TIME",
    "R2TMB_SCORE",
    "R2TMB_TIME",
]


@dataclass(frozen=True)
class TrainConfig:
    hcap_csv_zip: str
    output_dir: str
    mmse_threshold: int = 24
    n_estimators: int = 300
    max_depth: int | None = 100
    min_samples_leaf: int = 1
    n_splits: int = 5
    test_size: float = 0.2
    val_size: float = 0.2
    seed: int = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_hcap_respondent_csv(zip_path: Path) -> pd.DataFrame:
    if not zip_path.is_file():
        raise FileNotFoundError(f"Missing HCAP CSV zip: {zip_path}")

    usecols = {"HHID", "PN", TARGET_COLUMN, *RAW_FEATURE_COLUMNS}
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("Hc22hp_r.csv") as file:
            return pd.read_csv(file, usecols=lambda column: column in usecols)


def prepare_hcap_frame(zip_path: Path, mmse_threshold: int) -> pd.DataFrame:
    data = read_hcap_respondent_csv(zip_path)
    missing_features = sorted(set(RAW_FEATURE_COLUMNS) - set(data.columns))
    if missing_features:
        raise ValueError(f"HCAP respondent file is missing feature columns: {missing_features}")

    data["R2TMA_TIME"] = data["R2TMA_MIN"] * 60 + data["R2TMA_SEC"]
    data["R2TMB_TIME"] = data["R2TMB_MIN"] * 60 + data["R2TMB_SEC"]
    data["subject_id"] = data["HHID"].astype(str).str.zfill(6) + "-" + data["PN"].astype(str).str.zfill(3)
    data = data[["subject_id", "HHID", "PN", TARGET_COLUMN, *FEATURE_COLUMNS]].dropna().reset_index(drop=True)
    data["label"] = (data[TARGET_COLUMN] < mmse_threshold).astype(int)
    data["label_name"] = np.where(data["label"].eq(1), "Impaired", "Normal")

    if data["label"].nunique() != 2:
        raise ValueError("HCAP frame must contain both Normal and Impaired labels after filtering.")
    return data


def split_frame(data: pd.DataFrame, config: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        data,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=data["label"],
    )
    train, validation = train_test_split(
        train_val,
        test_size=config.val_size,
        random_state=config.seed + 1,
        stratify=train_val["label"],
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def make_classifier(config: TrainConfig, seed: int | None = None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=config.seed if seed is None else seed,
        n_jobs=-1,
    )


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true.tolist())) > 1 else None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": None if auc is None else float(auc),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Impaired"],
            output_dict=True,
            zero_division=0,
        ),
    }


def predict_frame(model: RandomForestClassifier, frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    x = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_prob = model.predict_proba(x)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return pd.DataFrame(
        {
            "split": split_name,
            "subject_id": frame["subject_id"].to_numpy(),
            "hcap_csv": "dataset/games/HC22/HC22csv.zip::Hc22hp_r.csv",
            "mmse_score": frame[TARGET_COLUMN].to_numpy(),
            "label": frame["label"].to_numpy(),
            "label_name": frame["label_name"].to_numpy(),
            "impairment_probability": y_prob,
            "prediction": y_pred,
            "prediction_name": np.where(y_pred == 1, "Impaired", "Normal"),
        }
    )


def cross_validated_predictions(data: pd.DataFrame, config: TrainConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    x = data[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = data["label"].to_numpy()
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    rows: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []

    for fold, (train_index, test_index) in enumerate(skf.split(x, y), start=1):
        model = make_classifier(config, seed=config.seed + fold)
        model.fit(x[train_index], y[train_index])
        y_prob = model.predict_proba(x[test_index])[:, 1]
        metrics = evaluate_predictions(y[test_index], y_prob)
        metrics["fold"] = fold
        fold_metrics.append(metrics)
        fold_frame = data.iloc[test_index]
        y_pred = (y_prob >= 0.5).astype(int)
        rows.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "subject_id": fold_frame["subject_id"].to_numpy(),
                    "mmse_score": fold_frame[TARGET_COLUMN].to_numpy(),
                    "label": y[test_index],
                    "label_name": fold_frame["label_name"].to_numpy(),
                    "impairment_probability": y_prob,
                    "prediction": y_pred,
                    "prediction_name": np.where(y_pred == 1, "Impaired", "Normal"),
                }
            )
        )

    predictions = pd.concat(rows, ignore_index=True)
    overall_metrics = evaluate_predictions(predictions["label"].to_numpy(), predictions["impairment_probability"].to_numpy())
    overall_metrics["folds"] = fold_metrics
    return predictions, overall_metrics


def train(config: TrainConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_hcap_frame(Path(config.hcap_csv_zip), config.mmse_threshold)
    train_frame, validation_frame, test_frame = split_frame(data, config)

    model = make_classifier(config)
    model.fit(train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32), train_frame["label"].to_numpy())

    split_predictions = [
        predict_frame(model, train_frame, "train"),
        predict_frame(model, validation_frame, "validation"),
        predict_frame(model, test_frame, "test"),
    ]
    predictions = pd.concat(split_predictions, ignore_index=True)
    split_metrics = {
        split_name: evaluate_predictions(
            frame["label"].to_numpy(),
            pred["impairment_probability"].to_numpy(),
        )
        for split_name, frame, pred in (
            ("train", train_frame, split_predictions[0]),
            ("validation", validation_frame, split_predictions[1]),
            ("test", test_frame, split_predictions[2]),
        )
    }

    cv_predictions, cv_metrics = cross_validated_predictions(data, config)
    legacy_predictions = cv_predictions[["label", "impairment_probability"]].rename(
        columns={"label": "True", "impairment_probability": "Prob"}
    )

    metadata = {
        "model_version": "hcap_games_random_forest_v1",
        "label_mapping": LABELS,
        "target": f"{TARGET_COLUMN} < {config.mmse_threshold}",
        "config": asdict(config),
        "n_records": int(len(data)),
        "n_train": int(len(train_frame)),
        "n_validation": int(len(validation_frame)),
        "n_test": int(len(test_frame)),
        "feature_columns": FEATURE_COLUMNS,
        "metrics": split_metrics,
        "cross_validation": cv_metrics,
    }

    predictions_path = output_dir / "hcap_games_predictions.csv"
    cv_predictions_path = output_dir / "hcap_games_cv_predictions.csv"
    legacy_predictions_path = output_dir / "hcap_preds.csv"
    metrics_path = output_dir / "hcap_games_metrics.json"
    model_path = output_dir / "hcap_games_model.pkl"

    predictions.to_csv(predictions_path, index=False)
    cv_predictions.to_csv(cv_predictions_path, index=False)
    legacy_predictions.to_csv(legacy_predictions_path, index=False)
    metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with model_path.open("wb") as file:
        pickle.dump(
            {
                "model": model,
                "config": asdict(config),
                "feature_columns": FEATURE_COLUMNS,
                "label_mapping": LABELS,
            },
            file,
        )

    return {
        "metadata": metadata,
        "predictions_path": predictions_path,
        "cv_predictions_path": cv_predictions_path,
        "legacy_predictions_path": legacy_predictions_path,
        "metrics_path": metrics_path,
        "model_path": model_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an HCAP cognitive games classifier.")
    parser.add_argument("--hcap-csv-zip", type=Path, default=HCAP_CSV_ZIP)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--mmse-threshold", type=int, default=24)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=100)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        hcap_csv_zip=str(args.hcap_csv_zip),
        output_dir=str(args.output_dir),
        mmse_threshold=args.mmse_threshold,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    result = train(config)
    test_metrics = result["metadata"]["metrics"]["test"]
    cv_metrics = result["metadata"]["cross_validation"]
    print("HCAP games model complete")
    print(f"Test accuracy: {test_metrics['accuracy']:.3f}")
    print(f"Test ROC AUC: {test_metrics['roc_auc']}")
    print(f"CV ROC AUC: {cv_metrics['roc_auc']}")
    print(f"Predictions CSV: {result['predictions_path']}")
    print(f"Metrics JSON: {result['metrics_path']}")
    print(f"Model checkpoint: {result['model_path']}")
    print(f"Fusion-compatible CSV: {result['legacy_predictions_path']}")


if __name__ == "__main__":
    main()
