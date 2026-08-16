from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.speech.analyze_cha_stats import run_analysis


RANDOM_STATE = 42
MODEL_VERSION = "unlabeled_kmeans_speech_baseline_v1"

MODEL_FEATURE_COLUMNS = [
    "total_turns",
    "recording_duration_min",
    "speech_density",
    "silent_gap_s",
    "overlap_s",
    "par_turns",
    "inv_turns",
    "par_speech_share",
    "inv_speech_share",
    "mean_turn_duration_s",
    "median_turn_duration_s",
    "max_turn_duration_s",
    "mean_par_duration_s",
    "median_par_duration_s",
    "mean_inv_duration_s",
    "median_inv_duration_s",
    "mean_gap_s",
    "median_gap_s",
    "max_gap_s",
    "turn_switch_rate",
    "par_turns_per_minute",
    "inv_turns_per_minute",
]

RISK_WEIGHTS = {
    "speech_density": -0.25,
    "silent_gap_s": 0.55,
    "overlap_s": 0.35,
    "par_speech_share": -0.85,
    "mean_par_duration_s": -0.45,
    "mean_gap_s": 0.65,
    "max_gap_s": 0.35,
    "turn_switch_rate": -0.35,
    "par_turns_per_minute": -0.65,
}


def _prepare_matplotlib_cache() -> None:
    cache_dir = REPO_ROOT / "output_performance" / "speech" / ".cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


def _feature_matrix(stats: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(MODEL_FEATURE_COLUMNS).difference(stats.columns))
    if missing:
        raise ValueError(f"Session stats are missing model features: {missing}")

    features = stats.loc[:, MODEL_FEATURE_COLUMNS].copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    return features


def train_unlabeled_baseline(stats: pd.DataFrame) -> dict[str, Any]:
    features = _feature_matrix(stats)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features)
    z_features = pd.DataFrame(x_scaled, columns=MODEL_FEATURE_COLUMNS, index=stats.index)

    pca = PCA(n_components=2)
    pca_values = pca.fit_transform(x_scaled)

    kmeans = KMeans(n_clusters=2, n_init=50, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(x_scaled)

    weighted_risk = np.zeros(len(stats), dtype=float)
    total_weight = 0.0
    for column, weight in RISK_WEIGHTS.items():
        weighted_risk += z_features[column].to_numpy() * weight
        total_weight += abs(weight)
    weighted_risk = weighted_risk / max(total_weight, 1e-9)
    weighted_risk = (weighted_risk - weighted_risk.mean()) / max(weighted_risk.std(), 1e-9)

    temp = pd.DataFrame({"cluster": clusters, "risk_index": weighted_risk})
    cluster_risk = temp.groupby("cluster")["risk_index"].mean()
    high_risk_cluster = int(cluster_risk.idxmax())
    low_risk_cluster = int(cluster_risk.idxmin())

    distances = np.linalg.norm(x_scaled[:, None, :] - kmeans.cluster_centers_[None, :, :], axis=2)
    cluster_margin = distances[:, low_risk_cluster] - distances[:, high_risk_cluster]
    cluster_probability = _sigmoid(cluster_margin / max(cluster_margin.std(), 1e-9))
    heuristic_probability = _sigmoid(weighted_risk)
    ad_probability = np.clip(0.65 * cluster_probability + 0.35 * heuristic_probability, 0.0, 1.0)

    predictions = pd.DataFrame(
        {
            "subject_id": stats["subject_id"],
            "ad_probability": ad_probability,
            "ad_prediction": (ad_probability >= 0.5).astype(int),
            "ad_confidence": np.abs(ad_probability - 0.5) * 2.0,
            "predicted_mmse": np.clip(30.0 - 12.0 * ad_probability, 0.0, 30.0),
            "cluster": clusters,
            "risk_cluster": np.where(clusters == high_risk_cluster, "higher", "lower"),
            "risk_index": weighted_risk,
            "pca_1": pca_values[:, 0],
            "pca_2": pca_values[:, 1],
        }
    )
    predictions["predicted_mmse"] = predictions["predicted_mmse"].round(1)

    silhouette = float(silhouette_score(x_scaled, clusters)) if len(set(clusters)) > 1 else 0.0
    metrics = {
        "model_version": MODEL_VERSION,
        "label_status": "No ground-truth diagnosis or MMSE labels were found; these are model diagnostics, not clinical performance metrics.",
        "n_subjects": int(len(predictions)),
        "prediction_coverage": {
            "ad_classification": 1.0,
            "mmse_regression": 1.0,
        },
        "valid_prediction_rate": {
            "ad_classification": float(predictions["ad_prediction"].isin([0, 1]).mean()),
            "mmse_regression": float(predictions["predicted_mmse"].between(0, 30).mean()),
        },
        "classification": {
            "predicted_class_counts": {
                str(key): int(value)
                for key, value in predictions["ad_prediction"].value_counts().sort_index().items()
            },
            "mean_ad_probability": float(predictions["ad_probability"].mean()),
            "mean_confidence": float(predictions["ad_confidence"].mean()),
            "median_confidence": float(predictions["ad_confidence"].median()),
            "uncertain_subjects_confidence_below_0_20": int((predictions["ad_confidence"] < 0.20).sum()),
        },
        "regression": {
            "mean_predicted_mmse": float(predictions["predicted_mmse"].mean()),
            "median_predicted_mmse": float(predictions["predicted_mmse"].median()),
            "min_predicted_mmse": float(predictions["predicted_mmse"].min()),
            "max_predicted_mmse": float(predictions["predicted_mmse"].max()),
        },
        "clustering": {
            "silhouette_score": silhouette,
            "high_risk_cluster": high_risk_cluster,
            "low_risk_cluster": low_risk_cluster,
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "pca_total_explained_variance": float(pca.explained_variance_ratio_.sum()),
        },
        "features_used": MODEL_FEATURE_COLUMNS,
    }

    return {
        "predictions": predictions,
        "metrics": metrics,
        "pca": pca,
        "kmeans": kmeans,
    }


def write_prediction_files(predictions: pd.DataFrame, metrics: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "model_predictions_unlabeled.csv"
    metrics_path = output_dir / "model_unlabeled_metrics.json"
    task1_path = output_dir / "test_results_task1_generated.csv"
    task2_path = output_dir / "test_results_task2_generated.csv"

    predictions.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(_json_ready(metrics), indent=2), encoding="utf-8")
    predictions.loc[:, ["subject_id", "ad_prediction"]].rename(
        columns={"subject_id": "ID", "ad_prediction": "Prediction"}
    ).to_csv(task1_path, index=False)
    predictions.loc[:, ["subject_id", "predicted_mmse"]].rename(
        columns={"subject_id": "ID", "predicted_mmse": "Prediction"}
    ).to_csv(task2_path, index=False)

    return {
        "predictions": predictions_path,
        "metrics": metrics_path,
        "task1": task1_path,
        "task2": task2_path,
    }


def write_diagnostic_figures(
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    _prepare_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / "model_diagnostics.png"
    rankings_path = output_dir / "model_prediction_rankings.png"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Speech Model Output Diagnostics", fontsize=15)

    metric_labels = [
        "AD valid",
        "MMSE valid",
        "Mean conf",
        "Silhouette",
        "PCA var",
    ]
    metric_values = [
        metrics["valid_prediction_rate"]["ad_classification"] * 100,
        metrics["valid_prediction_rate"]["mmse_regression"] * 100,
        metrics["classification"]["mean_confidence"] * 100,
        metrics["clustering"]["silhouette_score"] * 100,
        metrics["clustering"]["pca_total_explained_variance"] * 100,
    ]
    metric_colors = ["#2563eb", "#0891b2", "#d97706", "#7c3aed", "#059669"]
    bars = axes[0, 0].bar(metric_labels, metric_values, color=metric_colors)
    axes[0, 0].set_title("Core Model Diagnostics")
    axes[0, 0].set_ylabel("Percent / normalized score")
    axes[0, 0].set_ylim(0, 108)
    axes[0, 0].tick_params(axis="x", labelrotation=20)
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            f"{height:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[0, 1].hist(predictions["ad_confidence"], bins=np.linspace(0, 1, 11), color="#2563eb", alpha=0.84)
    axes[0, 1].axvline(
        metrics["classification"]["mean_confidence"],
        color="#111827",
        lw=1.3,
        label=f"Mean {metrics['classification']['mean_confidence']:.2f}",
    )
    axes[0, 1].axvline(0.20, color="#d97706", lw=1.3, linestyle="--", label="Uncertain < 0.20")
    axes[0, 1].set_title("AD Prediction Confidence")
    axes[0, 1].set_xlabel("Confidence, distance from 0.50")
    axes[0, 1].set_ylabel("Subjects")
    axes[0, 1].legend()

    scatter = axes[1, 0].scatter(
        predictions["pca_1"],
        predictions["pca_2"],
        c=predictions["ad_probability"],
        cmap="viridis",
        s=54,
        alpha=0.86,
        edgecolor="#111827",
        linewidth=0.35,
    )
    axes[1, 0].set_title(
        "Model Latent Space "
        f"(silhouette={metrics['clustering']['silhouette_score']:.3f})"
    )
    axes[1, 0].set_xlabel("PCA 1")
    axes[1, 0].set_ylabel("PCA 2")
    colorbar = fig.colorbar(scatter, ax=axes[1, 0], fraction=0.046, pad=0.04)
    colorbar.set_label("AD probability")

    axes[1, 1].hist(predictions["predicted_mmse"], bins=12, color="#0891b2", alpha=0.84)
    axes[1, 1].axvline(
        metrics["regression"]["mean_predicted_mmse"],
        color="#111827",
        lw=1.3,
        label=f"Mean {metrics['regression']['mean_predicted_mmse']:.1f}",
    )
    axes[1, 1].set_title("Predicted MMSE Distribution")
    axes[1, 1].set_xlabel("Predicted MMSE")
    axes[1, 1].set_ylabel("Subjects")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(diagnostics_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    fig.suptitle("Speech Model Subject-Level Predictions", fontsize=15)

    high_risk = predictions.sort_values("ad_probability", ascending=False).head(20)
    high_risk = high_risk.sort_values("ad_probability")
    axes[0].barh(high_risk["subject_id"], high_risk["ad_probability"], color="#2563eb")
    axes[0].axvline(0.5, color="#111827", lw=1.2, linestyle="--")
    axes[0].set_title("Highest Predicted AD Probability")
    axes[0].set_xlabel("AD probability")
    axes[0].set_xlim(0, 1)

    uncertain = predictions.assign(distance=(predictions["ad_probability"] - 0.5).abs())
    uncertain = uncertain.sort_values("distance").head(20).sort_values("ad_confidence", ascending=False)
    axes[1].barh(uncertain["subject_id"], uncertain["ad_confidence"], color="#d97706")
    axes[1].axvline(0.20, color="#111827", lw=1.2, linestyle="--")
    axes[1].set_title("Most Uncertain AD Predictions")
    axes[1].set_xlabel("Confidence")
    axes[1].set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(rankings_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "diagnostics": diagnostics_path,
        "rankings": rankings_path,
    }


def run_model() -> dict[str, Any]:
    analysis = run_analysis()
    output_dir = Path(analysis["output_dir"])
    trained = train_unlabeled_baseline(analysis["stats"])
    file_paths = write_prediction_files(trained["predictions"], trained["metrics"], output_dir)
    figure_paths = write_diagnostic_figures(trained["predictions"], trained["metrics"], output_dir)

    return {
        "analysis": analysis,
        "predictions": trained["predictions"],
        "metrics": trained["metrics"],
        "files": file_paths,
        "figures": figure_paths,
    }


def main() -> None:
    result = run_model()
    metrics = result["metrics"]
    print("Speech baseline model complete")
    print(f"Model: {metrics['model_version']}")
    print(f"Subjects: {metrics['n_subjects']}")
    print(f"AD coverage: {metrics['prediction_coverage']['ad_classification'] * 100:.0f}%")
    print(f"MMSE coverage: {metrics['prediction_coverage']['mmse_regression'] * 100:.0f}%")
    print(f"Mean confidence: {metrics['classification']['mean_confidence']:.3f}")
    print(f"Silhouette score: {metrics['clustering']['silhouette_score']:.3f}")
    print(f"Predictions CSV: {result['files']['predictions']}")
    print(f"Metrics JSON: {result['files']['metrics']}")
    print(f"Diagnostics figure: {result['figures']['diagnostics']}")
    print(f"Rankings figure: {result['figures']['rankings']}")


if __name__ == "__main__":
    main()
