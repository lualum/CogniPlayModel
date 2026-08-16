from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEECH_DATASET_DIR = REPO_ROOT / "dataset" / "speech"
SPEECH_OUTPUT_DIR = REPO_ROOT / "output_performance" / "speech"

SPEAKER_COLORS = {
    "PAR": "#2563eb",
    "INV": "#d97706",
}

PREDICTION_OUTPUT_GLOBS = (
    "test_results_task*.csv",
    "test_results-task*.csv",
    "model_predictions*.csv",
    "model_outputs*.csv",
    "predictions*.csv",
)

DATASET_ONLY_OUTPUT_NAMES = (
    "adresso_stats_overview.png",
    "recording_duration_distribution.png",
    "speaker_duration_by_subject.png",
    "participant_share_vs_duration.png",
    "adresso_segmentation_stats_visualizer.html",
)

def _subject_sort_key(path_or_id: Path | str) -> tuple[str, int | str]:
    subject_id = Path(path_or_id).stem if isinstance(path_or_id, Path) else str(path_or_id)
    prefix = "".join(ch for ch in subject_id if not ch.isdigit())
    suffix = "".join(ch for ch in subject_id if ch.isdigit())
    return prefix, int(suffix) if suffix else subject_id


def resolve_adresso_dir(adresso_dir: str | Path | None = None) -> Path:
    """Find the ADReSSo data directory from the organized dataset layout or legacy paths."""
    if adresso_dir is not None:
        candidates = [Path(adresso_dir)]
    else:
        here = Path(__file__).resolve().parent
        cwd = Path.cwd()
        candidates = [
            SPEECH_DATASET_DIR / "ADReSSo",
            cwd / "ADReSSo",
            cwd / "dataset" / "speech" / "ADReSSo",
            cwd / "speech" / "ADReSSo",
            here / "ADReSSo",
        ]

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "segmentation").is_dir():
            return candidate

    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find ADReSSo/segmentation. Searched:\n" + searched
    )


def _clean_segment_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")])

    required = {"speaker", "begin", "end"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.loc[:, ["speaker", "begin", "end"]].copy()
    df["subject_id"] = path.stem
    df["speaker"] = df["speaker"].astype(str).str.strip().str.upper()
    df["begin"] = pd.to_numeric(df["begin"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df = df.dropna(subset=["speaker", "begin", "end"])
    df = df[df["end"] >= df["begin"]]
    df = df.sort_values(["begin", "end", "speaker"], kind="mergesort").reset_index(drop=True)

    df.insert(1, "turn_index", np.arange(1, len(df) + 1))
    df["begin_ms"] = df["begin"].astype(int)
    df["end_ms"] = df["end"].astype(int)
    df = df.drop(columns=["begin", "end"])
    df["duration_ms"] = df["end_ms"] - df["begin_ms"]
    df["duration_s"] = df["duration_ms"] / 1000.0

    previous_end = df["end_ms"].shift()
    df["gap_from_previous_ms"] = np.where(
        previous_end.isna(), 0, np.maximum(df["begin_ms"] - previous_end, 0)
    ).astype(int)
    df["overlap_with_previous_ms"] = np.where(
        previous_end.isna(), 0, np.maximum(previous_end - df["begin_ms"], 0)
    ).astype(int)
    return df


def load_segments(adresso_dir: str | Path | None = None) -> pd.DataFrame:
    """Load and normalize every ADReSSo segmentation CSV into one tidy table."""
    root = resolve_adresso_dir(adresso_dir)
    files = sorted((root / "segmentation").glob("*.csv"), key=_subject_sort_key)
    if not files:
        raise FileNotFoundError(f"No segmentation CSV files found under {root / 'segmentation'}")

    frames = [_clean_segment_file(path) for path in files]
    segments = pd.concat(frames, ignore_index=True)
    segments = segments[
        [
            "subject_id",
            "turn_index",
            "speaker",
            "begin_ms",
            "end_ms",
            "duration_ms",
            "duration_s",
            "gap_from_previous_ms",
            "overlap_with_previous_ms",
        ]
    ]
    return segments


def _safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else 0.0


def _safe_median(values: pd.Series) -> float:
    return float(values.median()) if len(values) else 0.0


def compute_session_stats(segments: pd.DataFrame) -> pd.DataFrame:
    """Build per-subject stats from cleaned segmentation rows."""
    records: list[dict[str, Any]] = []

    for subject_id, group in segments.groupby("subject_id", sort=False):
        group = group.sort_values(["begin_ms", "end_ms"], kind="mergesort")
        recording_duration_s = max(
            (group["end_ms"].max() - group["begin_ms"].min()) / 1000.0, 0.001
        )
        total_turns = int(len(group))

        speaker_counts = group["speaker"].value_counts()
        speaker_duration = group.groupby("speaker")["duration_s"].sum()
        par = group[group["speaker"] == "PAR"]
        inv = group[group["speaker"] == "INV"]
        gaps_s = group["gap_from_previous_ms"] / 1000.0
        nonzero_gaps_s = gaps_s[gaps_s > 0]
        switches = int((group["speaker"] != group["speaker"].shift()).sum() - 1)
        switches = max(switches, 0)
        total_speech_s = float(group["duration_s"].sum())
        par_duration_s = float(speaker_duration.get("PAR", 0.0))
        inv_duration_s = float(speaker_duration.get("INV", 0.0))

        flags = []
        if par.empty:
            flags.append("no_PAR")
        if inv.empty:
            flags.append("no_INV")
        if group["overlap_with_previous_ms"].sum() > 0:
            flags.append("overlap")
        if group["duration_ms"].min() == 0:
            flags.append("zero_duration")

        records.append(
            {
                "subject_id": subject_id,
                "total_turns": total_turns,
                "recording_duration_s": recording_duration_s,
                "recording_duration_min": recording_duration_s / 60.0,
                "total_speech_s": total_speech_s,
                "speech_density": total_speech_s / recording_duration_s,
                "silent_gap_s": float(group["gap_from_previous_ms"].sum() / 1000.0),
                "overlap_s": float(group["overlap_with_previous_ms"].sum() / 1000.0),
                "par_turns": int(speaker_counts.get("PAR", 0)),
                "inv_turns": int(speaker_counts.get("INV", 0)),
                "par_duration_s": par_duration_s,
                "inv_duration_s": inv_duration_s,
                "par_speech_share": par_duration_s / total_speech_s if total_speech_s else 0.0,
                "inv_speech_share": inv_duration_s / total_speech_s if total_speech_s else 0.0,
                "mean_turn_duration_s": _safe_mean(group["duration_s"]),
                "median_turn_duration_s": _safe_median(group["duration_s"]),
                "max_turn_duration_s": float(group["duration_s"].max()),
                "mean_par_duration_s": _safe_mean(par["duration_s"]),
                "median_par_duration_s": _safe_median(par["duration_s"]),
                "mean_inv_duration_s": _safe_mean(inv["duration_s"]),
                "median_inv_duration_s": _safe_median(inv["duration_s"]),
                "mean_gap_s": _safe_mean(nonzero_gaps_s),
                "median_gap_s": _safe_median(nonzero_gaps_s),
                "max_gap_s": float(gaps_s.max()) if len(gaps_s) else 0.0,
                "turn_switches": switches,
                "turn_switch_rate": switches / max(total_turns - 1, 1),
                "par_turns_per_minute": int(speaker_counts.get("PAR", 0))
                / max(recording_duration_s / 60.0, 0.001),
                "inv_turns_per_minute": int(speaker_counts.get("INV", 0))
                / max(recording_duration_s / 60.0, 0.001),
                "quality_flags": ", ".join(flags) if flags else "ok",
            }
        )

    stats = pd.DataFrame.from_records(records)
    return stats.sort_values("subject_id", key=lambda col: col.map(_subject_sort_key)).reset_index(
        drop=True
    )


def compute_dataset_summary(stats: pd.DataFrame, segments: pd.DataFrame) -> dict[str, Any]:
    longest = stats.loc[stats["recording_duration_s"].idxmax()]
    most_par = stats.loc[stats["par_duration_s"].idxmax()]
    return {
        "sessions": int(stats["subject_id"].nunique()),
        "segments": int(len(segments)),
        "total_recording_minutes": float(stats["recording_duration_min"].sum()),
        "mean_recording_duration_s": float(stats["recording_duration_s"].mean()),
        "median_recording_duration_s": float(stats["recording_duration_s"].median()),
        "median_turns_per_session": float(stats["total_turns"].median()),
        "mean_par_speech_share": float(stats["par_speech_share"].mean()),
        "total_silent_gap_s": float(stats["silent_gap_s"].sum()),
        "total_overlap_s": float(stats["overlap_s"].sum()),
        "longest_session_id": str(longest["subject_id"]),
        "longest_session_duration_s": float(longest["recording_duration_s"]),
        "most_participant_speech_id": str(most_par["subject_id"]),
        "most_participant_speech_s": float(most_par["par_duration_s"]),
    }


def _format_number(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def remove_dataset_only_outputs(output_dir: str | Path) -> None:
    """Remove stale plots/pages that describe only the dataset, not model performance."""
    output_dir = Path(output_dir)
    for name in DATASET_ONLY_OUTPUT_NAMES:
        (output_dir / name).unlink(missing_ok=True)


def _standardize_features(stats: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values = stats.loc[:, columns].astype(float)
    centered = values - values.mean()
    scaled = centered / values.std(ddof=0).replace(0, 1)
    return scaled.fillna(0.0)


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def _prediction_present_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return series.notna() & text.ne("").fillna(False)


def _infer_prediction_task(path: Path) -> tuple[str, str, str]:
    stem = path.stem.lower().replace("-", "_")
    if "task1" in stem:
        return "classification", "AD classification", "numeric 0/1 diagnosis predictions"
    if "task2" in stem:
        return "regression", "MMSE regression", "numeric MMSE predictions from 0 to 30"
    return "prediction_output", "Prediction output", "non-empty prediction values"


def discover_prediction_output_files(adresso_dir: str | Path, output_dir: str | Path) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()

    def append(path: Path) -> None:
        path = path.resolve()
        if path.is_file() and path not in seen:
            discovered.append(path)
            seen.add(path)

    for root in (
        Path(output_dir),
        SPEECH_OUTPUT_DIR / "adresso_submissions",
        Path(adresso_dir),
    ):
        if not root.exists():
            continue
        for pattern in PREDICTION_OUTPUT_GLOBS:
            for candidate in sorted(root.glob(pattern)):
                append(candidate)

    return discovered


def summarize_prediction_output(path: Path) -> dict[str, Any]:
    raw = pd.read_csv(path)
    pred_col = _find_column(
        raw,
        ("prediction", "y_pred", "pred", "predicted", "predicted_label", "predicted_mmse"),
    )
    task, task_label, validation_rule = _infer_prediction_task(path)
    total = int(len(raw))

    if pred_col is None:
        return {
            "file": path.name,
            "task": task,
            "task_label": task_label,
            "records": total,
            "predictions_filled": 0,
            "predictions_missing": total,
            "completion_rate": 0.0,
            "valid_prediction_count": 0,
            "valid_prediction_rate": 0.0,
            "validation_rule": f"No prediction column found; expected {validation_rule}",
        }

    present_mask = _prediction_present_mask(raw[pred_col])
    predictions = raw.loc[present_mask, pred_col]
    prediction_numeric = pd.to_numeric(predictions, errors="coerce")
    present_count = int(present_mask.sum())

    if task == "classification":
        valid_mask = prediction_numeric.notna() & prediction_numeric.isin([0, 1])
    elif task == "regression":
        valid_mask = prediction_numeric.notna() & prediction_numeric.between(0, 30)
    else:
        valid_mask = pd.Series(True, index=predictions.index)

    valid_count = int(valid_mask.sum()) if present_count else 0
    return {
        "file": path.name,
        "task": task,
        "task_label": task_label,
        "records": total,
        "predictions_filled": present_count,
        "predictions_missing": total - present_count,
        "completion_rate": present_count / total if total else 0.0,
        "valid_prediction_count": valid_count,
        "valid_prediction_rate": valid_count / total if total else 0.0,
        "validation_rule": validation_rule,
    }


def write_model_overview_figure(
    stats: pd.DataFrame,
    output_dir: str | Path,
    adresso_dir: str | Path,
) -> Path:
    """Write a compact matplotlib figure with model-output metrics first."""
    cache_dir = SPEECH_OUTPUT_DIR / ".cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "model_input_feature_overview.png"
    prediction_summaries = [
        summarize_prediction_output(path)
        for path in discover_prediction_output_files(adresso_dir, output_dir)
    ]

    feature_columns = [
        "recording_duration_min",
        "par_speech_share",
        "par_turns_per_minute",
        "mean_par_duration_s",
        "mean_gap_s",
        "overlap_s",
    ]
    feature_labels = [
        "Recording min",
        "PAR speech share",
        "PAR turns/min",
        "Mean PAR turn s",
        "Mean gap s",
        "Overlap s",
    ]

    ordered = stats.sort_values("par_speech_share", kind="mergesort").reset_index(drop=True)
    heatmap = _standardize_features(ordered, feature_columns)

    flag_counts: dict[str, int] = {}
    for raw_flags in stats["quality_flags"].fillna("ok"):
        for flag in str(raw_flags).split(","):
            flag = flag.strip() or "ok"
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    flag_items = sorted(flag_counts.items(), key=lambda item: (-item[1], item[0]))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Speech Model Metrics and Input Checks", fontsize=15)

    if prediction_summaries:
        labels = [summary["task_label"] for summary in prediction_summaries]
        x = np.arange(len(labels))
        width = 0.36
        completion = [summary["completion_rate"] * 100 for summary in prediction_summaries]
        valid = [summary["valid_prediction_rate"] * 100 for summary in prediction_summaries]
        bars_filled = axes[0, 0].bar(x - width / 2, completion, width, label="Filled", color="#2563eb")
        bars_valid = axes[0, 0].bar(x + width / 2, valid, width, label="Valid", color="#0891b2")
        axes[0, 0].set_title("Model Prediction Coverage")
        axes[0, 0].set_ylabel("Rows (%)")
        axes[0, 0].set_ylim(0, 100)
        axes[0, 0].set_xticks(x, labels, rotation=20, ha="right")
        axes[0, 0].legend()
        for bars in (bars_filled, bars_valid):
            for bar in bars:
                height = bar.get_height()
                axes[0, 0].text(
                    bar.get_x() + bar.get_width() / 2,
                    min(height + 2, 96),
                    f"{height:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        filled = [summary["predictions_filled"] for summary in prediction_summaries]
        missing = [summary["predictions_missing"] for summary in prediction_summaries]
        axes[0, 1].bar(x, filled, label="Filled", color="#2563eb")
        axes[0, 1].bar(x, missing, bottom=filled, label="Missing", color="#d97706")
        axes[0, 1].set_title("Prediction Rows Filled vs Missing")
        axes[0, 1].set_ylabel("Rows")
        axes[0, 1].set_xticks(x, labels, rotation=20, ha="right")
        axes[0, 1].legend()
        for index, summary in enumerate(prediction_summaries):
            axes[0, 1].text(
                index,
                summary["records"] + max(summary["records"] * 0.02, 1),
                f"{summary['predictions_filled']}/{summary['records']}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        for ax in axes[0]:
            ax.axis("off")
        axes[0, 0].text(
            0.02,
            0.62,
            "No model prediction files found.",
            fontsize=13,
            weight="bold",
            transform=axes[0, 0].transAxes,
        )
        axes[0, 0].text(
            0.02,
            0.42,
            "Add predictions or labeled y_true/y_pred results to compute model metrics.",
            fontsize=11,
            transform=axes[0, 0].transAxes,
        )

    image = axes[1, 0].imshow(heatmap.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    axes[1, 0].set_title("Speech Feature Matrix")
    axes[1, 0].set_xticks(range(len(feature_labels)), feature_labels, rotation=35, ha="right")
    axes[1, 0].set_yticks([])
    axes[1, 0].set_ylabel("Subjects sorted by PAR speech share")
    heatmap_bar = fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)
    heatmap_bar.set_label("z-score")

    axes[1, 1].bar([label for label, _ in flag_items], [count for _, count in flag_items], color="#d97706")
    axes[1, 1].set_title("Segmentation Quality Flags")
    axes[1, 1].set_xlabel("Flag")
    axes[1, 1].set_ylabel("Subjects")
    axes[1, 1].tick_params(axis="x", labelrotation=25)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def run_analysis(
    adresso_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    adresso_root = resolve_adresso_dir(adresso_dir)
    speech_dir = adresso_root.parent
    output_root = Path(output_dir) if output_dir is not None else SPEECH_OUTPUT_DIR / "cha_stats"
    output_root.mkdir(parents=True, exist_ok=True)

    segments = load_segments(adresso_root)
    stats = compute_session_stats(segments)
    summary = compute_dataset_summary(stats, segments)

    segments_path = output_root / "adresso_segments_clean.csv"
    stats_path = output_root / "adresso_session_stats.csv"
    summary_path = output_root / "adresso_dataset_summary.json"
    segments.to_csv(segments_path, index=False)
    stats.to_csv(stats_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    remove_dataset_only_outputs(output_root)
    model_overview_plot_path = write_model_overview_figure(stats, output_root, adresso_root)

    return {
        "adresso_dir": adresso_root,
        "speech_dir": speech_dir,
        "output_dir": output_root,
        "segments": segments,
        "stats": stats,
        "summary": summary,
        "segments_path": segments_path,
        "stats_path": stats_path,
        "summary_path": summary_path,
        "feature_plot_path": model_overview_plot_path,
        "model_overview_plot_path": model_overview_plot_path,
        "performance_results_hint": output_root / "model_performance_results.csv",
    }


def main() -> None:
    results = run_analysis()
    summary = results["summary"]
    print("ADReSSo segmentation analysis complete")
    print(f"Sessions: {summary['sessions']}")
    print(f"Segments: {summary['segments']}")
    print(f"Total recording minutes: {_format_number(summary['total_recording_minutes'])}")
    print(f"Mean PAR speech share: {_format_number(summary['mean_par_speech_share'] * 100)}%")
    print(f"Feature CSV: {results['stats_path']}")
    print(f"Model overview figure: {results['model_overview_plot_path']}")
    print(f"Performance input hint: {results['performance_results_hint']}")


if __name__ == "__main__":
    main()
