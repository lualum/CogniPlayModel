import itertools
import json
import os
import pathlib
import zipfile
from dataclasses import dataclass
from html import escape
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(pathlib.Path(__file__).resolve().parents[2] / ".matplotlib-cache"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(pathlib.Path(__file__).resolve().parents[2] / ".cache"),
)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "output_performance" / "fusion"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
N_BOOTSTRAP = 500
INTERP_FPR = np.linspace(0, 1, 501)

PALETTE = {
    "Speech": "#4C72B0",
    "Clock Drawing": "#DD8452",
    "Games": "#55A868",
    "Speech + Clock Drawing": "#9B59B6",
    "Speech + Games": "#E74C3C",
    "Clock Drawing + Games": "#1ABC9C",
    "Speech + Clock Drawing + Games": "#2C3E50",
}


@dataclass(frozen=True)
class PredictionSpec:
    name: str
    path: pathlib.Path
    label_col: str
    probability_col: str


PREDICTION_SPECS = (
    PredictionSpec(
        name="Speech",
        path=REPO_ROOT
        / "output_performance"
        / "speech"
        / "pitt_transformer"
        / "pitt_transformer_predictions.csv",
        label_col="label",
        probability_col="dementia_probability",
    ),
    PredictionSpec(
        name="Clock Drawing",
        path=REPO_ROOT
        / "output_performance"
        / "clocks"
        / "nhats_cnn"
        / "nhats_clock_cnn_predictions.csv",
        label_col="label",
        probability_col="impairment_probability",
    ),
    PredictionSpec(
        name="Games",
        path=REPO_ROOT
        / "output_performance"
        / "games"
        / "hcap"
        / "hcap_games_predictions.csv",
        label_col="label",
        probability_col="impairment_probability",
    ),
)


def choose_eval_split(df: pd.DataFrame) -> str:
    if "split" not in df.columns:
        return "all"
    available = set(df["split"].dropna().astype(str))
    for split in ("test", "validation", "valid"):
        if split in available:
            return split
    return "all"


def load_predictions(spec: PredictionSpec) -> pd.DataFrame:
    if not spec.path.exists():
        raise FileNotFoundError(f"Missing prediction file for {spec.name}: {spec.path}")

    df = pd.read_csv(spec.path)
    missing = {spec.label_col, spec.probability_col} - set(df.columns)
    if missing:
        raise ValueError(f"{spec.path} is missing required columns: {sorted(missing)}")

    eval_split = choose_eval_split(df)
    if eval_split != "all":
        df = df[df["split"].astype(str) == eval_split].copy()
    else:
        df = df.copy()

    out = pd.DataFrame(
        {
            "modality": spec.name,
            "split": eval_split,
            "subject_id": df["subject_id"].astype(str) if "subject_id" in df else "",
            "label": pd.to_numeric(df[spec.label_col], errors="coerce"),
            "probability": pd.to_numeric(df[spec.probability_col], errors="coerce"),
        }
    ).dropna(subset=["label", "probability"])

    out["label"] = out["label"].astype(int)
    out["probability"] = out["probability"].clip(0.0, 1.0)

    class_count = out["label"].nunique()
    if class_count != 2:
        raise ValueError(
            f"{spec.name} {eval_split} split must contain two classes; found {class_count}"
        )
    return out


def roc_summary(labels: np.ndarray, scores: np.ndarray) -> Dict[str, object]:
    fpr, tpr, _ = roc_curve(labels, scores)
    interp_tpr = np.interp(INTERP_FPR, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tpr[-1] = 1.0
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "fpr": fpr,
        "tpr": tpr,
        "interp_tpr": interp_tpr,
    }


def bootstrap_roc(
    labels: np.ndarray,
    scores: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
) -> Tuple[np.ndarray, np.ndarray]:
    tpr_runs: List[np.ndarray] = []
    n = len(labels)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if np.unique(labels[idx]).size < 2:
            continue
        fpr, tpr, _ = roc_curve(labels[idx], scores[idx])
        interp_tpr = np.interp(INTERP_FPR, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        tpr_runs.append(interp_tpr)

    tpr_matrix = np.vstack(tpr_runs)
    return tpr_matrix.mean(axis=0), tpr_matrix.std(axis=0)


def make_score_cohort(
    datasets: Dict[str, pd.DataFrame],
    combo: Tuple[str, ...],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    positives = [
        datasets[name][datasets[name]["label"] == 1]["probability"].to_numpy()
        for name in combo
    ]
    negatives = [
        datasets[name][datasets[name]["label"] == 0]["probability"].to_numpy()
        for name in combo
    ]
    n_pos = min(len(values) for values in positives)
    n_neg = min(len(values) for values in negatives)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Cannot fuse {combo}: one class has no examples.")

    fused_pos = np.zeros(n_pos)
    fused_neg = np.zeros(n_neg)
    for weight, pos_scores, neg_scores in zip(weights, positives, negatives):
        fused_pos += weight * rng.choice(pos_scores, size=n_pos, replace=False)
        fused_neg += weight * rng.choice(neg_scores, size=n_neg, replace=False)

    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    scores = np.concatenate([fused_pos, fused_neg])
    return labels, scores


def simplex_grid(n_dims: int, step: float = 0.05) -> Iterable[np.ndarray]:
    ticks = np.round(np.arange(0.0, 1.0 + step, step), 10)
    if n_dims == 1:
        yield np.ones(1)
        return

    for values in itertools.product(ticks, repeat=n_dims):
        if abs(sum(values) - 1.0) <= 1e-9:
            yield np.array(values, dtype=float)


def optimize_fusion_weights(
    datasets: Dict[str, pd.DataFrame],
    combo: Tuple[str, ...],
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    best_auc = -np.inf
    best_weights = None
    best_labels = None
    best_scores = None
    rng = np.random.default_rng(RNG_SEED + len(combo))

    for weights in simplex_grid(len(combo)):
        labels, scores = make_score_cohort(datasets, combo, weights, rng)
        score_auc = roc_auc_score(labels, scores)
        if score_auc > best_auc:
            best_auc = score_auc
            best_weights = weights.copy()
            best_labels = labels
            best_scores = scores

    assert best_weights is not None
    mean_tpr, std_tpr = bootstrap_roc(best_labels, best_scores, rng)
    return best_weights, float(best_auc), best_labels, mean_tpr, std_tpr


def label_for_combo(combo: Tuple[str, ...]) -> str:
    return " + ".join(combo)


def save_curve_csv(curve_data: Dict[str, Dict[str, object]]) -> None:
    rows = []
    for key, data in curve_data.items():
        std_tpr = data.get("std_tpr")
        if std_tpr is None:
            std_tpr = np.zeros_like(data["mean_tpr"])
        for fpr, tpr, std in zip(INTERP_FPR, data["mean_tpr"], std_tpr):
            rows.append(
                {
                    "configuration": key,
                    "false_positive_rate": fpr,
                    "true_positive_rate": tpr,
                    "true_positive_rate_std": std,
                    "auc": data["auc"],
                    "method": data["method"],
                }
            )
    pd.DataFrame(rows).to_csv(OUT_DIR / "auc_curves.csv", index=False)


def plot_bar(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    x = np.arange(len(summary))
    bars = ax.bar(
        x,
        summary["auc"],
        color=[PALETTE[name] for name in summary["configuration"]],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["configuration"], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.9)
    for bar, value in zip(bars, summary["auc"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    plt.tight_layout()
    for filename in (
        "auc_modality_configuration.png",
        "auc_modality_configuration.svg",
        "auc_modality_configuration.jpg",
    ):
        fig.savefig(FIG_DIR / filename, dpi=180, bbox_inches="tight")
    for filename in (
        "auc_modality_configuration_no_legend.png",
        "auc_modality_configuration_no_legend.svg",
        "auc_modality_configuration_no_legend.jpg",
    ):
        fig.savefig(OUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rocs(curve_data: Dict[str, Dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    for key, data in curve_data.items():
        is_fusion = "+" in key
        linewidth = 2.8 if key.count("+") == 2 else (2.1 if is_fusion else 1.8)
        linestyle = "-" if key.count("+") == 2 else ("--" if is_fusion else "-")
        ax.plot(
            INTERP_FPR,
            data["mean_tpr"],
            color=PALETTE[key],
            linewidth=linewidth,
            linestyle=linestyle,
            label=f"{key} (AUC={data['auc']:.3f})",
        )
        std_tpr = data.get("std_tpr")
        if std_tpr is not None:
            ax.fill_between(
                INTERP_FPR,
                np.clip(data["mean_tpr"] - std_tpr, 0, 1),
                np.clip(data["mean_tpr"] + std_tpr, 0, 1),
                color=PALETTE[key],
                alpha=0.08,
                linewidth=0,
            )
    ax.plot(
        [0, 1],
        [0, 1],
        color="#555555",
        linestyle=":",
        linewidth=1.2,
        label="Chance",
    )
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.22)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    for filename in (
        "auc_curves.png",
        "auc_curves.svg",
        "auc_curves.jpg",
        "multimodal_fusion_bayes_roc.png",
        "multimodal_fusion_bayes_roc.jpg",
    ):
        fig.savefig(FIG_DIR / filename, dpi=180, bbox_inches="tight")
    fig.savefig(OUT_DIR / "multimodal_fusion_bayes_roc.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT_DIR / "multimodal_fusion_bayes_roc.jpg", dpi=180, bbox_inches="tight")
    plt.close(fig)


def figure_legend_lines(summary: pd.DataFrame) -> List[str]:
    auc_values = {
        row["configuration"]: float(row["auc"])
        for _, row in summary.iterrows()
    }
    method_values = {
        row["configuration"]: row["method"]
        for _, row in summary.iterrows()
    }
    color_text = "; ".join(
        f"{name}: {PALETTE[name]}" for name in summary["configuration"]
    )
    style_text = (
        "solid lines: individual modalities and full three-modality fusion estimate; "
        "dashed lines: two-modality fusion estimates; dotted gray diagonal: chance."
    )
    auc_text = "; ".join(
        f"{name}: AUC {auc_values[name]:.4f}" for name in summary["configuration"]
    )
    fusion_note = (
        "Individual modality curves were calculated directly from held-out test "
        "prediction files. Combined modality curves were estimated by optimized "
        "weighted fusion of class-specific held-out score distributions because "
        "the source prediction files had no overlapping subject IDs across "
        "modalities."
    )
    return [
        (
            "Figure 1: Area under the receiver operating characteristic curve by "
            "modality configuration for dementia or cognitive impairment "
            "classification. Bars show AUC values from the held-out test split. "
            f"{auc_text}. {fusion_note} Color mapping: {color_text}."
        ),
        (
            "Figure 2: Receiver operating characteristic curves by modality "
            "configuration. Curves show true positive rate versus false positive "
            "rate. Shaded bands on combined modality curves indicate +/- 1 "
            "standard deviation across 500 bootstrap resamples. "
            f"{auc_text}. {fusion_note} Line styles: {style_text} Color mapping: "
            f"{color_text}."
        ),
    ]


def write_docx(paragraphs: List[str], path: pathlib.Path) -> None:
    body = "".join(
        f"<w:p><w:r><w:t>{escape(paragraph)}</w:t></w:r></w:p>"
        for paragraph in paragraphs
    )
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="720" w:right="1440" w:bottom="720" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>
"""
    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types_xml)
        docx.writestr("_rels/.rels", rels_xml)
        docx.writestr("word/document.xml", document_xml)


def write_figure_legends(summary: pd.DataFrame) -> None:
    lines = figure_legend_lines(summary)
    text = "\n\n".join(lines) + "\n"
    (OUT_DIR / "figure_legends.txt").write_text(text)
    (OUT_DIR / "figure_legends.md").write_text(text)
    write_docx(lines, OUT_DIR / "figure_legends.docx")


def main() -> None:
    datasets = {spec.name: load_predictions(spec) for spec in PREDICTION_SPECS}
    modality_names = [spec.name for spec in PREDICTION_SPECS]
    subject_sets = {
        name: set(df["subject_id"].astype(str))
        for name, df in datasets.items()
        if "subject_id" in df
    }
    shared_subject_counts = {
        label_for_combo(combo): len(
            set.intersection(*(subject_sets[name] for name in combo))
        )
        for r in range(2, len(modality_names) + 1)
        for combo in itertools.combinations(modality_names, r)
    }

    curve_data: Dict[str, Dict[str, object]] = {}
    rows = []

    for name in modality_names:
        df = datasets[name]
        labels = df["label"].to_numpy()
        scores = df["probability"].to_numpy()
        roc = roc_summary(labels, scores)
        curve_data[name] = {
            "auc": roc["auc"],
            "mean_tpr": roc["interp_tpr"],
            "std_tpr": None,
            "method": "direct_prediction_file",
        }
        rows.append(
            {
                "configuration": name,
                "auc": roc["auc"],
                "method": "direct_prediction_file",
                "split": df["split"].iloc[0],
                "n": len(df),
                "n_positive": int(labels.sum()),
                "n_negative": int((labels == 0).sum()),
                "weights": name + ":1.000",
            }
        )

    for r in range(2, len(modality_names) + 1):
        for combo in itertools.combinations(modality_names, r):
            key = label_for_combo(combo)
            weights, fusion_auc, labels, mean_tpr, std_tpr = optimize_fusion_weights(
                datasets, combo
            )
            curve_data[key] = {
                "auc": fusion_auc,
                "mean_tpr": mean_tpr,
                "std_tpr": std_tpr,
                "method": "score_distribution_fusion_no_subject_overlap",
            }
            rows.append(
                {
                    "configuration": key,
                    "auc": fusion_auc,
                    "method": "score_distribution_fusion_no_subject_overlap",
                    "split": "test",
                    "n": len(labels),
                    "n_positive": int(labels.sum()),
                    "n_negative": int((labels == 0).sum()),
                    "weights": ";".join(
                        f"{name}:{weight:.3f}" for name, weight in zip(combo, weights)
                    ),
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "auc_summary.csv", index=False)
    save_curve_csv(curve_data)

    metadata = {
        "source_files": {
            spec.name: str(spec.path.relative_to(REPO_ROOT))
            for spec in PREDICTION_SPECS
        },
        "rng_seed": RNG_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "fusion_note": (
            "No subject IDs overlap across modalities, so fused curves are estimated by "
            "combining held-out score distributions within class labels. Direct "
            "subject-level fusion requires prediction files for the same participants."
        ),
        "shared_subject_counts": shared_subject_counts,
    }
    (OUT_DIR / "auc_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    plot_bar(summary)
    plot_rocs(curve_data)
    write_figure_legends(summary)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved summaries and figures to {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
