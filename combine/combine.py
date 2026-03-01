"""
Multimodal Fusion Pipeline with Monte Carlo Synthetic Dataset Generation
========================================================================
- Late fusion via weighted averaging across modalities
- Weights derived from each modality's unimodal confusion matrix performance
- Monte Carlo simulation to generate synthetic paired datasets
- AUC scoring and per-modality contribution analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DEFINE MODALITY CONFUSION MATRICES
#     Format: [[TN, FP], [FN, TP]]  (binary: Group A = 0, Group B = 1)
# ─────────────────────────────────────────────────────────────────────────────

MODALITY_CMS = {
    "Speech": np.array([
        [44, 56],
        [12, 88],
    ]),
"NHATS": np.array([
[40,60],
[7,93],
]),
    "HCAP": np.array([
        [46, 54],
        [5, 95],
    ])

}

CLASS_PRIOR = 0.5          # P(Group B) — adjust as needed
N_SYNTHETIC  = 10_000      # Monte Carlo subjects
N_HELD_OUT   = 2_000       # held-out synthetic evaluation set
RANDOM_SEED  = 42
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def cm_rates(cm: np.ndarray) -> dict:
    """Extract TN, FP, FN, TP and derived rates from a 2×2 confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    return dict(
        TN=tn, FP=fp, FN=fn, TP=tp,
        TPR=tp / (tp + fn + 1e-9),   # sensitivity / recall
        TNR=tn / (tn + fp + 1e-9),   # specificity
        PPV=tp / (tp + fp + 1e-9),   # precision
        ACC=(tp + tn) / (total + 1e-9),
        F1=2*tp / (2*tp + fp + fn + 1e-9),
        AUC_approx=0.5 * (tp/(tp+fn+1e-9) + tn/(tn+fp+1e-9)),  # balanced acc
    )


def confidence_weight(rates: dict) -> float:
    """
    Derive a scalar confidence weight for a modality from its confusion matrix.
    Uses balanced accuracy (= macro-averaged recall = approx AUC for binary).
    """
    return rates["AUC_approx"]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MONTE CARLO SYNTHETIC DATASET
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(cms: dict, n: int, class_prior: float) -> dict:
    """
    Generate n synthetic subjects.

    For each subject:
      1. Sample ground-truth label y ~ Bernoulli(class_prior)
      2. For each modality m, sample predicted label conditionally:
           if y=1: predict 1 w.p. TPR_m, else 0
           if y=0: predict 0 w.p. TNR_m, else 1

    Returns dict with 'y_true' and per-modality 'y_pred_{name}' arrays,
    plus soft scores (p_hat) drawn uniformly within the correct probability bin.
    """
    y_true = (np.random.rand(n) < class_prior).astype(int)
    data = {"y_true": y_true}

    for name, cm in cms.items():
        rates = cm_rates(cm)
        tpr, tnr = rates["TPR"], rates["TNR"]
        fpr = 1 - tnr

        # Hard predictions
        y_pred = np.where(
            y_true == 1,
            (np.random.rand(n) < tpr).astype(int),
            (np.random.rand(n) < fpr).astype(int),
        )

        # Soft probability scores (useful for AUC / weighted fusion)
        # Positive class: scores drawn from [tpr-δ, 1] or [0, fpr+δ]
        delta = 0.15
        p_hat = np.where(
            y_true == 1,
            np.random.uniform(np.clip(tpr - delta, 0, 1), 1.0, n),
            np.random.uniform(0.0, np.clip(fpr + delta, 0, 1), n),
        )
        # Re-jitter for misclassified samples so scores straddle the boundary
        p_hat = np.where(y_pred != y_true,
                         np.random.uniform(0.35, 0.65, n), p_hat)

        data[f"y_pred_{name}"] = y_pred
        data[f"p_hat_{name}"]  = np.clip(p_hat, 0, 1)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LATE FUSION
# ─────────────────────────────────────────────────────────────────────────────

def compute_initial_weights(cms: dict) -> dict:
    """Assign initial weights proportional to each modality's balanced accuracy."""
    raw = {name: confidence_weight(cm_rates(cm)) for name, cm in cms.items()}
    total = sum(raw.values())
    return {name: w / total for name, w in raw.items()}


def fused_score(data: dict, weights: dict) -> np.ndarray:
    """Weighted average of soft probability scores across modalities."""
    names = list(weights.keys())
    stack = np.stack([data[f"p_hat_{n}"] for n in names], axis=1)  # (n, M)
    w_arr = np.array([weights[n] for n in names])
    return stack @ w_arr  # (n,)


def negative_auc(w_raw: np.ndarray, names: list, data: dict) -> float:
    """Objective: –AUC for scipy.optimize (minimise)."""
    w = np.abs(w_raw) / (np.abs(w_raw).sum() + 1e-9)
    weights = dict(zip(names, w))
    scores = fused_score(data, weights)
    try:
        return -roc_auc_score(data["y_true"], scores)
    except Exception:
        return 0.0


def optimise_weights(cms: dict, data: dict) -> dict:
    """Learn fusion weights on synthetic training data to maximise AUC."""
    names = list(cms.keys())
    w0 = np.array([confidence_weight(cm_rates(cm)) for cm in cms.values()])
    w0 /= w0.sum()

    result = minimize(
        negative_auc, w0,
        args=(names, data),
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-5},
    )
    w_opt = np.abs(result.x) / np.abs(result.x).sum()
    return dict(zip(names, w_opt))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, scores)
    rates = cm_rates(cm)
    return dict(cm=cm, auc=auc, **rates)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(cms: dict, train_data: dict, test_data: dict,
             init_weights: dict, opt_weights: dict) -> None:

    modalities = list(cms.keys())
    n_mod = len(modalities)

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 26), facecolor="#0f1117")
    fig.suptitle(
        "Multimodal Fusion — Confusion Matrices, Weights & AUC",
        fontsize=18, fontweight="bold", color="white", y=0.98,
    )

    palette = {
        "bg":      "#0f1117",
        "panel":   "#1a1d27",
        "border":  "#2e3248",
        "accent":  "#5b6af7",
        "accent2": "#f76b8a",
        "text":    "#e2e8f0",
        "muted":   "#8892b0",
        "green":   "#43e97b",
        "amber":   "#f7c948",
    }

    cm_colors  = plt.cm.Blues
    roc_colors = ["#5b6af7","#f76b8a","#43e97b","#f7c948","#ff9f43"]

    # ── Row 1: Unimodal confusion matrices ─────────────────────────────────
    gs_top = gridspec.GridSpec(
        1, n_mod, figure=fig,
        top=0.91, bottom=0.72, left=0.04, right=0.96, wspace=0.35,
    )
    for i, name in enumerate(modalities):
        ax = fig.add_subplot(gs_top[i])
        ax.set_facecolor(palette["panel"])
        for spine in ax.spines.values():
            spine.set_edgecolor(palette["border"])

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cms[name],
            display_labels=["Group A", "Group B"],
        )
        disp.plot(ax=ax, colorbar=False, cmap=cm_colors)
        disp.im_.set_clim(0, cms[name].max())

        r = cm_rates(cms[name])
        ax.set_title(
            f"{name}\nACC={r['ACC']:.2f}  AUC≈{r['AUC_approx']:.2f}  F1={r['F1']:.2f}",
            fontsize=10, color=palette["text"], pad=8,
        )
        ax.tick_params(colors=palette["muted"])
        ax.xaxis.label.set_color(palette["muted"])
        ax.yaxis.label.set_color(palette["muted"])
        for text in disp.text_.ravel():
            text.set_color("white")
            text.set_fontsize(12)

    # ── Row 2: Fusion confusion matrices (initial vs optimised) + weight bars
    gs_mid = gridspec.GridSpec(
        1, 3, figure=fig,
        top=0.69, bottom=0.48, left=0.04, right=0.96, wspace=0.38,
    )

    for col, (label, weights) in enumerate(
        [("Initial Weights\n(Balanced-Acc Prior)", init_weights),
         ("Optimised Weights\n(AUC-Maximised)", opt_weights)]
    ):
        ax = fig.add_subplot(gs_mid[col])
        ax.set_facecolor(palette["panel"])
        for spine in ax.spines.values():
            spine.set_edgecolor(palette["border"])

        scores  = fused_score(test_data, weights)
        ev      = evaluate(test_data["y_true"], scores)
        disp    = ConfusionMatrixDisplay(
            confusion_matrix=ev["cm"],
            display_labels=["Group A", "Group B"],
        )
        disp.plot(ax=ax, colorbar=False, cmap=cm_colors)
        ax.set_title(
            f"Fused — {label}\nACC={ev['ACC']:.3f}  AUC={ev['auc']:.3f}  F1={ev['F1']:.3f}",
            fontsize=10, color=palette["text"], pad=8,
        )
        ax.tick_params(colors=palette["muted"])
        ax.xaxis.label.set_color(palette["muted"])
        ax.yaxis.label.set_color(palette["muted"])
        for text in disp.text_.ravel():
            text.set_color("white")
            text.set_fontsize(13)

    # Weight comparison bar chart
    ax_w = fig.add_subplot(gs_mid[2])
    ax_w.set_facecolor(palette["panel"])
    for spine in ax_w.spines.values():
        spine.set_edgecolor(palette["border"])

    x     = np.arange(n_mod)
    width = 0.35
    bars1 = ax_w.bar(x - width/2, [init_weights[m] for m in modalities],
                     width, label="Initial", color=palette["accent"], alpha=0.85)
    bars2 = ax_w.bar(x + width/2, [opt_weights[m] for m in modalities],
                     width, label="Optimised", color=palette["accent2"], alpha=0.85)
    ax_w.set_xticks(x)
    ax_w.set_xticklabels(modalities, rotation=25, ha="right",
                          fontsize=9, color=palette["text"])
    ax_w.set_ylabel("Fusion Weight", color=palette["muted"])
    ax_w.set_title("Modality Fusion Weights\nInitial vs Optimised",
                   fontsize=10, color=palette["text"])
    ax_w.tick_params(colors=palette["muted"])
    ax_w.legend(fontsize=9, labelcolor=palette["text"],
                facecolor=palette["panel"], edgecolor=palette["border"])
    ax_w.yaxis.grid(True, color=palette["border"], linewidth=0.5)
    ax_w.set_axisbelow(True)
    for bar in [*bars1, *bars2]:
        bar.set_edgecolor(palette["border"])

    # ── Row 3: ROC curves ──────────────────────────────────────────────────
    gs_roc = gridspec.GridSpec(
        1, 2, figure=fig,
        top=0.45, bottom=0.27, left=0.04, right=0.96, wspace=0.35,
    )

    for col, (title, weights) in enumerate(
        [("ROC — Initial Fusion Weights", init_weights),
         ("ROC — Optimised Fusion Weights", opt_weights)]
    ):
        ax = fig.add_subplot(gs_roc[col])
        ax.set_facecolor(palette["panel"])
        for spine in ax.spines.values():
            spine.set_edgecolor(palette["border"])

        # Unimodal ROC curves
        for i, name in enumerate(modalities):
            scores_m = test_data[f"p_hat_{name}"]
            fpr_m, tpr_m, _ = roc_curve(test_data["y_true"], scores_m)
            auc_m = roc_auc_score(test_data["y_true"], scores_m)
            ax.plot(fpr_m, tpr_m, color=roc_colors[i], lw=1.3, alpha=0.6,
                    linestyle="--", label=f"{name} (AUC={auc_m:.3f})")

        # Fused ROC
        scores_f = fused_score(test_data, weights)
        fpr_f, tpr_f, _ = roc_curve(test_data["y_true"], scores_f)
        auc_f = roc_auc_score(test_data["y_true"], scores_f)
        ax.plot(fpr_f, tpr_f, color="white", lw=2.5,
                label=f"Fused (AUC={auc_f:.3f})")

        ax.plot([0,1],[0,1], color=palette["muted"], lw=0.8, linestyle=":")
        ax.set_xlabel("False Positive Rate", color=palette["muted"])
        ax.set_ylabel("True Positive Rate", color=palette["muted"])
        ax.set_title(title, fontsize=10, color=palette["text"])
        ax.tick_params(colors=palette["muted"])
        ax.legend(fontsize=8, labelcolor=palette["text"],
                  facecolor=palette["panel"], edgecolor=palette["border"],
                  loc="lower right")
        ax.yaxis.grid(True, color=palette["border"], linewidth=0.5)
        ax.xaxis.grid(True, color=palette["border"], linewidth=0.5)

    # ── Row 4: Per-modality contribution & summary table ──────────────────
    gs_bot = gridspec.GridSpec(
        1, 2, figure=fig,
        top=0.24, bottom=0.04, left=0.04, right=0.96, wspace=0.38,
    )

    # Contribution = weight × unimodal AUC
    ax_contrib = fig.add_subplot(gs_bot[0])
    ax_contrib.set_facecolor(palette["panel"])
    for spine in ax_contrib.spines.values():
        spine.set_edgecolor(palette["border"])

    unimodal_aucs = {
        name: roc_auc_score(test_data["y_true"], test_data[f"p_hat_{name}"])
        for name in modalities
    }
    contributions = {
        name: opt_weights[name] * unimodal_aucs[name]
        for name in modalities
    }
    total_contrib = sum(contributions.values())
    contrib_pct   = {k: v/total_contrib*100 for k, v in contributions.items()}

    wedges, texts, autotexts = ax_contrib.pie(
        [contrib_pct[m] for m in modalities],
        labels=modalities,
        autopct="%1.1f%%",
        colors=roc_colors,
        startangle=140,
        wedgeprops=dict(edgecolor=palette["bg"], linewidth=2),
        pctdistance=0.78,
    )
    for t in texts:
        t.set_color(palette["text"]); t.set_fontsize(9)
    for at in autotexts:
        at.set_color("white"); at.set_fontsize(8)
    ax_contrib.set_title(
        "Modality Contribution\n(Optimised Weight × Unimodal AUC)",
        fontsize=10, color=palette["text"],
    )

    # Summary table
    ax_tbl = fig.add_subplot(gs_bot[1])
    ax_tbl.set_facecolor(palette["panel"])
    ax_tbl.axis("off")

    col_labels = ["Modality", "Unimodal\nAUC", "Unimodal\nF1",
                  "Init Wt", "Opt Wt", "Contribution"]
    rows = []
    for name in modalities:
        r = cm_rates(cms[name])
        rows.append([
            name,
            f"{unimodal_aucs[name]:.3f}",
            f"{r['F1']:.3f}",
            f"{init_weights[name]:.3f}",
            f"{opt_weights[name]:.3f}",
            f"{contrib_pct[name]:.1f}%",
        ])

    # Add fused row
    fused_ev = evaluate(test_data["y_true"], fused_score(test_data, opt_weights))
    rows.append([
        "FUSED (opt)",
        f"{fused_ev['auc']:.3f}",
        f"{fused_ev['F1']:.3f}",
        "—", "—", "100%",
    ])

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.7)

    # Style table cells
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(palette["panel"] if row > 0 else palette["accent"])
        cell.set_edgecolor(palette["border"])
        cell.set_text_props(
            color="white" if row == 0 else palette["text"],
            fontweight="bold" if row == 0 or row == len(rows) else "normal",
        )
        if row == len(rows):                      # fused summary row
            cell.set_facecolor("#1e2d3d")

    ax_tbl.set_title(
        "Performance Summary", fontsize=11, color=palette["text"], pad=14,
    )

    fig.patch.set_facecolor(palette["bg"])
    plt.savefig(
        "/mnt/user-data/outputs/multimodal_fusion_results.png",
        dpi=150, bbox_inches="tight", facecolor=palette["bg"],
    )
    plt.close()
    print("Figure saved → multimodal_fusion_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  MULTIMODAL FUSION PIPELINE")
    print("=" * 62)

    # ── Step 1: Unimodal statistics ───────────────────────────────────────
    print("\n[1] Unimodal Confusion Matrix Statistics")
    print("-" * 62)
    for name, cm in MODALITY_CMS.items():
        r = cm_rates(cm)
        print(f"  {name:<15} ACC={r['ACC']:.3f}  TPR={r['TPR']:.3f}  "
              f"TNR={r['TNR']:.3f}  AUC≈{r['AUC_approx']:.3f}  F1={r['F1']:.3f}")

    # ── Step 2: Initial (prior) weights ───────────────────────────────────
    init_weights = compute_initial_weights(MODALITY_CMS)
    print("\n[2] Initial Fusion Weights (proportional to balanced accuracy)")
    print("-" * 62)
    for name, w in init_weights.items():
        print(f"  {name:<15} weight = {w:.4f}")

    # ── Step 3: Monte Carlo synthetic dataset ─────────────────────────────
    print(f"\n[3] Generating Synthetic Dataset  (N={N_SYNTHETIC + N_HELD_OUT:,})")
    print("-" * 62)
    all_data  = generate_synthetic_dataset(
        MODALITY_CMS, N_SYNTHETIC + N_HELD_OUT, CLASS_PRIOR
    )
    # Split into train / held-out
    train_data = {k: v[:N_SYNTHETIC]  for k, v in all_data.items()}
    test_data  = {k: v[N_SYNTHETIC:]  for k, v in all_data.items()}
    print(f"  Train: {N_SYNTHETIC:,}  |  Held-out test: {N_HELD_OUT:,}")
    pos_rate = test_data["y_true"].mean()
    print(f"  Test class balance: Group B = {pos_rate:.1%}, Group A = {1-pos_rate:.1%}")

    # ── Step 4: Unimodal AUC on held-out ─────────────────────────────────
    print("\n[4] Unimodal AUC on Held-Out Synthetic Test Set")
    print("-" * 62)
    for name in MODALITY_CMS:
        auc = roc_auc_score(test_data["y_true"], test_data[f"p_hat_{name}"])
        print(f"  {name:<15} AUC = {auc:.4f}")

    # ── Step 5: Initial fusion evaluation ────────────────────────────────
    init_scores = fused_score(test_data, init_weights)
    init_ev     = evaluate(test_data["y_true"], init_scores)
    print("\n[5] Fused Model — Initial Weights (held-out test)")
    print("-" * 62)
    print(f"  AUC = {init_ev['auc']:.4f}  ACC = {init_ev['ACC']:.4f}  "
          f"F1 = {init_ev['F1']:.4f}  TPR = {init_ev['TPR']:.4f}  TNR = {init_ev['TNR']:.4f}")

    # ── Step 6: Optimise weights on training synthetic data ───────────────
    print("\n[6] Optimising Fusion Weights on Synthetic Train Set …")
    print("-" * 62)
    opt_weights = optimise_weights(MODALITY_CMS, train_data)
    for name, w in opt_weights.items():
        print(f"  {name:<15} weight = {w:.4f}")

    # ── Step 7: Optimised fusion evaluation ──────────────────────────────
    opt_scores = fused_score(test_data, opt_weights)
    opt_ev     = evaluate(test_data["y_true"], opt_scores)
    print("\n[7] Fused Model — Optimised Weights (held-out test)")
    print("-" * 62)
    print(f"  AUC = {opt_ev['auc']:.4f}  ACC = {opt_ev['ACC']:.4f}  "
          f"F1 = {opt_ev['F1']:.4f}  TPR = {opt_ev['TPR']:.4f}  TNR = {opt_ev['TNR']:.4f}")

    delta_auc = opt_ev['auc'] - init_ev['auc']
    print(f"\n  Δ AUC (opt − init) = {delta_auc:+.4f}")

    # ── Step 8: Per-modality contribution ─────────────────────────────────
    print("\n[8] Per-Modality Contribution (optimised weight × unimodal AUC)")
    print("-" * 62)
    contribs = {}
    for name in MODALITY_CMS:
        auc_m = roc_auc_score(test_data["y_true"], test_data[f"p_hat_{name}"])
        contribs[name] = opt_weights[name] * auc_m
    total = sum(contribs.values())
    for name, c in sorted(contribs.items(), key=lambda x: -x[1]):
        print(f"  {name:<15} {c/total*100:5.1f}%")

    # ── Step 9: Plot ───────────────────────────────────────────────────────
    # print("\n[9] Generating Visualisations …")
    # plot_all(MODALITY_CMS, train_data, test_data, init_weights, opt_weights)

    # print("\n" + "=" * 62)
    # print("  PIPELINE COMPLETE")
    # print("  Output → multimodal_fusion_results.png")
    # print("=" * 62)


if __name__ == "__main__":
    main()