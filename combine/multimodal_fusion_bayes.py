import itertools
import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import norm
from scipy.interpolate import interp1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler

import pathlib

OUT_DIR = pathlib.Path(__file__).parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODALITY_CMS = {
    "Speech": np.array([[72.2, 27.8], [40.7, 59.3]]),
    "Clock Drawing": np.array(
        [
            [87.18553459119497, 12.814465408805031],
            [27.142857142857142, 72.85714285714285],
        ]
    ),
    "Games": np.array([[97.31, 2.69], [68.97, 31.03]]),
}
CLASS_PRIOR = 0.5

N_SAMPLES = 100_000
N_BO_ITER = 40
N_INIT = 5
N_RESTARTS = 10
RNG_SEED = 42
INTERP_FPR = np.linspace(0, 1, 300)


def cm_to_rates(cm):
    cm = cm / cm.sum(axis=1, keepdims=True)
    tnr, fpr = cm[0]
    fnr, tpr = cm[1]
    return tnr, fpr, fnr, tpr


def rates_to_gaussian_params(tnr, fpr, fnr, tpr):
    eps = 1e-6
    fpr_c = float(np.clip(fpr, eps, 1 - eps))
    tpr_c = float(np.clip(tpr, eps, 1 - eps))
    mu0 = -norm.ppf(fpr_c)
    mu1 = norm.ppf(tpr_c)
    if mu1 < mu0:
        mu0, mu1 = -mu0, -mu1
    return (mu0, 1.0), (mu1, 1.0)


def synthesise_scores(mu0, s0, mu1, s1, n, prior, rng):
    n1 = int(round(n * prior))
    n0 = n - n1
    s = np.concatenate([rng.normal(mu0, s0, n0), rng.normal(mu1, s1, n1)])
    lbl = np.concatenate([np.zeros(n0), np.ones(n1)])
    return s, lbl


def weighted_fuse(score_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    return score_matrix @ w


def evaluate_weights(
    combo: Tuple[str, ...],
    weights: np.ndarray,
    modality_params: Dict,
    rng: np.random.Generator,
) -> Tuple[float, np.ndarray, np.ndarray]:
    score_cols, labels_ref = [], None
    for mod in combo:
        (mu0, s0), (mu1, s1) = modality_params[mod]
        scores, labels = synthesise_scores(
            mu0, s0, mu1, s1, N_SAMPLES, CLASS_PRIOR, rng
        )
        score_cols.append(scores)
        if labels_ref is None:
            labels_ref = labels
    fused = weighted_fuse(np.column_stack(score_cols), weights)
    auc = roc_auc_score(labels_ref, fused)
    fpr_r, tpr_r, _ = roc_curve(labels_ref, fused)
    interp_tpr = interp1d(fpr_r, tpr_r, bounds_error=False, fill_value=(0.0, 1.0))(
        INTERP_FPR
    )
    return auc, INTERP_FPR, interp_tpr


def expected_improvement(
    X_cand: np.ndarray, gp: GaussianProcessRegressor, y_best: float, xi: float = 0.01
) -> np.ndarray:
    mu, sigma = gp.predict(X_cand, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    Z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei


def next_candidate(
    gp: GaussianProcessRegressor,
    y_best: float,
    n_dims: int,
    rng: np.random.Generator,
    n_restarts: int = N_RESTARTS,
) -> np.ndarray:
    candidates = rng.dirichlet(np.ones(n_dims), size=2000)
    ei_vals = expected_improvement(candidates, gp, y_best)
    return candidates[np.argmax(ei_vals)]


modality_params: Dict = {}
print("=" * 65)
print(f"{'Modality':<10}  {'mu0':>7}  {'mu1':>7}  {'TPR':>6}  {'FPR':>6}  {'Flipped?'}")
print("-" * 65)
for name, cm in MODALITY_CMS.items():
    tnr, fpr, fnr, tpr = cm_to_rates(cm)
    eps = 1e-6
    fpr_c = float(np.clip(fpr, eps, 1 - eps))
    tpr_c = float(np.clip(tpr, eps, 1 - eps))
    mu0_raw = -norm.ppf(fpr_c)
    mu1_raw = norm.ppf(tpr_c)
    flipped = mu1_raw < mu0_raw
    (mu0, s0), (mu1, s1) = rates_to_gaussian_params(tnr, fpr, fnr, tpr)
    modality_params[name] = ((mu0, s0), (mu1, s1))
    print(
        f"{name:<10}  {mu0:>7.3f}  {mu1:>7.3f}  {tpr:>6.3f}  {fpr:>6.3f}  {'YES ← fixed' if flipped else 'no'}"
    )
print("=" * 65, "\n")

modality_names = list(modality_params.keys())

results = {}
results_curve = {}
roc_data = {}
best_weights = {}

kernel = Matern(nu=2.5)

for r in range(1, len(modality_names) + 1):
    for combo in itertools.combinations(modality_names, r):
        key = " + ".join(combo)
        n_dims = len(combo)
        rng = np.random.default_rng(RNG_SEED)

        X_obs: List[np.ndarray] = []
        y_obs: List[float] = []
        tpr_runs_best: List[np.ndarray] = []

        print(f"  BO → {key}  ({N_BO_ITER} iters, {n_dims}-D simplex)")

        best_so_far = -np.inf
        curve = []

        for it in range(N_BO_ITER):
            if it < N_INIT or n_dims == 1:
                w = rng.dirichlet(np.ones(n_dims))
            else:
                gp = GaussianProcessRegressor(
                    kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=3
                )
                gp.fit(np.array(X_obs), np.array(y_obs))
                w = next_candidate(gp, best_so_far, n_dims, rng)

            auc, fpr_arr, tpr_arr = evaluate_weights(combo, w, modality_params, rng)
            X_obs.append(w)
            y_obs.append(auc)

            if auc > best_so_far:
                best_so_far = auc
                best_w = w.copy()
                best_fpr = fpr_arr
                best_tpr = tpr_arr

            curve.append(best_so_far)

        results[key] = best_so_far
        results_curve[key] = curve
        best_weights[key] = best_w

        tpr_runs = []
        for rep in range(30):
            rng2 = np.random.default_rng(RNG_SEED + rep + 1000)
            _, _, tpr_arr = evaluate_weights(combo, best_w, modality_params, rng2)
            tpr_runs.append(tpr_arr)
        tpr_mat = np.array(tpr_runs)

        roc_data[key] = dict(
            mean_fpr=INTERP_FPR,
            mean_tpr=tpr_mat.mean(axis=0),
            std_tpr=tpr_mat.std(axis=0),
            auc_mean=best_so_far,
            auc_std=tpr_mat.std(axis=0).mean(),
        )
        print(f"    best AUC = {best_so_far:.4f}  weights = {np.round(best_w, 3)}")


def shapley_contributions(modality_names, results):
    n = len(modality_names)
    shapley = {m: 0.0 for m in modality_names}
    for r in range(1, n + 1):
        for combo in itertools.combinations(modality_names, r):
            key = " + ".join(combo)
            v_with = results[key]
            for mod in combo:
                without = tuple(m for m in combo if m != mod)
                v_without = 0.5 if not without else results[" + ".join(without)]
                marginal = v_with - v_without
                weight = (
                    math.factorial(len(without))
                    * math.factorial(n - len(combo))
                    / math.factorial(n)
                )
                shapley[mod] += weight * marginal
    return shapley


shapley = shapley_contributions(modality_names, results)
shap_total = sum(shapley.values())
shapley_pct = {m: v / shap_total * 100 for m, v in shapley.items()}

print(f"\n{'Combo':<35}  {'Optimal Weights':<45}  {'Best AUC':>9}")
print("-" * 97)
for r in range(1, len(modality_names) + 1):
    for combo in itertools.combinations(modality_names, r):
        key = " + ".join(combo)
        bw = best_weights[key]
        wstr = "  ".join(f"{m}: {bw[i]*100:.1f}%" for i, m in enumerate(combo))
        print(f"{key:<35}  {wstr:<45}  {results[key]:.4f}")

print("\n" + "=" * 97)
print("Shapley-based modality contributions (from BO-optimal AUCs):")
for m, pct in shapley_pct.items():
    print(f"  {m:<10}: {pct:.2f}%  (Shapley value = {shapley[m]:.5f})")

PALETTE = {
    "Speech": "#4C72B0",
    "Clock Drawing": "#DD8452",
    "Games": "#55A868",
    "Speech + Clock Drawing": "#9B59B6",
    "Speech + Games": "#E74C3C",
    "Clock Drawing + Games": "#1ABC9C",
    "Speech + Clock Drawing + Games": "#2C3E50",
}
all_keys = list(results.keys())

fig1, ax1 = plt.subplots(figsize=(13, 6))
bars = ax1.bar(
    np.arange(len(all_keys)),
    [results[k] for k in all_keys],
    color=[PALETTE[k] for k in all_keys],
    edgecolor="white",
    linewidth=0.8,
    alpha=0.88,
)
ax1.set_xticks(np.arange(len(all_keys)))
ax1.set_xticklabels(all_keys, rotation=25, ha="right", fontsize=10)
ax1.set_ylabel("Best AUC (Bayesian Optimisation)", fontsize=12)
ax1.set_title(
    f"Bayesian Optimisation Fusion AUC — All Modality Combinations\n"
    f"(N={N_SAMPLES} subjects × {N_BO_ITER} BO evaluations per combo)",
    fontsize=13,
)
ax1.set_ylim(0.0, 1.08)
ax1.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
for bar, k in zip(bars, all_keys):
    mv = results[k]
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.012,
        f"{mv:.3f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )
    bw = best_weights[k]
    combo = tuple(k.split(" + "))
    wlabel = "\n".join(f"{m[:3]}:{bw[i]*100:.0f}%" for i, m in enumerate(combo))
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        0.02,
        wlabel,
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="white",
        alpha=0.85,
    )

legend_els = [Patch(facecolor=PALETTE[k], label=k) for k in all_keys]
ax1.legend(handles=legend_els, loc="upper right", fontsize=8, ncol=2)
plt.tight_layout()
fig1.savefig(OUT_DIR / "multimodal_fusion_bayes_bar.png", dpi=150)
print("\nSaved → multimodal_fusion_bayes_bar.png")

single_keys = [k for k in all_keys if "+" not in k]

PANEL_SIZE = 5.6
GAP_H = 0.52
GAP_W = 0.42
fig2, axes2 = plt.subplots(
    2,
    2,
    figsize=(PANEL_SIZE * 2 + GAP_W * 3, PANEL_SIZE * 2 + GAP_H * 3),
)
fig2.subplots_adjust(hspace=0.38, wspace=0.32)
fig2.suptitle(
    f"ROC Curves — Individual Modalities & Full Combined Fusion (± 1 SD)\n"
    f"N={N_SAMPLES} subjects · {N_BO_ITER} BO evaluations per combo",
    fontsize=14,
    y=0.995,
)


def plot_roc_panel(ax, key, color, title=None):
    rd = roc_data[key]
    fpr, tpr, std = rd["mean_fpr"], rd["mean_tpr"], rd["std_tpr"]
    auc_m = rd["auc_mean"]
    bw = best_weights[key]
    combo = tuple(key.split(" + "))
    wlabel = "  ".join(f"{m[:3]}:{bw[i]*100:.0f}%" for i, m in enumerate(combo))
    ax.plot(fpr, tpr, color=color, lw=2.4, label=f"AUC = {auc_m:.3f}\n{wlabel}")
    ax.fill_between(
        fpr,
        np.clip(tpr - std, 0, 1),
        np.clip(tpr + std, 0, 1),
        alpha=0.20,
        color=color,
    )
    ax.plot([0, 1], [0, 1], "k--", lw=0.9, alpha=0.45)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(title or key, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.tick_params(labelsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)


positions = [(0, 0), (0, 1), (1, 0)]
for (row, col), key in zip(positions, single_keys):
    plot_roc_panel(axes2[row][col], key, PALETTE[key])

ax_all = axes2[1][1]
draw_order = (
    [k for k in all_keys if "+" not in k]
    + [k for k in all_keys if k.count("+") == 1]
    + [k for k in all_keys if k.count("+") == 2]
)
for key in draw_order:
    rd = roc_data[key]
    is_triple = key.count("+") == 2
    is_pair = key.count("+") == 1
    lw = 3.0 if is_triple else 1.8
    ls = "-" if is_triple else ("--" if is_pair else "-.")
    zord = 3 if is_triple else 2
    ax_all.plot(
        rd["mean_fpr"],
        rd["mean_tpr"],
        color=PALETTE[key],
        lw=lw,
        linestyle=ls,
        label=f"{key}  (AUC={rd['auc_mean']:.3f})",
        zorder=zord,
    )
    ax_all.fill_between(
        rd["mean_fpr"],
        np.clip(rd["mean_tpr"] - rd["std_tpr"], 0, 1),
        np.clip(rd["mean_tpr"] + rd["std_tpr"], 0, 1),
        alpha=0.10 if is_triple else 0.05,
        color=PALETTE[key],
    )
ax_all.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.4, label="Chance")
ax_all.set_xlim(-0.02, 1.02)
ax_all.set_ylim(-0.02, 1.08)
ax_all.set_xlabel("False Positive Rate", fontsize=10)
ax_all.set_ylabel("True Positive Rate", fontsize=10)
ax_all.set_title("All Combinations Overlaid", fontsize=12, fontweight="bold")
ax_all.legend(
    fontsize=7.5, loc="lower right", ncol=1, framealpha=0.9, edgecolor="#cccccc"
)
ax_all.tick_params(labelsize=9)
ax_all.set_aspect("equal")
ax_all.grid(alpha=0.2)

plt.savefig(OUT_DIR / "multimodal_fusion_bayes_roc.png", dpi=150, bbox_inches="tight")
print("Saved → multimodal_fusion_bayes_roc.png")

fig3, ax3 = plt.subplots(figsize=(11, 5))
for key in all_keys:
    curve = results_curve[key]
    ax3.plot(
        range(1, len(curve) + 1),
        curve,
        color=PALETTE[key],
        lw=2,
        label=key,
        marker="o",
        markersize=3,
        markevery=5,
    )
ax3.axvline(
    N_INIT, color="grey", linestyle=":", lw=1.2, label=f"GP starts (iter {N_INIT})"
)
ax3.set_xlabel("BO Iteration", fontsize=11)
ax3.set_ylabel("Best AUC so far", fontsize=11)
ax3.set_title("Bayesian Optimisation Convergence — All Combos", fontsize=13)
ax3.legend(fontsize=8, ncol=2, loc="lower right")
ax3.set_ylim(0.45, 1.02)
ax3.grid(alpha=0.3)
plt.tight_layout()
fig3.savefig(OUT_DIR / "multimodal_fusion_bayes_conv.png", dpi=150)
print("Saved → multimodal_fusion_bayes_conv.png")

plt.show()
print("\nDone.")
