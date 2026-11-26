#!/usr/bin/env python
"""
Generates publication-grade Polish pilot hyperparameter selection plots for Section 4
of the master's thesis (docs/Praca_magisterska/figures/).

Implements all methodological requirements:
1. Budget selection strictly by intra-method repeatability (rho_rep across 5 seeds)
   and computational cost — NEVER by RQ1 perturbation stability (shown only as ceiling reference).
2. Uncertainty bands (+/- SEM across N reviews) on ALL plotted series
   (repeatability, execution time, completeness error, and RQ1 ceiling reference).
3. Explicit sample and seed counts in titles/annotations (N reviews, S = 5 seeds -> 10 pairs/review).
4. Length-stratified analysis of Permutation SHAP (short/medium/long review terciles)
   comparing fixed `max_evals` vs length-scaled `B(n) = c * n`.
5. Consistent terminology: Permutation SHAP (`max_evals`), never KernelSHAP (`nsamples`).

Usage:
    .venv\\Scripts\\python.exe scripts/generate_pilot_hyperparam_plots.py --only-plots
"""

import os
import sys
import argparse

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_IG_CSV = os.path.join(PROJECT_ROOT, "results", "hyperparameter_tuning", "ig", "ig_convergence_stabilized.csv")
DEFAULT_GRID_CSV = os.path.join(PROJECT_ROOT, "results", "rq1_grid", "rq1_grid_summary.csv")
DEFAULT_PILOT_5SEED_CSV = os.path.join(PROJECT_ROOT, "results", "hyperparameter_tuning", "pilot_5seed_raw.csv")
THESIS_FIG_DIR = os.path.join(PROJECT_ROOT, "docs", "Praca_magisterska", "figures")


def _comma_fmt(decimals: int = 2):
    return FuncFormatter(lambda val, _: f"{val:.{decimals}f}".replace(".", ","))


def _int_fmt():
    return FuncFormatter(lambda val, _: f"{int(round(val))}")


def _configure_matplotlib():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
    })


def _save_fig(fig, basename: str, extra_dir: str = None):
    os.makedirs(THESIS_FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = os.path.join(THESIS_FIG_DIR, f"{basename}.{ext}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] {out_path}")
        if extra_dir:
            os.makedirs(extra_dir, exist_ok=True)
            extra_path = os.path.join(extra_dir, f"{basename}.{ext}")
            fig.savefig(extra_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_budget_apriori(x_vals: np.ndarray, rep_means: np.ndarray, threshold: float, delta_tol: float = 0.03) -> float:
    """
    A-priori selection rule:
    Smallest budget where rho_rep >= threshold AND marginal gain to the next budget step < delta_tol
    (or first budget >= threshold if no larger step satisfies both).
    """
    candidates = []
    for i in range(len(x_vals)):
        if rep_means[i] >= threshold:
            if i == len(x_vals) - 1 or (rep_means[i + 1] - rep_means[i]) < delta_tol:
                return float(x_vals[i])
            candidates.append(float(x_vals[i]))
    if candidates:
        return candidates[0]
    return float(x_vals[-1])


def _aggregate_5seed(df_raw: pd.DataFrame, family: str) -> pd.DataFrame:
    sub = df_raw[df_raw["method_family"] == family].copy()
    agg = sub.groupby("param_value", as_index=False).agg(
        n_reviews=("sample_id", "nunique"),
        n_seeds=("n_seeds", "first"),
        rep_mean=("rep_spearman_mean", "mean"),
        rep_sem=("rep_spearman_mean", "sem"),
        time_mean=("time_s_mean", "mean"),
        time_sem=("time_s_mean", "sem"),
        fwd_mean=("forward_passes_mean", "mean"),
    ).sort_values("param_value")
    return agg


def plot_shap_pilot(df_pilot: pd.DataFrame, df_grid: pd.DataFrame = None, model_label: str = "Qwen3-1.7B-Base"):
    agg = _aggregate_5seed(df_pilot, "SHAP_fixed")
    x = agg["param_value"].to_numpy(dtype=float)
    y_rep = agg["rep_mean"].to_numpy(dtype=float)
    y_rep_sem = agg["rep_sem"].fillna(0.0).to_numpy(dtype=float)
    y_time = agg["time_mean"].to_numpy(dtype=float)
    y_time_sem = agg["time_sem"].fillna(0.0).to_numpy(dtype=float)
    n_rev = int(agg["n_reviews"].iloc[0])
    n_seeds = int(agg["n_seeds"].iloc[0])

    chosen_b = select_budget_apriori(x, y_rep, threshold=0.80, delta_tol=0.03)

    fig, ax1 = plt.subplots(figsize=(7.8, 4.6), dpi=300)
    c_rep = "#2b5c8f"
    c_stab = "#1b9e77"
    c_time = "#d95f02"

    l1 = ax1.plot(x, y_rep, marker="o", markersize=5.5, color=c_rep, linewidth=2.2,
                  label=r"Powtarzalność własna $\rho_{\mathrm{rep}}$ (± SEM)")
    ax1.fill_between(x, np.maximum(0.0, y_rep - y_rep_sem), np.minimum(1.0, y_rep + y_rep_sem),
                     color=c_rep, alpha=0.18)

    l_thr = ax1.axhline(0.80, color="#666666", linestyle=":", linewidth=1.1, alpha=0.85,
                        label=r"Próg reguły a priori ($\rho_{\mathrm{rep}} \geq 0{,}80$)")

    lines = l1 + [l_thr]
    if df_grid is not None and "SHAP" in df_grid["method_family"].values:
        g_shap = df_grid[df_grid["method_family"] == "SHAP"].sort_values("param_value")
        gx = g_shap["param_value"].to_numpy(dtype=float)
        gy = g_shap["spearman_mean"].to_numpy(dtype=float)
        gy_sem = g_shap["spearman_sem"].to_numpy(dtype=float)
        l2 = ax1.plot(gx, gy, marker="s", markersize=4.8, color=c_stab, linewidth=1.6, linestyle="--", alpha=0.8,
                      label="Stabilność RQ1 — ilustracja sufitu (± SEM, nie kryterium)")
        ax1.fill_between(gx, gy - gy_sem, gy + gy_sem, color=c_stab, alpha=0.12)
        lines = lines + l2

    ax1.axvline(chosen_b, color="#555555", linestyle="-.", linewidth=1.3, alpha=0.85)
    ax1.annotate(
        f"Wybrany budżet\n(max_evals = {int(chosen_b)})",
        xy=(chosen_b, 0.20),
        xytext=(max(100.0, chosen_b - 680), 0.14),
        fontsize=8.5,
        color="#333333",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )

    ax1.set_xlabel(r"Budżet Permutation SHAP ($\mathrm{max\_evals}$)")
    ax1.set_ylabel(r"Korelacja rang Spearmana $\rho$", color=c_rep)
    ax1.set_ylim(0.0, 1.04)
    ax1.set_xlim(0, max(x) * 1.06)
    ax1.set_xticks(x)
    ax1.yaxis.set_major_formatter(_comma_fmt(2))
    ax1.tick_params(axis="y", labelcolor=c_rep)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)

    ax2 = ax1.twinx()
    l3 = ax2.plot(x, y_time, marker="^", markersize=5.5, color=c_time, linewidth=1.9, linestyle=":",
                  label="Średni czas na wyjaśnienie [s] (± SEM)")
    ax2.fill_between(x, np.maximum(0.0, y_time - y_time_sem), y_time + y_time_sem, color=c_time, alpha=0.16)
    ax2.set_ylabel("Średni czas na wyjaśnienie [s]", color=c_time)
    ax2.set_ylim(0.0, float(np.max(y_time + y_time_sem)) * 1.18)
    ax2.yaxis.set_major_formatter(_comma_fmt(1))
    ax2.tick_params(axis="y", labelcolor=c_time)
    ax2.spines["top"].set_visible(False)

    lines = lines + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95)

    ax1.set_title(f"Pilotażowy dobór budżetu Permutation SHAP ({model_label}, N={n_rev}, {n_seeds} ziaren)", pad=10)
    fig.tight_layout()
    extra_dir = os.path.join(PROJECT_ROOT, "results", "hyperparameter_tuning", "shap", "plots")
    _save_fig(fig, "pilot_shap_convergence", extra_dir=extra_dir)
    return chosen_b


def plot_shap_length_scaling(df_pilot: pd.DataFrame, model_label: str = "Qwen3-1.7B-Base"):
    """
    Two-panel plot comparing fixed max_evals vs length-scaled B(n) = c * n
    across review token-length terciles (short, medium, long).
    """
    bin_meta = [
        ("short", "Krótkie recenzje", "#2b5c8f", "o"),
        ("medium", "Średnie recenzje", "#1b9e77", "s"),
        ("long", "Długie recenzje", "#d95f02", "^"),
    ]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.4), dpi=300, sharey=True)

    # Panel A: Fixed max_evals by length tercile
    sub_fix = df_pilot[df_pilot["method_family"] == "SHAP_fixed"].copy()
    for bkey, blabel, color, marker in bin_meta:
        bsub = sub_fix[sub_fix["length_bin"] == bkey]
        if bsub.empty:
            continue
        n_min, n_max = int(bsub["n_tokens"].min()), int(bsub["n_tokens"].max())
        agg = bsub.groupby("param_value", as_index=False).agg(
            rep_mean=("rep_spearman_mean", "mean"),
            rep_sem=("rep_spearman_mean", "sem"),
        ).sort_values("param_value")
        x = agg["param_value"].to_numpy(dtype=float)
        y = agg["rep_mean"].to_numpy(dtype=float)
        y_sem = agg["rep_sem"].fillna(0.0).to_numpy(dtype=float)
        axL.plot(x, y, marker=marker, markersize=5.2, color=color, linewidth=2.0,
                 label=f"{blabel} (n={n_min}–{n_max} tok.)")
        axL.fill_between(x, np.maximum(0.0, y - y_sem), np.minimum(1.0, y + y_sem), color=color, alpha=0.15)

    axL.axhline(0.80, color="#666666", linestyle=":", linewidth=1.1, alpha=0.85)
    axL.set_xlabel(r"Stały budżet ($\mathrm{max\_evals}$)")
    axL.set_ylabel(r"Powtarzalność własna $\rho_{\mathrm{rep}}$ (± SEM)")
    axL.set_title("A. Stały budżet w podziale na długość recenzji $n$")
    axL.set_ylim(0.0, 1.04)
    axL.yaxis.set_major_formatter(_comma_fmt(2))
    axL.grid(True, linestyle="--", alpha=0.35)
    axL.spines["top"].set_visible(False)
    axL.spines["right"].set_visible(False)
    axL.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

    # Panel B: Length-scaled B(n) = c * n by length tercile
    sub_sc = df_pilot[df_pilot["method_family"] == "SHAP_scaled"].copy()
    for bkey, blabel, color, marker in bin_meta:
        bsub = sub_sc[sub_sc["length_bin"] == bkey]
        if bsub.empty:
            continue
        n_min, n_max = int(bsub["n_tokens"].min()), int(bsub["n_tokens"].max())
        agg = bsub.groupby("param_value", as_index=False).agg(
            rep_mean=("rep_spearman_mean", "mean"),
            rep_sem=("rep_spearman_mean", "sem"),
        ).sort_values("param_value")
        x = agg["param_value"].to_numpy(dtype=float)
        y = agg["rep_mean"].to_numpy(dtype=float)
        y_sem = agg["rep_sem"].fillna(0.0).to_numpy(dtype=float)
        axR.plot(x, y, marker=marker, markersize=5.2, color=color, linewidth=2.0,
                 label=f"{blabel} (n={n_min}–{n_max} tok.)")
        axR.fill_between(x, np.maximum(0.0, y - y_sem), np.minimum(1.0, y + y_sem), color=color, alpha=0.15)

    axR.axhline(0.80, color="#666666", linestyle=":", linewidth=1.1, alpha=0.85,
                label=r"Próg $\rho_{\mathrm{rep}} = 0{,}80$")
    axR.set_xlabel(r"Mnożnik budżetu na token ($c = \mathrm{max\_evals} / n$)")
    axR.set_title(r"B. Budżet skalowany $B(n) = c \cdot n$")
    axR.grid(True, linestyle="--", alpha=0.35)
    axR.spines["top"].set_visible(False)
    axR.spines["right"].set_visible(False)
    axR.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

    n_rev = int(df_pilot["sample_id"].nunique())
    n_seeds = int(df_pilot["n_seeds"].iloc[0])
    fig.suptitle(
        f"Wpływ długości recenzji $n$ na powtarzalność Permutation SHAP ({model_label}, N={n_rev}, {n_seeds} ziaren)",
        fontsize=11.5, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    extra_dir = os.path.join(PROJECT_ROOT, "results", "hyperparameter_tuning", "shap", "plots")
    _save_fig(fig, "pilot_shap_length_scaling", extra_dir=extra_dir)


def plot_lime_pilot(df_pilot: pd.DataFrame, df_grid: pd.DataFrame = None, model_label: str = "Qwen3-1.7B-Base"):
    agg = _aggregate_5seed(df_pilot, "LIME")
    x = agg["param_value"].to_numpy(dtype=float)
    y_rep = agg["rep_mean"].to_numpy(dtype=float)
    y_rep_sem = agg["rep_sem"].fillna(0.0).to_numpy(dtype=float)
    y_time = agg["time_mean"].to_numpy(dtype=float)
    y_time_sem = agg["time_sem"].fillna(0.0).to_numpy(dtype=float)
    n_rev = int(agg["n_reviews"].iloc[0])
    n_seeds = int(agg["n_seeds"].iloc[0])

    chosen_n = select_budget_apriori(x, y_rep, threshold=0.95, delta_tol=0.03)

    fig, ax1 = plt.subplots(figsize=(7.8, 4.6), dpi=300)
    c_rep = "#2b5c8f"
    c_stab = "#1b9e77"
    c_time = "#d95f02"

    l1 = ax1.plot(x, y_rep, marker="o", markersize=5.5, color=c_rep, linewidth=2.2,
                  label=r"Powtarzalność własna $\rho_{\mathrm{rep}}$ (± SEM)")
    ax1.fill_between(x, np.maximum(0.0, y_rep - y_rep_sem), np.minimum(1.0, y_rep + y_rep_sem),
                     color=c_rep, alpha=0.18)

    l_thr = ax1.axhline(0.95, color="#666666", linestyle=":", linewidth=1.1, alpha=0.85,
                        label=r"Próg reguły a priori ($\rho_{\mathrm{rep}} \geq 0{,}95$)")

    lines = l1 + [l_thr]
    if df_grid is not None and "LIME" in df_grid["method_family"].values:
        g_lime = df_grid[df_grid["method_family"] == "LIME"].sort_values("param_value")
        gx = g_lime["param_value"].to_numpy(dtype=float)
        gy = g_lime["spearman_mean"].to_numpy(dtype=float)
        gy_sem = g_lime["spearman_sem"].to_numpy(dtype=float)
        l2 = ax1.plot(gx, gy, marker="s", markersize=4.8, color=c_stab, linewidth=1.6, linestyle="--", alpha=0.8,
                      label="Stabilność RQ1 — ilustracja sufitu (± SEM, nie kryterium)")
        ax1.fill_between(gx, gy - gy_sem, gy + gy_sem, color=c_stab, alpha=0.12)
        lines = lines + l2

    ax1.axvline(chosen_n, color="#555555", linestyle="-.", linewidth=1.3, alpha=0.85)
    ax1.annotate(
        f"Wybrana liczba masek\n(N_LIME = {int(chosen_n)})",
        xy=(chosen_n, 0.50),
        xytext=(chosen_n + 80, 0.48),
        fontsize=8.5,
        color="#333333",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )

    ax1.set_xlabel(r"Liczba losowanych masek sąsiedztwa ($N_{\mathrm{LIME}}$)")
    ax1.set_ylabel(r"Korelacja rang Spearmana $\rho$", color=c_rep)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlim(0, max(x) * 1.06)
    ax1.set_xticks(x)
    ax1.yaxis.set_major_formatter(_comma_fmt(2))
    ax1.tick_params(axis="y", labelcolor=c_rep)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)

    ax2 = ax1.twinx()
    l3 = ax2.plot(x, y_time, marker="^", markersize=5.5, color=c_time, linewidth=1.9, linestyle=":",
                  label="Średni czas na wyjaśnienie [s] (± SEM)")
    ax2.fill_between(x, np.maximum(0.0, y_time - y_time_sem), y_time + y_time_sem, color=c_time, alpha=0.16)
    ax2.set_ylabel("Średni czas na wyjaśnienie [s]", color=c_time)
    ax2.set_ylim(0.0, float(np.max(y_time + y_time_sem)) * 1.18)
    ax2.yaxis.set_major_formatter(_comma_fmt(1))
    ax2.tick_params(axis="y", labelcolor=c_time)
    ax2.spines["top"].set_visible(False)

    lines = lines + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95)

    ax1.set_title(f"Pilotażowy dobór liczby masek LIME ({model_label}, N={n_rev}, {n_seeds} ziaren)", pad=10)
    fig.tight_layout()
    extra_dir = os.path.join(PROJECT_ROOT, "results", "hyperparameter_tuning", "lime", "plots")
    _save_fig(fig, "pilot_lime_convergence", extra_dir=extra_dir)
    return chosen_n


def plot_ig_pilot(df_ig: pd.DataFrame, model_label: str = "Qwen3-1.7B-Base"):
    df = df_ig.sort_values("steps").copy()
    x = df["steps"].to_numpy(dtype=float)
    y_err = df["completeness_error_mean_pct"].to_numpy(dtype=float)
    y_err_sem = df["completeness_error_sem_pct"].to_numpy(dtype=float)
    y_time = df["time_per_sample_s"].to_numpy(dtype=float)
    y_time_sem = df["time_per_sample_sem_s"].to_numpy(dtype=float) if "time_per_sample_sem_s" in df.columns else np.zeros_like(y_time)
    n_rev = int(df["n_samples"].iloc[0]) if "n_samples" in df.columns else 20

    fig, ax1 = plt.subplots(figsize=(7.8, 4.6), dpi=300)
    c_err = "#2b5c8f"
    c_time = "#d95f02"

    l1 = ax1.plot(x, y_err, marker="o", markersize=5.5, color=c_err, linewidth=2.1,
                  label=r"Błąd aksjomatu zupełności $\delta_{\mathrm{comp}}$ (± SEM)")
    ax1.fill_between(x, np.maximum(0.0, y_err - y_err_sem), y_err + y_err_sem, color=c_err, alpha=0.16)

    err_top = max(8.0, float(np.max(y_err + y_err_sem)) * 1.18)
    ax1.set_ylim(0.0, err_top)
    ax1.set_xlim(0, 210)
    ax1.set_xticks([5, 10, 20, 50, 100, 200])
    ax1.set_xlabel("Liczba kroków całkowania ($m$)")
    ax1.set_ylabel(r"Błąd aksjomatu zupełności $\delta_{\mathrm{comp}}$ [%]", color=c_err)
    ax1.yaxis.set_major_formatter(_int_fmt())
    ax1.tick_params(axis="y", labelcolor=c_err)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)

    ax2 = ax1.twinx()
    l2 = ax2.plot(x, y_time, marker="^", markersize=5.5, color=c_time, linewidth=1.9, linestyle=":",
                  label="Średni czas na wyjaśnienie [s] (± SEM)")
    ax2.fill_between(x, np.maximum(0.0, y_time - y_time_sem), y_time + y_time_sem, color=c_time, alpha=0.16)
    ax2.set_ylim(0.0, float(np.max(y_time + y_time_sem)) * 1.18)
    ax2.set_ylabel("Średni czas na wyjaśnienie [s]", color=c_time)
    ax2.yaxis.set_major_formatter(_comma_fmt(1))
    ax2.tick_params(axis="y", labelcolor=c_time)
    ax2.spines["top"].set_visible(False)

    ax1.axvline(100, color="#555555", linestyle="-.", linewidth=1.3, alpha=0.85)
    ax1.annotate(
        "Wybrana liczba kroków\n(m = 100)",
        xy=(100, err_top * 0.32),
        xytext=(108, err_top * 0.25),
        fontsize=8.5,
        color="#333333",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.56, 0.99),
               frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95)

    ax1.set_title(f"Pilotażowy dobór liczby kroków Integrated Gradients ({model_label}, N={n_rev}, deterministyczna)", pad=10)
    fig.tight_layout()
    extra_dir = os.path.join(PROJECT_ROOT, "results", "hyperparameter_tuning", "ig", "plots")
    _save_fig(fig, "pilot_ig_convergence", extra_dir=extra_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate pilot hyperparameter convergence plots for thesis.")
    parser.add_argument("--pilot-csv", default=DEFAULT_PILOT_5SEED_CSV)
    parser.add_argument("--grid-csv", default=None, help="Optional RQ1 grid summary CSV for ceiling reference")
    parser.add_argument("--ig-csv", default=DEFAULT_IG_CSV)
    parser.add_argument("--model-label", default="Qwen3-1.7B-Base")
    parser.add_argument("--only-plots", action="store_true", help="Kept for CLI compatibility.")
    args = parser.parse_args()

    _configure_matplotlib()

    df_pilot = pd.read_csv(args.pilot_csv)
    df_grid = pd.read_csv(args.grid_csv) if (args.grid_csv and os.path.exists(args.grid_csv)) else None
    df_ig = pd.read_csv(args.ig_csv) if (args.ig_csv and os.path.exists(args.ig_csv)) else None

    chosen_b = plot_shap_pilot(df_pilot, df_grid, model_label=args.model_label)
    plot_shap_length_scaling(df_pilot, model_label=args.model_label)
    chosen_n = plot_lime_pilot(df_pilot, df_grid, model_label=args.model_label)
    if df_ig is not None:
        plot_ig_pilot(df_ig, model_label=args.model_label)

    print(f"[Summary] A-priori selected budgets: Permutation SHAP max_evals={int(chosen_b)}, LIME N={int(chosen_n)}, IG m=100")


if __name__ == "__main__":
    main()
