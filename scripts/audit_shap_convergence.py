#!/usr/bin/env python
"""
SHAP Convergence & Budget Optimization Study (Ablation Experiment).
Analyzes the impact of `max_evals` on:
1. Estimator variance / Repeatability across independent seeds.
2. Attribution stability (Spearman rho) under prompt perturbations.
3. Execution time per explanation on GPU.

Produces publication-grade Pareto convergence curves for Chapter 3/4.
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import SentimentCausalLM
from src.explainers.shap_explainer import SHAPExplainer
from src.perturbations import TextPerturber


def run_shap_convergence_study(
    model_name: str = "microsoft/phi-2",
    eval_grid: list = [50, 100, 250, 500, 1000],
    num_samples: int = 15,
    seed: int = 42,
    output_dir: str = "results/shap_convergence"
):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 70)
    print("  SHAP CONVERGENCE & EVALUATION BUDGET OPTIMIZATION STUDY")
    print(f"  Model: {model_name} | Samples: {num_samples} | Grid: {eval_grid}")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)

    # 1. Load model
    model = SentimentCausalLM(model_name=model_name)

    # 2. Load dataset
    data_path = os.path.join(PROJECT_ROOT, "data", "sst2_validation.json")
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(all_data), size=num_samples, replace=False)
    samples = [all_data[i] for i in sorted(indices)]

    records = []

    # Test perturbation: neutral prefix
    test_prefix = "Review: "

    for max_evals in eval_grid:
        print(f"\n---> Testing SHAP with max_evals = {max_evals}...")
        explainer = SHAPExplainer(model, max_evals=max_evals)

        times = []
        repeatability_rhos = []
        stability_rhos = []

        for sample in tqdm(samples, desc=f"evals={max_evals}"):
            text = sample["sentence"]
            pred_class, _ = model.predict_single(text)

            # Measure run 1
            t0 = time.time()
            res1 = explainer.explain(text, target_class=pred_class)
            t_elapsed = time.time() - t0
            times.append(t_elapsed)

            # Measure run 2 on identical text (Repeatability check)
            res2 = explainer.explain(text, target_class=pred_class)
            rho_rep, _ = spearmanr(res1.attributions, res2.attributions)
            if not np.isnan(rho_rep):
                repeatability_rhos.append(rho_rep)

            # Measure stability under punctuation perturbation (e.g. exclamation mark swap)
            pert_text = text.rstrip(".!?") + "!"
            _, pert_probs = model.predict_single(pert_text)
            if int(np.argmax(pert_probs)) == pred_class: # decision-preserved
                res_pert = explainer.explain(pert_text, target_class=pred_class)
                # Compare overlapping tokens
                min_len = min(len(res1.attributions), len(res_pert.attributions))
                if min_len >= 4:
                    rho_stab, _ = spearmanr(res1.attributions[:min_len], res_pert.attributions[:min_len])
                    if not np.isnan(rho_stab):
                        stability_rhos.append(rho_stab)

        rec = {
            "max_evals": max_evals,
            "time_per_sample_s": float(np.mean(times)),
            "repeatability_rho_mean": float(np.mean(repeatability_rhos)),
            "repeatability_rho_std": float(np.std(repeatability_rhos)),
            "stability_rho_mean": float(np.mean(stability_rhos)) if stability_rhos else 0.0,
            "stability_rho_std": float(np.std(stability_rhos)) if stability_rhos else 0.0,
        }
        records.append(rec)
        print(f"Results for max_evals={max_evals}: Time={rec['time_per_sample_s']:.2f}s | "
              f"Repeatability rho={rec['repeatability_rho_mean']:.3f} | "
              f"Stability rho={rec['stability_rho_mean']:.3f}")

    df_results = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "shap_convergence_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[Saved] CSV summary saved to: {csv_path}")

    # Generate Dual-Axis Publication Chart
    fig, ax1 = plt.subplots(figsize=(8.5, 5), dpi=300)

    x = df_results["max_evals"].values
    y_rep = df_results["repeatability_rho_mean"].values
    y_rep_err = df_results["repeatability_rho_std"].values / np.sqrt(num_samples)
    y_stab = df_results["stability_rho_mean"].values
    y_time = df_results["time_per_sample_s"].values

    color_rep = "#2b5c8f"
    color_stab = "#1b9e77"
    color_time = "#d95f02"

    ax1.set_xlabel("Liczba ewaluacji permutacji (max_evals)", fontsize=11)
    ax1.set_ylabel("Korelacja rang Spearmana $\\rho$", fontsize=11, color=color_rep)
    
    line1 = ax1.plot(x, y_rep, marker="o", color=color_rep, linewidth=2.2, label="Powtarzalność (Repeatability)")
    ax1.fill_between(x, y_rep - y_rep_err, y_rep + y_rep_err, color=color_rep, alpha=0.15)
    
    line2 = ax1.plot(x, y_stab, marker="s", color=color_stab, linewidth=2.2, linestyle="--", label="Stabilność na perturbacje (RQ1)")
    ax1.tick_params(axis="y", labelcolor=color_rep)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Second axis for execution time
    ax2 = ax1.twinx()
    ax2.set_ylabel("Czas obliczeń na zdanie [s]", fontsize=11, color=color_time)
    line3 = ax2.plot(x, y_time, marker="^", color=color_time, linewidth=2.0, linestyle=":", label="Czas wykonania [s]")
    ax2.tick_params(axis="y", labelcolor=color_time)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)

    plt.title("Zbieżność i wariancja estymatora SHAP w funkcji budżetu ewaluacji", fontsize=12, fontweight="bold")
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, "shap_convergence_curve.png")
    pdf_path = os.path.join(plots_dir, "shap_convergence_curve.pdf")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"[Saved] Publication charts saved to: {plot_path} and {pdf_path}")
    return df_results


def main():
    parser = argparse.ArgumentParser(description="Badanie zbieżności i budżetu ewaluacji SHAP.")
    parser.add_argument("--samples", type=int, default=15, help="Liczba zdań do testu (domyślnie: 15)")
    parser.add_argument("--model", type=str, default="microsoft/phi-2", help="Model LLM")
    parser.add_argument("--grid", type=str, default="50,100,250,500,1000", help="Siatka ewaluacji rozdzielona przecinkami")
    parser.add_argument("--output-dir", type=str, default="results/shap_convergence")
    args = parser.parse_args()

    eval_grid = [int(x.strip()) for x in args.grid.split(",") if x.strip()]
    run_shap_convergence_study(
        model_name=args.model,
        eval_grid=eval_grid,
        num_samples=args.samples,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
