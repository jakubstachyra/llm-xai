#!/usr/bin/env python
"""
Unified Hyperparameter Ablation & Convergence Study for LLM Interpretability.
Grounded in foundational XAI literature:
1. Integrated Gradients (IG):
   - Sundararajan, M., Taly, A., & Yan, Q. (2017). "Axiomatic Attribution for Deep Networks", ICML 2017.
   - Criterion: Axiom of Completeness deficit error and Riemann discretization convergence.
2. LIME:
   - Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "'Why Should I Trust You?': Explaining the Predictions of Any Classifier", ACM SIGKDD 2016.
   - Visani, G. et al. (2022). "Statistical stability indices for LIME", JORS.
   - Bansal, H. et al. (2020). "SAM: Sampling-based Approximations for Model Interpretability", arXiv.
   - Criterion: Inter-run Monte Carlo repeatability (rho), surrogate R^2 fidelity, and forward pass budget.
3. SHAP (PermutationExplainer):
   - Lundberg, S. M., & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017.
   - Merrick, L., & Taly, A. (2020). "The Explanation Game: Explaining Machine Learning Models Using Shapley Values".
   - Criterion: Coalition permutation sampling variance (rho) and budget Pareto optimality.
4. Parameter-Free Baselines:
   - Occlusion (Leave-One-Out) and Gradient x Input have zero hyperparameters and deterministic guarantees.
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
from src.explainers.integrated_gradients import IntegratedGradientsExplainer
from src.explainers.lime_explainer import LIMEExplainer

try:
    from src.explainers.shap_explainer import SHAPExplainer
except ImportError:
    SHAPExplainer = None


def load_dataset_samples(num_samples: int = 15, seed: int = 42):
    data_path = os.path.join(PROJECT_ROOT, "data", "sst2_validation.json")
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(all_data), size=num_samples, replace=False)
    return [all_data[i] for i in sorted(indices)]


# ==============================================================================
# 1. INTEGRATED GRADIENTS ABLATION (Sundararajan et al., 2017)
# ==============================================================================
def audit_integrated_gradients(model, samples, steps_grid=[5, 10, 20, 50, 100], output_dir="results/hyperparameter_tuning"):
    ig_dir = os.path.join(output_dir, "ig")
    plots_dir = os.path.join(ig_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  AUDIT 1: INTEGRATED GRADIENTS CONVERGENCE & COMPLETENESS AXIOM")
    print(f"  Reference: Sundararajan et al. (ICML 2017) 'Axiomatic Attribution for Deep Networks'")
    print(f"  Riemann Steps Grid: {steps_grid} | Samples: {len(samples)}")
    print("=" * 70)

    # First compute reference high-budget explanations (m_max) for rank convergence
    max_steps = max(steps_grid)
    print(f"Computing gold-standard reference with steps = {max_steps}...")
    ref_explainer = IntegratedGradientsExplainer(model, steps=max_steps)
    gold_attributions = {}
    for s_idx, sample in enumerate(samples):
        res = ref_explainer.explain(sample["sentence"])
        gold_attributions[s_idx] = res.attributions

    records = []
    for steps in steps_grid:
        print(f"\n---> Evaluating Integrated Gradients with steps = {steps}...")
        explainer = IntegratedGradientsExplainer(model, steps=steps)

        times = []
        completeness_errors = []
        convergence_rhos = []
        stability_rhos = []

        for s_idx, sample in enumerate(tqdm(samples, desc=f"IG steps={steps}")):
            text = sample["sentence"]
            pred_class, _ = model.predict_single(text)

            t0 = time.time()
            res = explainer.explain(text, target_class=pred_class)
            t_elapsed = time.time() - t0
            times.append(t_elapsed)

            # Completeness deficit ratio
            if res.metadata and "completeness_error" in res.metadata:
                completeness_errors.append(res.metadata["completeness_error"])

            # Convergence to max_steps ranking
            gold_attr = gold_attributions[s_idx]
            min_l = min(len(res.attributions), len(gold_attr))
            if min_l >= 3:
                rho_conv, _ = spearmanr(res.attributions[:min_l], gold_attr[:min_l])
                if not np.isnan(rho_conv):
                    convergence_rhos.append(rho_conv)

            # Stability under slight punctuation tweak
            pert_text = text.rstrip(".!?") + "!"
            _, pert_probs = model.predict_single(pert_text)
            if int(np.argmax(pert_probs)) == pred_class:
                res_pert = explainer.explain(pert_text, target_class=pred_class)
                min_len = min(len(res.attributions), len(res_pert.attributions))
                if min_len >= 3:
                    rho_stab, _ = spearmanr(res.attributions[:min_len], res_pert.attributions[:min_len])
                    if not np.isnan(rho_stab):
                        stability_rhos.append(rho_stab)

        rec = {
            "steps": steps,
            "backward_passes": steps,
            "time_per_sample_s": float(np.mean(times)),
            "completeness_error_pct": float(np.mean(completeness_errors) * 100.0),
            "completeness_error_std": float(np.std(completeness_errors) * 100.0),
            "convergence_rho_mean": float(np.mean(convergence_rhos)),
            "convergence_rho_std": float(np.std(convergence_rhos)),
            "stability_rho_mean": float(np.mean(stability_rhos)) if stability_rhos else 0.0,
        }
        records.append(rec)
        print(f"Steps={steps:3d} | Time={rec['time_per_sample_s']:.3f}s | "
              f"Completeness Error={rec['completeness_error_pct']:.2f}% | "
              f"Convergence rho={rec['convergence_rho_mean']:.4f}")

    df_ig = pd.DataFrame(records)
    csv_path = os.path.join(ig_dir, "ig_convergence_results.csv")
    df_ig.to_csv(csv_path, index=False)

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=300)
    x = df_ig["steps"].values
    y_err = df_ig["completeness_error_pct"].values
    y_conv = df_ig["convergence_rho_mean"].values
    y_time = df_ig["time_per_sample_s"].values

    color_err = "#d95f02"
    color_conv = "#2b5c8f"
    color_time = "#7570b3"

    ax1.set_xlabel("Liczba kroków całkowania Riemanna ($m$)", fontsize=11)
    ax1.set_ylabel("Błąd aksjomatu zupełności $\\delta$ [%]", fontsize=11, color=color_err)
    line1 = ax1.plot(x, y_err, marker="o", color=color_err, linewidth=2.2, label="Błąd zupełności [%]")
    ax1.axhline(5.0, color="#e7298a", linestyle=":", alpha=0.7, label="Próg akceptowalności (5%)")
    ax1.tick_params(axis="y", labelcolor=color_err)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Zbieżność rangowa Spearmana $\\rho(m, m_{\\max})$", fontsize=11, color=color_conv)
    line2 = ax2.plot(x, y_conv, marker="s", color=color_conv, linewidth=2.2, linestyle="--", label="Zbieżność rang $\\rho$")
    ax2.tick_params(axis="y", labelcolor=color_conv)
    ax2.set_ylim(0.85, 1.01)

    lines = line1 + line2
    labels = [l.get_label() for l in lines] + ["Próg akceptowalności (5%)"]
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", frameon=True, facecolor="white")

    plt.title("Integrated Gradients: Błąd zupełności i zbieżność rang (Sundararajan et al., 2017)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "ig_completeness_convergence.png"), bbox_inches="tight")
    plt.savefig(os.path.join(plots_dir, "ig_completeness_convergence.pdf"), bbox_inches="tight")
    plt.close()

    print(f"[Done] IG convergence saved to: {csv_path}")
    return df_ig


# ==============================================================================
# 2. LIME ABLATION (Ribeiro et al., 2016; Visani et al., 2022)
# ==============================================================================
def audit_lime(model, samples, samples_grid=[25, 50, 100, 200, 400], output_dir="results/hyperparameter_tuning"):
    lime_dir = os.path.join(output_dir, "lime")
    plots_dir = os.path.join(lime_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  AUDIT 2: LIME MONTE CARLO SAMPLE CONVERGENCE & STABILITY")
    print("  References: Ribeiro et al. (KDD 2016) | Visani et al. (JORS 2022) | Bansal et al. (2020)")
    print(f"  Num Samples Grid: {samples_grid} | Input Sentences: {len(samples)}")
    print("=" * 70)

    # Compute reference with highest sample budget for ranking convergence
    max_num = max(samples_grid)
    print(f"Computing LIME reference baseline with num_samples = {max_num}...")
    ref_explainer = LIMEExplainer(model, num_samples=max_num, seed=42)
    gold_attributions = {}
    for s_idx, sample in enumerate(samples):
        res = ref_explainer.explain(sample["sentence"])
        gold_attributions[s_idx] = res.attributions

    records = []
    for num_s in samples_grid:
        print(f"\n---> Evaluating LIME with num_samples = {num_s}...")
        explainer = LIMEExplainer(model, num_samples=num_s, seed=42)

        times = []
        repeatability_rhos = []
        r2_scores = []
        convergence_rhos = []

        for s_idx, sample in enumerate(tqdm(samples, desc=f"LIME N={num_s}")):
            text = sample["sentence"]
            pred_class, _ = model.predict_single(text)

            t0 = time.time()
            res1 = explainer.explain(text, target_class=pred_class, seed=42)
            t_elapsed = time.time() - t0
            times.append(t_elapsed)

            if res1.metadata and "r2_score" in res1.metadata:
                r2_scores.append(res1.metadata["r2_score"])

            # Run 2 on identical instance with independent random seed (Repeatability check)
            res2 = explainer.explain(text, target_class=pred_class, seed=999)
            min_l = min(len(res1.attributions), len(res2.attributions))
            if min_l >= 3:
                rho_rep, _ = spearmanr(res1.attributions[:min_l], res2.attributions[:min_l])
                if not np.isnan(rho_rep):
                    repeatability_rhos.append(rho_rep)

            # Convergence to N_max
            gold_attr = gold_attributions[s_idx]
            if min(len(res1.attributions), len(gold_attr)) >= 3:
                rho_conv, _ = spearmanr(res1.attributions[:min_l], gold_attr[:min_l])
                if not np.isnan(rho_conv):
                    convergence_rhos.append(rho_conv)

        rec = {
            "num_samples": num_s,
            "forward_passes": num_s,
            "time_per_sample_s": float(np.mean(times)),
            "repeatability_rho_mean": float(np.mean(repeatability_rhos)),
            "repeatability_rho_std": float(np.std(repeatability_rhos)),
            "r2_score_mean": float(np.mean(r2_scores)) if r2_scores else 0.0,
            "convergence_rho_mean": float(np.mean(convergence_rhos)),
        }
        records.append(rec)
        print(f"N={num_s:3d} | Time={rec['time_per_sample_s']:.2f}s | "
              f"Repeatability rho={rec['repeatability_rho_mean']:.3f} | "
              f"Surrogate R^2={rec['r2_score_mean']:.3f} | "
              f"Convergence rho={rec['convergence_rho_mean']:.3f}")

    df_lime = pd.DataFrame(records)
    csv_path = os.path.join(lime_dir, "lime_convergence_results.csv")
    df_lime.to_csv(csv_path, index=False)

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=300)
    x = df_lime["num_samples"].values
    y_rep = df_lime["repeatability_rho_mean"].values
    y_rep_err = df_lime["repeatability_rho_std"].values / np.sqrt(len(samples))
    y_r2 = df_lime["r2_score_mean"].values
    y_time = df_lime["time_per_sample_s"].values

    color_rep = "#1b9e77"
    color_r2 = "#7570b3"
    color_time = "#d95f02"

    ax1.set_xlabel("Liczba próbek sąsiedztwa perturbacyjnego ($N$)", fontsize=11)
    ax1.set_ylabel("Wskaźnik powtarzalności $\\rho_{\\mathrm{rep}}$ (Spearman)", fontsize=11, color=color_rep)
    line1 = ax1.plot(x, y_rep, marker="o", color=color_rep, linewidth=2.2, label="Powtarzalność inter-run $\\rho$")
    ax1.fill_between(x, y_rep - y_rep_err, y_rep + y_rep_err, color=color_rep, alpha=0.15)
    line2 = ax1.plot(x, y_r2, marker="s", color=color_r2, linewidth=2.0, linestyle="--", label="Dopasowanie aproksymacji $R^2$")
    ax1.tick_params(axis="y", labelcolor=color_rep)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Czas obliczeń na próbkę [s]", fontsize=11, color=color_time)
    line3 = ax2.plot(x, y_time, marker="^", color=color_time, linewidth=2.0, linestyle=":", label="Czas wykonania [s]")
    ax2.tick_params(axis="y", labelcolor=color_time)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True, facecolor="white")

    plt.title("LIME: Wariancja próbkowania i jakość aproksymacji (Visani et al., 2022)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "lime_sampling_convergence.png"), bbox_inches="tight")
    plt.savefig(os.path.join(plots_dir, "lime_sampling_convergence.pdf"), bbox_inches="tight")
    plt.close()

    print(f"[Done] LIME convergence saved to: {csv_path}")
    return df_lime


# ==============================================================================
# 3. SHAP ABLATION (Lundberg & Lee, 2017)
# ==============================================================================
def audit_shap(model, samples, eval_grid=[50, 100, 250, 500, 1000], output_dir="results/hyperparameter_tuning"):
    from scripts.audit_shap_convergence import run_shap_convergence_study
    shap_dir = os.path.join(output_dir, "shap")
    return run_shap_convergence_study(
        model_name=model.model_name,
        eval_grid=eval_grid,
        num_samples=len(samples),
        output_dir=shap_dir
    )


# ==============================================================================
# UNIFIED LATEX SUMMARY TABLE (CHAPTER 3 METODOLOGIA)
# ==============================================================================
def generate_latex_parameter_table(output_dir="results/hyperparameter_tuning"):
    tex_path = os.path.join(output_dir, "hyperparameter_selection_table.tex")
    
    table_content = r"""\begin{table*}[t]
\centering
\small
\caption{Metodologiczne uzasadnienie doboru hiperparametrów metod atrybucji oraz kryteria zbieżności w oparciu o literaturę źródłową.}
\label{tab:hyperparameter_tuning_rationale}
\begin{tabular}{lllll}
\toprule
\textbf{Metoda XAI} & \textbf{Badany parametr} & \textbf{Przestrzeń siatki} & \textbf{Kryterium zbieżności / Aksjomat} & \textbf{Literatura źródłowa} \\
\midrule
\textbf{Integrated Gradients} & Liczba kroków ($m$) & $\{5, 10, 20, 50, 100\}$ & Błąd domknięcia zupełności $\delta < 5\%$ & Sundararajan et al. (ICML 2017) \\
\textbf{LIME} & Liczba próbek ($N$) & $\{25, 50, 100, 200, 400\}$ & Powtarzalność $\rho_{\text{rep}} \ge 0.90$, dopasowanie $R^2$ & Ribeiro (2016), Visani (2022) \\
\textbf{SHAP (Permutation)} & Budżet koalicji ($B$) & $\{50, 100, 250, 500, 1000\}$ & Stabilność permutacji $\rho \to 1.0$, front Pareto & Lundberg (2017), Merrick (2020) \\
\midrule
\textbf{Occlusion (LOO)} & \textit{Brak} & Deterministyczny ($L$) & Determinizm, brak losowości próbkowej & Zeiler \& Fergus (ECCV 2014) \\
\textbf{Gradient $\times$ Input} & \textit{Brak} & Deterministyczny ($1$) & Dokładny gradient pojedynczego przejścia & Simonyan et al. (ICLR 2014) \\
\bottomrule
\end{tabular}
\end{table*}
"""
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(table_content)
    print(f"\n[Saved] Academic LaTeX table generated: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Wielometodowy audyt i zbieżność hiperparametrów XAI.")
    parser.add_argument("--method", type=str, default="all", choices=["all", "ig", "lime", "shap"],
                        help="Metoda do audytu: 'ig', 'lime', 'shap' lub 'all'")
    parser.add_argument("--samples", type=int, default=15, help="Liczba zdań do testu")
    parser.add_argument("--model", type=str, default="microsoft/phi-2", help="Model LLM")
    parser.add_argument("--ig-steps", type=str, default="5,10,20,50,100", help="Siatka kroków IG")
    parser.add_argument("--lime-samples", type=str, default="25,50,100,200,400", help="Siatka próbek LIME")
    parser.add_argument("--shap-evals", type=str, default="50,100,250,500,1000", help="Siatka ewaluacji SHAP")
    parser.add_argument("--output-dir", type=str, default="results/hyperparameter_tuning")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generate_latex_parameter_table(args.output_dir)

    samples = load_dataset_samples(num_samples=args.samples)
    print(f"Loaded {len(samples)} validation reviews for hyperparameter audit.")

    model = None
    if args.method in ["all", "ig", "lime", "shap"]:
        print(f"Loading model {args.model}...")
        model = SentimentCausalLM(model_name=args.model)

    if args.method in ["all", "ig"]:
        ig_grid = [int(x.strip()) for x in args.ig_steps.split(",") if x.strip()]
        audit_integrated_gradients(model, samples, steps_grid=ig_grid, output_dir=args.output_dir)

    if args.method in ["all", "lime"]:
        lime_grid = [int(x.strip()) for x in args.lime_samples.split(",") if x.strip()]
        audit_lime(model, samples, samples_grid=lime_grid, output_dir=args.output_dir)

    if args.method in ["all", "shap"]:
        shap_grid = [int(x.strip()) for x in args.shap_evals.split(",") if x.strip()]
        audit_shap(model, samples, eval_grid=shap_grid, output_dir=args.output_dir)

    print("\n" + "=" * 70)
    print("  ALL HYPERPARAMETER ABLATION AUDITS COMPLETED SUCCESSFULLY!")
    print(f"  Artifacts saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
