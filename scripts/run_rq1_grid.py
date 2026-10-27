#!/usr/bin/env python
"""
RQ1 grid runner: stability of ALL methods over their hyperparameter grids, in one resumable run.

Parameter-free methods (Occlusion, Gradient x Input) are computed once. Parametric methods are
evaluated at every grid value on the SAME samples and the SAME perturbations:
    - Integrated Gradients: Riemann steps m   (Sundararajan et al., 2017: 20-300 steps recommended)
    - LIME:                 neighbourhood size N (Ribeiro et al., 2016 / lime package default: 5000)
    - SHAP (Permutation):   evaluation budget B  (Lundberg & Lee, 2017; pilot: rho_rep still rising at B=2500)

For each explanation the actual number of forward/backward passes is measured (hardware-invariant
cost). For stochastic methods (LIME, SHAP) a noise-floor control (same text, different random seed)
is added, so that perturbation instability can be separated from Monte-Carlo variance.

Output (in --output-dir):
    rq1_grid_raw.csv      one row per (sample, configuration, perturbation) + control rows (checkpoint)
    rq1_grid_summary.csv  one row per configuration: stability, noise floor, cost

Usage (Colab, L4):
    python scripts/run_rq1_grid.py --samples 50 --seed 42 --output-dir /content/drive/MyDrive/llm_xai_results/rq1_grid
Re-running the same command resumes from the checkpoint (flushed after every sample).
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd

from models import SentimentCausalLM
from explainers import (
    OcclusionExplainer,
    GradientXInputExplainer,
    IntegratedGradientsExplainer,
    LIMEExplainer,
    SHAPExplainer,
)
from experiments.rq1_stability import RQ1StabilityExperiment, NOISE_FLOOR_CATEGORY


def _grid(text: str):
    return [int(x) for x in text.split(",") if x.strip()]


def _tag(explainer, family: str, label: str, param=None):
    explainer.config_family = family
    explainer.config_label = label
    explainer.config_param = param
    return explainer


def build_explainers(model, methods, ig_grid, lime_grid, shap_grid, seed):
    explainers = []
    if "occlusion" in methods:
        explainers.append(_tag(OcclusionExplainer(model), "Occlusion", "Occlusion"))
    if "grad_x_input" in methods:
        explainers.append(_tag(GradientXInputExplainer(model), "GradientXInput", "GradientXInput"))
    if "ig" in methods:
        for m in ig_grid:
            explainers.append(_tag(IntegratedGradientsExplainer(model, steps=m), "IntegratedGradients", f"IG_m{m}", m))
    if "lime" in methods:
        for n in lime_grid:
            explainers.append(_tag(LIMEExplainer(model, num_samples=n, seed=seed), "LIME", f"LIME_N{n}", n))
    if "shap" in methods:
        for b in shap_grid:
            explainers.append(_tag(SHAPExplainer(model, max_evals=b, seed=seed), "SHAP", f"SHAP_B{b}", b))
    return explainers


def summarize(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Per-configuration summary: stability (preserved pairs), noise floor, cost."""
    main = df_raw[(df_raw["perturbation_category"] != NOISE_FLOOR_CATEGORY) & (df_raw["is_preserved"])]
    ctrl = df_raw[df_raw["perturbation_category"] == NOISE_FLOOR_CATEGORY]

    summ = main.groupby(["method_family", "explainer_name"], dropna=False).agg(
        param_value=("param_value", "first"),
        n_pairs=("spearman", "size"),
        n_samples=("sample_id", "nunique"),
        spearman_mean=("spearman", "mean"),
        spearman_sem=("spearman", "sem"),
        kendall_mean=("kendall", "mean"),
        jaccard_k1_mean=("jaccard_k1", "mean"),
        jaccard_k3_mean=("jaccard_k3", "mean"),
        jaccard_k5_mean=("jaccard_k5", "mean"),
        time_s_mean=("orig_time_s", "mean"),
        forward_passes_mean=("orig_forward_passes", "mean"),
        backward_passes_mean=("orig_backward_passes", "mean"),
        vram_mb_max=("vram_mb", "max"),
    ).reset_index()

    if len(ctrl) > 0:
        floor = ctrl.groupby("explainer_name").agg(
            noise_floor_spearman=("spearman", "mean"),
            noise_floor_jaccard_k3=("jaccard_k3", "mean"),
        ).reset_index()
        summ = summ.merge(floor, on="explainer_name", how="left")
    return summ.sort_values(["method_family", "param_value"]).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description="RQ1: stabilność wszystkich metod w siatce hiperparametrów.")
    p.add_argument("--model", default="microsoft/phi-2")
    p.add_argument("--samples", type=int, default=50, help="Liczba zdań SST-2 (wspólna dla wszystkich konfiguracji)")
    p.add_argument("--seed", type=int, default=42, help="Ziarno doboru próbek i metod losowych")
    p.add_argument("--methods", default="occlusion,grad_x_input,ig,lime,shap")
    p.add_argument("--ig-grid", default="10,20,50,100,200,300")
    p.add_argument("--lime-grid", default="50,150,500,1000,1500,2500,3000")
    p.add_argument("--shap-grid", default="100,250,500,1000,1500,2500,3000")
    p.add_argument("--no-noise-floor", action="store_true", help="Wyłącz kontrolę reseed dla LIME/SHAP")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--max-run-samples", type=int, default=None, help="Limit nowych próbek w tej sesji")
    p.add_argument("--output-dir", default="results/rq1_grid")
    args = p.parse_args()

    out_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "rq1_grid_raw.csv")
    summary_path = os.path.join(out_dir, "rq1_grid_summary.csv")
    if args.no_resume and os.path.exists(raw_path):
        os.remove(raw_path)

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    model = SentimentCausalLM(model_name=args.model)
    explainers = build_explainers(
        model, methods, _grid(args.ig_grid), _grid(args.lime_grid), _grid(args.shap_grid), args.seed
    )
    print("Konfiguracje:", [e.config_label for e in explainers])

    exp = RQ1StabilityExperiment(
        model=model,
        explainers=explainers,
        data_path=os.path.join(PROJECT_ROOT, "data", "sst2_validation.json"),
    )
    exp.run(
        num_samples=args.samples,
        seed=args.seed,
        filter_decision_preserved=True,
        checkpoint_path=raw_path,
        max_samples_this_run=args.max_run_samples,
        include_noise_floor=not args.no_noise_floor,
        checkpoint_interval=1,
    )

    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
        df_raw = pd.read_csv(raw_path)
        df_raw = df_raw.drop_duplicates(subset=["sample_id", "explainer_name", "perturbation_name"])
        summary = summarize(df_raw)
        summary.to_csv(summary_path, index=False)
        print("\n" + summary.round(3).to_string(index=False))
        print(f"\nZapisano: {raw_path}\n          {summary_path}")
    else:
        print(f"\n[Ostrzeżenie] Brak wyników do podsumowania w: {raw_path}")


if __name__ == "__main__":
    main()
