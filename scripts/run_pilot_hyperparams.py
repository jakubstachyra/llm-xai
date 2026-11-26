#!/usr/bin/env python
"""
Pilot Hyperparameter Selection Study (Stage 0) on microsoft/phi-2.

Addresses all methodological requirements:
1. Selection criterion is strictly intra-method repeatability (rho_rep across 5 seeds)
   and computational cost (time / forward passes) — NEVER RQ1 perturbation stability.
2. Evaluates 5 independent seeds per review -> C(5, 2) = 10 pairwise Spearman correlations
   per review, reporting mean +/- SEM across N stratified reviews for ALL series
   (repeatability AND execution time).
3. Stratified sampling across 3 review token-length terciles (short, medium, long).
4. Evaluates both fixed Permutation SHAP budgets (max_evals) and length-scaled budgets
   (B(n) = c * n, where n is the number of review tokens), plus LIME (N_LIME).

Usage:
    .venv\\Scripts\\python.exe scripts/run_pilot_hyperparams.py --samples-per-bin 6
"""

import os
import sys
import json
import time
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import SentimentCausalLM
from src.explainers.shap_explainer import SHAPExplainer
from src.explainers.lime_explainer import LIMEExplainer
from src.explainers.integrated_gradients import IntegratedGradientsExplainer


def select_stratified_samples(model: SentimentCausalLM, samples_per_bin: int = 6, seed: int = 42):
    """Selects N = 3 * samples_per_bin reviews stratified by review token length terciles."""
    data_path = os.path.join(PROJECT_ROOT, "data", "sst2_validation.json")
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    records = []
    for idx, item in enumerate(all_data):
        text = item["sentence"]
        rs, re_, _ = model.find_review_span(text)
        n_tok = int(re_ - rs)
        if n_tok >= 4:
            records.append({"idx": idx, "sentence": text, "n_tokens": n_tok})

    df = pd.DataFrame(records)
    q1, q2 = np.quantile(df["n_tokens"], [1 / 3, 2 / 3])

    def _bin(n):
        if n <= q1:
            return "short"
        elif n <= q2:
            return "medium"
        return "long"

    df["length_bin"] = df["n_tokens"].apply(_bin)
    rng = np.random.default_rng(seed)

    selected = []
    for b in ("short", "medium", "long"):
        sub = df[df["length_bin"] == b]
        chosen_idx = rng.choice(len(sub), size=min(samples_per_bin, len(sub)), replace=False)
        for ci in sorted(chosen_idx):
            row = sub.iloc[ci]
            selected.append({
                "sample_id": int(row["idx"]),
                "sentence": row["sentence"],
                "n_tokens": int(row["n_tokens"]),
                "length_bin": b,
            })

    return selected, float(q1), float(q2)


def pairwise_spearman(vectors: list[np.ndarray]) -> float:
    """Computes mean pairwise Spearman correlation across all C(S, 2) seed runs."""
    rhos = []
    for v1, v2 in combinations(vectors, 2):
        rho, _ = spearmanr(v1, v2)
        if not np.isnan(rho):
            rhos.append(float(rho))
    return float(np.mean(rhos)) if rhos else np.nan


def run_pilot(
    model_name: str = "microsoft/phi-2",
    samples_per_bin: int = 6,
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    length_bins: tuple[str, ...] = ("short", "medium", "long"),
    shap_fixed_grid: tuple[int, ...] = (250, 500, 1000, 1500, 2500),
    shap_scaled_grid: tuple[int, ...] = (25, 50, 75, 100),
    lime_grid: tuple[int, ...] = (100, 250, 500, 1000, 1500, 2000),
    output_csv: str = os.path.join(PROJECT_ROOT, "results", "hyperparameter_tuning", "pilot_5seed_raw.csv"),
    run_methods: tuple[str, ...] = ("shap_fixed", "shap_scaled", "lime"),
):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    print(f"[Pilot 5-seed] Loading {model_name}...")
    model = SentimentCausalLM(model_name=model_name, calibrate_logits=True)

    samples, q1, q2 = select_stratified_samples(model, samples_per_bin=samples_per_bin, seed=42)
    samples = [s for s in samples if s["length_bin"] in length_bins]
    print(
        f"[Pilot 5-seed] Selected {len(samples)} reviews ({samples_per_bin}/bin, bins={length_bins}); "
        f"tercile cutoffs: q1={q1:.1f}, q2={q2:.1f} tokens"
    )

    # Precompute frozen predicted class once per review
    frozen_preds = {}
    for s in samples:
        pred_class, _ = model.predict_single(s["sentence"])
        frozen_preds[s["sample_id"]] = pred_class

    # Resume support
    done_keys = set()
    if os.path.exists(output_csv):
        try:
            existing = pd.read_csv(output_csv)
            for _, r in existing.iterrows():
                done_keys.add((int(r["sample_id"]), str(r["method_family"]), str(r["param_label"])))
            print(f"[Resume] Found {len(done_keys)} completed (sample, method, param) units.")
        except Exception:
            pass

    configs = []
    if "shap_fixed" in run_methods:
        for b in shap_fixed_grid:
            configs.append(("SHAP_fixed", f"B={b}", float(b), lambda m, b=b: SHAPExplainer(m, max_evals=b)))
    if "shap_scaled" in run_methods:
        for c in shap_scaled_grid:
            configs.append(("SHAP_scaled", f"c={c}", float(c), lambda m, c=c: SHAPExplainer(m, evals_per_token=c)))
    if "lime" in run_methods:
        for n_mask in lime_grid:
            configs.append(("LIME", f"N={n_mask}", float(n_mask), lambda m, n_mask=n_mask: LIMEExplainer(m, num_samples=n_mask)))

    total_units = len(samples) * len(configs)
    done_now = 0
    t_start_all = time.time()

    for s in samples:
        sid = s["sample_id"]
        text = s["sentence"]
        n_tok = s["n_tokens"]
        lbin = s["length_bin"]
        target_cls = frozen_preds[sid]

        for family, plabel, pval, builder in configs:
            if (sid, family, plabel) in done_keys:
                done_now += 1
                continue

            explainer = builder(model)
            rs, re_, _ = model.find_review_span(text)

            seed_attrs = []
            seed_times = []
            seed_fwds = []

            for sd in seeds:
                model.reset_counters()
                res = explainer.explain(text, target_class=target_cls, seed=sd)
                span_attr = np.asarray(res.attributions[rs:re_], dtype=np.float64)
                seed_attrs.append(span_attr)
                seed_times.append(float(res.execution_time))
                seed_fwds.append(int(model.forward_pass_count))

            rho_rep = pairwise_spearman(seed_attrs)
            row = {
                "sample_id": sid,
                "n_tokens": n_tok,
                "length_bin": lbin,
                "method_family": family,
                "param_label": plabel,
                "param_value": pval,
                "n_seeds": len(seeds),
                "n_pairs": len(seeds) * (len(seeds) - 1) // 2,
                "rep_spearman_mean": rho_rep,
                "time_s_mean": float(np.mean(seed_times)),
                "time_s_std": float(np.std(seed_times, ddof=1)) if len(seed_times) > 1 else 0.0,
                "forward_passes_mean": float(np.mean(seed_fwds)),
            }

            pd.DataFrame([row]).to_csv(
                output_csv,
                mode="a",
                header=not os.path.exists(output_csv),
                index=False,
            )
            done_keys.add((sid, family, plabel))
            done_now += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            elapsed = time.time() - t_start_all
            print(
                f"[{done_now}/{total_units}] sid={sid} (n={n_tok}, {lbin}) | {family} {plabel} | "
                f"rho_rep={rho_rep:.3f} | t={row['time_s_mean']:.2f}s | elapsed={elapsed/60:.1f}m"
            )

    print(f"[Done] Raw 5-seed pilot saved to {output_csv}")


def run_ig_pilot(
    model: SentimentCausalLM,
    samples: list[dict],
    frozen_preds: dict[int, int],
    ig_grid: tuple[int, ...] = (5, 10, 20, 50, 100, 200),
    output_ig_csv: str = os.path.join(PROJECT_ROOT, "results", "hyperparameter_tuning", "ig", "ig_convergence_stabilized.csv"),
):
    """Runs deterministic IG convergence audit on the same pilot sample set."""
    os.makedirs(os.path.dirname(output_ig_csv), exist_ok=True)
    steps_sorted = sorted(int(m) for m in ig_grid)
    max_steps = max(steps_sorted)

    print(f"[Pilot IG] Computing reference at m={max_steps} for {len(samples)} reviews...")
    ref_exp = IntegratedGradientsExplainer(model, steps=max_steps)
    gold_spans = {}
    for s in samples:
        sid = s["sample_id"]
        res = ref_exp.explain(s["sentence"], target_class=frozen_preds[sid])
        rs, re_, _ = model.find_review_span(s["sentence"], num_tokens=len(res.attributions))
        gold_spans[sid] = np.asarray(res.attributions[rs:re_], dtype=np.float64)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    records = []
    for m in steps_sorted:
        exp = IntegratedGradientsExplainer(model, steps=m)
        times, comp_errs, rhos = [], [], []
        for s in samples:
            sid = s["sample_id"]
            res = exp.explain(s["sentence"], target_class=frozen_preds[sid])
            times.append(float(res.execution_time))
            comp_errs.append(float(res.metadata["completeness_error"]) * 100.0)
            rs, re_, _ = model.find_review_span(s["sentence"], num_tokens=len(res.attributions))
            span_attr = np.asarray(res.attributions[rs:re_], dtype=np.float64)
            gold_attr = gold_spans[sid]
            min_len = min(len(span_attr), len(gold_attr))
            if m == max_steps:
                rhos.append(1.0)
            elif min_len >= 3:
                rho, _ = spearmanr(span_attr[:min_len], gold_attr[:min_len])
                if not np.isnan(rho):
                    rhos.append(float(rho))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        records.append({
            "steps": m,
            "backward_passes": m,
            "n_samples": len(samples),
            "time_per_sample_s": float(np.mean(times)),
            "time_per_sample_sem_s": float(np.std(times, ddof=1) / np.sqrt(len(times))) if len(times) > 1 else 0.0,
            "completeness_error_mean_pct": float(np.mean(comp_errs)),
            "completeness_error_sem_pct": float(np.std(comp_errs, ddof=1) / np.sqrt(len(comp_errs))) if len(comp_errs) > 1 else 0.0,
            "completeness_error_median_pct": float(np.median(comp_errs)),
            "convergence_rho_mean": float(np.mean(rhos)),
            "convergence_rho_sem": float(np.std(rhos, ddof=1) / np.sqrt(len(rhos))) if len(rhos) > 1 and m != max_steps else 0.0,
        })
        print(
            f"[Pilot IG] m={m:3d} | t={records[-1]['time_per_sample_s']:.3f}s | "
            f"comp_err={records[-1]['completeness_error_mean_pct']:.2f}% | rho={records[-1]['convergence_rho_mean']:.4f}"
        )

    df_ig = pd.DataFrame(records)
    df_ig.to_csv(output_ig_csv, index=False)
    print(f"[Done] IG pilot saved to {output_ig_csv}")


def _parse_int_tuple(val: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in val.split(",") if x.strip())


def _parse_str_tuple(val: str) -> tuple[str, ...]:
    return tuple(x.strip().lower() for x in val.split(",") if x.strip())


def main():
    parser = argparse.ArgumentParser(description="5-seed stratified pilot hyperparameter study.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--samples-per-bin", type=int, default=10, help="Reviews per length tercile (total N = 3 * samples-per-bin)")
    parser.add_argument("--bins", default="short,medium,long", help="Comma-separated length bins to evaluate (short,medium,long)")
    parser.add_argument("--seeds", default="42,43,44,45,46", help="Comma-separated random seeds")
    parser.add_argument("--methods", default="shap_fixed,shap_scaled,lime,ig")
    parser.add_argument("--shap-grid", default="250,500,1000,1500,2500", help="Fixed Permutation SHAP max_evals grid")
    parser.add_argument("--shap-scaled-grid", default="25,50,75,100", help="Length-scaled Permutation SHAP c=evals_per_token grid")
    parser.add_argument("--lime-grid", default="100,250,500,1000,1500,2000", help="LIME num_samples grid")
    parser.add_argument("--ig-grid", default="5,10,20,50,100,200", help="IG steps grid")
    parser.add_argument("--output-csv", default=os.path.join(PROJECT_ROOT, "results", "hyperparameter_tuning", "pilot_5seed_raw.csv"))
    parser.add_argument("--ig-csv", default=None, help="Optional output CSV path for IG convergence")
    args = parser.parse_args()

    methods = _parse_str_tuple(args.methods)
    length_bins = _parse_str_tuple(args.bins)
    seeds = _parse_int_tuple(args.seeds)
    shap_fixed_grid = _parse_int_tuple(args.shap_grid)
    shap_scaled_grid = _parse_int_tuple(args.shap_scaled_grid)
    lime_grid = _parse_int_tuple(args.lime_grid)
    ig_grid = _parse_int_tuple(args.ig_grid)

    stochastic_methods = tuple(m for m in methods if m != "ig")
    if stochastic_methods:
        run_pilot(
            model_name=args.model,
            samples_per_bin=args.samples_per_bin,
            seeds=seeds,
            length_bins=length_bins,
            shap_fixed_grid=shap_fixed_grid,
            shap_scaled_grid=shap_scaled_grid,
            lime_grid=lime_grid,
            output_csv=args.output_csv,
            run_methods=stochastic_methods,
        )
    if "ig" in methods:
        model = SentimentCausalLM(model_name=args.model, calibrate_logits=True)
        samples, _, _ = select_stratified_samples(model, samples_per_bin=args.samples_per_bin, seed=42)
        samples = [s for s in samples if s["length_bin"] in length_bins]
        frozen_preds = {s["sample_id"]: model.predict_single(s["sentence"])[0] for s in samples}
        ig_out = args.ig_csv or os.path.join(os.path.dirname(args.output_csv), f"ig_convergence_{args.model.split('/')[-1]}.csv")
        run_ig_pilot(model, samples, frozen_preds, ig_grid=ig_grid, output_ig_csv=ig_out)


if __name__ == "__main__":
    main()
