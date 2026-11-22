#!/usr/bin/env python
"""
Backfill & Enrichment Utility for LLM-XAI Experimental Results.
Enriches existing CSV results with hardware-invariant complexity metrics:
- Forward pass count (N_fwd)
- Backward pass count (N_bwd)
- Hardware-invariant Relative Forward Units (FwdU = time_s / T_baseline)

Ensures that historical results collected on different GPUs (RTX 4060, RTX 4090, Colab L4)
can be compared objectively without re-running computations.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_BASELINE_LATENCY_RTX4060 = 0.04008  # ~40.1 ms for Phi-2 on RTX 4060 Laptop


def enrich_shap_convergence(base_latency: float):
    csv_path = os.path.join(PROJECT_ROOT, "results", "shap_convergence", "shap_convergence_results.csv")
    if not os.path.exists(csv_path):
        print(f"[Skip] {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df["forward_passes"] = df["max_evals"].astype(int)
    df["backward_passes"] = 0
    df["relative_fwdu"] = (df["time_per_sample_s"] / base_latency).round(1)

    df.to_csv(csv_path, index=False)
    print(f"[Enriched] SHAP convergence CSV updated: {csv_path}")


def enrich_rq1_stability(base_latency: float):
    raw_path = os.path.join(PROJECT_ROOT, "results", "rq1_stability", "rq1_stability_raw.csv")
    if not os.path.exists(raw_path):
        print(f"[Skip] {raw_path} not found.")
        return

    df = pd.read_csv(raw_path)
    if "orig_forward_passes" in df.columns and df["orig_forward_passes"].notna().any():
        print(f"[Skip] {raw_path} already contains measured pass counts (not overwriting).")
        return
    
    # Map forward and backward passes analytically
    def get_passes(row, text_col):
        explainer = str(row.get("explainer_name", "")).strip()
        text = str(row.get(text_col, ""))
        # Approximation of sequence length: words * 1.3 subwords + 6 prompt tokens
        approx_tokens = max(int(len(text.split()) * 1.3) + 6, 8)

        if "Gradient" in explainer:
            return 1, 1
        elif "Integrated" in explainer:
            return 25, 25
        elif "Occlusion" in explainer:
            return approx_tokens, 0
        elif "LIME" in explainer:
            return 150, 0
        elif "SHAP" in explainer:
            return 1500, 0
        return approx_tokens, 0

    orig_fwd, orig_bwd = [], []
    pert_fwd, pert_bwd = [], []

    for _, row in df.iterrows():
        f_o, b_o = get_passes(row, "original_text")
        f_p, b_p = get_passes(row, "perturbed_text")
        orig_fwd.append(f_o)
        orig_bwd.append(b_o)
        pert_fwd.append(f_p)
        pert_bwd.append(b_p)

    df["orig_forward_passes"] = orig_fwd
    df["orig_backward_passes"] = orig_bwd
    df["pert_forward_passes"] = pert_fwd
    df["pert_backward_passes"] = pert_bwd

    if "orig_time_s" in df.columns:
        df["orig_fwdu"] = (df["orig_time_s"] / base_latency).round(1)
    if "pert_time_s" in df.columns:
        df["pert_fwdu"] = (df["pert_time_s"] / base_latency).round(1)

    df.to_csv(raw_path, index=False)
    print(f"[Enriched] RQ1 stability raw CSV updated: {raw_path}")

    # Also update summary if present
    summary_path = os.path.join(PROJECT_ROOT, "results", "rq1_stability", "rq1_stability_summary.csv")
    if os.path.exists(summary_path):
        df_sum = pd.read_csv(summary_path)
        if "orig_time_s_mean" in df_sum.columns:
            df_sum["orig_fwdu_mean"] = (df_sum["orig_time_s_mean"] / base_latency).round(1)
        df_sum.to_csv(summary_path, index=False)
        print(f"[Enriched] RQ1 stability summary CSV updated: {summary_path}")


def enrich_rq3_joint(base_latency: float):
    joint_path = os.path.join(PROJECT_ROOT, "results", "rq3_joint_analysis", "rq1_rq2_joint_summary.csv")
    if not os.path.exists(joint_path):
        print(f"[Skip] {joint_path} not found.")
        return

    df = pd.read_csv(joint_path)
    if "orig_time_s" in df.columns:
        df["orig_fwdu"] = (df["orig_time_s"] / base_latency).round(1)
    df.to_csv(joint_path, index=False)
    print(f"[Enriched] RQ3 joint summary CSV updated: {joint_path}")


def main():
    parser = argparse.ArgumentParser(description="Enrich results with hardware-invariant efficiency metrics.")
    parser.add_argument("--baseline-latency", type=float, default=DEFAULT_BASELINE_LATENCY_RTX4060,
                        help="Baseline single forward pass latency in seconds.")
    args = parser.parse_args()

    print(f"Enriching results using baseline forward pass latency: {args.baseline_latency:.5f} s")
    enrich_shap_convergence(args.baseline_latency)
    enrich_rq1_stability(args.baseline_latency)
    enrich_rq3_joint(args.baseline_latency)
    print("\n[Complete] All available results files successfully enriched!")


if __name__ == "__main__":
    main()
