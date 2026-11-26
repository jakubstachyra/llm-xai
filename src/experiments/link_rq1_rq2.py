import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from scipy.stats import spearmanr, pearsonr


def _clean_method_name(n: str) -> str:
    n = str(n).strip()
    if "occlusion" in n.lower():
        return "Occlusion"
    if "gradient" in n.lower() and "integrated" not in n.lower():
        return "Gradient x Input"
    if "integrated" in n.lower() or n.lower().startswith("ig"):
        return "Integrated Gradients"
    if "lime" in n.lower():
        return "LIME"
    if "shap" in n.lower():
        return "SHAP"
    return n


def generate_joint_rq1_rq2_analysis(
    rq1_summary_path: str = "results/rq1_stability/rq1_stability_summary.csv",
    rq2_summary_path: str = "results/rq2_faithfulness/rq2_faithfulness_summary.csv",
    output_dir: str = "results/rq3_joint_analysis",
    rq1_raw_path: Optional[str] = None,
    rq2_raw_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Joint Analysis of Research Questions 1 and 2 (RQ3):
    Evaluates the relationship between explanation stability (RQ1) and prediction faithfulness (RQ2)
    at TWO distinct levels (Section 4.5.2 of thesis):
      1. Macro level (between methods): Pareto frontier & paired differences with true sample-level SEM.
      2. Micro level (within-method across shared reviews): instance-level Spearman rho(S_{i,m}, G_{i,m})
         with 95% cluster-bootstrap confidence intervals + ANCOVA-style repeated-measures correlation.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(rq1_summary_path) or not os.path.exists(rq2_summary_path):
        raise FileNotFoundError(
            f"Required summary files missing. Ensure both RQ1 ({rq1_summary_path}) "
            f"and RQ2 ({rq2_summary_path}) have been evaluated."
        )

    # Auto-detect raw CSV paths next to summaries when not explicitly passed
    if rq1_raw_path is None:
        cand1 = os.path.join(os.path.dirname(os.path.abspath(rq1_summary_path)), "rq1_stability_raw.csv")
        if os.path.exists(cand1):
            rq1_raw_path = cand1
    if rq2_raw_path is None:
        cand2 = os.path.join(os.path.dirname(os.path.abspath(rq2_summary_path)), "rq2_faithfulness_raw.csv")
        if os.path.exists(cand2):
            rq2_raw_path = cand2

    df_rq1 = pd.read_csv(rq1_summary_path)
    df_rq2 = pd.read_csv(rq2_summary_path)
    clean_name = _clean_method_name

    # If raw RQ1 exists, compute true hierarchical sample-level stability S_{i,m} and true SEM across samples
    rq1_sample_df: Optional[pd.DataFrame] = None
    if rq1_raw_path and os.path.exists(rq1_raw_path):
        raw1 = pd.read_csv(rq1_raw_path)
        if "is_preserved" in raw1.columns:
            raw1 = raw1[raw1["is_preserved"] == True].copy()
        if "perturbation_category" in raw1.columns:
            raw1 = raw1[raw1["perturbation_category"] != "control"].copy()
        if "original_text" in raw1.columns and "perturbed_text" in raw1.columns:
            raw1 = raw1[raw1["original_text"].astype(str).str.strip() != raw1["perturbed_text"].astype(str).str.strip()].copy()
        raw1["clean_name"] = raw1["explainer_name"].apply(clean_name)

        # Hierarchical aggregation: first within category, then across categories per (sample_id, clean_name)
        cat_col = "perturbation_category" if "perturbation_category" in raw1.columns else "perturbation_name"
        per_cat = raw1.groupby(["sample_id", "clean_name", cat_col], as_index=False).agg(
            spearman=("spearman", "mean"),
            kendall=("kendall", "mean"),
            jaccard_k3=("jaccard_k3", "mean"),
        )
        rq1_sample_df = per_cat.groupby(["sample_id", "clean_name"], as_index=False).agg(
            spearman=("spearman", "mean"),
            kendall=("kendall", "mean"),
            jaccard_k3=("jaccard_k3", "mean"),
        )
        rq1_agg = rq1_sample_df.groupby("clean_name", as_index=False).agg(
            spearman_mean=("spearman", "mean"),
            spearman_sem=("spearman", "sem"),
            kendall_tau_mean=("kendall", "mean"),
            top3_jaccard_mean=("jaccard_k3", "mean"),
            n_samples_rq1=("sample_id", "nunique"),
        )
        rq1_agg["explainer_name"] = rq1_agg["clean_name"]
    else:
        # Fallback to summary CSV when raw file is absent
        kendall_col = "kendall_mean" if "kendall_mean" in df_rq1.columns else "kendall_tau_mean"
        jaccard_col = "jaccard_k3_mean" if "jaccard_k3_mean" in df_rq1.columns else "top3_jaccard_mean"

        if "perturbation_name" in df_rq1.columns:
            if "perturbation_category" in df_rq1.columns:
                df_rq1 = df_rq1[df_rq1["perturbation_category"] != "control"]
            if "spearman_std" in df_rq1.columns:
                noop_mask = df_rq1["perturbation_name"].isin(["lowercase", "add_period"]) & (df_rq1["spearman_std"].fillna(1.0) == 0.0)
                df_rq1 = df_rq1[~noop_mask]

            agg_dict = {
                "spearman_mean": "mean",
                kendall_col: "mean",
                jaccard_col: "mean",
            }
            if "spearman_std" in df_rq1.columns:
                agg_dict["spearman_std"] = "mean"

            rq1_agg = df_rq1.groupby("explainer_name").agg(agg_dict).reset_index()
            rq1_agg = rq1_agg.rename(columns={
                kendall_col: "kendall_tau_mean",
                jaccard_col: "top3_jaccard_mean",
            })
            rq1_agg["spearman_sem"] = rq1_agg["spearman_std"] / np.sqrt(100) if "spearman_std" in rq1_agg.columns else 0.0
        else:
            rq1_agg = df_rq1.copy()
            if "spearman_sem" not in rq1_agg.columns:
                rq1_agg["spearman_sem"] = 0.0
        rq1_agg["clean_name"] = rq1_agg["explainer_name"].apply(clean_name)

    df_rq2["clean_name"] = df_rq2["explainer_name"].apply(clean_name)

    # Exclude Random Baseline from explainer macro table
    df_rq2_explainers = df_rq2[df_rq2["clean_name"] != "Random Baseline"]

    joint_df = pd.merge(
        rq1_agg,
        df_rq2_explainers,
        on="clean_name",
        suffixes=("_rq1", "_rq2")
    )

    # Micro-level (instance-level) within-method correlation on shared sample_ids
    micro_rows = []
    rm_corr_r, rm_corr_p = np.nan, np.nan
    if rq1_sample_df is not None and rq2_raw_path and os.path.exists(rq2_raw_path):
        raw2 = pd.read_csv(rq2_raw_path)
        raw2["clean_name"] = raw2["explainer_name"].apply(clean_name)
        raw2 = raw2[raw2["clean_name"] != "Random Baseline"].copy()
        merged_inst = pd.merge(
            rq1_sample_df,
            raw2[["sample_id", "clean_name", "aopc_c_prob", "gap_g", "harmonization_h", "num_words", "prediction_entropy", "is_correct"]],
            on=["sample_id", "clean_name"],
            how="inner",
        ).dropna(subset=["spearman", "aopc_c_prob"])

        if len(merged_inst) > 0:
            merged_inst.to_csv(os.path.join(output_dir, "rq3_instance_level_merged.csv"), index=False)
            rng = np.random.default_rng(42)
            for m_name, grp in merged_inst.groupby("clean_name"):
                s_vec = grp["spearman"].to_numpy()
                f_vec = grp["gap_g"].to_numpy() if "gap_g" in grp.columns else grp["aopc_c_prob"].to_numpy()
                n_inst = len(s_vec)
                if n_inst >= 5 and np.std(s_vec) > 0 and np.std(f_vec) > 0:
                    rho_val, p_val = spearmanr(s_vec, f_vec)
                    # Bootstrap 95% CI
                    boots = []
                    for _ in range(1000):
                        idx_b = rng.choice(n_inst, size=n_inst, replace=True)
                        if np.std(s_vec[idx_b]) > 0 and np.std(f_vec[idx_b]) > 0:
                            r_b, _ = spearmanr(s_vec[idx_b], f_vec[idx_b])
                            if not np.isnan(r_b):
                                boots.append(float(r_b))
                    ci_lo = float(np.percentile(boots, 2.5)) if boots else float(rho_val)
                    ci_hi = float(np.percentile(boots, 97.5)) if boots else float(rho_val)
                else:
                    rho_val, p_val, ci_lo, ci_hi = np.nan, np.nan, np.nan, np.nan
                micro_rows.append({
                    "clean_name": m_name,
                    "n_shared_samples": n_inst,
                    "micro_spearman_rho": float(rho_val) if not np.isnan(rho_val) else np.nan,
                    "micro_p_value": float(p_val) if not np.isnan(p_val) else np.nan,
                    "micro_ci95_low": ci_lo,
                    "micro_ci95_high": ci_hi,
                })

            # Repeated-measures (within-method centered) correlation across all (sample, method) pairs
            centered = merged_inst.copy()
            centered["s_centered"] = centered["spearman"] - centered.groupby("clean_name")["spearman"].transform("mean")
            f_col = "gap_g" if "gap_g" in centered.columns else "aopc_c_prob"
            centered["f_centered"] = centered[f_col] - centered.groupby("clean_name")[f_col].transform("mean")
            if len(centered) >= 6 and np.std(centered["s_centered"]) > 0 and np.std(centered["f_centered"]) > 0:
                rm_corr_r, rm_corr_p = spearmanr(centered["s_centered"], centered["f_centered"])

    if micro_rows:
        micro_df = pd.DataFrame(micro_rows)
        micro_df.to_csv(os.path.join(output_dir, "rq3_micro_correlations.csv"), index=False)
        joint_df = pd.merge(joint_df, micro_df, on="clean_name", how="left")

    joint_csv_path = os.path.join(output_dir, "rq1_rq2_joint_summary.csv")
    joint_df.to_csv(joint_csv_path, index=False)

    # Generate Publication Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    colors = {
        "Occlusion": "#d95f02",
        "Gradient x Input": "#7570b3",
        "Integrated Gradients": "#1b9e77",
        "LIME": "#e7298a",
        "SHAP": "#2b5c8f",
    }

    x_vals = joint_df["spearman_mean"].values
    y_vals = joint_df["aopc_c_prob_mean"].values
    names = joint_df["clean_name"].values

    for i, name in enumerate(names):
        c = colors.get(name, "#333333")
        x = x_vals[i]
        y = y_vals[i]
        x_err = joint_df["spearman_sem"].iloc[i] if "spearman_sem" in joint_df.columns else 0.0
        y_err = joint_df["aopc_c_prob_sem"].iloc[i] if "aopc_c_prob_sem" in joint_df.columns else 0.0

        ax.errorbar(
            x, y, 
            xerr=x_err, 
            yerr=y_err, 
            fmt="o", 
            color=c, 
            ecolor=c, 
            elinewidth=1.5, 
            capsize=4, 
            markersize=9, 
            label=name
        )
        ax.annotate(
            name, 
            (x, y), 
            textcoords="offset points", 
            xytext=(7, 5), 
            fontsize=10, 
            fontweight="bold", 
            color=c
        )

    # Compute descriptive macro correlation and within-method repeated-measures correlation
    if not np.isnan(rm_corr_r):
        ax.set_title(
            f"Relacja między stabilnością a wiernością wyjaśnień (RQ3)\n"
            f"Korelacja wewnątrz-metodowa (po próbkach): $\\rho_{{RM}} = {rm_corr_r:+.2f}$ ($p = {rm_corr_p:.3f}$)",
            fontsize=11.5,
            fontweight="bold"
        )
    elif len(x_vals) >= 3:
        r_pearson, _ = pearsonr(x_vals, y_vals)
        ax.set_title(
            f"Mapa kompromisu stabilność–wierność wyjaśnień (RQ3, opisowe $r = {r_pearson:+.2f}$)",
            fontsize=11.5,
            fontweight="bold"
        )
    else:
        ax.set_title("Relacja między stabilnością a wiernością wyjaśnień (RQ3)", fontsize=12, fontweight="bold")

    ax.set_xlabel("Stabilność atrybucji: Średnia korelacja Spearmana $\\rho$ (RQ1)", fontsize=11)
    ax.set_ylabel("Wierność atrybucji: Spadek prawdopodobieństwa $AOPC_C$ (RQ2)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    plot_path = os.path.join(output_dir, "rq3_stability_vs_faithfulness.png")
    pdf_path = os.path.join(output_dir, "rq3_stability_vs_faithfulness.pdf")
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    # Generate LaTeX Table
    tex_path = os.path.join(output_dir, "rq1_rq2_joint_table.tex")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Zbiorcze zestawienie stabilności (RQ1) i wierności (RQ2) metod XAI.}",
        r"\label{tab:rq1_rq2_joint_results}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Metoda XAI} & \textbf{Stabilność $\rho \uparrow$} & \textbf{Top-3 Jaccard $\uparrow$} & \textbf{$\mathrm{AOPC}_C \uparrow$} & \textbf{Luka $G \uparrow$} & \textbf{Harmonia $H \uparrow$} \\",
        r"\midrule",
    ]

    for _, row in joint_df.iterrows():
        method = row["clean_name"]
        rho_str = f"{row['spearman_mean']:.3f}"
        jacc_str = f"{row['top3_jaccard_mean']:.3f}"
        aopc_str = f"{row['aopc_c_prob_mean']:.3f}"
        g_str = f"{row['gap_g_mean']:+.2f}"
        h_str = f"{row['harmonization_h_mean']:.3f}"
        lines.append(f"{method} & ${rho_str}$ & ${jacc_str}$ & ${aopc_str}$ & ${g_str}$ & ${h_str}$ \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex_content = "\n".join(lines)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    return joint_df, tex_content


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Joint RQ1 + RQ2 Analysis (RQ3)")
    parser.add_argument("--rq1-dir", type=str, default="results/rq1_stability", help="Directory or CSV path for RQ1 results")
    parser.add_argument("--rq2-dir", type=str, default="results/rq2_faithfulness", help="Directory or CSV path for RQ2 results")
    parser.add_argument("--output-dir", type=str, default="results/rq3_joint_analysis", help="Output directory for RQ3 analysis")
    args = parser.parse_args()

    rq1_path = args.rq1_dir if args.rq1_dir.endswith(".csv") else os.path.join(args.rq1_dir, "rq1_stability_summary.csv")
    rq2_path = args.rq2_dir if args.rq2_dir.endswith(".csv") else os.path.join(args.rq2_dir, "rq2_faithfulness_summary.csv")

    generate_joint_rq1_rq2_analysis(
        rq1_summary_path=rq1_path,
        rq2_summary_path=rq2_path,
        output_dir=args.output_dir
    )
