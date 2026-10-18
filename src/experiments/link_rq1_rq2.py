import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from scipy.stats import spearmanr, pearsonr


def generate_joint_rq1_rq2_analysis(
    rq1_summary_path: str = "results/rq1_stability/rq1_stability_summary.csv",
    rq2_summary_path: str = "results/rq2_faithfulness/rq2_faithfulness_summary.csv",
    output_dir: str = "results/rq3_joint_analysis"
) -> Tuple[pd.DataFrame, str]:
    """
    Joint Analysis of Research Questions 1 and 2 (RQ3):
    Evaluates whether explanation stability correlates with prediction faithfulness.
    Produces joint summary CSV, LaTeX table for Chapter 4, and publication scatter plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(rq1_summary_path) or not os.path.exists(rq2_summary_path):
        raise FileNotFoundError(
            f"Required summary files missing. Ensure both RQ1 ({rq1_summary_path}) "
            f"and RQ2 ({rq2_summary_path}) have been evaluated."
        )

    df_rq1 = pd.read_csv(rq1_summary_path)
    df_rq2 = pd.read_csv(rq2_summary_path)

    # Standardize explainer names for clean joining
    def clean_name(n: str) -> str:
        n = str(n).strip()
        if "occlusion" in n.lower():
            return "Occlusion"
        if "gradient" in n.lower() and "integrated" not in n.lower():
            return "Gradient x Input"
        if "integrated" in n.lower():
            return "Integrated Gradients"
        if "lime" in n.lower():
            return "LIME"
        if "shap" in n.lower():
            return "SHAP"
        return n

    # Group RQ1 to overall mean per explainer
    kendall_col = "kendall_mean" if "kendall_mean" in df_rq1.columns else "kendall_tau_mean"
    jaccard_col = "jaccard_k3_mean" if "jaccard_k3_mean" in df_rq1.columns else "top3_jaccard_mean"

    if "perturbation_name" in df_rq1.columns:
        agg_dict = {
            "spearman_mean": "mean",
            kendall_col: "mean",
            jaccard_col: "mean",
        }
        if "spearman_std" in df_rq1.columns:
            agg_dict["spearman_std"] = "mean"

        rq1_agg = df_rq1.groupby("explainer_name").agg(agg_dict).reset_index()
        # Rename to standardized names
        rq1_agg = rq1_agg.rename(columns={
            kendall_col: "kendall_tau_mean",
            jaccard_col: "top3_jaccard_mean",
        })
        if "spearman_std" in rq1_agg.columns:
            rq1_agg["spearman_sem"] = rq1_agg["spearman_std"] / np.sqrt(16)
        else:
            rq1_agg["spearman_sem"] = 0.0
    else:
        rq1_agg = df_rq1.copy()
        if "spearman_sem" not in rq1_agg.columns:
            rq1_agg["spearman_sem"] = 0.0


    rq1_agg["clean_name"] = rq1_agg["explainer_name"].apply(clean_name)
    df_rq2["clean_name"] = df_rq2["explainer_name"].apply(clean_name)

    # Exclude Random Baseline from correlation calculation
    df_rq2_explainers = df_rq2[df_rq2["clean_name"] != "Random Baseline"]

    joint_df = pd.merge(
        rq1_agg,
        df_rq2_explainers,
        on="clean_name",
        suffixes=("_rq1", "_rq2")
    )

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

    # Compute correlation if at least 3 methods present
    if len(x_vals) >= 3:
        r_pearson, p_pearson = pearsonr(x_vals, y_vals)
        r_spearman, p_spearman = spearmanr(x_vals, y_vals)
        ax.set_title(
            f"Relacja między stabilnością a wiernością wyjaśnień (RQ3)\n"
            f"Korelacja Pearsona: r = {r_pearson:+.2f} (p = {p_pearson:.3f})",
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
    generate_joint_rq1_rq2_analysis()
