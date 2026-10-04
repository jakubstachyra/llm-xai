import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List, Dict

# Set clean publication-ready style
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#ebebeb",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

def plot_rq1_metric_distribution(
    df_raw: pd.DataFrame,
    metric: str = "spearman",
    metric_label: str = "Korelacja rang Spearmana",
    save_path: Optional[str] = None
):
    """
    Plots a comparison of metric distributions across perturbation types for each XAI method.
    """
    if metric not in df_raw.columns:
        raise ValueError(f"Metric '{metric}' not found in dataframe columns: {df_raw.columns}")

    perturbations = df_raw["perturbation_name"].unique()
    methods = df_raw["explainer_name"].unique()

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    
    # Define color palette
    colors = {"SHAP": "#2b5c8f", "Occlusion": "#d95f02", "Integrated": "#1b9e77", "Default": "#7570b3"}


    x = np.arange(len(perturbations))
    num_methods = len(methods)
    bar_width = 0.8 / num_methods

    for i, method in enumerate(methods):
        method_means = []
        method_stds = []
        
        for p in perturbations:
            subset = df_raw[(df_raw["explainer_name"] == method) & (df_raw["perturbation_name"] == p)][metric]
            method_means.append(subset.mean() if len(subset) > 0 else 0.0)
            method_stds.append(subset.std() if len(subset) > 0 else 0.0)

        offset = (i - (num_methods - 1) / 2) * bar_width
        color = colors.get(method.split()[0], colors["Default"])
        
        ax.bar(
            x + offset, 
            method_means, 
            yerr=method_stds, 
            width=bar_width, 
            label=method,
            color=color, 
            capsize=4, 
            alpha=0.88,
            edgecolor="black",
            linewidth=0.5
        )

    ax.set_ylabel(metric_label, fontsize=12, fontweight="normal")
    ax.set_title(f"Ocena stabilności atrybucji XAI: {metric_label}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(perturbations, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        # Save pdf version if requested or alongside
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        plt.savefig(pdf_path, bbox_inches="tight")
        
    return fig, ax


def plot_rq1_jaccard_k_curves(
    df_raw: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plots Top-K Jaccard similarity across k in {1, 3, 5} for each method.
    Features bounds clipping ([0, 1]), distinct line styles and markers,
    and slight horizontal jitter to prevent overlapping series from occluding each other.
    """
    methods = list(df_raw["explainer_name"].unique())
    k_cols = [col for col in ["jaccard_k1", "jaccard_k3", "jaccard_k5"] if col in df_raw.columns]
    k_labels = [col.replace("jaccard_k", "K=") for col in k_cols]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Style configuration per explainer for visual clarity and accessibility
    style_specs = {
        "Occlusion": {"color": "#2b5c8f", "linestyle": "-", "marker": "o"},
        "IntegratedGradients": {"color": "#d95f02", "linestyle": "--", "marker": "s"},
        "SHAP": {"color": "#1b9e77", "linestyle": ":", "marker": "^"},
    }
    fallback_palette = ["#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    fallback_linestyles = ["-.", "-", "--", ":"]
    fallback_markers = ["D", "v", "p", "*"]

    # Horizontal jitter setup so curves with near-identical means/stds do not eclipse each other
    x_indices = np.arange(len(k_cols))
    n_methods = len(methods)
    jitter_width = 0.05
    offsets = [(i - (n_methods - 1) / 2) * jitter_width for i in range(n_methods)]

    for i, method in enumerate(methods):
        sub = df_raw[df_raw["explainer_name"] == method]
        means = np.array([sub[k_col].mean() for k_col in k_cols])
        stds = np.array([sub[k_col].std() for k_col in k_cols])

        # Clip bounds to [0.0, 1.0] since Jaccard index is mathematically bounded in [0, 1]
        lower_bound = np.clip(means - stds, 0.0, 1.0)
        upper_bound = np.clip(means + stds, 0.0, 1.0)

        # Style lookup
        style = style_specs.get(method, {
            "color": fallback_palette[i % len(fallback_palette)],
            "linestyle": fallback_linestyles[i % len(fallback_linestyles)],
            "marker": fallback_markers[i % len(fallback_markers)]
        })

        x_coords = x_indices + offsets[i]

        ax.plot(
            x_coords, 
            means, 
            marker=style["marker"], 
            linestyle=style["linestyle"],
            linewidth=2.2, 
            markersize=7.5, 
            label=method, 
            color=style["color"],
            alpha=0.95
        )
        ax.fill_between(
            x_coords, 
            lower_bound, 
            upper_bound, 
            alpha=0.15, 
            color=style["color"]
        )

    ax.set_xticks(x_indices)
    ax.set_xticklabels(k_labels, fontsize=11)
    ax.set_ylabel("Top-K Jaccard Similarity", fontsize=12)
    ax.set_xlabel("Rozmiar zbioru kluczowych tokenów (K)", fontsize=11)
    ax.set_title("Stabilność zbioru najważniejszych tokenów (Top-K Jaccard)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(-0.02, 1.03)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        plt.savefig(pdf_path, bbox_inches="tight")

    return fig, ax


def plot_rq1_token_attribution_comparison(
    res_orig,
    res_pert,
    pert_name: str,
    top_n: int = 10,
    save_path: Optional[str] = None
):
    """
    Visualizes side-by-side token importance for a single sample before and after perturbation.
    """
    # Clean token dictionary
    def get_token_scores(res):
        from metrics import _extract_token_map
        return _extract_token_map(res, filter_prompt_template=True)

    map_orig = get_token_scores(res_orig)
    map_pert = get_token_scores(res_pert)

    # Shared tokens sorted by original importance
    shared = [t for t in map_orig if t in map_pert]
    shared = sorted(shared, key=lambda t: abs(map_orig[t]), reverse=True)[:top_n]
    shared.reverse() # For bottom-to-top horizontal bar chart

    if not shared:
        return None, None

    y = np.arange(len(shared))
    height = 0.35

    fig, ax = plt.subplots(figsize=(9, max(4.5, len(shared) * 0.45)), dpi=300)

    vals_orig = [map_orig[t] for t in shared]
    vals_pert = [map_pert[t] for t in shared]

    ax.barh(y + height/2, vals_orig, height, label="Oryginalny prompt", color="#2b5c8f", alpha=0.85)
    ax.barh(y - height/2, vals_pert, height, label=f"Zaburzony ({pert_name})", color="#e7298a", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(shared, fontsize=10)
    ax.set_xlabel("Wartość atrybucji", fontsize=11)
    ax.set_title(f"Zmiana atrybucji tokenów po perturbacji ({res_orig.explainer_name})", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        plt.savefig(pdf_path, bbox_inches="tight")

    return fig, ax


def plot_rq2_comprehensiveness_curves(
    df_raw: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plots the Comprehensiveness retention curves C(k) across perturbation steps k=1..5.
    Compares all XAI methods against the Random Baseline with shaded standard error bands.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    
    colors = {
        "Occlusion": "#d95f02",
        "Gradient": "#7570b3",
        "Integrated": "#1b9e77",
        "LIME": "#e7298a",
        "SHAP": "#2b5c8f",
        "Random": "#777777"
    }
    
    methods = sorted(df_raw["explainer_name"].unique())
    k_cols = [c for c in df_raw.columns if c.startswith("c_prob_k")]
    k_vals = [int(c.split("_k")[-1]) for c in k_cols]

    for method in methods:
        sub = df_raw[df_raw["explainer_name"] == method]
        means = [sub[col].mean() for col in k_cols]
        sems = [sub[col].sem() for col in k_cols]
        
        color_key = next((k for k in colors if k in method), "Default")
        c = colors.get(color_key, "#333333")
        is_random = "random" in method.lower()
        
        linestyle = "--" if is_random else "-"
        marker = "s" if is_random else "o"
        linewidth = 1.8 if is_random else 2.2

        ax.plot(
            k_vals, 
            means, 
            label=method, 
            color=c, 
            linestyle=linestyle, 
            marker=marker, 
            linewidth=linewidth,
            alpha=0.9
        )
        ax.fill_between(
            k_vals, 
            np.array(means) - np.array(sems), 
            np.array(means) + np.array(sems), 
            color=c, 
            alpha=0.15
        )

    ax.set_xlabel("Liczba usuniętych najważniejszych słów ($k$)", fontsize=11)
    ax.set_ylabel("Comprehensiveness: Spadek prawdopodobieństwa $C(k)$", fontsize=11)
    ax.set_title("Wierność atrybucji (RQ2): Krzywe Comprehensiveness $C(k)$", fontsize=12, fontweight="bold")
    ax.set_xticks(k_vals)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        plt.savefig(pdf_path, bbox_inches="tight")

    return fig, ax


def plot_rq2_aopc_comparison(
    df_summary: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Bar chart comparing AOPC Comprehensiveness vs AOPC Sufficiency for each method.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    
    methods = df_summary["explainer_name"].tolist()
    x = np.arange(len(methods))
    width = 0.35

    aopc_c = df_summary["aopc_c_prob_mean"].values
    aopc_u = df_summary["aopc_u_prob_mean"].values

    ax.bar(x - width/2, aopc_c, width, label="Comprehensiveness ($AOPC_C$)", color="#2b5c8f", alpha=0.85)
    ax.bar(x + width/2, aopc_u, width, label="Sufficiency ($AOPC_U$)", color="#e7298a", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Wartość AOPC (Probability Drop)", fontsize=11)
    ax.set_title("Porównanie wierności wyjaśnień: $AOPC_C$ vs $AOPC_U$", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        plt.savefig(pdf_path, bbox_inches="tight")

    return fig, ax

