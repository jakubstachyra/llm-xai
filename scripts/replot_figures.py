#!/usr/bin/env python
"""
Utility script to regenerate RQ1 plots directly from saved raw CSV results
without re-running LLM inferences or XAI calculations.

Usage:
    .venv\\Scripts\\python scripts/replot_figures.py
    .venv\\Scripts\\python scripts/replot_figures.py --input results/rq1_stability/rq1_stability_raw.csv --output-dir results/rq1_stability/plots
"""

import os
import sys
import argparse
import pandas as pd

# Add src/experiments to sys.path so we can import visualizations directly
# without triggering heavy dependencies (torch, captum, shap, spacy) in experiments/__init__.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "src", "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for faster script execution

from visualizations import (
    plot_rq1_metric_distribution,
    plot_rq1_jaccard_k_curves,
)

def main():
    parser = argparse.ArgumentParser(description="Regenerate RQ1 plots from raw CSV without model inference.")
    parser.add_argument(
        "--input", 
        type=str, 
        default="results/rq1_stability/rq1_stability_raw.csv",
        help="Path to raw CSV results"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results/rq1_stability/plots",
        help="Directory to save generated plots"
    )

    args = parser.parse_args()

    input_path = args.input if os.path.isabs(args.input) else os.path.join(PROJECT_ROOT, args.input)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(PROJECT_ROOT, args.output_dir)

    if not os.path.exists(input_path):
        print(f"BŁĄD: Plik danych {input_path} nie istnieje!")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"[Wczytywanie] {input_path}...")
    df_raw = pd.read_csv(input_path)
    print(f"Wczytano {len(df_raw)} wierszy.")

    print("\n[Generowanie] 1/3: Wykres Jaccard K-curves (poprawiony: clip [0, 1], jitter, odróżnialne linie)...")
    jaccard_png = os.path.join(output_dir, "rq1_jaccard_curves.png")
    plot_rq1_jaccard_k_curves(df_raw, save_path=jaccard_png)
    print(f"  -> Zapisano: {jaccard_png}")

    print("\n[Generowanie] 2/3: Rozkład korelacji Spearmana...")
    spearman_png = os.path.join(output_dir, "rq1_spearman_distribution.png")
    plot_rq1_metric_distribution(
        df_raw,
        metric="spearman",
        metric_label=r"Korelacja rang Spearmana ($\rho$)",
        save_path=spearman_png
    )
    print(f"  -> Zapisano: {spearman_png}")

    print("\n[Generowanie] 3/3: Rozkład współczynnika Kendalla...")
    kendall_png = os.path.join(output_dir, "rq1_kendall_distribution.png")
    plot_rq1_metric_distribution(
        df_raw,
        metric="kendall",
        metric_label=r"Współczynnik $\tau$ Kendalla",
        save_path=kendall_png
    )
    print(f"  -> Zapisano: {kendall_png}")

    print("\nGotowe! Wszystkie wykresy zostały pomyślnie zaktualizowane bez ponownego uruchamiania modelu.")

if __name__ == "__main__":
    main()
