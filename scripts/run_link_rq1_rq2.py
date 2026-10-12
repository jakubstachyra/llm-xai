#!/usr/bin/env python
"""
CLI Runner for Research Question 3 (RQ3 - Joint Stability & Faithfulness Analysis).

Usage:
    python scripts/run_link_rq1_rq2.py
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.link_rq1_rq2 import generate_joint_rq1_rq2_analysis


def main():
    print("=" * 60)
    print("  Uruchamianie analizy łączonej RQ1–RQ2 (RQ3)")
    print("  Sprawdzanie korelacji między stabilnością a wiernością...")
    print("=" * 60)

    try:
        joint_df, tex_table = generate_joint_rq1_rq2_analysis()
        print("\n[Sukces] Analiza wygenerowana:")
        print(f"  - Tabela podsumowująca: results/rq3_joint_analysis/rq1_rq2_joint_summary.csv")
        print(f"  - Wykres korelacji: results/rq3_joint_analysis/rq3_stability_vs_faithfulness.pdf")
        print(f"  - Tabela LaTeX: results/rq3_joint_analysis/rq1_rq2_joint_table.tex\n")
        print(tex_table)
    except FileNotFoundError as e:
        print(f"\n[Info] {e}")
        print("Najpierw uruchom badania RQ1 i RQ2, a następnie wywołaj ten skrypt.")


if __name__ == "__main__":
    main()
