#!/usr/bin/env python
"""
CLI Runner for Prompt Sensitivity Control Experiment.

Evaluates:
1. Zero-shot classification performance (Accuracy, F1, ROC-AUC, Agreement vs. base prompt)
   across a literature baseline prompt (Zhao et al., ICML 2021) and 3 paraphrases.
2. Attribution rank stability (Spearman rho, Kendall tau, Top-k Jaccard) between
   paraphrases and the base prompt on the decision-preserved subset.

Usage:
    .venv\\Scripts\\python scripts/run_prompt_sensitivity.py --model microsoft/phi-2
    .venv\\Scripts\\python scripts/run_prompt_sensitivity.py --model Qwen/Qwen3-8B-Base --clf-samples 872 --attr-samples 50
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models import SentimentCausalLM
from experiments.prompt_sensitivity import PromptSensitivityExperiment


def main():
    parser = argparse.ArgumentParser(
        description="Eksperyment kontrolny wrażliwości na prompt (klasyfikacja + stabilność atrybucji)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",
        help="Nazwa modelu HuggingFace (domyślnie: 'microsoft/phi-2')",
    )
    parser.add_argument(
        "--clf-samples",
        type=int,
        default=872,
        help="Liczba próbek SST-2 do oceny accuracy/F1/ROC-AUC (domyślnie: 872)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Podzbiór SST-2 używany do doboru/kontroli promptu (domyślnie: 'train')",
    )
    parser.add_argument(
        "--attr-samples",
        type=int,
        default=50,
        help="Liczba próbek do oceny stabilności rang atrybucji między parafrazami (domyślnie: 50)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="grad_x_input,occlusion,ig",
        help="Metody XAI rozdzielone przecinkiem (domyślnie: 'grad_x_input,occlusion,ig')",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=100,
        help="Liczba kroków dla Integrated Gradients (domyślnie: 100)",
    )
    parser.add_argument(
        "--lime-samples",
        type=int,
        default=1000,
        help="Liczba próbek LIME (domyślnie: 1000)",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=2500,
        help="Budżet wywołań SHAP (domyślnie: 2500)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Wyłącza kalibrację kontekstową Zhao et al. (2021)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Ziarno losowości (domyślnie: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Katalog wynikowy (domyślnie: results/prompt_sensitivity_<model>)",
    )
    args = parser.parse_args()

    safe_model = args.model.split("/")[-1]
    output_dir = args.output_dir or os.path.join("results", f"prompt_sensitivity_{safe_model}")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    model = SentimentCausalLM(
        model_name=args.model,
        calibrate_logits=not args.no_calibrate,
    )
    exp = PromptSensitivityExperiment(model=model, split=args.split)
    df_clf, _, df_summary = exp.run(
        clf_samples=args.clf_samples,
        attr_samples=args.attr_samples,
        methods=methods,
        seed=args.seed,
        ig_steps=args.ig_steps,
        lime_samples=args.lime_samples,
        max_evals_shap=args.max_evals,
        calibrate=not args.no_calibrate,
        output_dir=output_dir,
    )

    print("\n=== Podsumowanie klasyfikacji zero-shot ===")
    print(df_clf.to_string(index=False))
    print("\n=== Podsumowanie stabilności atrybucji (vs. zhao2021_base) ===")
    print(df_summary.to_string(index=False))
    print(f"\nWyniki zapisano w: {output_dir}")


if __name__ == "__main__":
    main()
