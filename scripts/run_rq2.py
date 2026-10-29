#!/usr/bin/env python
"""
CLI Runner for Research Question 2 (RQ2 - Faithfulness Evaluation).

Usage:
    # Pilot run (5 samples, fast methods):
    python scripts/run_rq2.py --samples 5 --methods occlusion,grad_x_input,ig

    # Main production run (400 samples, all 5 methods, auto-checkpointing):
    python scripts/run_rq2.py --samples 400 --methods occlusion,grad_x_input,ig,lime,shap --model microsoft/phi-2

    # Operator embedding masking (100 samples):
    python scripts/run_rq2.py --samples 100 --operator embedding_mask --output-dir results/rq2_embedding_mask
"""

import sys
import os
import argparse

# Ensure src/ is on python path relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import SentimentCausalLM
from src.explainers import (
    OcclusionExplainer,
    GradientXInputExplainer,
    IntegratedGradientsExplainer,
    LIMEExplainer,
    SHAPExplainer,
)
from src.experiments.rq2_faithfulness import RQ2FaithfulnessExperiment


def main():
    parser = argparse.ArgumentParser(
        description="Eksperyment RQ2: Ocena wierności wyjaśnień (Comprehensiveness, Sufficiency, AOPC, luka G)."
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=400, 
        help="Liczba próbek walidacyjnych SST-2 (domyślnie: 400)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="microsoft/phi-2", 
        help="Nazwa lub identyfikator modelu Causal LM (domyślnie: 'microsoft/phi-2')"
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="occlusion,grad_x_input,ig,lime,shap",
        help="Rozdzielona przecinkami lista metod: occlusion, grad_x_input, ig, lime, shap"
    )
    parser.add_argument(
        "--operator",
        type=str,
        default="deletion",
        choices=["deletion", "embedding_mask"],
        help="Operator zaburzający: 'deletion' (usuwanie słów) lub 'embedding_mask' (maskowanie embeddingów)"
    )
    parser.add_argument(
        "--max-evals-shap",
        type=int,
        default=2500,
        help="Liczba ewaluacji Monte Carlo dla PermutationExplainer w SHAP (domyślnie: 2500)"
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=100,
        help="Liczba kroków interpolacji dla Integrated Gradients (domyślnie: 100)"
    )
    parser.add_argument(
        "--lime-samples",
        type=int,
        default=1000,
        help="Liczba próbek zaburzających dla LIME (domyślnie: 1000)"
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=20,
        help="Liczba losowych permutacji B dla linii bazowej (domyślnie: 20)"
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=5,
        help="Maksymalna liczba usuwanych słów Top-k (domyślnie: 5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/rq2_faithfulness",
        help="Katalog zapisu wyników (domyślnie: 'results/rq2_faithfulness')"
    )
    parser.add_argument(
        "--max-run-samples",
        type=int,
        default=None,
        help="Opcjonalny limit próbek na daną sesję (np. dla partii 12h)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignoruj istniejący checkpoint i zacznij eksperyment od nowa"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Ziarno losowości dla powtarzalności (domyślnie: 42)"
    )

    args = parser.parse_args()

    # Load model
    print(f"\n[CLI] Inicjalizacja modelu '{args.model}'...")
    model_wrapper = SentimentCausalLM(model_name=args.model)

    # Instantiate requested explainers
    selected_methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    explainers = []

    for m in selected_methods:
        if m in ["occlusion", "loo"]:
            explainers.append(OcclusionExplainer(model_wrapper))
        elif m in ["grad_x_input", "gradientxinput", "grad", "saliency"]:
            explainers.append(GradientXInputExplainer(model_wrapper))
        elif m in ["ig", "integrated_gradients", "integratedgradients"]:
            explainers.append(IntegratedGradientsExplainer(model_wrapper, steps=args.ig_steps))
        elif m in ["lime"]:
            explainers.append(LIMEExplainer(model_wrapper, num_samples=args.lime_samples, seed=args.seed))
        elif m in ["shap"]:
            explainers.append(SHAPExplainer(model_wrapper, max_evals=args.max_evals_shap))
        else:
            print(f"[CLI Ostrzeżenie] Nieznana metoda '{m}', pomijam.")

    if not explainers:
        print("[CLI Błąd] Nie wybrano żadnej poprawnej metody wyjaśniającej!")
        sys.exit(1)

    experiment = RQ2FaithfulnessExperiment(
        model=model_wrapper,
        explainers=explainers,
        operator=args.operator,
        max_k=args.max_k,
        random_trials=args.random_trials,
        output_dir=args.output_dir
    )

    experiment.run(
        num_samples=args.samples,
        seed=args.seed,
        resume=not args.no_resume,
        max_run_samples=args.max_run_samples
    )

    print("\n" + "=" * 60)
    print("  Eksperyment RQ2 zakończony sukcesem!")
    print(f"  Wyniki zapisano w: {args.output_dir}")
    print(f"  - Plik surowy: {os.path.join(args.output_dir, 'rq2_faithfulness_raw.csv')}")
    print(f"  - Podsumowanie: {os.path.join(args.output_dir, 'rq2_faithfulness_summary.csv')}")
    print(f"  - Tabela LaTeX: {os.path.join(args.output_dir, 'rq2_table.tex')}")
    print(f"  - Wykresy (PNG/PDF): {os.path.join(args.output_dir, 'plots')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
