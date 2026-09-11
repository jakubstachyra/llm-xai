#!/usr/bin/env python
"""
CLI Runner for Phase 1: Perturbation Behavioral Validation (Sanity Check).
Verifies Flip Rate and Probability Drift (|Delta P|) on the SST-2 dataset.

Usage:
    python scripts/run_validate_perturbations.py --model microsoft/phi-2 --output-dir results/perturbation_validation
"""

import sys
import os
import argparse

# Ensure src/ is on python path relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models import SentimentCausalLM
from experiments.validate_perturbations import PerturbationValidator

def main():
    parser = argparse.ArgumentParser(
        description="Faza 1: Empiryczna weryfikacja behawioralna perturbacji promptu (Flip Rate & Delta P)."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="microsoft/phi-2", 
        help="Nazwa lub ścieżka modelu HuggingFace (domyślnie: 'microsoft/phi-2')"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Opcjonalny limit próbek do szybkiego testu (domyślnie: None = pełny zbiór 872 zdań)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results/perturbation_validation", 
        help="Katalog zapisu wyników i wykresów (domyślnie: 'results/perturbation_validation')"
    )
    
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)

    print("=" * 70)
    print("FAZA 1: WALIDACJA BEHAWIORALNA PERTURBACJI NA MODELU CAUSAL LM")
    print(f"Model: {args.model}")
    print(f"Próbki: {'Cały zbiór (872 zdania)' if args.limit is None else f'{args.limit} zdań'}")
    print(f"Katalog wyników: {output_dir}")
    print("=" * 70)

    # 1. Initialize model
    model = SentimentCausalLM(model_name=args.model)

    # 2. Run validator
    validator = PerturbationValidator(
        model=model,
        data_path=os.path.join(PROJECT_ROOT, "data", "sst2_validation.json")
    )
    
    print("\n[Start] Rozpoczęcie walidacji predykcji modelu przed i po perturbacjach...")
    validator.run(limit=args.limit, progress_bar=True)

    # 3. Save artifacts
    print("\n[Zapis] Zapisywanie tabeli LaTeX, statystyk CSV i wykresów...")
    paths = validator.save_results(output_dir=output_dir)

    print("\n" + "=" * 70)
    print("ZAKOŃCZONO POMYŚLNIE FAZĘ 1 (SANITY CHECK PERTURBACJI)!")
    print(f"Tabela do pracy magisterskiej: {paths['latex']}")
    print(f"Podsumowanie CSV: {paths['summary']}")
    print(f"Wykresy: {paths['plots']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
