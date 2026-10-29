#!/usr/bin/env python
"""
CLI Runner for Research Question 1 (RQ1 - Stability Evaluation).

Usage:
    # Uruchomienie w wirtualnym środowisku:
    .venv\\Scripts\\activate
    python scripts/run_rq1.py --samples 100 --methods occlusion,ig,shap --model microsoft/phi-2

    # Lub bezpośrednio przez interpreter środowiska:
    .venv\\Scripts\\python scripts/run_rq1.py --samples 100 --methods occlusion,ig,shap --model microsoft/phi-2
"""

import sys
import os
import argparse

# Check if running inside virtual environment (suppressed in Google Colab / Kaggle environments)
in_colab = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules
in_venv = (sys.prefix != sys.base_prefix) or ("VIRTUAL_ENV" in os.environ)
if not in_venv and not in_colab:
    print("\n" + "!" * 70)
    print("OSTRZEŻENIE: Uruchomiono skrypt w systemowym interpreterze Pythona:")
    print(f"  {sys.executable}")
    print("Wymagane pakiety (np. 'shap', 'torch' z CUDA) znajdują się w '.venv'!")
    print("Aby uniknąć błędów 'ModuleNotFoundError', uruchom:")
    print("  .venv\\Scripts\\activate")
    print("  python scripts/run_rq1.py ...")
    print("LUB bezpośrednio:")
    print("  .venv\\Scripts\\python scripts/run_rq1.py ...")
    print("!" * 70 + "\n")

# Ensure src/ is on python path relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models import SentimentCausalLM
from explainers import (
    OcclusionExplainer,
    SHAPExplainer,
    IntegratedGradientsExplainer,
    GradientXInputExplainer,
    LIMEExplainer,
)
from experiments.rq1_stability import RQ1StabilityExperiment


def main():
    parser = argparse.ArgumentParser(
        description="Eksperyment RQ1: Ocena stabilności wyjaśnień XAI przy perturbacjach promptu (Decision-Preserved Subset)."
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=100, 
        help="Liczba próbek ze zbioru SST-2 do ewaluacji (domyślnie: 100)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="microsoft/phi-2", 
        help="Nazwa lub ścieżka modelu HuggingFace (domyślnie: 'microsoft/phi-2')"
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="occlusion,grad_x_input,ig,lime,shap",
        help="Metody do ewaluacji rozdzielone przecinkiem: 'occlusion', 'grad_x_input', 'ig', 'lime', 'shap' (domyślnie: wszystkie 5)"
    )
    parser.add_argument(
        "--max-evals", 
        type=int, 
        default=2500, 
        help="Maksymalna liczba ewaluacji dla permutacyjnego SHAP (domyślnie: 2500)"
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=100,
        help="Liczba kroków integracji numerycznej dla Integrated Gradients (domyślnie: 100)"
    )
    parser.add_argument(
        "--lime-samples",
        type=int,
        default=1000,
        help="Liczba próbek sąsiedztwa dla LIME (domyślnie: 1000)"
    )
    parser.add_argument(
        "--no-filter-preserved",
        action="store_true",
        help="Wyłącza filtr par ze spójną predykcją (domyślnie: filtr WŁĄCZONY)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Ziarno losowości dla powtarzalności próbek (domyślnie: 42)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results/rq1_stability", 
        help="Katalog zapisu wyników i wykresów (domyślnie: 'results/rq1_stability')"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignoruje istniejący checkpoint i rozpoczyna obliczenia od zera (domyślnie: wznawianie włączone)."
    )
    parser.add_argument(
        "--max-run-samples",
        type=int,
        default=None,
        help="Maksymalna liczba nowych próbek do przeliczenia w bieżącej sesji (np. podział na partie 12-godzinne)."
    )
    
    parser.add_argument(
        "--no-noise-floor",
        action="store_true",
        help="Wyłącza kontrolę szumu własnego (reseed control) dla LIME/SHAP (domyślnie: WŁĄCZONA)"
    )

    args = parser.parse_args()

    selected_methods = [m.strip().lower() for m in args.methods.split(",")]

    # Resolve output-dir relative to project root if relative
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)

    filter_preserved = not args.no_filter_preserved

    print("=" * 70)
    print("LLM XAI EVALUATION FRAMEWORK - RESEARCH QUESTION 1 (STABILITY)")
    print(f"Model: {args.model}")
    print(f"Próbki SST-2: {args.samples} (seed={args.seed})")
    print(f"Filtr par ze spójną predykcją (Decision-Preserved): {'TAK' if filter_preserved else 'NIE'}")
    print(f"Wybrane metody: {', '.join(selected_methods)}")
    print(f"SHAP max_evals: {args.max_evals} | LIME samples: {args.lime_samples} | IG steps: {args.ig_steps}")
    print(f"Katalog wyników: {output_dir}")
    print(f"Auto-wznawianie (Checkpoint): {'WYŁĄCZONE' if args.no_resume else 'WŁĄCZONE'}")
    if args.max_run_samples:
        print(f"Limit próbek w tej sesji: {args.max_run_samples}")
    print("=" * 70)

    model_name = args.model.strip()
    if model_name in ["microsoft/phi", "phi", "phi-2", "phi2"]:
        model_name = "microsoft/phi-2"

    # 1. Initialize model
    model = SentimentCausalLM(model_name=model_name)

    # 2. Initialize explainers
    explainers = []
    if "occlusion" in selected_methods or "loo" in selected_methods:
        explainers.append(OcclusionExplainer(model))
    if any(m in selected_methods for m in ["grad_x_input", "gradientxinput", "grad", "saliency"]):
        explainers.append(GradientXInputExplainer(model))
    if any(m in selected_methods for m in ["ig", "integrated_gradients", "integratedgradients"]):
        if IntegratedGradientsExplainer is None:
            raise ImportError("Brak modułu IntegratedGradientsExplainer lub jego zależności!")
        explainers.append(IntegratedGradientsExplainer(model, steps=args.ig_steps))
    if "lime" in selected_methods:
        explainers.append(LIMEExplainer(model, num_samples=args.lime_samples, seed=args.seed))
    if "shap" in selected_methods:
        if SHAPExplainer is None:
            raise ImportError("Pakiet 'shap' nie jest zainstalowany w aktywnym środowisku! Uruchom skrypt w .venv.")
        explainers.append(SHAPExplainer(model, max_evals=args.max_evals))

    if not explainers:
        raise ValueError(f"Nie wybrano żadnej poprawnej metody z: {args.methods}")


    # 3. Create and execute experiment
    experiment = RQ1StabilityExperiment(
        model=model,
        explainers=explainers,
        max_evals_shap=args.max_evals,
        ig_steps=args.ig_steps,
        data_path=os.path.join(PROJECT_ROOT, "data", "sst2_validation.json")
    )

    checkpoint_file = None if args.no_resume else os.path.join(output_dir, "rq1_stability_raw.csv")

    print("\n[Start] Rozpoczęcie procedury eksperymentalnej...")
    experiment.run(
        num_samples=args.samples, 
        seed=args.seed, 
        filter_decision_preserved=filter_preserved,
        progress_bar=True,
        checkpoint_path=checkpoint_file,
        max_samples_this_run=args.max_run_samples,
        include_noise_floor=not args.no_noise_floor
    )

    # 4. Save artifacts, summary, LaTeX and plots
    print("\n[Zapis] Generowanie podsumowań, tabeli LaTeX oraz wykresów...")
    paths = experiment.save_results(output_dir=output_dir)

    print("\n" + "=" * 70)
    print("ZAKOŃCZONO POMYŚLNIE EKSPERYMENT DLA PYTANIA BADAWCZEGO 1!")
    print(f"Tabela LaTeX gotowa w: {paths['latex_table']}")
    print(f"Wykresy do pracy magisterskiej: {paths['plots_dir']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
