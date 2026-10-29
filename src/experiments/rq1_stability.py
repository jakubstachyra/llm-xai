import os
import json
import time
import inspect
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from models import SentimentCausalLM
from explainers import SHAPExplainer, OcclusionExplainer, BaseExplainer, IntegratedGradientsExplainer
from perturbations import TextPerturber
from hardware_tracker import HardwareSessionTracker
from metrics import (
    stability_spearman, 
    stability_kendall_tau, 
    stability_pearson,
    stability_topk_jaccard
)
from .visualizations import (
    plot_rq1_metric_distribution, 
    plot_rq1_jaccard_k_curves,
    plot_rq1_token_attribution_comparison
)


NOISE_FLOOR_CATEGORY = "control"
NOISE_FLOOR_NAME = "reseed_same_text"


def explainer_label(explainer: BaseExplainer) -> str:
    """Unique key of an explainer configuration (e.g. 'SHAP_B500'); falls back to the class name."""
    label = getattr(explainer, "config_label", None)
    return label if label else explainer.__class__.__name__.replace("Explainer", "")


class RQ1StabilityExperiment:
    """
    Research Question 1 (RQ1) Experiment Runner:
    Evaluates the stability of token attribution methods (e.g. Occlusion, IG, SHAP)
    under meaning-preserving prompt and text perturbations on SST-2.
    
    Includes:
    - Confounding factor filtering (Decision-Preserved Subset)
    - Resource profiling (Time, Model calls, GPU VRAM) for RQ4
    - Academic table and plot generation for Chapter 4
    """

    def __init__(
        self,
        model: Optional[SentimentCausalLM] = None,
        model_name: str = "microsoft/phi-2",
        explainers: Optional[List[BaseExplainer]] = None,
        max_evals_shap: int = 100,
        ig_steps: int = 20,
        data_path: str = "data/sst2_validation.json"
    ):
        if model is None:
            print(f"[RQ1] Inicjalizacja modelu '{model_name}'...")
            self.model = SentimentCausalLM(model_name=model_name)
        else:
            self.model = model

        if explainers is None:
            self.explainers = [
                OcclusionExplainer(self.model),
                IntegratedGradientsExplainer(self.model, steps=ig_steps),
                SHAPExplainer(self.model, max_evals=max_evals_shap)
            ]
        else:
            self.explainers = explainers

        self.data_path = data_path
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_summary: Optional[pd.DataFrame] = None
        self.last_explanations_cache: Dict[str, Any] = {}

    def load_dataset(self, num_samples: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
        """Loads SST-2 validation samples from local JSON or Hugging Face datasets."""
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            from datasets import load_dataset
            print(f"[RQ1] Pobieranie zbioru SST-2 z HuggingFace...")
            ds = load_dataset("stanfordnlp/sst2", split="validation")
            data = [{"sentence": item["sentence"], "label": item["label"]} for item in ds]

        # Deterministic prefix-preserving permutation (compatible with all N up to full dataset 872)
        np.random.seed(seed)
        indices = np.random.permutation(len(data))[:min(num_samples, len(data))]
        samples = [data[i] for i in indices]

        print(f"[RQ1] Załadowano {len(samples)} próbek do ewaluacji (ziarno: {seed}).")
        return samples

    def run(
        self, 
        num_samples: int = 100, 
        seed: int = 42,
        filter_decision_preserved: bool = True,
        progress_bar: bool = True,
        checkpoint_path: Optional[str] = None,
        max_samples_this_run: Optional[int] = None,
        include_noise_floor: bool = False,
        checkpoint_interval: int = 10
    ) -> pd.DataFrame:
        """
        Executes the stability evaluation benchmark with incremental checkpointing.
        
        Args:
            num_samples: Total number of SST-2 sentences to evaluate.
            seed: Random seed for sample selection.
            filter_decision_preserved: If True, only pairs where y_orig == y_pert are included in stability calculations.
            progress_bar: Display tqdm progress bar.
            checkpoint_path: Path to raw CSV file for auto-resume and per-sample flushing.
            max_samples_this_run: Optional limit on number of new samples to process in this session.
        """
        samples = self.load_dataset(num_samples=num_samples, seed=seed)
        records: List[Dict[str, Any]] = []
        completed_units: set = set()

        # Load existing checkpoint if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                df_existing = pd.read_csv(checkpoint_path)
                if len(df_existing) > 0 and "sample_id" in df_existing.columns and "explainer_name" in df_existing.columns:
                    records = df_existing.to_dict("records")
                    for r in records:
                        completed_units.add((int(r["sample_id"]), str(r["explainer_name"]), str(r.get("perturbation_name", ""))))
                    distinct_samples = set(r["sample_id"] for r in records)
                    print(f"[RQ1 Checkpoint] Wczytano {len(records)} rekordów z checkpointa ({len(distinct_samples)} częściowo lub w pełni gotowych próbek).")
            except Exception as e:
                print(f"[RQ1 Checkpoint] Ostrzeżenie przy wczytywaniu checkpointa ({e}). Start od zera.")
                records = []

        total_pairs = 0
        preserved_pairs = 0
        samples_processed_this_run = 0

        tracker = HardwareSessionTracker(self.model, experiment_name="RQ1-Stability", num_samples=num_samples)
        tracker.start()

        iterator = tqdm(enumerate(samples), total=len(samples), desc="Ewaluacja próbek RQ1") if progress_bar else enumerate(samples)

        unflushed_records: List[Dict[str, Any]] = []
        # checkpoint_interval: flush to disk every N processed samples (parameter)

        try:
            for sample_idx, item in iterator:
                if max_samples_this_run is not None and samples_processed_this_run >= max_samples_this_run:
                    print(f"\n[RQ1] Osiągnięto limit partii sesji ({max_samples_this_run} próbek). Zapisywanie stanu i pauza.")
                    break

                text = item["sentence"]
                true_label = item["label"]
                
                # Check if all explainers for this sample_idx are already completed
                explainer_names = [explainer_label(e) for e in self.explainers]
                cat_perts = TextPerturber.get_categorized_perturbations(text)
                all_pert_names = [p_name for perts in cat_perts.values() for p_name in perts.keys()]

                def _needs_control(exp) -> bool:
                    return include_noise_floor and getattr(exp, "stochastic", False)

                all_done = all(
                    (sample_idx, exp_name, p_name) in completed_units
                    for exp_name in explainer_names
                    for p_name in all_pert_names
                ) and all(
                    (sample_idx, explainer_label(e), NOISE_FLOOR_NAME) in completed_units
                    for e in self.explainers if _needs_control(e)
                )
                if all_done:
                    continue  # Skip already fully evaluated sample

                # Predict original sentence
                pred_class, probs = self.model.predict_single(text)
                p_target = float(probs[pred_class])

                # Pre-evaluate perturbations to verify decision preservation
                pert_status: Dict[str, Dict[str, Any]] = {}
                for category, perts in cat_perts.items():
                    for pert_name, pert_text in perts.items():
                        total_pairs += 1
                        pert_pred, pert_probs = self.model.predict_single(pert_text)
                        is_preserved = bool(pert_pred == pred_class)
                        if is_preserved:
                            preserved_pairs += 1

                        pert_status[pert_name] = {
                            "category": category,
                            "text": pert_text,
                            "pred_class": pert_pred,
                            "is_preserved": is_preserved,
                            "p_target": float(pert_probs[pred_class])
                        }

                sample_new_records: List[Dict[str, Any]] = []

                # Generate explanations and compute stability
                for explainer in self.explainers:
                    explainer_name = explainer_label(explainer)
                    method_family = getattr(explainer, "config_family", explainer.__class__.__name__.replace("Explainer", ""))
                    param_value = getattr(explainer, "config_param", None)
                    
                    # Filter perturbations that are already in checkpoint for this explainer
                    perts_to_run = [
                        (p_name, p_info) for p_name, p_info in pert_status.items()
                        if (sample_idx, explainer_name, p_name) not in completed_units
                    ]
                    run_control = _needs_control(explainer) and (sample_idx, explainer_name, NOISE_FLOOR_NAME) not in completed_units
                    if not perts_to_run and not run_control:
                        continue

                    print(f"  [Próbka {sample_idx+1}/{len(samples)}] Liczenie: {explainer_name} ({len(perts_to_run)} perturbacji)...", flush=True)

                    # Reset GPU memory stats before explanation
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    # Explain original prompt (measure hardware-invariant cost via model counters)
                    c0 = self.model.get_counters()
                    orig_res = explainer.explain(text, target_class=pred_class)
                    c1 = self.model.get_counters()
                    orig_fwd = c1["forward_passes"] - c0["forward_passes"]
                    orig_bwd = c1["backward_passes"] - c0["backward_passes"]
                    orig_time = orig_res.execution_time
                    orig_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0

                    if sample_idx == 0:
                        self.last_explanations_cache[f"{explainer_name}_orig"] = orig_res

                    for pert_name, p_info in perts_to_run:
                        pert_text = p_info["text"]
                        category = p_info["category"]
                        is_preserved = p_info["is_preserved"]

                        # Skip calculation if filtering enabled and decision flipped
                        if filter_decision_preserved and not is_preserved:
                            continue

                        # Reset GPU memory stats for perturbed run
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()

                        c0 = self.model.get_counters()
                        pert_res = explainer.explain(pert_text, target_class=pred_class)
                        c1 = self.model.get_counters()
                        pert_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0

                        if sample_idx == 0 and pert_name == "prefix_overall":
                            self.last_explanations_cache[f"{explainer_name}_pert"] = pert_res

                        # Calculate stability metrics
                        spearman = stability_spearman(orig_res, pert_res)
                        kendall = stability_kendall_tau(orig_res, pert_res)
                        pearson = stability_pearson(orig_res, pert_res)
                        jaccard_k1 = stability_topk_jaccard(orig_res, pert_res, k=1)
                        jaccard_k3 = stability_topk_jaccard(orig_res, pert_res, k=3)
                        jaccard_k5 = stability_topk_jaccard(orig_res, pert_res, k=5)

                        row = {
                            "sample_id": sample_idx,
                            "device_name": getattr(self.model, "device_name", "Unknown"),
                            "explainer_name": explainer_name,
                            "method_family": method_family,
                            "param_value": param_value,
                            "perturbation_category": category,
                            "perturbation_name": pert_name,
                            "is_preserved": is_preserved,
                            "original_text": text,
                            "perturbed_text": pert_text,
                            "true_label": true_label,
                            "orig_pred": pred_class,
                            "pert_pred": p_info["pred_class"],
                            "p_orig": p_target,
                            "p_pert": p_info["p_target"],
                            "orig_num_tokens": len(orig_res.tokens),
                            "spearman": spearman,
                            "kendall": kendall,
                            "pearson": pearson,
                            "jaccard_k1": jaccard_k1,
                            "jaccard_k3": jaccard_k3,
                            "jaccard_k5": jaccard_k5,
                            "orig_time_s": orig_time,
                            "pert_time_s": pert_res.execution_time,
                            "vram_mb": max(orig_vram_mb, pert_vram_mb),
                            "orig_forward_passes": orig_fwd,
                            "orig_backward_passes": orig_bwd,
                            "pert_forward_passes": c1["forward_passes"] - c0["forward_passes"],
                            "pert_backward_passes": c1["backward_passes"] - c0["backward_passes"]
                        }
                        records.append(row)
                        sample_new_records.append(row)
                        completed_units.add((sample_idx, explainer_name, pert_name))

                    # Noise-floor control: same text, different random seed. Separates intrinsic
                    # Monte-Carlo variance of LIME/SHAP from sensitivity to text perturbations
                    # (Visani et al. 2022; Alvarez-Melis & Jaakkola 2018).
                    if run_control:
                        ctrl_kwargs = {}
                        if "seed" in inspect.signature(explainer.explain).parameters:
                            ctrl_kwargs["seed"] = int(getattr(explainer, "seed", 42)) + 1000
                        c0 = self.model.get_counters()
                        ctrl_res = explainer.explain(text, target_class=pred_class, **ctrl_kwargs)
                        c1 = self.model.get_counters()
                        row = {
                            "sample_id": sample_idx,
                            "device_name": getattr(self.model, "device_name", "Unknown"),
                            "explainer_name": explainer_name,
                            "method_family": method_family,
                            "param_value": param_value,
                            "perturbation_category": NOISE_FLOOR_CATEGORY,
                            "perturbation_name": NOISE_FLOOR_NAME,
                            "is_preserved": True,
                            "original_text": text,
                            "perturbed_text": text,
                            "true_label": true_label,
                            "orig_pred": pred_class,
                            "pert_pred": pred_class,
                            "p_orig": p_target,
                            "p_pert": p_target,
                            "orig_num_tokens": len(orig_res.tokens),
                            "spearman": stability_spearman(orig_res, ctrl_res),
                            "kendall": stability_kendall_tau(orig_res, ctrl_res),
                            "pearson": stability_pearson(orig_res, ctrl_res),
                            "jaccard_k1": stability_topk_jaccard(orig_res, ctrl_res, k=1),
                            "jaccard_k3": stability_topk_jaccard(orig_res, ctrl_res, k=3),
                            "jaccard_k5": stability_topk_jaccard(orig_res, ctrl_res, k=5),
                            "orig_time_s": orig_time,
                            "pert_time_s": ctrl_res.execution_time,
                            "vram_mb": orig_vram_mb,
                            "orig_forward_passes": orig_fwd,
                            "orig_backward_passes": orig_bwd,
                            "pert_forward_passes": c1["forward_passes"] - c0["forward_passes"],
                            "pert_backward_passes": c1["backward_passes"] - c0["backward_passes"]
                        }
                        records.append(row)
                        sample_new_records.append(row)
                        completed_units.add((sample_idx, explainer_name, NOISE_FLOOR_NAME))

                if sample_new_records:
                    unflushed_records.extend(sample_new_records)
                    samples_processed_this_run += 1

                # Flush to checkpoint file every checkpoint_interval (10) samples
                if checkpoint_path and len(unflushed_records) > 0 and (samples_processed_this_run % checkpoint_interval == 0):
                    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
                    df_chunk = pd.DataFrame(unflushed_records)
                    write_header = not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0
                    df_chunk.to_csv(checkpoint_path, mode="a", header=write_header, index=False)
                    unflushed_records.clear()
                    # Also save live hardware snapshot to checkpoint directory
                    live_hw_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "hardware_live_snapshot.json")
                    try:
                        with open(live_hw_path, "w", encoding="utf-8") as f_hw:
                            json.dump(tracker.snapshot_stats(samples_processed_this_run), f_hw, indent=2)
                    except Exception:
                        pass

                # Free CUDA cache between samples to prevent memory fragmentation on 8GB GPUs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Flush any remaining buffered records at the end of loop
            if checkpoint_path and len(unflushed_records) > 0:
                os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
                df_chunk = pd.DataFrame(unflushed_records)
                write_header = not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0
                df_chunk.to_csv(checkpoint_path, mode="a", header=write_header, index=False)
                unflushed_records.clear()

        except KeyboardInterrupt:
            print("\n[RQ1] Przerwano działanie skryptu przez użytkownika (Ctrl+C).")
            if checkpoint_path and len(unflushed_records) > 0:
                os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
                df_chunk = pd.DataFrame(unflushed_records)
                write_header = not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0
                df_chunk.to_csv(checkpoint_path, mode="a", header=write_header, index=False)
                unflushed_records.clear()
            if checkpoint_path:
                print(f"[RQ1] Wszystkie dotychczasowe wyniki zostały bezpiecznie zapisane w checkpointcie: {checkpoint_path}")

        # Assemble and deduplicate full raw dataframe
        self.df_raw = pd.DataFrame(records)
        if len(self.df_raw) > 0:
            self.df_raw.drop_duplicates(subset=["sample_id", "explainer_name", "perturbation_name"], inplace=True)

        print(f"\n[RQ1] Zakończono sesję. Łącznie w buforze: {len(self.df_raw)} wierszy ({len(set(self.df_raw['sample_id'])) if len(self.df_raw) > 0 else 0} unikalnych próbek).")
        self._compute_summary()

        stats = tracker.stop(num_samples=samples_processed_this_run)
        extra_hw_paths = []
        if checkpoint_path:
            ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
            extra_hw_paths.append(os.path.join(ckpt_dir, "hardware_session_log.csv"))
            extra_hw_paths.append(os.path.join(os.path.dirname(ckpt_dir), "hardware_session_log.csv"))
        tracker.log_session(stats=stats, extra_csv_paths=extra_hw_paths)
        tracker.print_summary(stats=stats)

        return self.df_raw

    def _compute_summary(self):
        """Computes mean and std aggregation across methods and perturbation categories."""
        if self.df_raw is None or len(self.df_raw) == 0:
            return

        # Always aggregate strictly on decision-preserved pairs
        df_valid = self.df_raw[self.df_raw["is_preserved"]] if "is_preserved" in self.df_raw.columns else self.df_raw

        grouped = df_valid.groupby(["explainer_name", "perturbation_category", "perturbation_name"])
        agg_funcs = {
            "spearman": ["mean", "std"],
            "kendall": ["mean", "std"],
            "pearson": ["mean", "std"],
            "jaccard_k1": ["mean"],
            "jaccard_k3": ["mean", "std"],
            "jaccard_k5": ["mean"],
            "pert_time_s": ["mean"],
            "vram_mb": ["max"]
        }
        summary = grouped.agg(agg_funcs)
        summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
        self.df_summary = summary.reset_index()

    def generate_latex_table(self) -> str:
        """
        Generates publication-ready LaTeX table for Chapter 3/4 of the thesis.
        """
        if self.df_summary is None:
            self._compute_summary()

        latex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Ocena stabilności metod interpretowalności dla par z zachowaną predykcją (RQ1)}",
            r"\label{tab:rq1_stability}",
            r"\begin{tabular}{llcccc}",
            r"\hline",
            r"\textbf{Metoda} & \textbf{Perturbacja} & \textbf{Spearman $\rho$} & \textbf{Kendall $\tau$} & \textbf{Jaccard ($K=3$)} & \textbf{Jaccard ($K=5$)} \\",
            r"\hline"
        ]

        current_method = None
        for _, row in self.df_summary.iterrows():
            m = row['explainer_name']
            p = row['perturbation_name'].replace("_", r"\_")
            sp_mean, sp_std = row['spearman_mean'], row['spearman_std']
            kd_mean, kd_std = row['kendall_mean'], row['kendall_std']
            j3_mean, j3_std = row['jaccard_k3_mean'], row['jaccard_k3_std']
            j5_mean = row['jaccard_k5_mean']

            # Add horizontal separator between methods
            if current_method is not None and current_method != m:
                latex_lines.append(r"\hline")
            current_method = m

            sp_std_str = f"{sp_std:.2f}" if not np.isnan(sp_std) else "0.00"
            kd_std_str = f"{kd_std:.2f}" if not np.isnan(kd_std) else "0.00"
            j3_std_str = f"{j3_std:.2f}" if not np.isnan(j3_std) else "0.00"

            line = f"{m} & \\texttt{{{p}}} & {sp_mean:.3f} $\\pm$ {sp_std_str} & {kd_mean:.3f} $\\pm$ {kd_std_str} & {j3_mean:.3f} $\\pm$ {j3_std_str} & {j5_mean:.3f} \\\\"
            latex_lines.append(line)

        latex_lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}"
        ])

        return "\n".join(latex_lines)

    def save_results(self, output_dir: str = "results/rq1_stability") -> Dict[str, str]:
        """
        Saves raw CSV, summary CSV, LaTeX table and generates thesis visualization plots.
        """
        if self.df_raw is None:
            raise RuntimeError("Najpierw uruchom metodę .run(), aby uzyskać wyniki!")

        os.makedirs(output_dir, exist_ok=True)
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # 1. Save CSV and JSON
        raw_csv = os.path.join(output_dir, "rq1_stability_raw.csv")
        summary_csv = os.path.join(output_dir, "rq1_stability_summary.csv")
        self.df_raw.to_csv(raw_csv, index=False)
        self.df_summary.to_csv(summary_csv, index=False)

        # 2. Save LaTeX Table
        latex_path = os.path.join(output_dir, "rq1_table.tex")
        latex_str = self.generate_latex_table()
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex_str)

        # 3. Generate Visualizations
        p1 = os.path.join(plots_dir, "rq1_spearman_distribution.png")
        plot_rq1_metric_distribution(
            self.df_raw, 
            metric="spearman", 
            metric_label=r"Korelacja rang Spearmana ($\rho$)", 
            save_path=p1
        )

        p2 = os.path.join(plots_dir, "rq1_kendall_distribution.png")
        plot_rq1_metric_distribution(
            self.df_raw, 
            metric="kendall", 
            metric_label=r"Współczynnik $\tau$ Kendalla", 
            save_path=p2
        )

        p3 = os.path.join(plots_dir, "rq1_jaccard_curves.png")
        plot_rq1_jaccard_k_curves(self.df_raw, save_path=p3)

        # Example attribution shift plot
        p4 = os.path.join(plots_dir, "rq1_token_attribution_example.png")
        for explainer_name in ["Occlusion", "IntegratedGradients", "SHAP"]:
            orig_k = f"{explainer_name}_orig"
            pert_k = f"{explainer_name}_pert"
            if orig_k in self.last_explanations_cache and pert_k in self.last_explanations_cache:
                plot_rq1_token_attribution_comparison(
                    self.last_explanations_cache[orig_k],
                    self.last_explanations_cache[pert_k],
                    pert_name="prefix_overall",
                    save_path=p4
                )
                break

        print(f"\n[RQ1] Wyniki zapisano pomyślnie w katalogu '{output_dir}':")
        print(f" - Surowe dane: {raw_csv}")
        print(f" - Podsumowanie: {summary_csv}")
        print(f" - Tabela LaTeX: {latex_path}")
        print(f" - Wykresy (PNG/PDF): {plots_dir}")

        return {
            "raw_csv": raw_csv,
            "summary_csv": summary_csv,
            "latex_table": latex_path,
            "plots_dir": plots_dir
        }
