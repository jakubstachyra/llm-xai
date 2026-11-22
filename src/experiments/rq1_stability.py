import os
import json
import time
import zlib
import inspect
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon, spearmanr
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

try:
    from models import SentimentCausalLM
    from explainers import SHAPExplainer, OcclusionExplainer, BaseExplainer, IntegratedGradientsExplainer
    from perturbations import TextPerturber
    from hardware_tracker import HardwareSessionTracker
    from metrics import (
        align_word_attributions,
        compute_all_stability_metrics,
        compute_perturbation_strength,
    )
except ImportError:
    from src.models import SentimentCausalLM
    from src.explainers import SHAPExplainer, OcclusionExplainer, BaseExplainer, IntegratedGradientsExplainer
    from src.perturbations import TextPerturber
    from src.hardware_tracker import HardwareSessionTracker
    from src.metrics import (
        align_word_attributions,
        compute_all_stability_metrics,
        compute_perturbation_strength,
    )
from .visualizations import (
    plot_rq1_metric_distribution,
    plot_rq1_jaccard_k_curves,
    plot_rq1_token_attribution_comparison
)


NOISE_FLOOR_CATEGORY = "control"
NOISE_FLOOR_NAME = "reseed_same_text"


def _independent_pert_seed(base_seed: int, sample_idx: int, pert_name: str) -> int:
    """
    Deterministic, statistically independent seed per (sample_idx, pert_name) pair.
    Prevents Common Random Numbers (CRN) coupling between original and perturbed runs
    in stochastic explainers (LIME, SHAP), ensuring unbiased comparison against noise floor.
    """
    tag = f"{base_seed}:{sample_idx}:{pert_name}".encode("utf-8")
    return int(zlib.crc32(tag) & 0x7FFFFFFF)


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
        self.df_decisions: Optional[pd.DataFrame] = None
        self.df_preservation: Optional[pd.DataFrame] = None
        self.df_strength_analysis: Optional[pd.DataFrame] = None
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
        samples = [
            {"id": int(data[i].get("id", i)), "sentence": str(data[i]["sentence"]).strip(), "label": int(data[i]["label"])}
            for i in indices
        ]

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
        checkpoint_interval: int = 10,
        overwrite_methods: Optional[List[str]] = None,
        save_perturbed_attributions: bool = False,
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
            overwrite_methods: Optional list of explainer_name strings to purge from checkpoint and recompute.
            save_perturbed_attributions: If False (default), saves only original (__orig__) attributions to
                rq1_attributions.jsonl so RQ2 can reuse them cleanly without bloating disk space.
        """
        samples = self.load_dataset(num_samples=num_samples, seed=seed)
        records: List[Dict[str, Any]] = []
        decision_records: List[Dict[str, Any]] = []
        completed_units: set = set()
        fully_completed_sample_explainers: set = set()
        logged_decision_pairs: set = set()
        all_flipped_samples: set = set()

        decisions_csv_path = (
            os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "rq1_perturbation_decisions.csv")
            if checkpoint_path else None
        )

        # Load existing checkpoint if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                df_existing = pd.read_csv(checkpoint_path)
                if overwrite_methods and len(df_existing) > 0 and "explainer_name" in df_existing.columns:
                    before_len = len(df_existing)
                    df_existing = df_existing[~df_existing["explainer_name"].isin(overwrite_methods)].copy()
                    df_existing.to_csv(checkpoint_path, index=False)
                    print(f"[RQ1 Checkpoint] Usunięto {before_len - len(df_existing)} starych wierszy dla nadpisywanych metod: {overwrite_methods}")
                if len(df_existing) > 0 and "sample_id" in df_existing.columns and "explainer_name" in df_existing.columns:
                    records = df_existing.to_dict("records")
                    for r in records:
                        sid = int(r["sample_id"])
                        ename = str(r["explainer_name"])
                        pname = str(r.get("perturbation_name", ""))
                        completed_units.add((sid, ename, pname))
                        if pname != NOISE_FLOOR_NAME:
                            # If any non-control perturbation was recorded for (sample_id, explainer_name),
                            # all perturbations for that (sample_id, explainer_name) were already evaluated
                            # in a single atomic loop (missing perturbations are decision flips).
                            fully_completed_sample_explainers.add((sid, ename))
                    distinct_samples = set(r["sample_id"] for r in records)
                    print(f"[RQ1 Checkpoint] Wczytano {len(records)} rekordów z checkpointa ({len(distinct_samples)} częściowo lub w pełni gotowych próbek).")
            except Exception as e:
                print(f"[RQ1 Checkpoint] Ostrzeżenie przy wczytywaniu checkpointa ({e}). Start od zera.")
                records = []

        if decisions_csv_path and os.path.exists(decisions_csv_path):
            try:
                df_dec_existing = pd.read_csv(decisions_csv_path)
                if len(df_dec_existing) > 0 and "sample_id" in df_dec_existing.columns:
                    decision_records = df_dec_existing.to_dict("records")
                    for r in decision_records:
                        logged_decision_pairs.add((int(r["sample_id"]), str(r["perturbation_name"])))
                    for sid, grp in df_dec_existing.groupby("sample_id"):
                        if len(grp) >= 1 and not grp["is_preserved"].astype(bool).any():
                            all_flipped_samples.add(int(sid))
            except Exception:
                decision_records = []

        total_pairs = 0
        preserved_pairs = 0
        samples_processed_this_run = 0

        tracker = HardwareSessionTracker(self.model, experiment_name="RQ1-Stability", num_samples=num_samples)
        tracker.start()

        iterator = tqdm(enumerate(samples), total=len(samples), desc="Ewaluacja próbek RQ1") if progress_bar else enumerate(samples)

        unflushed_records: List[Dict[str, Any]] = []
        unflushed_decision_records: List[Dict[str, Any]] = []
        unflushed_attr_records: List[Dict[str, Any]] = []
        attr_jsonl_path = (
            os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "rq1_attributions.jsonl")
            if checkpoint_path else None
        )

        def _flush_csv_safe(csv_path: str, chunk_records: List[Dict[str, Any]]):
            if not chunk_records:
                return
            df_chunk = pd.DataFrame(chunk_records)
            file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
            if not file_exists:
                df_chunk.to_csv(csv_path, index=False)
            else:
                existing_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
                if existing_cols == list(df_chunk.columns):
                    df_chunk.to_csv(csv_path, mode="a", header=False, index=False)
                else:
                    df_old = pd.read_csv(csv_path)
                    df_merged = pd.concat([df_old, df_chunk], ignore_index=True)
                    df_merged.to_csv(csv_path, index=False)

        def _flush_checkpoints():
            if not checkpoint_path:
                return
            os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
            if unflushed_records:
                _flush_csv_safe(checkpoint_path, unflushed_records)
                unflushed_records.clear()
            if decisions_csv_path and unflushed_decision_records:
                _flush_csv_safe(decisions_csv_path, unflushed_decision_records)
                unflushed_decision_records.clear()
            if attr_jsonl_path and unflushed_attr_records:
                with open(attr_jsonl_path, "a", encoding="utf-8") as f_attr:
                    for rec in unflushed_attr_records:
                        f_attr.write(json.dumps(rec, ensure_ascii=False) + "\n")
                unflushed_attr_records.clear()

        try:
            for sample_idx, item in iterator:
                if max_samples_this_run is not None and samples_processed_this_run >= max_samples_this_run:
                    print(f"\n[RQ1] Osiągnięto limit partii sesji ({max_samples_this_run} próbek). Zapisywanie stanu i pauza.")
                    break

                text = item["sentence"].strip()
                true_label = item["label"]
                
                # Check if all explainers for this sample_idx are already completed
                explainer_names = [explainer_label(e) for e in self.explainers]
                cat_perts = TextPerturber.get_categorized_perturbations(text)
                # Filter out accidental identity perturbations (pert_text.strip() == text.strip())
                filtered_cat_perts: Dict[str, Dict[str, str]] = {}
                for category, perts in cat_perts.items():
                    non_ident = {
                        p_name: p_text.strip()
                        for p_name, p_text in perts.items()
                        if p_text.strip() != text
                    }
                    if non_ident:
                        filtered_cat_perts[category] = non_ident
                cat_perts = filtered_cat_perts
                all_pert_names = [p_name for perts in cat_perts.values() for p_name in perts.keys()]

                def _needs_control(exp) -> bool:
                    return include_noise_floor and getattr(exp, "stochastic", False)

                all_done = (
                    sample_idx in all_flipped_samples
                    and not any(_needs_control(e) for e in self.explainers)
                ) or (
                    all(
                        (sample_idx, exp_name) in fully_completed_sample_explainers or all(
                            (sample_idx, exp_name, p_name) in completed_units
                            for p_name in all_pert_names
                        )
                        for exp_name in explainer_names
                    ) and all(
                        (sample_idx, explainer_label(e), NOISE_FLOOR_NAME) in completed_units
                        for e in self.explainers if _needs_control(e)
                    )
                )
                if all_done:
                    continue  # Skip already fully evaluated sample

                # Batch-predict original sentence + all perturbations in a single call
                flat_pert_items: List[Tuple[str, str, str]] = [
                    (category, pert_name, pert_text)
                    for category, perts in cat_perts.items()
                    for pert_name, pert_text in perts.items()
                ]
                all_texts_to_predict = [text] + [pt for _, _, pt in flat_pert_items]
                all_probs = self.model.predict_texts_batch(all_texts_to_predict)
                probs = all_probs[0]
                pred_class = int(np.argmax(probs))
                p_target = float(probs[pred_class])

                pert_status: Dict[str, Dict[str, Any]] = {}
                for idx_p, (category, pert_name, pert_text) in enumerate(flat_pert_items):
                    total_pairs += 1
                    pert_probs = all_probs[idx_p + 1]
                    pert_pred = int(np.argmax(pert_probs))
                    is_preserved = bool(pert_pred == pred_class)
                    if is_preserved:
                        preserved_pairs += 1

                    p_pert_target = float(pert_probs[pred_class])
                    abs_dp = abs(p_target - p_pert_target)
                    pert_status[pert_name] = {
                        "category": category,
                        "text": pert_text,
                        "pred_class": pert_pred,
                        "is_preserved": is_preserved,
                        "p_target": p_pert_target,
                        "pert_Base_prob": float(pert_probs[pert_pred]),
                        "abs_delta_p": abs_dp,
                    }
                    if (sample_idx, pert_name) not in logged_decision_pairs:
                        char_er = compute_perturbation_strength(
                            text, pert_text, len(text.split()), len(pert_text.split())
                        )["char_edit_ratio"]
                        dec_row = {
                            "sample_id": sample_idx,
                            "perturbation_category": category,
                            "perturbation_name": pert_name,
                            "is_preserved": is_preserved,
                            "true_label": true_label,
                            "orig_pred": pred_class,
                            "pert_pred": pert_pred,
                            "p_orig": p_target,
                            "p_pert": p_pert_target,
                            "abs_delta_p": abs_dp,
                            "char_edit_ratio": char_er,
                        }
                        decision_records.append(dec_row)
                        unflushed_decision_records.append(dec_row)
                        logged_decision_pairs.add((sample_idx, pert_name))

                sample_new_records: List[Dict[str, Any]] = []

                # Generate explanations and compute stability
                for explainer in self.explainers:
                    explainer_name = explainer_label(explainer)
                    method_family = getattr(explainer, "config_family", explainer.__class__.__name__.replace("Explainer", ""))
                    param_value = getattr(explainer, "config_param", None)
                    has_seed_param = "seed" in inspect.signature(explainer.explain).parameters
                    has_base_prob_param = "baseline_prob" in inspect.signature(explainer.explain).parameters
                    base_exp_seed = int(getattr(explainer, "seed", seed))

                    # Filter perturbations that are already in checkpoint for this explainer
                    if (sample_idx, explainer_name) in fully_completed_sample_explainers:
                        perts_to_run = []
                    else:
                        perts_to_run = [
                            (p_name, p_info) for p_name, p_info in pert_status.items()
                            if (sample_idx, explainer_name, p_name) not in completed_units
                        ]
                    active_perts_to_run = [
                        (p_name, p_info) for p_name, p_info in perts_to_run
                        if (not filter_decision_preserved) or p_info["is_preserved"]
                    ]
                    run_control = _needs_control(explainer) and (sample_idx, explainer_name, NOISE_FLOOR_NAME) not in completed_units
                    if not active_perts_to_run and not run_control:
                        for p_name, _ in perts_to_run:
                            completed_units.add((sample_idx, explainer_name, p_name))
                        if perts_to_run:
                            fully_completed_sample_explainers.add((sample_idx, explainer_name))
                        continue

                    print(f"  [Próbka {sample_idx+1}/{len(samples)}] Liczenie: {explainer_name} ({len(active_perts_to_run)} aktywnych perturbacji)...", flush=True)

                    # Reset GPU memory stats before explanation
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                    # Explain original prompt (pass precomputed baseline_prob to avoid redundant forward pass)
                    orig_kwargs: Dict[str, Any] = {}
                    if has_base_prob_param:
                        orig_kwargs["baseline_prob"] = p_target
                    if has_seed_param:
                        orig_kwargs["seed"] = _independent_pert_seed(base_exp_seed, sample_idx, "__orig__")

                    c0 = self.model.get_counters()
                    orig_res = explainer.explain(text, target_class=pred_class, **orig_kwargs)
                    c1 = self.model.get_counters()
                    orig_fwd = c1["forward_passes"] - c0["forward_passes"]
                    orig_bwd = c1["backward_passes"] - c0["backward_passes"]
                    orig_time = orig_res.execution_time
                    orig_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
                    orig_meta = dict(orig_res.metadata or {})
                    if hasattr(explainer, "steps"):
                        orig_meta.setdefault("steps", getattr(explainer, "steps"))
                    if hasattr(explainer, "baseline_type"):
                        orig_meta.setdefault("baseline_type", getattr(explainer, "baseline_type"))
                    if hasattr(explainer, "num_samples"):
                        orig_meta.setdefault("num_samples", getattr(explainer, "num_samples"))
                    if hasattr(explainer, "max_evals"):
                        orig_meta.setdefault("max_evals", getattr(explainer, "max_evals"))

                    unflushed_attr_records.append({
                        "sample_id": sample_idx,
                        "model_name": self.model.model_name,
                        "explainer_name": explainer_name,
                        "Canonical_name": explainer.explainer_name,
                        "variant": "__orig__",
                        "text": text,
                        "target_class": int(pred_class),
                        "baseline_prob": float(p_target),
                        "execution_time": float(orig_time),
                        "forward_passes": int(orig_fwd),
                        "backward_passes": int(orig_bwd),
                        "tokens": list(orig_res.tokens),
                        "attributions": [round(float(x), 6) for x in orig_res.attributions],
                        "metadata": orig_meta,
                    })

                    if sample_idx == 0:
                        self.last_explanations_cache[f"{explainer_name}_orig"] = orig_res

                    for p_name, _ in perts_to_run:
                        if filter_decision_preserved and not pert_status[p_name]["is_preserved"]:
                            completed_units.add((sample_idx, explainer_name, p_name))

                    for pert_name, p_info in active_perts_to_run:
                        pert_text = p_info["text"]
                        category = p_info["category"]
                        is_preserved = p_info["is_preserved"]

                        # Reset GPU memory stats for perturbed run
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()

                        pert_kwargs: Dict[str, Any] = {}
                        if has_base_prob_param:
                            pert_kwargs["baseline_prob"] = float(p_info["p_target"])
                        if has_seed_param:
                            # Statistically independent seed per (sample, perturbation)
                            pert_kwargs["seed"] = _independent_pert_seed(base_exp_seed, sample_idx, pert_name)

                        c0 = self.model.get_counters()
                        pert_res = explainer.explain(pert_text, target_class=pred_class, **pert_kwargs)
                        c1 = self.model.get_counters()
                        pert_vram_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
                        pert_meta = pert_res.metadata or {}

                        if save_perturbed_attributions:
                            unflushed_attr_records.append({
                                "sample_id": sample_idx,
                                "model_name": self.model.model_name,
                                "explainer_name": explainer_name,
                                "Canonical_name": explainer.explainer_name,
                                "variant": pert_name,
                                "text": pert_text,
                                "target_class": int(pred_class),
                                "baseline_prob": float(p_info["p_target"]),
                                "execution_time": float(pert_res.execution_time),
                                "forward_passes": int(c1["forward_passes"] - c0["forward_passes"]),
                                "backward_passes": int(c1["backward_passes"] - c0["backward_passes"]),
                                "tokens": list(pert_res.tokens),
                                "attributions": [round(float(x), 6) for x in pert_res.attributions],
                                "metadata": pert_meta,
                            })

                        if sample_idx == 0 and pert_name == "prefix_overall":
                            self.last_explanations_cache[f"{explainer_name}_pert"] = pert_res

                        # Calculate word-aligned stability metrics and perturbation strength in a single alignment pass
                        stab = compute_all_stability_metrics(orig_res, pert_res)
                        strength = compute_perturbation_strength(
                            text, pert_text, len(orig_res.tokens), len(pert_res.tokens)
                        )

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
                            "abs_delta_p": p_info["abs_delta_p"],
                            "char_edit_ratio": strength["char_edit_ratio"],
                            "token_len_diff": strength["token_len_diff"],
                            "token_len_rel_diff": strength["token_len_rel_diff"],
                            "orig_num_tokens": len(orig_res.tokens),
                            "aligned_words": stab["aligned_words"],
                            "spearman": stab["spearman"],
                            "kendall": stab["kendall"],
                            "pearson": stab["pearson"],
                            "jaccard_k1": stab["jaccard_k1"],
                            "jaccard_k3": stab["jaccard_k3"],
                            "jaccard_k5": stab["jaccard_k5"],
                            "orig_completeness_err": orig_meta.get("completeness_error", np.nan),
                            "pert_completeness_err": pert_meta.get("completeness_error", np.nan),
                            "orig_r2": orig_meta.get("r2_score", np.nan),
                            "pert_r2": pert_meta.get("r2_score", np.nan),
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

                    # Noise-floor control: same text, independent random seed
                    if run_control:
                        ctrl_kwargs: Dict[str, Any] = {}
                        if has_base_prob_param:
                            ctrl_kwargs["baseline_prob"] = p_target
                        if has_seed_param:
                            ctrl_kwargs["seed"] = _independent_pert_seed(base_exp_seed, sample_idx, NOISE_FLOOR_NAME)
                        c0 = self.model.get_counters()
                        ctrl_res = explainer.explain(text, target_class=pred_class, **ctrl_kwargs)
                        c1 = self.model.get_counters()
                        ctrl_meta = ctrl_res.metadata or {}
                        stab_c = compute_all_stability_metrics(orig_res, ctrl_res)
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
                            "abs_delta_p": 0.0,
                            "char_edit_ratio": 0.0,
                            "token_len_diff": 0,
                            "token_len_rel_diff": 0.0,
                            "orig_num_tokens": len(orig_res.tokens),
                            "aligned_words": stab_c["aligned_words"],
                            "spearman": stab_c["spearman"],
                            "kendall": stab_c["kendall"],
                            "pearson": stab_c["pearson"],
                            "jaccard_k1": stab_c["jaccard_k1"],
                            "jaccard_k3": stab_c["jaccard_k3"],
                            "jaccard_k5": stab_c["jaccard_k5"],
                            "orig_completeness_err": orig_meta.get("completeness_error", np.nan),
                            "pert_completeness_err": ctrl_meta.get("completeness_error", np.nan),
                            "orig_r2": orig_meta.get("r2_score", np.nan),
                            "pert_r2": ctrl_meta.get("r2_score", np.nan),
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
                if checkpoint_path and (len(unflushed_records) > 0 or len(unflushed_decision_records) > 0) and (samples_processed_this_run > 0 and samples_processed_this_run % checkpoint_interval == 0):
                    _flush_checkpoints()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Also save live hardware snapshot and append to timeline CSV
                    live_hw_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "hardware_live_snapshot.json")
                    live_hw_csv = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "hardware_timeline.csv")
                    try:
                        snap = tracker.snapshot_stats(samples_processed_this_run)
                        with open(live_hw_path, "w", encoding="utf-8") as f_hw:
                            json.dump(snap, f_hw, indent=2)
                        df_snap = pd.DataFrame([snap])
                        write_hw_hdr = not os.path.exists(live_hw_csv) or os.path.getsize(live_hw_csv) == 0
                        df_snap.to_csv(live_hw_csv, mode="a", header=write_hw_hdr, index=False)
                    except Exception:
                        pass

            # Flush any remaining buffered records at the end of loop
            _flush_checkpoints()

        except KeyboardInterrupt:
            print("\n[RQ1] Przerwano działanie skryptu przez użytkownika (Ctrl+C).")
            _flush_checkpoints()
            if checkpoint_path:
                print(f"[RQ1] Wszystkie dotychczasowe wyniki zostały bezpiecznie zapisane w checkpointcie: {checkpoint_path}")

        # Assemble and deduplicate full raw and decision dataframes
        self.df_raw = pd.DataFrame(records)
        if len(self.df_raw) > 0:
            self.df_raw.drop_duplicates(subset=["sample_id", "explainer_name", "perturbation_name"], inplace=True)
        self.df_decisions = pd.DataFrame(decision_records)
        if len(self.df_decisions) > 0:
            self.df_decisions.drop_duplicates(subset=["sample_id", "perturbation_name"], inplace=True)

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
        """
        Computes per-perturbation summary AND hierarchical sample-level summary
        with paired cluster bootstrap 95% confidence intervals and Friedman/Wilcoxon-Holm tests.
        """
        if self.df_raw is None or len(self.df_raw) == 0:
            return

        # Always aggregate strictly on decision-preserved pairs (excluding accidental exact identities)
        df_valid = self.df_raw[self.df_raw["is_preserved"]].copy() if "is_preserved" in self.df_raw.columns else self.df_raw.copy()
        if "original_text" in df_valid.columns and "perturbed_text" in df_valid.columns:
            non_ctrl = df_valid["perturbation_category"] != NOISE_FLOOR_CATEGORY
            ident_mask = non_ctrl & (df_valid["original_text"].astype(str).str.strip() == df_valid["perturbed_text"].astype(str).str.strip())
            df_valid = df_valid[~ident_mask]

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

        # Preservation rate & Flip Rate summary from rq1_perturbation_decisions.csv
        if getattr(self, "df_decisions", None) is not None and len(self.df_decisions) > 0:
            pres_grp = self.df_decisions.groupby(["perturbation_category", "perturbation_name"], as_index=False).agg(
                n_total=("is_preserved", "size"),
                n_preserved=("is_preserved", "sum"),
                mean_abs_delta_p=("abs_delta_p", "mean"),
                mean_char_edit_ratio=("char_edit_ratio", "mean"),
            )
            pres_grp["n_preserved"] = pres_grp["n_preserved"].astype(int)
            pres_grp["preservation_rate_pct"] = 100.0 * pres_grp["n_preserved"] / pres_grp["n_total"].clip(lower=1)
            pres_grp["flip_rate_pct"] = 100.0 - pres_grp["preservation_rate_pct"]
            self.df_preservation = pres_grp.sort_values(["perturbation_category", "preservation_rate_pct"])

        # Hierarchical per-sample aggregation (Section 4.7.1 of thesis):
        # First average within each (sample_id, explainer_name, category), then across categories per sample
        df_main = df_valid[df_valid["perturbation_category"] != NOISE_FLOOR_CATEGORY]
        df_ctrl = df_valid[df_valid["perturbation_category"] == NOISE_FLOOR_CATEGORY]
        if len(df_main) == 0:
            return

        # Perturbation strength analysis (Spearman correlation + low/mid/high strength tertiles)
        def _safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
            if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
                r_v, p_v = spearmanr(x, y)
                return (float(r_v) if not np.isnan(r_v) else np.nan, float(p_v) if not np.isnan(p_v) else np.nan)
            return np.nan, np.nan

        def _tertile_means(sub_df: pd.DataFrame, strength_col: str) -> Tuple[float, float, float]:
            if strength_col not in sub_df.columns:
                return np.nan, np.nan, np.nan
            clean_s = sub_df[[strength_col, "spearman"]].dropna()
            if len(clean_s) < 3:
                return np.nan, np.nan, np.nan
            s_vals = clean_s[strength_col].to_numpy(dtype=float)
            if np.std(s_vals) == 0:
                return np.nan, float(clean_s["spearman"].mean()), np.nan
            try:
                bins = pd.qcut(clean_s[strength_col], q=3, labels=False, duplicates="drop")
            except Exception:
                bins = None
            if bins is not None and bins.nunique() == 3:
                m_map = clean_s.groupby(bins)["spearman"].mean().to_dict()
                return float(m_map.get(0, np.nan)), float(m_map.get(1, np.nan)), float(m_map.get(2, np.nan))
            q33, q67 = float(np.quantile(s_vals, 1.0 / 3.0)), float(np.quantile(s_vals, 2.0 / 3.0))
            low_m = clean_s[clean_s[strength_col] <= q33]["spearman"].mean()
            mid_m = clean_s[(clean_s[strength_col] > q33) & (clean_s[strength_col] <= q67)]["spearman"].mean()
            high_m = clean_s[clean_s[strength_col] > q67]["spearman"].mean()
            return (
                float(low_m) if not pd.isna(low_m) else np.nan,
                float(mid_m) if not pd.isna(mid_m) else np.nan,
                float(high_m) if not pd.isna(high_m) else np.nan,
            )

        strength_rows = []
        for m_name, m_grp in df_main.groupby("explainer_name"):
            valid_sp = m_grp.dropna(subset=["spearman"])
            sp_arr = valid_sp["spearman"].to_numpy(dtype=float)
            rho_dp, p_dp = (
                _safe_spearman(valid_sp["abs_delta_p"].to_numpy(dtype=float), sp_arr)
                if "abs_delta_p" in valid_sp.columns else (np.nan, np.nan)
            )
            rho_ce, p_ce = (
                _safe_spearman(valid_sp["char_edit_ratio"].to_numpy(dtype=float), sp_arr)
                if "char_edit_ratio" in valid_sp.columns else (np.nan, np.nan)
            )
            rho_td, p_td = (
                _safe_spearman(valid_sp["token_len_rel_diff"].to_numpy(dtype=float), sp_arr)
                if "token_len_rel_diff" in valid_sp.columns else (np.nan, np.nan)
            )
            dp_lo, dp_mid, dp_hi = _tertile_means(valid_sp, "abs_delta_p")
            ed_lo, ed_mid, ed_hi = _tertile_means(valid_sp, "char_edit_ratio")
            strength_rows.append({
                "explainer_name": m_name,
                "n_pairs": int(len(valid_sp)),
                "rho_vs_delta_p": rho_dp,
                "p_vs_delta_p": p_dp,
                "rho_vs_char_edit": rho_ce,
                "p_vs_char_edit": p_ce,
                "rho_vs_tok_diff": rho_td,
                "p_vs_tok_diff": p_td,
                "spearman_delta_p_low": dp_lo,
                "spearman_delta_p_mid": dp_mid,
                "spearman_delta_p_high": dp_hi,
                "spearman_edit_low": ed_lo,
                "spearman_edit_mid": ed_mid,
                "spearman_edit_high": ed_hi,
            })
        self.df_strength_analysis = pd.DataFrame(strength_rows)

        per_sample_cat = df_main.groupby(["sample_id", "explainer_name", "perturbation_category"], as_index=False).agg(
            spearman=("spearman", "mean"),
            kendall=("kendall", "mean"),
            jaccard_k1=("jaccard_k1", "mean"),
            jaccard_k3=("jaccard_k3", "mean"),
            jaccard_k5=("jaccard_k5", "mean"),
            orig_time_s=("orig_time_s", "mean"),
            orig_fwd=("orig_forward_passes", "mean"),
            orig_bwd=("orig_backward_passes", "mean"),
        )
        per_sample = per_sample_cat.groupby(["sample_id", "explainer_name"], as_index=False).agg(
            spearman=("spearman", "mean"),
            kendall=("kendall", "mean"),
            jaccard_k1=("jaccard_k1", "mean"),
            jaccard_k3=("jaccard_k3", "mean"),
            jaccard_k5=("jaccard_k5", "mean"),
            orig_time_s=("orig_time_s", "mean"),
            orig_fwd=("orig_fwd", "mean"),
            orig_bwd=("orig_bwd", "mean"),
        )
        self.df_sample_level = per_sample

        # Cluster bootstrap 95% CI over samples (B=1000, seed=42)
        rng = np.random.default_rng(42)
        method_rows = []
        ctrl_map = (
            df_ctrl.groupby("explainer_name")["spearman"].mean().to_dict()
            if len(df_ctrl) > 0 else {}
        )
        for m_name, grp in per_sample.groupby("explainer_name"):
            vals = grp["spearman"].dropna().to_numpy()
            n_s = len(vals)
            mean_v = float(np.mean(vals)) if n_s > 0 else np.nan
            sem_v = float(np.std(vals, ddof=1) / np.sqrt(n_s)) if n_s > 1 else 0.0
            if n_s >= 2:
                boot_means = [float(np.mean(rng.choice(vals, size=n_s, replace=True))) for _ in range(1000)]
                ci_low, ci_high = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
            else:
                ci_low, ci_high = mean_v, mean_v
            nf = float(ctrl_map.get(m_name, np.nan))
            method_rows.append({
                "explainer_name": m_name,
                "n_samples": n_s,
                "spearman_mean": mean_v,
                "spearman_sem": sem_v,
                "spearman_ci95_low": ci_low,
                "spearman_ci95_high": ci_high,
                "noise_floor_spearman": nf,
                "spearman_norm_by_floor": (mean_v / nf) if (not np.isnan(nf) and nf > 0) else mean_v,
                "kendall_mean": float(grp["kendall"].mean()),
                "jaccard_k3_mean": float(grp["jaccard_k3"].mean()),
                "jaccard_k5_mean": float(grp["jaccard_k5"].mean()),
                "orig_time_s_mean": float(grp["orig_time_s"].mean()),
                "fwdu_mean": float(grp["orig_fwd"].mean() + 2.0 * grp["orig_bwd"].mean()),
            })
        self.df_method_summary = pd.DataFrame(method_rows).sort_values("spearman_mean", ascending=False)

        # Non-parametric Friedman + pairwise Wilcoxon with Holm correction across paired samples
        pivot = per_sample.pivot(index="sample_id", columns="explainer_name", values="spearman").dropna()
        self.stats_tests: Dict[str, Any] = {}
        if pivot.shape[1] >= 2 and pivot.shape[0] >= 5:
            cols = list(pivot.columns)
            if len(cols) >= 3:
                try:
                    stat_f, p_f = friedmanchisquare(*[pivot[c].to_numpy() for c in cols])
                    self.stats_tests["friedman"] = {"chi2": float(stat_f), "p_value": float(p_f), "n_samples": int(pivot.shape[0])}
                except Exception:
                    pass
            pw_list = []
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    a, b = pivot[cols[i]].to_numpy(), pivot[cols[j]].to_numpy()
                    diff = a - b
                    if np.allclose(diff, 0):
                        continue
                    try:
                        w_stat, p_val = wilcoxon(a, b)
                        pw_list.append({
                            "method_a": cols[i],
                            "method_b": cols[j],
                            "mean_diff": float(np.mean(diff)),
                            "median_diff": float(np.median(diff)),
                            "wilcoxon_stat": float(w_stat),
                            "p_uncorrected": float(p_val),
                        })
                    except Exception:
                        pass
            if pw_list:
                pw_df = pd.DataFrame(pw_list).sort_values("p_uncorrected").reset_index(drop=True)
                m_tests = len(pw_df)
                holm_p = []
                running_max = 0.0
                for idx_row, r_pw in pw_df.iterrows():
                    adj = min(1.0, float(r_pw["p_uncorrected"]) * (m_tests - idx_row))
                    running_max = max(running_max, adj)
                    holm_p.append(running_max)
                pw_df["p_holm"] = holm_p
                self.stats_tests["pairwise_wilcoxon_holm"] = pw_df.to_dict("records")

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

        if getattr(self, "df_method_summary", None) is not None:
            self.df_method_summary.to_csv(os.path.join(output_dir, "rq1_method_summary.csv"), index=False)
        if getattr(self, "df_sample_level", None) is not None:
            self.df_sample_level.to_csv(os.path.join(output_dir, "rq1_sample_level.csv"), index=False)
        decisions_csv = None
        if getattr(self, "df_decisions", None) is not None and len(self.df_decisions) > 0:
            decisions_csv = os.path.join(output_dir, "rq1_perturbation_decisions.csv")
            self.df_decisions.to_csv(decisions_csv, index=False)
        preservation_csv = None
        if getattr(self, "df_preservation", None) is not None and len(self.df_preservation) > 0:
            preservation_csv = os.path.join(output_dir, "rq1_preservation_summary.csv")
            self.df_preservation.to_csv(preservation_csv, index=False)
        strength_csv = None
        if getattr(self, "df_strength_analysis", None) is not None and len(self.df_strength_analysis) > 0:
            strength_csv = os.path.join(output_dir, "rq1_strength_analysis.csv")
            self.df_strength_analysis.to_csv(strength_csv, index=False)
        if getattr(self, "stats_tests", None):
            with open(os.path.join(output_dir, "rq1_statistical_tests.json"), "w", encoding="utf-8") as f_st:
                json.dump(self.stats_tests, f_st, indent=2)

        # 2. Save LaTeX Table
        latex_path = os.path.join(output_dir, "rq1_table.tex")
        latex_str = self.generate_latex_table()
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex_str)

        # 3. Generate Visualizations (exclude control noise-floor and identical pairs)
        df_plot = self.df_raw.copy()
        if len(df_plot) > 0:
            if "is_preserved" in df_plot.columns:
                df_plot = df_plot[df_plot["is_preserved"] == True]
            if "perturbation_category" in df_plot.columns:
                df_plot = df_plot[df_plot["perturbation_category"] != NOISE_FLOOR_CATEGORY]
            if "original_text" in df_plot.columns and "perturbed_text" in df_plot.columns:
                df_plot = df_plot[
                    df_plot["original_text"].astype(str).str.strip()
                    != df_plot["perturbed_text"].astype(str).str.strip()
                ]
        if len(df_plot) == 0:
            df_plot = self.df_raw

        p1 = os.path.join(plots_dir, "rq1_spearman_distribution.png")
        plot_rq1_metric_distribution(
            df_plot,
            metric="spearman",
            metric_label=r"Korelacja rang Spearmana ($\rho$)",
            save_path=p1
        )

        p2 = os.path.join(plots_dir, "rq1_kendall_distribution.png")
        plot_rq1_metric_distribution(
            df_plot,
            metric="kendall",
            metric_label=r"Współczynnik $\tau$ Kendalla",
            save_path=p2
        )

        p3 = os.path.join(plots_dir, "rq1_jaccard_curves.png")
        plot_rq1_jaccard_k_curves(df_plot, save_path=p3)

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
        plt.close("all")

        print(f"\n[RQ1] Wyniki zapisano pomyślnie w katalogu '{output_dir}':")
        print(f" - Surowe dane: {raw_csv}")
        print(f" - Podsumowanie: {summary_csv}")
        if preservation_csv:
            print(f" - Preservation Rate: {preservation_csv}")
        if strength_csv:
            print(f" - Siła perturbacji: {strength_csv}")
        print(f" - Tabela LaTeX: {latex_path}")
        print(f" - Wykresy (PNG/PDF): {plots_dir}")

        return {
            "raw_csv": raw_csv,
            "summary_csv": summary_csv,
            "decisions_csv": decisions_csv,
            "preservation_summary_csv": preservation_csv,
            "strength_analysis_csv": strength_csv,
            "latex_table": latex_path,
            "plots_dir": plots_dir
        }
