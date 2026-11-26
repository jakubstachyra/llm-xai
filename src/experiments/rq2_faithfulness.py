import os
import json
import time
import inspect
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare, spearmanr, kendalltau, pearsonr
from typing import List, Dict, Any, Optional, Literal, Tuple
from tqdm import tqdm
from src.models import SentimentCausalLM
from src.hardware_tracker import HardwareSessionTracker
from src.explainers import (
    BaseExplainer,
    OcclusionExplainer,
    GradientXInputExplainer,
    IntegratedGradientsExplainer,
    LIMEExplainer,
    SHAPExplainer,
)
from src.explainers.base import AttributionResult
from src.faithfulness import (
    WordUnit,
    extract_word_units,
    aggregate_word_attributions,
    rank_word_units,
    apply_word_deletion,
    apply_word_retention,
    apply_embedding_mask,
    calculate_aopc,
    compute_additive_gap,
    evaluate_faithfulness_curves,
)
from src.experiments.visualizations import (
    plot_rq2_comprehensiveness_curves,
    plot_rq2_morf_vs_lerf_and_sufficiency,
    plot_rq2_aopc_comparison,
    plot_rq2_operator_robustness,
)


def _holm_bonferroni(p_values: List[float]) -> List[float]:
    """Applies Holm-Bonferroni step-down FWER adjustment to a list of p-values."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: (1.0 if np.isnan(x[1]) else x[1]))
    adjusted = [np.nan] * n
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        if np.isnan(p_val):
            adjusted[orig_idx] = np.nan
            continue
        scaled = min(1.0, float(p_val) * (n - rank))
        running_max = max(running_max, scaled)
        adjusted[orig_idx] = running_max
    return adjusted


class RQ2FaithfulnessExperiment:
    """
    Research Question 2 (RQ2) Experiment Runner:
    Evaluates the prediction faithfulness of token/word attributions across XAI methods.
    
    Key Features:
    - Tokenizer-agnostic WordUnit abstraction with signed subtoken aggregation.
    - Two perturbation operators: Discrete Deletion and Vector Embedding Masking
      (plus 'both' mode which computes expensive XAI attributions ONCE per sample and
      evaluates both operators simultaneously, cutting dual-operator runtime in half).
    - Metrics: Comprehensiveness C(k), Sufficiency U(k), AOPC, Normalized Gap G, and Retention Harmonization H.
    - Incremental checkpointing flushed to CSV every 10 samples with automatic resume.
    - Publication-grade LaTeX tables and vector graphics (PDF/PNG).
    """

    def __init__(
        self,
        model: Optional[SentimentCausalLM] = None,
        model_name: str = "microsoft/phi-2",
        explainers: Optional[List[BaseExplainer]] = None,
        max_evals_shap: int = 500,
        ig_steps: int = 20,
        lime_samples: int = 150,
        operator: Literal["deletion", "embedding_mask", "both"] = "deletion",
        max_k: int = 5,
        random_trials: int = 20,
        stochastic_num_seeds: int = 3,
        data_path: str = "data/sst2_validation.json",
        output_dir: str = "results/rq2_faithfulness"
    ):
        self.output_dir = output_dir
        self.stochastic_num_seeds = max(1, int(stochastic_num_seeds))
        self.operator = operator
        if operator == "both":
            base_dir = output_dir[:-5] if output_dir.endswith("_both") else output_dir
            self.base_output_dir = base_dir
            self.active_operators: List[str] = ["deletion", "embedding_mask"]
            self.op_dirs: Dict[str, str] = {
                "deletion": f"{base_dir}_deletion",
                "embedding_mask": f"{base_dir}_embedding_mask",
            }
        else:
            self.base_output_dir = output_dir
            self.active_operators = [operator]
            self.op_dirs = {operator: output_dir}

        for d in self.op_dirs.values():
            os.makedirs(d, exist_ok=True)

        if model is None:
            print(f"[RQ2] Loading model '{model_name}'...")
            self.model = SentimentCausalLM(model_name=model_name)
        else:
            self.model = model

        if explainers is None:
            self.explainers = [
                OcclusionExplainer(self.model),
                GradientXInputExplainer(self.model),
                IntegratedGradientsExplainer(self.model, steps=ig_steps),
                LIMEExplainer(self.model, num_samples=lime_samples, seed=42),
                SHAPExplainer(self.model, max_evals=max_evals_shap),
            ]
        else:
            self.explainers = explainers

        self.max_k = max_k
        self.random_trials = random_trials
        self.data_path = data_path
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_summary: Optional[pd.DataFrame] = None
        self.df_raw_by_op: Dict[str, pd.DataFrame] = {}
        self.df_summary_by_op: Dict[str, pd.DataFrame] = {}

    def load_dataset(self, num_samples: int = 400, seed: int = 42) -> List[Dict[str, Any]]:
        """Loads stratified SST-2 validation samples."""
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            from datasets import load_dataset
            print(f"[RQ2] Downloading SST-2 validation split...")
            ds = load_dataset("glue", "sst2", split="validation")
            data = [{"id": i, "sentence": row["sentence"], "label": row["label"]} for i, row in enumerate(ds)]

        # Deterministic prefix-preserving permutation (identical sample_id order to RQ1 up to full dataset 872)
        np.random.seed(seed)
        indices = np.random.permutation(len(data))[:min(num_samples, len(data))]
        data = [
            {"id": sample_idx, "orig_dataset_id": int(data[i].get("id", i)), "sentence": str(data[i]["sentence"]).strip(), "label": int(data[i]["label"])}
            for sample_idx, i in enumerate(indices)
        ]

        return data

    def _eval_ranked_units_for_operator(
        self,
        op: str,
        text: str,
        units: List[WordUnit],
        ranked_units: List[WordUnit],
        k_list: List[int],
        c_star: int,
        p_star: float,
        f_star: float,
        base_embeds: Optional[torch.Tensor] = None,
        least_ranked_units: Optional[List[WordUnit]] = None,
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
        """Evaluates Comprehensiveness C(k), Sufficiency U(k), and optional LeRF C_least(k) curves."""
        c_prob_steps: List[float] = []
        c_log_steps: List[float] = []
        u_prob_steps: List[float] = []
        u_log_steps: List[float] = []
        c_least_prob_steps: List[float] = []
        c_least_log_steps: List[float] = []
        stride = 3 if least_ranked_units is not None else 2
        sub_bs = max(1, int(getattr(self.model, "default_sub_batch_size", 64)))

        if op == "deletion":
            # Batch all deletion, retention, and optional least-important deletion prompts into one GPU call
            batch_texts: List[str] = []
            for k in k_list:
                batch_texts.append(apply_word_deletion(text, ranked_units[:k]))
                batch_texts.append(apply_word_retention(text, units, ranked_units[:k]))
                if least_ranked_units is not None:
                    batch_texts.append(apply_word_deletion(text, least_ranked_units[:k]))
            probs_batch = self.model.predict_texts_batch(batch_texts)
            for idx_k in range(len(k_list)):
                base = stride * idx_k
                p_del_c = float(probs_batch[base, c_star])
                p_ret_c = float(probs_batch[base + 1, c_star])
                c_prob_steps.append(p_star - p_del_c)
                c_log_steps.append(f_star - float(np.log(max(p_del_c, 1e-12))))
                u_prob_steps.append(p_star - p_ret_c)
                u_log_steps.append(f_star - float(np.log(max(p_ret_c, 1e-12))))
                if least_ranked_units is not None:
                    p_least_c = float(probs_batch[base + 2, c_star])
                    c_least_prob_steps.append(p_star - p_least_c)
                    c_least_log_steps.append(f_star - float(np.log(max(p_least_c, 1e-12))))
        else:
            # Embedding Masking operator (batched across all k in k_list since seq_len is identical)
            if base_embeds is None:
                input_ids = self.model.encode_prompt(text, return_tensors="pt")
                with torch.inference_mode():
                    base_embeds = self.model.get_input_embeddings()(input_ids).detach()

            batch_embeds_list = []
            for k in k_list:
                del_embeds = apply_embedding_mask(base_embeds, ranked_units[:k], mask_value="zero")
                rem_units = [u for u in units if u not in ranked_units[:k]]
                ret_embeds = apply_embedding_mask(base_embeds, rem_units, mask_value="zero")
                batch_embeds_list.append(del_embeds)
                batch_embeds_list.append(ret_embeds)
                if least_ranked_units is not None:
                    least_embeds = apply_embedding_mask(base_embeds, least_ranked_units[:k], mask_value="zero")
                    batch_embeds_list.append(least_embeds)

            stacked_embeds = torch.cat(batch_embeds_list, dim=0)
            if stacked_embeds.shape[0] <= sub_bs:
                probs_batch = self.model.predict_embeddings(stacked_embeds).cpu().numpy()
            else:
                probs_chunks = []
                for chunk_start in range(0, stacked_embeds.shape[0], sub_bs):
                    chunk_e = stacked_embeds[chunk_start:chunk_start + sub_bs]
                    probs_chunks.append(self.model.predict_embeddings(chunk_e).cpu().numpy())
                probs_batch = np.vstack(probs_chunks)
            del stacked_embeds, batch_embeds_list

            for idx_k in range(len(k_list)):
                base = stride * idx_k
                p_del_c = float(probs_batch[base, c_star])
                p_ret_c = float(probs_batch[base + 1, c_star])
                c_prob_steps.append(p_star - p_del_c)
                c_log_steps.append(f_star - float(np.log(max(p_del_c, 1e-12))))
                u_prob_steps.append(p_star - p_ret_c)
                u_log_steps.append(f_star - float(np.log(max(p_ret_c, 1e-12))))
                if least_ranked_units is not None:
                    p_least_c = float(probs_batch[base + 2, c_star])
                    c_least_prob_steps.append(p_star - p_least_c)
                    c_least_log_steps.append(f_star - float(np.log(max(p_least_c, 1e-12))))

        return (
            c_prob_steps,
            c_log_steps,
            u_prob_steps,
            u_log_steps,
            c_least_prob_steps,
            c_least_log_steps,
        )

    def _is_cache_compatible(
        self,
        explainer: BaseExplainer,
        cached_obj: Optional[Dict[str, Any]],
        overwrite_methods: Optional[List[str]] = None,
    ) -> bool:
        """Checks whether a cached attribution record matches the current model and explainer config."""
        if not cached_obj:
            return False
        if overwrite_methods and explainer.explainer_name in overwrite_methods:
            return False
        cached_model = cached_obj.get("model_name")
        if cached_model is not None and str(cached_model) != str(self.model.model_name):
            return False
        tokens = cached_obj.get("tokens")
        attrs = cached_obj.get("attributions")
        if not isinstance(tokens, list) or not isinstance(attrs, list) or len(tokens) == 0 or len(tokens) != len(attrs):
            return False

        meta = cached_obj.get("metadata") or {}
        if bool(meta.get("calibrated", False)) != bool(getattr(self.model, "calibrate_logits", False)):
            return False
        if isinstance(explainer, IntegratedGradientsExplainer) or (hasattr(explainer, "steps") and hasattr(explainer, "baseline_type")):
            if "steps" in meta:
                if int(meta["steps"]) != int(explainer.steps):
                    return False
            else:
                cname = str(cached_obj.get("explainer_name", ""))
                if f"steps={explainer.steps}" not in cname:
                    return False
            if "baseline_type" in meta:
                if str(meta["baseline_type"]) != str(explainer.baseline_type):
                    return False
            elif str(explainer.baseline_type) != "zero":
                return False
        elif isinstance(explainer, LIMEExplainer) or hasattr(explainer, "num_samples"):
            if "num_samples" in meta and int(meta["num_samples"]) != int(explainer.num_samples):
                return False
        elif isinstance(explainer, SHAPExplainer) or hasattr(explainer, "max_evals"):
            if "max_evals" in meta:
                if int(meta["max_evals"]) != int(explainer.max_evals):
                    return False
            else:
                cname = str(cached_obj.get("explainer_name", ""))
                if "max_evals=" in cname and f"max_evals={explainer.max_evals}" not in cname:
                    return False
        return True

    def run(
        self,
        num_samples: int = 400,
        seed: int = 42,
        resume: bool = True,
        max_run_samples: Optional[int] = None,
        overwrite_methods: Optional[List[str]] = None,
        rq1_attributions_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Executes the RQ2 evaluation suite with atomic checkpointing.
        When operator='both', computes each XAI explanation ONCE per sample and evaluates
        both 'deletion' and 'embedding_mask' operators in the same pass.
        Can also reuse original explanations precomputed in RQ1 (rq1_attributions.jsonl).
        """
        dataset = self.load_dataset(num_samples=num_samples, seed=seed)

        # Load cached original explanations from RQ1 or previous RQ2 runs if available
        cached_orig_attrs: Dict[Tuple[int, str], Dict[str, Any]] = {}
        model_tag = self.model.model_name.split("/")[-1]
        parent_of_base = os.path.dirname(os.path.abspath(self.base_output_dir))
        candidate_attr_paths: List[str] = []
        if rq1_attributions_path:
            candidate_attr_paths.append(rq1_attributions_path)
        for op in self.active_operators:
            candidate_attr_paths.append(os.path.join(self.op_dirs[op], "rq2_attributions.jsonl"))
        candidate_attr_paths.append(os.path.join(parent_of_base, f"rq1_stability_{model_tag}", "rq1_attributions.jsonl"))
        candidate_attr_paths.append(os.path.join("results", f"rq1_stability_{model_tag}", "rq1_attributions.jsonl"))
        if model_tag.lower() in ["phi-2", "phi2"]:
            candidate_attr_paths.append(os.path.join("results", "rq1_stability", "rq1_attributions.jsonl"))

        seen_attr_paths = set()
        for apath in candidate_attr_paths:
            if not apath:
                continue
            abs_apath = os.path.abspath(apath)
            if abs_apath in seen_attr_paths or not os.path.exists(abs_apath):
                continue
            seen_attr_paths.add(abs_apath)
            try:
                with open(abs_apath, "r", encoding="utf-8") as fa:
                    for line in fa:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        if obj.get("variant", "__orig__") == "__orig__":
                            sid = int(obj["sample_id"])
                            cname = str(obj.get("Canonical_name") or obj.get("explainer_name"))
                            ename = str(obj.get("explainer_name"))
                            if (sid, cname) not in cached_orig_attrs:
                                cached_orig_attrs[(sid, cname)] = obj
                            if (sid, ename) not in cached_orig_attrs:
                                cached_orig_attrs[(sid, ename)] = obj
            except Exception:
                pass

        raw_csv_paths: Dict[str, str] = {
            op: os.path.join(self.op_dirs[op], "rq2_faithfulness_raw.csv")
            for op in self.active_operators
        }
        completed_keys: Dict[str, set] = {op: set() for op in self.active_operators}
        existing_records: Dict[str, List[Dict[str, Any]]] = {op: [] for op in self.active_operators}
        unflushed_records: Dict[str, List[Dict[str, Any]]] = {op: [] for op in self.active_operators}
        rand_aopc_lookup: Dict[str, Dict[int, float]] = {op: {} for op in self.active_operators}
        unflushed_attr_records: List[Dict[str, Any]] = []
        rq2_attr_path = os.path.join(self.op_dirs[self.active_operators[0]], "rq2_attributions.jsonl")

        for op in self.active_operators:
            raw_csv_path = raw_csv_paths[op]
            if resume and os.path.exists(raw_csv_path):
                try:
                    df_existing = pd.read_csv(raw_csv_path)
                    if overwrite_methods and len(df_existing) > 0 and "explainer_name" in df_existing.columns:
                        before_len = len(df_existing)
                        df_existing = df_existing[~df_existing["explainer_name"].isin(overwrite_methods)].copy()
                        df_existing.to_csv(raw_csv_path, index=False)
                        print(f"[RQ2 Checkpoint ({op})] Removed {before_len - len(df_existing)} old rows for overwritten methods: {overwrite_methods}")
                    existing_records[op] = df_existing.to_dict("records")
                    for r in existing_records[op]:
                        sid = int(r["sample_id"])
                        ename = str(r["explainer_name"])
                        completed_keys[op].add((sid, ename, str(r["operator"])))
                        if ename == "Random Baseline":
                            rand_aopc_lookup[op][sid] = float(r["aopc_c_prob"])
                    print(f"[RQ2 Checkpoint ({op})] Loaded {len(existing_records[op])} existing records ({len(completed_keys[op])} unique evaluations).")
                except Exception as e:
                    print(f"[RQ2 Checkpoint Warning ({op})] Could not parse existing checkpoint: {e}. Starting fresh.")

        samples_processed_in_this_run = 0

        def flush_buffer():
            for op in self.active_operators:
                if not unflushed_records[op]:
                    continue
                raw_csv_path = raw_csv_paths[op]
                df_chunk = pd.DataFrame(unflushed_records[op])
                file_exists = os.path.exists(raw_csv_path) and os.path.getsize(raw_csv_path) > 0
                if not file_exists:
                    df_chunk.to_csv(raw_csv_path, index=False)
                else:
                    existing_cols = pd.read_csv(raw_csv_path, nrows=0).columns.tolist()
                    if existing_cols == list(df_chunk.columns):
                        df_chunk.to_csv(raw_csv_path, mode="a", header=False, index=False)
                    else:
                        df_old = pd.read_csv(raw_csv_path)
                        df_merged = pd.concat([df_old, df_chunk], ignore_index=True)
                        df_merged.to_csv(raw_csv_path, index=False)
                existing_records[op].extend(unflushed_records[op])
                unflushed_records[op] = []
            if unflushed_attr_records:
                with open(rq2_attr_path, "a", encoding="utf-8") as fa:
                    for rec in unflushed_attr_records:
                        fa.write(json.dumps(rec, ensure_ascii=False) + "\n")
                unflushed_attr_records.clear()

        print(f"\n=======================================================")
        print(f"  Starting RQ2 Faithfulness Benchmark (Operator: {self.operator})")
        print(f"  Active Operators: {self.active_operators}")
        print(f"  Samples: {len(dataset)} | Explainers: {[e.__class__.__name__ for e in self.explainers]}")
        for op in self.active_operators:
            print(f"  Checkpoint ({op}): {raw_csv_paths[op]}")
        print(f"=======================================================\n")

        tracker = HardwareSessionTracker(self.model, experiment_name=f"RQ2-Faithfulness-{self.operator}", num_samples=len(dataset))
        tracker.start()

        try:
            for sample_idx, sample in enumerate(tqdm(dataset, desc="Evaluating Faithfulness")):
                sample_id = int(sample.get("id", sample_idx))
                text = sample["sentence"]

                # Check if all explainers and Random Baseline are already completed across all active operators
                all_done = all(
                    all(
                        (sample_id, explainer.explainer_name, op) in completed_keys[op]
                        for explainer in self.explainers
                    ) and ((sample_id, "Random Baseline", op) in completed_keys[op])
                    for op in self.active_operators
                )

                if all_done:
                    continue

                if max_run_samples is not None and samples_processed_in_this_run >= max_run_samples:
                    print(f"\n[RQ2 Session] Reached session limit of {max_run_samples} samples. Stopping for this batch.")
                    break

                # 1. Base prediction
                pred_class, probs_orig = self.model.predict_single(text)
                true_label = int(sample.get("label", -1))
                c_star = pred_class
                p_star = float(probs_orig[c_star])
                f_star = float(np.log(max(p_star, 1e-12)))
                p_clip = np.clip(probs_orig, 1e-12, 1.0)
                pred_entropy = float(-np.sum(p_clip * np.log2(p_clip)))

                # 2. Extract WordUnits
                units = extract_word_units(text, self.model.tokenizer)
                if len(units) < 2:
                    continue

                eff_k = min(self.max_k, max(1, len(units) // 2))
                k_list = list(range(1, eff_k + 1))

                # Precompute base_embeds once if embedding_mask is active
                base_embeds: Optional[torch.Tensor] = None
                if "embedding_mask" in self.active_operators:
                    input_ids = self.model.encode_prompt(text, return_tensors="pt")
                    with torch.inference_mode():
                        base_embeds = self.model.get_input_embeddings()(input_ids).detach()

                # 3. Random Baseline (Monte Carlo over B permutations) per active operator
                rand_aopc_c_prob_by_op: Dict[str, float] = {}
                sub_bs = max(1, int(getattr(self.model, "default_sub_batch_size", 64)))
                for op in self.active_operators:
                    rand_key = (sample_id, "Random Baseline", op)
                    if rand_key not in completed_keys[op]:
                        c_prob_rand_trials = {k: [] for k in k_list}
                        c_log_rand_trials = {k: [] for k in k_list}
                        u_prob_rand_trials = {k: [] for k in k_list}
                        u_log_rand_trials = {k: [] for k in k_list}

                        if op == "deletion":
                            # Deduplicate identical random deletion/retention prompts before GPU inference
                            unique_texts: List[str] = []
                            text_to_idx: Dict[str, int] = {}
                            trial_index_pairs: List[Tuple[int, int, int]] = []
                            for b in range(self.random_trials):
                                rand_units = rank_word_units(units, ranking_mode="random", seed=seed + sample_idx * 100 + b)
                                for k in k_list:
                                    del_t = apply_word_deletion(text, rand_units[:k])
                                    ret_t = apply_word_retention(text, units, rand_units[:k])
                                    if del_t not in text_to_idx:
                                        text_to_idx[del_t] = len(unique_texts)
                                        unique_texts.append(del_t)
                                    if ret_t not in text_to_idx:
                                        text_to_idx[ret_t] = len(unique_texts)
                                        unique_texts.append(ret_t)
                                    trial_index_pairs.append((k, text_to_idx[del_t], text_to_idx[ret_t]))

                            probs_all = self.model.predict_texts_batch(unique_texts)
                            for k, del_idx, ret_idx in trial_index_pairs:
                                p_del_c = float(probs_all[del_idx, c_star])
                                p_ret_c = float(probs_all[ret_idx, c_star])
                                c_prob_rand_trials[k].append(p_star - p_del_c)
                                c_log_rand_trials[k].append(f_star - float(np.log(max(p_del_c, 1e-12))))
                                u_prob_rand_trials[k].append(p_star - p_ret_c)
                                u_log_rand_trials[k].append(f_star - float(np.log(max(p_ret_c, 1e-12))))
                        else:
                            # Deduplicate identical masked unit sets across random trials and batch adaptively
                            unique_embeds_list: List[torch.Tensor] = []
                            mask_key_to_idx: Dict[Tuple[int, ...], int] = {}
                            trial_index_pairs = []
                            for b in range(self.random_trials):
                                rand_units = rank_word_units(units, ranking_mode="random", seed=seed + sample_idx * 100 + b)
                                for k in k_list:
                                    top_k = rand_units[:k]
                                    del_key = tuple(sorted(u.index for u in top_k))
                                    if del_key not in mask_key_to_idx:
                                        mask_key_to_idx[del_key] = len(unique_embeds_list)
                                        unique_embeds_list.append(apply_embedding_mask(base_embeds, top_k, mask_value="zero"))

                                    rem_units = [u for u in units if u not in top_k]
                                    ret_key = tuple(sorted(u.index for u in rem_units))
                                    if ret_key not in mask_key_to_idx:
                                        mask_key_to_idx[ret_key] = len(unique_embeds_list)
                                        unique_embeds_list.append(apply_embedding_mask(base_embeds, rem_units, mask_value="zero"))

                                    trial_index_pairs.append((k, mask_key_to_idx[del_key], mask_key_to_idx[ret_key]))

                            stacked_rand_embeds = torch.cat(unique_embeds_list, dim=0)
                            probs_chunks = []
                            for chunk_start in range(0, stacked_rand_embeds.shape[0], sub_bs):
                                chunk_e = stacked_rand_embeds[chunk_start:chunk_start + sub_bs]
                                probs_chunks.append(self.model.predict_embeddings(chunk_e).cpu().numpy())
                            probs_all = np.vstack(probs_chunks)
                            del stacked_rand_embeds, unique_embeds_list

                            for k, del_idx, ret_idx in trial_index_pairs:
                                p_del_c = float(probs_all[del_idx, c_star])
                                p_ret_c = float(probs_all[ret_idx, c_star])
                                c_prob_rand_trials[k].append(p_star - p_del_c)
                                c_log_rand_trials[k].append(f_star - float(np.log(max(p_del_c, 1e-12))))
                                u_prob_rand_trials[k].append(p_star - p_ret_c)
                                u_log_rand_trials[k].append(f_star - float(np.log(max(p_ret_c, 1e-12))))

                        c_prob_rand = [float(np.mean(c_prob_rand_trials[k])) for k in k_list]
                        c_log_rand = [float(np.mean(c_log_rand_trials[k])) for k in k_list]
                        u_prob_rand = [float(np.mean(u_prob_rand_trials[k])) for k in k_list]
                        u_log_rand = [float(np.mean(u_log_rand_trials[k])) for k in k_list]

                        rand_result = evaluate_faithfulness_curves(
                            k_values=k_list,
                            c_prob=c_prob_rand,
                            c_log=c_log_rand,
                            u_prob=u_prob_rand,
                            u_log=u_log_rand,
                            random_aopc_c_prob=None
                        )

                        rand_row = {
                            "sample_id": sample_id,
                            "device_name": getattr(self.model, "device_name", "Unknown"),
                            "explainer_name": "Random Baseline",
                            "operator": op,
                            "original_text": text,
                            "true_label": true_label,
                            "target_class": c_star,
                            "is_correct": bool(c_star == true_label),
                            "baseline_prob": p_star,
                            "prediction_entropy": pred_entropy,
                            "num_words": len(units),
                            "max_k": eff_k,
                            "aopc_c_prob": rand_result.aopc_comprehensiveness_prob,
                            "aopc_c_log": rand_result.aopc_comprehensiveness_log,
                            "aopc_u_prob": rand_result.aopc_sufficiency_prob,
                            "aopc_u_log": rand_result.aopc_sufficiency_log,
                            "aopc_least_prob": np.nan,
                            "aopc_least_log": np.nan,
                            "gap_g": 0.0,
                            "gap_g_rel": 0.0,
                            "gap_g_least": np.nan,
                            "harmonization_h": rand_result.harmonization_h,
                            "execution_time": 0.0,
                            "forward_passes": 0,
                            "backward_passes": 0,
                        }
                        for i, k in enumerate(k_list):
                            rand_row[f"c_prob_k{k}"] = c_prob_rand[i]
                            rand_row[f"u_prob_k{k}"] = u_prob_rand[i]
                            rand_row[f"c_least_prob_k{k}"] = np.nan
                            rand_row[f"c_log_k{k}"] = c_log_rand[i]
                            rand_row[f"u_log_k{k}"] = u_log_rand[i]
                            rand_row[f"c_least_log_k{k}"] = np.nan

                        unflushed_records[op].append(rand_row)
                        completed_keys[op].add(rand_key)
                        rand_aopc_lookup[op][sample_id] = rand_result.aopc_comprehensiveness_prob
                        rand_aopc_c_prob_by_op[op] = rand_result.aopc_comprehensiveness_prob
                    else:
                        rand_aopc_c_prob_by_op[op] = float(rand_aopc_lookup[op].get(sample_id, 0.0))

                # 4. Evaluate each XAI explainer ONCE per sample (or reuse from RQ1 cache) and apply to all needed operators
                for explainer in self.explainers:
                    method_name = explainer.explainer_name
                    ops_needed = [
                        op for op in self.active_operators
                        if (sample_id, method_name, op) not in completed_keys[op]
                    ]
                    if not ops_needed:
                        continue

                    is_stoch = bool(getattr(explainer, "stochastic", False))
                    n_seeds = self.stochastic_num_seeds if is_stoch else 1
                    base_exp_seed = int(getattr(explainer, "seed", seed))

                    if n_seeds == 1:
                        cached = cached_orig_attrs.get((sample_id, method_name))
                        if (
                            cached is not None
                            and cached.get("text", "").strip() == text
                            and int(cached.get("target_class", -1)) == c_star
                            and self._is_cache_compatible(explainer, cached, overwrite_methods=overwrite_methods)
                        ):
                            res = AttributionResult(
                                tokens=list(cached["tokens"]),
                                attributions=np.asarray(cached["attributions"], dtype=float),
                                target_class=c_star,
                                execution_time=float(cached.get("execution_time", 0.0)),
                                explainer_name=method_name,
                                baseline_prob=p_star,
                                metadata=cached.get("metadata", {}),
                            )
                            exp_fwd = int(cached.get("forward_passes", 0))
                            exp_bwd = int(cached.get("backward_passes", 0))
                        else:
                            exp_kwargs: Dict[str, Any] = {}
                            if "baseline_prob" in inspect.signature(explainer.explain).parameters:
                                exp_kwargs["baseline_prob"] = p_star
                            c0 = self.model.get_counters()
                            res = explainer.explain(text, target_class=c_star, **exp_kwargs)
                            c1 = self.model.get_counters()
                            exp_fwd = c1["forward_passes"] - c0["forward_passes"]
                            exp_bwd = c1["backward_passes"] - c0["backward_passes"]

                            rec_meta = dict(res.metadata or {})
                            rec_meta["calibrated"] = bool(getattr(self.model, "calibrate_logits", False))
                            if isinstance(explainer, IntegratedGradientsExplainer):
                                rec_meta.setdefault("steps", explainer.steps)
                                rec_meta.setdefault("baseline_type", explainer.baseline_type)
                            elif isinstance(explainer, LIMEExplainer):
                                rec_meta.setdefault("num_samples", explainer.num_samples)
                            elif isinstance(explainer, SHAPExplainer):
                                rec_meta.setdefault("max_evals", explainer.max_evals)

                            attr_rec = {
                                "sample_id": sample_id,
                                "model_name": self.model.model_name,
                                "explainer_name": method_name,
                                "Canonical_name": method_name,
                                "variant": "__orig__",
                                "text": text,
                                "target_class": int(c_star),
                                "baseline_prob": float(p_star),
                                "execution_time": float(res.execution_time),
                                "forward_passes": int(exp_fwd),
                                "backward_passes": int(exp_bwd),
                                "tokens": list(res.tokens),
                                "attributions": [round(float(x), 6) for x in res.attributions],
                                "metadata": rec_meta,
                            }
                            unflushed_attr_records.append(attr_rec)
                            cached_orig_attrs[(sample_id, method_name)] = attr_rec

                        aggregate_word_attributions(units, res.attributions, method="signed_sum")
                        ranked_units = rank_word_units(units, ranking_mode="top_signed")
                        least_units = rank_word_units(units, ranking_mode="near_zero")

                        for op in ops_needed:
                            (
                                c_prob_steps,
                                c_log_steps,
                                u_prob_steps,
                                u_log_steps,
                                c_least_prob_steps,
                                c_least_log_steps,
                            ) = self._eval_ranked_units_for_operator(
                                op=op,
                                text=text,
                                units=units,
                                ranked_units=ranked_units,
                                k_list=k_list,
                                c_star=c_star,
                                p_star=p_star,
                                f_star=f_star,
                                base_embeds=base_embeds,
                                least_ranked_units=least_units,
                            )

                            faith_res = evaluate_faithfulness_curves(
                                k_values=k_list,
                                c_prob=c_prob_steps,
                                c_log=c_log_steps,
                                u_prob=u_prob_steps,
                                u_log=u_log_steps,
                                random_aopc_c_prob=rand_aopc_c_prob_by_op[op]
                            )
                            additive_g = compute_additive_gap(
                                faith_res.aopc_comprehensiveness_prob,
                                rand_aopc_c_prob_by_op[op]
                            )
                            aopc_least_prob = calculate_aopc(c_least_prob_steps)
                            aopc_least_log = calculate_aopc(c_least_log_steps)
                            gap_g_least = float(faith_res.aopc_comprehensiveness_prob - aopc_least_prob)

                            row = {
                                "sample_id": sample_id,
                                "device_name": getattr(self.model, "device_name", "Unknown"),
                                "explainer_name": method_name,
                                "operator": op,
                                "original_text": text,
                                "true_label": true_label,
                                "target_class": c_star,
                                "is_correct": bool(c_star == true_label),
                                "baseline_prob": p_star,
                                "prediction_entropy": pred_entropy,
                                "num_words": len(units),
                                "max_k": eff_k,
                                "n_seeds": 1,
                                "aopc_c_prob": faith_res.aopc_comprehensiveness_prob,
                                "aopc_c_log": faith_res.aopc_comprehensiveness_log,
                                "aopc_u_prob": faith_res.aopc_sufficiency_prob,
                                "aopc_u_log": faith_res.aopc_sufficiency_log,
                                "aopc_least_prob": aopc_least_prob,
                                "aopc_least_log": aopc_least_log,
                                "gap_g": additive_g,
                                "gap_g_rel": faith_res.relative_gap_g,
                                "gap_g_least": gap_g_least,
                                "harmonization_h": faith_res.harmonization_h,
                                "execution_time": res.execution_time,
                                "forward_passes": exp_fwd,
                                "backward_passes": exp_bwd,
                            }
                            for i, k in enumerate(k_list):
                                row[f"c_prob_k{k}"] = c_prob_steps[i]
                                row[f"u_prob_k{k}"] = u_prob_steps[i]
                                row[f"c_least_prob_k{k}"] = c_least_prob_steps[i]
                                row[f"c_log_k{k}"] = c_log_steps[i]
                                row[f"u_log_k{k}"] = u_log_steps[i]
                                row[f"c_least_log_k{k}"] = c_least_log_steps[i]

                            unflushed_records[op].append(row)
                            completed_keys[op].add((sample_id, method_name, op))
                    else:
                        # Multi-seed averaging for stochastic explainers (R independent seeds per review)
                        seed_rows_by_op: Dict[str, List[Dict[str, Any]]] = {op: [] for op in ops_needed}
                        has_seed_arg = "seed" in inspect.signature(explainer.explain).parameters
                        has_bprob_arg = "baseline_prob" in inspect.signature(explainer.explain).parameters

                        for r_idx in range(n_seeds):
                            run_seed = base_exp_seed + r_idx
                            exp_kwargs = {}
                            if has_bprob_arg:
                                exp_kwargs["baseline_prob"] = p_star
                            if has_seed_arg:
                                exp_kwargs["seed"] = run_seed

                            c0 = self.model.get_counters()
                            res = explainer.explain(text, target_class=c_star, **exp_kwargs)
                            c1 = self.model.get_counters()
                            exp_fwd = c1["forward_passes"] - c0["forward_passes"]
                            exp_bwd = c1["backward_passes"] - c0["backward_passes"]

                            aggregate_word_attributions(units, res.attributions, method="signed_sum")
                            ranked_units = rank_word_units(units, ranking_mode="top_signed")
                            least_units = rank_word_units(units, ranking_mode="near_zero")

                            for op in ops_needed:
                                (
                                    c_prob_steps,
                                    c_log_steps,
                                    u_prob_steps,
                                    u_log_steps,
                                    c_least_prob_steps,
                                    c_least_log_steps,
                                ) = self._eval_ranked_units_for_operator(
                                    op=op,
                                    text=text,
                                    units=units,
                                    ranked_units=ranked_units,
                                    k_list=k_list,
                                    c_star=c_star,
                                    p_star=p_star,
                                    f_star=f_star,
                                    base_embeds=base_embeds,
                                    least_ranked_units=least_units,
                                )
                                faith_res = evaluate_faithfulness_curves(
                                    k_values=k_list,
                                    c_prob=c_prob_steps,
                                    c_log=c_log_steps,
                                    u_prob=u_prob_steps,
                                    u_log=u_log_steps,
                                    random_aopc_c_prob=rand_aopc_c_prob_by_op[op],
                                )
                                additive_g = compute_additive_gap(
                                    faith_res.aopc_comprehensiveness_prob,
                                    rand_aopc_c_prob_by_op[op],
                                )
                                aopc_least_prob = calculate_aopc(c_least_prob_steps)
                                aopc_least_log = calculate_aopc(c_least_log_steps)
                                gap_g_least = float(faith_res.aopc_comprehensiveness_prob - aopc_least_prob)

                                s_metrics: Dict[str, float] = {
                                    "aopc_c_prob": float(faith_res.aopc_comprehensiveness_prob),
                                    "aopc_c_log": float(faith_res.aopc_comprehensiveness_log),
                                    "aopc_u_prob": float(faith_res.aopc_sufficiency_prob),
                                    "aopc_u_log": float(faith_res.aopc_sufficiency_log),
                                    "aopc_least_prob": float(aopc_least_prob),
                                    "aopc_least_log": float(aopc_least_log),
                                    "gap_g": float(additive_g),
                                    "gap_g_rel": float(faith_res.relative_gap_g),
                                    "gap_g_least": float(gap_g_least),
                                    "harmonization_h": float(faith_res.harmonization_h),
                                    "execution_time": float(res.execution_time),
                                    "forward_passes": float(exp_fwd),
                                    "backward_passes": float(exp_bwd),
                                }
                                for i, k in enumerate(k_list):
                                    s_metrics[f"c_prob_k{k}"] = float(c_prob_steps[i])
                                    s_metrics[f"u_prob_k{k}"] = float(u_prob_steps[i])
                                    s_metrics[f"c_least_prob_k{k}"] = float(c_least_prob_steps[i])
                                    s_metrics[f"c_log_k{k}"] = float(c_log_steps[i])
                                    s_metrics[f"u_log_k{k}"] = float(u_log_steps[i])
                                    s_metrics[f"c_least_log_k{k}"] = float(c_least_log_steps[i])
                                seed_rows_by_op[op].append(s_metrics)

                        for op in ops_needed:
                            s_list = seed_rows_by_op[op]
                            avg_metrics = {
                                key: float(np.mean([sr[key] for sr in s_list]))
                                for key in s_list[0].keys()
                            }
                            row = {
                                "sample_id": sample_id,
                                "device_name": getattr(self.model, "device_name", "Unknown"),
                                "explainer_name": method_name,
                                "operator": op,
                                "original_text": text,
                                "true_label": true_label,
                                "target_class": c_star,
                                "is_correct": bool(c_star == true_label),
                                "baseline_prob": p_star,
                                "prediction_entropy": pred_entropy,
                                "num_words": len(units),
                                "max_k": eff_k,
                                "n_seeds": n_seeds,
                                **avg_metrics,
                            }
                            unflushed_records[op].append(row)
                            completed_keys[op].add((sample_id, method_name, op))

                del base_embeds
                samples_processed_in_this_run += 1
                if samples_processed_in_this_run % 10 == 0:
                    flush_buffer()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for op in self.active_operators:
                        op_abs = os.path.abspath(self.op_dirs[op])
                        live_hw_path = os.path.join(op_abs, "hardware_live_snapshot.json")
                        live_hw_csv = os.path.join(op_abs, "hardware_timeline.csv")
                        try:
                            snap = tracker.snapshot_stats(samples_processed_in_this_run)
                            with open(live_hw_path, "w", encoding="utf-8") as f_hw:
                                json.dump(snap, f_hw, indent=2)
                            df_snap = pd.DataFrame([snap])
                            write_hw_hdr = not os.path.exists(live_hw_csv) or os.path.getsize(live_hw_csv) == 0
                            df_snap.to_csv(live_hw_csv, mode="a", header=write_hw_hdr, index=False)
                        except Exception:
                            pass

        except KeyboardInterrupt:
            print("\n[RQ2] Interrupted by user! Flushing pending checkpoints to disk...")
            flush_buffer()
            print("[RQ2 Checkpoint] Safely saved. You can resume at any time.")

        flush_buffer()

        for op in self.active_operators:
            self.df_raw_by_op[op] = pd.DataFrame(existing_records[op])
            self.df_raw = self.df_raw_by_op[op]
            self.output_dir = self.op_dirs[op]
            self.df_summary_by_op[op] = self.generate_summary(operator_label=op)
            self.generate_statistical_tests()
            self.generate_latex_table()
            self.generate_plots()

        if len(self.active_operators) >= 2:
            self.generate_operator_robustness_report()

        stats = tracker.stop(num_samples=samples_processed_in_this_run)
        extra_hw_paths = []
        for op in self.active_operators:
            out_abs = os.path.abspath(self.op_dirs[op])
            extra_hw_paths.append(os.path.join(out_abs, "hardware_session_log.csv"))
            parent_log = os.path.join(os.path.dirname(out_abs), "hardware_session_log.csv")
            if parent_log not in extra_hw_paths:
                extra_hw_paths.append(parent_log)
        tracker.log_session(stats=stats, extra_csv_paths=extra_hw_paths)
        tracker.print_summary(stats=stats)

        return self.df_raw

    def generate_summary(self, operator_label: Optional[str] = None) -> pd.DataFrame:
        """Aggregates raw results into mean, std, sem, and bootstrap 95% CI per explainer."""
        if self.df_raw is None or len(self.df_raw) == 0:
            return pd.DataFrame()

        op_str = operator_label if operator_label is not None else self.operator
        rand_sub = self.df_raw[self.df_raw["explainer_name"] == "Random Baseline"]
        rand_mean_aopc = float(rand_sub["aopc_c_prob"].mean()) if len(rand_sub) > 0 else 0.0
        rng = np.random.default_rng(42)

        has_least = "aopc_least_prob" in self.df_raw.columns
        has_gap_least = "gap_g_least" in self.df_raw.columns

        summary_rows = []
        for method, group in self.df_raw.groupby("explainer_name"):
            c_vals = group["aopc_c_prob"].dropna().to_numpy()
            n_c = len(c_vals)
            if n_c >= 2:
                boots = [float(np.mean(rng.choice(c_vals, size=n_c, replace=True))) for _ in range(1000)]
                ci_low, ci_high = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
            else:
                ci_low = ci_high = float(np.mean(c_vals)) if n_c > 0 else 0.0
            mean_c = float(group["aopc_c_prob"].mean())
            ratio_g_rel = ((mean_c / rand_mean_aopc) - 1.0) if (rand_mean_aopc > 1e-6 and method != "Random Baseline") else 0.0
            aopc_least_mean = float(group["aopc_least_prob"].mean()) if (has_least and group["aopc_least_prob"].notna().any()) else np.nan
            aopc_least_sem = float(group["aopc_least_prob"].sem()) if (has_least and group["aopc_least_prob"].notna().sum() >= 2) else np.nan
            gap_least_mean = float(group["gap_g_least"].mean()) if (has_gap_least and group["gap_g_least"].notna().any()) else np.nan
            gap_least_sem = float(group["gap_g_least"].sem()) if (has_gap_least and group["gap_g_least"].notna().sum() >= 2) else np.nan
            summary_rows.append({
                "explainer_name": method,
                "operator": op_str,
                "count": len(group),
                "aopc_c_prob_mean": mean_c,
                "aopc_c_prob_std": group["aopc_c_prob"].std(),
                "aopc_c_prob_sem": group["aopc_c_prob"].sem(),
                "aopc_c_ci95_low": ci_low,
                "aopc_c_ci95_high": ci_high,
                "aopc_c_log_mean": group["aopc_c_log"].mean(),
                "aopc_u_prob_mean": group["aopc_u_prob"].mean(),
                "aopc_u_prob_std": group["aopc_u_prob"].std(),
                "aopc_u_prob_sem": group["aopc_u_prob"].sem(),
                "aopc_least_prob_mean": aopc_least_mean,
                "aopc_least_prob_sem": aopc_least_sem,
                "gap_g_mean": group["gap_g"].mean(),
                "gap_g_sem": group["gap_g"].sem(),
                "gap_g_rel_ratio": ratio_g_rel,
                "gap_g_least_mean": gap_least_mean,
                "gap_g_least_sem": gap_least_sem,
                "harmonization_h_mean": group["harmonization_h"].mean(),
                "time_mean": group["execution_time"].mean(),
            })

        self.df_summary = pd.DataFrame(summary_rows).sort_values("aopc_c_prob_mean", ascending=False)
        summary_csv = os.path.join(self.output_dir, "rq2_faithfulness_summary.csv")
        self.df_summary.to_csv(summary_csv, index=False)
        return self.df_summary

    def generate_statistical_tests(self) -> pd.DataFrame:
        """
        Runs paired Wilcoxon signed-rank tests (MoRF vs Random Baseline and MoRF vs Least-Important LeRF)
        with Holm-Bonferroni FWER adjustment, plus a Friedman test across XAI methods.
        """
        if self.df_raw is None or len(self.df_raw) == 0:
            return pd.DataFrame()

        rand_df = self.df_raw[self.df_raw["explainer_name"] == "Random Baseline"][["sample_id", "aopc_c_prob"]].rename(
            columns={"aopc_c_prob": "rand_aopc_c_prob"}
        )
        explainer_names = [m for m in self.df_raw["explainer_name"].unique() if m != "Random Baseline"]

        rows = []
        p_rand_list: List[float] = []
        p_least_list: List[float] = []

        for method in sorted(explainer_names):
            sub = self.df_raw[self.df_raw["explainer_name"] == method].copy()
            # 1. Paired comparison against Random Baseline
            n_rand = 0
            mean_diff_rand = np.nan
            w_rand_stat = np.nan
            p_rand_two = np.nan
            p_rand_greater = np.nan
            if len(rand_df) > 0:
                merged_rand = pd.merge(sub[["sample_id", "aopc_c_prob"]], rand_df, on="sample_id", how="inner").dropna()
                n_rand = len(merged_rand)
                if n_rand >= 1:
                    diff_r = merged_rand["aopc_c_prob"].to_numpy(dtype=float) - merged_rand["rand_aopc_c_prob"].to_numpy(dtype=float)
                    mean_diff_rand = float(np.mean(diff_r))
                    if n_rand >= 2 and np.any(np.abs(diff_r) > 1e-12):
                        w_rand_stat, p_rand_two = wilcoxon(diff_r, alternative="two-sided", zero_method="wilcox")
                        _, p_rand_greater = wilcoxon(diff_r, alternative="greater", zero_method="wilcox")
                        w_rand_stat, p_rand_two, p_rand_greater = float(w_rand_stat), float(p_rand_two), float(p_rand_greater)
                    else:
                        w_rand_stat, p_rand_two, p_rand_greater = 0.0, 1.0, 1.0

            # 2. Paired comparison against Least-Important (LeRF)
            n_least = 0
            mean_diff_least = np.nan
            w_least_stat = np.nan
            p_least_two = np.nan
            p_least_greater = np.nan
            if "aopc_least_prob" in sub.columns:
                sub_least = sub[["sample_id", "aopc_c_prob", "aopc_least_prob"]].dropna()
                n_least = len(sub_least)
                if n_least >= 1:
                    diff_l = sub_least["aopc_c_prob"].to_numpy(dtype=float) - sub_least["aopc_least_prob"].to_numpy(dtype=float)
                    mean_diff_least = float(np.mean(diff_l))
                    if n_least >= 2 and np.any(np.abs(diff_l) > 1e-12):
                        w_least_stat, p_least_two = wilcoxon(diff_l, alternative="two-sided", zero_method="wilcox")
                        _, p_least_greater = wilcoxon(diff_l, alternative="greater", zero_method="wilcox")
                        w_least_stat, p_least_two, p_least_greater = float(w_least_stat), float(p_least_two), float(p_least_greater)
                    else:
                        w_least_stat, p_least_two, p_least_greater = 0.0, 1.0, 1.0

            p_rand_list.append(p_rand_greater)
            p_least_list.append(p_least_greater)
            rows.append({
                "explainer_name": method,
                "n_vs_random": n_rand,
                "mean_diff_vs_random": mean_diff_rand,
                "wilcoxon_stat_vs_random": w_rand_stat,
                "p_vs_random_two_sided": p_rand_two,
                "p_vs_random_greater": p_rand_greater,
                "n_vs_least": n_least,
                "mean_diff_vs_least": mean_diff_least,
                "wilcoxon_stat_vs_least": w_least_stat,
                "p_vs_least_two_sided": p_least_two,
                "p_vs_least_greater": p_least_greater,
            })

        df_tests = pd.DataFrame(rows)
        if len(df_tests) > 0:
            df_tests["p_holm_vs_random"] = _holm_bonferroni(p_rand_list)
            df_tests["p_holm_vs_least"] = _holm_bonferroni(p_least_list)

        tests_csv = os.path.join(self.output_dir, "rq2_statistical_tests.csv")
        df_tests.to_csv(tests_csv, index=False)

        # Friedman omnibus test across explainers on shared sample_ids
        non_rand = self.df_raw[self.df_raw["explainer_name"] != "Random Baseline"]
        if len(explainer_names) >= 3:
            wide = non_rand.pivot_table(index="sample_id", columns="explainer_name", values="aopc_c_prob", aggfunc="mean").dropna()
            if len(wide) >= 5 and wide.shape[1] >= 3:
                chi2, p_friedman = friedmanchisquare(*[wide[col].to_numpy() for col in wide.columns])
                n_s, k_m = wide.shape
                kendall_w = float(chi2 / (n_s * (k_m - 1))) if (n_s > 0 and k_m > 1) else np.nan
                friedman_info = {
                    "n_samples": int(n_s),
                    "n_methods": int(k_m),
                    "methods": list(wide.columns),
                    "friedman_chi2": float(chi2),
                    "p_value": float(p_friedman),
                    "kendall_w": kendall_w,
                    "mean_ranks": wide.rank(axis=1, ascending=False).mean().to_dict(),
                }
                with open(os.path.join(self.output_dir, "rq2_friedman_test.json"), "w", encoding="utf-8") as ff:
                    json.dump(friedman_info, ff, indent=2)

        return df_tests

    def generate_latex_table(self) -> str:
        """Generates an academic publication-grade LaTeX table for thesis Chapter 4."""
        if self.df_summary is None or len(self.df_summary) == 0:
            return ""

        table_path = os.path.join(self.output_dir, "rq2_table.tex")
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\caption{Wyniki ewaluacji wierności wyjaśnień (RQ2) na zbiorze walidacyjnym SST-2.}",
            r"\label{tab:rq2_faithfulness_results}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"\textbf{Metoda XAI} & \textbf{$\mathrm{AOPC}_C \uparrow$} & \textbf{$\mathrm{AOPC}_U \downarrow$} & \textbf{$\mathrm{AOPC}_{\text{least}} \downarrow$} & \textbf{Luka $G \uparrow$} & \textbf{Harmonia $H \uparrow$} & \textbf{Czas [s]} \\",
            r"\midrule",
        ]

        for _, r in self.df_summary.iterrows():
            method = r["explainer_name"]
            c_val = f"{r['aopc_c_prob_mean']:.3f} \\pm {r['aopc_c_prob_sem']:.3f}"
            u_val = f"{r['aopc_u_prob_mean']:.3f}"
            least_mean = r.get("aopc_least_prob_mean", np.nan)
            least_val = f"{least_mean:.3f}" if (method != "Random Baseline" and not pd.isna(least_mean)) else "-"
            g_val = f"{r['gap_g_mean']:+.2f}" if method != "Random Baseline" else "0.00"
            h_val = f"{r['harmonization_h_mean']:.3f}"
            t_val = f"{r['time_mean']:.2f}" if r["time_mean"] > 0 else "-"

            if method == "Random Baseline":
                lines.append(r"\midrule")
                lines.append(f"{method} & ${c_val}$ & ${u_val}$ & {least_val} & {g_val} & ${h_val}$ & {t_val} \\\\")
            else:
                lines.append(f"{method} & ${c_val}$ & ${u_val}$ & ${least_val}$ & ${g_val}$ & ${h_val}$ & {t_val} \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        tex_content = "\n".join(lines)
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(tex_content)
        return tex_content

    def generate_plots(self):
        """Generates high-DPI PDF and PNG plots for the thesis."""
        if self.df_raw is None or len(self.df_raw) == 0:
            return

        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        plot_rq2_comprehensiveness_curves(
            self.df_raw,
            save_path=os.path.join(plots_dir, "rq2_comprehensiveness_curves.png")
        )
        plot_rq2_morf_vs_lerf_and_sufficiency(
            self.df_raw,
            save_path=os.path.join(plots_dir, "rq2_morf_vs_lerf_and_sufficiency.png")
        )

        if self.df_summary is not None:
            plot_rq2_aopc_comparison(
                self.df_summary,
                save_path=os.path.join(plots_dir, "rq2_aopc_comparison.png")
            )
        plt.close("all")

    def generate_operator_robustness_report(self) -> Dict[str, Any]:
        """
        Cross-operator robustness report comparing 'deletion' vs. 'embedding_mask':
        - Method-level ranking agreement (Kendall tau and Spearman rho on AOPC_C, AOPC_U, G, AOPC_least)
        - Per-method instance-level paired correlation (Spearman rho, Pearson r, MAE) across shared sample_ids
        - LaTeX comparison table and side-by-side figure.
        """
        if "deletion" not in self.df_raw_by_op or "embedding_mask" not in self.df_raw_by_op:
            return {}

        df_del = self.df_raw_by_op["deletion"]
        df_mask = self.df_raw_by_op["embedding_mask"]
        sum_del = self.df_summary_by_op.get("deletion")
        sum_mask = self.df_summary_by_op.get("embedding_mask")
        if df_del is None or df_mask is None or len(df_del) == 0 or len(df_mask) == 0:
            return {}

        comp_dir = f"{self.base_output_dir}_operator_comparison"
        plots_dir = os.path.join(comp_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # 1. Method-level ranking agreement (excluding Random Baseline)
        ranking_stats: Dict[str, Any] = {}
        if sum_del is not None and sum_mask is not None:
            sd = sum_del[sum_del["explainer_name"] != "Random Baseline"]
            sm = sum_mask[sum_mask["explainer_name"] != "Random Baseline"]
            merged_sum = pd.merge(sd, sm, on="explainer_name", suffixes=("_del", "_mask"))
            ranking_stats["n_methods"] = int(len(merged_sum))
            for col in ["aopc_c_prob_mean", "aopc_u_prob_mean", "gap_g_mean", "aopc_least_prob_mean"]:
                c_del = f"{col}_del"
                c_mask = f"{col}_mask"
                if c_del in merged_sum.columns and c_mask in merged_sum.columns:
                    sub_m = merged_sum[[c_del, c_mask]].dropna()
                    if len(sub_m) >= 2 and np.std(sub_m[c_del]) > 0 and np.std(sub_m[c_mask]) > 0:
                        tau, tau_p = kendalltau(sub_m[c_del], sub_m[c_mask])
                        rho, rho_p = spearmanr(sub_m[c_del], sub_m[c_mask])
                        ranking_stats[col] = {
                            "kendall_tau": float(tau),
                            "kendall_p": float(tau_p),
                            "spearman_rho": float(rho),
                            "spearman_p": float(rho_p),
                        }

        with open(os.path.join(comp_dir, "operator_robustness_ranking.json"), "w", encoding="utf-8") as fr:
            json.dump(ranking_stats, fr, indent=2)

        # 2. Instance-level paired agreement per method
        method_rows = []
        common_methods = sorted(set(df_del["explainer_name"].unique()) & set(df_mask["explainer_name"].unique()))
        for method in common_methods:
            d1 = df_del[df_del["explainer_name"] == method][["sample_id", "aopc_c_prob", "aopc_u_prob"]]
            d2 = df_mask[df_mask["explainer_name"] == method][["sample_id", "aopc_c_prob", "aopc_u_prob"]]
            paired = pd.merge(d1, d2, on="sample_id", suffixes=("_del", "_mask")).dropna()
            n_shared = len(paired)

            rho_c = p_rho_c = pearson_c = mae_c = np.nan
            rho_u = p_rho_u = np.nan
            mean_c_del = float(d1["aopc_c_prob"].mean()) if len(d1) > 0 else np.nan
            mean_c_mask = float(d2["aopc_c_prob"].mean()) if len(d2) > 0 else np.nan
            mean_u_del = float(d1["aopc_u_prob"].mean()) if len(d1) > 0 else np.nan
            mean_u_mask = float(d2["aopc_u_prob"].mean()) if len(d2) > 0 else np.nan

            if n_shared >= 2:
                x_c = paired["aopc_c_prob_del"].to_numpy(dtype=float)
                y_c = paired["aopc_c_prob_mask"].to_numpy(dtype=float)
                mae_c = float(np.mean(np.abs(x_c - y_c)))
                if np.std(x_c) > 0 and np.std(y_c) > 0:
                    rho_c, p_rho_c = spearmanr(x_c, y_c)
                    pearson_c, _ = pearsonr(x_c, y_c)
                    rho_c, p_rho_c, pearson_c = float(rho_c), float(p_rho_c), float(pearson_c)

                x_u = paired["aopc_u_prob_del"].to_numpy(dtype=float)
                y_u = paired["aopc_u_prob_mask"].to_numpy(dtype=float)
                if np.std(x_u) > 0 and np.std(y_u) > 0:
                    rho_u, p_rho_u = spearmanr(x_u, y_u)
                    rho_u, p_rho_u = float(rho_u), float(p_rho_u)

            method_rows.append({
                "explainer_name": method,
                "n_shared": n_shared,
                "aopc_c_del_mean": mean_c_del,
                "aopc_c_mask_mean": mean_c_mask,
                "mae_aopc_c": mae_c,
                "spearman_aopc_c": rho_c,
                "spearman_aopc_c_p": p_rho_c,
                "pearson_aopc_c": pearson_c,
                "aopc_u_del_mean": mean_u_del,
                "aopc_u_mask_mean": mean_u_mask,
                "spearman_aopc_u": rho_u,
                "spearman_aopc_u_p": p_rho_u,
            })

        df_comp = pd.DataFrame(method_rows).sort_values("aopc_c_del_mean", ascending=False)
        df_comp.to_csv(os.path.join(comp_dir, "operator_robustness_methods.csv"), index=False)

        # 3. LaTeX comparison table
        tex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\caption{Porównanie odporności wyników RQ2 względem operatora perturbacji (\texttt{deletion} vs.~\texttt{embedding\_mask}).}",
            r"\label{tab:rq2_operator_robustness}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"\textbf{Metoda XAI} & \textbf{$\mathrm{AOPC}_C$ (Del)} & \textbf{$\mathrm{AOPC}_C$ (Mask)} & \textbf{MAE} & \textbf{$\rho$ (Del vs.~Mask)} & \textbf{$r$ (Del vs.~Mask)} \\",
            r"\midrule",
        ]
        for _, r in df_comp.iterrows():
            rho_str = f"{r['spearman_aopc_c']:.3f}" if not pd.isna(r["spearman_aopc_c"]) else "-"
            r_str = f"{r['pearson_aopc_c']:.3f}" if not pd.isna(r["pearson_aopc_c"]) else "-"
            mae_str = f"{r['mae_aopc_c']:.3f}" if not pd.isna(r["mae_aopc_c"]) else "-"
            tex_lines.append(
                f"{r['explainer_name']} & ${r['aopc_c_del_mean']:.3f}$ & ${r['aopc_c_mask_mean']:.3f}$ & ${mae_str}$ & ${rho_str}$ & ${r_str}$ \\\\"
            )
        tex_lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        with open(os.path.join(comp_dir, "operator_robustness_table.tex"), "w", encoding="utf-8") as ft:
            ft.write("\n".join(tex_lines))

        # 4. Plot
        if sum_del is not None and sum_mask is not None:
            plot_rq2_operator_robustness(
                sum_del,
                sum_mask,
                save_path=os.path.join(plots_dir, "operator_robustness_comparison.png"),
            )
            plt.close("all")

        return {
            "comparison_dir": comp_dir,
            "ranking": ranking_stats,
            "methods_df": df_comp,
        }
