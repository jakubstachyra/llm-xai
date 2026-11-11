import os
import json
import time
import pandas as pd
import numpy as np
import torch
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
from src.faithfulness import (
    WordUnit,
    extract_word_units,
    aggregate_word_attributions,
    rank_word_units,
    apply_word_deletion,
    apply_word_retention,
    apply_embedding_mask,
    evaluate_faithfulness_curves,
)
from src.experiments.visualizations import (
    plot_rq2_comprehensiveness_curves,
    plot_rq2_aopc_comparison,
)


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
        data_path: str = "data/sst2_validation.json",
        output_dir: str = "results/rq2_faithfulness"
    ):
        self.output_dir = output_dir
        self.operator = operator
        if operator == "both":
            base_dir = output_dir[:-5] if output_dir.endswith("_both") else output_dir
            self.active_operators: List[str] = ["deletion", "embedding_mask"]
            self.op_dirs: Dict[str, str] = {
                "deletion": f"{base_dir}_deletion",
                "embedding_mask": f"{base_dir}_embedding_mask",
            }
        else:
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
        data = [data[i] for i in indices]

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
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Evaluates Comprehensiveness C(k) and Sufficiency U(k) step curves for a given operator."""
        c_prob_steps: List[float] = []
        c_log_steps: List[float] = []
        u_prob_steps: List[float] = []
        u_log_steps: List[float] = []

        if op == "deletion":
            for k in k_list:
                # Comprehensiveness: Excise top-k
                text_del = apply_word_deletion(text, ranked_units[:k])
                _, p_del = self.model.predict_single(text_del)
                p_del_c = float(p_del[c_star])
                c_prob_steps.append(p_star - p_del_c)
                c_log_steps.append(f_star - float(np.log(max(p_del_c, 1e-12))))

                # Sufficiency: Retain only top-k
                text_ret = apply_word_retention(text, units, ranked_units[:k])
                _, p_ret = self.model.predict_single(text_ret)
                p_ret_c = float(p_ret[c_star])
                u_prob_steps.append(p_star - p_ret_c)
                u_log_steps.append(f_star - float(np.log(max(p_ret_c, 1e-12))))
        else:
            # Embedding Masking operator (batched across all k in k_list since seq_len is identical)
            if base_embeds is None:
                prompt = self.model.format_prompt(text)
                enc = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    base_embeds = self.model.get_input_embeddings()(enc.input_ids).detach()

            batch_embeds_list = []
            for k in k_list:
                del_embeds = apply_embedding_mask(base_embeds, ranked_units[:k], mask_value="zero")
                rem_units = [u for u in units if u not in ranked_units[:k]]
                ret_embeds = apply_embedding_mask(base_embeds, rem_units, mask_value="zero")
                batch_embeds_list.append(del_embeds)
                batch_embeds_list.append(ret_embeds)

            stacked_embeds = torch.cat(batch_embeds_list, dim=0)  # shape: (2*K, seq_len, hidden_dim)
            probs_batch = self.model.predict_embeddings(stacked_embeds).cpu().numpy()

            for idx_k in range(len(k_list)):
                p_del_c = float(probs_batch[2 * idx_k, c_star])
                p_ret_c = float(probs_batch[2 * idx_k + 1, c_star])
                c_prob_steps.append(p_star - p_del_c)
                c_log_steps.append(f_star - float(np.log(max(p_del_c, 1e-12))))
                u_prob_steps.append(p_star - p_ret_c)
                u_log_steps.append(f_star - float(np.log(max(p_ret_c, 1e-12))))

        return c_prob_steps, c_log_steps, u_prob_steps, u_log_steps

    def run(
        self,
        num_samples: int = 400,
        seed: int = 42,
        resume: bool = True,
        max_run_samples: Optional[int] = None,
        overwrite_methods: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Executes the RQ2 evaluation suite with atomic checkpointing.
        When operator='both', computes each XAI explanation ONCE per sample and evaluates
        both 'deletion' and 'embedding_mask' operators in the same pass.
        """
        dataset = self.load_dataset(num_samples=num_samples, seed=seed)

        raw_csv_paths: Dict[str, str] = {
            op: os.path.join(self.op_dirs[op], "rq2_faithfulness_raw.csv")
            for op in self.active_operators
        }
        completed_keys: Dict[str, set] = {op: set() for op in self.active_operators}
        existing_records: Dict[str, List[Dict[str, Any]]] = {op: [] for op in self.active_operators}
        unflushed_records: Dict[str, List[Dict[str, Any]]] = {op: [] for op in self.active_operators}

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
                        completed_keys[op].add((int(r["sample_id"]), str(r["explainer_name"]), str(r["operator"])))
                    print(f"[RQ2 Checkpoint ({op})] Loaded {len(existing_records[op])} existing records ({len(completed_keys[op])} unique evaluations).")
                except Exception as e:
                    print(f"[RQ2 Checkpoint Warning ({op})] Could not parse existing checkpoint: {e}. Starting fresh.")

        samples_processed_in_this_run = 0

        def flush_buffer():
            for op in self.active_operators:
                if not unflushed_records[op]:
                    continue
                all_records = existing_records[op] + unflushed_records[op]
                df_to_save = pd.DataFrame(all_records)
                raw_csv_path = raw_csv_paths[op]
                tmp_path = raw_csv_path + ".tmp"
                df_to_save.to_csv(tmp_path, index=False)
                if os.path.exists(raw_csv_path):
                    os.remove(raw_csv_path)
                os.rename(tmp_path, raw_csv_path)
                existing_records[op] = all_records
                unflushed_records[op] = []

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
                    prompt = self.model.format_prompt(text)
                    enc = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    with torch.no_grad():
                        base_embeds = self.model.get_input_embeddings()(enc.input_ids).detach()

                # 3. Random Baseline (Monte Carlo over B permutations) per active operator
                rand_aopc_c_prob_by_op: Dict[str, float] = {}
                for op in self.active_operators:
                    rand_key = (sample_id, "Random Baseline", op)
                    if rand_key not in completed_keys[op]:
                        c_prob_rand_trials = {k: [] for k in k_list}
                        c_log_rand_trials = {k: [] for k in k_list}
                        u_prob_rand_trials = {k: [] for k in k_list}
                        u_log_rand_trials = {k: [] for k in k_list}

                        if op == "deletion":
                            for b in range(self.random_trials):
                                rand_units = rank_word_units(units, ranking_mode="random", seed=seed + sample_idx * 100 + b)
                                for k in k_list:
                                    text_del = apply_word_deletion(text, rand_units[:k])
                                    _, p_del = self.model.predict_single(text_del)
                                    p_del_c = float(p_del[c_star])
                                    c_prob_rand_trials[k].append(p_star - p_del_c)
                                    c_log_rand_trials[k].append(f_star - float(np.log(max(p_del_c, 1e-12))))

                                    text_ret = apply_word_retention(text, units, rand_units[:k])
                                    _, p_ret = self.model.predict_single(text_ret)
                                    p_ret_c = float(p_ret[c_star])
                                    u_prob_rand_trials[k].append(p_star - p_ret_c)
                                    u_log_rand_trials[k].append(f_star - float(np.log(max(p_ret_c, 1e-12))))
                        else:
                            # Fast batched embedding masking across all random trials
                            all_embeds_list = []
                            trial_k_pairs = []
                            for b in range(self.random_trials):
                                rand_units = rank_word_units(units, ranking_mode="random", seed=seed + sample_idx * 100 + b)
                                for k in k_list:
                                    del_embeds = apply_embedding_mask(base_embeds, rand_units[:k], mask_value="zero")
                                    rem_units = [u for u in units if u not in rand_units[:k]]
                                    ret_embeds = apply_embedding_mask(base_embeds, rem_units, mask_value="zero")
                                    all_embeds_list.append(del_embeds)
                                    all_embeds_list.append(ret_embeds)
                                    trial_k_pairs.append(k)

                            stacked_rand_embeds = torch.cat(all_embeds_list, dim=0)
                            # Process in memory-safe sub-batches of 64
                            probs_chunks = []
                            for chunk_start in range(0, stacked_rand_embeds.shape[0], 64):
                                chunk_e = stacked_rand_embeds[chunk_start:chunk_start + 64]
                                probs_chunks.append(self.model.predict_embeddings(chunk_e).cpu().numpy())
                            probs_all = np.vstack(probs_chunks)
                            del stacked_rand_embeds, all_embeds_list

                            for pair_idx, k in enumerate(trial_k_pairs):
                                p_del_c = float(probs_all[2 * pair_idx, c_star])
                                p_ret_c = float(probs_all[2 * pair_idx + 1, c_star])
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
                            "gap_g": 0.0,
                            "harmonization_h": rand_result.harmonization_h,
                            "execution_time": 0.0,
                            "forward_passes": 0,
                            "backward_passes": 0,
                        }
                        for i, k in enumerate(k_list):
                            rand_row[f"c_prob_k{k}"] = c_prob_rand[i]
                            rand_row[f"u_prob_k{k}"] = u_prob_rand[i]
                            rand_row[f"c_log_k{k}"] = c_log_rand[i]
                            rand_row[f"u_log_k{k}"] = u_log_rand[i]

                        unflushed_records[op].append(rand_row)
                        completed_keys[op].add(rand_key)
                        rand_aopc_c_prob_by_op[op] = rand_result.aopc_comprehensiveness_prob
                    else:
                        rec = next(
                            r for r in (existing_records[op] + unflushed_records[op])
                            if int(r["sample_id"]) == sample_id and str(r["explainer_name"]) == "Random Baseline"
                        )
                        rand_aopc_c_prob_by_op[op] = float(rec["aopc_c_prob"])

                # 4. Evaluate each XAI explainer ONCE per sample and apply to all needed operators
                for explainer in self.explainers:
                    method_name = explainer.explainer_name
                    ops_needed = [
                        op for op in self.active_operators
                        if (sample_id, method_name, op) not in completed_keys[op]
                    ]
                    if not ops_needed:
                        continue

                    c0 = self.model.get_counters()
                    res = explainer.explain(text, target_class=c_star)
                    c1 = self.model.get_counters()
                    exp_fwd = c1["forward_passes"] - c0["forward_passes"]
                    exp_bwd = c1["backward_passes"] - c0["backward_passes"]
                    aggregate_word_attributions(units, res.attributions, method="signed_sum")
                    ranked_units = rank_word_units(units, ranking_mode="top_signed")

                    for op in ops_needed:
                        c_prob_steps, c_log_steps, u_prob_steps, u_log_steps = self._eval_ranked_units_for_operator(
                            op=op,
                            text=text,
                            units=units,
                            ranked_units=ranked_units,
                            k_list=k_list,
                            c_star=c_star,
                            p_star=p_star,
                            f_star=f_star,
                            base_embeds=base_embeds,
                        )

                        faith_res = evaluate_faithfulness_curves(
                            k_values=k_list,
                            c_prob=c_prob_steps,
                            c_log=c_log_steps,
                            u_prob=u_prob_steps,
                            u_log=u_log_steps,
                            random_aopc_c_prob=rand_aopc_c_prob_by_op[op]
                        )

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
                            "aopc_c_prob": faith_res.aopc_comprehensiveness_prob,
                            "aopc_c_log": faith_res.aopc_comprehensiveness_log,
                            "aopc_u_prob": faith_res.aopc_sufficiency_prob,
                            "aopc_u_log": faith_res.aopc_sufficiency_log,
                            "gap_g": faith_res.normalized_gap_g,
                            "harmonization_h": faith_res.harmonization_h,
                            "execution_time": res.execution_time,
                            "forward_passes": exp_fwd,
                            "backward_passes": exp_bwd,
                        }
                        for i, k in enumerate(k_list):
                            row[f"c_prob_k{k}"] = c_prob_steps[i]
                            row[f"u_prob_k{k}"] = u_prob_steps[i]
                            row[f"c_log_k{k}"] = c_log_steps[i]
                            row[f"u_log_k{k}"] = u_log_steps[i]

                        unflushed_records[op].append(row)
                        completed_keys[op].add((sample_id, method_name, op))

                del base_embeds
                samples_processed_in_this_run += 1
                if samples_processed_in_this_run % 10 == 0:
                    flush_buffer()
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
            self.generate_latex_table()
            self.generate_plots()

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
        """Aggregates raw results into mean, std, and sem per explainer."""
        if self.df_raw is None or len(self.df_raw) == 0:
            return pd.DataFrame()

        op_str = operator_label if operator_label is not None else self.operator
        summary_rows = []
        for method, group in self.df_raw.groupby("explainer_name"):
            summary_rows.append({
                "explainer_name": method,
                "operator": op_str,
                "count": len(group),
                "aopc_c_prob_mean": group["aopc_c_prob"].mean(),
                "aopc_c_prob_std": group["aopc_c_prob"].std(),
                "aopc_c_prob_sem": group["aopc_c_prob"].sem(),
                "aopc_c_log_mean": group["aopc_c_log"].mean(),
                "aopc_u_prob_mean": group["aopc_u_prob"].mean(),
                "aopc_u_prob_std": group["aopc_u_prob"].std(),
                "gap_g_mean": group["gap_g"].mean(),
                "harmonization_h_mean": group["harmonization_h"].mean(),
                "time_mean": group["execution_time"].mean(),
            })

        self.df_summary = pd.DataFrame(summary_rows).sort_values("aopc_c_prob_mean", ascending=False)
        summary_csv = os.path.join(self.output_dir, "rq2_faithfulness_summary.csv")
        self.df_summary.to_csv(summary_csv, index=False)
        return self.df_summary

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
            r"\textbf{Metoda XAI} & \textbf{$\mathrm{AOPC}_C \uparrow$} & \textbf{$\mathrm{AOPC}_U \downarrow$} & \textbf{Luka $G \uparrow$} & \textbf{Harmonia $H \uparrow$} & \textbf{Czas [s]} \\",
            r"\midrule",
        ]

        for _, r in self.df_summary.iterrows():
            method = r["explainer_name"]
            c_val = f"{r['aopc_c_prob_mean']:.3f} \\pm {r['aopc_c_prob_sem']:.3f}"
            u_val = f"{r['aopc_u_prob_mean']:.3f}"
            g_val = f"{r['gap_g_mean']:+.2f}" if method != "Random Baseline" else "0.00"
            h_val = f"{r['harmonization_h_mean']:.3f}"
            t_val = f"{r['time_mean']:.2f}" if r["time_mean"] > 0 else "-"

            if method == "Random Baseline":
                lines.append(r"\midrule")
                lines.append(f"{method} & ${c_val}$ & ${u_val}$ & {g_val} & ${h_val}$ & {t_val} \\\\")
            else:
                lines.append(f"{method} & ${c_val}$ & ${u_val}$ & ${g_val}$ & ${h_val}$ & {t_val} \\\\")

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

        if self.df_summary is not None:
            plot_rq2_aopc_comparison(
                self.df_summary,
                save_path=os.path.join(plots_dir, "rq2_aopc_comparison.png")
            )
