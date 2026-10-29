import os
import json
import time
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Literal
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
    - Two perturbation operators: Discrete Deletion and Vector Embedding Masking.
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
        operator: Literal["deletion", "embedding_mask"] = "deletion",
        max_k: int = 5,
        random_trials: int = 20,
        data_path: str = "data/sst2_validation.json",
        output_dir: str = "results/rq2_faithfulness"
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

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

        self.operator = operator
        self.max_k = max_k
        self.random_trials = random_trials
        self.data_path = data_path
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_summary: Optional[pd.DataFrame] = None

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

    def run(
        self,
        num_samples: int = 400,
        seed: int = 42,
        resume: bool = True,
        max_run_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Executes the RQ2 evaluation suite with atomic checkpointing.
        """
        dataset = self.load_dataset(num_samples=num_samples, seed=seed)
        raw_csv_path = os.path.join(self.output_dir, "rq2_faithfulness_raw.csv")

        completed_keys = set()
        existing_records = []

        if resume and os.path.exists(raw_csv_path):
            try:
                df_existing = pd.read_csv(raw_csv_path)
                existing_records = df_existing.to_dict("records")
                for r in existing_records:
                    completed_keys.add((r["sample_id"], r["explainer_name"], r["operator"]))
                print(f"[RQ2 Checkpoint] Loaded {len(existing_records)} existing records ({len(completed_keys)} unique evaluations).")
            except Exception as e:
                print(f"[RQ2 Checkpoint Warning] Could not parse existing checkpoint: {e}. Starting fresh.")

        unflushed_records = []
        samples_processed_in_this_run = 0

        def flush_buffer():
            nonlocal unflushed_records, existing_records
            if not unflushed_records:
                return
            all_records = existing_records + unflushed_records
            df_to_save = pd.DataFrame(all_records)
            tmp_path = raw_csv_path + ".tmp"
            df_to_save.to_csv(tmp_path, index=False)
            if os.path.exists(raw_csv_path):
                os.remove(raw_csv_path)
            os.rename(tmp_path, raw_csv_path)
            existing_records = all_records
            unflushed_records = []

        print(f"\n=======================================================")
        print(f"  Starting RQ2 Faithfulness Benchmark (Operator: {self.operator})")
        print(f"  Samples: {len(dataset)} | Explainers: {[e.__class__.__name__ for e in self.explainers]}")
        print(f"  Checkpoints: Saved every 10 samples to {raw_csv_path}")
        print(f"=======================================================\n")

        tracker = HardwareSessionTracker(self.model, experiment_name=f"RQ2-Faithfulness-{self.operator}", num_samples=len(dataset))
        tracker.start()

        try:
            for sample_idx, sample in enumerate(tqdm(dataset, desc="Evaluating Faithfulness")):
                sample_id = sample.get("id", sample_idx)
                text = sample["sentence"]

                # Check if all explainers are already completed for this sample
                all_done = all(
                    (sample_id, explainer.explainer_name, self.operator) in completed_keys
                    for explainer in self.explainers
                ) and ((sample_id, "Random Baseline", self.operator) in completed_keys)

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

                # 3. Random Baseline (Monte Carlo over B permutations)
                rand_key = (sample_id, "Random Baseline", self.operator)
                if rand_key not in completed_keys:
                    c_prob_rand_trials = {k: [] for k in k_list}
                    c_log_rand_trials = {k: [] for k in k_list}
                    u_prob_rand_trials = {k: [] for k in k_list}
                    u_log_rand_trials = {k: [] for k in k_list}

                    for b in range(self.random_trials):
                        rand_units = rank_word_units(units, ranking_mode="random", seed=seed + sample_idx * 100 + b)
                        for k in k_list:
                            # Deletion
                            text_del = apply_word_deletion(text, rand_units[:k])
                            _, p_del = self.model.predict_single(text_del)
                            p_del_c = float(p_del[c_star])
                            c_prob_rand_trials[k].append(p_star - p_del_c)
                            c_log_rand_trials[k].append(f_star - float(np.log(max(p_del_c, 1e-12))))

                            # Retention
                            text_ret = apply_word_retention(text, units, rand_units[:k])
                            _, p_ret = self.model.predict_single(text_ret)
                            p_ret_c = float(p_ret[c_star])
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
                        "operator": self.operator,
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

                    unflushed_records.append(rand_row)
                    completed_keys.add(rand_key)
                    rand_aopc_c_prob = rand_result.aopc_comprehensiveness_prob
                else:
                    # Find random AOPC from existing
                    rec = next(r for r in existing_records if r["sample_id"] == sample_id and r["explainer_name"] == "Random Baseline")
                    rand_aopc_c_prob = rec["aopc_c_prob"]

                # 4. Evaluate each XAI explainer
                for explainer in self.explainers:
                    method_name = explainer.explainer_name
                    key = (sample_id, method_name, self.operator)
                    if key in completed_keys:
                        continue

                    c0 = self.model.get_counters()
                    res = explainer.explain(text, target_class=c_star)
                    c1 = self.model.get_counters()
                    exp_fwd = c1["forward_passes"] - c0["forward_passes"]
                    exp_bwd = c1["backward_passes"] - c0["backward_passes"]
                    aggregate_word_attributions(units, res.attributions, method="signed_sum")
                    ranked_units = rank_word_units(units, ranking_mode="top_signed")

                    c_prob_steps = []
                    c_log_steps = []
                    u_prob_steps = []
                    u_log_steps = []

                    for k in k_list:
                        if self.operator == "deletion":
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
                            # Embedding Masking operator
                            prompt = self.model.format_prompt(text)
                            enc = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                            base_embeds = self.model.get_input_embeddings()(enc.input_ids)

                            # Mask top-k
                            del_embeds = apply_embedding_mask(base_embeds, ranked_units[:k], mask_value="zero")
                            p_del = self.model.predict_embeddings(del_embeds)[0].cpu().numpy()
                            p_del_c = float(p_del[c_star])
                            c_prob_steps.append(p_star - p_del_c)
                            c_log_steps.append(f_star - float(np.log(max(p_del_c, 1e-12))))

                            # Retain top-k (mask remaining)
                            rem_units = [u for u in units if u not in ranked_units[:k]]
                            ret_embeds = apply_embedding_mask(base_embeds, rem_units, mask_value="zero")
                            p_ret = self.model.predict_embeddings(ret_embeds)[0].cpu().numpy()
                            p_ret_c = float(p_ret[c_star])
                            u_prob_steps.append(p_star - p_ret_c)
                            u_log_steps.append(f_star - float(np.log(max(p_ret_c, 1e-12))))

                    faith_res = evaluate_faithfulness_curves(
                        k_values=k_list,
                        c_prob=c_prob_steps,
                        c_log=c_log_steps,
                        u_prob=u_prob_steps,
                        u_log=u_log_steps,
                        random_aopc_c_prob=rand_aopc_c_prob
                    )

                    row = {
                        "sample_id": sample_id,
                        "device_name": getattr(self.model, "device_name", "Unknown"),
                        "explainer_name": method_name,
                        "operator": self.operator,
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

                    unflushed_records.append(row)
                    completed_keys.add(key)

                samples_processed_in_this_run += 1
                if samples_processed_in_this_run % 10 == 0:
                    flush_buffer()

        except KeyboardInterrupt:
            print("\n[RQ2] Interrupted by user! Flushing pending checkpoints to disk...")
            flush_buffer()
            print("[RQ2 Checkpoint] Safely saved. You can resume at any time.")

        flush_buffer()
        self.df_raw = pd.DataFrame(existing_records)
        self.generate_summary()
        self.generate_latex_table()
        self.generate_plots()

        stats = tracker.stop(num_samples=samples_processed_in_this_run)
        tracker.log_session(stats=stats)
        tracker.print_summary(stats=stats)

        return self.df_raw

    def generate_summary(self) -> pd.DataFrame:
        """Aggregates raw results into mean, std, and sem per explainer."""
        if self.df_raw is None or len(self.df_raw) == 0:
            return pd.DataFrame()

        summary_rows = []
        for method, group in self.df_raw.groupby("explainer_name"):
            summary_rows.append({
                "explainer_name": method,
                "operator": self.operator,
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
