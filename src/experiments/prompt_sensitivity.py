import os
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

try:
    from models import SentimentCausalLM
    from explainers import (
        BaseExplainer,
        OcclusionExplainer,
        GradientXInputExplainer,
        IntegratedGradientsExplainer,
        LIMEExplainer,
        SHAPExplainer,
    )
    from metrics import compute_all_stability_metrics
except ImportError:
    from src.models import SentimentCausalLM
    from src.explainers import (
        BaseExplainer,
        OcclusionExplainer,
        GradientXInputExplainer,
        IntegratedGradientsExplainer,
        LIMEExplainer,
        SHAPExplainer,
    )
    from src.metrics import compute_all_stability_metrics


@dataclass(frozen=True)
class PromptTemplateSpec:
    """Specification of a zero-shot sentiment classification prompt template."""
    name: str
    description: str
    prefix: str
    suffix: str
    neg_label: str = " negative"
    pos_label: str = " positive"
    is_baseline: bool = False


DEFAULT_PROMPT_TEMPLATES: List[PromptTemplateSpec] = [
    PromptTemplateSpec(
        name="zhao2021_base",
        description=(
            "Baseline template adapted from the SST-2 format of Zhao et al. "
            "(ICML 2021, 'Calibrate Before Use'): 'Review: {text}\\nSentiment:'. "
            "Used here in a zero-shot setting (no in-context examples)."
        ),
        prefix="Review: ",
        suffix="\nSentiment:",
        is_baseline=True,
    ),
    PromptTemplateSpec(
        name="paraphrase_instruction",
        description=(
            "Own wording based on a common prompt pattern (task instruction "
            "followed by the input): 'Classify the sentiment of the following "
            "movie review.\\nReview: {text}\\nSentiment:'. Generic formulation, "
            "widely used in prompting tutorials; not attributed to a single source."
        ),
        prefix="Classify the sentiment of the following movie review.\nReview: ",
        suffix="\nSentiment:",
    ),
    PromptTemplateSpec(
        name="paraphrase_question",
        description=(
            "Own wording based on a common prompt pattern (task phrased as a "
            "question with an 'Answer:' cue): 'Movie review: {text}\\nQuestion: "
            "Is the sentiment positive or negative?\\nAnswer:'. Similar in style "
            "to the Review/Answer format of Zhao et al. (2021); exact wording "
            "is not taken from a specific source."
        ),
        prefix="Movie review: ",
        suffix="\nQuestion: Is the sentiment positive or negative?\nAnswer:",
    ),
    PromptTemplateSpec(
        name="paraphrase_polarity",
        description=(
            "Own wording based on the baseline: the input field name is changed "
            "('Review' -> 'Text') and the label cue is changed ('Sentiment' -> "
            "'Overall sentiment'): 'Text: {text}\\nOverall sentiment:'. "
            "Generic formulation; no specific source identified."
        ),
        prefix="Text: ",
        suffix="\nOverall sentiment:",
    ),
]


def load_sst2_samples(
    data_path: str = "data/sst2_validation.json",
    num_samples: int = 872,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Loads deterministic SST-2 validation samples matching RQ1/RQ2 ordering."""
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        from datasets import load_dataset
        ds = load_dataset("stanfordnlp/sst2", split="validation")
        data = [{"sentence": item["sentence"], "label": item["label"]} for item in ds]

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(data))[: min(num_samples, len(data))]
    return [
        {
            "id": int(data[i].get("id", i)),
            "sentence": str(data[i]["sentence"]).strip(),
            "label": int(data[i]["label"]),
        }
        for i in indices
    ]


class PromptSensitivityExperiment:
    """
    Control experiment evaluating prompt format sensitivity in zero-shot Causal LM
    sentiment classification and its downstream impact on XAI attribution stability.

    Protocol:
    1. Classification check on SST-2 (Accuracy, F1, ROC-AUC, Agreement vs. Base).
    2. Attribution rank-correlation check (Spearman rho, Kendall tau, Top-k Jaccard)
       between each prompt paraphrase and the frozen literature baseline (Zhao et al., 2021).
    """

    def __init__(
        self,
        model: SentimentCausalLM,
        templates: Optional[List[PromptTemplateSpec]] = None,
        data_path: str = "data/sst2_validation.json",
    ):
        self.model = model
        self.templates = templates if templates is not None else list(DEFAULT_PROMPT_TEMPLATES)
        self.data_path = data_path
        if not any(t.is_baseline for t in self.templates):
            raise ValueError("At least one template must have is_baseline=True.")

    def _build_explainers(
        self,
        methods: List[str],
        ig_steps: int = 100,
        ig_baseline: str = "mean",
        lime_samples: int = 1000,
        max_evals_shap: int = 2500,
    ) -> List[BaseExplainer]:
        explainers: List[BaseExplainer] = []
        for m in methods:
            key = m.strip().lower()
            if key == "occlusion":
                explainers.append(OcclusionExplainer(self.model))
            elif key in ("grad_x_input", "gradient_x_input"):
                explainers.append(GradientXInputExplainer(self.model))
            elif key in ("ig", "integrated_gradients"):
                explainers.append(
                    IntegratedGradientsExplainer(
                        self.model, steps=ig_steps, baseline_type=ig_baseline
                    )
                )
            elif key == "lime":
                explainers.append(LIMEExplainer(self.model, num_samples=lime_samples))
            elif key == "shap":
                explainers.append(SHAPExplainer(self.model, max_evals=max_evals_shap))
            else:
                raise ValueError(f"Unknown explainer method: {m}")
        return explainers

    def evaluate_classification(
        self,
        samples: List[Dict[str, Any]],
        calibrate: bool = True,
        progress_bar: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[int, Tuple[int, float]]]]:
        """
        Evaluates zero-shot classification metrics for all prompt templates.
        Returns summary DataFrame and per-template prediction lookup {sample_id: (pred, p_pos)}.
        """
        y_true = np.array([s["label"] for s in samples], dtype=int)
        preds_by_template: Dict[str, Dict[int, Tuple[int, float]]] = {}
        rows: List[Dict[str, Any]] = []

        base_name = next(t.name for t in self.templates if t.is_baseline)

        for spec in self.templates:
            t0 = time.perf_counter()
            self.model.set_prompt_template(
                prompt_prefix=spec.prefix,
                prompt_suffix=spec.suffix,
                neg_label=spec.neg_label,
                pos_label=spec.pos_label,
                calibrate_logits=calibrate,
            )

            preds: List[int] = []
            p_pos_list: List[float] = []
            pred_map: Dict[int, Tuple[int, float]] = {}

            iterator = tqdm(samples, desc=f"Clf [{spec.name}]", disable=not progress_bar)
            for s in iterator:
                pred_cls, probs = self.model.predict_single(s["sentence"])
                p_pos = float(probs[1])
                preds.append(int(pred_cls))
                p_pos_list.append(p_pos)
                pred_map[s["id"]] = (int(pred_cls), p_pos)

            preds_arr = np.array(preds, dtype=int)
            probs_arr = np.array(p_pos_list, dtype=float)
            preds_by_template[spec.name] = pred_map

            acc = float(accuracy_score(y_true, preds_arr))
            f1 = float(f1_score(y_true, preds_arr, zero_division=0))
            auc = float(roc_auc_score(y_true, probs_arr)) if len(np.unique(y_true)) > 1 else float("nan")
            pos_rate = float(np.mean(preds_arr))
            elapsed = time.perf_counter() - t0

            rows.append({
                "model": self.model.model_name,
                "template": spec.name,
                "is_baseline": spec.is_baseline,
                "calibrated": calibrate,
                "n_samples": len(samples),
                "accuracy": acc,
                "f1": f1,
                "roc_auc": auc,
                "pos_pred_rate": pos_rate,
                "elapsed_s": round(elapsed, 2),
            })

        base_preds = preds_by_template[base_name]
        for row in rows:
            cur_preds = preds_by_template[row["template"]]
            agreements = [
                1.0 if cur_preds[s["id"]][0] == base_preds[s["id"]][0] else 0.0
                for s in samples
            ]
            agree_rate = float(np.mean(agreements))
            row["agreement_vs_base"] = agree_rate
            row["flip_rate_vs_base"] = 1.0 - agree_rate

        return pd.DataFrame(rows), preds_by_template

    def evaluate_attribution_stability(
        self,
        samples: List[Dict[str, Any]],
        preds_by_template: Dict[str, Dict[int, Tuple[int, float]]],
        methods: List[str],
        ig_steps: int = 100,
        ig_baseline: str = "mean",
        lime_samples: int = 1000,
        max_evals_shap: int = 2500,
        calibrate: bool = True,
        progress_bar: bool = True,
    ) -> pd.DataFrame:
        """
        Computes word-level attribution rank correlation between each paraphrase template
        and the baseline template across selected XAI explainers.
        """
        base_spec = next(t for t in self.templates if t.is_baseline)
        explainers = self._build_explainers(
            methods=methods,
            ig_steps=ig_steps,
            ig_baseline=ig_baseline,
            lime_samples=lime_samples,
            max_evals_shap=max_evals_shap,
        )

        # Store attributions for each template -> explainer_name -> sample_id -> AttributionResult
        attr_cache: Dict[str, Dict[str, Dict[int, Any]]] = {t.name: {} for t in self.templates}

        for spec in self.templates:
            self.model.set_prompt_template(
                prompt_prefix=spec.prefix,
                prompt_suffix=spec.suffix,
                neg_label=spec.neg_label,
                pos_label=spec.pos_label,
                calibrate_logits=calibrate,
            )

            for exp in explainers:
                exp_name = exp.explainer_name
                attr_cache[spec.name][exp_name] = {}
                iterator = tqdm(
                    samples,
                    desc=f"XAI [{spec.name} | {exp_name}]",
                    disable=not progress_bar,
                )
                for s in iterator:
                    sid = s["id"]
                    text = s["sentence"]
                    pred_cls, p_pos = preds_by_template[spec.name][sid]
                    base_cls, _ = preds_by_template[base_spec.name][sid]
                    # Explain the frozen baseline decision when preserved, else variant's own decision
                    target_cls = base_cls if pred_cls == base_cls else pred_cls
                    probs_arr = [1.0 - p_pos, p_pos]
                    res = exp.explain(
                        text=text,
                        target_class=target_cls,
                        baseline_prob=float(probs_arr[target_cls]),
                    )
                    span_start, span_end, _ = self.model.find_review_span(
                        text, num_tokens=len(res.tokens)
                    )
                    if res.metadata is None:
                        res.metadata = {}
                    res.metadata["review_span"] = (span_start, span_end)
                    attr_cache[spec.name][exp_name][sid] = res
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        raw_rows: List[Dict[str, Any]] = []
        for spec in self.templates:
            if spec.is_baseline:
                continue
            for exp in explainers:
                exp_name = exp.explainer_name
                for s in samples:
                    sid = s["id"]
                    base_cls = preds_by_template[base_spec.name][sid][0]
                    var_cls = preds_by_template[spec.name][sid][0]
                    preserved = bool(base_cls == var_cls)

                    res_base = attr_cache[base_spec.name][exp_name][sid]
                    res_var = attr_cache[spec.name][exp_name][sid]
                    metrics = compute_all_stability_metrics(
                        res_base, res_var, filter_prompt_template=True
                    )
                    raw_rows.append({
                        "model": self.model.model_name,
                        "sample_id": sid,
                        "explainer": exp_name,
                        "base_template": base_spec.name,
                        "variant_template": spec.name,
                        "base_pred": base_cls,
                        "variant_pred": var_cls,
                        "decision_preserved": preserved,
                        **metrics,
                    })

        return pd.DataFrame(raw_rows)

    def run(
        self,
        clf_samples: int = 872,
        attr_samples: int = 50,
        methods: Optional[List[str]] = None,
        seed: int = 42,
        ig_steps: int = 100,
        ig_baseline: str = "mean",
        lime_samples: int = 1000,
        max_evals_shap: int = 2500,
        calibrate: bool = True,
        output_dir: str = "results/prompt_sensitivity",
        progress_bar: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        os.makedirs(output_dir, exist_ok=True)
        if methods is None:
            methods = ["grad_x_input", "occlusion", "ig"]

        all_samples = load_sst2_samples(
            data_path=self.data_path,
            num_samples=max(clf_samples, attr_samples),
            seed=seed,
        )
        clf_set = all_samples[:clf_samples]
        attr_set = all_samples[:attr_samples]

        df_clf, preds_by_template = self.evaluate_classification(
            samples=clf_set,
            calibrate=calibrate,
            progress_bar=progress_bar,
        )
        df_clf.to_csv(os.path.join(output_dir, "prompt_classification_metrics.csv"), index=False)

        df_attr_raw = self.evaluate_attribution_stability(
            samples=attr_set,
            preds_by_template=preds_by_template,
            methods=methods,
            ig_steps=ig_steps,
            ig_baseline=ig_baseline,
            lime_samples=lime_samples,
            max_evals_shap=max_evals_shap,
            calibrate=calibrate,
            progress_bar=progress_bar,
        )
        df_attr_raw.to_csv(os.path.join(output_dir, "prompt_attribution_raw.csv"), index=False)

        df_summary = self.summarize_and_plot(
            df_clf=df_clf,
            df_attr_raw=df_attr_raw,
            model_name=self.model.model_name,
            output_dir=output_dir,
        )
        return df_clf, df_attr_raw, df_summary

    @classmethod
    def summarize_and_plot(
        cls,
        df_clf: pd.DataFrame,
        df_attr_raw: pd.DataFrame,
        model_name: str,
        output_dir: str,
    ) -> pd.DataFrame:
        os.makedirs(output_dir, exist_ok=True)
        metric_cols = [
            "spearman", "kendall", "pearson",
            "jaccard_k1", "jaccard_k3", "jaccard_k5",
        ]
        summary_rows: List[Dict[str, Any]] = []
        for (exp_name, var_name), grp in df_attr_raw.groupby(["explainer", "variant_template"]):
            grp_pres = grp[grp["decision_preserved"]]
            row: Dict[str, Any] = {
                "model": model_name,
                "explainer": exp_name,
                "variant_template": var_name,
                "n_total": int(len(grp)),
                "n_preserved": int(len(grp_pres)),
                "decision_preservation_rate": float(len(grp_pres) / max(1, len(grp))),
            }
            for col in metric_cols:
                vals = grp_pres[col].dropna() if len(grp_pres) > 0 else grp[col].dropna()
                row[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else float("nan")
                row[f"{col}_std"] = float(vals.std()) if len(vals) > 1 else float("nan")
            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(os.path.join(output_dir, "prompt_attribution_summary.csv"), index=False)

        cls._plot_summary_static(df_clf, df_summary, model_name, output_dir)
        return df_summary

    @staticmethod
    def _plot_summary_static(
        df_clf: pd.DataFrame,
        df_summary: pd.DataFrame,
        model_name: str,
        output_dir: str,
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left panel: Classification accuracy & ROC-AUC
        x = np.arange(len(df_clf))
        width = 0.35
        axes[0].bar(x - width / 2, df_clf["accuracy"], width, label="Accuracy", color="#2b5c8f")
        axes[0].bar(x + width / 2, df_clf["roc_auc"], width, label="ROC-AUC", color="#4f9da6")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df_clf["template"], rotation=20, ha="right")
        axes[0].set_ylim(0.5, 1.02)
        axes[0].set_ylabel("Score")
        axes[0].set_title(f"Zero-Shot Classification ({model_name})")
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)

        # Right panel: Attribution Spearman rho vs base prompt
        if not df_summary.empty:
            piv = df_summary.pivot(
                index="variant_template", columns="explainer", values="spearman_mean"
            )
            piv.plot(kind="bar", ax=axes[1], rot=20)
            axes[1].set_ylim(-0.1, 1.05)
            axes[1].set_ylabel("Spearman $\\rho$ vs. Base Prompt")
            axes[1].set_title("Attribution Rank Stability Across Prompt Paraphrases")
            axes[1].legend(title="Explainer")
            axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prompt_sensitivity_summary.png"), dpi=200)
        plt.close(fig)
