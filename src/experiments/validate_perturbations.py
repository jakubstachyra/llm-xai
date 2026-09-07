import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from models import SentimentCausalLM
from perturbations import TextPerturber

class PerturbationValidator:
    """
    Phase 1: Behavioral Validation of Semantic Invariance of Text Perturbations.
    Measures whether the LLM's classification decision is preserved (Flip Rate)
    and quantifies probability drift (|Delta P|) across the SST-2 dataset.
    """
    def __init__(
        self,
        model: Optional[SentimentCausalLM] = None,
        model_name: str = "microsoft/phi-2",
        data_path: str = "data/sst2_validation.json"
    ):
        if model is None:
            print(f"[Validator] Inicjalizacja modelu '{model_name}'...")
            self.model = SentimentCausalLM(model_name=model_name)
        else:
            self.model = model

        self.data_path = data_path
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_summary: Optional[pd.DataFrame] = None
        self.global_metrics: Dict[str, Any] = {}

    def load_dataset(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Loads SST-2 validation dataset."""
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            from datasets import load_dataset
            print("[Validator] Pobieranie SST-2 z HuggingFace...")
            ds = load_dataset("stanfordnlp/sst2", split="validation")
            data = [{"sentence": item["sentence"], "label": item["label"]} for item in ds]

        if limit is not None and limit < len(data):
            data = data[:limit]

        print(f"[Validator] Załadowano {len(data)} zdań do walidacji perturbacji.")
        return data

    def run(self, limit: Optional[int] = None, progress_bar: bool = True) -> pd.DataFrame:
        """
        Runs empirical validation of all perturbations on the model.
        """
        dataset = self.load_dataset(limit=limit)
        records = []
        
        strict_preserved_count = 0
        total_samples = len(dataset)
        
        iterator = tqdm(enumerate(dataset), total=total_samples, desc="Walidacja behawioralna") if progress_bar else enumerate(dataset)

        for idx, item in iterator:
            orig_text = item["sentence"]
            true_label = int(item["label"])

            # Predict original sentence
            orig_pred, orig_probs = self.model.predict_single(orig_text)
            orig_p_target = float(orig_probs[orig_pred])
            orig_is_correct = (orig_pred == true_label)

            # Get categorized perturbations
            cat_perts = TextPerturber.get_categorized_perturbations(orig_text)
            all_pert_preserved = True

            for category, perts in cat_perts.items():
                for pert_name, pert_text in perts.items():
                    pert_pred, pert_probs = self.model.predict_single(pert_text)
                    
                    # Probability of original predicted class after perturbation
                    pert_p_orig_target = float(pert_probs[orig_pred])
                    prob_diff = orig_p_target - pert_p_orig_target
                    abs_prob_diff = abs(prob_diff)
                    
                    # Did decision flip?
                    decision_flip = (orig_pred != pert_pred)
                    is_preserved = not decision_flip
                    
                    if decision_flip:
                        all_pert_preserved = False

                    pert_is_correct = (pert_pred == true_label)

                    records.append({
                        "sample_id": idx,
                        "perturbation_category": category,
                        "perturbation_name": pert_name,
                        "original_text": orig_text,
                        "perturbed_text": pert_text,
                        "true_label": true_label,
                        "orig_pred": orig_pred,
                        "pert_pred": pert_pred,
                        "orig_prob": orig_p_target,
                        "pert_prob": pert_p_orig_target,
                        "prob_diff": prob_diff,
                        "abs_prob_diff": abs_prob_diff,
                        "decision_flip": decision_flip,
                        "is_preserved": is_preserved,
                        "orig_correct": orig_is_correct,
                        "pert_correct": pert_is_correct
                    })

            if all_pert_preserved:
                strict_preserved_count += 1

        self.df_raw = pd.DataFrame(records)
        self.global_metrics = {
            "total_samples": total_samples,
            "strict_preserved_samples": strict_preserved_count,
            "strict_preserved_pct": (strict_preserved_count / total_samples) * 100 if total_samples > 0 else 0.0
        }

        self._compute_summary()
        return self.df_raw

    def _compute_summary(self):
        """Computes summary statistics for each perturbation."""
        if self.df_raw is None or len(self.df_raw) == 0:
            return

        grouped = self.df_raw.groupby(["perturbation_category", "perturbation_name"])
        summary_rows = []

        for (cat, name), group in grouped:
            total = len(group)
            flips = group["decision_flip"].sum()
            flip_rate = (flips / total) * 100.0
            preserved_rate = 100.0 - flip_rate
            mean_abs_delta_p = group["abs_prob_diff"].mean()
            std_abs_delta_p = group["abs_prob_diff"].std()
            orig_acc = group["orig_correct"].mean() * 100.0
            pert_acc = group["pert_correct"].mean() * 100.0

            summary_rows.append({
                "perturbation_category": cat,
                "perturbation_name": name,
                "total_samples": total,
                "preserved_samples": total - flips,
                "flip_count": flips,
                "flip_rate_pct": flip_rate,
                "preserved_rate_pct": preserved_rate,
                "mean_abs_delta_p": mean_abs_delta_p,
                "std_abs_delta_p": std_abs_delta_p,
                "orig_acc_pct": orig_acc,
                "pert_acc_pct": pert_acc,
                "acc_change_pct": pert_acc - orig_acc
            })

        self.df_summary = pd.DataFrame(summary_rows)

    def generate_latex_table(self) -> str:
        """
        Generates publication-ready Table 1 in LaTeX for thesis Chapter 3/4.
        """
        if self.df_summary is None:
            self._compute_summary()

        latex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Empiryczna weryfikacja niezmienniczości semantycznej perturbacji promptu (Faza 1)}",
            r"\label{tab:perturbation_validation}",
            r"\begin{tabular}{llcccc}",
            r"\hline",
            r"\textbf{Kategoria} & \textbf{Perturbacja} & \textbf{Zachowane [\%]} & \textbf{Flip Rate [\%]} & \textbf{Średni $|\Delta P|$} & \textbf{Acc modelu [\%]} \\",
            r"\hline"
        ]

        for _, row in self.df_summary.iterrows():
            cat = row["perturbation_category"]
            name = row["perturbation_name"].replace("_", r"\_")
            pres = f"{row['preserved_rate_pct']:.1f}\\%"
            flip = f"{row['flip_rate_pct']:.1f}\\%"
            dp = f"{row['mean_abs_delta_p']:.4f} $\\pm$ {row['std_abs_delta_p']:.3f}"
            acc = f"{row['pert_acc_pct']:.1f}\\%"

            latex_lines.append(f"{cat} & \\texttt{{{name}}} & {pres} & {flip} & {dp} & {acc} \\\\")

        orig_acc_str = f"{self.df_raw['orig_correct'].mean()*100:.1f}" if self.df_raw is not None else "N/A"
        strict_pct_str = f"{self.global_metrics.get('strict_preserved_pct', 0):.1f}"
        strict_samples_str = f"{self.global_metrics.get('strict_preserved_samples', 0)}"
        total_samples_str = f"{self.global_metrics.get('total_samples', 0)}"

        latex_lines.extend([
            r"\hline",
            r"\multicolumn{6}{l}{\footnotesize \textit{Uwaga:} Dokładność bazowa modelu bez perturbacji: " + orig_acc_str + r"\%." + r"} \\",
            r"\multicolumn{6}{l}{\footnotesize \textit{Rdzeń ściśle niezmienniczy (brak flipa przy żadnej perturbacji):} " + strict_pct_str + r"\% (" + strict_samples_str + r"/" + total_samples_str + r" próbek)." + r"} \\",
            r"\end{tabular}",
            r"\end{table}"
        ])

        return "\n".join(latex_lines)

    def save_results(self, output_dir: str = "results/perturbation_validation") -> Dict[str, str]:
        """Saves raw data, summary, LaTeX table, plots, and filtered index mapping."""
        if self.df_raw is None:
            raise RuntimeError("Najpierw uruchom metodę .run()!")

        os.makedirs(output_dir, exist_ok=True)
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        raw_path = os.path.join(output_dir, "perturbation_validation_raw.csv")
        summary_path = os.path.join(output_dir, "perturbation_validation_summary.csv")
        latex_path = os.path.join(output_dir, "perturbation_validation_table.tex")
        preserved_pairs_path = os.path.join(output_dir, "decision_preserved_pairs.json")

        self.df_raw.to_csv(raw_path, index=False)
        self.df_summary.to_csv(summary_path, index=False)

        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(self.generate_latex_table())

        # Save mapping of preserved pairs: {(sample_id, pert_name): True}
        preserved_records = self.df_raw[self.df_raw["is_preserved"]][["sample_id", "perturbation_name"]].to_dict(orient="records")
        with open(preserved_pairs_path, "w", encoding="utf-8") as f:
            json.dump(preserved_records, f, indent=2)

        # Generate plots
        self._generate_plots(plots_dir)

        print(f"\n[Validator] Wyniki Fazy 1 zapisano w '{output_dir}':")
        print(f" - Dane surowe: {raw_path}")
        print(f" - Podsumowanie: {summary_path}")
        print(f" - Tabela LaTeX: {latex_path}")
        print(f" - Zapisane poprawne pary (Decision-Preserved): {preserved_pairs_path}")
        print(f" - Wykresy: {plots_dir}")

        return {
            "raw": raw_path,
            "summary": summary_path,
            "latex": latex_path,
            "pairs": preserved_pairs_path,
            "plots": plots_dir
        }

    def _generate_plots(self, plots_dir: str):
        """Generates publication plots for perturbation validation."""
        if self.df_summary is None:
            return

        # Plot 1: Flip Rate per Perturbation
        fig, ax = plt.subplots(figsize=(13, 5.2), dpi=300)
        x = np.arange(len(self.df_summary))
        names = [n.replace("prefix_", "").replace("suffix_", "") for n in self.df_summary["perturbation_name"]]
        rates = self.df_summary["flip_rate_pct"]

        bars = ax.bar(x, rates, color="#d95f02", alpha=0.85, edgecolor="black", linewidth=0.6, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9.5)
        ax.set_ylabel("Współczynnik zmiany decyzji (Flip Rate) [%]", fontsize=11)
        ax.set_title("Wpływ perturbacji promptu na zmianę predykcji modelu (Flip Rate)", fontsize=12, fontweight="bold", pad=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="medium")

        plt.tight_layout()
        p1 = os.path.join(plots_dir, "perturbation_flip_rate.png")
        plt.savefig(p1, bbox_inches="tight")
        plt.savefig(os.path.splitext(p1)[0] + ".pdf", bbox_inches="tight")
        plt.close()

        # Plot 2: Mean Absolute Delta P
        fig, ax = plt.subplots(figsize=(13, 5.2), dpi=300)
        means = self.df_summary["mean_abs_delta_p"]
        stds = self.df_summary["std_abs_delta_p"]

        bars = ax.bar(x, means, yerr=stds, capsize=4, color="#2b5c8f", alpha=0.85, edgecolor="black", linewidth=0.6, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9.5)
        ax.set_ylabel("Średni dryf prawdopodobieństwa |ΔP|", fontsize=11)
        ax.set_title("Dryf prawdopodobieństwa etykiety modelu pod wpływem perturbacji (|ΔP|)", fontsize=12, fontweight="bold", pad=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)

        
        plt.tight_layout()
        p2 = os.path.join(plots_dir, "perturbation_prob_drift.png")
        plt.savefig(p2, bbox_inches="tight")
        plt.savefig(os.path.splitext(p2)[0] + ".pdf", bbox_inches="tight")
        plt.close()
