import json
import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.explainers.base import BaseExplainer, AttributionResult
from src.explainers.integrated_gradients import IntegratedGradientsExplainer
from src.experiments.rq1_stability import RQ1StabilityExperiment
from src.experiments.rq2_faithfulness import RQ2FaithfulnessExperiment
from src.models import SentimentCausalLM
from src.metrics import (
    compute_all_stability_metrics,
    stability_spearman,
    stability_kendall_tau,
    stability_pearson,
    stability_topk_jaccard,
)


class _DummyTokenizer:
    is_fast = True
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [i + 10 for i, _ in enumerate(text.split())]

    def decode(self, token_ids):
        return f"tok_{token_ids[0]}"


class _MockRQ1Model:
    def __init__(self, flip_all_perturbations: bool = False):
        self.model_name = "Qwen/Qwen2.5-3B"
        self.device = "cpu"
        self.device_name = "MockCPU"
        self.tokenizer = _DummyTokenizer()
        self.pad_token_id = 0
        self.default_sub_batch_size = 8
        self.param_count = 3_000_000_000
        self.flip_all_perturbations = flip_all_perturbations
        self.forward_pass_count = 0
        self.backward_pass_count = 0
        self.forward_token_count = 0

    def format_prompt(self, text: str) -> str:
        return f"Review: {text}\nSentiment:"

    def predict_texts_batch(self, texts):
        self.forward_pass_count += len(texts)
        if self.flip_all_perturbations and len(texts) > 1:
            out = [[0.15, 0.85]] + [[0.85, 0.15] for _ in texts[1:]]
            return np.array(out, dtype=np.float32)

        # Preserve prediction with slight probability variation across perturbation lengths
        probs = []
        for idx, t in enumerate(texts):
            shift = min(0.25, 0.008 * abs(len(t) - len(texts[0])))
            p_pos = max(0.55, 0.90 - shift) if idx > 0 else 0.90
            probs.append([1.0 - p_pos, p_pos])
        return np.array(probs, dtype=np.float32)

    def get_counters(self):
        return {
            "forward_passes": self.forward_pass_count,
            "backward_passes": self.backward_pass_count,
            "forward_tokens": self.forward_token_count,
        }

    def benchmark_baseline_forward(self, num_trials=5):
        return 0.01

    def compute_relative_fwdu(self, elapsed_seconds: float) -> float:
        return elapsed_seconds / 0.01


class StubIGExplainer(IntegratedGradientsExplainer):
    """Stub IG inheriting real config/class for cache handoff verification."""
    stochastic = False

    def __init__(self, model_wrapper, steps=25, baseline_type="mean"):
        self.model_wrapper = model_wrapper
        self.steps = steps
        self.baseline_type = baseline_type
        self.explain_calls = 0

    @property
    def explainer_name(self) -> str:
        return f"Integrated Gradients (steps={self.steps})"

    def explain(self, text: str, target_class=None, baseline_prob=None) -> AttributionResult:
        self.explain_calls += 1
        words = text.strip().split()
        tokens = ["Review:"] + [f" {w}" for w in words] + ["\nSentiment:"]
        attrs = np.zeros(len(tokens), dtype=float)
        for i, w in enumerate(words):
            attrs[1 + i] = float((sum(ord(c) for c in w.lower()) % 17) + 1) / 20.0
        return AttributionResult(
            tokens=tokens,
            attributions=attrs,
            target_class=int(target_class if target_class is not None else 1),
            execution_time=0.01,
            explainer_name=self.explainer_name,
            baseline_prob=baseline_prob,
            metadata={"steps": self.steps, "baseline_type": self.baseline_type, "completeness_error": 0.01},
        )


class StubStochasticExplainer(BaseExplainer):
    stochastic = True

    def __init__(self, model_wrapper):
        super().__init__(model_wrapper)
        self.seed = 42
        self.num_samples = 100
        self.explain_calls = 0

    def explain(self, text: str, target_class=None, seed=None, baseline_prob=None) -> AttributionResult:
        self.explain_calls += 1
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        words = text.strip().split()
        tokens = ["Review:"] + [f" {w}" for w in words] + ["\nSentiment:"]
        attrs = np.zeros(len(tokens), dtype=float)
        for i, w in enumerate(words):
            base = float((sum(ord(c) for c in w.lower()) % 13) + 1) / 15.0
            attrs[1 + i] = base + float(rng.normal(0.0, 0.05))
        return AttributionResult(
            tokens=tokens,
            attributions=attrs,
            target_class=int(target_class if target_class is not None else 1),
            execution_time=0.01,
            explainer_name="LIME",
            baseline_prob=baseline_prob,
            metadata={"r2_score": 0.92, "num_samples": self.num_samples},
        )


class TestRQ1StabilityPipeline(unittest.TestCase):
    def test_compute_all_stability_metrics_matches_individual_functions(self):
        res1 = AttributionResult(
            tokens=["Review:", " a", " great", " movie", " with", " good", " pacing", "\nSentiment:"],
            attributions=np.array([0.0, 0.1, 0.9, 0.3, 0.05, 0.6, 0.2, 0.0]),
            target_class=1,
            execution_time=0.01,
            explainer_name="Stub",
        )
        res2 = AttributionResult(
            tokens=["Review:", " overall", ",", " a", " great", " film", " with", " good", " pacing", "\nSentiment:"],
            attributions=np.array([0.0, 0.02, 0.01, 0.12, 0.85, 0.25, 0.04, 0.55, 0.30, 0.0]),
            target_class=1,
            execution_time=0.01,
            explainer_name="Stub",
        )
        all_m = compute_all_stability_metrics(res1, res2)
        assert np.isclose(all_m["spearman"], stability_spearman(res1, res2))
        assert np.isclose(all_m["kendall"], stability_kendall_tau(res1, res2))
        assert np.isclose(all_m["pearson"], stability_pearson(res1, res2))
        assert np.isclose(all_m["jaccard_k1"], stability_topk_jaccard(res1, res2, k=1))
        assert np.isclose(all_m["jaccard_k3"], stability_topk_jaccard(res1, res2, k=3))
        assert np.isclose(all_m["jaccard_k5"], stability_topk_jaccard(res1, res2, k=5))
        assert all_m["aligned_words"] == 5

    def test_rq1_end_to_end_outputs_preservation_strength_and_rq2_cache_handoff(self):
        mock_model = _MockRQ1Model(flip_all_perturbations=False)
        stub_ig = StubIGExplainer(mock_model, steps=25, baseline_type="mean")
        stub_lime = StubStochasticExplainer(mock_model)

        sample_data = [
            {"id": i, "sentence": s, "label": 1}
            for i, s in enumerate([
                "the movie was charming and deeply moving overall.",
                "a fantastic film with brilliant acting and pacing!",
                "it's an engaging story with great visual style.",
                "simply a great cinema experience from start to end.",
            ])
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "sst2_mock.json")
            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)
            out_dir = os.path.join(tmpdir, "rq1_out")
            ckpt_path = os.path.join(out_dir, "rq1_stability_raw.csv")

            exp = RQ1StabilityExperiment(
                model=mock_model,
                explainers=[stub_ig, stub_lime],
                data_path=data_file,
            )
            exp.run(
                num_samples=4,
                seed=42,
                filter_decision_preserved=True,
                progress_bar=False,
                checkpoint_path=ckpt_path,
                include_noise_floor=True,
                checkpoint_interval=2,
            )
            paths = exp.save_results(output_dir=out_dir)

            assert os.path.exists(paths["raw_csv"])
            assert os.path.exists(paths["decisions_csv"])
            assert os.path.exists(paths["preservation_summary_csv"])
            assert os.path.exists(paths["strength_analysis_csv"])
            assert os.path.exists(paths["latex_table"])

            df_raw = pd.read_csv(paths["raw_csv"])
            for col in ("char_edit_ratio", "token_len_diff", "token_len_rel_diff", "abs_delta_p", "aligned_words"):
                assert col in df_raw.columns, f"Missing column {col}"

            df_pres = pd.read_csv(paths["preservation_summary_csv"])
            assert "preservation_rate_pct" in df_pres.columns
            assert "flip_rate_pct" in df_pres.columns
            assert np.allclose(df_pres["preservation_rate_pct"] + df_pres["flip_rate_pct"], 100.0)

            df_strength = pd.read_csv(paths["strength_analysis_csv"])
            assert len(df_strength) == 2
            for col in ("rho_vs_delta_p", "rho_vs_char_edit", "spearman_delta_p_low", "spearman_edit_high"):
                assert col in df_strength.columns

            attr_jsonl = os.path.join(out_dir, "rq1_attributions.jsonl")
            assert os.path.exists(attr_jsonl)
            with open(attr_jsonl, "r", encoding="utf-8") as fa:
                recs = [json.loads(line) for line in fa if line.strip()]
            assert len(recs) > 0
            assert all(r["variant"] == "__orig__" for r in recs)
            assert all(r.get("model_name") == "Qwen/Qwen2.5-3B" for r in recs)

            # Verify RQ2 cache compatibility accepts matching IG(steps=25, mean) and rejects mismatched configs
            ig_rec = next(r for r in recs if "Integrated Gradients" in str(r.get("Canonical_name", "")))
            rq2_exp = RQ2FaithfulnessExperiment(model=mock_model, explainers=[], output_dir=tmpdir)
            assert rq2_exp._is_cache_compatible(stub_ig, ig_rec) is True
            assert rq2_exp._is_cache_compatible(StubIGExplainer(mock_model, steps=25, baseline_type="zero"), ig_rec) is False
            assert rq2_exp._is_cache_compatible(StubIGExplainer(mock_model, steps=100, baseline_type="mean"), ig_rec) is False

    def test_rq1_safe_resume_from_legacy_29_col_csv(self):
        mock_model = _MockRQ1Model(flip_all_perturbations=False)
        stub_ig = StubIGExplainer(mock_model, steps=25, baseline_type="mean")
        sample_data = [
            {"id": 0, "sentence": "the movie was charming and deeply moving.", "label": 1},
            {"id": 1, "sentence": "a fantastic film with brilliant acting.", "label": 1},
        ]
        legacy_cols = [
            "sample_id", "device_name", "explainer_name", "method_family", "param_value",
            "perturbation_category", "perturbation_name", "is_preserved", "original_text", "perturbed_text",
            "true_label", "orig_pred", "pert_pred", "p_orig", "p_pert", "orig_num_tokens",
            "spearman", "kendall", "pearson", "jaccard_k1", "jaccard_k3", "jaccard_k5",
            "orig_time_s", "pert_time_s", "vram_mb", "orig_forward_passes", "orig_backward_passes",
            "pert_forward_passes", "pert_backward_passes",
        ]
        legacy_row = {
            "sample_id": 0, "device_name": "NVIDIA A100-SXM4-40GB", "explainer_name": "StubIG",
            "method_family": "StubIG", "param_value": None, "perturbation_category": "surface",
            "perturbation_name": "uppercase", "is_preserved": True,
            "original_text": "the movie was charming and deeply moving.",
            "perturbed_text": "THE MOVIE WAS CHARMING AND DEEPLY MOVING.",
            "true_label": 1, "orig_pred": 1, "pert_pred": 1, "p_orig": 0.90, "p_pert": 0.88,
            "orig_num_tokens": 9, "spearman": 0.777, "kendall": 0.666, "pearson": 0.81,
            "jaccard_k1": 1.0, "jaccard_k3": 0.5, "jaccard_k5": 0.75,
            "orig_time_s": 0.5, "pert_time_s": 0.5, "vram_mb": 15000.0,
            "orig_forward_passes": 27, "orig_backward_passes": 25,
            "pert_forward_passes": 27, "pert_backward_passes": 25,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "sst2_mock.json")
            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)
            out_dir = os.path.join(tmpdir, "rq1_legacy")
            os.makedirs(out_dir, exist_ok=True)
            ckpt_path = os.path.join(out_dir, "rq1_stability_raw.csv")
            pd.DataFrame([legacy_row], columns=legacy_cols).to_csv(ckpt_path, index=False)

            exp = RQ1StabilityExperiment(model=mock_model, explainers=[stub_ig], data_path=data_file)
            exp.run(
                num_samples=2,
                seed=42,
                filter_decision_preserved=True,
                progress_bar=False,
                checkpoint_path=ckpt_path,
                include_noise_floor=False,
                checkpoint_interval=1,
            )
            exp.save_results(output_dir=out_dir)

            df_merged = pd.read_csv(ckpt_path)
            assert set(df_merged["sample_id"].unique()) == {0, 1}
            s0 = df_merged[df_merged["sample_id"] == 0].iloc[0]
            assert np.isclose(s0["spearman"], 0.777)
            assert np.isclose(s0["jaccard_k3"], 0.5)
            s1 = df_merged[df_merged["sample_id"] == 1]
            assert len(s1) > 0
            assert s1["char_edit_ratio"].notna().all()
            assert s1["aligned_words"].notna().all()

    def test_rq1_skips_orig_explain_when_all_perturbations_flip(self):
        mock_model = _MockRQ1Model(flip_all_perturbations=True)
        stub_ig = StubIGExplainer(mock_model, steps=25, baseline_type="mean")
        sample_data = [
            {"id": 0, "sentence": "the movie was charming and deeply moving.", "label": 1},
            {"id": 1, "sentence": "a fantastic film with brilliant acting.", "label": 1},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "sst2_mock.json")
            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)
            out_dir = os.path.join(tmpdir, "rq1_flips")
            ckpt_path = os.path.join(out_dir, "rq1_stability_raw.csv")

            exp = RQ1StabilityExperiment(model=mock_model, explainers=[stub_ig], data_path=data_file)
            exp.run(
                num_samples=2,
                seed=42,
                filter_decision_preserved=True,
                progress_bar=False,
                checkpoint_path=ckpt_path,
                include_noise_floor=False,
            )
            assert stub_ig.explain_calls == 0
            dec_csv = os.path.join(out_dir, "rq1_perturbation_decisions.csv")
            assert os.path.exists(dec_csv)
            df_dec = pd.read_csv(dec_csv)
            assert len(df_dec) > 0
            assert (df_dec["is_preserved"] == False).all()

    def test_effective_sub_batch_size_scales_down_on_long_sequences(self):
        dummy = SentimentCausalLM.__new__(SentimentCausalLM)
        dummy.default_sub_batch_size = 24
        dummy.max_tokens_per_batch = 768

        assert dummy._effective_sub_batch_size(None, seq_len=20) == 24
        assert dummy._effective_sub_batch_size(None, seq_len=40) == 19
        assert dummy._effective_sub_batch_size(None, seq_len=95) == 8
        assert dummy._effective_sub_batch_size(None, seq_len=300) == 4
        assert dummy._effective_sub_batch_size(16, seq_len=95) == 8
        assert dummy._effective_sub_batch_size(6, seq_len=95) == 6


if __name__ == "__main__":
    unittest.main()
