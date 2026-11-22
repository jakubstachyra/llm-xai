import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
import numpy as np
import pandas as pd
from src.faithfulness import (
    WordUnit,
    apply_word_deletion,
    apply_word_retention,
    apply_embedding_mask,
    evaluate_faithfulness_curves,
    calculate_aopc,
    compute_normalized_gap,
)
from src.explainers.base import BaseExplainer, AttributionResult
from src.explainers.integrated_gradients import IntegratedGradientsExplainer
from src.explainers.lime_explainer import LIMEExplainer
from src.explainers.shap_explainer import SHAPExplainer
from src.experiments.rq2_faithfulness import RQ2FaithfulnessExperiment
from src.experiments.link_rq1_rq2 import generate_joint_rq1_rq2_analysis


from sklearn.linear_model import Ridge
import shap


class _AttrDict(dict):
    """Dict subclass supporting attribute access like HuggingFace BatchEncoding."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _WhitespaceTokenizer:
    """Fast offline mock tokenizer matching HuggingFace fast offset_mapping contract."""
    is_fast = True
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text_or_list, return_offsets_mapping=False, add_special_tokens=False, return_tensors=None, padding=False):
        if isinstance(text_or_list, list):
            ids_list = [self.encode(t) for t in text_or_list]
            max_len = max((len(x) for x in ids_list), default=1)
            padded = [[0] * (max_len - len(x)) + x for x in ids_list]
            masks = [[0] * (max_len - len(x)) + [1] * len(x) for x in ids_list]
            return _AttrDict({
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            })

        text = text_or_list
        ids = []
        offsets = []
        i = 0
        n = len(text)
        while i < n:
            if text[i].isspace():
                j = i + 1
                while j < n and text[j].isspace():
                    j += 1
                i = j
                continue
            j = i + 1
            while j < n and not text[j].isspace() and text[j].isalnum() == text[i].isalnum():
                j += 1
            word = text[i:j]
            ids.append((abs(hash(word)) % 5000) + 10)
            offsets.append((i, j))
            i = j
        out = _AttrDict({"input_ids": ids})
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        if return_tensors == "pt":
            out["input_ids"] = torch.tensor([ids], dtype=torch.long)
        return out

    def encode(self, text, add_special_tokens=False):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def decode(self, token_ids):
        return f"tok_{token_ids[0]}"


class _MockCausalModelWrapper:
    """Offline deterministic mock for SentimentCausalLM supporting texts, token arrays, and embeds."""
    def __init__(self, model_name="mock/model", sub_batch_size=8):
        self.model_name = model_name
        self.device = "cpu"
        self.device_name = "MockCPU"
        self.tokenizer = _WhitespaceTokenizer()
        self.pad_token_id = 0
        self.default_sub_batch_size = sub_batch_size
        self.forward_pass_count = 0
        self.backward_pass_count = 0
        self.forward_token_count = 0
        self.batch_call_sizes = []
        self._embed_layer = torch.nn.Embedding(6000, 8)
        with torch.no_grad():
            self._embed_layer.weight.fill_(1.0)
            self._embed_layer.weight[0].zero_()

    def format_prompt(self, text: str) -> str:
        return f"Review: {text}\nSentiment:"

    def encode_prompt(self, text: str, return_tensors="pt"):
        return self.tokenizer(self.format_prompt(text), return_tensors=return_tensors)["input_ids"]

    def find_review_span(self, text: str, num_tokens=None):
        prompt = self.format_prompt(text)
        encoded = self.tokenizer(prompt, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        t_start = len("Review: ")
        t_end = t_start + len(text)
        matching = [i for i, (s, e) in enumerate(encoded["offset_mapping"]) if max(s, t_start) < min(e, t_end)]
        return matching[0], matching[-1] + 1, ids

    def get_input_embeddings(self):
        return self._embed_layer

    def _score_text(self, text: str):
        pos_words = ("fantastic", "brilliant", "great", "charming", "engaging", "moving")
        neg_words = ("terrible", "awful", "bad", "boring", "dreadful")
        low = text.lower()
        score = 0.35 * sum(w in low for w in pos_words) - 0.35 * sum(w in low for w in neg_words) + 0.05 * len(text.split())
        p_pos = float(1.0 / (1.0 + np.exp(-score)))
        return np.array([1.0 - p_pos, p_pos], dtype=np.float32)

    def predict_single(self, text: str):
        self.forward_pass_count += 1
        probs = self._score_text(text)
        return int(np.argmax(probs)), probs

    def predict_texts_batch(self, texts):
        self.forward_pass_count += len(texts)
        return np.vstack([self._score_text(t) for t in texts])

    def predict_tokens_batch(self, token_ids_batch, sub_batch_size=None):
        arr = np.asarray(token_ids_batch)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self.batch_call_sizes.append(len(arr))
        self.forward_pass_count += len(arr)
        non_pad = (arr != 0)
        non_pad_cnt = np.maximum(non_pad.sum(axis=1), 1)
        logits = (non_pad.sum(axis=1) * 0.15) + (((arr % 7) * non_pad).sum(axis=1) / non_pad_cnt * 0.25)
        p_pos = 1.0 / (1.0 + np.exp(-logits))
        return np.stack([1.0 - p_pos, p_pos], axis=1).astype(np.float32)

    def predict_embeddings(self, inputs_embeds, attention_mask=None):
        self.forward_pass_count += int(inputs_embeds.shape[0])
        active_mass = inputs_embeds.abs().sum(dim=(1, 2))
        p_pos = torch.sigmoid(0.05 * active_mass - 0.5)
        return torch.stack([1.0 - p_pos, p_pos], dim=1)

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


class _MockKeywordExplainer(BaseExplainer):
    """Offline mock explainer assigning higher score to positive/longer review tokens."""
    def __init__(self, model_wrapper, name_override: str, bias: float = 0.0):
        super().__init__(model_wrapper)
        self._name_override = name_override
        self._bias = bias

    @property
    def explainer_name(self) -> str:
        return self._name_override

    def explain(self, text: str, target_class=None, baseline_prob=None) -> AttributionResult:
        rs, re_idx, ids = self.model_wrapper.find_review_span(text)
        tokens = [self.model_wrapper.tokenizer.decode([tid]) for tid in ids]
        attrs = np.zeros(len(ids), dtype=float)
        pos_hints = {"fantastic", "brilliant", "great", "charming", "engaging", "moving"}
        words = text.split()
        for offset_i, idx in enumerate(range(rs, re_idx)):
            w = words[offset_i].strip(".,!").lower() if offset_i < len(words) else ""
            attrs[idx] = (0.9 if w in pos_hints else 0.05 * (offset_i + 1)) + self._bias
        return AttributionResult(
            tokens=tokens,
            attributions=attrs,
            target_class=int(target_class or 1),
            execution_time=0.01,
            explainer_name=self._name_override,
            baseline_prob=baseline_prob,
        )


class TestFaithfulnessOperatorsAndMetrics(unittest.TestCase):
    def test_discrete_word_deletion_and_retention(self):
        text = "The movie was extraordinarily fantastic and brilliant!"
        # Manually create mock WordUnits with exact character offsets
        # "extraordinarily" is at chars 14:29
        # "fantastic" is at chars 30:39
        unit_extra = WordUnit(index=0, text="extraordinarily", start_char=14, end_char=29, score=0.8)
        unit_fantastic = WordUnit(index=1, text="fantastic", start_char=30, end_char=39, score=0.9)
        unit_brilliant = WordUnit(index=2, text="brilliant", start_char=44, end_char=53, score=0.7)
        all_units = [unit_extra, unit_fantastic, unit_brilliant]

        # Deletion of Top-1 ('fantastic')
        deleted_text = apply_word_deletion(text, [unit_fantastic])
        assert "fantastic" not in deleted_text
        assert "extraordinarily" in deleted_text
        assert "brilliant" in deleted_text
        assert deleted_text == "The movie was extraordinarily and brilliant!"

        # Deletion of multiple units
        deleted_two = apply_word_deletion(text, [unit_extra, unit_fantastic])
        assert "extraordinarily" not in deleted_two
        assert "fantastic" not in deleted_two
        assert deleted_two == "The movie was and brilliant!"

        # Retention of only 'fantastic'
        retained = apply_word_retention(text, all_units, [unit_fantastic])
        assert "fantastic" in retained
        assert "extraordinarily" not in retained
        assert "brilliant" not in retained

    def test_embedding_mask_operator(self):
        # Batch size 1, sequence length 6, embedding dim 4
        embeds = torch.ones((1, 6, 4), dtype=torch.float32)
        unit = WordUnit(index=0, text="test", start_char=0, end_char=4, token_indices=[2, 3])

        masked = apply_embedding_mask(embeds, [unit], mask_value="zero")
        
        # Indices 2 and 3 should be exactly zero
        assert torch.all(masked[:, 2, :] == 0.0)
        assert torch.all(masked[:, 3, :] == 0.0)
        
        # Indices 0, 1, 4, 5 should remain untouched (all ones)
        assert torch.all(masked[:, 0, :] == 1.0)
        assert torch.all(masked[:, 1, :] == 1.0)
        assert torch.all(masked[:, 4, :] == 1.0)
        assert torch.all(masked[:, 5, :] == 1.0)

    def test_metrics_aopc_and_normalized_gap(self):
        k_values = [1, 2, 3]
        c_prob = [0.10, 0.20, 0.30] # mean = 0.20
        c_log = [0.20, 0.40, 0.60]  # mean = 0.40
        u_prob = [0.05, 0.05, 0.05] # mean = 0.05
        u_log = [0.10, 0.10, 0.10]  # mean = 0.10
        random_aopc = 0.08

        res = evaluate_faithfulness_curves(
            k_values=k_values,
            c_prob=c_prob,
            c_log=c_log,
            u_prob=u_prob,
            u_log=u_log,
            random_aopc_c_prob=random_aopc
        )

        assert np.isclose(res.aopc_comprehensiveness_prob, 0.20)
        assert np.isclose(res.aopc_comprehensiveness_log, 0.40)
        assert np.isclose(res.aopc_sufficiency_prob, 0.05)
        assert np.isclose(res.harmonization_h, 0.20 - 0.05)

        # G metric: (0.20 - 0.08) / 0.08 = 0.12 / 0.08 = 1.50 (+150% gain over random)
        expected_g = (0.20 - 0.08) / (0.08 + 1e-6)
        assert np.isclose(res.normalized_gap_g, expected_g, atol=1e-3)

    def test_rq2_dual_operator_with_least_important_and_robustness(self):
        mock_model = _MockCausalModelWrapper()
        explainers = [
            _MockKeywordExplainer(mock_model, "Occlusion (Leave-One-Out)", bias=0.1),
            _MockKeywordExplainer(mock_model, "Gradient x Input", bias=0.02),
            _MockKeywordExplainer(mock_model, "LIME", bias=-0.01),
        ]
        sample_data = [
            {"id": i, "sentence": s, "label": 1}
            for i, s in enumerate([
                "The movie was fantastic and truly brilliant with moving acting.",
                "A charming and engaging film with great visual polish.",
                "Really great storytelling and fantastic dialogue throughout the movie.",
                "An engaging drama with brilliant pacing and moving scenes.",
            ])
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "sst2_mock.json")
            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)
            out_dir = os.path.join(tmpdir, "rq2_out")

            exp = RQ2FaithfulnessExperiment(
                model=mock_model,
                explainers=explainers,
                operator="both",
                max_k=3,
                random_trials=4,
                data_path=data_file,
                output_dir=out_dir,
            )
            exp.run(num_samples=4, seed=42, resume=False)

            for op in ("deletion", "embedding_mask"):
                op_dir = f"{out_dir}_{op}"
                raw_csv = os.path.join(op_dir, "rq2_faithfulness_raw.csv")
                sum_csv = os.path.join(op_dir, "rq2_faithfulness_summary.csv")
                stats_csv = os.path.join(op_dir, "rq2_statistical_tests.csv")
                tex_file = os.path.join(op_dir, "rq2_table.tex")
                assert os.path.exists(raw_csv)
                assert os.path.exists(sum_csv)
                assert os.path.exists(stats_csv)
                assert os.path.exists(tex_file)

                df_raw = pd.read_csv(raw_csv)
                for col in ("aopc_least_prob", "aopc_least_log", "gap_g_least", "c_least_prob_k1"):
                    assert col in df_raw.columns, f"Missing column {col}"

                non_rand = df_raw[df_raw["explainer_name"] != "Random Baseline"]
                assert non_rand["aopc_least_prob"].notna().all()

                df_stats = pd.read_csv(stats_csv)
                assert len(df_stats) == 3
                for col in ("p_holm_vs_random", "p_holm_vs_least"):
                    assert col in df_stats.columns

                with open(tex_file, "r", encoding="utf-8") as ft:
                    tex = ft.read()
                assert r"\begin{tabular}{lcccccc}" in tex
                assert r"\mathrm{AOPC}_{\text{least}}" in tex
                # Verify 7-column LaTeX header matches every data row (6 '&' separators)
                header_line = [ln for ln in tex.splitlines() if r"\textbf{Metoda XAI}" in ln][0]
                assert header_line.count("&") == 6

                for p_name in (
                    "rq2_comprehensiveness_curves.png",
                    "rq2_morf_vs_lerf_and_sufficiency.png",
                    "rq2_aopc_comparison.png",
                ):
                    assert os.path.exists(os.path.join(op_dir, "plots", p_name))

            comp_dir = f"{out_dir}_operator_comparison"
            assert os.path.exists(os.path.join(comp_dir, "operator_robustness_methods.csv"))
            assert os.path.exists(os.path.join(comp_dir, "operator_robustness_ranking.json"))
            assert os.path.exists(os.path.join(comp_dir, "operator_robustness_table.tex"))
            assert os.path.exists(os.path.join(comp_dir, "plots", "operator_robustness_comparison.png"))

            # Verify RQ3 joint analysis still merges cleanly
            rq1_sum = os.path.join(tmpdir, "rq1_summary.csv")
            pd.DataFrame([
                {"explainer_name": "Occlusion", "perturbation_name": "synonym_domain", "spearman_mean": 0.65, "kendall_mean": 0.55, "jaccard_k3_mean": 0.50},
                {"explainer_name": "Gradient x Input", "perturbation_name": "synonym_domain", "spearman_mean": 0.45, "kendall_mean": 0.35, "jaccard_k3_mean": 0.30},
                {"explainer_name": "LIME", "perturbation_name": "synonym_domain", "spearman_mean": 0.72, "kendall_mean": 0.60, "jaccard_k3_mean": 0.55},
            ]).to_csv(rq1_sum, index=False)
            joint_df, _ = generate_joint_rq1_rq2_analysis(
                rq1_summary_path=rq1_sum,
                rq2_summary_path=os.path.join(f"{out_dir}_deletion", "rq2_faithfulness_summary.csv"),
                output_dir=os.path.join(tmpdir, "rq3"),
            )
            assert len(joint_df) == 3

    def test_rq2_cache_compatibility_guard(self):
        mock_model = _MockCausalModelWrapper(model_name="Qwen/Qwen2.5-7B")
        exp = RQ2FaithfulnessExperiment(model=mock_model, explainers=[], output_dir=tempfile.gettempdir())
        ig_mean = IntegratedGradientsExplainer(mock_model, steps=100, baseline_type="mean")
        ig_zero = IntegratedGradientsExplainer(mock_model, steps=100, baseline_type="zero")

        cached_old_zero = {
            "tokens": ["a", "b"],
            "attributions": [0.1, 0.2],
            "explainer_name": "Integrated Gradients (steps=100)",
            "metadata": {"steps": 100, "baseline_type": "zero"},
        }
        assert exp._is_cache_compatible(ig_zero, cached_old_zero) is True
        assert exp._is_cache_compatible(ig_mean, cached_old_zero) is False
        assert exp._is_cache_compatible(ig_zero, cached_old_zero, overwrite_methods=[ig_zero.explainer_name]) is False

        cached_wrong_steps = {
            "tokens": ["a", "b"],
            "attributions": [0.1, 0.2],
            "explainer_name": "Integrated Gradients (steps=50)",
            "metadata": {"steps": 50, "baseline_type": "zero"},
        }
        assert exp._is_cache_compatible(ig_zero, cached_wrong_steps) is False

        cached_other_model = {
            "model_name": "microsoft/phi-2",
            "tokens": ["a", "b"],
            "attributions": [0.1, 0.2],
            "metadata": {"steps": 100, "baseline_type": "zero"},
        }
        assert exp._is_cache_compatible(ig_zero, cached_other_model) is False

    def test_lime_dedup_and_dynamic_padding_determinism(self):
        mock_model = _MockCausalModelWrapper(sub_batch_size=8)
        lime = LIMEExplainer(mock_model, num_samples=120, seed=42)
        text = "Short great film."
        res1 = lime.explain(text, target_class=1, baseline_prob=0.8)
        res2 = lime.explain(text, target_class=1, baseline_prob=0.8)

        np.testing.assert_allclose(res1.attributions, res2.attributions, atol=1e-7)
        assert res1.metadata["num_samples"] == 120
        assert res1.metadata["unique_evals"] < 120
        assert mock_model.forward_pass_count == 2 * res1.metadata["unique_evals"]

        # Exact equivalence against un-deduplicated manual Ridge reference
        rs, re_idx, input_ids = mock_model.find_review_span(text)
        n_rev = re_idx - rs
        rng = np.random.default_rng(42)
        masks = rng.choice([0, 1], size=(120, n_rev), p=[0.3, 0.7])
        masks[0] = 1
        seqs = []
        for m in masks:
            act = [input_ids[rs + j] for j in range(n_rev) if m[j] == 1]
            if not act:
                act = [input_ids[rs]]
                m[0] = 1
            seqs.append(input_ids[:rs] + act + input_ids[re_idx:])
        max_len = max(len(s) for s in seqs)
        padded = np.array([[0] * (max_len - len(s)) + s for s in seqs], dtype=np.int64)
        y_ref = mock_model.predict_tokens_batch(padded)[:, 1]
        dists = 1.0 - (masks.sum(axis=1) / float(n_rev))
        weights = np.exp(-(dists ** 2) / (lime.kernel_width ** 2))
        ridge = Ridge(alpha=lime.alpha, fit_intercept=True).fit(masks, y_ref, sample_weight=weights)
        np.testing.assert_allclose(res1.attributions[rs:re_idx], ridge.coef_, atol=1e-6)

    def test_shap_in_call_cache_exactness(self):
        mock_model = _MockCausalModelWrapper(sub_batch_size=8)
        shap_exp = SHAPExplainer(mock_model, max_evals=60, seed=42)

        text1 = "A great movie."
        res1 = shap_exp.explain(text1, target_class=1, baseline_prob=0.8)
        calls_after_first = mock_model.forward_pass_count
        res2 = shap_exp.explain("A completely different sentence.", target_class=1, baseline_prob=0.8)

        assert res1.metadata["max_evals"] == 60
        assert res1.metadata["unique_evals"] < 60
        assert calls_after_first == res1.metadata["unique_evals"]
        # Second sentence must not reuse first sentence's cache
        assert mock_model.forward_pass_count > calls_after_first
        assert not np.isnan(res2.attributions).any()

        # Exact equivalence against uncached PermutationExplainer
        rs, re_idx, ids = mock_model.find_review_span(text1)
        ids_arr = np.array(ids, dtype=np.int64)
        prefix_arr = ids_arr[:rs]
        suffix_arr = ids_arr[re_idx:]
        rev_ids = ids_arr[rs:re_idx].reshape(1, -1)
        bg = np.full(rev_ids.shape, shap_exp.mask_token_id, dtype=rev_ids.dtype)

        def _uncached_fn(rev_b):
            if rev_b.ndim == 1:
                rev_b = rev_b.reshape(1, -1)
            B = rev_b.shape[0]
            full_b = np.concatenate(
                [np.broadcast_to(prefix_arr, (B, prefix_arr.size)), rev_b, np.broadcast_to(suffix_arr, (B, suffix_arr.size))],
                axis=1,
            )
            return mock_model.predict_tokens_batch(full_b)

        ref_exp = shap.explainers.PermutationExplainer(_uncached_fn, bg, seed=42)
        ref_vals = ref_exp(rev_ids, max_evals=max(60, 2 * rev_ids.shape[1] + 1), batch_size=8).values[0][:, 1]
        np.testing.assert_allclose(res1.attributions[rs:re_idx], ref_vals, atol=1e-6)
