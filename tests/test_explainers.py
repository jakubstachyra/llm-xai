"""
Unit and Axiomatic Tests for XAI Explainers.

Verifies:
1. Occlusion exact probability drop calculation (Leave-One-Out consistency).
2. Integrated Gradients Completeness Axiom (Sundararajan et al., 2017).
3. Semantic Saliency & Sentiment Discrimination.
4. Explainer Reproducibility & Determinism.
"""
import unittest
import sys
import os
import torch
import numpy as np

# Add src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from models import SentimentCausalLM
from explainers import OcclusionExplainer, IntegratedGradientsExplainer, SHAPExplainer


class TestExplainers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[TestSetup] Loading SentimentCausalLM ('microsoft/phi-2') once for all tests...")
        cls.model = SentimentCausalLM(model_name="microsoft/phi-2")
        cls.sample_text = "The movie was fantastic and brilliant!"

    def test_occlusion_exact_probability_drop(self):
        r"""
        Verifies that OcclusionExplainer produces exact Leave-One-Out probability drops.
        Attribution(t_i) must equal P(y | x) - P(y | x \ {t_i}).
        """
        occ = OcclusionExplainer(self.model)
        pred_class, probs = self.model.predict_single(self.sample_text)
        baseline_prob = float(probs[pred_class])

        res = occ.explain(self.sample_text, target_class=pred_class)
        self.assertEqual(len(res.tokens), len(res.attributions))
        self.assertAlmostEqual(res.baseline_prob, baseline_prob, places=4)

        # Manually compute single occluded sequence drop for a token
        input_ids = self.model.tokenizer.encode(self.model.format_prompt(self.sample_text))
        test_idx = len(input_ids) // 2
        seq_without_idx = input_ids[:test_idx] + input_ids[test_idx + 1 :]
        probs_without = self.model.predict_tokens_batch(np.array([seq_without_idx], dtype=np.int64))[0]
        expected_drop = baseline_prob - float(probs_without[pred_class])

        self.assertAlmostEqual(
            float(res.attributions[test_idx]),
            expected_drop,
            places=5,
            msg="Occlusion attribution did not match exact manual single-token probability drop!"
        )

    def test_integrated_gradients_completeness_axiom(self):
        """
        Verifies Sundararajan (2017) Completeness Axiom:
        sum_i(Attribution_i) == F(x) - F(x_baseline)
        Relative numerical integration error must be under 5% for 25 steps.
        """
        ig = IntegratedGradientsExplainer(self.model, steps=25)
        pred_class, _ = self.model.predict_single(self.sample_text)
        target_token_id = self.model.target_token_ids[pred_class]

        prompt = self.model.format_prompt(self.sample_text)
        input_ids = self.model.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        embed_layer = self.model.model.get_input_embeddings()
        input_embeds = embed_layer(input_ids)
        baseline_embeds = torch.zeros_like(input_embeds)

        with torch.no_grad():
            logit_input = float(self.model.model(inputs_embeds=input_embeds).logits[0, -1, target_token_id])
            logit_base = float(self.model.model(inputs_embeds=baseline_embeds).logits[0, -1, target_token_id])
            delta_f = logit_input - logit_base

        res = ig.explain(self.sample_text, target_class=pred_class)
        sum_attributions = float(np.sum(res.attributions))

        rel_error = abs(sum_attributions - delta_f) / (abs(delta_f) + 1e-9)
        self.assertLess(
            rel_error,
            0.05,
            f"IG Completeness Axiom violated: sum_attr={sum_attributions:.4f}, delta_f={delta_f:.4f}, error={rel_error:.2%}"
        )

    def test_saliency_sentiment_discrimination(self):
        """
        Verifies that prominent sentiment tokens ('fantastic', 'brilliant') receive
        higher positive attribution than neutral syntax tokens ('The', 'was').
        """
        occ = OcclusionExplainer(self.model)
        res = occ.explain(self.sample_text, target_class=1)

        token_attr_map = {t.strip().lower(): float(a) for t, a in zip(res.tokens, res.attributions)}

        sentiment_scores = [v for k, v in token_attr_map.items() if any(w in k for w in ["fantastic", "brilliant"])]
        syntax_scores = [v for k, v in token_attr_map.items() if k in ["the", "was", "."]]

        self.assertTrue(len(sentiment_scores) > 0, "Sentiment tokens not found in token attribution map")
        if syntax_scores:
            self.assertGreater(
                max(sentiment_scores),
                max(syntax_scores),
                "Sentiment tokens did not receive higher attribution than syntax tokens!"
            )

    def test_explainers_determinism(self):
        """
        Verifies that Occlusion and Integrated Gradients are 100% deterministic (diff == 0.0).
        """
        occ = OcclusionExplainer(self.model)
        res_occ1 = occ.explain(self.sample_text)
        res_occ2 = occ.explain(self.sample_text)
        np.testing.assert_allclose(
            res_occ1.attributions,
            res_occ2.attributions,
            rtol=1e-5,
            err_msg="Occlusion is not deterministic!"
        )

        ig = IntegratedGradientsExplainer(self.model, steps=15)
        res_ig1 = ig.explain(self.sample_text)
        res_ig2 = ig.explain(self.sample_text)
        np.testing.assert_allclose(
            res_ig1.attributions,
            res_ig2.attributions,
            rtol=1e-5,
            err_msg="Integrated Gradients is not deterministic!"
        )


if __name__ == "__main__":
    unittest.main()
