import unittest
import numpy as np
from src.models import SentimentCausalLM
from src.explainers import GradientXInputExplainer, LIMEExplainer


class TestNewExplainers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n[TestSetup] Loading SentimentCausalLM ('microsoft/phi-2') for new explainer tests...")
        cls.model = SentimentCausalLM(model_name="microsoft/phi-2")
        cls.pos_text = "The movie was fantastic and brilliant!"
        cls.neg_text = "The acting was terrible and utterly dreadful."

    def test_gradient_x_input_execution_and_sentiment_direction(self):
        explainer = GradientXInputExplainer(self.model)
        res_pos = explainer.explain(self.pos_text, target_class=1)

        assert res_pos is not None
        assert len(res_pos.tokens) == len(res_pos.attributions)
        assert res_pos.execution_time > 0.0
        assert not np.isnan(res_pos.attributions).any()
        assert not np.isinf(res_pos.attributions).any()

        # Top tokens should contain positive words
        top_k = res_pos.top_k_tokens(k=3)
        top_tokens = [tok.strip().lower() for tok, score in top_k]
        assert any(w in top_tokens for w in ["fantastic", "brilliant"]), (
            f"Expected positive sentiment tokens, got {top_tokens}"
        )

    def test_lime_explainer_execution_and_determinism(self):
        explainer1 = LIMEExplainer(self.model, num_samples=30, seed=42)
        explainer2 = LIMEExplainer(self.model, num_samples=30, seed=42)

        res1 = explainer1.explain(self.pos_text, target_class=1)
        res2 = explainer2.explain(self.pos_text, target_class=1)

        assert len(res1.tokens) == len(res1.attributions)
        assert not np.isnan(res1.attributions).any()
        
        # Test determinism given identical seed
        np.testing.assert_allclose(
            res1.attributions, 
            res2.attributions, 
            atol=1e-5, 
            err_msg="LIME explanations must be deterministic with identical seed"
        )
