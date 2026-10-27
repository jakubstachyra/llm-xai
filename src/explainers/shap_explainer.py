import time
import shap
import numpy as np
from typing import Optional
from .base import BaseExplainer, AttributionResult

class SHAPExplainer(BaseExplainer):
    """
    SHAP Permutation Explainer for Causal LLM Token Attribution.
    """
    stochastic = True

    def __init__(self, model_wrapper, max_evals: int = 100, mask_token_id: Optional[int] = None, seed: int = 42):
        super().__init__(model_wrapper)
        self.max_evals = max_evals
        self.seed = seed
        if mask_token_id is not None:
            self.mask_token_id = mask_token_id
        else:
            self.mask_token_id = getattr(self.model_wrapper.tokenizer, "pad_token_id", None)
            if self.mask_token_id is None:
                self.mask_token_id = getattr(self.model_wrapper.tokenizer, "eos_token_id", 0)

    def explain(self, text: str, target_class: Optional[int] = None, seed: Optional[int] = None) -> AttributionResult:
        start_time = time.time()
        
        prompt = self.model_wrapper.format_prompt(text)
        input_ids = self.model_wrapper.tokenizer.encode(prompt, return_tensors="np")
        input_ids_flat = input_ids[0]
        
        _, probs = self.model_wrapper.predict_single(text)
        if target_class is None:
            target_class = int(np.argmax(probs))
            
        baseline_prob = float(probs[target_class])
        
        background = np.full(input_ids.shape, self.mask_token_id, dtype=input_ids.dtype)
        effective_seed = self.seed if seed is None else seed
        explainer = shap.explainers.PermutationExplainer(
            self.model_wrapper.predict_tokens_batch, background, seed=effective_seed
        )
        
        # PermutationExplainer requires at least 2 * num_features + 1
        min_evals = 2 * len(input_ids_flat) + 1
        evals = max(self.max_evals, min_evals)
        shap_values = explainer(input_ids, max_evals=evals)
        
        # shap_values.values shape: (1, seq_len, 2) or (1, seq_len)
        values = shap_values.values[0]
        if len(values.shape) > 1:
            attributions = values[:, target_class]
        else:
            attributions = values
            
        tokens = [self.model_wrapper.tokenizer.decode([tid]) for tid in input_ids_flat]
        execution_time = time.time() - start_time
        
        return AttributionResult(
            tokens=tokens,
            attributions=np.array(attributions, dtype=float),
            target_class=target_class,
            execution_time=execution_time,
            explainer_name=f"SHAP (max_evals={self.max_evals})",
            baseline_prob=baseline_prob
        )
