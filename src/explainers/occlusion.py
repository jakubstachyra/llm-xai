import time
import numpy as np
import torch
from typing import Optional
from .base import BaseExplainer, AttributionResult

class OcclusionExplainer(BaseExplainer):
    """
    Leave-One-Out (LOO) / Occlusion Token Explainer.
    Measures the drop in probability of target class when a token is removed.
    """
    def __init__(self, model_wrapper):
        super().__init__(model_wrapper)

    def explain(self, text: str, target_class: Optional[int] = None) -> AttributionResult:
        start_time = time.time()
        
        prompt = self.model_wrapper.format_prompt(text)
        input_ids_tensor = self.model_wrapper.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids_tensor[0].tolist()
        num_tokens = len(input_ids)
        
        _, probs_orig = self.model_wrapper.predict_single(text)
        if target_class is None:
            target_class = int(np.argmax(probs_orig))
            
        baseline_prob = float(probs_orig[target_class])
        
        # Build batch of occluded token sequences (removing token i)
        # Sequence batch size: num_tokens
        attributions = np.zeros(num_tokens)
        
        # We can run occlusion in a single batch
        batch_seqs = []
        max_len = num_tokens - 1
        
        for i in range(num_tokens):
            seq_without_i = input_ids[:i] + input_ids[i+1:]
            batch_seqs.append(seq_without_i)
            
        # Predict probabilities for all occluded sequences
        # Since sequences have fixed length (num_tokens - 1), we can stack them into 2D numpy array
        batch_arr = np.array(batch_seqs, dtype=np.int64)
        probs_occluded = self.model_wrapper.predict_tokens_batch(batch_arr) # shape: (num_tokens, 2)
        
        # Attribution = Baseline_Prob - Occluded_Prob
        attributions = baseline_prob - probs_occluded[:, target_class]
        
        tokens = [self.model_wrapper.tokenizer.decode([tid]) for tid in input_ids]
        execution_time = time.time() - start_time
        
        return AttributionResult(
            tokens=tokens,
            attributions=attributions,
            target_class=target_class,
            execution_time=execution_time,
            explainer_name="Occlusion (Leave-One-Out)",
            baseline_prob=baseline_prob
        )