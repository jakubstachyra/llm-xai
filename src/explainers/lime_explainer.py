import time
import numpy as np
from typing import Optional
from sklearn.linear_model import Ridge
from .base import BaseExplainer, AttributionResult


class LIMEExplainer(BaseExplainer):
    """
    Local Interpretable Model-agnostic Explanations (LIME, Ribeiro et al., 2016)
    adapted for Causal Language Model sentiment classification.
    
    Generates binary perturbation masks over input tokens, scores them via batched model inference,
    weights samples using an exponential kernel on cosine/Hamming distance, and fits a weighted
    Ridge regression surrogate model to estimate local token importance.
    """
    stochastic = True

    def __init__(
        self, 
        model_wrapper, 
        num_samples: int = 150, 
        kernel_width: float = 0.25,
        alpha: float = 1.0,
        seed: int = 42
    ):
        super().__init__(model_wrapper)
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self.alpha = alpha
        self.seed = seed

    def explain(
        self, 
        text: str, 
        target_class: Optional[int] = None, 
        seed: Optional[int] = None
    ) -> AttributionResult:
        start_time = time.time()
        
        prompt = self.model_wrapper.format_prompt(text)
        input_ids = self.model_wrapper.tokenizer.encode(
            prompt, 
            add_special_tokens=False
        )
        num_tokens = len(input_ids)
        
        # Determine target class and baseline probability
        _, probs_orig = self.model_wrapper.predict_single(text)
        if target_class is None:
            target_class = int(np.argmax(probs_orig))
            
        baseline_prob = float(probs_orig[target_class])
        
        # Identify review tokens (excluding "Review: " prefix and "\nSentiment:" suffix)
        prefix_ids = self.model_wrapper.tokenizer.encode("Review: ", add_special_tokens=False)
        suffix_ids = self.model_wrapper.tokenizer.encode("\nSentiment:", add_special_tokens=False)
        
        prefix_len = len(prefix_ids)
        suffix_len = len(suffix_ids)
        
        if num_tokens > prefix_len + suffix_len:
            review_start = prefix_len
            review_end = num_tokens - suffix_len
        else:
            review_start = 0
            review_end = num_tokens
            
        num_review_tokens = review_end - review_start
        
        if num_review_tokens <= 1:
            # Trivial case: single token
            attributions = np.zeros(num_tokens)
            if num_review_tokens == 1:
                attributions[review_start] = baseline_prob
            tokens = [self.model_wrapper.tokenizer.decode([tid]) for tid in input_ids]
            return AttributionResult(
                tokens=tokens,
                attributions=attributions,
                target_class=target_class,
                execution_time=time.time() - start_time,
                explainer_name="LIME",
                baseline_prob=baseline_prob
            )

        # Generate binary perturbation masks for review tokens: shape (num_samples, num_review_tokens)
        effective_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(effective_seed)
        
        # Mask 0 is all ones (original unperturbed instance)
        masks = rng.choice([0, 1], size=(self.num_samples, num_review_tokens), p=[0.3, 0.7])
        masks[0] = 1 # ensure original instance is included
        
        # Build batched token sequences corresponding to the masks
        prefix_part = input_ids[:review_start]
        suffix_part = input_ids[review_end:]
        review_token_ids = input_ids[review_start:review_end]
        
        perturbed_seqs = []
        for mask in masks:
            active_review = [review_token_ids[j] for j in range(num_review_tokens) if mask[j] == 1]
            if not active_review:
                # If all masked out, retain at least one token to avoid empty sequence
                active_review = [review_token_ids[0]]
                mask[0] = 1
            seq = prefix_part + active_review + suffix_part
            perturbed_seqs.append(seq)
            
        # Left-pad sequences to uniform length for batched GPU forward pass
        max_len = max(len(s) for s in perturbed_seqs)
        pad_id = self.model_wrapper.pad_token_id
        padded_batch = []
        for s in perturbed_seqs:
            padded_batch.append([pad_id] * (max_len - len(s)) + s)
            
        batch_arr = np.array(padded_batch, dtype=np.int64)
        probs_perturbed = self.model_wrapper.predict_tokens_batch(batch_arr)
        y = probs_perturbed[:, target_class]
        
        # Calculate sample weights using exponential kernel on Hamming distance
        # Normalized distance: fraction of tokens flipped to 0
        distances = 1.0 - (masks.sum(axis=1) / float(num_review_tokens))
        weights = np.exp(- (distances ** 2) / (self.kernel_width ** 2))
        
        # Fit weighted Ridge regression surrogate
        ridge = Ridge(alpha=self.alpha, fit_intercept=True)
        ridge.fit(masks, y, sample_weight=weights)

        # Compute weighted R^2 score of surrogate fit
        y_pred = ridge.predict(masks)
        weighted_mean_y = np.average(y, weights=weights)
        ss_tot = np.sum(weights * (y - weighted_mean_y) ** 2)
        ss_res = np.sum(weights * (y - y_pred) ** 2)
        r2_score = float(1.0 - (ss_res / (ss_tot + 1e-12))) if ss_tot > 0 else 1.0
        
        # Map estimated coefficients back to full prompt token attributions
        attributions = np.zeros(num_tokens)
        attributions[review_start:review_end] = ridge.coef_
        
        tokens = [self.model_wrapper.tokenizer.decode([tid]) for tid in input_ids]
        execution_time = time.time() - start_time
        
        return AttributionResult(
            tokens=tokens,
            attributions=attributions,
            target_class=target_class,
            execution_time=execution_time,
            explainer_name="LIME",
            baseline_prob=baseline_prob,
            metadata={"r2_score": r2_score, "seed": effective_seed}
        )
