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
        seed: Optional[int] = None,
        baseline_prob: Optional[float] = None,
    ) -> AttributionResult:
        start_time = time.time()
        
        target_class, baseline_prob = self._resolve_target(text, target_class, baseline_prob)
        review_start, review_end, input_ids = self.model_wrapper.find_review_span(text)
        num_tokens = len(input_ids)
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

        # Fix all-zero masks BEFORE deduplication so masks matrix stays 1:1 with Ridge targets
        empty_rows = masks.sum(axis=1) == 0
        if np.any(empty_rows):
            masks[empty_rows, 0] = 1

        # Deduplicate identical binary masks prior to GPU inference
        unique_masks, inv_idx = np.unique(masks, axis=0, return_inverse=True)

        # Build batched token sequences corresponding to unique masks only
        prefix_part = input_ids[:review_start]
        suffix_part = input_ids[review_end:]
        review_token_ids = input_ids[review_start:review_end]

        unique_seqs = []
        for mask in unique_masks:
            active_review = [review_token_ids[j] for j in range(num_review_tokens) if mask[j] == 1]
            unique_seqs.append(prefix_part + active_review + suffix_part)

        # Sort by sequence length for dynamic per-batch left-padding
        seq_lengths = np.array([len(s) for s in unique_seqs], dtype=np.int64)
        order = np.argsort(seq_lengths, kind="stable")
        sub_bs = max(1, int(getattr(self.model_wrapper, "default_sub_batch_size", 64)))
        pad_id = self.model_wrapper.pad_token_id

        unique_probs_sorted = []
        for start in range(0, len(order), sub_bs):
            chunk_indices = order[start:start + sub_bs]
            chunk_max_len = int(seq_lengths[chunk_indices[-1]])
            padded_chunk = [
                [pad_id] * (chunk_max_len - len(unique_seqs[idx])) + unique_seqs[idx]
                for idx in chunk_indices
            ]
            batch_arr = np.array(padded_chunk, dtype=np.int64)
            probs_chunk = self.model_wrapper.predict_tokens_batch(batch_arr, sub_batch_size=sub_bs)
            unique_probs_sorted.append(probs_chunk)

        unique_probs_ordered = np.vstack(unique_probs_sorted)
        unique_probs = np.empty_like(unique_probs_ordered)
        unique_probs[order] = unique_probs_ordered
        probs_perturbed = unique_probs[inv_idx]
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
            metadata={
                "r2_score": r2_score,
                "seed": effective_seed,
                "num_samples": self.num_samples,
                "unique_evals": int(len(unique_masks)),
            }
        )
