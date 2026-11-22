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

    def __init__(
        self,
        model_wrapper,
        max_evals: int = 100,
        mask_token_id: Optional[int] = None,
        seed: int = 42,
        mask_prompt_template: bool = True,
    ):
        super().__init__(model_wrapper)
        self.max_evals = max_evals
        self.seed = seed
        self.mask_prompt_template = mask_prompt_template
        if mask_token_id is not None:
            self.mask_token_id = mask_token_id
        else:
            self.mask_token_id = getattr(self.model_wrapper.tokenizer, "pad_token_id", None)
            if self.mask_token_id is None:
                self.mask_token_id = getattr(self.model_wrapper.tokenizer, "eos_token_id", 0)

    def explain(
        self,
        text: str,
        target_class: Optional[int] = None,
        seed: Optional[int] = None,
        baseline_prob: Optional[float] = None,
    ) -> AttributionResult:
        start_time = time.time()

        target_class, baseline_prob = self._resolve_target(text, target_class, baseline_prob)
        review_start, review_end, input_ids_list = self.model_wrapper.find_review_span(text)
        input_ids_flat = np.array(input_ids_list, dtype=np.int64)
        num_tokens = len(input_ids_flat)
        effective_seed = self.seed if seed is None else seed
        row_cache: dict[bytes, np.ndarray] = {}
        shap_batch_size = getattr(self.model_wrapper, "default_sub_batch_size", "auto")

        def _predict_cached(batch_2d: np.ndarray, fn) -> np.ndarray:
            if batch_2d.ndim == 1:
                batch_2d = batch_2d.reshape(1, -1)
            n_rows = batch_2d.shape[0]
            keys = [np.ascontiguousarray(batch_2d[i]).tobytes() for i in range(n_rows)]
            missing_rows = []
            missing_keys = []
            seen_missing = set()
            for i, k in enumerate(keys):
                if k not in row_cache and k not in seen_missing:
                    seen_missing.add(k)
                    missing_rows.append(batch_2d[i])
                    missing_keys.append(k)
            if missing_rows:
                missing_arr = np.stack(missing_rows, axis=0)
                preds = fn(missing_arr)
                for k, p in zip(missing_keys, preds):
                    row_cache[k] = p
            return np.stack([row_cache[k] for k in keys], axis=0)

        if self.mask_prompt_template and 0 <= review_start < review_end <= num_tokens:
            prefix_arr = input_ids_flat[:review_start]
            suffix_arr = input_ids_flat[review_end:]
            review_ids = input_ids_flat[review_start:review_end].reshape(1, -1)
            num_review = review_ids.shape[1]

            def _predict_review_only(rev_batch: np.ndarray) -> np.ndarray:
                def _run_batch(missing_rev: np.ndarray) -> np.ndarray:
                    B = missing_rev.shape[0]
                    parts = []
                    if prefix_arr.size > 0:
                        parts.append(np.broadcast_to(prefix_arr, (B, prefix_arr.size)))
                    parts.append(missing_rev)
                    if suffix_arr.size > 0:
                        parts.append(np.broadcast_to(suffix_arr, (B, suffix_arr.size)))
                    full_batch = np.concatenate(parts, axis=1)
                    return self.model_wrapper.predict_tokens_batch(full_batch)

                return _predict_cached(rev_batch, _run_batch)

            background = np.full(review_ids.shape, self.mask_token_id, dtype=review_ids.dtype)
            explainer = shap.explainers.PermutationExplainer(
                _predict_review_only, background, seed=effective_seed
            )
            min_evals = 2 * num_review + 1
            evals = max(self.max_evals, min_evals)
            shap_values = explainer(review_ids, max_evals=evals, batch_size=shap_batch_size)

            values = shap_values.values[0]
            rev_attr = values[:, target_class] if len(values.shape) > 1 else values
            attributions = np.zeros(num_tokens, dtype=float)
            attributions[review_start:review_end] = rev_attr
        else:
            input_ids_2d = input_ids_flat.reshape(1, -1)
            background = np.full(input_ids_2d.shape, self.mask_token_id, dtype=input_ids_2d.dtype)

            def _predict_full_cached(full_batch: np.ndarray) -> np.ndarray:
                return _predict_cached(full_batch, self.model_wrapper.predict_tokens_batch)

            explainer = shap.explainers.PermutationExplainer(
                _predict_full_cached, background, seed=effective_seed
            )
            min_evals = 2 * num_tokens + 1
            evals = max(self.max_evals, min_evals)
            shap_values = explainer(input_ids_2d, max_evals=evals, batch_size=shap_batch_size)

            values = shap_values.values[0]
            attributions = (values[:, target_class] if len(values.shape) > 1 else values).astype(float)

        tokens = [self.model_wrapper.tokenizer.decode([int(tid)]) for tid in input_ids_flat]
        execution_time = time.time() - start_time

        return AttributionResult(
            tokens=tokens,
            attributions=np.array(attributions, dtype=float),
            target_class=target_class,
            execution_time=execution_time,
            explainer_name=f"SHAP (max_evals={self.max_evals})",
            baseline_prob=baseline_prob,
            metadata={
                "seed": effective_seed,
                "mask_prompt_template": self.mask_prompt_template,
                "max_evals": self.max_evals,
                "unique_evals": len(row_cache),
            },
        )
