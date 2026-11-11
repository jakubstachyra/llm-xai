import time
import torch
import numpy as np
from typing import Optional, Literal, Tuple
from .base import BaseExplainer, AttributionResult


class IntegratedGradientsExplainer(BaseExplainer):
    """
    Integrated Gradients (Sundararajan et al., 2017) adapted for Causal Language Models
    with LayerNorm / RMSNorm architectures (Sikdar et al., 2021; Enguehard, 2023).

    Key improvements over naive zero-baseline IG on autoregressive Transformers:
    1. Non-singular embedding baseline ('mean' vocabulary embedding or 'pad') avoids the
       RMSNorm(0) = 0 singularity where path scaling alpha * x is cancelled by normalization
       for all alpha > 0 and gradients explode as 1/alpha near alpha -> 0.
    2. Prompt-template freezing (mask_prompt_template=True): interpolates only the review
       tokens while keeping prompt scaffolding ('Review: ', '\\nSentiment:') fixed at alpha=1,
       preserving the classification instruction along the entire integration path.
    3. Contrastive target objective ('logit_diff'): differentiates the binary logit margin
       z(target) - z(other) instead of a single raw logit, removing common-mode logit shifts
       under prompt perturbations.
    4. Midpoint Riemann quadrature ((k - 0.5)/m) with float32 gradient accumulation to
       prevent bfloat16 rounding/cancellation across hidden_dim.
    """
    def __init__(
        self,
        model_wrapper,
        steps: int = 25,
        baseline_type: Literal["mean", "pad", "zero"] = "zero",
        target_mode: Optional[Literal["logit_diff", "logit"]] = None,
        mask_prompt_template: Optional[bool] = None,
        quadrature: Optional[Literal["midpoint", "riemann_right", "trapezoid"]] = None,
    ):
        super().__init__(model_wrapper)
        self.steps = steps
        self.baseline_type = baseline_type
        # When baseline_type == "zero", default to standard IG settings used in RQ1 for consistency;
        # when baseline_type == "mean" (or "pad"), enable LayerNorm-stabilized settings by default.
        is_stabilized = (baseline_type != "zero")
        self.target_mode = target_mode if target_mode is not None else ("logit_diff" if is_stabilized else "logit")
        self.mask_prompt_template = mask_prompt_template if mask_prompt_template is not None else is_stabilized
        self.quadrature = quadrature if quadrature is not None else ("midpoint" if is_stabilized else "riemann_right")
        self._cached_mean_embed: Optional[torch.Tensor] = None

    def _find_review_span(self, prompt: str, text: str, num_tokens: int) -> Tuple[int, int]:
        """Finds [start, end) token indices corresponding to the review text inside the prompt."""
        prefix = "Review: "
        text_start = len(prefix)
        text_end = len(prefix) + len(text)

        if getattr(self.model_wrapper.tokenizer, "is_fast", False):
            try:
                encoded = self.model_wrapper.tokenizer(
                    prompt, return_offsets_mapping=True, add_special_tokens=False
                )
                matching = [
                    i for i, (s, e) in enumerate(encoded.offset_mapping)
                    if max(s, text_start) < min(e, text_end)
                ]
                if matching and len(encoded.input_ids) == num_tokens:
                    return matching[0], matching[-1] + 1
            except Exception:
                pass

        prefix_ids = self.model_wrapper.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = self.model_wrapper.tokenizer.encode("\nSentiment:", add_special_tokens=False)
        if num_tokens > len(prefix_ids) + len(suffix_ids):
            return len(prefix_ids), num_tokens - len(suffix_ids)
        return 0, num_tokens

    def _get_reference_embedding(self, embedding_layer: torch.nn.Module, input_ids: torch.Tensor, input_embeds: torch.Tensor) -> torch.Tensor:
        """Builds full-sequence reference embeddings before optional prompt-template freezing."""
        if self.baseline_type == "zero":
            return torch.zeros_like(input_embeds)
        elif self.baseline_type == "pad":
            pad_id = self.model_wrapper.tokenizer.pad_token_id
            baseline_ids = torch.full_like(input_ids, fill_value=pad_id)
            return embedding_layer(baseline_ids)
        else:  # "mean": expectation over vocabulary embeddings (norm-preserving, neutral semantics)
            if self._cached_mean_embed is None or self._cached_mean_embed.device != input_embeds.device:
                with torch.no_grad():
                    w = embedding_layer.weight.detach()
                    mean_vec = w.to(torch.float32).mean(dim=0).to(input_embeds.dtype)
                    self._cached_mean_embed = mean_vec.view(1, 1, -1)
            return self._cached_mean_embed.expand_as(input_embeds).clone()

    def _compute_target_objective(self, logits: torch.Tensor, target_class: int) -> torch.Tensor:
        """Computes scalar/batched target score from final-position logits."""
        target_token_id = self.model_wrapper.target_token_ids[target_class]
        if self.target_mode == "logit_diff" and len(self.model_wrapper.target_token_ids) >= 2:
            other_class = 1 - int(target_class)
            other_token_id = self.model_wrapper.target_token_ids[other_class]
            return logits[:, -1, target_token_id].to(torch.float32) - logits[:, -1, other_token_id].to(torch.float32)
        return logits[:, -1, target_token_id].to(torch.float32)

    def build_baseline_embeddings(self, text: str, input_ids: torch.Tensor, input_embeds: torch.Tensor) -> torch.Tensor:
        """Constructs the baseline embedding tensor (exposed for axiomatic unit tests)."""
        embedding_layer = self.model_wrapper.model.get_input_embeddings()
        ref_embeds = self._get_reference_embedding(embedding_layer, input_ids, input_embeds)
        if not self.mask_prompt_template:
            return ref_embeds

        prompt = self.model_wrapper.format_prompt(text)
        rev_start, rev_end = self._find_review_span(prompt, text, input_ids.shape[1])
        baseline_embeds = input_embeds.clone()
        baseline_embeds[:, rev_start:rev_end, :] = ref_embeds[:, rev_start:rev_end, :]
        return baseline_embeds

    def explain(self, text: str, target_class: Optional[int] = None) -> AttributionResult:
        start_time = time.time()

        prompt = self.model_wrapper.format_prompt(text)
        if getattr(self.model_wrapper.tokenizer, "is_fast", False):
            input_ids = self.model_wrapper.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.model_wrapper.device)
        else:
            input_ids = self.model_wrapper.tokenizer.encode(
                prompt, return_tensors="pt"
            ).to(self.model_wrapper.device)

        # Determine target class and baseline probability
        _, probs = self.model_wrapper.predict_single(text)
        if target_class is None:
            target_class = int(np.argmax(probs))

        baseline_prob = float(probs[target_class])

        # Get input and baseline embeddings
        embedding_layer = self.model_wrapper.model.get_input_embeddings()
        input_embeds = embedding_layer(input_ids).detach()  # shape: (1, seq_len, hidden_dim)
        baseline_embeds = self.build_baseline_embeddings(text, input_ids, input_embeds).detach()

        delta = input_embeds - baseline_embeds
        # Accumulate gradients in float32 to avoid bfloat16 precision loss across steps and hidden_dim
        total_grads = torch.zeros_like(input_embeds, dtype=torch.float32)

        # Compute true endpoint objectives F(x') and F(x) without gradient
        with torch.no_grad():
            baseline_outputs = self.model_wrapper.model(inputs_embeds=baseline_embeds)
            baseline_score = float(self._compute_target_objective(baseline_outputs.logits, target_class).item())
            input_outputs = self.model_wrapper.model(inputs_embeds=input_embeds)
            input_score = float(self._compute_target_objective(input_outputs.logits, target_class).item())
            del baseline_outputs, input_outputs
        self.model_wrapper.forward_pass_count += 2

        # Dynamic backward micro-batching to prevent autograd activation OOM on 7B+ models
        param_count = sum(p.numel() for p in self.model_wrapper.model.parameters()) if hasattr(self.model_wrapper, "model") else 7e9
        is_large_model = param_count > 4e9

        if self.model_wrapper.device == "cuda" and torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if is_large_model:
                ig_batch_size = 16 if total_mem >= 30 else (8 if total_mem >= 16 else 4)
            else:
                ig_batch_size = 32 if total_mem >= 30 else (16 if total_mem >= 16 else 8)
        else:
            ig_batch_size = 4

        # Quadrature nodes and weights on [0, 1]
        if self.quadrature == "midpoint":
            # Midpoint rule: second-order accurate O(1/m^2) and avoids boundary singularities at alpha=0
            alphas = torch.linspace(
                0.5 / self.steps, 1.0 - 0.5 / self.steps, steps=self.steps,
                device=self.model_wrapper.device, dtype=torch.float32
            )
            weights = torch.full((self.steps,), 1.0 / self.steps, device=self.model_wrapper.device, dtype=torch.float32)
        elif self.quadrature == "trapezoid" and self.steps >= 2:
            alphas = torch.linspace(0.0, 1.0, steps=self.steps, device=self.model_wrapper.device, dtype=torch.float32)
            weights = torch.full((self.steps,), 1.0 / (self.steps - 1), device=self.model_wrapper.device, dtype=torch.float32)
            weights[0] *= 0.5
            weights[-1] *= 0.5
        else:
            alphas = torch.linspace(
                1.0 / self.steps, 1.0, steps=self.steps,
                device=self.model_wrapper.device, dtype=torch.float32
            )
            weights = torch.full((self.steps,), 1.0 / self.steps, device=self.model_wrapper.device, dtype=torch.float32)

        base_f32 = baseline_embeds.to(torch.float32)
        delta_f32 = delta.to(torch.float32)

        for i in range(0, self.steps, ig_batch_size):
            chunk_alphas = alphas[i:i + ig_batch_size]
            chunk_weights = weights[i:i + ig_batch_size]
            B = chunk_alphas.shape[0]

            # Interpolate in float32, then cast to model dtype for forward pass
            interp_f32 = base_f32 + chunk_alphas.view(B, 1, 1) * delta_f32
            interpolated = interp_f32.to(input_embeds.dtype).detach().clone().requires_grad_(True)

            outputs = self.model_wrapper.model(inputs_embeds=interpolated)
            target_scores = self._compute_target_objective(outputs.logits, target_class)

            grad = torch.autograd.grad(target_scores.sum(), interpolated)[0]
            weighted_grad = grad.to(torch.float32) * chunk_weights.view(B, 1, 1)
            total_grads += weighted_grad.sum(dim=0, keepdim=True).detach()

            del interpolated, interp_f32, outputs, target_scores, grad, weighted_grad
            self.model_wrapper.forward_pass_count += B
            self.model_wrapper.increment_backward_passes(B)

        # Token attribution: (input - baseline) * integrated_grads, summed over embedding dimension in float32
        attributions_tensor = (delta_f32 * total_grads).sum(dim=-1)[0]  # shape: (seq_len,)
        attributions = attributions_tensor.detach().cpu().numpy()

        del baseline_embeds, input_embeds, delta, base_f32, delta_f32, total_grads, attributions_tensor

        tokens = [
            self.model_wrapper.tokenizer.decode([tid])
            for tid in input_ids[0].tolist()
        ]
        execution_time = time.time() - start_time

        # Calculate Completeness Axiom error (Sundararajan et al., 2017)
        delta_f = input_score - baseline_score
        sum_attr = float(attributions.sum())
        denom = max(abs(delta_f), 1.0)
        completeness_error = float(abs(sum_attr - delta_f) / denom)

        return AttributionResult(
            tokens=tokens,
            attributions=attributions,
            target_class=target_class,
            execution_time=execution_time,
            explainer_name=f"Integrated Gradients (steps={self.steps})",
            baseline_prob=baseline_prob,
            metadata={
                "steps": self.steps,
                "baseline_type": self.baseline_type,
                "target_mode": self.target_mode,
                "mask_prompt_template": self.mask_prompt_template,
                "quadrature": self.quadrature,
                "completeness_error": completeness_error,
                "delta_f": delta_f,
                "sum_attributions": sum_attr,
            }
        )
