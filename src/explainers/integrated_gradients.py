import time
import torch
import numpy as np
from typing import Optional
from .base import BaseExplainer, AttributionResult

class IntegratedGradientsExplainer(BaseExplainer):
    """
    Integrated Gradients (Sundararajan et al., 2017) for Causal Language Models.
    Computes path-integral of gradients along the straight line from a baseline embedding
    to the input token embeddings for the target sentiment token logit.
    """
    def __init__(
        self, 
        model_wrapper, 
        steps: int = 25, 
        baseline_type: str = "zero"
    ):
        super().__init__(model_wrapper)
        self.steps = steps
        self.baseline_type = baseline_type

    def explain(self, text: str, target_class: Optional[int] = None) -> AttributionResult:
        start_time = time.time()
        
        prompt = self.model_wrapper.format_prompt(text)
        input_ids = self.model_wrapper.tokenizer.encode(
            prompt, 
            return_tensors="pt"
        ).to(self.model_wrapper.device)
        
        # Determine target class and baseline probability
        _, probs = self.model_wrapper.predict_single(text)
        if target_class is None:
            target_class = int(np.argmax(probs))
            
        baseline_prob = float(probs[target_class])
        target_token_id = self.model_wrapper.target_token_ids[target_class]
        
        # Get input embeddings
        embedding_layer = self.model_wrapper.model.get_input_embeddings()
        input_embeds = embedding_layer(input_ids) # shape: (1, seq_len, hidden_dim)
        
        # Define baseline embeddings
        if self.baseline_type == "zero":
            baseline_embeds = torch.zeros_like(input_embeds)
        else:
            pad_id = self.model_wrapper.tokenizer.pad_token_id
            baseline_ids = torch.full_like(input_ids, fill_value=pad_id)
            baseline_embeds = embedding_layer(baseline_ids)

        delta = input_embeds - baseline_embeds
        total_grads = torch.zeros_like(input_embeds)

        # Compute true baseline logit F(x') without gradient
        with torch.no_grad():
            baseline_outputs = self.model_wrapper.model(inputs_embeds=baseline_embeds)
            baseline_logit = float(baseline_outputs.logits[0, -1, target_token_id].item())
            input_outputs = self.model_wrapper.model(inputs_embeds=input_embeds)
            input_logit = float(input_outputs.logits[0, -1, target_token_id].item())
            del baseline_outputs, input_outputs
        self.model_wrapper.forward_pass_count += 2

        # Numerical integration using Riemann sum over steps k/m (Sundararajan et al., 2017)
        # Vectorized/batched integration to saturate GPU Tensor Cores (A100/L4)
        if self.model_wrapper.device == "cuda" and torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            ig_batch_size = 32 if total_mem >= 16 else 8
        else:
            ig_batch_size = 4

        alphas = torch.linspace(1.0 / self.steps, 1.0, steps=self.steps, device=self.model_wrapper.device)
        
        for i in range(0, self.steps, ig_batch_size):
            chunk_alphas = alphas[i:i + ig_batch_size]
            B = chunk_alphas.shape[0]
            
            # Shape: (B, seq_len, hidden_dim)
            interpolated = (baseline_embeds.expand(B, -1, -1) + chunk_alphas.view(B, 1, 1) * delta.expand(B, -1, -1)).detach().clone().requires_grad_(True)
            
            outputs = self.model_wrapper.model(inputs_embeds=interpolated)
            target_logits = outputs.logits[:, -1, target_token_id]
            
            grad = torch.autograd.grad(target_logits.sum(), interpolated)[0]
            total_grads += grad.sum(dim=0, keepdim=True).detach()
            
            del interpolated, outputs, target_logits, grad
            self.model_wrapper.forward_pass_count += B
            self.model_wrapper.increment_backward_passes(B)

        avg_grads = total_grads / self.steps
        # Token attribution: (input - baseline) * avg_grads, summed over embedding dimension
        attributions_tensor = (delta * avg_grads).sum(dim=-1)[0] # shape: (seq_len,)
        attributions = attributions_tensor.detach().cpu().to(torch.float32).numpy()
        
        del baseline_embeds, input_embeds, delta, total_grads, avg_grads, attributions_tensor
        
        tokens = [
            self.model_wrapper.tokenizer.decode([tid]) 
            for tid in input_ids[0].tolist()
        ]
        execution_time = time.time() - start_time
        
        # Calculate Completeness Axiom error (Sundararajan et al., 2017)
        delta_f = input_logit - baseline_logit
        sum_attr = float(attributions.sum())
        # Normalized relative error (with lower bound on denominator to prevent div-by-zero on neutral inputs)
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
                "completeness_error": completeness_error,
                "delta_f": delta_f,
                "sum_attributions": float(attributions.sum())
            }
        )
