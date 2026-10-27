import time
import torch
import numpy as np
from typing import Optional
from .base import BaseExplainer, AttributionResult


class GradientXInputExplainer(BaseExplainer):
    """
    Gradient x Input (Input * Gradient) for Causal Language Models.
    Computes local feature attribution via the dot product between input token embeddings
    and the gradient of the target sentiment logit:
        attr_i = <x_i, grad_{x_i} F(x)>
    Requires only 1 backward pass, operates directly on the true input distribution,
    and avoids the numerical instability of path integration in deep LayerNorm architectures.
    """
    def __init__(self, model_wrapper):
        super().__init__(model_wrapper)

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
        
        # Get input embeddings and enable gradient tracking
        embedding_layer = self.model_wrapper.model.get_input_embeddings()
        input_embeds = embedding_layer(input_ids).detach().clone().requires_grad_(True)
        
        # Forward pass
        outputs = self.model_wrapper.model(inputs_embeds=input_embeds)
        target_logit = outputs.logits[0, -1, target_token_id]
        
        # Backward pass
        grad = torch.autograd.grad(target_logit, input_embeds)[0] # shape: (1, seq_len, hidden_dim)
        self.model_wrapper.forward_pass_count += 1
        self.model_wrapper.increment_backward_passes(1)
        
        # Dot product across embedding dimension: <x_i, g_i>
        token_attr = (input_embeds * grad).sum(dim=-1).squeeze(0) # shape: (seq_len,)
        attributions = token_attr.detach().float().cpu().numpy()
        
        tokens = [self.model_wrapper.tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
        execution_time = time.time() - start_time
        
        return AttributionResult(
            tokens=tokens,
            attributions=attributions,
            target_class=target_class,
            execution_time=execution_time,
            explainer_name="Gradient x Input",
            baseline_prob=baseline_prob
        )
