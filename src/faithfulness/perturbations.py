import re
from typing import List, Literal, Optional
import torch
from .units import WordUnit


def apply_word_deletion(text: str, units_to_remove: List[WordUnit]) -> str:
    """
    Applies the Discrete Deletion Operator by excising the specified WordUnits from the original text.
    Processes spans in descending order of character offsets to maintain slice index validity.
    Normalizes residual whitespace so the resulting text remains natural.
    
    Args:
        text: Original review text.
        units_to_remove: List of WordUnit instances to be deleted.
        
    Returns:
        Perturbed text with the specified words excised.
    """
    if not units_to_remove:
        return text

    # Sort descending by start_char so earlier offsets are never invalidated
    sorted_units = sorted(units_to_remove, key=lambda u: u.start_char, reverse=True)
    
    chars = list(text)
    for unit in sorted_units:
        s = max(0, unit.start_char)
        e = min(len(chars), unit.end_char)
        # Clear the characters of this unit
        for i in range(s, e):
            chars[i] = ""
            
    perturbed = "".join(chars)
    # Clean up double spaces created by deletion while preserving punctuation
    perturbed = re.sub(r"[ \t]+", " ", perturbed).strip()
    return perturbed


def apply_word_retention(text: str, all_units: List[WordUnit], units_to_keep: List[WordUnit]) -> str:
    """
    Applies the Discrete Retention Operator for Sufficiency evaluations (U(k)).
    Preserves ONLY the specified WordUnits and excises all others.
    
    Args:
        text: Original review text.
        all_units: Complete list of WordUnits in text.
        units_to_keep: Sublist of WordUnits to be retained.
        
    Returns:
        Perturbed text containing only the retained words.
    """
    keep_indices = {u.index for u in units_to_keep}
    units_to_remove = [u for u in all_units if u.index not in keep_indices]
    return apply_word_deletion(text, units_to_remove)


def apply_embedding_mask(
    inputs_embeds: torch.Tensor,
    units_to_mask: List[WordUnit],
    mask_value: Literal["zero", "mean"] = "zero"
) -> torch.Tensor:
    """
    Applies the Vector Embedding Masking Operator.
    Directly replaces the token embedding vectors corresponding to the specified WordUnits
    with a neutral vector (zero or mean embedding), eliminating syntax and RoPE length shifts.
    
    Args:
        inputs_embeds: Tensor of shape (batch_size, seq_len, hidden_dim).
        units_to_mask: List of WordUnits whose constituent token vectors are to be masked.
        mask_value: 'zero' (zeros out vectors) or 'mean' (replaces with sequence mean).
        
    Returns:
        Masked embedding tensor of identical shape.
    """
    masked_embeds = inputs_embeds.clone()
    
    if mask_value == "mean":
        replacement = inputs_embeds.mean(dim=1, keepdim=True) # shape: (batch_size, 1, hidden_dim)
    else:
        replacement = torch.zeros(
            (inputs_embeds.shape[0], 1, inputs_embeds.shape[-1]),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device
        )

    for unit in units_to_mask:
        for tidx in unit.token_indices:
            if tidx < masked_embeds.shape[1]:
                masked_embeds[:, tidx, :] = replacement.squeeze(1)
                
    return masked_embeds
