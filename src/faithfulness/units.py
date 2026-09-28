from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal
import numpy as np


@dataclass
class WordUnit:
    """
    Represents a coherent lexical unit (word or punctuation symbol) mapped to its constituent subtokens.
    Enables tokenizer-agnostic faithfulness evaluation and prevents out-of-distribution (OOD) subtoken fragments.
    """
    index: int
    text: str
    start_char: int
    end_char: int
    token_indices: List[int] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    subtoken_scores: List[float] = field(default_factory=list)
    score: float = 0.0

    def __repr__(self) -> str:
        return (
            f"WordUnit(idx={self.index}, text='{self.text}', span=({self.start_char}, {self.end_char}), "
            f"tokens={self.tokens}, score={self.score:+.4f})"
        )


def extract_word_units(
    text: str,
    tokenizer,
    prompt_prefix: str = "Review: ",
    prompt_suffix: str = "\nSentiment:"
) -> List[WordUnit]:
    """
    Extracts word-level lexical units from the review text within the formatted prompt.
    Groups subtoken continuations (BPE/WordPiece fragments) into natural words.
    
    Args:
        text: The raw review text (e.g. "A masterfully-crafted film.")
        tokenizer: HuggingFace tokenizer with fast offset_mapping support.
        prompt_prefix: Prompt template prefix before the text.
        prompt_suffix: Prompt template suffix after the text.
        
    Returns:
        List of WordUnit instances with character offsets and token index mappings.
    """
    full_prompt = f"{prompt_prefix}{text}{prompt_suffix}"
    encoded = tokenizer(full_prompt, return_offsets_mapping=True, add_special_tokens=False)
    
    tokens = [tokenizer.decode([tid]) for tid in encoded.input_ids]
    offsets = encoded.offset_mapping
    
    text_start_char = len(prompt_prefix)
    text_end_char = len(prompt_prefix) + len(text)
    
    # 1. Identify all prompt tokens that overlap with the review text span
    review_token_indices = []
    for idx, (t_start, t_end) in enumerate(offsets):
        # A token belongs to the review text if its character span overlaps with [text_start_char, text_end_char]
        overlap_start = max(t_start, text_start_char)
        overlap_end = min(t_end, text_end_char)
        if overlap_start < overlap_end:
            review_token_indices.append(idx)
            
    if not review_token_indices:
        return []
        
    # 2. Group subtokens into coherent WordUnits
    units: List[WordUnit] = []
    current_unit: Optional[dict] = None
    
    for tidx in review_token_indices:
        tok_str = tokens[tidx]
        t_start, t_end = offsets[tidx]
        
        # Clamp offsets to review text boundaries
        clamped_start = max(t_start, text_start_char) - text_start_char
        clamped_end = min(t_end, text_end_char) - text_start_char
        
        # Determine whether this token continues the previous word unit
        is_continuation = False
        if current_unit is not None:
            # Token continues previous unit if:
            # - It begins immediately where the previous unit ended (no space in original text)
            # - The token string does not start with whitespace
            # - Both preceding char and current char are alphanumeric
            if clamped_start == current_unit.end_char and not tok_str.startswith(" "):
                prev_char = text[current_unit.end_char - 1] if current_unit.end_char > 0 else ""
                curr_char = text[clamped_start] if clamped_start < len(text) else ""
                if prev_char.isalnum() and curr_char.isalnum():
                    is_continuation = True
                    
        if is_continuation and current_unit is not None:
            current_unit.end_char = clamped_end
            current_unit.token_indices.append(tidx)
            current_unit.tokens.append(tok_str)
            current_unit.text = text[current_unit.start_char:clamped_end]
        else:
            # If current token has leading space in text slice, strip leading space from the unit span
            actual_start = clamped_start
            while actual_start < clamped_end and text[actual_start].isspace():
                actual_start += 1
                
            if actual_start < clamped_end:
                current_unit = WordUnit(
                    index=len(units),
                    text=text[actual_start:clamped_end],
                    start_char=actual_start,
                    end_char=clamped_end,
                    token_indices=[tidx],
                    tokens=[tok_str],
                )
                units.append(current_unit)
            else:
                # Token was only whitespace
                current_unit = None

    return units


def aggregate_word_attributions(
    units: List[WordUnit],
    token_attributions: np.ndarray,
    method: Literal["signed_sum", "signed_max", "mean"] = "signed_sum"
) -> List[WordUnit]:
    """
    Aggregates subtoken attribution scores into word-level scores.
    
    Args:
        units: List of WordUnit instances with mapped token indices.
        token_attributions: 1D numpy array of attribution scores across all prompt tokens.
        method: Aggregation strategy:
            - 'signed_sum': Algebraic sum of subtoken scores (preserves net directional impact).
            - 'signed_max': Sign of max-magnitude subtoken multiplied by its absolute value.
            - 'mean': Arithmetic mean of subtoken scores.
            
    Returns:
        The same list of WordUnit instances with updated subtoken_scores and score fields.
    """
    for unit in units:
        scores = [float(token_attributions[idx]) for idx in unit.token_indices if idx < len(token_attributions)]
        unit.subtoken_scores = scores
        
        if not scores:
            unit.score = 0.0
            continue
            
        if method == "signed_sum":
            unit.score = float(np.sum(scores))
        elif method == "signed_max":
            max_idx = int(np.argmax(np.abs(scores)))
            unit.score = float(scores[max_idx])
        elif method == "mean":
            unit.score = float(np.mean(scores))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
            
    return units


def rank_word_units(
    units: List[WordUnit],
    ranking_mode: Literal["top_signed", "bottom_signed", "magnitude", "near_zero", "random"] = "top_signed",
    seed: Optional[int] = None
) -> List[WordUnit]:
    """
    Ranks word units according to the specified experimental evaluation order.
    Resolves ties deterministically using sequence index.
    
    Args:
        units: List of WordUnit instances with assigned scores.
        ranking_mode:
            - 'top_signed': Highest positive attribution first (Comprehensiveness: tokens favoring c*).
            - 'bottom_signed': Most negative attribution first (Sanity check: tokens opposing c*).
            - 'magnitude': Highest absolute attribution first.
            - 'near_zero': Smallest absolute attribution first (Neutral / uninformative tokens).
            - 'random': Uniform random ordering (Baseline null hypothesis).
        seed: Random seed for 'random' mode.
        
    Returns:
        Ordered copy of WordUnit instances.
    """
    ordered = list(units)
    if ranking_mode == "top_signed":
        # Descending score; tie-break: ascending sequence index
        ordered.sort(key=lambda u: (-u.score, u.index))
    elif ranking_mode == "bottom_signed":
        # Ascending score; tie-break: ascending sequence index
        ordered.sort(key=lambda u: (u.score, u.index))
    elif ranking_mode == "magnitude":
        # Descending absolute score; tie-break: ascending sequence index
        ordered.sort(key=lambda u: (-abs(u.score), u.index))
    elif ranking_mode == "near_zero":
        # Ascending absolute score; tie-break: ascending sequence index
        ordered.sort(key=lambda u: (abs(u.score), u.index))
    elif ranking_mode == "random":
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(ordered))
        ordered = [ordered[i] for i in indices]
    else:
        raise ValueError(f"Unknown ranking mode: {ranking_mode}")
        
    return ordered
