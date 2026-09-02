import re
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
from typing import List, Tuple, Dict, Any, Set, Optional

try:
    from explainers.base import AttributionResult
    from models import SentimentCausalLM
except ImportError:
    from .explainers.base import AttributionResult
    from .models import SentimentCausalLM

DEFAULT_PROMPT_STOPWORDS: Set[str] = {
    "review", "sentiment", "sent", "iment", ":", "\n", "review:", "sentiment:", "rating", "rating:"
}

def _extract_token_map(
    res: AttributionResult,
    filter_prompt_template: bool = True,
    ignored_tokens: Optional[Set[str]] = None
) -> Dict[str, float]:
    """
    Extracts a dictionary mapping normalized token text to aggregated attribution.
    Averages attributions if the same token appears multiple times.
    Optionally filters out prompt scaffolding tokens (e.g. 'Review', 'Sentiment').
    """
    if ignored_tokens is None:
        ignored_tokens = DEFAULT_PROMPT_STOPWORDS if filter_prompt_template else set()

    t_map: Dict[str, List[float]] = {}
    for t, a in zip(res.tokens, res.attributions):
        t_clean = t.strip().lower()
        if not t_clean or (filter_prompt_template and t_clean in ignored_tokens):
            continue
        if t_clean not in t_map:
            t_map[t_clean] = []
        t_map[t_clean].append(float(a))

    return {k: float(np.mean(v)) for k, v in t_map.items()}


def stability_spearman(
    res1: AttributionResult, 
    res2: AttributionResult,
    filter_prompt_template: bool = True
) -> float:
    """
    Computes Spearman rank correlation between token attributions of shared tokens.
    Measures monotonic rank preservation between original and perturbed explanations.
    """
    map1 = _extract_token_map(res1, filter_prompt_template=filter_prompt_template)
    map2 = _extract_token_map(res2, filter_prompt_template=filter_prompt_template)
    
    shared_tokens = [t for t in map1 if t in map2]
    if len(shared_tokens) < 2:
        return 0.0
        
    vec1 = [map1[t] for t in shared_tokens]
    vec2 = [map2[t] for t in shared_tokens]
    
    corr, _ = spearmanr(vec1, vec2)
    return float(corr) if not np.isnan(corr) else 0.0


def stability_kendall_tau(
    res1: AttributionResult, 
    res2: AttributionResult,
    filter_prompt_template: bool = True
) -> float:
    """
    Computes Kendall's Tau-b rank correlation for shared tokens.
    Measures pairwise ordering concordance between explanations.
    """
    map1 = _extract_token_map(res1, filter_prompt_template=filter_prompt_template)
    map2 = _extract_token_map(res2, filter_prompt_template=filter_prompt_template)
    
    shared_tokens = [t for t in map1 if t in map2]
    if len(shared_tokens) < 2:
        return 0.0
        
    vec1 = [map1[t] for t in shared_tokens]
    vec2 = [map2[t] for t in shared_tokens]
    
    tau, _ = kendalltau(vec1, vec2)
    return float(tau) if not np.isnan(tau) else 0.0


def stability_pearson(
    res1: AttributionResult, 
    res2: AttributionResult,
    filter_prompt_template: bool = True
) -> float:
    """
    Computes Pearson linear correlation of raw attribution scores for shared tokens.
    """
    map1 = _extract_token_map(res1, filter_prompt_template=filter_prompt_template)
    map2 = _extract_token_map(res2, filter_prompt_template=filter_prompt_template)
    
    shared_tokens = [t for t in map1 if t in map2]
    if len(shared_tokens) < 2:
        return 0.0
        
    vec1 = [map1[t] for t in shared_tokens]
    vec2 = [map2[t] for t in shared_tokens]
    
    # In case of constant vector variance
    if np.std(vec1) == 0 or np.std(vec2) == 0:
        return 0.0
        
    corr, _ = pearsonr(vec1, vec2)
    return float(corr) if not np.isnan(corr) else 0.0


def stability_topk_jaccard(
    res1: AttributionResult, 
    res2: AttributionResult, 
    k: int = 3,
    filter_prompt_template: bool = True
) -> float:
    """
    Computes Jaccard Similarity between Top K tokens of two explanations:
    Jaccard(TopK_1, TopK_2) = |TopK_1 ∩ TopK_2| / |TopK_1 ∪ TopK_2|
    """
    map1 = _extract_token_map(res1, filter_prompt_template=filter_prompt_template)
    map2 = _extract_token_map(res2, filter_prompt_template=filter_prompt_template)

    sorted1 = sorted(map1.items(), key=lambda x: x[1], reverse=True)
    sorted2 = sorted(map2.items(), key=lambda x: x[1], reverse=True)

    top1 = set([t[0] for t in sorted1[:k]])
    top2 = set([t[0] for t in sorted2[:k]])
    
    union = top1.union(top2)
    if not union:
        return 0.0
    intersection = top1.intersection(top2)
    return float(len(intersection) / len(union))


def faithfulness_prob_drop(
    model_wrapper: SentimentCausalLM,
    text: str,
    attribution_result: AttributionResult,
    k: int = 2
) -> float:
    """
    Measures Faithfulness via Probability Drop.
    Removes Top K most important tokens from text and computes probability drop.
    
    Returns:
        Probability Drop = P(target | original) - P(target | pruned)
    """
    target_class = attribution_result.target_class
    baseline_prob = attribution_result.baseline_prob
    if baseline_prob is None:
        _, probs = model_wrapper.predict_single(text)
        baseline_prob = float(probs[target_class])
        
    # Get top K tokens (excluding prompt template words)
    map_toks = _extract_token_map(attribution_result, filter_prompt_template=True)
    sorted_toks = sorted(map_toks.items(), key=lambda x: x[1], reverse=True)
    top_tokens = [t[0] for t in sorted_toks[:k]]
    
    pruned_text = text
    for clean_t in top_tokens:
        if clean_t:
            # Word-boundary regex matching with case insensitivity
            # Prevents partial substring destruction (e.g. erasing 'in' inside 'original')
            pattern = re.compile(rf'\b{re.escape(clean_t)}\b', flags=re.IGNORECASE)
            if pattern.search(pruned_text):
                pruned_text = pattern.sub("", pruned_text, count=1)
            else:
                pruned_text = re.sub(re.escape(clean_t), "", pruned_text, count=1, flags=re.IGNORECASE)
            
    # Remove extra spaces
    pruned_text = " ".join(pruned_text.split())
    if not pruned_text:
        pruned_text = "a" # fallback if everything was removed
        
    _, pruned_probs = model_wrapper.predict_single(pruned_text)
    pruned_prob = float(pruned_probs[target_class])
    
    prob_drop = baseline_prob - pruned_prob
    return float(prob_drop)
