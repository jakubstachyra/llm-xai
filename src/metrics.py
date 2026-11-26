import re
import difflib
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


def _slice_review_tokens(
    res: AttributionResult,
    filter_prompt_template: bool = True,
) -> Tuple[List[str], np.ndarray]:
    """
    Isolates review tokens and attributions from the prompt scaffolding ('Review: ', '\\nSentiment:')
    using positional span detection rather than string-matching stopwords inside the review.
    """
    tokens = list(res.tokens)
    attrs = np.asarray(res.attributions, dtype=float)
    if not filter_prompt_template or len(tokens) == 0:
        return tokens, attrs

    if isinstance(res.metadata, dict) and "review_span" in res.metadata:
        span = res.metadata["review_span"]
        if isinstance(span, (tuple, list)) and len(span) == 2:
            s_idx, e_idx = int(span[0]), int(span[1])
            if 0 <= s_idx < e_idx <= len(tokens):
                return tokens[s_idx:e_idx], attrs[s_idx:e_idx]

    # Positional detection of 'Review:' prefix at the start of the prompt
    start_idx = 0
    joined_prefix = ""
    for i in range(min(24, len(tokens))):
        joined_prefix += tokens[i]
        if "review:" in joined_prefix.lower():
            start_idx = i + 1
            break

    # Positional detection of '\nSentiment:' suffix at the end of the prompt
    end_idx = len(tokens)
    joined_suffix = ""
    for i in range(len(tokens) - 1, max(start_idx - 1, len(tokens) - 8), -1):
        joined_suffix = tokens[i] + joined_suffix
        if "sentiment:" in joined_suffix.lower():
            end_idx = i
            break

    if start_idx < end_idx:
        return tokens[start_idx:end_idx], attrs[start_idx:end_idx]
    return tokens, attrs


def _group_subtokens_to_words(
    tokens: List[str],
    attrs: np.ndarray,
) -> List[Tuple[str, float]]:
    """
    Groups BPE subtokens into coherent word-level units with signed-sum attribution aggregation.
    Makes RQ1 alignment invariant to BPE subtoken splitting under uppercase or typo perturbations,
    matching the WordUnit granularity of RQ2.
    """
    words: List[Tuple[str, float]] = []
    curr_chars: List[str] = []
    curr_score: float = 0.0

    def flush():
        nonlocal curr_chars, curr_score
        if curr_chars:
            w = "".join(curr_chars).strip().lower()
            w_alnum = re.sub(r"[^\w]", "", w)
            key = w_alnum if w_alnum else w
            if key:
                words.append((key, float(curr_score)))
            curr_chars = []
            curr_score = 0.0

    for tok, score in zip(tokens, attrs):
        if not tok or tok.isspace():
            flush()
            continue
        starts_new = tok[0].isspace()
        clean = tok.strip()
        if not clean:
            flush()
            continue
        if starts_new:
            flush()
        # Separate standalone punctuation from alphanumeric words
        if curr_chars and (clean.isalnum() != "".join(curr_chars).isalnum()):
            flush()
        curr_chars.append(clean)
        curr_score += float(score)

    flush()
    return words


def align_word_attributions(
    res1: AttributionResult,
    res2: AttributionResult,
    filter_prompt_template: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Aligns word-level attributions between original and perturbed explanations using
    order-preserving longest common subsequence matching (SequenceMatcher), falling back
    to Levenshtein-similar token matching for typo_swap / contractions when needed.
    Preserves duplicate word occurrences in sequence order instead of collapsing them.
    """
    toks1, attrs1 = _slice_review_tokens(res1, filter_prompt_template=filter_prompt_template)
    toks2, attrs2 = _slice_review_tokens(res2, filter_prompt_template=filter_prompt_template)

    words1 = _group_subtokens_to_words(toks1, attrs1)
    words2 = _group_subtokens_to_words(toks2, attrs2)
    # Keep alphanumeric words primarily; if too short, keep all units
    alnum1 = [(w, s) for w, s in words1 if any(c.isalnum() for c in w)]
    alnum2 = [(w, s) for w, s in words2 if any(c.isalnum() for c in w)]
    if len(alnum1) >= 2 and len(alnum2) >= 2:
        words1, words2 = alnum1, alnum2

    keys1 = [w for w, _ in words1]
    keys2 = [w for w, _ in words2]

    matcher = difflib.SequenceMatcher(None, keys1, keys2, autojunk=False)
    vec1: List[float] = []
    vec2: List[float] = []
    matched_labels: List[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                vec1.append(words1[i1 + offset][1])
                vec2.append(words2[j1 + offset][1])
                matched_labels.append(f"{keys1[i1 + offset]}@{i1 + offset}")
        elif tag == "replace" and (i2 - i1) == (j2 - j1):
            # Pairwise replacement (e.g. typo_swap 'masterfully' -> 'masetrfully'): match if edit similarity >= 0.7
            for offset in range(i2 - i1):
                w_a, s_a = words1[i1 + offset]
                w_b, s_b = words2[j1 + offset]
                if len(w_a) >= 4 and len(w_b) >= 4:
                    sim = difflib.SequenceMatcher(None, w_a, w_b).ratio()
                    if sim >= 0.7:
                        vec1.append(s_a)
                        vec2.append(s_b)
                        matched_labels.append(f"{w_a}@{i1 + offset}")

    return np.asarray(vec1, dtype=float), np.asarray(vec2, dtype=float), matched_labels


def _extract_token_map(
    res: AttributionResult,
    filter_prompt_template: bool = True,
    ignored_tokens: Optional[Set[str]] = None
) -> Dict[str, float]:
    """
    Extracts a dictionary mapping normalized word text to aggregated attribution.
    Uses positional prompt-template slicing and word-level subtoken grouping.
    """
    toks, attrs = _slice_review_tokens(res, filter_prompt_template=filter_prompt_template)
    words = _group_subtokens_to_words(toks, attrs)
    if ignored_tokens:
        words = [(w, s) for w, s in words if w not in ignored_tokens]

    t_map: Dict[str, List[float]] = {}
    for w, a in words:
        if w not in t_map:
            t_map[w] = []
        t_map[w].append(float(a))

    return {k: float(np.mean(v)) for k, v in t_map.items()}


def stability_spearman(
    res1: AttributionResult, 
    res2: AttributionResult,
    filter_prompt_template: bool = True
) -> float:
    """
    Computes Spearman rank correlation between word-level attributions of aligned words.
    Measures monotonic rank preservation between original and perturbed explanations.
    Returns np.nan when fewer than 2 shared units exist or variance is zero.
    """
    vec1, vec2, _ = align_word_attributions(res1, res2, filter_prompt_template=filter_prompt_template)
    if len(vec1) < 2 or np.std(vec1) == 0 or np.std(vec2) == 0:
        return float("nan")

    corr, _ = spearmanr(vec1, vec2)
    return float(corr) if not np.isnan(corr) else float("nan")


def stability_kendall_tau(
    res1: AttributionResult, 
    res2: AttributionResult,
    filter_prompt_template: bool = True
) -> float:
    """
    Computes Kendall's Tau-b rank correlation for aligned word units.
    Measures pairwise ordering concordance between explanations.
    """
    vec1, vec2, _ = align_word_attributions(res1, res2, filter_prompt_template=filter_prompt_template)
    if len(vec1) < 2 or np.std(vec1) == 0 or np.std(vec2) == 0:
        return float("nan")

    tau, _ = kendalltau(vec1, vec2)
    return float(tau) if not np.isnan(tau) else float("nan")


def stability_pearson(
    res1: AttributionResult, 
    res2: AttributionResult,
    filter_prompt_template: bool = True
) -> float:
    """
    Computes Pearson linear correlation of raw attribution scores for aligned word units.
    """
    vec1, vec2, _ = align_word_attributions(res1, res2, filter_prompt_template=filter_prompt_template)
    if len(vec1) < 2 or np.std(vec1) == 0 or np.std(vec2) == 0:
        return float("nan")

    corr, _ = pearsonr(vec1, vec2)
    return float(corr) if not np.isnan(corr) else float("nan")


def stability_topk_jaccard(
    res1: AttributionResult, 
    res2: AttributionResult, 
    k: int = 3,
    filter_prompt_template: bool = True
) -> float:
    """
    Computes Jaccard Similarity between Top K aligned word units of two explanations:
    Jaccard(TopK_1, TopK_2) = |TopK_1 ∩ TopK_2| / |TopK_1 ∪ TopK_2|
    """
    vec1, vec2, labels = align_word_attributions(res1, res2, filter_prompt_template=filter_prompt_template)
    if len(labels) == 0:
        return float("nan")

    eff_k = min(k, len(labels))
    idx1 = np.argsort(-vec1, kind="mergesort")[:eff_k]
    idx2 = np.argsort(-vec2, kind="mergesort")[:eff_k]

    top1 = {labels[i] for i in idx1}
    top2 = {labels[i] for i in idx2}

    union = top1.union(top2)
    if not union:
        return float("nan")
    intersection = top1.intersection(top2)
    return float(len(intersection) / len(union))


def compute_all_stability_metrics(
    res1: AttributionResult,
    res2: AttributionResult,
    filter_prompt_template: bool = True,
) -> Dict[str, Any]:
    """
    Computes all word-aligned stability metrics in a single alignment pass
    (avoids redundant SequenceMatcher alignment calls per perturbation pair).
    """
    vec1, vec2, labels = align_word_attributions(
        res1, res2, filter_prompt_template=filter_prompt_template
    )
    aligned_words = int(len(vec1))

    if aligned_words < 2 or np.std(vec1) == 0 or np.std(vec2) == 0:
        sp = kd = pr = float("nan")
    else:
        sp_val, _ = spearmanr(vec1, vec2)
        kd_val, _ = kendalltau(vec1, vec2)
        pr_val, _ = pearsonr(vec1, vec2)
        sp = float(sp_val) if not np.isnan(sp_val) else float("nan")
        kd = float(kd_val) if not np.isnan(kd_val) else float("nan")
        pr = float(pr_val) if not np.isnan(pr_val) else float("nan")

    j_scores: Dict[str, float] = {}
    for k in (1, 3, 5):
        if len(labels) == 0:
            j_scores[f"jaccard_k{k}"] = float("nan")
            continue
        eff_k = min(k, len(labels))
        idx1 = np.argsort(-vec1, kind="mergesort")[:eff_k]
        idx2 = np.argsort(-vec2, kind="mergesort")[:eff_k]
        top1 = {labels[i] for i in idx1}
        top2 = {labels[i] for i in idx2}
        union = top1.union(top2)
        j_scores[f"jaccard_k{k}"] = float(len(top1.intersection(top2)) / len(union)) if union else float("nan")

    return {
        "aligned_words": aligned_words,
        "spearman": sp,
        "kendall": kd,
        "pearson": pr,
        **j_scores,
    }


def compute_perturbation_strength(
    original_text: str,
    perturbed_text: str,
    orig_num_tokens: int,
    pert_num_tokens: int,
) -> Dict[str, float]:
    """
    Computes surface/token perturbation strength indicators for RQ1 strength analysis:
    - char_edit_ratio: normalized character edit distance (1 - SequenceMatcher ratio)
    - token_len_diff: absolute token count change
    - token_len_rel_diff: relative token count change
    """
    char_edit_ratio = float(
        1.0 - difflib.SequenceMatcher(None, original_text.strip(), perturbed_text.strip()).ratio()
    )
    token_len_diff = int(abs(int(pert_num_tokens) - int(orig_num_tokens)))
    token_len_rel_diff = float(token_len_diff / max(int(orig_num_tokens), 1))
    return {
        "char_edit_ratio": char_edit_ratio,
        "token_len_diff": token_len_diff,
        "token_len_rel_diff": token_len_rel_diff,
    }


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
