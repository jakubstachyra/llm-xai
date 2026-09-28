"""
Faithfulness evaluation package for LLM explanation attribution.
Includes word unit extraction, signed subtoken aggregation, deletion & masking operators, and faithfulness metrics.
"""

from src.faithfulness.units import (
    WordUnit,
    extract_word_units,
    aggregate_word_attributions,
    rank_word_units,
)
from src.faithfulness.perturbations import (
    apply_word_deletion,
    apply_word_retention,
    apply_embedding_mask,
)
from src.faithfulness.metrics import (
    FaithfulnessCurveResult,
    calculate_aopc,
    compute_normalized_gap,
    evaluate_faithfulness_curves,
)

__all__ = [
    "WordUnit",
    "extract_word_units",
    "aggregate_word_attributions",
    "rank_word_units",
    "apply_word_deletion",
    "apply_word_retention",
    "apply_embedding_mask",
    "FaithfulnessCurveResult",
    "calculate_aopc",
    "compute_normalized_gap",
    "evaluate_faithfulness_curves",
]
