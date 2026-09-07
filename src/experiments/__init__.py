"""
LLM XAI Evaluation Framework - Experiments Module
"""
from .rq1_stability import RQ1StabilityExperiment
from .visualizations import (
    plot_rq1_metric_distribution,
    plot_rq1_jaccard_k_curves,
    plot_rq1_token_attribution_comparison
)

__all__ = [
    "RQ1StabilityExperiment",
    "plot_rq1_metric_distribution",
    "plot_rq1_jaccard_k_curves",
    "plot_rq1_token_attribution_comparison"
]
