from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class FaithfulnessCurveResult:
    """Stores the perturbation step curves and aggregate scalar metrics for an explanation."""
    k_values: List[int]
    comprehensiveness_prob: List[float] # Delta P (prob drop)
    comprehensiveness_log: List[float]  # Delta log P (log-prob drop)
    sufficiency_prob: List[float]
    sufficiency_log: List[float]
    aopc_comprehensiveness_prob: float
    aopc_comprehensiveness_log: float
    aopc_sufficiency_prob: float
    aopc_sufficiency_log: float
    normalized_gap_g: float = 0.0        # G metric relative to random baseline
    harmonization_h: float = 0.0         # AOPC_C - AOPC_U


def calculate_aopc(curve: List[float]) -> float:
    """Calculates Area Over the Perturbation Curve (AOPC) as the arithmetic mean of step drops."""
    if not curve:
        return 0.0
    return float(np.mean(curve))


def compute_normalized_gap(
    aopc_method: float, 
    aopc_random: float, 
    eps: float = 1e-6
) -> float:
    """
    Computes the Normalized Information Gap G metric:
        G = (AOPC_method - AOPC_random) / (|AOPC_random| + eps)
    Measures the percentage gain over random feature removal.
    """
    return float((aopc_method - aopc_random) / (abs(aopc_random) + eps))


def evaluate_faithfulness_curves(
    k_values: List[int],
    c_prob: List[float],
    c_log: List[float],
    u_prob: List[float],
    u_log: List[float],
    random_aopc_c_prob: Optional[float] = None
) -> FaithfulnessCurveResult:
    """
    Assembles step drop curves into a FaithfulnessCurveResult and computes AOPC, G, and H metrics.
    """
    aopc_c_prob = calculate_aopc(c_prob)
    aopc_c_log = calculate_aopc(c_log)
    aopc_u_prob = calculate_aopc(u_prob)
    aopc_u_log = calculate_aopc(u_log)
    
    g_metric = 0.0
    if random_aopc_c_prob is not None:
        g_metric = compute_normalized_gap(aopc_c_prob, random_aopc_c_prob)
        
    h_metric = aopc_c_prob - aopc_u_prob

    return FaithfulnessCurveResult(
        k_values=k_values,
        comprehensiveness_prob=c_prob,
        comprehensiveness_log=c_log,
        sufficiency_prob=u_prob,
        sufficiency_log=u_log,
        aopc_comprehensiveness_prob=aopc_c_prob,
        aopc_comprehensiveness_log=aopc_c_log,
        aopc_sufficiency_prob=aopc_u_prob,
        aopc_sufficiency_log=aopc_u_log,
        normalized_gap_g=g_metric,
        harmonization_h=h_metric
    )
