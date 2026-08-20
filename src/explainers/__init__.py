from .base import BaseExplainer, AttributionResult
from .occlusion import OcclusionExplainer

try:
    from .shap_explainer import SHAPExplainer
except ImportError:
    SHAPExplainer = None

try:
    from .integrated_gradients import IntegratedGradientsExplainer
except ImportError:
    IntegratedGradientsExplainer = None

__all__ = [
    "BaseExplainer",
    "AttributionResult",
    "OcclusionExplainer",
    "SHAPExplainer",
    "IntegratedGradientsExplainer"
]
