from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class AttributionResult:
    """Class to hold explanation attribution results."""
    tokens: List[str]
    attributions: np.ndarray  # 1D array of importance values for each token
    target_class: int         # 0: negative, 1: positive
    execution_time: float     # time taken in seconds
    explainer_name: str
    baseline_prob: Optional[float] = None

    def top_k_tokens(self, k: int = 3) -> List[tuple]:
        """Returns top K tokens with highest positive attributions."""
        sorted_indices = np.argsort(self.attributions)[::-1]
        top_indices = sorted_indices[:k]
        return [(self.tokens[i], float(self.attributions[i])) for i in top_indices]


class BaseExplainer(ABC):
    """Abstract Base Class for LLM Interpretability Explainer Methods."""
    
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        
    @property
    def explainer_name(self) -> str:
        """Returns standard readable name of the explainer."""
        name = self.__class__.__name__.replace("Explainer", "")
        if name == "GradientXInput":
            return "Gradient x Input"
        if name == "IntegratedGradients":
            return "Integrated Gradients"
        return name

    @abstractmethod
    def explain(self, text: str, target_class: Optional[int] = None) -> AttributionResult:
        """
        Generates token attribution explanation for the input text.
        
        Args:
            text: Input sentence/review text.
            target_class: 0 for negative, 1 for positive. If None, predicted class is used.
            
        Returns:
            AttributionResult object containing tokens, attributions, and metadata.
        """
        pass

