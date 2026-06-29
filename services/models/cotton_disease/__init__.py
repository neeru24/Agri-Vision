"""
Cotton Disease Detection Ensemble Module

Implements multi-model ensemble for cotton disease detection with 95% accuracy.
Features: Ensemble voting, uncertainty quantification, Grad-CAM explainability.
"""

from .ensemble_model import CottonDiseaseEnsemble
from .uncertainty import UncertaintyEstimator

__all__ = ['CottonDiseaseEnsemble', 'UncertaintyEstimator']
