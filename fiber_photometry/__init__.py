"""
Fiber Photometry Analysis Pipeline

A robust, reproducible analysis pipeline for fiber-photometry recordings,
inspired by GuPPy (Scientific Reports, 2021).
"""

__version__ = "0.1.0"
__author__ = "Scientific Python Engineer"

from .parser import FluorescenceParser
from .preprocessing import Preprocessor
from .normalization import Normalizer
from .analysis import EventAnalyzer, TransientAnalyzer
from .visualization import Visualizer

__all__ = [
    'FluorescenceParser',
    'Preprocessor', 
    'Normalizer',
    'EventAnalyzer',
    'TransientAnalyzer',
    'Visualizer'
]