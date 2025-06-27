"""
UQEF-Dynamic: Framework for Efficient Uncertainty Quantification and Sensitivity Analysis of Time-dependent Model Outputs

This package provides tools and models for uncertainty quantification and sensitivity analysis
of time-dependent model outputs.
"""

__version__ = "0.1"
__author__ = "Ivana Jovanovic Buha"
__email__ = "ivana.jovanovic@tum.de"

# Import key modules and utilities
from . import utils
from . import models
from . import scientific_pipelines

# Make commonly used utilities easily accessible
from .utils import utility

# Version information
__all__ = [
    "utils",
    "models", 
    "scientific_pipelines",
    "utility",
    "__version__",
    "__author__",
    "__email__"
]
