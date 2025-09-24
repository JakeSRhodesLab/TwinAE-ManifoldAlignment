"""
Graph Manifold Alignment - Twin Autoencoders

Implementation of "Guided Manifold Alignment with Geometry-Regularized Twin Autoencoders" - MMAI 2025 @ IEEE ICDM

This package provides novel Geometric Regularized Autoencoders (GRAE) for manifold 
alignment, along with comprehensive baseline methods for comparison.

Key Features:
- Twin Autoencoder architecture with geometric regularization
- GRAEAnchor: Anchor-guided alignment for improved accuracy  
- Support for multimodal data (neuroimaging, tabular, images)
- Comprehensive evaluation and visualization tools
"""

__version__ = "0.1.0"
__author__ = "Jake Rhodes Lab"
__paper__ = "Guided Manifold Alignment with Geometry-Regularized Twin Autoencoders, MMAI 2025 @ IEEE ICDM"

from . import alignment_methods
from . import helpers
from . import adni
from . import autoencoder

__all__ = [
    "alignment_methods",
    "helpers", 
    "adni",
    "autoencoder"
]
