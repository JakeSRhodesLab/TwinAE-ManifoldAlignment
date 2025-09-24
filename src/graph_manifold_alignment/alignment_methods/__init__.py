"""
Alignment Methods Module

This module contains various manifold alignment algorithms.
"""

from .DTA_andres import *
from .jlma import *
from .ma_procrustes import *
from .mali import *
from .MASH_MD import *
from .ssma import *

# Optional imports for TensorFlow-dependent modules
try:
    from .MAGAN import *
except ImportError:
    pass  # TensorFlow may not be available in all environments

__all__ = [
    # Add specific function/class names as needed
]
