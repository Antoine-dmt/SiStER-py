"""
SiSteR-py: Staggered-grid Iterative Solver for the Earth's Rheology in Python

A production-grade geodynamic simulation framework with fully-staggered grids,
modular rheology, and high-performance material property evaluation.
"""

__version__ = "0.1.0"
__author__ = "SiSteR-py Contributors"

from sister_py.config import ConfigurationManager, Material

__all__ = ["ConfigurationManager", "Material"]
