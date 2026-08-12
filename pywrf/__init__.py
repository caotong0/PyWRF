"""PyWRF — a PyTorch-based, differentiable reimplementation of the WRF model.

This package contains a pure-Python/PyTorch port of the WRF (Weather Research
and Forecasting) dynamical core, physics and lateral boundary conditions,
written to support differentiable / AI-for-weather research.

See the README for usage and the MIT license for terms.

Public API:
    ``pywrf.WrfSolver`` — the model solver (:meth:`~WrfSolver.solve` runs one
    full integration). ``pywrf.main`` — the command-line entry point.
"""

from pywrf.solver import WrfSolver, main

__all__ = ["WrfSolver", "main"]
__version__ = "0.1.0"
