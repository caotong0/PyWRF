"""PyWRF 9 km example — command-line entry point.

Runs the PyWRF model end to end for the shipped 9 km case configuration.
This file is deliberately tiny: the actual solver lives in
:mod:`pywrf.solver` (see ``WrfSolver.solve``).

Usage:
    python examples/run_wrf_9km.py

The WRF NetCDF input files must be available under ``data/`` (or point
``PYWRF_DATA_DIR`` / ``PYWRF_RUN_NAME`` at them) — see ``data/README.md``.
"""
import os
import sys

# Make the repository root importable when running this file directly without
# installing the package (`pip install -e .` or `python -m pywrf`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pywrf.solver import main  # noqa: E402

if __name__ == "__main__":
    main()
