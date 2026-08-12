"""PyWRF 9 km example — command-line entry point.

Runs the PyWRF model end to end for the shipped 9 km case configuration.
This file is deliberately tiny: the actual solver lives in
:mod:`pywrf.solver` (see ``WrfSolver.solve``).

Usage:
    python examples/run_wrf_9km.py

The real 9 km case data ships with the repo under ``data/`` — see
``data/README.md`` (``PYWRF_DATA_DIR`` / ``PYWRF_RUN_NAME`` override the paths).
"""
import os
import sys

# Make the repository root importable when running this file directly without
# installing the package (`pip install -e .` or `python -m pywrf`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pywrf.solver import main  # noqa: E402

if __name__ == "__main__":
    main()
