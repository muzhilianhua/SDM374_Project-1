"""SDM374 Project 1: surrogate models and optimizers for building-controller tuning."""
import os

# Multi-threaded BLAS on these small matrices oversubscribes cores: ridge2.fit goes from ~1 s to ~15 s.
# Must run before numpy is first imported, as baseline.py does.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

__version__ = "0.1.0"
