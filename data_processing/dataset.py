"""Compatibility wrapper for dataset loading utilities.

The codebase currently stores the implementation in `dataset/dataset.py`,
while some entrypoints still import `data_processing.dataset`.
This module re-exports the public API so both import paths work.
"""

from dataset.dataset import *  # noqa: F401,F403
