"""
Utility functions and pipeline entry points extracted from the kernel_model notebook.

The modules in this package break the notebook into reusable pieces so the
notebook can focus on experiments while the heavy lifting lives in Python code.
"""

from .config import (
    KernelBankConfig,
    PatchExtractionConfig,
    PipelineConfig,
    SelectionConfig,
    SubsetSearchConfig,
    TrainingConfig,
)
from .pipeline import run_pipeline

__all__ = [
    "KernelBankConfig",
    "PatchExtractionConfig",
    "PipelineConfig",
    "SelectionConfig",
    "SubsetSearchConfig",
    "TrainingConfig",
    "run_pipeline",
]
