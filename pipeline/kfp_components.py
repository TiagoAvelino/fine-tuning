"""Re-export KFP components for convenience.

All components are defined in training_pipeline.py.
"""

from pipeline.training_pipeline import (
    prepare_dataset,
    train_model,
    evaluate_model,
    upload_model,
)

__all__ = [
    "prepare_dataset",
    "train_model",
    "evaluate_model",
    "upload_model",
]
