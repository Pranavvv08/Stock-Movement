"""
Utility modules for Stock Movement Prediction system.

This package provides shared utilities for data alignment, preprocessing,
and feature engineering used across training and inference pipelines.
"""

from .data_alignment import validate_datasets, align_tweets_to_stock_data, prepare_aligned_dataset
from .preprocessing import (
    prepare_features,
    validate_feature_shape,
    save_scaler,
    load_scaler
)

__all__ = [
    'validate_datasets',
    'align_tweets_to_stock_data',
    'prepare_aligned_dataset',
    'prepare_features',
    'validate_feature_shape',
    'save_scaler',
    'load_scaler'
]
