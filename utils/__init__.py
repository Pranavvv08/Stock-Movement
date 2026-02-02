"""
Utility modules for Stock Movement Prediction system.

This package provides shared utilities for data alignment, preprocessing,
and feature engineering used across training and inference pipelines.
"""

from .data_alignment import align_tweet_to_trading_day, validate_datasets
from .preprocessing import (
    prepare_features,
    validate_feature_shape,
    save_scaler,
    load_scaler
)

__all__ = [
    'align_tweet_to_trading_day',
    'validate_datasets',
    'prepare_features',
    'validate_feature_shape',
    'save_scaler',
    'load_scaler'
]
