"""
Preprocessing utilities for feature engineering and scaling.

This module ensures consistent preprocessing between training and inference.
"""

import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler


# Model input shape constants - DO NOT CHANGE
# These values are specific to the trained models and must remain consistent
FEATURE_SKIP = 2  # Skip first 2 features after normalization (historical model compatibility)
TIME_STEPS = 35   # Reshape to 35 time steps
FEATURES_PER_STEP = 22  # 22 features per time step
EXPECTED_FEATURE_COUNT = TIME_STEPS * FEATURES_PER_STEP  # 770 total features


def validate_feature_shape(X, stage="unknown"):
    """
    Validate that feature array has the correct shape for the model.
    
    Args:
        X: Feature array after skipping first FEATURE_SKIP features
        stage: String describing when this validation occurs (for error messages)
    
    Raises:
        ValueError: If feature shape is incorrect
    """
    if X.ndim == 1:
        # Single sample - should have EXPECTED_FEATURE_COUNT features
        if X.shape[0] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Feature shape validation failed at {stage}: "
                f"Expected {EXPECTED_FEATURE_COUNT} features after skipping first {FEATURE_SKIP}, "
                f"got {X.shape[0]}. "
                f"Full feature vector should have {EXPECTED_FEATURE_COUNT + FEATURE_SKIP} features."
            )
    elif X.ndim == 2:
        # Multiple samples
        if X.shape[1] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Feature shape validation failed at {stage}: "
                f"Expected {EXPECTED_FEATURE_COUNT} features after skipping first {FEATURE_SKIP}, "
                f"got {X.shape[1]}. "
                f"Full feature vector should have {EXPECTED_FEATURE_COUNT + FEATURE_SKIP} features."
            )
    else:
        raise ValueError(f"Feature array has invalid number of dimensions: {X.ndim}")


def prepare_features(bert_embeddings, stock_features, normalize=True, scaler=None, fit_scaler=False):
    """
    Prepare features for model input by merging, normalizing, and reshaping.
    
    This function:
    1. Merges BERT embeddings (768 dims) with stock features (4 dims) -> 772 total
    2. Normalizes using MinMaxScaler
    3. Skips first FEATURE_SKIP (2) features -> 770 features
    4. Validates feature count matches TIME_STEPS * FEATURES_PER_STEP (35 * 22 = 770)
    5. Reshapes to (batch_size, TIME_STEPS, FEATURES_PER_STEP) for LSTM/GRU input
    
    Args:
        bert_embeddings: Array of BERT embeddings, shape (n_samples, 768)
        stock_features: Array of stock features, shape (n_samples, 4) [Open, High, Low, Close]
        normalize: Whether to normalize features
        scaler: Pre-fitted MinMaxScaler (for inference). If None and normalize=True, creates new scaler
        fit_scaler: If True, fit the scaler on this data (training only)
    
    Returns:
        tuple: (X_reshaped, scaler_used)
            X_reshaped: Features shaped as (n_samples, TIME_STEPS, FEATURES_PER_STEP)
            scaler_used: The scaler object (fitted if fit_scaler=True)
    """
    # Validate inputs
    if bert_embeddings.shape[0] != stock_features.shape[0]:
        raise ValueError(
            f"Sample count mismatch: BERT embeddings has {bert_embeddings.shape[0]} samples, "
            f"stock features has {stock_features.shape[0]} samples"
        )
    
    if bert_embeddings.shape[1] != 768:
        raise ValueError(f"BERT embeddings should have 768 dimensions, got {bert_embeddings.shape[1]}")
    
    if stock_features.shape[1] != 4:
        raise ValueError(f"Stock features should have 4 dimensions [Open,High,Low,Close], got {stock_features.shape[1]}")
    
    # Merge BERT embeddings and stock features
    # Result: (n_samples, 772)
    X = np.hstack((bert_embeddings, stock_features))
    
    # Normalize if requested
    scaler_used = None
    if normalize:
        if scaler is None:
            # Create new scaler
            scaler_used = MinMaxScaler((0, 1))
            if fit_scaler:
                X = scaler_used.fit_transform(X)
            else:
                raise ValueError("normalize=True but no scaler provided and fit_scaler=False")
        else:
            # Use provided scaler
            scaler_used = scaler
            if fit_scaler:
                X = scaler_used.fit_transform(X)
            else:
                X = scaler_used.transform(X)
    
    # Skip first FEATURE_SKIP features
    # Original notebook behavior: skip first 2 features for model compatibility
    X = X[:, FEATURE_SKIP:]
    
    # Validate feature count
    validate_feature_shape(X, stage="after_skip")
    
    # Reshape to 3D for LSTM/GRU: (n_samples, TIME_STEPS, FEATURES_PER_STEP)
    X_reshaped = X.reshape(X.shape[0], TIME_STEPS, FEATURES_PER_STEP)
    
    return X_reshaped, scaler_used


def save_scaler(scaler, filepath):
    """
    Save a fitted scaler to disk.
    
    Args:
        scaler: Fitted MinMaxScaler object
        filepath: Path to save the scaler (e.g., 'model/scaler.pkl')
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {filepath}")


def load_scaler(filepath):
    """
    Load a fitted scaler from disk.
    
    Args:
        filepath: Path to the saved scaler (e.g., 'model/scaler.pkl')
    
    Returns:
        MinMaxScaler object, or None if file doesn't exist
    """
    if not os.path.exists(filepath):
        print(f"Warning: Scaler file not found at {filepath}")
        return None
    
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {filepath}")
    return scaler


def prepare_single_prediction(tweet_embedding, stock_data_dict, scaler):
    """
    Prepare features for a single prediction (used in Streamlit app).
    
    Args:
        tweet_embedding: BERT embedding for one tweet, shape (768,) or (1, 768)
        stock_data_dict: Dictionary with keys 'Open', 'High', 'Low', 'Close'
        scaler: Pre-fitted MinMaxScaler from training
    
    Returns:
        numpy array: Features shaped as (1, TIME_STEPS, FEATURES_PER_STEP)
    """
    # Ensure tweet embedding is 2D
    if tweet_embedding.ndim == 1:
        tweet_embedding = tweet_embedding.reshape(1, -1)
    
    # Validate tweet embedding shape
    if tweet_embedding.shape[1] != 768:
        raise ValueError(f"Tweet embedding should have 768 dimensions, got {tweet_embedding.shape[1]}")
    
    # Extract stock features in correct order
    stock_features = np.array([[
        float(stock_data_dict['Open']),
        float(stock_data_dict['High']),
        float(stock_data_dict['Low']),
        float(stock_data_dict['Close'])
    ]])
    
    # Use prepare_features with the scaler
    X_reshaped, _ = prepare_features(
        tweet_embedding,
        stock_features,
        normalize=True,
        scaler=scaler,
        fit_scaler=False
    )
    
    return X_reshaped
