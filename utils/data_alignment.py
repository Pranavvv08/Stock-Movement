"""
Data alignment utilities for Stock Movement Prediction.

This module provides functions to align tweets to stock data using index-based pairing
and validate datasets.

Note: Timestamp-based alignment has been removed to improve accuracy.
The system now uses simple index-based pairing which assumes tweets and stock data
are pre-aligned in the dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def validate_datasets(stock_df, tweets_df):
    """
    Validate that datasets have required columns and proper format.
    
    Args:
        stock_df: DataFrame with stock data
        tweets_df: DataFrame with tweets data
    
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Validate stock dataset
    required_stock_cols = ['Date', 'Open', 'High', 'Low', 'Close']
    for col in required_stock_cols:
        if col not in stock_df.columns:
            errors.append(f"Stock dataset missing required column: {col}")
    
    if 'Date' in stock_df.columns:
        try:
            # Ensure Date is datetime
            if not pd.api.types.is_datetime64_any_dtype(stock_df['Date']):
                errors.append("Stock dataset 'Date' column is not datetime type")
        except Exception as e:
            errors.append(f"Error checking stock Date column: {e}")
    
    # Validate tweets dataset
    required_tweet_cols = ['Tweets']
    for col in required_tweet_cols:
        if col not in tweets_df.columns:
            errors.append(f"Tweets dataset missing required column: {col}")
    
    is_valid = len([e for e in errors if not e.startswith('WARNING')]) == 0
    return is_valid, errors


def create_labels_from_stock_movement(stock_df):
    """
    Create binary labels from stock price movements.
    
    Label = 1 if Close[t] > Close[t-1] (price went up)
    Label = 0 if Close[t] <= Close[t-1] (price went down or stayed same)
    
    Args:
        stock_df: DataFrame with 'Close' column
    
    Returns:
        numpy array: Binary labels
    """
    if 'Close' not in stock_df.columns:
        raise ValueError("Stock dataset must have 'Close' column")
    
    # Calculate price changes
    close_prices = stock_df['Close'].values
    labels = np.zeros(len(close_prices), dtype=int)
    
    # First day has no previous day, default to 0
    for i in range(1, len(close_prices)):
        labels[i] = 1 if close_prices[i] > close_prices[i-1] else 0
    
    return labels


def align_tweets_to_stock_data(tweets_df, stock_df, timestamp_col=None):
    """
    Align tweets to stock data using simple index-based pairing.
    
    This function uses index-based alignment which assumes tweets and stock data
    are pre-aligned in the dataset. Timestamp-based alignment has been removed
    to improve model accuracy.
    
    Args:
        tweets_df: DataFrame with tweets
        stock_df: DataFrame with stock data (must have 'Date' column)
        timestamp_col: Ignored (kept for API compatibility)
    
    Returns:
        tuple: (aligned_df, alignment_info)
    """
    # Always use index-based pairing for better accuracy
    print("Using index-based pairing (tweets and stock data assumed pre-aligned).")
    
    # Ensure both datasets have same length or truncate to minimum
    min_len = min(len(tweets_df), len(stock_df))
    aligned_tweets = tweets_df.iloc[:min_len].copy()
    aligned_stock = stock_df.iloc[:min_len].copy()
    
    alignment_info = {
        'method': 'index_based',
        'original_tweets': len(tweets_df),
        'original_stock': len(stock_df),
        'aligned_count': min_len
    }
    
    return aligned_tweets, aligned_stock, alignment_info


def prepare_aligned_dataset(tweets_df, stock_df, timestamp_col=None, use_existing_labels=True):
    """
    Prepare a fully aligned dataset with proper labels using index-based pairing.
    
    Args:
        tweets_df: DataFrame with tweets (and optional Label column)
        stock_df: DataFrame with stock data (and optional Label column)
        timestamp_col: Ignored (kept for API compatibility)
        use_existing_labels: If True and Label exists, use it; otherwise compute from price movement
    
    Returns:
        tuple: (merged_df, labels, alignment_info)
    """
    # Validate datasets
    is_valid, errors = validate_datasets(stock_df, tweets_df)
    if not is_valid:
        raise ValueError(f"Dataset validation failed: {errors}")
    
    # Print warnings (but ignore timestamp warnings since we use index-based)
    for error in errors:
        if error.startswith('WARNING') and 'timestamp' not in error.lower():
            print(error)
    
    # Always use index-based alignment for better accuracy
    min_len = min(len(tweets_df), len(stock_df))
    # Handle duplicate column names by removing Label from stock_df if present in both
    stock_to_concat = stock_df.iloc[:min_len].reset_index(drop=True)
    if 'Label' in tweets_df.columns and 'Label' in stock_df.columns:
        stock_to_concat = stock_to_concat.drop(columns=['Label'])
    merged = pd.concat([
        tweets_df.iloc[:min_len].reset_index(drop=True),
        stock_to_concat
    ], axis=1)
    alignment_info = {
        'method': 'index_based',
        'aligned_count': min_len
    }
    
    # Determine labels
    # Handle case where Label column might still be duplicated (e.g., from merge)
    if use_existing_labels and 'Label' in merged.columns:
        label_data = merged['Label']
        # If Label is a DataFrame (duplicate columns), take the first column
        if isinstance(label_data, pd.DataFrame):
            labels = label_data.iloc[:, 0].values
        else:
            labels = label_data.values
        alignment_info['label_source'] = 'existing'
    else:
        # Compute labels from stock movement
        labels = create_labels_from_stock_movement(merged)
        alignment_info['label_source'] = 'computed_from_price_movement'
    
    return merged, labels, alignment_info
