"""
Data alignment utilities for Stock Movement Prediction.

This module provides functions to align tweets to trading days and validate datasets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def align_tweet_to_trading_day(tweet_timestamp, trading_days_df):
    """
    Align a tweet timestamp to the corresponding trading day.
    
    If the tweet timestamp falls on a trading day, return that day.
    Otherwise, return the most recent previous trading day.
    
    Args:
        tweet_timestamp: pandas Timestamp or datetime object
        trading_days_df: DataFrame with 'Date' column containing trading days (as datetime)
    
    Returns:
        pandas Timestamp: The aligned trading day
    """
    if not isinstance(tweet_timestamp, (pd.Timestamp, datetime)):
        raise ValueError(f"tweet_timestamp must be a datetime or Timestamp, got {type(tweet_timestamp)}")
    
    # Ensure trading days are sorted
    trading_days = trading_days_df['Date'].sort_values().values
    
    # Convert to datetime64 for comparison
    tweet_date = pd.Timestamp(tweet_timestamp).normalize()
    
    # Find the trading day on or before the tweet date
    mask = trading_days <= tweet_date
    
    if not mask.any():
        # Tweet is before all trading days - use first trading day
        return pd.Timestamp(trading_days[0])
    
    # Return the most recent trading day on or before the tweet
    return pd.Timestamp(trading_days[mask][-1])


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
    
    # Check if tweets have timestamps (optional but recommended)
    has_timestamp = False
    for col in ['timestamp', 'Timestamp', 'Date', 'date', 'created_at']:
        if col in tweets_df.columns:
            has_timestamp = True
            break
    
    if not has_timestamp:
        errors.append(
            "WARNING: Tweets dataset has no timestamp column. "
            "Using index-based pairing as fallback. "
            "For accurate alignment, add a timestamp column (e.g., 'timestamp', 'Date')."
        )
    
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
    Align tweets to stock data using timestamps or index-based pairing.
    
    Args:
        tweets_df: DataFrame with tweets
        stock_df: DataFrame with stock data (must have 'Date' column)
        timestamp_col: Name of timestamp column in tweets_df (if None, auto-detect)
    
    Returns:
        tuple: (aligned_tweets_df, aligned_stock_df, alignment_info)
    """
    # Auto-detect timestamp column if not provided
    if timestamp_col is None:
        possible_cols = ['timestamp', 'Timestamp', 'Date', 'date', 'created_at']
        for col in possible_cols:
            if col in tweets_df.columns:
                timestamp_col = col
                break
    
    if timestamp_col is None:
        # Fallback: use index-based pairing
        print("WARNING: No timestamp column found. Using index-based pairing.")
        print("This assumes tweets and stock data are pre-aligned by index.")
        
        # Ensure both datasets have same length or truncate to minimum
        min_len = min(len(tweets_df), len(stock_df))
        aligned_tweets = tweets_df.iloc[:min_len].copy()
        aligned_stock = stock_df.iloc[:min_len].copy()
        
        alignment_info = {
            'method': 'index_based',
            'original_tweets': len(tweets_df),
            'original_stock': len(stock_df),
            'aligned_count': min_len,
            'warning': 'Timestamp-based alignment not available'
        }
        
        return aligned_tweets, aligned_stock, alignment_info
    
    # Timestamp-based alignment
    print(f"Using timestamp column: {timestamp_col}")
    
    # Ensure timestamps are datetime
    tweets_with_time = tweets_df.copy()
    tweets_with_time['timestamp'] = pd.to_datetime(tweets_with_time[timestamp_col])
    
    # Ensure stock dates are datetime
    stock_with_date = stock_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(stock_with_date['Date']):
        stock_with_date['Date'] = pd.to_datetime(stock_with_date['Date'])
    
    # Align each tweet to a trading day
    tweets_with_time['aligned_trading_day'] = tweets_with_time['timestamp'].apply(
        lambda ts: align_tweet_to_trading_day(ts, stock_with_date)
    )
    
    # Join with stock data
    merged = tweets_with_time.merge(
        stock_with_date,
        left_on='aligned_trading_day',
        right_on='Date',
        how='inner'
    )
    
    alignment_info = {
        'method': 'timestamp_based',
        'timestamp_column': timestamp_col,
        'original_tweets': len(tweets_df),
        'original_stock': len(stock_df),
        'aligned_count': len(merged),
        'date_range': f"{merged['Date'].min()} to {merged['Date'].max()}"
    }
    
    # Return aligned data
    # Extract original columns from tweets and stock
    tweet_cols = [c for c in tweets_df.columns if c != timestamp_col] + ['timestamp', 'aligned_trading_day']
    stock_cols = [c for c in stock_df.columns]
    
    return merged, alignment_info


def prepare_aligned_dataset(tweets_df, stock_df, timestamp_col=None, use_existing_labels=True):
    """
    Prepare a fully aligned dataset with proper labels.
    
    Args:
        tweets_df: DataFrame with tweets (and optional Label column)
        stock_df: DataFrame with stock data (and optional Label column)
        timestamp_col: Name of timestamp column in tweets_df
        use_existing_labels: If True and Label exists, use it; otherwise compute from price movement
    
    Returns:
        tuple: (merged_df, labels, alignment_info)
    """
    # Validate datasets
    is_valid, errors = validate_datasets(stock_df, tweets_df)
    if not is_valid:
        raise ValueError(f"Dataset validation failed: {errors}")
    
    # Print warnings
    for error in errors:
        if error.startswith('WARNING'):
            print(error)
    
    # Align datasets
    if timestamp_col or any(col in tweets_df.columns for col in ['timestamp', 'Timestamp', 'Date', 'date', 'created_at']):
        merged, alignment_info = align_tweets_to_stock_data(tweets_df, stock_df, timestamp_col)
    else:
        # Index-based fallback
        min_len = min(len(tweets_df), len(stock_df))
        merged = pd.concat([
            tweets_df.iloc[:min_len].reset_index(drop=True),
            stock_df.iloc[:min_len].reset_index(drop=True)
        ], axis=1)
        alignment_info = {
            'method': 'index_based',
            'aligned_count': min_len
        }
    
    # Determine labels
    if use_existing_labels and 'Label' in merged.columns:
        labels = merged['Label'].values
        alignment_info['label_source'] = 'existing'
    else:
        # Compute labels from stock movement
        labels = create_labels_from_stock_movement(merged)
        alignment_info['label_source'] = 'computed_from_price_movement'
    
    return merged, labels, alignment_info
