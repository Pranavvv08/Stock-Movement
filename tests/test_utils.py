"""
Unit tests for data alignment and preprocessing utilities.

Run with: python -m pytest tests/test_utils.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_alignment import (
    align_tweet_to_trading_day,
    validate_datasets,
    create_labels_from_stock_movement
)
from utils.preprocessing import (
    prepare_features,
    validate_feature_shape,
    FEATURE_SKIP,
    TIME_STEPS,
    FEATURES_PER_STEP,
    EXPECTED_FEATURE_COUNT
)


class TestDataAlignment:
    """Tests for data alignment functions."""
    
    def test_align_tweet_to_trading_day_exact_match(self):
        """Test alignment when tweet is on a trading day."""
        # Create sample trading days
        trading_days = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10, freq='D')
        })
        
        tweet_timestamp = pd.Timestamp('2020-01-05')
        aligned = align_tweet_to_trading_day(tweet_timestamp, trading_days)
        
        assert aligned == pd.Timestamp('2020-01-05')
    
    def test_align_tweet_to_trading_day_weekend(self):
        """Test alignment when tweet is on a non-trading day."""
        # Trading days (weekdays only)
        trading_days = pd.DataFrame({
            'Date': pd.to_datetime(['2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09', '2020-01-10'])
        })
        
        # Saturday - should align to Friday
        tweet_timestamp = pd.Timestamp('2020-01-11')
        aligned = align_tweet_to_trading_day(tweet_timestamp, trading_days)
        
        assert aligned == pd.Timestamp('2020-01-10')
    
    def test_validate_datasets_valid(self):
        """Test validation with valid datasets."""
        stock_df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=5),
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106]
        })
        
        tweets_df = pd.DataFrame({
            'Tweets': ['Tweet 1', 'Tweet 2', 'Tweet 3', 'Tweet 4', 'Tweet 5']
        })
        
        is_valid, errors = validate_datasets(stock_df, tweets_df)
        
        # Should have warning about missing timestamp but still be valid
        assert is_valid == True
        assert any('WARNING' in e for e in errors)
    
    def test_validate_datasets_missing_columns(self):
        """Test validation with missing required columns."""
        stock_df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=5),
            'Open': [100, 101, 102, 103, 104]
            # Missing High, Low, Close
        })
        
        tweets_df = pd.DataFrame({
            'Tweets': ['Tweet 1', 'Tweet 2']
        })
        
        is_valid, errors = validate_datasets(stock_df, tweets_df)
        
        assert is_valid == False
        assert any('High' in e for e in errors)
    
    def test_create_labels_from_stock_movement(self):
        """Test label creation from price movements."""
        stock_df = pd.DataFrame({
            'Close': [100, 102, 101, 105, 103]  # up, down, up, down
        })
        
        labels = create_labels_from_stock_movement(stock_df)
        
        assert len(labels) == 5
        assert labels[0] == 0  # First day (no previous)
        assert labels[1] == 1  # 102 > 100
        assert labels[2] == 0  # 101 < 102
        assert labels[3] == 1  # 105 > 101
        assert labels[4] == 0  # 103 < 105


class TestPreprocessing:
    """Tests for preprocessing functions."""
    
    def test_validate_feature_shape_correct(self):
        """Test validation with correct feature shape."""
        # Single sample with 770 features
        X = np.random.randn(EXPECTED_FEATURE_COUNT)
        
        # Should not raise
        validate_feature_shape(X, stage="test")
    
    def test_validate_feature_shape_incorrect(self):
        """Test validation with incorrect feature shape."""
        # Wrong number of features
        X = np.random.randn(500)
        
        with pytest.raises(ValueError, match="Feature shape validation failed"):
            validate_feature_shape(X, stage="test")
    
    def test_validate_feature_shape_batch(self):
        """Test validation with batch of samples."""
        # Batch with correct features
        X = np.random.randn(10, EXPECTED_FEATURE_COUNT)
        
        # Should not raise
        validate_feature_shape(X, stage="test")
        
        # Batch with wrong features
        X_wrong = np.random.randn(10, 500)
        
        with pytest.raises(ValueError, match="Feature shape validation failed"):
            validate_feature_shape(X_wrong, stage="test")
    
    def test_prepare_features_shape(self):
        """Test feature preparation produces correct shape."""
        n_samples = 10
        
        # Create fake BERT embeddings (768 dims)
        bert_embeddings = np.random.randn(n_samples, 768)
        
        # Create fake stock features (4 dims)
        stock_features = np.random.randn(n_samples, 4)
        
        # Prepare features with a new scaler
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler((0, 1))
        
        X_reshaped, _ = prepare_features(
            bert_embeddings,
            stock_features,
            normalize=True,
            scaler=scaler,
            fit_scaler=True
        )
        
        # Check output shape
        assert X_reshaped.shape == (n_samples, TIME_STEPS, FEATURES_PER_STEP)
    
    def test_prepare_features_values_normalized(self):
        """Test that features are properly normalized."""
        n_samples = 5
        
        bert_embeddings = np.random.randn(n_samples, 768) * 100  # Large values
        stock_features = np.random.randn(n_samples, 4) * 1000
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler((0, 1))
        
        X_reshaped, fitted_scaler = prepare_features(
            bert_embeddings,
            stock_features,
            normalize=True,
            scaler=scaler,
            fit_scaler=True
        )
        
        # Values should be in [0, 1] after normalization
        assert X_reshaped.min() >= 0
        assert X_reshaped.max() <= 1
    
    def test_prepare_features_mismatched_samples(self):
        """Test error handling for mismatched sample counts."""
        bert_embeddings = np.random.randn(10, 768)
        stock_features = np.random.randn(5, 4)  # Different count
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler((0, 1))
        
        with pytest.raises(ValueError, match="Sample count mismatch"):
            prepare_features(
                bert_embeddings,
                stock_features,
                normalize=True,
                scaler=scaler,
                fit_scaler=True
            )
    
    def test_prepare_features_wrong_dimensions(self):
        """Test error handling for wrong feature dimensions."""
        n_samples = 10
        
        # Wrong BERT dimension
        bert_embeddings = np.random.randn(n_samples, 500)  # Should be 768
        stock_features = np.random.randn(n_samples, 4)
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler((0, 1))
        
        with pytest.raises(ValueError, match="BERT embeddings should have 768 dimensions"):
            prepare_features(
                bert_embeddings,
                stock_features,
                normalize=True,
                scaler=scaler,
                fit_scaler=True
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
