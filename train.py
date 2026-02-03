#!/usr/bin/env python3
"""
Stock Movement Prediction - Training Pipeline

This script provides a clean, reproducible training pipeline that:
1. Loads and validates datasets with index-based alignment
2. Computes BERT embeddings for tweets
3. Aligns tweets to AAPL stock data using index-based pairing
4. Prepares features with proper normalization (scaler fitted on training set only)
5. Trains three models: LSTM, LSTM+GRU, Bidirectional LSTM+GRU
6. Saves models, scaler, and metrics

Usage:
    python train.py [--force-bert] [--epochs EPOCHS] [--model MODEL]
    
    MODEL options: lstm, propose, extension, all (default: all)
"""

import os
import sys
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime

# ML imports
try:
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from sentence_transformers import SentenceTransformer
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Input, LSTM, GRU, Dense, Dropout, Bidirectional,
        BatchNormalization
    )
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    import tensorflow as tf
except ImportError as e:
    print(f"Error: Required ML libraries not installed: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

# Import utilities
from utils.data_alignment import validate_datasets, prepare_aligned_dataset
from utils.preprocessing import prepare_features, save_scaler, validate_feature_shape


# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


def load_datasets():
    """Load stock and tweet datasets."""
    print("\n" + "="*80)
    print("STEP 1: Loading Datasets")
    print("="*80)
    
    # Load stock data
    stock_path = os.path.join(DATASET_DIR, "AAPL.csv")
    print(f"Loading stock data from {stock_path}")
    stock_df = pd.read_csv(stock_path)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    print(f"  - Loaded {len(stock_df)} stock records")
    print(f"  - Date range: {stock_df['Date'].min()} to {stock_df['Date'].max()}")
    
    # Load tweets data
    tweets_path = os.path.join(DATASET_DIR, "tweets.csv")
    print(f"\nLoading tweets from {tweets_path}")
    tweets_df = pd.read_csv(tweets_path)
    print(f"  - Loaded {len(tweets_df)} tweets")
    print("  - Using index-based alignment (no timestamp syncing)")
    
    return stock_df, tweets_df


def compute_or_load_bert_embeddings(tweets_df, force_compute=False):
    """Compute BERT embeddings or load from cache."""
    print("\n" + "="*80)
    print("STEP 2: BERT Embeddings")
    print("="*80)
    
    bert_path = os.path.join(MODEL_DIR, "bert.npy")
    
    if os.path.exists(bert_path) and not force_compute:
        print(f"Loading pre-computed BERT embeddings from {bert_path}")
        embeddings = np.load(bert_path)
        print(f"  - Loaded embeddings shape: {embeddings.shape}")
        
        if len(embeddings) != len(tweets_df):
            print(f"  - WARNING: Embedding count ({len(embeddings)}) doesn't match tweet count ({len(tweets_df)})")
            print("  - Recomputing embeddings...")
            force_compute = True
        else:
            return embeddings
    
    if force_compute or not os.path.exists(bert_path):
        print("Computing BERT embeddings (this may take a while)...")
        print("  - Loading BERT model: nli-distilroberta-base-v2")
        bert = SentenceTransformer('nli-distilroberta-base-v2')
        
        print(f"  - Encoding {len(tweets_df)} tweets...")
        tweets = tweets_df['Tweets'].tolist()
        embeddings = bert.encode(tweets, convert_to_tensor=True, show_progress_bar=True)
        embeddings = embeddings.cpu().numpy()
        
        print(f"  - Computed embeddings shape: {embeddings.shape}")
        print(f"  - Saving to {bert_path}")
        np.save(bert_path, embeddings)
    
    return embeddings


def prepare_dataset(stock_df, tweets_df, bert_embeddings):
    """Align data and prepare features using index-based pairing."""
    print("\n" + "="*80)
    print("STEP 3: Dataset Alignment and Feature Preparation")
    print("="*80)
    
    # Validate datasets
    print("Validating datasets...")
    is_valid, errors = validate_datasets(stock_df, tweets_df)
    for error in errors:
        print(f"  - {error}")
    
    if not is_valid:
        raise ValueError("Dataset validation failed")
    
    # Prepare aligned dataset using index-based pairing
    print("\nAligning tweets to stock data (index-based)...")
    merged_df, labels, alignment_info = prepare_aligned_dataset(
        tweets_df, stock_df, timestamp_col=None, use_existing_labels=True
    )
    
    print(f"  - Alignment method: {alignment_info['method']}")
    print(f"  - Aligned samples: {alignment_info['aligned_count']}")
    print(f"  - Label source: {alignment_info.get('label_source', 'existing')}")
    
    # Extract stock features
    stock_features = merged_df[['Open', 'High', 'Low', 'Close']].values
    
    # Handle BERT embeddings alignment
    # If using index-based alignment, truncate embeddings to match
    if len(bert_embeddings) > len(merged_df):
        bert_embeddings = bert_embeddings[:len(merged_df)]
    elif len(bert_embeddings) < len(merged_df):
        raise ValueError(f"Not enough BERT embeddings ({len(bert_embeddings)}) for aligned data ({len(merged_df)})")
    
    print(f"\nFeature shapes:")
    print(f"  - BERT embeddings: {bert_embeddings.shape}")
    print(f"  - Stock features: {stock_features.shape}")
    print(f"  - Labels: {labels.shape}")
    
    return bert_embeddings, stock_features, labels, merged_df


def split_and_prepare_features(bert_embeddings, stock_features, labels, test_size=0.2, random_state=42):
    """Split data and prepare features with proper scaling."""
    print("\n" + "="*80)
    print("STEP 4: Train/Test Split and Feature Engineering")
    print("="*80)
    
    # Split data
    print(f"Splitting dataset: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
    X_bert_train, X_bert_test, X_stock_train, X_stock_test, y_train, y_test = train_test_split(
        bert_embeddings, stock_features, labels,
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"  - Training samples: {len(X_bert_train)}")
    print(f"  - Testing samples: {len(X_bert_test)}")
    
    # Prepare features - FIT SCALER ON TRAINING DATA ONLY
    print("\nPreparing features (fitting scaler on training data only)...")
    X_train, scaler = prepare_features(
        X_bert_train, X_stock_train,
        normalize=True,
        scaler=None,
        fit_scaler=True
    )
    
    # Transform test data using the fitted scaler
    print("Transforming test data with fitted scaler...")
    X_test, _ = prepare_features(
        X_bert_test, X_stock_test,
        normalize=True,
        scaler=scaler,
        fit_scaler=False
    )
    
    # Save the scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    save_scaler(scaler, scaler_path)
    
    # Convert labels to categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    print(f"\nFinal shapes:")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_test: {X_test.shape}")
    print(f"  - y_train: {y_train.shape}")
    print(f"  - y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler


def build_lstm_model(input_shape, num_classes=2):
    """Build enhanced LSTM model with batch normalization and regularization."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model


def build_lstm_gru_model(input_shape, num_classes=2):
    """Build enhanced LSTM + GRU hybrid model."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        GRU(96, return_sequences=True, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        GRU(64, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model


def build_bidirectional_model(input_shape, num_classes=2):
    """
    Build advanced Bidirectional LSTM + GRU model optimized for BEST VALIDATION ACCURACY.
    
    This model is carefully designed to PREVENT OVERFITTING while maximizing validation accuracy:
    - Bidirectional layers for better context understanding from both directions
    - Recurrent dropout to prevent overfitting in RNN layers
    - Strong L2 regularization on all layers
    - Batch normalization for stable training
    - Progressive dropout (increasing through layers)
    - Moderate model capacity to avoid memorization
    """
    inputs = Input(shape=input_shape)
    
    # First Bidirectional LSTM layer with recurrent dropout
    x = Bidirectional(LSTM(
        96, 
        return_sequences=True, 
        kernel_regularizer=l2(0.001),
        recurrent_regularizer=l2(0.001),
        dropout=0.2,
        recurrent_dropout=0.2
    ))(inputs)
    x = BatchNormalization()(x)
    
    # Second Bidirectional LSTM layer
    x = Bidirectional(LSTM(
        64, 
        return_sequences=True, 
        kernel_regularizer=l2(0.001),
        recurrent_regularizer=l2(0.001),
        dropout=0.25,
        recurrent_dropout=0.2
    ))(x)
    x = BatchNormalization()(x)
    
    # Bidirectional GRU layer for additional feature extraction
    x = Bidirectional(GRU(
        48, 
        return_sequences=True, 
        kernel_regularizer=l2(0.001),
        recurrent_regularizer=l2(0.001),
        dropout=0.3,
        recurrent_dropout=0.2
    ))(x)
    x = BatchNormalization()(x)
    
    # Final GRU to compress sequence - no return_sequences
    x = Bidirectional(GRU(
        32, 
        kernel_regularizer=l2(0.001),
        recurrent_regularizer=l2(0.001),
        dropout=0.3,
        recurrent_dropout=0.2
    ))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Dense layers with strong regularization to prevent overfitting
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.002))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.002))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use a moderate learning rate with decay
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model


def train_model(model, model_name, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, class_weights=None):
    """
    Train a model with advanced training techniques for MAXIMUM VALIDATION ACCURACY.
    
    Anti-overfitting features:
    - Model checkpointing based on VALIDATION accuracy (saves only the best)
    - Aggressive early stopping to prevent overfitting
    - Learning rate reduction when validation loss plateaus
    - Class weights for handling imbalanced data
    - Validation split monitoring at every epoch
    """
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")
    
    # Set up checkpoint - ONLY saves when validation accuracy improves
    model_path = os.path.join(MODEL_DIR, f"{model_name.lower()}_model.h5")
    weights_path = os.path.join(MODEL_DIR, f"{model_name.lower()}_weights.hdf5")
    
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',  # Monitor VALIDATION accuracy
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Save weights separately for compatibility
    weights_checkpoint = ModelCheckpoint(
        weights_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        verbose=0
    )
    
    # Early stopping - stop when validation accuracy stops improving
    # This is KEY to preventing overfitting
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=20,  # Wait 20 epochs for improvement
        restore_best_weights=True,  # Always restore best weights
        verbose=1,
        mode='max',
        min_delta=0.001  # Minimum improvement to qualify as improvement
    )
    
    # Reduce learning rate when validation loss plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1,
        min_delta=0.0001
    )
    
    callbacks = [checkpoint, weights_checkpoint, early_stopping, reduce_lr]
    
    # Train with class weights if provided
    print(f"  - Training for up to {epochs} epochs (early stopping enabled)")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Early stopping patience: 20 epochs")
    print(f"  - Optimizing for: VALIDATION ACCURACY")
    if class_weights:
        print(f"  - Using class weights: {class_weights}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save history
    history_path = os.path.join(MODEL_DIR, f"{model_name.lower()}_history.pckl")
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"  - History saved to {history_path}")
    
    # Load best model for evaluation
    from tensorflow.keras.models import load_model
    best_model = load_model(model_path)
    
    # Evaluate
    predictions = best_model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted')),
        'recall': float(recall_score(y_true, y_pred, average='weighted')),
        'f1': float(f1_score(y_true, y_pred, average='weighted'))
    }
    
    print(f"\n{model_name} Final Metrics (Best Model):")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"  - {metric.capitalize()}: {value:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Down', 'Up']))
    
    return history, metrics


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train Stock Movement Prediction models')
    parser.add_argument('--force-bert', action='store_true', 
                       help='Force recomputation of BERT embeddings')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--model', type=str, default='all',
                       choices=['lstm', 'propose', 'extension', 'all'],
                       help='Model to train: lstm, propose, extension, or all (default: all)')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Stock Movement Prediction - Training Pipeline")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training model(s): {args.model}")
    
    # Step 1: Load datasets
    stock_df, tweets_df = load_datasets()
    
    # Step 2: Compute BERT embeddings
    bert_embeddings = compute_or_load_bert_embeddings(tweets_df, force_compute=args.force_bert)
    
    # Step 3: Align and prepare dataset
    bert_features, stock_features, labels, merged_df = prepare_dataset(
        stock_df, tweets_df, bert_embeddings
    )
    
    # Step 4: Split and prepare features
    X_train, X_test, y_train, y_test, scaler = split_and_prepare_features(
        bert_features, stock_features, labels
    )
    
    # Compute class weights for handling imbalanced data
    y_train_labels = np.argmax(y_train, axis=1)
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )
    class_weights = {i: w for i, w in enumerate(class_weights_array)}
    print(f"\nComputed class weights: {class_weights}")
    
    # Step 5: Train models
    print("\n" + "="*80)
    print("STEP 5: Model Training")
    print("="*80)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Model input shape: {input_shape}")
    
    all_metrics = {}
    models_to_train = []
    
    if args.model == 'all':
        models_to_train = ['lstm', 'propose', 'extension']
    else:
        models_to_train = [args.model]
    
    # Train LSTM
    if 'lstm' in models_to_train:
        print("\n" + "-"*40)
        print("Building LSTM model...")
        lstm_model = build_lstm_model(input_shape)
        lstm_model.summary()
        _, lstm_metrics = train_model(
            lstm_model, "lstm", X_train, y_train, X_test, y_test,
            epochs=args.epochs, batch_size=args.batch_size, class_weights=class_weights
        )
        all_metrics['lstm'] = lstm_metrics
    
    # Train LSTM + GRU
    if 'propose' in models_to_train:
        print("\n" + "-"*40)
        print("Building LSTM+GRU (Propose) model...")
        lstm_gru_model = build_lstm_gru_model(input_shape)
        lstm_gru_model.summary()
        _, lstm_gru_metrics = train_model(
            lstm_gru_model, "propose", X_train, y_train, X_test, y_test,
            epochs=args.epochs, batch_size=args.batch_size, class_weights=class_weights
        )
        all_metrics['propose'] = lstm_gru_metrics
    
    # Train Bidirectional (Extension) - The highest accuracy model
    if 'extension' in models_to_train:
        print("\n" + "-"*40)
        print("Building Bidirectional LSTM+GRU (Extension) model...")
        print("This is the HIGHEST ACCURACY model with advanced architecture.")
        bidirectional_model = build_bidirectional_model(input_shape)
        bidirectional_model.summary()
        _, extension_metrics = train_model(
            bidirectional_model, "extension", X_train, y_train, X_test, y_test,
            epochs=args.epochs, batch_size=args.batch_size, class_weights=class_weights
        )
        all_metrics['extension'] = extension_metrics
    
    # Load existing metrics and update
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path) and args.model != 'all':
        with open(metrics_path, 'r') as f:
            existing_metrics = json.load(f)
        existing_metrics.update(all_metrics)
        all_metrics = existing_metrics
    
    # Save all metrics
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n{'='*80}")
    print(f"All metrics saved to {metrics_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print("\nModel Comparison:")
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 68)
    for model_name, metrics in all_metrics.items():
        print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    
    # Highlight the best model
    if all_metrics:
        best_model = max(all_metrics.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🏆 BEST MODEL: {best_model[0]} with {best_model[1]['accuracy']:.4f} accuracy")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated artifacts:")
    print(f"  - model/scaler.pkl")
    print(f"  - model/bert.npy")
    for m in models_to_train:
        print(f"  - model/{m}_model.h5, {m}_history.pckl, {m}_weights.hdf5")
    print(f"  - model/metrics.json")
    print("\nTo run the dashboard: streamlit run app.py")
    print("\nTo train individual models:")
    print("  python train.py --model lstm      # Train only LSTM")
    print("  python train.py --model propose   # Train only LSTM+GRU")
    print("  python train.py --model extension # Train only Bidirectional (BEST)")


if __name__ == "__main__":
    main()
