"""
CNN-RNN Model Architectures for Predictive Maintenance

This module contains the two core neural network architectures used 
in the predictive maintenance system.

Models:
    - build_cnn_rnn_model: Basic CNN-RNN architecture
    - build_weighted_model: Recall-optimized version for safety-critical applications

Author: Luca Zagaia
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, SimpleRNN, Dense, Input
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.optimizers import Adam
from typing import Tuple


def build_cnn_rnn_model(
    input_shape: Tuple[int, int],
    filters: int = 64,
    kernel_size: int = 3,
    rnn_units: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Build basic CNN-RNN model for predictive maintenance.
    
    Architecture: Conv1D -> MaxPooling -> Dropout -> SimpleRNN -> Dense(sigmoid)
    
    Args:
        input_shape: Shape of input data (timesteps, features)
        filters: Number of convolutional filters (default: 64)
        kernel_size: Size of convolutional kernel (default: 3)
        rnn_units: Number of RNN units (default: 32)
        dropout_rate: Dropout rate for regularization (default: 0.2)
        learning_rate: Learning rate for optimizer (default: 0.001)
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        SimpleRNN(units=rnn_units),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_weighted_model(
    input_shape: Tuple[int, int],
    filters: int = 64,
    kernel_size: int = 3,
    rnn_units: int = 32,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Build recall-optimized CNN-RNN model for safety-critical applications.
    
    Same architecture as basic model but with additional metrics tracking
    and higher dropout for regularization. Designed to be used with
    class weights to boost recall performance.
    
    Architecture: Conv1D -> MaxPooling -> Dropout -> SimpleRNN -> Dense(sigmoid)
    
    Args:
        input_shape: Shape of input data (timesteps, features)
        filters: Number of convolutional filters (default: 64)
        kernel_size: Size of convolutional kernel (default: 3)
        rnn_units: Number of RNN units (default: 32)
        dropout_rate: Dropout rate for regularization (default: 0.3)
        learning_rate: Learning rate for optimizer (default: 0.001)
        
    Returns:
        Compiled Keras model with recall and precision tracking
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        SimpleRNN(units=rnn_units),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            Recall(name='recall'),
            Precision(name='precision')
        ]
    )
    
    return model