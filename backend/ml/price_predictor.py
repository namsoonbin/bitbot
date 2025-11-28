# coding: utf-8
"""
LSTM Price Predictor for Multi-Timeframe Trading

PyTorch-based LSTM model for 3-class price prediction:
- UP: Price increase > threshold
- DOWN: Price decrease > threshold
- SIDEWAYS: Price change within threshold

Architecture:
- Input: 60 timesteps x N features
- Hidden: 128 units x 2 LSTM layers
- Output: 3 classes (Softmax)
- Dropout: 0.2 (overfitting prevention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path
from loguru import logger


# ============================================================================
# Configuration
# ============================================================================

class LSTMConfig:
    """LSTM model configuration"""

    # Architecture
    input_size: int = 34  # Number of features (29 technical + 5 OHLCV without timestamp)
    hidden_size: int = 128  # LSTM hidden units
    num_layers: int = 2  # Number of LSTM layers
    num_classes: int = 3  # UP/DOWN/SIDEWAYS
    dropout: float = 0.2  # Dropout rate

    # Sequence
    sequence_length: int = 60  # Timesteps to look back

    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 50
    early_stopping_patience: int = 10

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Dataset Class
# ============================================================================

class TimeSeriesDataset(Dataset):
    """Dataset for time series LSTM training"""

    def __init__(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        feature_cols: List[str] = None
    ):
        """
        Initialize dataset

        Args:
            df: DataFrame with features and labels
            sequence_length: Number of timesteps
            feature_cols: List of feature column names
        """
        self.sequence_length = sequence_length

        # Exclude non-feature columns
        exclude_cols = ['timestamp', 'label', 'future_return']
        if feature_cols is None:
            self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        else:
            self.feature_cols = feature_cols

        # Extract features and labels
        self.features = df[self.feature_cols].values.astype(np.float32)

        # Convert labels to integers
        label_map = {'UP': 0, 'SIDEWAYS': 1, 'DOWN': 2}
        self.labels = df['label'].map(label_map).values.astype(np.int64)

        # Calculate valid indices (we need sequence_length historical data)
        self.valid_indices = list(range(sequence_length, len(df)))

        logger.info(f"TimeSeriesDataset initialized:")
        logger.info(f"  Features: {len(self.feature_cols)} columns")
        logger.info(f"  Samples: {len(self.valid_indices)}")
        logger.info(f"  Sequence length: {sequence_length}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sequence and label"""
        actual_idx = self.valid_indices[idx]

        # Get sequence [sequence_length, num_features]
        start_idx = actual_idx - self.sequence_length
        end_idx = actual_idx

        sequence = self.features[start_idx:end_idx]
        label = self.labels[actual_idx]

        return torch.tensor(sequence), torch.tensor(label)


# ============================================================================
# LSTM Model
# ============================================================================

class LSTMPricePredictor(nn.Module):
    """LSTM-based price prediction model"""

    def __init__(self, config: LSTMConfig):
        """
        Initialize LSTM model

        Args:
            config: Model configuration
        """
        super(LSTMPricePredictor, self).__init__()

        self.config = config

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True  # Input shape: [batch, seq, features]
        )

        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)

        # Fully connected layer
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

        logger.info(f"LSTMPricePredictor initialized:")
        logger.info(f"  Input size: {config.input_size}")
        logger.info(f"  Hidden size: {config.hidden_size}")
        logger.info(f"  Num layers: {config.num_layers}")
        logger.info(f"  Num classes: {config.num_classes}")
        logger.info(f"  Dropout: {config.dropout}")
        logger.info(f"  Device: {config.device}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, sequence_length, input_size]

        Returns:
            Output tensor [batch_size, num_classes]
        """
        # LSTM forward
        # lstm_out: [batch, seq, hidden_size]
        # h_n: [num_layers, batch, hidden_size]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the last timestep output
        # last_output: [batch, hidden_size]
        last_output = lstm_out[:, -1, :]

        # Dropout
        out = self.dropout(last_output)

        # Fully connected
        # out: [batch, num_classes]
        out = self.fc(out)

        return out

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities

        Args:
            x: Input tensor [batch_size, sequence_length, input_size]

        Returns:
            Probability tensor [batch_size, num_classes]
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels

        Args:
            x: Input tensor [batch_size, sequence_length, input_size]

        Returns:
            Class labels [batch_size]
        """
        probs = self.predict_proba(x)
        predictions = torch.argmax(probs, dim=1)
        return predictions


# ============================================================================
# Model Utilities
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(
    model: nn.Module,
    config: LSTMConfig,
    save_path: str,
    metadata: Dict = None
):
    """
    Save model checkpoint

    Args:
        model: PyTorch model
        config: Model configuration
        save_path: Path to save checkpoint
        metadata: Additional metadata (e.g., training stats)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'input_size': config.input_size,
            'hidden_size': config.hidden_size,
            'num_layers': config.num_layers,
            'num_classes': config.num_classes,
            'dropout': config.dropout,
            'sequence_length': config.sequence_length
        },
        'metadata': metadata or {}
    }

    torch.save(checkpoint, save_path)
    logger.success(f"Model saved to {save_path}")


def load_model(
    load_path: str,
    device: str = 'cpu'
) -> Tuple[nn.Module, LSTMConfig]:
    """
    Load model checkpoint

    Args:
        load_path: Path to checkpoint
        device: Device to load model on

    Returns:
        Loaded model and config
    """
    checkpoint = torch.load(load_path, map_location=device)

    # Reconstruct config
    config = LSTMConfig()
    for key, value in checkpoint['config'].items():
        setattr(config, key, value)
    config.device = device

    # Reconstruct model
    model = LSTMPricePredictor(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.success(f"Model loaded from {load_path}")
    logger.info(f"  Metadata: {checkpoint.get('metadata', {})}")

    return model, config


# ============================================================================
# Prediction Function
# ============================================================================

def predict_next_movement(
    model: nn.Module,
    recent_data: pd.DataFrame,
    config: LSTMConfig,
    scaler = None
) -> Dict[str, any]:
    """
    Predict next price movement

    Args:
        model: Trained LSTM model
        recent_data: Recent OHLCV data (at least sequence_length rows)
        config: Model configuration
        scaler: Feature scaler (if features need normalization)

    Returns:
        Dict with prediction, probabilities, and confidence
    """
    model.eval()

    # Ensure we have enough data
    if len(recent_data) < config.sequence_length:
        raise ValueError(f"Need at least {config.sequence_length} rows, got {len(recent_data)}")

    # Extract features
    exclude_cols = ['timestamp', 'label', 'future_return']
    feature_cols = [col for col in recent_data.columns if col not in exclude_cols]
    features = recent_data[feature_cols].values.astype(np.float32)

    # Take last sequence_length rows
    sequence = features[-config.sequence_length:]

    # Convert to tensor and add batch dimension
    sequence_tensor = torch.tensor(sequence).unsqueeze(0)  # [1, seq_len, features]
    sequence_tensor = sequence_tensor.to(config.device)

    # Predict
    with torch.no_grad():
        probabilities = model.predict_proba(sequence_tensor)
        prediction = model.predict(sequence_tensor)

    # Convert to numpy
    probs = probabilities.cpu().numpy()[0]
    pred_class = prediction.cpu().item()

    # Map class to label
    class_labels = ['UP', 'SIDEWAYS', 'DOWN']
    predicted_label = class_labels[pred_class]
    confidence = probs[pred_class]

    result = {
        'prediction': predicted_label,
        'confidence': float(confidence),
        'probabilities': {
            'UP': float(probs[0]),
            'SIDEWAYS': float(probs[1]),
            'DOWN': float(probs[2])
        },
        'raw_class': int(pred_class)
    }

    return result


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == '__main__':
    # Test model creation
    config = LSTMConfig()
    model = LSTMPricePredictor(config)

    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, config.sequence_length, config.input_size)

    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    probs = model.predict_proba(dummy_input)
    predictions = model.predict(dummy_input)

    print(f"\nProbabilities shape: {probs.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample probabilities: {probs[0]}")
    print(f"Sample prediction: {predictions[0].item()}")
