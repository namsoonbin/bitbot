# coding: utf-8
"""
LSTM Training Pipeline

Trains the LSTM price prediction model using prepared data:
- 3-class classification: UP/DOWN/SIDEWAYS
- Adam optimizer with CrossEntropyLoss
- Early stopping (patience=10)
- Model checkpointing (best validation accuracy)
- Training/Validation monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from loguru import logger
import sys

# Import LSTM components
from price_predictor import (
    LSTMConfig,
    LSTMPricePredictor,
    TimeSeriesDataset,
    count_parameters,
    save_model,
    load_model
)


# ============================================================================
# Configuration
# ============================================================================

class TrainingConfig:
    """Training configuration"""

    # Paths
    data_dir: Path = Path(__file__).parent / 'data' / 'prepared'
    model_dir: Path = Path(__file__).parent / 'models'

    # Training
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5  # L2 regularization

    # Early stopping
    early_stopping_patience: int = 10
    min_delta: float = 0.001  # Minimum improvement

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Logging
    log_interval: int = 10  # Log every N batches


# ============================================================================
# Training Functions
# ============================================================================

def load_datasets(
    timeframe: str = '15m',
    config: LSTMConfig = None
) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
    """
    Load train/val/test datasets

    Args:
        timeframe: Timeframe to use ('15m', '1d', etc.)
        config: LSTM configuration

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    if config is None:
        config = LSTMConfig()

    data_dir = Path(__file__).parent / 'data' / 'prepared'

    # Load CSVs
    train_df = pd.read_csv(data_dir / f'train_{timeframe}.csv')
    val_df = pd.read_csv(data_dir / f'val_{timeframe}.csv')
    test_df = pd.read_csv(data_dir / f'test_{timeframe}.csv')

    logger.info(f"Loaded datasets:")
    logger.info(f"  Train: {len(train_df)} rows")
    logger.info(f"  Val: {len(val_df)} rows")
    logger.info(f"  Test: {len(test_df)} rows")

    # Create datasets
    train_dataset = TimeSeriesDataset(train_df, sequence_length=config.sequence_length)
    val_dataset = TimeSeriesDataset(val_df, sequence_length=config.sequence_length)
    test_dataset = TimeSeriesDataset(test_df, sequence_length=config.sequence_length)

    return train_dataset, val_dataset, test_dataset


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    log_interval: int = 10
) -> Dict[str, float]:
    """
    Train for one epoch

    Args:
        model: LSTM model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        log_interval: Logging frequency

    Returns:
        Dict with epoch metrics
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (sequences, labels) in enumerate(dataloader):
        # Move to device
        sequences = sequences.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Logging
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            logger.info(
                f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                f"Loss={avg_loss:.4f}, Acc={accuracy:.2f}%"
            )

    # Epoch metrics
    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = 100.0 * correct / total

    return {
        'loss': epoch_loss,
        'accuracy': epoch_accuracy
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]:
    """
    Validate model

    Args:
        model: LSTM model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Dict with validation metrics
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # Per-class metrics
    class_correct = [0, 0, 0]  # UP, SIDEWAYS, DOWN
    class_total = [0, 0, 0]

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1

    # Overall metrics
    val_loss = total_loss / len(dataloader)
    val_accuracy = 100.0 * correct / total

    # Per-class accuracy
    class_labels = ['UP', 'SIDEWAYS', 'DOWN']
    class_accuracies = {}
    for i, label in enumerate(class_labels):
        if class_total[i] > 0:
            class_accuracies[label] = 100.0 * class_correct[i] / class_total[i]
        else:
            class_accuracies[label] = 0.0

    return {
        'loss': val_loss,
        'accuracy': val_accuracy,
        'class_accuracies': class_accuracies
    }


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_accuracy: float) -> bool:
        """
        Check if training should stop

        Args:
            val_accuracy: Current validation accuracy

        Returns:
            True if training should stop
        """
        score = val_accuracy

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


# ============================================================================
# Main Training Loop
# ============================================================================

def train_model(
    timeframe: str = '15m',
    lstm_config: LSTMConfig = None,
    train_config: TrainingConfig = None,
    resume_from: str = None
) -> Dict:
    """
    Train LSTM model

    Args:
        timeframe: Timeframe to train on
        lstm_config: LSTM model configuration
        train_config: Training configuration
        resume_from: Path to checkpoint to resume from

    Returns:
        Dict with training history
    """
    # Initialize configs
    if lstm_config is None:
        lstm_config = LSTMConfig()
    if train_config is None:
        train_config = TrainingConfig()

    # Setup
    device = train_config.device
    logger.info(f"Training on device: {device}")

    # Create model directory
    train_config.model_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    logger.info(f"Loading {timeframe} datasets...")
    train_dataset, val_dataset, test_dataset = load_datasets(timeframe, lstm_config)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0  # Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create model
    logger.info("Initializing model...")
    model = LSTMPricePredictor(lstm_config).to(device)
    logger.info(f"Total parameters: {count_parameters(model):,}")

    # Show class distribution (no weighting applied)
    train_labels = train_dataset.labels
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)

    logger.info(f"Class distribution: UP={class_counts[0]}, SIDEWAYS={class_counts[1]}, DOWN={class_counts[2]}")
    logger.info(f"Class percentages: UP={100*class_counts[0]/total_samples:.1f}%, SIDEWAYS={100*class_counts[1]/total_samples:.1f}%, DOWN={100*class_counts[2]/total_samples:.1f}%")

    # Loss and optimizer (no class weighting - let more data handle imbalance naturally)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from and Path(resume_from).exists():
        logger.info(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # Early stopping
    early_stopping = EarlyStopping(
        patience=train_config.early_stopping_patience,
        min_delta=train_config.min_delta
    )

    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    best_val_accuracy = 0.0
    best_model_path = train_config.model_dir / f'best_model_{timeframe}.pt'

    # Training loop
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training for {train_config.num_epochs} epochs")
    logger.info(f"{'='*60}\n")

    for epoch in range(start_epoch, train_config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{train_config.num_epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device, train_config.log_interval
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Log epoch results
        logger.info(
            f"\n  Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.2f}%"
        )
        logger.info(
            f"  Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.2f}%"
        )
        logger.info(f"  Class Accuracies:")
        for class_name, acc in val_metrics['class_accuracies'].items():
            logger.info(f"    {class_name}: {acc:.2f}%")

        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])

        # Save best model
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            save_model(
                model,
                lstm_config,
                str(best_model_path),
                metadata={
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'class_accuracies': val_metrics['class_accuracies']
                }
            )
            logger.success(f"  New best model saved! Val Acc: {best_val_accuracy:.2f}%")

        # Early stopping check
        if early_stopping(val_metrics['accuracy']):
            logger.warning(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

        logger.info("")

    # Final evaluation on test set
    logger.info(f"\n{'='*60}")
    logger.info("Evaluating on test set...")
    logger.info(f"{'='*60}\n")

    # Load best model
    best_model, _ = load_model(str(best_model_path), device=device)
    test_metrics = validate(best_model, test_loader, criterion, device)

    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    logger.info(f"Class Accuracies:")
    for class_name, acc in test_metrics['class_accuracies'].items():
        logger.info(f"  {class_name}: {acc:.2f}%")

    # Save final results
    history['test_loss'] = test_metrics['loss']
    history['test_accuracy'] = test_metrics['accuracy']
    history['test_class_accuracies'] = test_metrics['class_accuracies']

    logger.success(f"\nTraining complete!")
    logger.success(f"Best model saved to: {best_model_path}")

    return history


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Train model
    history = train_model(timeframe='15m')

    # Print summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Best Validation Accuracy: {max(history['val_accuracy']):.2f}%")
    print(f"Test Accuracy: {history['test_accuracy']:.2f}%")
    print(f"Test Class Accuracies:")
    for class_name, acc in history['test_class_accuracies'].items():
        print(f"  {class_name}: {acc:.2f}%")
    print("="*60)
