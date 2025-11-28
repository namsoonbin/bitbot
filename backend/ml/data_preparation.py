# coding: utf-8
"""
Multi-Timeframe Data Preparation for ML Models

Handles:
- Multi-timeframe data collection (1d, 4h, 15m, 5m)
- Feature engineering and normalization
- Labeling (UP/DOWN/SIDEWAYS)
- Train/Validation/Test split
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
from loguru import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.ccxt_collector import CCXTDataCollector


# ============================================================================
# Configuration
# ============================================================================

TIMEFRAMES = {
    '1d': {'days': 730, 'table': 'ohlcv_btcusdt_1d'},    # 2 years daily
    '4h': {'days': 365, 'table': 'ohlcv_btcusdt_4h'},    # 1 year 4-hour
    '15m': {'days': 90, 'table': 'ohlcv_btcusdt_15m'},   # 3 months 15-min
    '5m': {'days': 30, 'table': 'ohlcv_btcusdt_5m'}      # 1 month 5-min
}

LABEL_THRESHOLDS = {
    '15m': {'up': 0.001, 'down': -0.001},   # 0.1% for 15-minute
    '5m': {'up': 0.0005, 'down': -0.0005},  # 0.05% for 5-minute
}


# ============================================================================
# Multi-Timeframe Data Collector
# ============================================================================

class MultiTimeframeCollector:
    """Collects and manages data from multiple timeframes"""

    def __init__(
        self,
        symbol: str = 'BTC/USDT',
        timeframes: List[str] = None,
        db_config: Dict = None
    ):
        """
        Initialize multi-timeframe collector

        Args:
            symbol: Trading pair
            timeframes: List of timeframes to collect (default: ['1d', '15m'])
            db_config: Database configuration dict
        """
        self.symbol = symbol
        self.timeframes = timeframes or ['1d', '15m']  # Start with Phase 1

        # Initialize collector
        if db_config:
            self.collector = CCXTDataCollector(**db_config)
        else:
            # Use defaults
            self.collector = CCXTDataCollector(connect_db=True)

        logger.info(f"MultiTimeframeCollector initialized for {symbol}")
        logger.info(f"Timeframes: {self.timeframes}")

    def collect_all_timeframes(
        self,
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect data from all configured timeframes

        Args:
            force_refresh: If True, fetch new data from exchange

        Returns:
            Dict mapping timeframe to DataFrame
        """
        data = {}

        for tf in self.timeframes:
            if tf not in TIMEFRAMES:
                logger.warning(f"Unknown timeframe: {tf}, skipping")
                continue

            config = TIMEFRAMES[tf]
            logger.info(f"Collecting {tf} data ({config['days']} days)...")

            try:
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=config['days'])

                # Collect data
                df = self.collector.collect_historical_data(
                    symbol=self.symbol,
                    timeframe=tf,
                    days=config['days'],
                    table_name=config['table']
                )

                # Validate data
                if df is None or len(df) == 0:
                    logger.warning(f"No data collected for {tf}")
                    continue

                logger.success(f"Collected {len(df)} candles for {tf}")
                data[tf] = df

            except Exception as e:
                logger.error(f"Error collecting {tf} data: {e}")
                continue

        logger.success(f"Collected data from {len(data)} timeframes")
        return data

    def save_datasets(
        self,
        data: Dict[str, pd.DataFrame],
        output_dir: str = 'backend/ml/data'
    ):
        """Save collected data to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for tf, df in data.items():
            file_path = output_path / f"{self.symbol.replace('/', '_')}_{tf}.csv"
            df.to_csv(file_path, index=False)
            logger.success(f"Saved {tf} data to {file_path}")

    def load_datasets(
        self,
        input_dir: str = 'backend/ml/data'
    ) -> Dict[str, pd.DataFrame]:
        """Load saved datasets from disk"""
        input_path = Path(input_dir)
        data = {}

        for tf in self.timeframes:
            file_path = input_path / f"{self.symbol.replace('/', '_')}_{tf}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path, parse_dates=['timestamp'])
                data[tf] = df
                logger.success(f"Loaded {tf} data: {len(df)} candles")
            else:
                logger.warning(f"File not found: {file_path}")

        return data


# ============================================================================
# Feature Engineering
# ============================================================================

def add_technical_features(df: pd.DataFrame, timeframe: str = '15m') -> pd.DataFrame:
    """
    Add technical indicators as features

    Args:
        df: OHLCV DataFrame
        timeframe: Timeframe for parameter adjustment

    Returns:
        DataFrame with additional feature columns
    """
    df = df.copy()

    # Price-based features
    df['price_change'] = df['close'].pct_change()
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['body_size'] = abs(df['close'] - df['open']) / df['close']
    df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close']
    df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close']

    # Volume features
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Moving averages (20/50/100/200 for multi-timeframe analysis)
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_100'] = df['close'].rolling(window=100).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()

    # EMA
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    # Volume Profile (VMP) - Rolling window approach
    # Calculate VP features over last N candles
    vp_window = 50  # 50 periods for VP calculation

    df['vp_poc'] = np.nan  # Point of Control (price with max volume)
    df['vp_vah'] = np.nan  # Value Area High
    df['vp_val'] = np.nan  # Value Area Low
    df['vp_position'] = np.nan  # Current price position in VP (0-1)
    df['vp_poc_dist'] = np.nan  # Distance from POC (%)

    # Calculate VP for each row using rolling window
    for i in range(vp_window, len(df)):
        window_data = df.iloc[i-vp_window:i]

        # Create price bins (10 bins within the price range)
        price_min = window_data['low'].min()
        price_max = window_data['high'].max()
        n_bins = 10
        bins = np.linspace(price_min, price_max, n_bins + 1)

        # Aggregate volume for each price level
        volume_profile = np.zeros(n_bins)
        for _, row in window_data.iterrows():
            # Distribute volume across bins that the candle touched
            candle_low_bin = np.digitize(row['low'], bins) - 1
            candle_high_bin = np.digitize(row['high'], bins) - 1

            # Clip to valid range
            candle_low_bin = max(0, min(candle_low_bin, n_bins - 1))
            candle_high_bin = max(0, min(candle_high_bin, n_bins - 1))

            # Distribute volume evenly across touched bins
            n_touched = candle_high_bin - candle_low_bin + 1
            if n_touched > 0:
                volume_per_bin = row['volume'] / n_touched
                for bin_idx in range(candle_low_bin, candle_high_bin + 1):
                    volume_profile[bin_idx] += volume_per_bin

        # Point of Control (POC) - bin with max volume
        poc_bin = np.argmax(volume_profile)
        poc_price = (bins[poc_bin] + bins[poc_bin + 1]) / 2

        # Value Area (70% of total volume)
        total_volume = volume_profile.sum()
        target_volume = total_volume * 0.70

        # Find Value Area by expanding from POC
        va_bins = [poc_bin]
        va_volume = volume_profile[poc_bin]

        left = poc_bin - 1
        right = poc_bin + 1

        while va_volume < target_volume and (left >= 0 or right < n_bins):
            left_vol = volume_profile[left] if left >= 0 else 0
            right_vol = volume_profile[right] if right < n_bins else 0

            if left_vol >= right_vol and left >= 0:
                va_bins.append(left)
                va_volume += left_vol
                left -= 1
            elif right < n_bins:
                va_bins.append(right)
                va_volume += right_vol
                right += 1
            else:
                break

        # Value Area High/Low
        va_bins_sorted = sorted(va_bins)
        vah_bin = va_bins_sorted[-1]
        val_bin = va_bins_sorted[0]

        vah_price = bins[vah_bin + 1]  # Upper bound of highest bin
        val_price = bins[val_bin]      # Lower bound of lowest bin

        # Current price position in VP (0 = VAL, 0.5 = POC, 1 = VAH)
        current_price = df.iloc[i]['close']
        if vah_price > val_price:
            vp_position = (current_price - val_price) / (vah_price - val_price)
            vp_position = max(0, min(1, vp_position))  # Clip to [0, 1]
        else:
            vp_position = 0.5

        # Distance from POC (percentage)
        poc_dist = ((current_price - poc_price) / poc_price) * 100

        # Store results
        df.at[df.index[i], 'vp_poc'] = poc_price
        df.at[df.index[i], 'vp_vah'] = vah_price
        df.at[df.index[i], 'vp_val'] = val_price
        df.at[df.index[i], 'vp_position'] = vp_position
        df.at[df.index[i], 'vp_poc_dist'] = poc_dist

    # Drop NaN rows (from rolling calculations)
    df = df.dropna()

    logger.info(f"Added {len(df.columns) - 6} technical features")  # -6 for OHLCV columns
    return df


# ============================================================================
# Labeling
# ============================================================================

def create_labels(
    df: pd.DataFrame,
    timeframe: str = '15m',
    lookahead: int = 1
) -> pd.DataFrame:
    """
    Create classification labels (UP/DOWN/SIDEWAYS)

    Args:
        df: DataFrame with OHLCV data
        timeframe: Timeframe for threshold selection
        lookahead: Number of periods to look ahead

    Returns:
        DataFrame with 'label' column added
    """
    df = df.copy()

    # Get thresholds
    thresholds = LABEL_THRESHOLDS.get(timeframe, {'up': 0.001, 'down': -0.001})

    # Calculate future return
    df['future_return'] = df['close'].pct_change(periods=lookahead).shift(-lookahead)

    # Create labels
    df['label'] = 'SIDEWAYS'  # Default
    df.loc[df['future_return'] > thresholds['up'], 'label'] = 'UP'
    df.loc[df['future_return'] < thresholds['down'], 'label'] = 'DOWN'

    # Drop last row (no future data)
    df = df[:-lookahead]

    # Log distribution
    label_counts = df['label'].value_counts()
    logger.info(f"Label distribution:\n{label_counts}")
    logger.info(f"UP: {label_counts.get('UP', 0)/len(df)*100:.1f}%")
    logger.info(f"DOWN: {label_counts.get('DOWN', 0)/len(df)*100:.1f}%")
    logger.info(f"SIDEWAYS: {label_counts.get('SIDEWAYS', 0)/len(df)*100:.1f}%")

    return df


# ============================================================================
# Normalization
# ============================================================================

def normalize_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: str = 'standard'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """
    Normalize features using StandardScaler or MinMaxScaler

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        method: 'standard' or 'minmax'

    Returns:
        Normalized train, val, test DataFrames and fitted scaler
    """
    # Select feature columns (exclude timestamp, label, future_return)
    exclude_cols = ['timestamp', 'label', 'future_return']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    # Initialize scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Fit on training data only
    scaler.fit(train_df[feature_cols])

    # Transform all datasets
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    logger.success(f"Normalized {len(feature_cols)} features using {method} scaler")

    return train_df, val_df, test_df, scaler


# ============================================================================
# Train/Val/Test Split
# ============================================================================

def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset chronologically (no shuffle for time series)

    Args:
        df: Full dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        train_df, val_df, test_df
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.success(f"Split dataset: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    logger.info(f"Train period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    logger.info(f"Val period: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
    logger.info(f"Test period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

    return train_df, val_df, test_df


# ============================================================================
# Full Pipeline
# ============================================================================

def prepare_ml_dataset(
    timeframe: str = '15m',
    symbol: str = 'BTC/USDT',
    force_refresh: bool = False,
    save_path: str = 'backend/ml/data/prepared'
) -> Dict[str, pd.DataFrame]:
    """
    Full pipeline: Collect -> Engineer -> Label -> Split -> Normalize

    Args:
        timeframe: Target timeframe for ML model
        symbol: Trading pair
        force_refresh: Re-collect data from exchange
        save_path: Where to save prepared datasets

    Returns:
        Dict with 'train', 'val', 'test' DataFrames
    """
    logger.info("="*60)
    logger.info(f"ML Data Preparation Pipeline - {timeframe}")
    logger.info("="*60)

    # Step 1: Collect data
    collector = MultiTimeframeCollector(symbol=symbol, timeframes=[timeframe])

    if force_refresh:
        logger.info("Step 1: Collecting data from exchange...")
        data = collector.collect_all_timeframes()
        collector.save_datasets(data)
    else:
        logger.info("Step 1: Loading cached data...")
        data = collector.load_datasets()
        if not data:
            logger.warning("No cached data found, collecting from exchange...")
            data = collector.collect_all_timeframes()
            collector.save_datasets(data)

    df = data[timeframe]

    # Step 2: Feature engineering
    logger.info("Step 2: Engineering features...")
    df = add_technical_features(df, timeframe=timeframe)

    # Step 3: Labeling
    logger.info("Step 3: Creating labels...")
    df = create_labels(df, timeframe=timeframe, lookahead=1)

    # Step 4: Split
    logger.info("Step 4: Splitting dataset...")
    train_df, val_df, test_df = split_dataset(df)

    # Step 5: Normalize
    logger.info("Step 5: Normalizing features...")
    train_df, val_df, test_df, scaler = normalize_features(train_df, val_df, test_df)

    # Save prepared datasets
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(save_dir / f'train_{timeframe}.csv', index=False)
    val_df.to_csv(save_dir / f'val_{timeframe}.csv', index=False)
    test_df.to_csv(save_dir / f'test_{timeframe}.csv', index=False)

    # Save scaler
    with open(save_dir / f'scaler_{timeframe}.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    logger.success("="*60)
    logger.success(f"ML dataset prepared and saved to {save_dir}")
    logger.success("="*60)

    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'scaler': scaler
    }


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare ML dataset')
    parser.add_argument('--timeframe', default='15m', help='Timeframe (15m, 5m, etc.)')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading pair')
    parser.add_argument('--refresh', action='store_true', help='Force refresh data')

    args = parser.parse_args()

    # Run pipeline
    datasets = prepare_ml_dataset(
        timeframe=args.timeframe,
        symbol=args.symbol,
        force_refresh=args.refresh
    )

    print("\nDataset shapes:")
    print(f"  Train: {datasets['train'].shape}")
    print(f"  Val: {datasets['val'].shape}")
    print(f"  Test: {datasets['test'].shape}")
