"""
Technical Analyst Module
Calculates technical indicators using the 'ta' library

Provides:
- Trend indicators (EMA, SMA, MACD)
- Momentum indicators (RSI, Stochastic)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators
- Support/Resistance detection
- Pattern recognition
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

# Technical Analysis Library
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


# ============================================================================
# Core Technical Indicators
# ============================================================================

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate Relative Strength Index (RSI)

    RSI > 70: Overbought
    RSI < 30: Oversold

    Args:
        df: DataFrame with 'close' column
        period: RSI period (default: 14)

    Returns:
        Current RSI value (0-100)
    """
    if len(df) < period:
        return 50.0  # Neutral default

    rsi = RSIIndicator(close=df['close'], window=period)
    rsi_values = rsi.rsi()

    return float(rsi_values.iloc[-1]) if not pd.isna(rsi_values.iloc[-1]) else 50.0


def calculate_macd(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate MACD (Moving Average Convergence Divergence)

    Returns:
        {
            'macd': MACD line,
            'signal': Signal line,
            'histogram': MACD - Signal,
            'trend': 'bullish' | 'bearish' | 'neutral'
        }
    """
    if len(df) < 26:
        return {
            'macd': 0.0,
            'signal': 0.0,
            'histogram': 0.0,
            'trend': 'neutral'
        }

    macd_indicator = MACD(
        close=df['close'],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )

    macd_line = macd_indicator.macd()
    signal_line = macd_indicator.macd_signal()
    histogram = macd_indicator.macd_diff()

    current_macd = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0
    current_signal = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0
    current_hist = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0

    # Determine trend
    if current_hist > 0 and current_macd > current_signal:
        trend = 'bullish'
    elif current_hist < 0 and current_macd < current_signal:
        trend = 'bearish'
    else:
        trend = 'neutral'

    return {
        'macd': round(current_macd, 2),
        'signal': round(current_signal, 2),
        'histogram': round(current_hist, 2),
        'trend': trend
    }


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
    """
    Calculate Bollinger Bands

    Returns:
        {
            'upper': Upper band,
            'middle': Middle band (SMA),
            'lower': Lower band,
            'position': 'above' | 'middle' | 'below',
            'width': Band width percentage
        }
    """
    if len(df) < period:
        current_price = float(df['close'].iloc[-1])
        return {
            'upper': current_price * 1.02,
            'middle': current_price,
            'lower': current_price * 0.98,
            'position': 'middle',
            'width': 4.0
        }

    bb = BollingerBands(
        close=df['close'],
        window=period,
        window_dev=std_dev
    )

    upper = bb.bollinger_hband()
    middle = bb.bollinger_mavg()
    lower = bb.bollinger_lband()

    current_price = float(df['close'].iloc[-1])
    current_upper = float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else current_price * 1.02
    current_middle = float(middle.iloc[-1]) if not pd.isna(middle.iloc[-1]) else current_price
    current_lower = float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else current_price * 0.98

    # Determine position
    if current_price > current_upper:
        position = 'above'
    elif current_price < current_lower:
        position = 'below'
    else:
        position = 'middle'

    # Calculate band width as percentage
    width = ((current_upper - current_lower) / current_middle) * 100 if current_middle > 0 else 4.0

    return {
        'upper': round(current_upper, 2),
        'middle': round(current_middle, 2),
        'lower': round(current_lower, 2),
        'position': position,
        'width': round(width, 2)
    }


def calculate_ema(df: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> Dict[str, float]:
    """
    Calculate Exponential Moving Averages

    Args:
        df: DataFrame with 'close' column
        periods: List of EMA periods

    Returns:
        {'ema_20': value, 'ema_50': value, 'ema_200': value}
    """
    result = {}
    current_price = float(df['close'].iloc[-1])

    for period in periods:
        if len(df) < period:
            result[f'ema_{period}'] = current_price
        else:
            ema = EMAIndicator(close=df['close'], window=period)
            ema_values = ema.ema_indicator()
            result[f'ema_{period}'] = float(ema_values.iloc[-1]) if not pd.isna(ema_values.iloc[-1]) else current_price

    return result


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate Average True Range (ATR)
    Used for volatility measurement and stop-loss calculation

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period

    Returns:
        Current ATR value
    """
    if len(df) < period or 'high' not in df.columns or 'low' not in df.columns:
        return 0.0

    atr = AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=period
    )

    atr_values = atr.average_true_range()
    return float(atr_values.iloc[-1]) if not pd.isna(atr_values.iloc[-1]) else 0.0


# ============================================================================
# Volume Indicators
# ============================================================================

def calculate_volume_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate volume-based indicators

    Returns:
        {
            'obv': On-Balance Volume,
            'vwap': Volume Weighted Average Price (if available),
            'volume_trend': 'increasing' | 'decreasing' | 'stable'
        }
    """
    if 'volume' not in df.columns or len(df) < 10:
        return {
            'obv': 0.0,
            'vwap': float(df['close'].iloc[-1]) if len(df) > 0 else 0.0,
            'volume_trend': 'stable'
        }

    # On-Balance Volume
    obv_indicator = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
    obv = obv_indicator.on_balance_volume()
    current_obv = float(obv.iloc[-1]) if not pd.isna(obv.iloc[-1]) else 0.0

    # VWAP (if high/low available)
    if 'high' in df.columns and 'low' in df.columns:
        try:
            vwap_indicator = VolumeWeightedAveragePrice(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume']
            )
            vwap = vwap_indicator.volume_weighted_average_price()
            current_vwap = float(vwap.iloc[-1]) if not pd.isna(vwap.iloc[-1]) else float(df['close'].iloc[-1])
        except:
            current_vwap = float(df['close'].iloc[-1])
    else:
        current_vwap = float(df['close'].iloc[-1])

    # Volume trend (last 5 candles)
    if len(df) >= 5:
        recent_volume = df['volume'].iloc[-5:].mean()
        previous_volume = df['volume'].iloc[-10:-5].mean() if len(df) >= 10 else recent_volume

        if recent_volume > previous_volume * 1.2:
            volume_trend = 'increasing'
        elif recent_volume < previous_volume * 0.8:
            volume_trend = 'decreasing'
        else:
            volume_trend = 'stable'
    else:
        volume_trend = 'stable'

    return {
        'obv': round(current_obv, 2),
        'vwap': round(current_vwap, 2),
        'volume_trend': volume_trend
    }


# ============================================================================
# Support & Resistance Detection
# ============================================================================

def detect_support_resistance(df: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
    """
    Detect support and resistance levels using local minima/maxima

    Args:
        df: DataFrame with 'high' and 'low' columns
        window: Window size for detecting local extrema

    Returns:
        {
            'support_levels': [price1, price2, ...],
            'resistance_levels': [price1, price2, ...]
        }
    """
    if len(df) < window * 2 or 'high' not in df.columns or 'low' not in df.columns:
        current_price = float(df['close'].iloc[-1]) if len(df) > 0 else 0.0
        return {
            'support_levels': [current_price * 0.95],
            'resistance_levels': [current_price * 1.05]
        }

    # Find local minima (support)
    lows = df['low'].values
    support_levels = []

    for i in range(window, len(lows) - window):
        if lows[i] == min(lows[i-window:i+window]):
            support_levels.append(float(lows[i]))

    # Find local maxima (resistance)
    highs = df['high'].values
    resistance_levels = []

    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i-window:i+window]):
            resistance_levels.append(float(highs[i]))

    # Keep only recent levels (last 3 of each)
    support_levels = sorted(support_levels, reverse=True)[:3]
    resistance_levels = sorted(resistance_levels, reverse=True)[:3]

    # Fallback if no levels found
    current_price = float(df['close'].iloc[-1])
    if not support_levels:
        support_levels = [current_price * 0.95]
    if not resistance_levels:
        resistance_levels = [current_price * 1.05]

    return {
        'support_levels': [round(x, 2) for x in support_levels],
        'resistance_levels': [round(x, 2) for x in resistance_levels]
    }


def calculate_pivot_points(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate Pivot Points (standard method)

    Returns:
        {
            'pivot': Pivot point,
            'r1': Resistance 1,
            'r2': Resistance 2,
            's1': Support 1,
            's2': Support 2
        }
    """
    if len(df) < 1 or 'high' not in df.columns or 'low' not in df.columns:
        current_price = float(df['close'].iloc[-1]) if len(df) > 0 else 0.0
        return {
            'pivot': current_price,
            'r1': current_price * 1.01,
            'r2': current_price * 1.02,
            's1': current_price * 0.99,
            's2': current_price * 0.98
        }

    # Use last candle's data
    high = float(df['high'].iloc[-1])
    low = float(df['low'].iloc[-1])
    close = float(df['close'].iloc[-1])

    # Standard pivot calculation
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)

    return {
        'pivot': round(pivot, 2),
        'r1': round(r1, 2),
        'r2': round(r2, 2),
        's1': round(s1, 2),
        's2': round(s2, 2)
    }


# ============================================================================
# Pattern Recognition
# ============================================================================

def detect_trend(df: pd.DataFrame, ema_periods: List[int] = [20, 50]) -> str:
    """
    Detect overall trend using EMAs

    Returns:
        'uptrend' | 'downtrend' | 'sideways'
    """
    if len(df) < max(ema_periods):
        return 'sideways'

    emas = calculate_ema(df, ema_periods)
    current_price = float(df['close'].iloc[-1])

    ema_20 = emas.get('ema_20', current_price)
    ema_50 = emas.get('ema_50', current_price)

    # Uptrend: Price > EMA20 > EMA50
    if current_price > ema_20 > ema_50:
        return 'uptrend'
    # Downtrend: Price < EMA20 < EMA50
    elif current_price < ema_20 < ema_50:
        return 'downtrend'
    else:
        return 'sideways'


def detect_momentum(df: pd.DataFrame) -> str:
    """
    Detect momentum using RSI

    Returns:
        'overbought' | 'oversold' | 'neutral'
    """
    rsi = calculate_rsi(df)

    if rsi > 70:
        return 'overbought'
    elif rsi < 30:
        return 'oversold'
    else:
        return 'neutral'


# ============================================================================
# Main Function: Calculate All Indicators
# ============================================================================

def calculate_technical_indicators(ohlcv_data: List[Dict]) -> Dict[str, Any]:
    """
    Calculate all technical indicators from OHLCV data

    Args:
        ohlcv_data: List of OHLCV candles from database
                    [{'timestamp': ..., 'open': ..., 'high': ..., 'low': ..., 'close': ..., 'volume': ...}, ...]

    Returns:
        Dictionary with all technical indicators:
        {
            'rsi': float,
            'macd': {...},
            'bollinger_bands': {...},
            'ema_20': float,
            'ema_50': float,
            'ema_200': float,
            'atr': float,
            'volume': {...},
            'support_resistance': {...},
            'pivot_points': {...},
            'trend': str,
            'momentum': str
        }
    """
    if not ohlcv_data or len(ohlcv_data) == 0:
        return {
            'rsi': 50.0,
            'macd': {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0, 'trend': 'neutral'},
            'bollinger_bands': {'upper': 0.0, 'middle': 0.0, 'lower': 0.0, 'position': 'middle', 'width': 0.0},
            'ema_20': 0.0,
            'ema_50': 0.0,
            'ema_200': 0.0,
            'atr': 0.0,
            'volume': {'obv': 0.0, 'vwap': 0.0, 'volume_trend': 'stable'},
            'support_resistance': {'support_levels': [], 'resistance_levels': []},
            'pivot_points': {'pivot': 0.0, 'r1': 0.0, 'r2': 0.0, 's1': 0.0, 's2': 0.0},
            'trend': 'sideways',
            'momentum': 'neutral'
        }

    # Convert to DataFrame
    df = pd.DataFrame(ohlcv_data)

    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    # Calculate all indicators
    rsi = calculate_rsi(df)
    macd = calculate_macd(df)
    bb = calculate_bollinger_bands(df)
    emas = calculate_ema(df, [20, 50, 200])
    atr = calculate_atr(df)
    volume = calculate_volume_indicators(df)
    sr = detect_support_resistance(df)
    pivots = calculate_pivot_points(df)
    trend = detect_trend(df)
    momentum = detect_momentum(df)

    return {
        'rsi': rsi,
        'macd': macd,
        'bollinger_bands': bb,
        'ema_20': emas.get('ema_20', 0.0),
        'ema_50': emas.get('ema_50', 0.0),
        'ema_200': emas.get('ema_200', 0.0),
        'atr': atr,
        'volume': volume,
        'support_resistance': sr,
        'pivot_points': pivots,
        'trend': trend,
        'momentum': momentum
    }
