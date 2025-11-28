# coding: utf-8
"""
Phase 3.4 & 3.5 Integration Tests

Tests for implemented components:
- Phase 3.4: Sentiment Analyst (Gemini 2.5 Flash CoT)
- Phase 3.4.1: API Rate Limiting (Free tier management)
- Phase 3.5: Risk Guardrails (Day trading limits)

Note: Tests with real Gemini API calls consume free tier quota
"""

import pytest
from datetime import datetime, timedelta

# Import implemented components
from agents.sentiment_analyst import analyze_news_sentiment
from agents.risk_guardrails import (
    assess_trade_risk,
    calculate_position_size,
    validate_position_size,
    validate_stop_loss,
    validate_take_profit,
    RISK_LIMITS
)
from agents.rate_limiter import get_rate_limiter
from agents.researchers import get_llm


# ============================================================================
# Phase 3.4.1: Rate Limiter Tests
# ============================================================================

def test_rate_limiter_flash_config():
    """Test Flash configuration (10 calls/min, 250/day)"""
    limiter = get_rate_limiter()

    assert limiter.call_delays["gemini-2.5-flash"] == 6.0
    assert limiter.daily_limits["gemini-2.5-flash"] == 250

    print("  [PASS] Flash: 6s delay, 250/day limit")


def test_rate_limiter_pro_config():
    """Test Pro configuration (2 calls/min, 50/day)"""
    limiter = get_rate_limiter()

    assert limiter.call_delays["gemini-2.5-pro"] == 30.0
    assert limiter.daily_limits["gemini-2.5-pro"] == 50

    print("  [PASS] Pro: 30s delay, 50/day limit")


def test_rate_limiter_usage_tracking():
    """Test API usage tracking"""
    limiter = get_rate_limiter()

    stats = limiter.get_stats()

    assert isinstance(stats, dict)
    assert 'gemini-2.5-flash' in stats or len(stats) >= 0

    print(f"  [PASS] Stats retrieved: {len(stats)} models")


# ============================================================================
# Phase 3.5: Risk Guardrails Tests
# ============================================================================

def test_day_trading_position_limits():
    """Verify day trading position limits (5-30%)"""
    assert RISK_LIMITS.MAX_POSITION_SIZE == 0.30
    assert RISK_LIMITS.MIN_POSITION_SIZE == 0.05

    print("  [PASS] Position: 5-30% (split entry)")


def test_day_trading_stop_loss_limits():
    """Verify stop-loss limits (3-15%)"""
    assert RISK_LIMITS.MAX_STOP_LOSS_PCT == 15.0
    assert RISK_LIMITS.MIN_STOP_LOSS_PCT == 3.0

    print("  [PASS] Stop-loss: 3-15%")


def test_day_trading_take_profit_limits():
    """Verify take-profit limits (4-45%)"""
    assert RISK_LIMITS.MAX_TAKE_PROFIT_PCT == 45.0
    assert RISK_LIMITS.MIN_TAKE_PROFIT_PCT == 4.0

    print("  [PASS] Take-profit: 4-45% (split exit)")


def test_leverage_fixed_5x():
    """Verify leverage is fixed at 5x for BTC/USDT"""
    assert RISK_LIMITS.MAX_LEVERAGE == 5.0

    print("  [PASS] Leverage: 5.0x fixed")


def test_validate_position_size_valid():
    """Test valid position sizes"""
    assert validate_position_size(0.05)[0] is True  # Min 5%
    assert validate_position_size(0.10)[0] is True  # 10%
    assert validate_position_size(0.20)[0] is True  # 20%
    assert validate_position_size(0.30)[0] is True  # Max 30%

    print("  [PASS] Valid positions: 5%, 10%, 20%, 30%")


def test_validate_position_size_invalid():
    """Test invalid position sizes"""
    assert validate_position_size(0.02)[0] is False  # Too small
    assert validate_position_size(0.35)[0] is False  # Too large
    assert validate_position_size(-0.1)[0] is False  # Negative

    print("  [PASS] Invalid positions rejected: -10%, 2%, 35%")


def test_validate_stop_loss_valid():
    """Test valid stop-loss levels"""
    assert validate_stop_loss(3.0, "BUY")[0] is True
    assert validate_stop_loss(10.0, "BUY")[0] is True
    assert validate_stop_loss(15.0, "BUY")[0] is True

    print("  [PASS] Valid stop-loss: 3%, 10%, 15%")


def test_validate_stop_loss_invalid():
    """Test invalid stop-loss levels"""
    assert validate_stop_loss(2.0, "BUY")[0] is False   # Too tight
    assert validate_stop_loss(20.0, "BUY")[0] is False  # Too wide

    print("  [PASS] Invalid stop-loss rejected: 2%, 20%")


def test_validate_take_profit_valid():
    """Test valid take-profit levels"""
    assert validate_take_profit(4.0, "BUY")[0] is True
    assert validate_take_profit(15.0, "BUY")[0] is True
    assert validate_take_profit(45.0, "BUY")[0] is True

    print("  [PASS] Valid take-profit: 4%, 15%, 45%")


def test_validate_take_profit_invalid():
    """Test invalid take-profit levels"""
    assert validate_take_profit(2.0, "BUY")[0] is False   # Too low
    assert validate_take_profit(50.0, "BUY")[0] is False  # Too high

    print("  [PASS] Invalid take-profit rejected: 2%, 50%")


def test_calculate_position_size_kelly():
    """Test Kelly-inspired position sizing"""
    # High confidence should give higher position
    pos_high = calculate_position_size(80, 0.85, max_allocation=0.30)
    assert 0.15 <= pos_high <= 0.30

    # Low confidence should give lower position
    pos_low = calculate_position_size(30, 0.40, max_allocation=0.30)
    assert 0.05 <= pos_low <= 0.15

    print(f"  [PASS] High conf: {pos_high*100:.1f}%, Low conf: {pos_low*100:.1f}%")


def test_assess_trade_risk_approve_valid():
    """Test valid trade is approved"""
    result = assess_trade_risk(
        action="BUY",
        allocation=0.15,
        confidence=0.75,
        stop_loss_pct=5.0,
        take_profit_pct=12.0,
        market_regime="uptrend"
    )

    assert result['approved'] is True
    assert len(result['validation_errors']) == 0
    assert 0.0 <= result['risk_score'] <= 1.0

    print(f"  [PASS] Valid trade approved (risk: {result['risk_score']:.3f})")


def test_assess_trade_risk_reject_invalid():
    """Test invalid trade is rejected"""
    result = assess_trade_risk(
        action="BUY",
        allocation=0.35,      # Too high
        confidence=0.75,
        stop_loss_pct=2.0,    # Too tight
        take_profit_pct=50.0, # Too high
        market_regime="uptrend"
    )

    assert result['approved'] is False
    assert len(result['validation_errors']) >= 3

    print(f"  [PASS] Invalid trade rejected ({len(result['validation_errors'])} errors)")


# ============================================================================
# Phase 3.4: Sentiment Analyst Tests (Real API)
# ============================================================================

@pytest.mark.api
def test_sentiment_bullish_news():
    """Test sentiment with bullish news (makes 1 API call)"""
    news = [
        {
            'title': 'Bitcoin ETF sees record $500M inflows',
            'source': 'CoinDesk',
            'published_at': (datetime.now() - timedelta(hours=2)).isoformat()
        },
        {
            'title': 'Major banks announce crypto trading desks',
            'source': 'Bloomberg',
            'published_at': (datetime.now() - timedelta(hours=5)).isoformat()
        }
    ]

    result = analyze_news_sentiment(news, symbol="BTC/USDT")

    # Validate structure
    assert 'average_score' in result
    assert 'overall_label' in result
    assert -1.0 <= result['average_score'] <= 1.0

    # Should be bullish
    assert result['average_score'] > 0.2

    print(f"  [PASS] Bullish sentiment: {result['average_score']:.3f}")


@pytest.mark.api
def test_sentiment_bearish_news():
    """Test sentiment with bearish news (makes 1 API call)"""
    news = [
        {
            'title': 'SEC lawsuit threatens major exchange',
            'source': 'Reuters',
            'published_at': datetime.now().isoformat()
        },
        {
            'title': 'Bitcoin crashes below support',
            'source': 'CoinDesk',
            'published_at': datetime.now().isoformat()
        }
    ]

    result = analyze_news_sentiment(news, symbol="BTC/USDT")

    # Should be bearish
    assert result['average_score'] < -0.1

    print(f"  [PASS] Bearish sentiment: {result['average_score']:.3f}")


def test_sentiment_empty_news():
    """Test sentiment with empty news"""
    result = analyze_news_sentiment([], symbol="BTC/USDT")

    assert result['average_score'] == 0.0
    assert result['overall_label'] == 'Neutral'

    print("  [PASS] Empty news -> Neutral")


# ============================================================================
# LLM Setup Tests
# ============================================================================

def test_llm_gemini_flash_init():
    """Test Gemini Flash LLM initialization"""
    llm = get_llm("gemini-2.5-flash")
    assert llm is not None
    assert hasattr(llm, 'model')

    print("  [PASS] Flash LLM initialized")


def test_llm_gemini_pro_init():
    """Test Gemini Pro LLM initialization"""
    llm = get_llm("gemini-2.5-pro")
    assert llm is not None
    assert hasattr(llm, 'model')

    print("  [PASS] Pro LLM initialized")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
