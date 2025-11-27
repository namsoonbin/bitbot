"""
Risk Management Guardrails

Defines risk limits and validation rules for trading decisions.

Conservative approach for cryptocurrency trading:
- Maximum position size: 10% of portfolio
- Maximum stop-loss: 20% from entry
- No leverage (spot trading only)
- Daily loss limit: 5% of portfolio

Future: Can integrate with Guardrails AI for LLM output validation
"""

from typing import Dict, List, Optional, Literal
from dataclasses import dataclass


# ============================================================================
# Risk Limits (Conservative)
# ============================================================================

@dataclass(frozen=True)
class RiskLimits:
    """Risk management limits"""

    # Position Sizing
    MAX_POSITION_SIZE: float = 0.10  # 10% max of portfolio
    MIN_POSITION_SIZE: float = 0.01  # 1% min (avoid dust trades)

    # Stop-Loss / Take-Profit
    MAX_STOP_LOSS_PCT: float = 20.0  # Maximum 20% stop-loss
    MIN_STOP_LOSS_PCT: float = 0.5   # Minimum 0.5% (avoid too tight)
    MAX_TAKE_PROFIT_PCT: float = 50.0  # Maximum 50% take-profit
    MIN_TAKE_PROFIT_PCT: float = 1.0   # Minimum 1% (worthwhile)

    # Confidence Thresholds
    MIN_CONFIDENCE_TO_TRADE: float = 0.55  # Minimum 55% confidence
    HIGH_CONFIDENCE_THRESHOLD: float = 0.75  # Consider "high confidence"

    # Portfolio Limits
    DAILY_LOSS_LIMIT_PCT: float = 5.0  # Max 5% daily loss
    MAX_LEVERAGE: float = 1.0  # No leverage (spot only)

    # Risk Scoring
    MAX_RISK_SCORE: float = 0.70  # Reject if risk score > 0.70


# Global instance
RISK_LIMITS = RiskLimits()


# ============================================================================
# Validation Functions
# ============================================================================

def validate_position_size(allocation: float) -> tuple[bool, Optional[str]]:
    """
    Validate position size

    Args:
        allocation: Proposed allocation (0.0 to 1.0)

    Returns:
        (is_valid, error_message)
    """
    if allocation < 0:
        return False, "Position size cannot be negative"

    if allocation > RISK_LIMITS.MAX_POSITION_SIZE:
        return False, f"Position size {allocation*100:.1f}% exceeds maximum {RISK_LIMITS.MAX_POSITION_SIZE*100:.1f}%"

    if 0 < allocation < RISK_LIMITS.MIN_POSITION_SIZE:
        return False, f"Position size {allocation*100:.1f}% below minimum {RISK_LIMITS.MIN_POSITION_SIZE*100:.1f}%"

    return True, None


def validate_stop_loss(stop_loss_pct: Optional[float], action: str) -> tuple[bool, Optional[str]]:
    """
    Validate stop-loss percentage

    Args:
        stop_loss_pct: Stop-loss percentage
        action: Trade action

    Returns:
        (is_valid, error_message)
    """
    if action == 'HOLD':
        return True, None  # No stop-loss needed for HOLD

    if stop_loss_pct is None:
        return False, "Stop-loss is required for BUY/SELL"

    if stop_loss_pct < RISK_LIMITS.MIN_STOP_LOSS_PCT:
        return False, f"Stop-loss {stop_loss_pct:.2f}% is too tight (min: {RISK_LIMITS.MIN_STOP_LOSS_PCT}%)"

    if stop_loss_pct > RISK_LIMITS.MAX_STOP_LOSS_PCT:
        return False, f"Stop-loss {stop_loss_pct:.2f}% is too wide (max: {RISK_LIMITS.MAX_STOP_LOSS_PCT}%)"

    return True, None


def validate_take_profit(take_profit_pct: Optional[float], action: str) -> tuple[bool, Optional[str]]:
    """
    Validate take-profit percentage

    Args:
        take_profit_pct: Take-profit percentage
        action: Trade action

    Returns:
        (is_valid, error_message)
    """
    if action == 'HOLD' or take_profit_pct is None:
        return True, None  # Optional for all actions

    if take_profit_pct < RISK_LIMITS.MIN_TAKE_PROFIT_PCT:
        return False, f"Take-profit {take_profit_pct:.2f}% is too small (min: {RISK_LIMITS.MIN_TAKE_PROFIT_PCT}%)"

    if take_profit_pct > RISK_LIMITS.MAX_TAKE_PROFIT_PCT:
        return False, f"Take-profit {take_profit_pct:.2f}% is too large (max: {RISK_LIMITS.MAX_TAKE_PROFIT_PCT}%)"

    return True, None


def validate_confidence(confidence: float) -> tuple[bool, Optional[str]]:
    """
    Validate confidence level

    Args:
        confidence: Confidence score (0.0 to 1.0)

    Returns:
        (is_valid, error_message)
    """
    if not (0.0 <= confidence <= 1.0):
        return False, f"Confidence {confidence:.2f} must be between 0.0 and 1.0"

    if confidence < RISK_LIMITS.MIN_CONFIDENCE_TO_TRADE:
        return False, f"Confidence {confidence:.2f} below minimum {RISK_LIMITS.MIN_CONFIDENCE_TO_TRADE:.2f}"

    return True, None


def validate_risk_score(risk_score: float) -> tuple[bool, Optional[str]]:
    """
    Validate risk score

    Args:
        risk_score: Risk score (0.0 to 1.0, higher = riskier)

    Returns:
        (is_valid, error_message)
    """
    if not (0.0 <= risk_score <= 1.0):
        return False, f"Risk score {risk_score:.2f} must be between 0.0 and 1.0"

    if risk_score > RISK_LIMITS.MAX_RISK_SCORE:
        return False, f"Risk score {risk_score:.2f} exceeds maximum {RISK_LIMITS.MAX_RISK_SCORE:.2f}"

    return True, None


def calculate_position_size(
    consensus_position: float,
    consensus_confidence: float,
    max_allocation: float = None
) -> float:
    """
    Calculate safe position size using Kelly-inspired approach

    Args:
        consensus_position: Position strength (-100 to 100)
        consensus_confidence: Confidence level (0.0 to 1.0)
        max_allocation: Override max allocation (default: RISK_LIMITS.MAX_POSITION_SIZE)

    Returns:
        Allocation (0.0 to max_allocation)
    """
    if max_allocation is None:
        max_allocation = RISK_LIMITS.MAX_POSITION_SIZE

    # Kelly-inspired: allocation = confidence * position_strength / 100
    raw_allocation = (consensus_confidence * abs(consensus_position)) / 100

    # Cap at max allocation
    allocation = min(raw_allocation, max_allocation)

    # Round to 2 decimal places
    return round(allocation, 4)


def calculate_risk_score(
    confidence: float,
    concerns: List[str],
    market_regime: Optional[str] = None
) -> float:
    """
    Calculate risk score (0.0 = low risk, 1.0 = high risk)

    Args:
        confidence: Confidence level
        concerns: List of risk concerns
        market_regime: Market regime (optional)

    Returns:
        Risk score (0.0 to 1.0)
    """
    # Base risk from low confidence
    base_risk = 1.0 - confidence

    # Add risk for each concern (+0.15 per concern)
    concern_risk = len(concerns) * 0.15

    # Add risk for sideways market
    market_risk = 0.10 if market_regime == 'sideways' else 0.0

    # Total risk (capped at 1.0)
    total_risk = min(1.0, base_risk + concern_risk + market_risk)

    return round(total_risk, 3)


# ============================================================================
# Risk Assessment
# ============================================================================

def assess_trade_risk(
    action: str,
    allocation: float,
    confidence: float,
    stop_loss_pct: Optional[float],
    take_profit_pct: Optional[float],
    market_regime: Optional[str] = None
) -> Dict[str, any]:
    """
    Comprehensive risk assessment of a proposed trade

    Args:
        action: Trade action
        allocation: Position size
        confidence: Confidence level
        stop_loss_pct: Stop-loss percentage
        take_profit_pct: Take-profit percentage
        market_regime: Market regime

    Returns:
        {
            'approved': bool,
            'risk_score': float,
            'concerns': List[str],
            'recommendations': List[str],
            'validation_errors': List[str]
        }
    """
    concerns = []
    recommendations = []
    validation_errors = []

    # Validate all parameters
    valid, error = validate_position_size(allocation)
    if not valid:
        validation_errors.append(error)

    valid, error = validate_confidence(confidence)
    if not valid:
        validation_errors.append(error)

    valid, error = validate_stop_loss(stop_loss_pct, action)
    if not valid:
        validation_errors.append(error)

    valid, error = validate_take_profit(take_profit_pct, action)
    if not valid:
        validation_errors.append(error)

    # Risk concerns
    if confidence < 0.60:
        concerns.append(f"Low confidence ({confidence:.2f})")

    if allocation > 0.08:  # 8% threshold for warning
        concerns.append(f"Large position ({allocation*100:.1f}%)")
        recommendations.append("Consider reducing to 5-7%")

    if market_regime == 'sideways' and action != 'HOLD':
        concerns.append("Sideways market - unclear trend")
        recommendations.append("Wait for trend confirmation")

    if action == 'SELL':
        concerns.append("Short position (high risk in bull market)")
        recommendations.append("Prefer HOLD over SELL in uncertain conditions")

    # Calculate risk score
    risk_score = calculate_risk_score(confidence, concerns, market_regime)

    # Approval decision
    approved = (
        len(validation_errors) == 0 and  # No validation errors
        risk_score <= RISK_LIMITS.MAX_RISK_SCORE and  # Risk acceptable
        confidence >= RISK_LIMITS.MIN_CONFIDENCE_TO_TRADE and  # Sufficient confidence
        action != 'SELL'  # Conservative: reject shorts
    )

    return {
        'approved': approved,
        'risk_score': risk_score,
        'concerns': concerns,
        'recommendations': recommendations,
        'validation_errors': validation_errors
    }


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    """Test risk guardrails"""
    print("=" * 80)
    print("Testing Risk Guardrails")
    print("=" * 80)

    # Test 1: Valid trade
    print("\n[TEST 1] Valid BUY trade:")
    result = assess_trade_risk(
        action='BUY',
        allocation=0.05,  # 5%
        confidence=0.75,
        stop_loss_pct=2.0,
        take_profit_pct=6.0
    )
    print(f"  Approved: {result['approved']}")
    print(f"  Risk Score: {result['risk_score']:.3f}")
    print(f"  Concerns: {result['concerns']}")

    # Test 2: Oversized position
    print("\n[TEST 2] Oversized position:")
    result = assess_trade_risk(
        action='BUY',
        allocation=0.15,  # 15% (too large)
        confidence=0.80,
        stop_loss_pct=3.0,
        take_profit_pct=9.0
    )
    print(f"  Approved: {result['approved']}")
    print(f"  Validation Errors: {result['validation_errors']}")

    # Test 3: Low confidence
    print("\n[TEST 3] Low confidence:")
    result = assess_trade_risk(
        action='BUY',
        allocation=0.03,
        confidence=0.45,  # Too low
        stop_loss_pct=1.5,
        take_profit_pct=4.0
    )
    print(f"  Approved: {result['approved']}")
    print(f"  Validation Errors: {result['validation_errors']}")

    print("\n" + "=" * 80)
