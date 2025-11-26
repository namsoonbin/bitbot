"""
AgentState Definition for HATS Trading System
Defines the state structure that flows through the LangGraph
"""

from typing import TypedDict, List, Dict, Optional, Literal
from datetime import datetime


class TechnicalIndicators(TypedDict):
    """Technical analysis indicators"""
    rsi: Optional[float]
    macd: Optional[Dict[str, float]]  # {'macd': value, 'signal': value, 'histogram': value}
    bollinger_bands: Optional[Dict[str, float]]  # {'upper': value, 'middle': value, 'lower': value}
    ema_20: Optional[float]
    ema_50: Optional[float]
    volume_sma: Optional[float]
    support_level: Optional[float]
    resistance_level: Optional[float]


class MarketData(TypedDict):
    """Current market data"""
    timestamp: datetime
    symbol: str
    current_price: float
    open: float
    high: float
    low: float
    volume: float
    price_change_24h: float
    price_change_pct_24h: float


class NewsItem(TypedDict):
    """News article with sentiment"""
    published_at: datetime
    title: str
    body: str
    source: str
    url: str
    sentiment_score: float  # -1.0 to 1.0
    votes_positive: int
    votes_negative: int


class ReasoningStep(TypedDict):
    """A single step in the chain-of-thought reasoning"""
    step_number: int
    role: Literal['Analyst', 'Researcher_Bull', 'Researcher_Bear', 'Risk_Manager', 'Trader']
    content: str
    timestamp: datetime
    confidence: Optional[float]


class DebateMessage(TypedDict):
    """A message in the Bull vs Bear debate"""
    role: Literal['Bull', 'Bear']
    content: str
    timestamp: datetime
    supporting_evidence: List[str]


class ProposedTrade(TypedDict):
    """Trade proposal from the agent"""
    action: Literal['BUY', 'SELL', 'HOLD']
    allocation: float  # 0.0 to 1.0 (percentage of portfolio)
    confidence: float  # 0.0 to 1.0
    stop_loss_pct: Optional[float]
    take_profit_pct: Optional[float]
    reasoning: str
    proposed_at: datetime


class RiskAssessment(TypedDict):
    """Risk manager's assessment"""
    approved: bool
    risk_score: float  # 0.0 to 1.0 (higher = riskier)
    concerns: List[str]
    recommendations: List[str]
    max_position_size: float
    suggested_stop_loss: Optional[float]
    feedback: str


class PortfolioState(TypedDict):
    """Current portfolio state"""
    cash_balance: float
    btc_balance: float
    total_value_usd: float
    unrealized_pnl: float
    realized_pnl: float
    current_position: Optional[Literal['LONG', 'SHORT', 'NEUTRAL']]
    entry_price: Optional[float]
    position_size: Optional[float]


class AgentState(TypedDict):
    """
    Complete state that flows through the LangGraph

    This state is passed between nodes and accumulates information
    as it progresses through the agent workflow.
    """

    # Workflow control
    current_node: str
    iteration: int
    should_continue: bool
    error: Optional[str]

    # Market data
    market_data: MarketData
    technical_indicators: TechnicalIndicators
    recent_news: List[NewsItem]
    historical_prices: List[Dict[str, float]]  # List of OHLCV dicts

    # Analysis results
    fundamental_analysis: Optional[str]
    technical_analysis: Optional[str]
    sentiment_analysis: Optional[str]
    sentiment_score: Optional[float]  # Aggregated sentiment from news

    # Reasoning process
    reasoning_trace: List[ReasoningStep]
    debate_transcript: List[DebateMessage]

    # Research outputs
    bull_case: Optional[str]
    bull_confidence: Optional[float]
    bear_case: Optional[str]
    bear_confidence: Optional[float]

    # Trade decision
    proposed_trade: Optional[ProposedTrade]
    risk_assessment: Optional[RiskAssessment]
    final_decision: Optional[Literal['BUY', 'SELL', 'HOLD']]

    # Portfolio context
    portfolio: PortfolioState

    # Metadata
    session_id: str
    thread_id: str
    started_at: datetime
    completed_at: Optional[datetime]

    # LLM tracking
    api_calls_count: int
    tokens_used: int
    cache_hits: int
    cache_misses: int


class MinimalAgentState(TypedDict):
    """
    Minimal state for quick iterations (used during development/testing)
    """
    current_node: str
    iteration: int
    market_data: MarketData
    reasoning_trace: List[ReasoningStep]
    final_decision: Optional[Literal['BUY', 'SELL', 'HOLD']]
    session_id: str


# Initial state factory
def create_initial_state(session_id: str, market_data: MarketData) -> AgentState:
    """Create a new initial AgentState"""
    now = datetime.now()

    return AgentState(
        # Workflow control
        current_node='start',
        iteration=0,
        should_continue=True,
        error=None,

        # Market data
        market_data=market_data,
        technical_indicators=TechnicalIndicators(
            rsi=None,
            macd=None,
            bollinger_bands=None,
            ema_20=None,
            ema_50=None,
            volume_sma=None,
            support_level=None,
            resistance_level=None
        ),
        recent_news=[],
        historical_prices=[],

        # Analysis results
        fundamental_analysis=None,
        technical_analysis=None,
        sentiment_analysis=None,
        sentiment_score=None,

        # Reasoning process
        reasoning_trace=[],
        debate_transcript=[],

        # Research outputs
        bull_case=None,
        bull_confidence=None,
        bear_case=None,
        bear_confidence=None,

        # Trade decision
        proposed_trade=None,
        risk_assessment=None,
        final_decision=None,

        # Portfolio context
        portfolio=PortfolioState(
            cash_balance=10000.0,  # Default starting capital
            btc_balance=0.0,
            total_value_usd=10000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            current_position='NEUTRAL',
            entry_price=None,
            position_size=None
        ),

        # Metadata
        session_id=session_id,
        thread_id=f"thread_{session_id}",
        started_at=now,
        completed_at=None,

        # LLM tracking
        api_calls_count=0,
        tokens_used=0,
        cache_hits=0,
        cache_misses=0
    )


def add_reasoning_step(
    state: AgentState,
    role: Literal['Analyst', 'Researcher_Bull', 'Researcher_Bear', 'Risk_Manager', 'Trader'],
    content: str,
    confidence: Optional[float] = None
) -> AgentState:
    """Add a reasoning step to the state"""
    step = ReasoningStep(
        step_number=len(state['reasoning_trace']) + 1,
        role=role,
        content=content,
        timestamp=datetime.now(),
        confidence=confidence
    )

    state['reasoning_trace'].append(step)
    return state


def add_debate_message(
    state: AgentState,
    role: Literal['Bull', 'Bear'],
    content: str,
    evidence: List[str]
) -> AgentState:
    """Add a debate message to the state"""
    message = DebateMessage(
        role=role,
        content=content,
        timestamp=datetime.now(),
        supporting_evidence=evidence
    )

    state['debate_transcript'].append(message)
    return state
