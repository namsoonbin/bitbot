"""
LangGraph Node Implementations
Each node represents a specialized agent in the trading workflow
"""

from .state import AgentState, add_reasoning_step, add_debate_message
from datetime import datetime, timedelta
from loguru import logger
import os
from typing import Dict
from contextlib import contextmanager

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.caches import BaseCache  # noqa: F401 - forward-ref fix
from langchain_core.callbacks import Callbacks  # noqa: F401 - forward-ref fix

# Ensure pydantic models are fully built (avoids forward-ref errors)
ChatOpenAI.model_rebuild()
ChatAnthropic.model_rebuild()

# Database imports
import psycopg2
from pymongo import MongoClient

# Technical Analysis
from .technical_analyst import calculate_technical_indicators

# Sentiment Analysis
from .sentiment_analyst import analyze_news_sentiment

# Risk Management
from .risk_guardrails import (
    assess_trade_risk,
    calculate_position_size,
    RISK_LIMITS
)


# ============================================
# Helpers
# ============================================

@contextmanager
def _postgres_conn():
    conn = None
    try:
        conn = get_postgres_connection()
        yield conn
    finally:
        if conn:
            conn.close()


@contextmanager
def _mongo_db():
    client = None
    try:
        client = MongoClient(
            f"mongodb://{os.getenv('MONGO_USER', 'hats_user')}:{os.getenv('MONGO_PASSWORD', 'hats_password')}@"
            f"{os.getenv('MONGO_HOST', 'localhost')}:{os.getenv('MONGO_PORT', '27017')}/"
        )
        yield client[os.getenv('MONGO_DB', 'hats_trading')]
    finally:
        if client:
            client.close()


def _build_news_summary(news_data: list) -> str:
    return "\n".join([
        f"- [{item.get('source', 'Unknown')}] {item.get('title', 'No title')} "
        f"(Sentiment: {item.get('sentiment', {}).get('score', 0):.2f})"
        for item in news_data[:5]
    ])


def _build_price_summary(ohlcv_data: list) -> str:
    if not ohlcv_data:
        return "No historical data available"
    latest = ohlcv_data[0]
    oldest = ohlcv_data[-1]
    try:
        week_change_pct = ((latest['close'] - oldest['close']) / oldest['close']) * 100
        return f"7-day change: {week_change_pct:+.2f}%"
    except Exception:
        return "Insufficient historical data"


def _run_analysis_llm(llm, state: AgentState, price_summary: str, news_summary: str):
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a cryptocurrency market analyst.
        Analyze the provided market data and news to form an initial assessment.
        Be concise but insightful. Focus on key signals and trends."""),
        ("user", """
Current Market Data:
- Symbol: {symbol}
- Current Price: ${current_price:,.2f}
- 24h Change: {price_change:+.2f}%
- {price_summary}

Recent News (Avg Sentiment: {sentiment:.2f}):
{news_summary}

Provide:
1. Fundamental Analysis: What do the news and market sentiment suggest?
2. Technical Analysis: What do the price trends indicate?
3. Key Concerns: Any major risks or uncertainties?

Keep analysis under 300 words.
        """)
    ])

    chain = analysis_prompt | llm | StrOutputParser()
    return chain.invoke({
        'symbol': state['market_data']['symbol'],
        'current_price': state['market_data'].get('current_price') or 0.0,
        'price_change': state['market_data'].get('price_change_pct_24h') or 0.0,
        'price_summary': price_summary,
        'sentiment': state.get('sentiment_score', 0.0) or 0.0,
        'news_summary': news_summary if news_summary else "No recent news available"
    })


# ============================================
# Database Utilities
# ============================================

def get_postgres_connection():
    """Get PostgreSQL connection"""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        dbname=os.getenv('POSTGRES_DB', 'hats_trading'),
        user=os.getenv('POSTGRES_USER', 'hats_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'hats_password')
    )


def get_mongo_connection():
    """Get MongoDB connection"""
    mongo_user = os.getenv('MONGO_USER', 'hats_user')
    mongo_password = os.getenv('MONGO_PASSWORD', 'hats_password')
    mongo_host = os.getenv('MONGO_HOST', 'localhost')
    mongo_port = os.getenv('MONGO_PORT', '27017')

    client = MongoClient(f'mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/')
    return client[os.getenv('MONGO_DB', 'hats_trading')]


def fetch_recent_ohlcv(symbol: str = 'BTC/USDT', hours: int = 24) -> list:
    """Fetch recent OHLCV data from PostgreSQL"""
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()

        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        # Fetch data
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_btcusdt_1h
            WHERE timestamp BETWEEN %s AND %s
            ORDER BY timestamp DESC
            LIMIT %s
        """

        cursor.execute(query, (start_time, end_time, hours))
        rows = cursor.fetchall()

        # Convert to list of dicts
        ohlcv_data = [
            {
                'timestamp': row[0],
                'open': float(row[1]),
                'high': float(row[2]),
                'low': float(row[3]),
                'close': float(row[4]),
                'volume': float(row[5])
            }
            for row in rows
        ]

        cursor.close()
        conn.close()

        logger.info(f"Fetched {len(ohlcv_data)} OHLCV candles from PostgreSQL")
        return ohlcv_data

    except Exception as e:
        logger.error(f"Error fetching OHLCV data: {e}")
        return []


def fetch_recent_news(hours: int = 24, limit: int = 10) -> list:
    """Fetch recent news from MongoDB"""
    try:
        db = get_mongo_connection()

        # Calculate time range
        start_time = datetime.now() - timedelta(hours=hours)

        # Fetch news
        news_items = db.news.find(
            {'published_at': {'$gte': start_time}},
            {'_id': 0}  # Exclude MongoDB ID
        ).sort('published_at', -1).limit(limit)

        news_list = list(news_items)
        logger.info(f"Fetched {len(news_list)} news items from MongoDB")

        return news_list

    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return []


# ============================================
# LLM Utilities
# ============================================

def get_llm(model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.7):
    """
    Get LLM instance

    Args:
        model: Model name ("claude-3-5-sonnet-20241022", "gpt-4o-mini", "gpt-4")
        temperature: Temperature for generation

    Returns:
        LLM instance
    """
    openai_key = os.getenv("OPENAI_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    # OpenAI 키가 없으면 Claude로 자동 전환
    if model.startswith("gpt") and not openai_key and anthropic_key:
        logger.warning("OPENAI_API_KEY가 없어 Claude 모델로 자동 전환합니다.")
        model = "claude-3-5-sonnet-20241022"

    if model.startswith("gpt"):
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=openai_key or None
        )
    elif model.startswith("claude"):
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=anthropic_key or None
        )
    else:
        raise ValueError(f"Unsupported model: {model}")


# ============================================
# Node Implementations
# ============================================

def analyst_node(state: AgentState) -> AgentState:
    """
    Fundamental & Technical Analyst Node

    Responsibilities:
    - Fetch recent market data from PostgreSQL
    - Fetch recent news from MongoDB
    - Perform fundamental analysis (news sentiment)
    - Perform technical analysis (price patterns, indicators)
    - Add analysis to reasoning trace
    """
    logger.info("=" * 60)
    logger.info("ANALYST NODE - Starting analysis")
    logger.info("=" * 60)

    state['current_node'] = 'analyst'
    state['iteration'] += 1

    try:
        # Fetch data
        logger.info("Fetching market data...")
        ohlcv_data = fetch_recent_ohlcv(hours=168)  # 1 week
        news_data = fetch_recent_news(hours=72, limit=20)  # 3 days, max 20 articles

        state['historical_prices'] = ohlcv_data
        state['recent_news'] = news_data

        # Calculate technical indicators
        logger.info("Calculating technical indicators...")
        technical_indicators = calculate_technical_indicators(ohlcv_data)
        state['technical_indicators'] = technical_indicators
        logger.info(f"Technical Indicators: RSI={technical_indicators['rsi']:.1f}, "
                   f"Trend={technical_indicators['trend']}, "
                   f"Momentum={technical_indicators['momentum']}")

        # Analyze sentiment using Gemini 2.5 Pro
        logger.info("Analyzing news sentiment with Gemini 2.5 Pro...")
        sentiment_result = analyze_news_sentiment(
            news_data,
            symbol=state['market_data'].get('symbol', 'BTC/USDT')
        )
        state['sentiment_score'] = sentiment_result['average_score']
        state['sentiment_analysis'] = sentiment_result['summary']
        state['news_sentiment'] = sentiment_result
        logger.info(f"Sentiment: {sentiment_result['overall_label']} "
                   f"(Score: {sentiment_result['average_score']:.3f})")

        # Build summaries
        news_summary = _build_news_summary(news_data)
        price_summary = _build_price_summary(ohlcv_data)

        # LLM analysis (OpenAI 사용)
        llm = get_llm(model="gpt-4o-mini", temperature=0.7)

        analysis_result = None
        try:
            analysis_result = _run_analysis_llm(llm, state, price_summary, news_summary)
            state['api_calls_count'] += 1
        except Exception as llm_err:
            # Graceful degrade when LLM 호출 실패 (네트워크/쿼터/무결제 등)
            logger.error(f"LLM call failed, using fallback analysis: {llm_err}")
            analysis_result = "LLM unavailable; using fallback analysis placeholder."
            state['error'] = str(llm_err)
            state['should_continue'] = False

        # Build technical analysis summary
        tech_summary = (
            f"RSI: {technical_indicators['rsi']:.1f} ({technical_indicators['momentum']}), "
            f"Trend: {technical_indicators['trend']}, "
            f"MACD: {technical_indicators['macd']['trend']}, "
            f"Bollinger: {technical_indicators['bollinger_bands']['position']}, "
            f"EMA20: ${technical_indicators['ema_20']:.0f}"
        )

        # Update state
        state['fundamental_analysis'] = analysis_result
        state['technical_analysis'] = tech_summary

        # Add to reasoning trace
        add_reasoning_step(
            state,
            role='Analyst',
            content=f"Market Analysis:\n{analysis_result}",
            confidence=0.7
        )

        logger.success("✓ Analysis complete")
        logger.info(f"Reasoning steps: {len(state['reasoning_trace'])}")

    except Exception as e:
        logger.error(f"✗ Error in analyst node: {e}")
        state['error'] = str(e)
        state['should_continue'] = False

    return state


def bull_researcher_node(state: AgentState) -> AgentState:
    """
    Bull Researcher Node - Argues for LONG position

    Responsibilities:
    - Build strongest case for buying/holding
    - Find supporting evidence from analysis
    - Present optimistic scenario
    """
    logger.info("=" * 60)
    logger.info("BULL RESEARCHER - Building case for LONG")
    logger.info("=" * 60)

    state['current_node'] = 'bull_researcher'

    # Placeholder implementation
    bull_case = """
    Bull Case:
    - Positive sentiment trend
    - Price showing support levels
    - News indicates growing adoption
    """

    state['bull_case'] = bull_case
    state['bull_confidence'] = 0.6

    add_debate_message(
        state,
        role='bull',
        content=bull_case,
        evidence=["Sentiment analysis", "Price support"]
    )

    add_reasoning_step(
        state,
        role='Researcher_Bull',
        content=bull_case,
        confidence=0.6
    )

    logger.success("✓ Bull case presented")
    return state


def bear_researcher_node(state: AgentState) -> AgentState:
    """
    Bear Researcher Node - Argues for SHORT/HOLD position

    Responsibilities:
    - Build strongest case against buying
    - Find risks and concerns
    - Present pessimistic scenario
    """
    logger.info("=" * 60)
    logger.info("BEAR RESEARCHER - Building case for SHORT/HOLD")
    logger.info("=" * 60)

    state['current_node'] = 'bear_researcher'

    # Placeholder implementation
    bear_case = """
    Bear Case:
    - Market volatility remains high
    - Regulatory uncertainties
    - Potential resistance at current levels
    """

    state['bear_case'] = bear_case
    state['bear_confidence'] = 0.5

    add_debate_message(
        state,
        role='bear',
        content=bear_case,
        evidence=["Market volatility", "Regulatory risks"]
    )

    add_reasoning_step(
        state,
        role='Researcher_Bear',
        content=bear_case,
        confidence=0.5
    )

    logger.success("✓ Bear case presented")
    return state


def risk_manager_node(state: AgentState) -> AgentState:
    """
    Risk Manager Node - Evaluates trade safety

    Phase 3: Uses consensus output from Bull/Bear debate
    - Validates consensus position and confidence
    - Assesses risk/reward ratio
    - Determines position sizing (Kelly criterion)
    - Approves or rejects trade
    """
    logger.info("=" * 60)
    logger.info("RISK MANAGER - Evaluating trade safety")
    logger.info("=" * 60)

    state['current_node'] = 'risk_manager'

    from .state import RiskAssessment, ProposedTrade

    # Get consensus from debate
    consensus = state.get('debate_consensus')

    if not consensus:
        logger.warning("No debate consensus found, defaulting to HOLD")
        state['proposed_trade'] = ProposedTrade(
            action='HOLD',
            allocation=0.0,
            confidence=0.5,
            stop_loss_pct=None,
            take_profit_pct=None,
            reasoning="No consensus available from debate",
            proposed_at=datetime.now()
        )
        state['risk_assessment'] = RiskAssessment(
            approved=False,
            risk_score=1.0,
            concerns=["No consensus data"],
            recommendations=["Wait for clearer signals"],
            max_position_size=0.0,
            suggested_stop_loss=None,
            feedback="Trade rejected: No debate consensus"
        )
        return state

    # Extract consensus data
    consensus_position = consensus.get('position', 0.0)  # -100 to 100
    consensus_confidence = consensus.get('confidence', 0.5)  # 0.0 to 1.0
    bull_weight = consensus.get('bull_weight', 0.5)
    bear_weight = consensus.get('bear_weight', 0.5)

    logger.info(f"Consensus Position: {consensus_position:.1f}% (Confidence: {consensus_confidence:.2f})")
    logger.info(f"Bull Weight: {bull_weight:.2f}, Bear Weight: {bear_weight:.2f}")

    # Determine action based on consensus position
    if consensus_position > 30:
        action = 'BUY'
    elif consensus_position < -30:
        action = 'SELL'
    else:
        action = 'HOLD'

    # Position sizing using risk guardrails (max 10% now, was 20%)
    allocation = calculate_position_size(
        consensus_position=consensus_position,
        consensus_confidence=consensus_confidence,
        max_allocation=RISK_LIMITS.MAX_POSITION_SIZE  # 10%
    )

    # Stop-loss and take-profit based on confidence
    # Higher confidence → wider stops (more conviction)
    # But stay within RISK_LIMITS (max 20% stop-loss)
    if consensus_confidence > 0.7:
        stop_loss_pct = min(3.0, RISK_LIMITS.MAX_STOP_LOSS_PCT)
        take_profit_pct = 8.0
    elif consensus_confidence > 0.5:
        stop_loss_pct = min(2.0, RISK_LIMITS.MAX_STOP_LOSS_PCT)
        take_profit_pct = 5.0
    else:
        stop_loss_pct = min(1.5, RISK_LIMITS.MAX_STOP_LOSS_PCT)
        take_profit_pct = 3.0

    # Create proposed trade
    state['proposed_trade'] = ProposedTrade(
        action=action,
        allocation=allocation,
        confidence=consensus_confidence,
        stop_loss_pct=stop_loss_pct if action in ['BUY', 'SELL'] else None,
        take_profit_pct=take_profit_pct if action in ['BUY', 'SELL'] else None,
        reasoning=consensus.get('summary', 'Consensus-based decision'),
        proposed_at=datetime.now()
    )

    # Comprehensive risk assessment using risk_guardrails
    market_regime = state.get('market_regime')

    risk_assessment_result = assess_trade_risk(
        action=action,
        allocation=allocation,
        confidence=consensus_confidence,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        market_regime=market_regime
    )

    # Extract results
    approved = risk_assessment_result['approved']
    risk_score = risk_assessment_result['risk_score']
    concerns = risk_assessment_result['concerns']
    recommendations = risk_assessment_result['recommendations']
    validation_errors = risk_assessment_result['validation_errors']

    # Add stop-loss/take-profit to recommendations
    if action != 'HOLD':
        recommendations.append(f"Stop-loss: {stop_loss_pct}%")
        recommendations.append(f"Take-profit: {take_profit_pct}%")

    # Build feedback message
    if validation_errors:
        feedback = f"Trade REJECTED (Validation Errors): {', '.join(validation_errors)}"
    elif not approved:
        feedback = f"Trade REJECTED (Risk too high): {', '.join(concerns)}"
    else:
        feedback = f"Trade APPROVED: {consensus.get('summary', 'Consensus-based decision')}"

    state['risk_assessment'] = RiskAssessment(
        approved=approved,
        risk_score=risk_score,
        concerns=concerns + validation_errors,  # Combine concerns and errors
        recommendations=recommendations,
        max_position_size=RISK_LIMITS.MAX_POSITION_SIZE,
        suggested_stop_loss=stop_loss_pct,
        feedback=feedback
    )

    add_reasoning_step(
        state,
        role='Risk_Manager',
        content=state['risk_assessment']['feedback'],
        confidence=0.8
    )

    logger.info(f"Risk assessment: {'APPROVED' if approved else 'REJECTED'}")
    logger.info(f"Proposed action: {action} ({allocation*100:.1f}% allocation)")
    logger.info(f"Risk score: {risk_score:.2f}")

    return state


def trader_node(state: AgentState) -> AgentState:
    """
    Trader Node - Executes approved trades

    Responsibilities:
    - Execute the trade (in backtest: log decision)
    - Update portfolio state
    - Record trade in database
    - Finalize decision
    """
    logger.info("=" * 60)
    logger.info("TRADER - Executing trade")
    logger.info("=" * 60)

    state['current_node'] = 'trader'

    proposed_trade = state['proposed_trade']
    state['final_decision'] = proposed_trade['action']

    add_reasoning_step(
        state,
        role='Trader',
        content=f"Executing {proposed_trade['action']} with {proposed_trade['allocation']*100}% allocation",
        confidence=proposed_trade['confidence']
    )

    state['completed_at'] = datetime.now()

    logger.success(f"✓ Trade decision: {state['final_decision']}")
    logger.info(f"Total reasoning steps: {len(state['reasoning_trace'])}")
    logger.info(f"Total API calls: {state['api_calls_count']}")

    return state
