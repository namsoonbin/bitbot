"""
LangGraph Node Implementations
Each node represents a specialized agent in the trading workflow
"""

from .state import AgentState, add_reasoning_step, add_debate_message
from datetime import datetime, timedelta
from loguru import logger
import os
from typing import Dict

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

        # Calculate sentiment
        if news_data:
            sentiment_scores = [item.get('sentiment', {}).get('score', 0) for item in news_data]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
            state['sentiment_score'] = avg_sentiment
            logger.info(f"Average news sentiment: {avg_sentiment:.3f}")

        # Prepare context for LLM
        current_price = state['market_data'].get('current_price') or 0.0
        price_change = state['market_data'].get('price_change_pct_24h') or 0.0

        # Build news summary
        news_summary = "\n".join([
            f"- [{item.get('source', 'Unknown')}] {item.get('title', 'No title')} (Sentiment: {item.get('sentiment', {}).get('score', 0):.2f})"
            for item in news_data[:5]  # Top 5 news
        ])

        # Build price summary
        if ohlcv_data:
            latest = ohlcv_data[0]
            oldest = ohlcv_data[-1]
            week_change_pct = ((latest['close'] - oldest['close']) / oldest['close']) * 100
            price_summary = f"7-day change: {week_change_pct:+.2f}%"
        else:
            price_summary = "No historical data available"

        # LLM analysis (OpenAI 사용)
        llm = get_llm(model="gpt-4o-mini", temperature=0.7)

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

        analysis_result = None
        try:
            chain = analysis_prompt | llm | StrOutputParser()
            analysis_result = chain.invoke({
                'symbol': state['market_data']['symbol'],
                'current_price': current_price,
                'price_change': price_change,
                'price_summary': price_summary,
                'sentiment': state.get('sentiment_score', 0.0) or 0.0,
                'news_summary': news_summary if news_summary else "No recent news available"
            })
            state['api_calls_count'] += 1
        except Exception as llm_err:
            # Graceful degrade when LLM 호출 실패 (네트워크/쿼터/무결제 등)
            logger.error(f"LLM call failed, using fallback analysis: {llm_err}")
            analysis_result = "LLM unavailable; using fallback analysis placeholder."
            state['error'] = str(llm_err)
            state['should_continue'] = False

        # Update state
        state['fundamental_analysis'] = analysis_result
        state['technical_analysis'] = f"Price Analysis: {price_summary}"

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
        role='Bull',
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
        role='Bear',
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

    Responsibilities:
    - Assess risk/reward ratio
    - Check position sizing
    - Validate stop-loss levels
    - Approve or reject trade
    """
    logger.info("=" * 60)
    logger.info("RISK MANAGER - Evaluating trade safety")
    logger.info("=" * 60)

    state['current_node'] = 'risk_manager'

    # Placeholder: Simple approval logic
    from .state import RiskAssessment, ProposedTrade

    # Create a proposed trade (this would normally come from previous node)
    if not state.get('proposed_trade'):
        # Determine action based on bull/bear confidence
        bull_conf = state.get('bull_confidence', 0)
        bear_conf = state.get('bear_confidence', 0)

        if bull_conf > bear_conf + 0.2:
            action = 'BUY'
        elif bear_conf > bull_conf + 0.2:
            action = 'SELL'
        else:
            action = 'HOLD'

        state['proposed_trade'] = ProposedTrade(
            action=action,
            allocation=0.1,  # 10% of portfolio
            confidence=max(bull_conf, bear_conf),
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            reasoning=f"Based on analysis: Bull={bull_conf:.2f}, Bear={bear_conf:.2f}",
            proposed_at=datetime.now()
        )

    # Risk assessment (placeholder)
    proposed_trade = state['proposed_trade']
    approved = proposed_trade['action'] != 'SELL'  # Simple rule: approve BUY/HOLD, reject SELL

    state['risk_assessment'] = RiskAssessment(
        approved=approved,
        risk_score=0.3,
        concerns=[] if approved else ["High volatility"],
        recommendations=["Set stop-loss at 2%"],
        max_position_size=0.15,
        suggested_stop_loss=2.0,
        feedback="Trade approved with risk controls" if approved else "Trade rejected due to high risk"
    )

    add_reasoning_step(
        state,
        role='Risk_Manager',
        content=state['risk_assessment']['feedback'],
        confidence=0.8
    )

    logger.info(f"Risk assessment: {'APPROVED' if approved else 'REJECTED'}")
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
