"""
Bull/Bear Researcher Implementation
Based on 2024-2025 Multi-Agent Debate research

References:
- TradingAgents Framework (Dec 2024)
- Multi-Agent Debate best practices
- Adaptive calibration for market regimes
"""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os

from .state import AgentState, DebateMessage


# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================

class ResearcherOutput(BaseModel):
    """Bull/Bear Researcher의 구조화된 출력"""

    thesis: str = Field(description="One-sentence summary of position (bullish or bearish)")
    evidence: List[str] = Field(description="3-5 key supporting points with specific data")
    counter_arguments: str = Field(description="Address opposing viewpoint directly")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    confidence_reasoning: str = Field(description="Explanation for confidence score")
    recommended_position: float = Field(
        ge=-100.0,
        le=100.0,
        description="Position recommendation: Bull (0-100%), Bear (-100-0%)"
    )


# ============================================================================
# Market Regime Detection
# ============================================================================

def detect_market_regime(state: AgentState) -> str:
    """
    시장 상황 자동 감지

    Returns:
        "bull_market", "bear_market", or "sideways"
    """
    market_data = state.get('market_data', {})
    technical_indicators = state.get('technical_indicators', {})
    news_sentiment = state.get('news_sentiment', {})

    # 가격 변화
    price_change_7d = market_data.get('price_change_7d', 0)
    price_change_30d = market_data.get('price_change_30d', 0)

    # 기술적 지표
    rsi = technical_indicators.get('rsi', 50)
    current_price = market_data.get('current_price', 0)
    ema_20 = technical_indicators.get('ema_20', current_price)
    ema_50 = technical_indicators.get('ema_50', current_price)

    # 뉴스 감성
    avg_sentiment = news_sentiment.get('average_score', 0)

    # Bull Market 기준
    is_bull = (
        price_change_7d > 5 and
        price_change_30d > 10 and
        current_price > ema_20 > ema_50 and
        rsi > 50 and
        avg_sentiment > 0.2
    )

    # Bear Market 기준
    is_bear = (
        price_change_7d < -5 and
        price_change_30d < -10 and
        current_price < ema_20 < ema_50 and
        rsi < 50 and
        avg_sentiment < -0.2
    )

    if is_bull:
        return "bull_market"
    elif is_bear:
        return "bear_market"
    else:
        return "sideways"


# ============================================================================
# Adaptive Reasoning Styles
# ============================================================================

def get_bull_reasoning_style(market_regime: str) -> Dict[str, Any]:
    """Bull Researcher의 시장 상황별 추론 스타일"""

    if market_regime == "bull_market":
        return {
            "emphasis": "subjective",
            "focus_areas": [
                "성장 잠재력 (growth potential)",
                "모멘텀 지속성 (momentum continuation)",
                "긍정적 시장 심리 (positive sentiment)",
                "기관 유입 신호 (institutional interest)"
            ],
            "confidence_threshold": 0.65,
            "position_sizing": "aggressive",
            "reasoning_instruction": (
                "Emphasize subjective growth potential and positive momentum. "
                "Focus on opportunities and upside catalysts. "
                "Be optimistic but data-backed."
            )
        }

    elif market_regime == "bear_market":
        return {
            "emphasis": "factual",
            "focus_areas": [
                "명확한 반전 신호 (clear reversal signals)",
                "과매도 지표 (oversold indicators)",
                "펀더멘털 개선 (fundamental improvements)",
                "저점 형성 패턴 (bottom formation)"
            ],
            "confidence_threshold": 0.80,
            "position_sizing": "conservative",
            "reasoning_instruction": (
                "Focus on factual reversal signals and concrete data. "
                "Wait for clear evidence of trend change. "
                "Be conservative and require strong proof."
            )
        }

    else:  # sideways
        return {
            "emphasis": "balanced",
            "focus_areas": [
                "돌파 가능성 (breakout potential)",
                "지지선 테스트 (support level tests)",
                "거래량 증가 (volume increase)",
                "범위 상단 접근 (approaching range top)"
            ],
            "confidence_threshold": 0.70,
            "position_sizing": "moderate",
            "reasoning_instruction": (
                "Look for breakout signals and range-bound patterns. "
                "Balance optimism with caution. "
                "Focus on risk/reward at range extremes."
            )
        }


def get_bear_reasoning_style(market_regime: str) -> Dict[str, Any]:
    """Bear Researcher의 시장 상황별 추론 스타일"""

    if market_regime == "bull_market":
        return {
            "emphasis": "factual",
            "focus_areas": [
                "과열 지표 (overheating indicators)",
                "밸류에이션 리스크 (valuation risk)",
                "기술적 과매수 (technical overbought)",
                "유동성 고갈 신호 (liquidity exhaustion)"
            ],
            "confidence_threshold": 0.75,
            "risk_weight": "moderate",
            "reasoning_instruction": (
                "Use factual data to temper excessive optimism. "
                "Focus on overheating signals and valuation concerns. "
                "Be the voice of caution with concrete evidence."
            )
        }

    elif market_regime == "bear_market":
        return {
            "emphasis": "subjective",
            "focus_areas": [
                "하락 모멘텀 (downside momentum)",
                "지지선 붕괴 (support breakdown)",
                "패닉 매도 리스크 (panic selling risk)",
                "시장 심리 악화 (negative sentiment)"
            ],
            "confidence_threshold": 0.60,
            "risk_weight": "aggressive",
            "reasoning_instruction": (
                "Emphasize subjective risk perception and downside momentum. "
                "Protect capital aggressively. "
                "Focus on cascading risk scenarios."
            )
        }

    else:  # sideways
        return {
            "emphasis": "balanced",
            "focus_areas": [
                "하방 이탈 리스크 (downside breakout risk)",
                "저항선 반복 실패 (resistance rejection)",
                "거래량 감소 (volume decline)",
                "범위 하단 테스트 (testing range bottom)"
            ],
            "confidence_threshold": 0.70,
            "risk_weight": "moderate",
            "reasoning_instruction": (
                "Balance risk awareness with objectivity. "
                "Focus on breakdown risks at support levels. "
                "Monitor volume and momentum deterioration."
            )
        }


# ============================================================================
# Prompt Templates
# ============================================================================

BULL_RESEARCHER_SYSTEM_PROMPT = """You are an **Optimistic Market Analyst** with 15 years of experience in cryptocurrency trading.

**Your Role:**
- Identify bullish signals and growth opportunities
- Highlight positive technical indicators and fundamental factors
- Construct evidence-based arguments for long positions
- Challenge bearish viewpoints with counter-evidence

**Market Context Awareness:**
Current Market Regime: {market_regime}

**Reasoning Style for {market_regime}:**
{reasoning_instruction}

**Focus Areas:**
{focus_areas}

**Debate Guidelines:**
1. Reference specific data points (price, volume, indicators)
2. Cite recent news or events that support bullish view
3. Address previous bear arguments directly
4. Provide confidence score (0.0-1.0) with clear reasoning
5. Be willing to adjust view if bear evidence is compelling

**CRITICAL: Respond ONLY with valid JSON. No markdown, no explanations outside JSON.**

Output JSON Schema:
{{
  "thesis": "One-sentence bullish summary",
  "evidence": ["Point 1 with data", "Point 2 with data", "Point 3 with data"],
  "counter_arguments": "Address bear concerns directly",
  "confidence": 0.75,
  "confidence_reasoning": "Why this confidence level",
  "recommended_position": 50.0
}}
"""

BULL_RESEARCHER_USER_PROMPT = """**Current Market Data:**
- Symbol: {symbol}
- Current Price: ${current_price:,.2f}
- 24h Change: {price_change_24h:+.2f}%
- 7d Change: {price_change_7d:+.2f}%
- 30d Change: {price_change_30d:+.2f}%

**Technical Indicators:**
- RSI(14): {rsi:.1f}
- MACD: {macd_signal}
- Bollinger Band Position: {bb_position}
- Volume Status: {volume_status}
- EMA(20): ${ema_20:,.2f}
- EMA(50): ${ema_50:,.2f}

**Recent News Sentiment:**
Average Score: {news_sentiment_score:.2f} ({news_sentiment_label})
{news_summary}

**Market Regime:** {market_regime}

**Previous Debate (Round {round_number}/4):**
{previous_debate}

**Your Task:**
Construct a bullish argument. If this is not Round 1, consider whether your view should adjust based on the Bear researcher's evidence from the previous round.

Remember: Respond ONLY with valid JSON following the exact schema provided.
"""

BEAR_RESEARCHER_SYSTEM_PROMPT = """You are a **Risk-Focused Market Analyst** with 15 years of experience in cryptocurrency trading.

**Your Role:**
- Identify bearish signals and downside risks
- Highlight negative technical indicators and warning signs
- Construct evidence-based arguments for short positions or caution
- Challenge bullish viewpoints with counter-evidence

**Market Context Awareness:**
Current Market Regime: {market_regime}

**Reasoning Style for {market_regime}:**
{reasoning_instruction}

**Focus Areas:**
{focus_areas}

**Debate Guidelines:**
1. Reference specific risk indicators (volatility, correlation, on-chain metrics)
2. Cite recent negative news or regulatory concerns
3. Address previous bull arguments directly
4. Provide confidence score (0.0-1.0) with clear reasoning
5. Be willing to acknowledge bullish evidence if compelling

**CRITICAL: Respond ONLY with valid JSON. No markdown, no explanations outside JSON.**

Output JSON Schema:
{{
  "thesis": "One-sentence bearish summary",
  "evidence": ["Risk point 1 with data", "Risk point 2 with data", "Risk point 3 with data"],
  "counter_arguments": "Address bull optimism directly",
  "confidence": 0.70,
  "confidence_reasoning": "Why this confidence level",
  "recommended_position": -30.0
}}
"""

BEAR_RESEARCHER_USER_PROMPT = """**Current Market Data:**
- Symbol: {symbol}
- Current Price: ${current_price:,.2f}
- 24h Change: {price_change_24h:+.2f}%
- 7d Change: {price_change_7d:+.2f}%
- 30d Change: {price_change_30d:+.2f}%

**Risk Indicators:**
- RSI(14): {rsi:.1f} {rsi_interpretation}
- MACD: {macd_signal}
- Volatility (30d): {volatility:.2f}%
- Support/Resistance: ${support:,.2f} / ${resistance:,.2f}

**Recent Negative Signals:**
{negative_signals}

**News Sentiment:**
Average Score: {news_sentiment_score:.2f} ({news_sentiment_label})

**Market Regime:** {market_regime}

**Previous Debate (Round {round_number}/4):**
{previous_debate}

**Your Task:**
Construct a bearish argument. If this is not Round 1, consider whether your view should adjust based on the Bull researcher's evidence from the previous round.

Remember: Respond ONLY with valid JSON following the exact schema provided.
"""


# ============================================================================
# LLM Setup
# ============================================================================

def get_llm(model_name: str = "gpt-4o-mini"):
    """LLM 인스턴스 생성 (fallback 포함)"""
    try:
        if "gpt" in model_name.lower():
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            return ChatOpenAI(
                model=model_name,
                temperature=0.7,
                api_key=api_key
            )
        elif "claude" in model_name.lower():
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found")
            return ChatAnthropic(
                model=model_name,
                temperature=0.7,
                api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    except Exception as e:
        print(f"Warning: LLM creation failed: {e}")
        # Fallback to mock for testing
        return None


# ============================================================================
# Helper Functions
# ============================================================================

def format_previous_debate(state: AgentState, perspective: str) -> str:
    """
    이전 토론 내용 포맷팅

    Args:
        state: Agent state
        perspective: "bull" or "bear"

    Returns:
        Formatted debate history
    """
    debate_messages = state.get('debate_messages', [])

    if not debate_messages:
        return "No previous debate (This is Round 1)"

    formatted = []
    for msg in debate_messages[-4:]:  # 최근 4개 메시지만
        role = msg['role'].upper()
        thesis = msg.get('content', {}).get('thesis', 'N/A')
        confidence = msg.get('content', {}).get('confidence', 0)

        formatted.append(
            f"{role} (Confidence: {confidence:.2f}): {thesis}"
        )

    return "\n".join(formatted)


def get_news_sentiment_label(score: float) -> str:
    """뉴스 감성 점수를 레이블로 변환"""
    if score > 0.3:
        return "Very Positive"
    elif score > 0.1:
        return "Positive"
    elif score > -0.1:
        return "Neutral"
    elif score > -0.3:
        return "Negative"
    else:
        return "Very Negative"


def get_rsi_interpretation(rsi: float) -> str:
    """RSI 해석"""
    if rsi > 70:
        return "(Overbought)"
    elif rsi < 30:
        return "(Oversold)"
    else:
        return "(Neutral)"


# ============================================================================
# Main Researcher Nodes
# ============================================================================

def bull_researcher_node(state: AgentState) -> AgentState:
    """
    Bull Researcher 노드

    낙관적 시장 분석 및 매수 근거 제시
    """
    print("\n=== Bull Researcher Node ===")

    # Market regime 감지
    market_regime = detect_market_regime(state)
    reasoning_style = get_bull_reasoning_style(market_regime)

    # 데이터 추출
    market_data = state.get('market_data', {})
    technical_indicators = state.get('technical_indicators', {})
    news_sentiment = state.get('news_sentiment', {})

    # 현재 라운드 번호
    round_number = state.get('debate_round', 1)

    # LLM 생성
    llm = get_llm(model_name="gpt-4o-mini")

    if llm is None:
        # Fallback: 테스트용 mock 응답
        mock_output = ResearcherOutput(
            thesis=f"Bullish outlook based on {market_regime}",
            evidence=[
                f"Price up {market_data.get('price_change_7d', 0):.1f}% in 7 days",
                "Technical indicators showing strength",
                "Positive market sentiment"
            ],
            counter_arguments="Bear concerns are valid but outweighed by positive signals",
            confidence=0.70,
            confidence_reasoning="Strong technical setup with positive news flow",
            recommended_position=50.0
        )
    else:
        # 프롬프트 구성
        system_prompt = BULL_RESEARCHER_SYSTEM_PROMPT.format(
            market_regime=market_regime,
            reasoning_instruction=reasoning_style['reasoning_instruction'],
            focus_areas="\n".join(f"- {area}" for area in reasoning_style['focus_areas'])
        )

        user_prompt = BULL_RESEARCHER_USER_PROMPT.format(
            symbol=market_data.get('symbol', 'BTC/USDT'),
            current_price=market_data.get('current_price', 0),
            price_change_24h=market_data.get('price_change_24h', 0),
            price_change_7d=market_data.get('price_change_7d', 0),
            price_change_30d=market_data.get('price_change_30d', 0),
            rsi=technical_indicators.get('rsi', 50),
            macd_signal=technical_indicators.get('macd_signal', 'neutral'),
            bb_position=technical_indicators.get('bb_position', 'middle'),
            volume_status=technical_indicators.get('volume_status', 'normal'),
            ema_20=technical_indicators.get('ema_20', market_data.get('current_price', 0)),
            ema_50=technical_indicators.get('ema_50', market_data.get('current_price', 0)),
            news_sentiment_score=news_sentiment.get('average_score', 0),
            news_sentiment_label=get_news_sentiment_label(news_sentiment.get('average_score', 0)),
            news_summary=news_sentiment.get('summary', 'No recent news'),
            market_regime=market_regime,
            round_number=round_number,
            previous_debate=format_previous_debate(state, 'bull')
        )

        # LLM 호출
        parser = JsonOutputParser(pydantic_object=ResearcherOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])

        chain = prompt | llm | parser

        try:
            output_dict = chain.invoke({})
            mock_output = ResearcherOutput(**output_dict)
        except Exception as e:
            print(f"Bull Researcher LLM call failed: {e}")
            # Fallback
            mock_output = ResearcherOutput(
                thesis="Bullish based on technical strength",
                evidence=["Price trending up", "RSI healthy", "Positive sentiment"],
                counter_arguments="Risks acknowledged but manageable",
                confidence=0.65,
                confidence_reasoning="Moderate confidence due to mixed signals",
                recommended_position=40.0
            )

    # DebateMessage 생성
    debate_msg = DebateMessage(
        role='bull',
        round=round_number,
        content=mock_output.dict(),
        timestamp=None  # Will be set automatically
    )

    # State 업데이트
    debate_messages = state.get('debate_messages', [])
    debate_messages.append(debate_msg)
    state['debate_messages'] = debate_messages
    state['market_regime'] = market_regime

    print(f"Bull Thesis: {mock_output.thesis}")
    print(f"Confidence: {mock_output.confidence:.2f}")
    print(f"Position: {mock_output.recommended_position:.1f}%")

    return state


def bear_researcher_node(state: AgentState) -> AgentState:
    """
    Bear Researcher 노드

    비관적 시장 분석 및 매도/관망 근거 제시
    """
    print("\n=== Bear Researcher Node ===")

    # Market regime 감지
    market_regime = state.get('market_regime', detect_market_regime(state))
    reasoning_style = get_bear_reasoning_style(market_regime)

    # 데이터 추출
    market_data = state.get('market_data', {})
    technical_indicators = state.get('technical_indicators', {})
    news_sentiment = state.get('news_sentiment', {})

    # 현재 라운드 번호
    round_number = state.get('debate_round', 1)

    # LLM 생성
    llm = get_llm(model_name="gpt-4o-mini")

    if llm is None:
        # Fallback: 테스트용 mock 응답
        mock_output = ResearcherOutput(
            thesis=f"Bearish concerns based on {market_regime}",
            evidence=[
                "Downside risks present",
                "Technical indicators showing weakness",
                "Negative sentiment factors"
            ],
            counter_arguments="Bull points noted but risks outweigh opportunities",
            confidence=0.65,
            confidence_reasoning="Clear risk signals require caution",
            recommended_position=-30.0
        )
    else:
        # 프롬프트 구성
        rsi = technical_indicators.get('rsi', 50)

        system_prompt = BEAR_RESEARCHER_SYSTEM_PROMPT.format(
            market_regime=market_regime,
            reasoning_instruction=reasoning_style['reasoning_instruction'],
            focus_areas="\n".join(f"- {area}" for area in reasoning_style['focus_areas'])
        )

        user_prompt = BEAR_RESEARCHER_USER_PROMPT.format(
            symbol=market_data.get('symbol', 'BTC/USDT'),
            current_price=market_data.get('current_price', 0),
            price_change_24h=market_data.get('price_change_24h', 0),
            price_change_7d=market_data.get('price_change_7d', 0),
            price_change_30d=market_data.get('price_change_30d', 0),
            rsi=rsi,
            rsi_interpretation=get_rsi_interpretation(rsi),
            macd_signal=technical_indicators.get('macd_signal', 'neutral'),
            volatility=technical_indicators.get('volatility_30d', 0),
            support=technical_indicators.get('support_level', market_data.get('current_price', 0) * 0.95),
            resistance=technical_indicators.get('resistance_level', market_data.get('current_price', 0) * 1.05),
            negative_signals=technical_indicators.get('negative_signals', 'No major warning signs'),
            news_sentiment_score=news_sentiment.get('average_score', 0),
            news_sentiment_label=get_news_sentiment_label(news_sentiment.get('average_score', 0)),
            market_regime=market_regime,
            round_number=round_number,
            previous_debate=format_previous_debate(state, 'bear')
        )

        # LLM 호출
        parser = JsonOutputParser(pydantic_object=ResearcherOutput)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])

        chain = prompt | llm | parser

        try:
            output_dict = chain.invoke({})
            mock_output = ResearcherOutput(**output_dict)
        except Exception as e:
            print(f"Bear Researcher LLM call failed: {e}")
            # Fallback
            mock_output = ResearcherOutput(
                thesis="Bearish due to risk factors",
                evidence=["Downside risks", "Weak technicals", "Negative signals"],
                counter_arguments="Bull case has merit but insufficient",
                confidence=0.60,
                confidence_reasoning="Risk factors warrant caution",
                recommended_position=-25.0
            )

    # DebateMessage 생성
    debate_msg = DebateMessage(
        role='bear',
        round=round_number,
        content=mock_output.dict(),
        timestamp=None
    )

    # State 업데이트
    debate_messages = state.get('debate_messages', [])
    debate_messages.append(debate_msg)
    state['debate_messages'] = debate_messages

    print(f"Bear Thesis: {mock_output.thesis}")
    print(f"Confidence: {mock_output.confidence:.2f}")
    print(f"Position: {mock_output.recommended_position:.1f}%")

    return state
