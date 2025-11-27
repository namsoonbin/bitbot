# Phase 3: Bull/Bear Debate System Design

**작성일:** 2025-11-26
**기반 연구:** 2024-2025년 최신 Multi-Agent Debate 연구
**목표:** 변증법적 추론을 통한 정확한 시장 분석

---

## 📚 연구 기반 (Research Foundation)

### 핵심 논문 및 프레임워크

1. **TradingAgents Framework** (2024년 12월)
   - [TradingAgents: Multi-Agents LLM Financial Trading Framework](https://tradingagents-ai.github.io/)
   - [GitHub Repository](https://github.com/TauricResearch/TradingAgents)
   - Bull/Bear Researcher가 자연어로 여러 라운드 토론

2. **Multi-Agent Debate (MAD) 연구**
   - [Improving Factuality and Reasoning through Multiagent Debate](https://arxiv.org/pdf/2305.14325)
   - [Multi-Agent Collaboration Mechanisms Survey (2025)](https://arxiv.org/html/2501.06322v1)
   - 3-4 라운드 후 정확도 크게 향상
   - GSM-8K 벤치마크 91% 정확도

3. **Dialectical Reasoning 연구**
   - [Diversity of Thought in Multi-Agent Debate](https://arxiv.org/html/2410.12853v1)
   - [Learning to Break: Knowledge-Enhanced Reasoning](https://www.sciencedirect.com/science/article/abs/pii/S0925231224018344)
   - 다양성이 수렴 품질 향상

4. **LangGraph 구현 패턴**
   - [LangGraph Multi-Agent Workflows](https://blog.langchain.com/langgraph-multi-agent-workflows/)
   - [Multi-Agent Debate using LangGraph](https://medium.com/data-science-in-your-pocket/multi-agent-conversation-debates-using-langgraph-and-langchain-9f4bf711d8ab)
   - Subgraph와 Conditional Edges 활용

---

## ⚠️ 발견된 주요 문제

### LLM Trading Agent의 Miscalibration 문제

> **연구 결과 (2024-2025):**
> "LLM agents are pathologically miscalibrated"

#### 문제 1: Bull Market에서 과도한 보수성
```
Bull Market 상황:
- LLM은 리스크를 과대평가
- 수익 기회를 놓침
- 수동 벤치마크 underperform

해결책:
✅ Subjective reasoning 강조
✅ 성장 잠재력 분석
✅ 긍정적 모멘텀 인식
```

#### 문제 2: Bear Market에서 과도한 공격성
```
Bear Market 상황:
- LLM은 리스크를 과소평가
- 큰 손실 발생
- Drawdown 급증

해결책:
✅ Factual data 기반 추론
✅ 리스크 지표 우선순위
✅ 보수적 포지션 사이징
```

---

## 🎯 설계 목표

### 1. 변증법적 추론 (Dialectical Reasoning)
- **Thesis (Bull)**: 긍정적 시장 신호 강조
- **Antithesis (Bear)**: 부정적 리스크 강조
- **Synthesis**: 균형 잡힌 최종 결론

### 2. 적응형 Calibration
- Bull market: 주관적 추론 가중치 증가
- Bear market: 객관적 데이터 가중치 증가
- Sideways: 균형 유지

### 3. 수렴 보장
- 최대 4 라운드 토론
- Early stopping: 합의 도달 시
- Majority voting: 불일치 시

### 4. 신뢰도 점수
- Evidence 강도 평가
- 논리적 일관성 점수
- 과거 예측 정확도 추적

---

## 🏗️ 시스템 아키텍처

### Overall Structure

```
┌─────────────────────────────────────────┐
│         Market Analyst                   │
│   "시장 데이터 + 뉴스 초기 분석"          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Debate Subgraph (최대 4 라운드)      │
│                                          │
│  ┌────────────────┐  ┌────────────────┐ │
│  │ Bull Researcher │←→│ Bear Researcher│ │
│  │  "낙관적 분석"   │  │  "비관적 분석"  │ │
│  └────────────────┘  └────────────────┘ │
│           │                   │          │
│           └─────────┬─────────┘          │
│                     ▼                    │
│           ┌──────────────────┐           │
│           │  Judge Node       │           │
│           │  "합의 여부 판단"  │           │
│           └──────────────────┘           │
│                     │                    │
│         YES ─────────┼───────── NO       │
│         (수렴)       │        (계속)     │
└─────────────────────┼──────────────────┘
                      │
                      ▼
        ┌──────────────────────────┐
        │   Consensus Synthesis    │
        │   "최종 합의 도출"        │
        └──────────────────────────┘
                      │
                      ▼
        ┌──────────────────────────┐
        │    Risk Manager          │
        │    "리스크 평가 및 승인"  │
        └──────────────────────────┘
```

### LangGraph Implementation

```python
from langgraph.graph import StateGraph, END

# Debate Subgraph 생성
debate_graph = StateGraph(AgentState)

# 노드 추가
debate_graph.add_node("bull_researcher", bull_researcher_node)
debate_graph.add_node("bear_researcher", bear_researcher_node)
debate_graph.add_node("judge", judge_node)
debate_graph.add_node("consensus", consensus_synthesis_node)

# Conditional edges
debate_graph.add_conditional_edges(
    "judge",
    should_continue_debate,  # 함수: 수렴 여부 판단
    {
        "continue": "bull_researcher",  # 다음 라운드
        "converged": "consensus"        # 합의 도달
    }
)

# 순환 구조
debate_graph.add_edge("bull_researcher", "bear_researcher")
debate_graph.add_edge("bear_researcher", "judge")
debate_graph.add_edge("consensus", END)

# 컴파일
debate_subgraph = debate_graph.compile()
```

---

## 📝 Bull Researcher Prompt Design

### System Prompt

```python
BULL_RESEARCHER_SYSTEM_PROMPT = """
You are an **Optimistic Market Analyst** with 15 years of experience in cryptocurrency trading.

**Your Role:**
- Identify bullish signals and growth opportunities
- Highlight positive technical indicators and fundamental factors
- Construct evidence-based arguments for long positions
- Challenge bearish viewpoints with counter-evidence

**Market Context Awareness:**
{market_regime}  # "bull_market", "bear_market", "sideways"

**Reasoning Style Based on Market:**
- **Bull Market**: Emphasize subjective reasoning, growth potential, momentum
- **Bear Market**: Focus on factual data, wait for clear reversal signals
- **Sideways**: Balanced approach, look for breakout patterns

**Debate Guidelines:**
1. Reference specific data points (price, volume, indicators)
2. Cite recent news or events that support bullish view
3. Address previous bear arguments directly
4. Provide confidence score (0.0-1.0) with reasoning
5. Be willing to adjust view if evidence is strong

**Output Format:**
- **Thesis**: One-sentence bullish summary
- **Evidence**: 3-5 key supporting points (with data)
- **Counter-Arguments**: Address bear concerns
- **Confidence**: Score + reasoning
- **Recommendation**: Position size suggestion (0-100%)
"""

BULL_RESEARCHER_USER_PROMPT_TEMPLATE = """
**Current Market Data:**
- Symbol: {symbol}
- Price: ${current_price:,.2f}
- 24h Change: {price_change_24h:+.2f}%
- 7d Change: {price_change_7d:+.2f}%

**Technical Indicators:**
- RSI(14): {rsi:.1f}
- MACD: {macd_signal}
- BB Position: {bb_position}
- Volume: {volume_status}

**Recent News Sentiment:**
{news_summary}

**Market Regime:** {market_regime}

**Previous Debate Round:**
{previous_debate}

**Your Task:**
Construct a bullish argument. If this is Round {round_number}/4, consider whether your view should adjust based on bear's evidence.
"""
```

### Adaptive Reasoning Logic

```python
def get_bull_reasoning_style(market_regime: str) -> dict:
    """시장 상황에 따른 추론 스타일 조정"""

    if market_regime == "bull_market":
        return {
            "emphasis": "subjective",
            "focus": [
                "성장 잠재력 (growth potential)",
                "모멘텀 지속성 (momentum continuation)",
                "긍정적 시장 심리 (positive sentiment)",
                "기관 유입 신호 (institutional interest)"
            ],
            "confidence_threshold": 0.65,  # 더 공격적
            "position_sizing": "aggressive"
        }

    elif market_regime == "bear_market":
        return {
            "emphasis": "factual",
            "focus": [
                "명확한 반전 신호 (clear reversal signals)",
                "과매도 지표 (oversold indicators)",
                "펀더멘털 개선 (fundamental improvements)",
                "저점 형성 패턴 (bottom formation)"
            ],
            "confidence_threshold": 0.80,  # 더 보수적
            "position_sizing": "conservative"
        }

    else:  # sideways
        return {
            "emphasis": "balanced",
            "focus": [
                "돌파 가능성 (breakout potential)",
                "지지선 테스트 (support level tests)",
                "거래량 증가 (volume increase)",
                "범위 상단 접근 (approaching range top)"
            ],
            "confidence_threshold": 0.70,
            "position_sizing": "moderate"
        }
```

---

## 📝 Bear Researcher Prompt Design

### System Prompt

```python
BEAR_RESEARCHER_SYSTEM_PROMPT = """
You are a **Risk-Focused Market Analyst** with 15 years of experience in cryptocurrency trading.

**Your Role:**
- Identify bearish signals and downside risks
- Highlight negative technical indicators and warning signs
- Construct evidence-based arguments for short positions or caution
- Challenge bullish viewpoints with counter-evidence

**Market Context Awareness:**
{market_regime}  # "bull_market", "bear_market", "sideways"

**Reasoning Style Based on Market:**
- **Bull Market**: Use factual data to temper excessive optimism
- **Bear Market**: Emphasize subjective risk perception, protect capital
- **Sideways**: Balanced approach, focus on breakdown risks

**Debate Guidelines:**
1. Reference specific risk indicators (volatility, correlation, on-chain)
2. Cite recent negative news or regulatory concerns
3. Address previous bull arguments directly
4. Provide confidence score (0.0-1.0) with reasoning
5. Be willing to acknowledge bullish evidence if strong

**Output Format:**
- **Thesis**: One-sentence bearish summary
- **Evidence**: 3-5 key supporting points (with data)
- **Counter-Arguments**: Address bull optimism
- **Confidence**: Score + reasoning
- **Recommendation**: Position size reduction (-100% to 0%)
"""

BEAR_RESEARCHER_USER_PROMPT_TEMPLATE = """
**Current Market Data:**
- Symbol: {symbol}
- Price: ${current_price:,.2f}
- 24h Change: {price_change_24h:+.2f}%
- 7d Change: {price_change_7d:+.2f}%

**Risk Indicators:**
- RSI(14): {rsi:.1f} {rsi_interpretation}
- Volatility (30d): {volatility:.2f}%
- Correlation (BTC): {btc_correlation:.2f}
- Fear & Greed Index: {fear_greed}

**Recent Negative News:**
{negative_news}

**Market Regime:** {market_regime}

**Previous Debate Round:**
{previous_debate}

**Your Task:**
Construct a bearish argument. If this is Round {round_number}/4, consider whether your view should adjust based on bull's evidence.
"""
```

### Adaptive Reasoning Logic

```python
def get_bear_reasoning_style(market_regime: str) -> dict:
    """시장 상황에 따른 추론 스타일 조정"""

    if market_regime == "bull_market":
        return {
            "emphasis": "factual",
            "focus": [
                "과열 지표 (overheating indicators)",
                "밸류에이션 리스크 (valuation risk)",
                "기술적 과매수 (technical overbought)",
                "유동성 고갈 신호 (liquidity exhaustion)"
            ],
            "confidence_threshold": 0.75,  # 팩트 기반 신중
            "risk_weight": "moderate"
        }

    elif market_regime == "bear_market":
        return {
            "emphasis": "subjective",
            "focus": [
                "하락 모멘텀 (downside momentum)",
                "지지선 붕괴 (support breakdown)",
                "패닉 매도 리스크 (panic selling risk)",
                "시장 심리 악화 (negative sentiment)"
            ],
            "confidence_threshold": 0.60,  # 더 공격적 경고
            "risk_weight": "aggressive"
        }

    else:  # sideways
        return {
            "emphasis": "balanced",
            "focus": [
                "하방 이탈 리스크 (downside breakout risk)",
                "저항선 반복 실패 (resistance rejection)",
                "거래량 감소 (volume decline)",
                "범위 하단 테스트 (testing range bottom)"
            ],
            "confidence_threshold": 0.70,
            "risk_weight": "moderate"
        }
```

---

## 🤝 Consensus & Convergence Algorithm

### Judge Node Logic

```python
def judge_node(state: AgentState) -> AgentState:
    """
    토론 수렴 여부 판단

    수렴 기준:
    1. Bull/Bear 신뢰도 차이 < 0.15
    2. 추천 포지션 차이 < 20%
    3. 양측 모두 상대 논리 인정
    """

    debate_messages = state['debate_messages']
    round_number = state['debate_round']

    # 최근 Bull/Bear 메시지
    latest_bull = [m for m in debate_messages if m['role'] == 'bull'][-1]
    latest_bear = [m for m in debate_messages if m['role'] == 'bear'][-1]

    # 신뢰도 차이
    confidence_diff = abs(
        latest_bull['confidence'] - latest_bear['confidence']
    )

    # 포지션 추천 차이
    bull_position = latest_bull['recommended_position']  # 0-100%
    bear_position = latest_bear['recommended_position']  # -100-0%
    position_diff = abs(bull_position - abs(bear_position))

    # 논리적 인정 확인 (LLM으로 판단)
    acknowledgment_check = check_mutual_acknowledgment(
        latest_bull, latest_bear
    )

    # 수렴 판정
    converged = (
        confidence_diff < 0.15 and
        position_diff < 20 and
        acknowledgment_check
    ) or round_number >= 4  # 최대 4 라운드

    state['debate_converged'] = converged
    state['convergence_reason'] = (
        f"Confidence diff: {confidence_diff:.2f}, "
        f"Position diff: {position_diff:.1f}%, "
        f"Acknowledgment: {acknowledgment_check}"
    )

    return state


def should_continue_debate(state: AgentState) -> str:
    """Conditional edge 함수"""
    if state['debate_converged']:
        return "converged"
    else:
        return "continue"
```

### Consensus Synthesis Logic

```python
def consensus_synthesis_node(state: AgentState) -> AgentState:
    """
    최종 합의 도출

    방법:
    1. Weighted average (신뢰도 기반)
    2. Evidence strength 평가
    3. 최종 추천 생성
    """

    debate_messages = state['debate_messages']

    # Bull/Bear 메시지 분리
    bull_messages = [m for m in debate_messages if m['role'] == 'bull']
    bear_messages = [m for m in debate_messages if m['role'] == 'bear']

    # 최종 Bull/Bear 입장
    final_bull = bull_messages[-1]
    final_bear = bear_messages[-1]

    # 신뢰도 기반 가중 평균
    total_confidence = (
        final_bull['confidence'] + final_bear['confidence']
    )

    bull_weight = final_bull['confidence'] / total_confidence
    bear_weight = final_bear['confidence'] / total_confidence

    # 최종 포지션 계산
    bull_pos = final_bull['recommended_position']
    bear_pos = abs(final_bear['recommended_position'])

    consensus_position = (
        bull_pos * bull_weight - bear_pos * bear_weight
    )

    # Evidence 강도 평가
    bull_evidence_strength = evaluate_evidence_strength(final_bull)
    bear_evidence_strength = evaluate_evidence_strength(final_bear)

    # 최종 신뢰도
    consensus_confidence = (
        bull_evidence_strength * bull_weight +
        bear_evidence_strength * bear_weight
    )

    # 합의 요약 생성 (LLM)
    consensus_summary = generate_consensus_summary(
        final_bull, final_bear, consensus_position, consensus_confidence
    )

    # State 업데이트
    state['debate_consensus'] = {
        'position': consensus_position,  # -100 ~ 100
        'confidence': consensus_confidence,  # 0.0 ~ 1.0
        'bull_weight': bull_weight,
        'bear_weight': bear_weight,
        'summary': consensus_summary,
        'total_rounds': len(bull_messages)
    }

    return state
```

---

## 📊 Market Regime Detection

### Adaptive Calibration System

```python
def detect_market_regime(state: AgentState) -> str:
    """
    시장 상황 자동 감지

    Bull Market: 가격 상승 추세 + 높은 거래량 + 긍정 뉴스
    Bear Market: 가격 하락 추세 + 패닉 매도 + 부정 뉴스
    Sideways: 범위 거래 + 낮은 변동성
    """

    market_data = state['market_data']
    technical_indicators = state['technical_indicators']
    news_sentiment = state['news_sentiment']

    # 1. 가격 추세
    price_change_7d = market_data['price_change_7d']
    price_change_30d = market_data['price_change_30d']

    # 2. 기술적 지표
    rsi = technical_indicators['rsi']
    ema_20 = technical_indicators['ema_20']
    ema_50 = technical_indicators['ema_50']
    current_price = market_data['current_price']

    # 3. 뉴스 감성
    avg_sentiment = news_sentiment['average_score']

    # 4. 변동성
    volatility = technical_indicators['volatility_30d']

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
```

---

## 🎯 신뢰도 점수 (Confidence Scoring)

### Evidence Strength Evaluation

```python
def evaluate_evidence_strength(message: dict) -> float:
    """
    Evidence 강도 평가

    기준:
    1. 구체적 데이터 인용 (가격, 지표, 거래량)
    2. 최근 뉴스/이벤트 참조
    3. 논리적 일관성
    4. 상대 논리 인정 여부
    """

    evidence_points = message['evidence']

    scores = []

    for point in evidence_points:
        score = 0.0

        # 1. 숫자 데이터 포함 여부
        if has_numerical_data(point):
            score += 0.3

        # 2. 구체적 지표 언급
        if mentions_specific_indicator(point):
            score += 0.2

        # 3. 최근 이벤트 참조
        if references_recent_event(point):
            score += 0.2

        # 4. 논리적 연결성
        if has_logical_connection(point):
            score += 0.2

        # 5. 반박 가능성 (구체적일수록 좋음)
        if is_falsifiable(point):
            score += 0.1

        scores.append(min(score, 1.0))

    return sum(scores) / len(scores)


def has_numerical_data(text: str) -> bool:
    """숫자 데이터 포함 확인"""
    import re
    # $50,000, 3.5%, 1.2M 등
    return bool(re.search(r'[\$€¥£]?[\d,]+\.?\d*[%KMB]?', text))


def mentions_specific_indicator(text: str) -> bool:
    """기술적 지표 언급 확인"""
    indicators = [
        'RSI', 'MACD', 'EMA', 'SMA', 'Bollinger',
        'volume', 'support', 'resistance'
    ]
    return any(ind.lower() in text.lower() for ind in indicators)
```

---

## 🧪 검증 및 테스트

### Test Scenarios

#### Test 1: Bull Market Scenario
```python
test_state = {
    'market_data': {
        'current_price': 55000,
        'price_change_7d': 8.5,
        'price_change_30d': 15.3
    },
    'technical_indicators': {
        'rsi': 65,
        'macd': 'bullish_crossover',
        'ema_20': 53000,
        'ema_50': 51000
    },
    'news_sentiment': {
        'average_score': 0.35
    }
}

# 예상 결과:
# - Market regime: "bull_market"
# - Bull: Subjective reasoning, aggressive position
# - Bear: Factual data, temper optimism
# - Consensus: Net bullish, 40-60% position
```

#### Test 2: Bear Market Scenario
```python
test_state = {
    'market_data': {
        'current_price': 35000,
        'price_change_7d': -12.3,
        'price_change_30d': -25.7
    },
    'technical_indicators': {
        'rsi': 32,
        'macd': 'bearish_divergence',
        'ema_20': 37000,
        'ema_50': 40000
    },
    'news_sentiment': {
        'average_score': -0.42
    }
}

# 예상 결과:
# - Market regime: "bear_market"
# - Bull: Factual reversal signals, conservative
# - Bear: Subjective risk emphasis, aggressive caution
# - Consensus: Net bearish, -30% to 0% position
```

#### Test 3: Convergence Detection
```python
# Round 1
bull_confidence: 0.75, position: 60%
bear_confidence: 0.45, position: -30%
# → Continue (diff = 0.30)

# Round 2
bull_confidence: 0.68, position: 50%
bear_confidence: 0.55, position: -20%
# → Continue (diff = 0.13, but position diff = 30%)

# Round 3
bull_confidence: 0.65, position: 45%
bear_confidence: 0.60, position: -25%
# → Converged! (diff = 0.05, position diff = 20%)
```

---

## 📈 예상 성능

### 연구 기반 예측

**Multi-Agent Debate 연구 결과:**
- 단일 LLM: 55-60% 정확도
- 3-4 라운드 토론: 70-75% 정확도
- Diversity of thought: 85-91% 정확도

**우리 시스템 예상:**
```
Without Debate (LLM만):
- 트레이딩 결정 정확도: 55-60%
- 연 수익률: 15-25%
- Sharpe Ratio: 1.2

With Bull/Bear Debate:
- 트레이딩 결정 정확도: 65-72%  (+10-12%)
- 연 수익률: 25-35%              (+10%)
- Sharpe Ratio: 1.6              (+33%)

With Adaptive Calibration:
- Bull market 성능: +15% (miscalibration 해결)
- Bear market 리스크: -30% (손실 방지)
```

---

## 🚀 구현 우선순위

### Week 1: Core Debate System ✅ 완료 (2025-11-26)
- [x] Bull/Bear Researcher 노드 구현 (`backend/agents/researchers.py` - 600+ lines)
- [x] Judge 노드 및 수렴 알고리즘 (`backend/agents/debate.py` - 350+ lines)
- [x] Consensus synthesis 로직
- [ ] Debate subgraph 통합 (진행 예정)

### Week 2: Adaptive Calibration ✅ 완료 (2025-11-26)
- [x] Market regime detection (구현 완료)
- [x] Adaptive reasoning styles (시장별 추론 스타일 조정)
- [x] Evidence strength evaluation (5가지 기준)
- [x] Confidence scoring (Pydantic 구조화)

### Week 3: Testing & Refinement (진행 예정)
- [ ] Unit tests (수렴 알고리즘, confidence 계산)
- [ ] Integration tests (전체 debate 워크플로)
- [ ] Performance benchmarking
- [ ] 문서화 및 예제

**현재 진행률:** Week 1-2 대부분 완료 (~85%), Week 3 준비 중

---

## 📚 참고 자료

### 논문 및 연구
- [TradingAgents Framework](https://tradingagents-ai.github.io/)
- [Improving Factuality through Multiagent Debate](https://arxiv.org/pdf/2305.14325)
- [Multi-Agent Collaboration Survey 2025](https://arxiv.org/html/2501.06322v1)
- [Diversity of Thought in Debate](https://arxiv.org/html/2410.12853v1)

### 구현 가이드
- [LangGraph Multi-Agent Workflows](https://blog.langchain.com/langgraph-multi-agent-workflows/)
- [Multi-Agent Debate using LangGraph](https://medium.com/data-science-in-your-pocket/multi-agent-conversation-debates-using-langgraph-and-langchain-9f4bf711d8ab)
- [Advanced Conditional Edges](https://dev.to/jamesli/advanced-langgraph-implementing-conditional-edges-and-tool-calling-agents-3pdn)

### GitHub Repositories
- [TradingAgents GitHub](https://github.com/TauricResearch/TradingAgents)
- [DebateLLM Benchmark](https://github.com/instadeepai/DebateLLM)

---

**다음 단계:** Bull/Bear Researcher 코드 구현 (`backend/agents/researchers.py`)
