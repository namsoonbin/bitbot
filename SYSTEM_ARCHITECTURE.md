# 🏗️ HATS Trading System - 시스템 아키텍처 및 작업 흐름

**작성일:** 2025-11-26
**버전:** 1.0

---

## 📋 목차

1. [전체 시스템 개요](#전체-시스템-개요)
2. [자동매매 프로세스](#자동매매-프로세스)
3. [기술 스택](#기술-스택)
4. [Agent 워크플로우](#agent-워크플로우)
5. [데이터 파이프라인](#데이터-파이프라인)
6. [LLM 추론 과정](#llm-추론-과정)
7. [의사결정 흐름](#의사결정-흐름)
8. [백테스팅 시스템](#백테스팅-시스템)

---

## 🎯 전체 시스템 개요

### HATS (Hybrid AI Trading System)란?

**핵심 컨셉:** LLM(대규모 언어 모델)을 활용한 다중 Agent 기반 자율 트레이딩 시스템

```
┌─────────────────────────────────────────────────────────────────┐
│                     HATS Trading System                         │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │ Data Layer   │ → │ Agent Layer   │ → │ Execution    │    │
│  │              │    │               │    │ Layer        │    │
│  │ - PostgreSQL │    │ - 5 Agents    │    │ - Backtest   │    │
│  │ - MongoDB    │    │ - LangGraph   │    │ - Live Trade │    │
│  │ - Redis      │    │ - LLM Reasoning│   │ - Risk Mgmt  │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 시스템 특징

1. **Multi-Agent 협업**: 5개의 전문 Agent가 각자의 역할 수행
2. **Chain-of-Thought**: 모든 의사결정 과정 추적 및 기록
3. **Dialectical Reasoning**: Bull vs Bear 토론을 통한 균형잡힌 판단
4. **Human-in-the-Loop**: 중요한 거래는 사람의 승인 필요 (옵션)
5. **Landscape of Thoughts**: 추론 과정 시각화

---

## 🔄 자동매매 프로세스

### 전체 흐름도

```
┌─────────────────────────────────────────────────────────────────────┐
│                        자동매매 실행 흐름                             │
└─────────────────────────────────────────────────────────────────────┘

1️⃣ 데이터 수집 (매 1시간)
   ┌─────────────────────────────────────┐
   │ • Binance: OHLCV 데이터              │
   │ • CryptoPanic: 뉴스 + 감성           │
   │ • Technical Indicators: RSI, MACD    │
   └─────────────────────────────────────┘
                    ↓
2️⃣ Agent 분석 시작 (LangGraph 실행)
   ┌─────────────────────────────────────┐
   │  START → Analyst Node                │
   │           ↓                          │
   │       [조건부 분기]                   │
   │      연구 필요? YES → Researcher     │
   │                NO  → END             │
   └─────────────────────────────────────┘
                    ↓
3️⃣ Analyst Node: 초기 분석
   ┌─────────────────────────────────────┐
   │ INPUT:                               │
   │  - 최근 1주일 가격 데이터             │
   │  - 최근 3일 뉴스 (20개)              │
   │  - 감성 점수 평균                     │
   │                                      │
   │ PROCESSING:                          │
   │  - LLM에게 질문: "이 데이터를           │
   │    보고 시장을 어떻게 해석하는가?"       │
   │                                      │
   │ OUTPUT:                              │
   │  - Fundamental Analysis (뉴스 기반)  │
   │  - Technical Analysis (가격 기반)    │
   │  - Key Concerns (주요 리스크)        │
   │  - 신뢰도 점수: 0.7                  │
   └─────────────────────────────────────┘
                    ↓
4️⃣ Bull Researcher: 상승 케이스 구축
   ┌─────────────────────────────────────┐
   │ ROLE: 낙관론자 (변호사 역할)          │
   │                                      │
   │ TASK:                                │
   │  - 왜 가격이 오를 것인가?             │
   │  - 긍정적 신호는?                     │
   │  - 상승 모멘텀은?                     │
   │                                      │
   │ OUTPUT:                              │
   │  - Bull Case: "긍정적 뉴스 트렌드,    │
   │    지지선 유지, 모멘텀 강화..."        │
   │  - 신뢰도: 0.6                       │
   └─────────────────────────────────────┘
                    ↓
5️⃣ Bear Researcher: 하락 케이스 구축
   ┌─────────────────────────────────────┐
   │ ROLE: 비관론자 (검사 역할)            │
   │                                      │
   │ TASK:                                │
   │  - 왜 가격이 떨어질 것인가?           │
   │  - 부정적 신호는?                     │
   │  - 하락 리스크는?                     │
   │                                      │
   │ OUTPUT:                              │
   │  - Bear Case: "변동성 높음,           │
   │    규제 불확실성, 저항선..."          │
   │  - 신뢰도: 0.5                       │
   └─────────────────────────────────────┘
                    ↓
6️⃣ Risk Manager: 리스크 평가
   ┌─────────────────────────────────────┐
   │ INPUT:                               │
   │  - Bull Case (신뢰도 0.6)            │
   │  - Bear Case (신뢰도 0.5)            │
   │  - 현재 포트폴리오 상태               │
   │                                      │
   │ CHECKS:                              │
   │  ✓ 포지션 크기 < 30% (MAX_POSITION)  │
   │  ✓ 일일 손실 < 5% (MAX_DAILY_LOSS)   │
   │  ✓ 신뢰도 차이 > 임계값               │
   │  ✓ Guardrails 검증 통과              │
   │                                      │
   │ DECISION:                            │
   │  - APPROVED: 거래 진행 ✅            │
   │  - REJECTED: 거래 거부 ❌            │
   │  - 리스크 점수: 0.3 (낮음)           │
   └─────────────────────────────────────┘
                    ↓
7️⃣ Trader Node: 거래 실행 기록
   ┌─────────────────────────────────────┐
   │ FINAL DECISION: HOLD                 │
   │                                      │
   │ TRADE DETAILS:                       │
   │  - Action: HOLD                      │
   │  - Allocation: 10%                   │
   │  - Stop Loss: 2%                     │
   │  - Take Profit: 5%                   │
   │  - Confidence: 0.6                   │
   │                                      │
   │ EXECUTION:                           │
   │  - 백테스트 모드: 로그만 기록         │
   │  - 실거래 모드: 거래소 API 호출       │
   └─────────────────────────────────────┘
                    ↓
8️⃣ 결과 저장
   ┌─────────────────────────────────────┐
   │ MongoDB:                             │
   │  - reasoning_logs: 전체 추론 과정    │
   │  - agent_checkpoints: 상태 저장      │
   │                                      │
   │ PostgreSQL:                          │
   │  - trades: 거래 기록                 │
   │  - portfolio_snapshots: 포트폴리오   │
   │  - backtest_results: 백테스트 결과   │
   └─────────────────────────────────────┘
```

---

## 🛠️ 기술 스택

### 1. 프레임워크 & 라이브러리

#### Core Frameworks
```python
┌─────────────────────────────────────────┐
│ LangGraph (Agent Orchestration)         │
│ - 버전: latest                           │
│ - 역할: 5개 Agent의 워크플로우 조정      │
│ - 기능: 조건부 엣지, 체크포인팅, 추론     │
│                                          │
│ 왜 LangGraph?                            │
│ ✓ CrewAI보다 세밀한 제어 가능            │
│ ✓ 순환(Cyclic) 워크플로우 지원           │
│ ✓ 상태 저장 및 복구 (Checkpointing)      │
│ ✓ 조건부 분기 (Conditional Edges)        │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ LangChain (LLM Integration)              │
│ - 역할: LLM 호출 추상화                  │
│ - 지원 모델:                             │
│   • OpenAI: GPT-4o-mini, GPT-4          │
│   • Anthropic: Claude-3.5-Sonnet         │
│ - 기능: Prompt 템플릿, 파서, 캐싱        │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Lumibot (Backtesting Engine)            │
│ - 역할: 이벤트 기반 백테스팅             │
│ - 특징: Lookahead Bias 방지              │
│ - 기능: 포트폴리오 관리, 수익률 계산     │
└─────────────────────────────────────────┘
```

#### Data & Database
```python
┌─────────────────────────────────────────┐
│ PostgreSQL + TimescaleDB                 │
│ - 용도: 시계열 데이터 (OHLCV)            │
│ - 최적화: Hypertable로 쿼리 성능 향상    │
│ - 데이터: 8,761 캔들 (1년치)             │
│                                          │
│ pgvector Extension                       │
│ - 용도: 벡터 임베딩 저장                 │
│ - 기능: RAG (Retrieval Augmented Gen)    │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ MongoDB                                  │
│ - 용도: 비정형 데이터 (뉴스, 추론 로그)  │
│ - 컬렉션:                                │
│   • reasoning_logs: Agent 추론 과정      │
│   • news: 뉴스 + 감성 데이터             │
│   • agent_checkpoints: 상태 스냅샷       │
│   • backtest_metadata: 백테스트 설정     │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Redis                                    │
│ - 용도: Semantic Caching                 │
│ - 기능: LLM 응답 캐싱 (40-68.8% 절감)    │
│ - 메커니즘:                              │
│   1. 프롬프트 → 임베딩 벡터              │
│   2. 유사도 검색 (코사인 유사도)         │
│   3. 임계값(0.95) 이상이면 캐시 사용     │
└─────────────────────────────────────────┘
```

#### AI/ML Libraries
```python
┌─────────────────────────────────────────┐
│ Guardrails AI                            │
│ - 역할: LLM 출력 검증 및 안전성          │
│ - 검증 항목:                             │
│   • Valid Range: 가격/수량 범위 체크     │
│   • Valid JSON: 출력 형식 검증           │
│   • Financial Tone: 금융 적절성 검사     │
│ - 실패 시: 재시도 또는 거부              │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ TA-Lib (Technical Analysis Library)      │
│ - 역할: 기술적 지표 계산                 │
│ - 지표:                                  │
│   • RSI (Relative Strength Index)        │
│   • MACD (Moving Average Conv/Div)       │
│   • Bollinger Bands (볼린저 밴드)        │
│   • Support/Resistance (지지/저항선)     │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Sentence Transformers                    │
│ - 모델: all-MiniLM-L6-v2                 │
│ - 역할: 텍스트 → 벡터 임베딩             │
│ - 용도: Semantic Caching, RAG            │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ UMAP (Dimensionality Reduction)          │
│ - 역할: 고차원 벡터 → 2D/3D 시각화       │
│ - 용도: Landscape of Thoughts 렌더링    │
└─────────────────────────────────────────┘
```

---

## 🤖 Agent 워크플로우

### LangGraph StateGraph 구조

```python
from langgraph.graph import StateGraph, END

# 1. Graph 생성
workflow = StateGraph(AgentState)

# 2. 노드 추가 (5개 Agent)
workflow.add_node("analyst", analyst_node)
workflow.add_node("bull_researcher", bull_researcher_node)
workflow.add_node("bear_researcher", bear_researcher_node)
workflow.add_node("risk_manager", risk_manager_node)
workflow.add_node("trader", trader_node)

# 3. 시작점 설정
workflow.set_entry_point("analyst")

# 4. 조건부 엣지 (Conditional Edges)
workflow.add_conditional_edges(
    "analyst",
    should_continue_research,  # 함수: 연구 필요 여부 판단
    {
        "bull_researcher": "bull_researcher",  # YES → 연구 진행
        "end": END                              # NO → 종료
    }
)

# 5. 순차 흐름
workflow.add_edge("bull_researcher", "bear_researcher")
workflow.add_edge("bear_researcher", "risk_manager")

# 6. 거래 실행 조건부
workflow.add_conditional_edges(
    "risk_manager",
    should_execute_trade,  # 함수: 리스크 승인 여부
    {
        "trader": "trader",  # APPROVED → 거래 실행
        "end": END           # REJECTED → 거래 거부
    }
)

workflow.add_edge("trader", END)

# 7. 컴파일
app = workflow.compile()
```

### AgentState (상태 관리)

```python
class AgentState(TypedDict):
    # 워크플로 제어
    current_node: str              # 현재 노드 이름
    iteration: int                 # 반복 횟수
    should_continue: bool          # 계속 진행 여부
    error: Optional[str]           # 에러 메시지

    # 시장 데이터
    market_data: MarketData        # 현재 가격, 거래량 등
    technical_indicators: TechnicalIndicators  # RSI, MACD 등
    recent_news: List[NewsItem]    # 최근 뉴스 (3일)
    historical_prices: List[Dict]  # 과거 가격 (1주일)

    # 분석 결과
    fundamental_analysis: str      # 펀더멘털 분석 (뉴스 기반)
    technical_analysis: str        # 기술적 분석 (차트 기반)
    sentiment_analysis: str        # 감성 분석
    sentiment_score: float         # 감성 점수 (-1 ~ +1)

    # 추론 과정 (Chain-of-Thought)
    reasoning_trace: List[ReasoningStep]  # 각 Agent의 사고 과정
    debate_transcript: List[DebateMessage]  # Bull vs Bear 토론

    # 연구 결과
    bull_case: str                 # 상승 케이스
    bull_confidence: float         # Bull 신뢰도
    bear_case: str                 # 하락 케이스
    bear_confidence: float         # Bear 신뢰도

    # 거래 결정
    proposed_trade: ProposedTrade  # 제안된 거래
    risk_assessment: RiskAssessment  # 리스크 평가
    final_decision: Literal['BUY', 'SELL', 'HOLD']  # 최종 결정

    # 포트폴리오
    portfolio: PortfolioState      # 현재 보유 자산

    # 메타데이터
    session_id: str                # 세션 ID
    thread_id: str                 # 스레드 ID
    started_at: datetime           # 시작 시간
    completed_at: datetime         # 완료 시간

    # LLM 추적
    api_calls_count: int           # API 호출 횟수
    tokens_used: int               # 사용한 토큰 수
    cache_hits: int                # 캐시 히트 횟수
    cache_misses: int              # 캐시 미스 횟수
```

---

## 📡 데이터 파이프라인

### 외부 API 및 데이터 소스

```
┌────────────────────────────────────────────────────────────────┐
│                     데이터 수집 파이프라인                       │
└────────────────────────────────────────────────────────────────┘

1️⃣ OHLCV 데이터 (가격 데이터)
   ┌─────────────────────────────────────┐
   │ API: Binance Public API              │
   │ 라이브러리: CCXT (Python)             │
   │ 엔드포인트: /api/v3/klines           │
   │                                      │
   │ 요청 예시:                            │
   │ GET https://api.binance.com/...      │
   │   ?symbol=BTCUSDT                    │
   │   &interval=1h                       │
   │   &limit=1000                        │
   │                                      │
   │ 응답 데이터:                          │
   │ [                                    │
   │   [1700000000,  // timestamp         │
   │    87500.00,    // open              │
   │    88000.00,    // high              │
   │    87000.00,    // low               │
   │    87800.00,    // close             │
   │    1000.5]      // volume            │
   │ ]                                    │
   │                                      │
   │ 저장: PostgreSQL ohlcv_btcusdt_1h    │
   │ 빈도: 1시간마다 자동 수집 (Cron)      │
   └─────────────────────────────────────┘

2️⃣ 뉴스 데이터 (감성 분석)
   ┌─────────────────────────────────────┐
   │ API: CryptoPanic API v2              │
   │ 엔드포인트: /api/developer/v2/posts  │
   │ 인증: API Token (Bearer)             │
   │                                      │
   │ 요청 예시:                            │
   │ GET https://cryptopanic.com/...      │
   │   ?auth_token=YOUR_TOKEN             │
   │   &currencies=BTC                    │
   │   &kind=news                         │
   │   &public=true                       │
   │                                      │
   │ 응답 데이터:                          │
   │ {                                    │
   │   "results": [                       │
   │     {                                │
   │       "title": "Bitcoin reaches...", │
   │       "published_at": "2025-11-25",  │
   │       "source": "CoinDesk",          │
   │       "votes": {                     │
   │         "positive": 10,  // 긍정     │
   │         "negative": 2,   // 부정     │
   │         "neutral": 5     // 중립     │
   │       }                              │
   │     }                                │
   │   ]                                  │
   │ }                                    │
   │                                      │
   │ 감성 점수 계산:                       │
   │ score = (positive - negative) / total│
   │ 범위: -1.0 (매우 부정) ~ +1.0 (매우긍정)│
   │                                      │
   │ 저장: MongoDB news 컬렉션            │
   │ 빈도: 30분마다 자동 수집              │
   └─────────────────────────────────────┘

3️⃣ 기술적 지표 (계산)
   ┌─────────────────────────────────────┐
   │ 라이브러리: TA-Lib                   │
   │ 입력: PostgreSQL OHLCV 데이터        │
   │                                      │
   │ 계산 지표:                            │
   │                                      │
   │ RSI (Relative Strength Index)        │
   │ - 과매수/과매도 판단                  │
   │ - 범위: 0-100                        │
   │ - 과매수: RSI > 70                   │
   │ - 과매도: RSI < 30                   │
   │                                      │
   │ MACD (이동평균 수렴/확산)             │
   │ - 추세 전환 신호                      │
   │ - 골든크로스: 매수 신호               │
   │ - 데드크로스: 매도 신호               │
   │                                      │
   │ Bollinger Bands (볼린저 밴드)        │
   │ - 변동성 측정                         │
   │ - 상단 밴드 돌파: 과열                │
   │ - 하단 밴드 터치: 반등 가능성         │
   │                                      │
   │ 저장: AgentState (메모리)            │
   │ 빈도: Agent 실행 시 실시간 계산       │
   └─────────────────────────────────────┘
```

---

## 🧠 LLM 추론 과정

### 1. Analyst Node - LLM 호출 예시

```python
# 1. 프롬프트 구성
from langchain_core.prompts import ChatPromptTemplate

analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a cryptocurrency market analyst with 10 years of experience.
    Analyze the provided market data and news to form an initial assessment.
    Be concise but insightful. Focus on key signals and trends.
    """),

    ("user", """
    Current Market Data:
    - Symbol: {symbol}
    - Current Price: ${current_price:,.2f}
    - 24h Change: {price_change:+.2f}%
    - 7-day change: {week_change:+.2f}%

    Technical Indicators:
    - RSI: {rsi:.1f} (Oversold < 30, Overbought > 70)
    - MACD: {macd_signal}
    - Bollinger: {bb_position}

    Recent News (Last 3 days, Average Sentiment: {sentiment:.2f}):
    {news_summary}

    Provide your analysis in the following format:

    1. Fundamental Analysis (150 words):
       What do the news and market sentiment suggest?

    2. Technical Analysis (150 words):
       What do the price trends and indicators show?

    3. Key Concerns (100 words):
       What are the major risks or uncertainties?
    """)
])

# 2. LLM 선택
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,  # 적당한 창의성
    max_tokens=500
)

# 3. 체인 구성 및 실행
chain = analysis_prompt | llm | StrOutputParser()

result = chain.invoke({
    'symbol': 'BTC/USDT',
    'current_price': 87500.0,
    'price_change': 0.58,
    'week_change': 2.34,
    'rsi': 45.2,
    'macd_signal': 'Bullish (Golden Cross)',
    'bb_position': 'Middle band',
    'sentiment': 0.35,
    'news_summary': """
    - [CoinDesk] Bitcoin reaches new resistance at $88K (Sentiment: 0.5)
    - [Bloomberg] Regulatory clarity expected in Q1 2026 (Sentiment: 0.4)
    - [CryptoNews] Institutional buying accelerates (Sentiment: 0.6)
    """
})

# 4. 결과 파싱 및 저장
state['fundamental_analysis'] = extract_section(result, "Fundamental Analysis")
state['technical_analysis'] = extract_section(result, "Technical Analysis")
state['key_concerns'] = extract_section(result, "Key Concerns")

# 5. Reasoning Trace에 추가
add_reasoning_step(
    state,
    role='Analyst',
    content=f"Market Analysis:\n{result}",
    confidence=0.7
)
```

### 2. Bull Researcher - 변증법적 추론

```python
bull_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a BULL (optimistic) crypto analyst.
    Your job is to build the STRONGEST possible case for why the price will GO UP.
    Be persuasive, find supporting evidence, present the optimistic scenario.
    BUT remain realistic - don't make up facts.
    """),

    ("user", """
    Initial Analysis:
    {analyst_summary}

    Your task:
    Build a compelling BULL CASE for why BTC/USDT will rise.

    Consider:
    - Positive news signals
    - Technical support levels
    - Market momentum
    - Fundamental drivers

    Structure:
    1. Main Thesis (2-3 sentences)
    2. Supporting Evidence (3-5 bullet points)
    3. Price Target & Timeline
    4. Confidence Level (0.0 - 1.0)
    """)
])

bull_result = chain.invoke({...})

# Debate에 추가
add_debate_message(
    state,
    speaker='Bull',
    message=bull_result,
    confidence=0.6
)
```

### 3. Bear Researcher - 반대 입장

```python
bear_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a BEAR (pessimistic) crypto analyst.
    Your job is to build the STRONGEST possible case for why the price will GO DOWN.
    Be critical, identify risks, present the pessimistic scenario.
    BUT remain realistic - don't make up facts.
    """),

    ("user", """
    Initial Analysis:
    {analyst_summary}

    Bull's Argument:
    {bull_case}

    Your task:
    Build a compelling BEAR CASE for why BTC/USDT will fall.
    COUNTER the Bull's arguments where appropriate.

    Consider:
    - Negative news signals
    - Technical resistance levels
    - Market risks
    - Fundamental concerns

    Structure:
    1. Main Thesis (2-3 sentences)
    2. Supporting Evidence (3-5 bullet points)
    3. Risk Factors
    4. Confidence Level (0.0 - 1.0)
    """)
])
```

### 4. Semantic Caching (Redis)

```python
# LLM 호출 전에 캐시 체크
def call_llm_with_cache(prompt: str, llm: ChatOpenAI) -> str:
    # 1. 프롬프트 → 임베딩 벡터
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    prompt_embedding = embedder.encode(prompt)

    # 2. Redis에서 유사 프롬프트 검색
    cached_responses = redis.search_similar(
        embedding=prompt_embedding,
        threshold=0.95,  # 95% 이상 유사하면 캐시 사용
        limit=1
    )

    if cached_responses:
        # 캐시 히트! 저장된 응답 반환
        state['cache_hits'] += 1
        logger.info("✓ Cache HIT - Saved API call!")
        return cached_responses[0]['response']

    # 3. 캐시 미스 - 실제 LLM 호출
    state['cache_misses'] += 1
    response = llm.invoke(prompt)

    # 4. Redis에 저장 (TTL 24시간)
    redis.store(
        embedding=prompt_embedding,
        response=response,
        ttl=86400
    )

    return response

# 비용 절감 효과:
# - 캐시 히트율 40-68.8%
# - GPT-4o-mini: $0.15/1M input tokens
# - 1000번 호출 → 400번 캐시 → 약 $0.06 절감
```

---

## 🎲 의사결정 흐름

### Risk Manager - 다단계 검증

```python
def risk_manager_node(state: AgentState) -> AgentState:
    """
    리스크 관리자: 거래 승인/거부 결정
    """

    # 1. 기본 정보 수집
    bull_conf = state['bull_confidence']  # 0.6
    bear_conf = state['bear_confidence']  # 0.5
    current_portfolio = state['portfolio']
    proposed_trade = state['proposed_trade']

    # 2. 신뢰도 차이 계산
    confidence_diff = abs(bull_conf - bear_conf)

    # 3. 다단계 검증 시작
    checks = []

    # ✓ Check 1: 포지션 크기 제한
    max_position = float(os.getenv('MAX_POSITION_SIZE', 0.3))  # 30%
    if proposed_trade.allocation > max_position:
        checks.append({
            'name': 'Position Size',
            'status': 'FAIL',
            'reason': f'Allocation {proposed_trade.allocation:.0%} > Max {max_position:.0%}'
        })
    else:
        checks.append({'name': 'Position Size', 'status': 'PASS'})

    # ✓ Check 2: 일일 손실 제한
    daily_loss_pct = calculate_daily_loss(current_portfolio)
    max_daily_loss = float(os.getenv('MAX_DAILY_LOSS', 0.05))  # 5%
    if daily_loss_pct > max_daily_loss:
        checks.append({
            'name': 'Daily Loss Limit',
            'status': 'FAIL',
            'reason': f'Loss {daily_loss_pct:.1%} > Max {max_daily_loss:.1%}'
        })
    else:
        checks.append({'name': 'Daily Loss Limit', 'status': 'PASS'})

    # ✓ Check 3: 신뢰도 임계값
    min_confidence_diff = 0.1  # Bull과 Bear 차이가 10% 이상이어야 거래
    if confidence_diff < min_confidence_diff:
        checks.append({
            'name': 'Confidence Threshold',
            'status': 'FAIL',
            'reason': f'Diff {confidence_diff:.2f} < Min {min_confidence_diff}'
        })
    else:
        checks.append({'name': 'Confidence Threshold', 'status': 'PASS'})

    # ✓ Check 4: Guardrails AI 검증
    try:
        from guardrails import Guard

        guard = Guard.from_rail_string("""
        <rail version="0.1">
        <output>
            <object name="trade">
                <string name="action" validators="valid-choices: choices=['BUY', 'SELL', 'HOLD']"/>
                <float name="allocation" validators="valid-range: min=0.0 max=1.0"/>
                <float name="confidence" validators="valid-range: min=0.0 max=1.0"/>
            </object>
        </output>
        </rail>
        """)

        validated_trade = guard.validate(proposed_trade)
        checks.append({'name': 'Guardrails Validation', 'status': 'PASS'})

    except Exception as e:
        checks.append({
            'name': 'Guardrails Validation',
            'status': 'FAIL',
            'reason': str(e)
        })

    # 4. 최종 결정
    all_passed = all(check['status'] == 'PASS' for check in checks)

    if all_passed:
        decision = 'APPROVED'
        risk_score = 0.3  # 낮음
        feedback = "All risk checks passed. Trade approved with monitoring."
    else:
        decision = 'REJECTED'
        risk_score = 0.8  # 높음
        failed_checks = [c['name'] for c in checks if c['status'] == 'FAIL']
        feedback = f"Trade rejected. Failed checks: {', '.join(failed_checks)}"

    # 5. 상태 업데이트
    state['risk_assessment'] = {
        'approved': decision == 'APPROVED',
        'risk_score': risk_score,
        'checks': checks,
        'feedback': feedback
    }

    add_reasoning_step(
        state,
        role='Risk_Manager',
        content=feedback,
        confidence=0.8
    )

    return state
```

---

## 🔬 백테스팅 시스템

### Lumibot 백테스트 워크플로우

```python
from lumibot.strategies import Strategy
from lumibot.backtesting import YahooDataBacktesting

class HATSStrategy(Strategy):
    """
    HATS Trading Agent를 Lumibot Strategy로 래핑
    """

    def initialize(self):
        """초기화"""
        self.agent_app = compile_trading_graph()
        self.sleeptime = "1H"  # 1시간마다 실행

    def on_trading_iteration(self):
        """매 시간마다 호출되는 메인 로직"""

        # 1. 현재 시장 데이터 수집
        current_price = self.get_last_price("BTC-USD")
        historical_data = self.get_historical_prices("BTC-USD", 168, "hour")

        # 2. AgentState 생성
        market_data = MarketData(
            timestamp=self.get_datetime(),
            symbol='BTC/USDT',
            current_price=current_price,
            # ... 기타 필드
        )

        initial_state = create_initial_state(
            session_id=f"backtest_{self.get_datetime()}",
            market_data=market_data
        )

        # 3. Agent 실행
        final_state = self.agent_app.invoke(initial_state)

        # 4. 거래 결정 실행
        decision = final_state['final_decision']
        proposed_trade = final_state['proposed_trade']

        if decision == 'BUY':
            # 매수
            quantity = self.portfolio_value * proposed_trade.allocation / current_price
            order = self.create_order("BTC-USD", quantity, "buy")
            self.submit_order(order)

            # 스탑로스 설정
            stop_loss_price = current_price * (1 - proposed_trade.stop_loss_pct)
            self.set_stop_loss(stop_loss_price)

        elif decision == 'SELL':
            # 매도
            position = self.get_position("BTC-USD")
            if position:
                order = self.create_order("BTC-USD", position.quantity, "sell")
                self.submit_order(order)

        # HOLD는 아무 것도 안 함

        # 5. 결과 로깅
        self.log_message(f"Decision: {decision}, Confidence: {proposed_trade.confidence}")

# 백테스트 실행
strategy = HATSStrategy()

results = strategy.backtest(
    YahooDataBacktesting,
    start_date=datetime(2024, 11, 26),
    end_date=datetime(2025, 11, 26),
    parameters={
        "symbol": "BTC-USD",
        "initial_capital": 10000
    }
)

# 결과 분석
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
```

### 백테스트 결과 저장

```sql
-- PostgreSQL: backtest_results 테이블
INSERT INTO backtest_results (
    run_id,
    strategy_name,
    start_date,
    end_date,
    initial_capital,
    final_capital,
    total_return,
    sharpe_ratio,
    max_drawdown,
    win_rate,
    total_trades,
    avg_trade_duration
) VALUES (
    'bt_20251126_001',
    'HATS_Agent_v1.0',
    '2024-11-26',
    '2025-11-26',
    10000.00,
    12500.00,
    0.25,  -- 25% return
    1.8,   -- Sharpe Ratio
    0.15,  -- 15% max drawdown
    0.58,  -- 58% win rate
    120,   -- 120 trades
    '3 days'
);
```

---

## 📊 시각화 시스템

### Landscape of Thoughts

```python
from sentence_transformers import SentenceTransformer
import umap
import plotly.graph_objects as go

def visualize_reasoning_landscape(reasoning_trace: List[ReasoningStep]):
    """
    추론 과정을 2D/3D 공간에 시각화
    """

    # 1. 각 추론 단계 → 임베딩 벡터
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    texts = [step['content'] for step in reasoning_trace]
    embeddings = embedder.encode(texts)

    # 2. UMAP으로 차원 축소 (384차원 → 2차원)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=5,
        min_dist=0.1,
        metric='cosine'
    )

    reduced_embeddings = reducer.fit_transform(embeddings)

    # 3. Plotly로 시각화
    fig = go.Figure()

    for i, step in enumerate(reasoning_trace):
        fig.add_trace(go.Scatter(
            x=[reduced_embeddings[i, 0]],
            y=[reduced_embeddings[i, 1]],
            mode='markers+text',
            name=step['role'],
            text=f"{step['role']} (conf: {step['confidence']:.2f})",
            marker=dict(
                size=step['confidence'] * 30,  # 신뢰도에 비례한 크기
                color=step['confidence'],
                colorscale='Viridis',
                showscale=True
            )
        ))

    # 4. 추론 경로 연결
    for i in range(len(reasoning_trace) - 1):
        fig.add_trace(go.Scatter(
            x=[reduced_embeddings[i, 0], reduced_embeddings[i+1, 0]],
            y=[reduced_embeddings[i, 1], reduced_embeddings[i+1, 1]],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False
        ))

    fig.update_layout(
        title='Landscape of Thoughts - HATS Agent Reasoning',
        xaxis_title='Thought Dimension 1',
        yaxis_title='Thought Dimension 2',
        hovermode='closest'
    )

    return fig

# Streamlit 대시보드에서 표시
st.plotly_chart(fig)
```

---

## 🔄 전체 시스템 통합

### 실시간 거래 시스템 (향후 Phase 7)

```python
# 1시간마다 실행되는 스케줄러
import schedule
import time

def run_trading_agent():
    """HATS Agent 1회 실행"""

    # 1. 최신 데이터 수집
    collect_latest_ohlcv()
    collect_latest_news()

    # 2. Agent 실행
    session_id = f"live_{datetime.now().isoformat()}"
    market_data = fetch_latest_market_data()

    initial_state = create_initial_state(session_id, market_data)

    # Checkpointing 활성화
    checkpointer = MemorySaver()
    app = compile_trading_graph(checkpointer=checkpointer)

    # Tracing 활성화
    with TracingContext(session_id=session_id, backtest_mode=False):
        final_state = app.invoke(
            initial_state,
            config={"configurable": {"thread_id": session_id}}
        )

    # 3. 거래 실행 (실거래 모드)
    if final_state['risk_assessment']['approved']:
        execute_real_trade(final_state['proposed_trade'])

    # 4. 결과 저장
    save_to_mongodb(final_state)

    logger.info(f"✓ Agent execution complete: {final_state['final_decision']}")

# 스케줄 설정
schedule.every().hour.at(":00").do(run_trading_agent)

# 무한 루프 실행
while True:
    schedule.run_pending()
    time.sleep(60)  # 1분마다 체크
```

---

## 🎯 결론

### HATS Trading System의 핵심 차별점

1. **Multi-Agent 협업**
   - 5개 전문 Agent가 각자의 역할 수행
   - 변증법적 추론으로 균형잡힌 의사결정

2. **완전한 투명성**
   - 모든 추론 과정 기록 (Chain-of-Thought)
   - 의사결정 근거 추적 가능
   - Landscape of Thoughts 시각화

3. **프로덕션 준비**
   - 체크포인팅으로 상태 복구
   - LangSmith 트레이싱으로 모니터링
   - Guardrails로 안전성 보장

4. **비용 최적화**
   - Semantic Caching으로 40-68.8% 절감
   - GPT-4o-mini 사용으로 저비용 운영

5. **확장 가능성**
   - 멀티 심볼 지원 가능
   - 다양한 거래 전략 적용
   - 온체인 데이터 통합 용이

---

**다음 읽을 문서:**
- `PHASE2_COMPLETE.md` - Agent 구현 상세
- `DATA_COLLECTION_COMPLETE.md` - 데이터 수집 결과
- `PROJECT_PLAN.md` - 전체 로드맵

**문의:**
- GitHub Issues
- 프로젝트 폴더 내 README 파일들 참조
