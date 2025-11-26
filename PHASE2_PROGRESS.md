# Phase 2: LangGraph Agent Foundation - 진행 상황

## 완료된 작업 (2025-11-26)

### ✅ Phase 2.1: AgentState TypedDict 정의

**파일:** `backend/agents/state.py`

**구현 내용:**
- 완전한 `AgentState` TypedDict 정의 (모든 노드 간 공유되는 상태)
- 하위 TypedDict 정의:
  - `TechnicalIndicators` - RSI, MACD, 볼린저 밴드 등
  - `MarketData` - 현재 시장 데이터
  - `NewsItem` - 뉴스 + 감성 분석
  - `ReasoningStep` - Chain-of-Thought 추론 단계
  - `DebateMessage` - Bull vs Bear 토론 메시지
  - `ProposedTrade` - 거래 제안
  - `RiskAssessment` - 리스크 평가
  - `PortfolioState` - 포트폴리오 상태

**핵심 기능:**
```python
# 초기 상태 생성
initial_state = create_initial_state(session_id, market_data)

# 추론 단계 추가
add_reasoning_step(state, role='Analyst', content='분석 내용', confidence=0.7)

# 토론 메시지 추가
add_debate_message(state, role='Bull', content='불 케이스', evidence=['증거1', '증거2'])
```

**상태 필드 (총 30개):**
1. **워크플로 제어** - current_node, iteration, should_continue, error
2. **시장 데이터** - market_data, technical_indicators, recent_news, historical_prices
3. **분석 결과** - fundamental_analysis, technical_analysis, sentiment_analysis
4. **추론 과정** - reasoning_trace (Chain-of-Thought), debate_transcript
5. **연구 결과** - bull_case, bear_case, confidence 점수
6. **거래 결정** - proposed_trade, risk_assessment, final_decision
7. **포트폴리오** - cash_balance, btc_balance, pnl, position
8. **메타데이터** - session_id, thread_id, timestamps
9. **LLM 추적** - api_calls_count, tokens_used, cache_hits/misses

---

### ✅ Phase 2.2: LangGraph 기본 그래프 구조 생성

**파일:** `backend/agents/graph.py`

**구현 내용:**
- LangGraph `StateGraph` 생성 및 컴파일
- 5개 노드로 구성된 워크플로:
  1. **Analyst** (분석가)
  2. **Bull Researcher** (불 연구원)
  3. **Bear Researcher** (베어 연구원)
  4. **Risk Manager** (리스크 관리자)
  5. **Trader** (거래 실행자)

**그래프 구조:**
```
START
  ↓
Analyst (시장 분석)
  ↓
  ├─→ [조건부] 연구 필요?
  │     YES: Bull Researcher
  │     NO:  END
  ↓
Bull Researcher (불 케이스)
  ↓
Bear Researcher (베어 케이스)
  ↓
Risk Manager (리스크 평가)
  ↓
  ├─→ [조건부] 승인?
  │     YES: Trader
  │     NO:  END (거래 거부)
  ↓
Trader (거래 실행)
  ↓
END
```

**조건부 엣지:**
1. `should_continue_research()` - 분석 후 연구 단계 진행 여부
2. `should_execute_trade()` - 리스크 관리자 승인 여부

**컴파일 옵션:**
```python
# 체크포인팅 없이 (테스트용)
app = compile_trading_graph(checkpointer=None)

# 체크포인팅 활성화 (프로덕션용)
from langgraph.checkpoint.memory import MemorySaver
app = compile_trading_graph(checkpointer=MemorySaver())
```

---

### ✅ Phase 2.3: Analyst 노드 구현

**파일:** `backend/agents/nodes.py`

**구현 내용:**

#### 1. Analyst Node (완전 구현)
- PostgreSQL에서 최근 OHLCV 데이터 조회 (1주일)
- MongoDB에서 최근 뉴스 조회 (3일, 최대 20개)
- 뉴스 감성 점수 집계
- GPT-4o-mini를 사용한 시장 분석:
  - **Fundamental Analysis** (뉴스 + 감성)
  - **Technical Analysis** (가격 추세)
  - **Key Concerns** (주요 리스크)
- 추론 트레이스에 분석 결과 기록

**사용 기술:**
- LangChain `ChatOpenAI` / `ChatAnthropic`
- `ChatPromptTemplate` + `StrOutputParser`
- PostgreSQL 직접 쿼리 (psycopg2)
- MongoDB 쿼리 (pymongo)

**LLM 프롬프트 구조:**
```python
System: "You are a cryptocurrency market analyst..."
User: """
Current Market Data:
- Symbol: {symbol}
- Current Price: ${current_price}
- 24h Change: {price_change}%

Recent News (Avg Sentiment: {sentiment}):
{news_summary}

Provide:
1. Fundamental Analysis
2. Technical Analysis
3. Key Concerns
"""
```

#### 2. 다른 노드 (기본 구현)
- **Bull Researcher** - 불 케이스 생성 (플레이스홀더)
- **Bear Researcher** - 베어 케이스 생성 (플레이스홀더)
- **Risk Manager** - 리스크 평가 (간단한 승인/거부 로직)
- **Trader** - 최종 결정 기록 (실제 거래는 백테스트에서 처리)

#### 3. 유틸리티 함수
```python
# 데이터베이스 연결
get_postgres_connection()
get_mongo_connection()

# 데이터 조회
fetch_recent_ohlcv(symbol='BTC/USDT', hours=24)
fetch_recent_news(hours=24, limit=10)

# LLM 인스턴스 생성
get_llm(model="gpt-4o-mini", temperature=0.7)
```

---

## 📁 생성된 파일 구조

```
backend/
├── agents/
│   ├── __init__.py          # 패키지 초기화
│   ├── state.py             # AgentState 정의 (270 lines)
│   ├── graph.py             # LangGraph 구조 (130 lines)
│   └── nodes.py             # 노드 구현 (440 lines)
│
└── tests/
    └── test_agent_basic.py  # 기본 Agent 테스트
```

---

## 🧪 테스트 방법

### 1. LangGraph 의존성 설치
```bash
pip install langgraph langchain langchain-openai langchain-anthropic langchain-core
```

### 2. 환경 변수 설정
`.env` 파일에 추가:
```bash
# LLM API Keys
OPENAI_API_KEY=your_openai_key_here
# 또는
ANTHROPIC_API_KEY=your_anthropic_key_here

# Database (이미 설정됨)
POSTGRES_HOST=localhost
MONGO_HOST=localhost
```

### 3. 기본 Agent 테스트 실행
```bash
cd backend
python tests/test_agent_basic.py
```

**예상 출력:**
```
TESTING HATS TRADING AGENT - Basic Workflow
Session ID: abc-123-def-456
Initial state created:
  - Current price: $87,500.00
  - 24h change: +0.58%
  - Portfolio cash: $10,000.00

Compiling agent graph...
✓ Graph compiled successfully

Executing agent workflow...
Completed node: analyst
Completed node: bull_researcher
Completed node: bear_researcher
Completed node: risk_manager
Completed node: trader

AGENT EXECUTION COMPLETE
Final Decision: BUY
Total Reasoning Steps: 5
API Calls: 1

Reasoning Trace:
  [Analyst] Step 1
    Market Analysis: ...
    Confidence: 0.70
  [Researcher_Bull] Step 2
    Bull Case: ...
    Confidence: 0.60
  ...

✓ Test completed successfully!
```

---

## 🔄 워크플로 예시

실제 Agent 실행 시 상태 변화:

```python
# 1. START → Analyst
state['current_node'] = 'analyst'
state['fundamental_analysis'] = "긍정적 뉴스 흐름..."
state['sentiment_score'] = 0.35
state['reasoning_trace'].append(...)
state['api_calls_count'] = 1

# 2. Analyst → Bull Researcher
state['current_node'] = 'bull_researcher'
state['bull_case'] = "상승 모멘텀..."
state['bull_confidence'] = 0.6
state['debate_transcript'].append(...)

# 3. Bull → Bear Researcher
state['current_node'] = 'bear_researcher'
state['bear_case'] = "변동성 높음..."
state['bear_confidence'] = 0.5

# 4. Bear → Risk Manager
state['current_node'] = 'risk_manager'
state['proposed_trade'] = {'action': 'BUY', 'allocation': 0.1, ...}
state['risk_assessment'] = {'approved': True, 'risk_score': 0.3, ...}

# 5. Risk Manager → Trader
state['current_node'] = 'trader'
state['final_decision'] = 'BUY'
state['completed_at'] = datetime.now()
```

---

## 📊 데이터 흐름

```
PostgreSQL (OHLCV)  ─┐
                     ├─→ Analyst Node
MongoDB (News)      ─┘        ↓
                         [분석 결과]
                              ↓
                     Bull Researcher
                              ↓
                     Bear Researcher
                              ↓
                      Risk Manager
                              ↓
                          Trader
                              ↓
                     [Final Decision]
```

---

## ⏭️ 다음 단계: Phase 2.4-2.5

### Phase 2.4: 체크포인팅 시스템
- MongoDB를 활용한 `MongoDBSaver` 구현
- 상태 저장 및 복원 기능
- 중단된 워크플로 재개

### Phase 2.5: LangSmith 트레이싱
- LangSmith API 연동
- Agent 실행 추적
- 디버깅 및 성능 분석

---

## 💡 주요 설계 결정

1. **TypedDict 사용**
   - 타입 안정성 확보
   - IDE 자동완성 지원
   - 런타임 오버헤드 없음

2. **조건부 엣지**
   - 동적 워크플로 제어
   - 불필요한 노드 실행 방지
   - 리스크 기반 거래 거부

3. **추론 트레이스**
   - 모든 결정 과정 기록
   - 디버깅 및 감사 가능
   - Landscape of Thoughts 시각화 준비

4. **하이브리드 데이터베이스**
   - PostgreSQL: 시계열 OHLCV
   - MongoDB: 비정형 뉴스 + 추론 로그
   - 각 DB의 강점 활용

---

**Phase 2 진행률: 60% (3/5 완료)**

다음: Phase 2.4 체크포인팅 시스템 구현
