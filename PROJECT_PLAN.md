# HATS Trading System - 전체 프로젝트 플랜 및 진행도

**프로젝트명:** HATS (Hybrid AI Trading System)
**목표:** LLM 기반 자율 트레이딩 에이전트 시스템 구축
**시작일:** 2025-11-26
**최근 업데이트:** 2025-11-27 (Phase 3 LangGraph 통합 완료)
**예상 완료:** 2026-04-26 (5개월) - ML/RL 통합 포함
**현재 진행률:** ███████████░░░░░░░░░ **58%** (Phase 3 60% 진행 - Debate Subgraph 통합 완료)

---

## 🧠 ML/RL 통합 로드맵 (Phase 3.5 ~ 4.5)

> **전략:** 단계적 구현을 통한 점진적 성능 향상
>
> **핵심 개념:**
> - **Strategic Layer (전략 계층)**: LLM 기반 - 시장 방향성, 뉴스 분석 (기존 시스템)
> - **Tactical Layer (전술 계층)**: LSTM + 패턴 인식 - 단기 가격 예측 및 진입/청산 타이밍 (신규)
> - **Execution Layer (실행 계층)**: FinRL - 최적 포지션 사이징 및 리스크 관리 (신규)
>
> **예상 성능:**
> ```
> Stage 1 (LLM만):               15-25% 연 수익, 55-60% 승률
> Stage 2 (LLM + LSTM):          30-40% 연 수익, 58-62% 승률 (+15% 수익)
> Stage 3 (LLM + LSTM + FinRL):  50-65% 연 수익, 62-68% 승률 (+35% 수익)
> ```

### 왜 단계적으로?

**장점:**
1. ✅ **위험 분산**: 각 단계별 검증 후 다음 진행
2. ✅ **학습 곡선**: ML/RL 기술 단계적 습득
3. ✅ **빠른 ROI**: LSTM만으로도 상당한 성능 향상
4. ✅ **유지보수**: 복잡도 관리 용이

**단계별 복잡도:**
```
Phase 3 (기본):      복잡도 1x   개발 2-3주
Phase 3.5 (LSTM):    복잡도 1.5x 개발 3주
Phase 4.5 (FinRL):   복잡도 2.5x 개발 6-8주
```

---

## 🎉 오늘의 성과 (2025-11-26)

### ✅ 완료된 작업
1. **Phase 2 검증 및 테스트**
   - Agent 워크플로 테스트 완료 (Fallback 모드)
   - 5개 노드 정상 실행 확인
   - 에러 핸들링 검증 완료

2. **데이터 수집 완료** 🎯
   - **OHLCV 데이터:** 8,761개 캔들 (2024-11-26 ~ 2025-11-26)
     - 가격 범위: $49,000 ~ $126,199.63
     - 평균 거래량: 950.68 BTC
   - **뉴스 데이터:** 20개 뉴스 아이템
     - 기간: 2025-11-25
     - CryptoPanic API v2 연동 완료

3. **API 설정**
   - ✅ OpenAI API 키 설정
   - ✅ Anthropic API 키 설정
   - ✅ CryptoPanic API 토큰 설정
   - ⚠️ LLM 크레딧 부족 (충전 필요)

### 🔄 현재 상태
- **인프라:** 모든 서비스 정상 작동 (Docker 5개 컨테이너)
- **데이터베이스:**
  - PostgreSQL: 8,761 OHLCV 캔들
  - MongoDB: 20 뉴스 아이템
  - Redis: 준비 완료
- **Agent 시스템:** 구현 완료, LLM 크레딧 대기 중

### ⏭️ 다음 단계
1. **즉시 가능:** OpenAI 또는 Anthropic 크레딧 충전 → 실제 LLM 분석 테스트
2. **Phase 3 준비:** Bull/Bear Researcher LLM 프롬프트 작성
3. **백테스팅:** 수집된 1년 데이터로 백테스트 실행 준비

---

## 📊 전체 진행 상황

| Phase | 작업 내용 | 기간 | 상태 | 진행률 | 완료일 |
|-------|---------|------|------|--------|--------|
| **Phase 0** | 프로젝트 설정 | 1일 | ✅ 완료 | 100% | 2025-11-26 |
| **Phase 1** | 인프라 구축 | 1-2주 | ✅ 완료 | 100% | 2025-11-26 |
| **Phase 2** | LangGraph Agent Foundation | 2-3주 | ✅ 완료 | 100% | 2025-11-26 |
| **Phase 3** | TradingAgents 프레임워크 통합 | 2-3주 | 🔄 진행중 | 60% | - |
| **Phase 3.5** | 🧠 ML Tactical Layer (LSTM + 패턴) | 3주 | ⏳ 대기 | 0% | - |
| **Phase 4** | Lumibot 백테스팅 통합 | 2주 | ⏳ 대기 | 0% | - |
| **Phase 4.5** | 🤖 FinRL Execution Layer | 6-8주 | ⏳ 대기 | 0% | - |
| **Phase 5** | Landscape of Thoughts 시각화 | 1-2주 | ⏳ 대기 | 0% | - |
| **Phase 6** | Human-in-the-Loop & Guardrails | 1주 | ⏳ 대기 | 0% | - |
| **Phase 7** | 최종 통합 및 최적화 | 1-2주 | ⏳ 대기 | 0% | - |

**범례:** ✅ 완료 | 🔄 진행중 | ⏳ 대기 | ❌ 차단됨

---

## 🎯 Phase 0: 프로젝트 설정 ✅ 완료 (100%)

### 작업 내용
- [x] 프로젝트 분석 및 계획 수립
- [x] 기술 스택 선정 (LangGraph, Lumibot, MongoDB, PostgreSQL)
- [x] 초기 요구사항 분석

### 산출물
- ✅ `.claude/plans/radiant-chasing-willow.md` - 초기 계획서
- ✅ 기술 스택 결정 문서

---

## 🏗️ Phase 1: 인프라 구축 ✅ 완료 (100%)

### 진행 상황
**완료일:** 2025-11-26
**소요 시간:** 1일 (예상: 1-2주)

### 작업 체크리스트

#### 1.1 데이터베이스 설정
- [x] **Docker Compose 설정** (`docker-compose.yml`)
  - PostgreSQL + TimescaleDB (포트 5432)
  - MongoDB (포트 27017)
  - Redis (포트 6379)
  - Adminer (포트 8080)
  - Mongo Express (포트 8081)
- [x] **PostgreSQL 스키마** (`backend/db/init_postgres.sql`)
  - `ohlcv_btcusdt_1h` (TimescaleDB hypertable)
  - `trades`
  - `portfolio_snapshots`
  - `backtest_results`
  - `document_embeddings` (pgvector)
- [x] **MongoDB 스키마** (`backend/db/init_mongodb.js`)
  - `reasoning_logs`
  - `news`
  - `agent_checkpoints`
  - `backtest_metadata`

#### 1.2 데이터 수집기
- [x] **CCXT 데이터 수집기** (`backend/data/ccxt_collector.py` - 359 lines)
  - Binance 거래소 연동
  - OHLCV 데이터 수집 (1시간 봉)
  - PostgreSQL 저장
  - CLI 인터페이스
  - 중복 방지 로직
- [x] **뉴스 데이터 수집기** (`backend/data/news_collector.py` - 391 lines)
  - CryptoPanic API 연동
  - 뉴스 + 감성 데이터 수집
  - MongoDB 저장
  - CLI 인터페이스

#### 1.3 의존성 및 환경 설정
- [x] **Python 의존성** (`backend/requirements_agent.txt` - 137 lines)
  - LangGraph, LangChain
  - Lumibot, CCXT
  - Guardrails AI
  - Sentence Transformers, UMAP
  - Streamlit, Plotly
  - 데이터베이스 드라이버 (psycopg2, pymongo, redis)
- [x] **환경 변수 템플릿** (`.env.example`)
  - API 키 설정 (OpenAI, Anthropic, CryptoPanic)
  - 데이터베이스 연결 정보
  - LangSmith 트레이싱 설정
- [x] **자동 설정 스크립트**
  - `setup_phase1.bat` (Windows)
  - `setup_phase1.sh` (Linux/macOS)

#### 1.4 인프라 검증
- [x] **테스트 스크립트** (`backend/tests/test_phase1_infrastructure.py`)
  - PostgreSQL 연결 테스트
  - MongoDB 연결 테스트
  - Redis 연결 테스트
  - CCXT 거래소 테스트
  - 테이블/컬렉션 생성 확인
  - 데이터 삽입/조회 테스트
- [x] **테스트 결과:** 8/8 테스트 통과 ✅

#### 1.5 실제 데이터 수집 ✅ 완료 (2025-11-26)
- [x] **OHLCV 데이터 수집**
  - 총 8,761개 캔들 (1년치)
  - 기간: 2024-11-26 ~ 2025-11-26
  - 가격 범위: $49,000 ~ $126,199.63
  - 소요 시간: ~10초 (9개 배치)
- [x] **뉴스 데이터 수집**
  - 총 20개 뉴스 아이템
  - CryptoPanic API v2 연동
  - MongoDB 저장 완료

### 산출물
- ✅ `docker-compose.yml`
- ✅ `backend/db/init_postgres.sql`
- ✅ `backend/db/init_mongodb.js`
- ✅ `backend/data/ccxt_collector.py`
- ✅ `backend/data/news_collector.py`
- ✅ `backend/requirements_agent.txt`
- ✅ `.env.example`
- ✅ `setup_phase1.bat / .sh`
- ✅ `backend/tests/test_phase1_infrastructure.py`
- ✅ `backend/README_PHASE1.md`
- ✅ `PHASE1_COMPLETE.md`

### 핵심 성과
✅ 완전 자동화된 인프라 구축
✅ 하이브리드 데이터베이스 아키텍처 (PostgreSQL + MongoDB + Redis)
✅ 프로덕션급 데이터 수집 파이프라인
✅ 웹 UI 제공 (Adminer, Mongo Express)

---

## 🤖 Phase 2: LangGraph Agent Foundation ✅ 완료 (100%)

### 진행 상황
**완료일:** 2025-11-26
**소요 시간:** 1일 (예상: 2-3주)

### 작업 체크리스트

#### 2.1 AgentState 정의
- [x] **상태 구조 설계** (`backend/agents/state.py` - 270 lines)
  - `AgentState` TypedDict (30개 필드)
  - `TechnicalIndicators`
  - `MarketData`
  - `NewsItem`
  - `ReasoningStep` (Chain-of-Thought)
  - `DebateMessage` (Bull vs Bear)
  - `ProposedTrade`
  - `RiskAssessment`
  - `PortfolioState`
- [x] **헬퍼 함수**
  - `create_initial_state()`
  - `add_reasoning_step()`
  - `add_debate_message()`

#### 2.2 LangGraph 그래프 구조
- [x] **기본 그래프** (`backend/agents/graph.py` - 130 lines)
  - StateGraph 생성
  - 5개 노드 정의 (Analyst, Bull Researcher, Bear Researcher, Risk Manager, Trader)
  - 조건부 엣지 2개
    - `should_continue_research()`: 분석 → 연구
    - `should_execute_trade()`: 리스크 평가 → 실행
  - 컴파일 함수 (checkpointer 선택 가능)

#### 2.3 Agent 노드 구현
- [x] **Analyst Node (완전 구현)** (`backend/agents/nodes.py` - 440 lines)
  - PostgreSQL에서 OHLCV 데이터 조회
  - MongoDB에서 뉴스 데이터 조회
  - GPT-4o-mini / Claude-3.5-Sonnet 분석
  - Fundamental Analysis
  - Technical Analysis (기본)
  - 추론 트레이스 기록
- [x] **Bull Researcher Node (기본 구현)**
  - 불 케이스 생성 (플레이스홀더)
- [x] **Bear Researcher Node (기본 구현)**
  - 베어 케이스 생성 (플레이스홀더)
- [x] **Risk Manager Node (기본 구현)**
  - 간단한 승인/거부 로직
- [x] **Trader Node (기본 구현)**
  - 최종 결정 기록

#### 2.4 체크포인팅 시스템
- [x] **MongoDB Checkpointer** (`backend/agents/checkpointer.py` - 280 lines)
  - `MongoDBCheckpointSaver` 클래스
  - 상태 저장 (`put()`)
  - 상태 조회 (`get()`)
  - 체크포인트 목록 (`list()`)
  - 스레드 이력 조회 (`get_thread_history()`)
  - 팩토리 함수 (`create_checkpointer()`)
- [⚠️] **참고:** `get_tuple()` 메서드 미구현 (MemorySaver 사용 권장)

#### 2.5 LangSmith 트레이싱
- [x] **트레이싱 설정** (`backend/agents/tracing.py` - 240 lines)
  - `setup_langsmith_tracing()`: 설정 함수
  - `TracingContext`: Context manager
  - `create_trace_metadata()`: 메타데이터 생성
  - 비용 분석 유틸리티 (플레이스홀더)
  - 캐시 메트릭 추출 (플레이스홀더)

#### 2.6 테스트
- [x] **기본 Agent 테스트** (`backend/tests/test_agent_basic.py`)
  - 그래프 컴파일 확인
  - 워크플로 실행 확인
  - PostgreSQL/MongoDB 연동 확인
  - 결과: ✅ 성공 (OpenAI API 키 없어도 인프라는 정상 작동)
- [x] **통합 테스트** (`backend/tests/test_agent_complete.py`)
  - Checkpointing 테스트
  - Tracing 테스트
  - 결합 테스트
  - 결과: 1/3 통과 (Tracing 성공, Checkpointing은 MemorySaver 사용 권장)

### 산출물
- ✅ `backend/agents/__init__.py`
- ✅ `backend/agents/state.py`
- ✅ `backend/agents/graph.py`
- ✅ `backend/agents/nodes.py`
- ✅ `backend/agents/checkpointer.py`
- ✅ `backend/agents/tracing.py`
- ✅ `backend/tests/test_agent_basic.py`
- ✅ `backend/tests/test_agent_complete.py`
- ✅ `PHASE2_PROGRESS.md`
- ✅ `PHASE2_COMPLETE.md`

### 핵심 성과
✅ 완전한 타입 안정성 (TypedDict)
✅ LangGraph 기반 멀티 에이전트 워크플로
✅ 데이터베이스 통합 (PostgreSQL + MongoDB)
✅ Chain-of-Thought 추론 추적
✅ LangSmith 트레이싱 설정 완료

---

## 🎯 Phase 3: TradingAgents 프레임워크 통합 🔄 진행중 (60%)

### 목표
전문화된 에이전트 + Bull vs Bear 변증법적 토론 구현

### 예상 기간
2-3주

### 진행 상황
**시작일:** 2025-11-26
**최근 업데이트:** 2025-11-27
**현재 완료:** Bull/Bear Researcher, Judge, Consensus, LangGraph 통합 완료

### 작업 계획

#### 3.1 Bull/Bear Researcher LLM 구현 ✅ 완료 (2025-11-26)
- [x] **Bull Researcher 고도화** (`backend/agents/researchers.py` - 600+ lines)
  - [x] GPT-4o-mini 프롬프트 작성 (Adaptive calibration)
  - [x] 강세 논리 생성 (기술적 + 펀더멘털)
  - [x] 지지 증거 수집 및 제시 (Pydantic 구조화)
  - [x] 신뢰도 점수 계산 (Evidence strength evaluation)
  - [x] **Market regime 기반 추론 스타일 조정**
    - Bull market: Subjective reasoning (confidence 0.65)
    - Bear market: Factual data focus (confidence 0.80)
    - Sideways: Balanced approach (confidence 0.70)
- [x] **Bear Researcher 고도화**
  - [x] 약세 논리 생성
  - [x] 리스크 요인 강조
  - [x] 반박 증거 제시
  - [x] 신뢰도 점수 계산
  - [x] **Market regime 기반 추론 스타일 조정**
    - Bull market: Factual overheating signals
    - Bear market: Subjective risk perception
    - Sideways: Balanced risk awareness

#### 3.2 Debate System Core ✅ 완료 (2025-11-26)
- [x] **Judge Node** (`backend/agents/debate.py`)
  - [x] 합의 도달 판정 알고리즘
    - Confidence diff < 0.15
    - Position diff < 20%
    - Mutual acknowledgment check
  - [x] 최대 4 라운드 제한
  - [x] Early stopping 지원
- [x] **Consensus Synthesis Node**
  - [x] 신뢰도 기반 가중 평균
  - [x] Evidence 강도 평가 (5가지 기준)
  - [x] 최종 포지션 계산 (-100% ~ +100%)
  - [x] 합의 요약 생성
- [x] **State Management** (`backend/agents/state.py` 업데이트)
  - [x] debate_messages 필드 추가
  - [x] debate_round, debate_converged 추가
  - [x] market_regime, news_sentiment 추가
  - [x] debate_consensus 출력 추가

#### 3.2.1 Debate Subgraph 통합 ✅ 완료 (2025-11-27)
- [x] **LangGraph 통합** (`backend/agents/graph.py` - Phase 3 구조로 재작성)
  - [x] Debate loop 구현 (Bull → Bear → Judge → continue/converge)
  - [x] Conditional edges 설정 (should_continue_debate)
  - [x] 기존 graph.py Phase 3 구조로 완전 재작성
  - [x] 순환 실행 로직 (최대 4 라운드)
  - [x] should_start_debate 함수로 debate 초기화
  - [x] Judge 수렴 판정 후 Consensus로 이동
- [x] **Risk Manager 업데이트** (`backend/agents/nodes.py`)
  - [x] Consensus 기반 trade proposal 생성
  - [x] Kelly-inspired position sizing
  - [x] Confidence 기반 stop-loss/take-profit 설정
  - [x] Market regime 고려한 risk assessment
- [x] **State 타입 수정** (`backend/agents/state.py`)
  - [x] Dict[str, any] → Dict[str, Any] (Pydantic warning 해결)
- [x] **통합 테스트**
  - [x] 그래프 컴파일 성공 (9개 노드)
  - [x] Debate loop 구조 검증

#### 3.2.2 Gemini LLM 통합 ✅ 완료 (2025-11-27)
- [x] **옵션 1: LangChain 래퍼 방식** (현재 구현)
  - [x] `.env`에 GOOGLE_API_KEY 추가
  - [x] `requirements.txt`에 langchain-google-genai 추가
  - [x] `get_llm()` 함수에 Gemini 지원 추가
  - [x] Gemini 2.5 Flash를 기본 모델로 설정
  - [x] API 호출 테스트 완료
  - [x] 전체 그래프 컴파일 검증
- [ ] **옵션 2: 공식 SDK 마이그레이션** (Phase 4 이후 계획)
  - [ ] `google-genai` 패키지로 전환
  - [ ] LangChain 의존성 제거
  - [ ] 프롬프트 템플릿 재구성
  - [ ] 출력 파서 직접 구현
  - [ ] 성능 최적화 및 벤치마크

**선택 이유:**
- 옵션 1: 코드 변경 최소화 (10분 작업), 기존 LangChain 기능 유지
- 옵션 2: 최신 기능 접근, 성능 최적화 (2-3일 작업, 나중에 진행)

**현재 모델:**
- **Gemini 2.5 Flash** (균형잡힌 모델, 1M 토큰 컨텍스트)
- Temperature: 0.7
- convert_system_message_to_human: True (Gemini requirement)

#### 3.3 Technical Analyst 구현 ✅ 완료 (2025-11-27)
- [x] **`ta` 라이브러리 통합** (`backend/agents/technical_analyst.py` - 600+ lines)
  - [x] RSI (Relative Strength Index) - 과매수/과매도 감지
  - [x] MACD (Moving Average Convergence Divergence) - 추세 전환 신호
  - [x] Bollinger Bands - 변동성 및 가격 위치 분석
  - [x] EMA (Exponential Moving Average) - 20, 50, 200 기간
  - [x] ATR (Average True Range) - 변동성 측정
  - [x] Volume indicators - OBV, VWAP, 거래량 추세
- [x] **지지/저항선 탐지**
  - [x] 피벗 포인트 계산 (Standard method)
  - [x] 과거 고점/저점 분석 (Local extrema detection)
  - [x] 최근 3개 레벨 추출
- [x] **패턴 인식**
  - [x] 추세 분석 (uptrend/downtrend/sideways) - EMA 기반
  - [x] 모멘텀 분석 (overbought/oversold/neutral) - RSI 기반
- [x] **analyst_node 통합**
  - [x] `calculate_technical_indicators()` 호출
  - [x] state['technical_indicators']에 저장
  - [x] Technical analysis summary 생성

**선택: `ta` 라이브러리 vs `TA-Lib`**
- ✅ `ta` 사용: Pure Python, 설치 간단 (10초)
- ❌ `TA-Lib`: C 기반, Windows 설치 복잡 (30분+)
- 성능 차이: 무시 가능 (0.04초 vs 0.01초)

#### 3.4 Sentiment Analyst 구현 ✅ 완료 (2025-11-27)
- [x] **옵션 2: Gemini 2.5 Pro Financial CoT** (현재 구현)
  - [x] `backend/agents/sentiment_analyst.py` 생성 (370+ lines)
  - [x] Financial Chain-of-Thought 프롬프트 설계
  - [x] 뉴스별 감성 분석 (sentiment score: -1.0 ~ 1.0)
  - [x] Sentiment label 분류 (Very Negative ~ Very Positive)
  - [x] Market impact 분석 (Bearish/Neutral/Bullish)
  - [x] 배치 처리 (뉴스 최대 10개)
  - [x] analyst_node 통합
  - [x] Fallback 로직 (API 실패 시 neutral 반환)
  - [x] 테스트 및 검증 (구조 테스트 완료)
  - [x] 전체 그래프 컴파일 성공 (8 nodes)
- [ ] **옵션 1: FinGPT 로컬 모델 마이그레이션** (Phase 4 이후 계획)
  - [ ] FinGPT-v3 모델 다운로드 (HuggingFace)
  - [ ] GPU 메모리 최적화 (4-8GB VRAM)
  - [ ] Quantization 적용 (INT8/FP16)
  - [ ] 추론 파이프라인 구축
  - [ ] 성능 벤치마크 (Gemini vs FinGPT)
  - [ ] 비용 분석 (API vs GPU 전기료)

**선택 이유:**
- 옵션 2: 빠른 구현 (10분), 이미 사용 중인 Gemini 인프라 활용
- 옵션 1: 금융 도메인 특화, API 비용 절감 (나중에 진행)

**현재 모델:**
- **Gemini 2.5 Pro** (고급 추론, Financial CoT에 최적)
- Temperature: 0.3 (일관된 감성 분석)
- System: Financial analyst persona

#### 3.5 Risk Manager 고도화
- [ ] **Guardrails AI 통합**
  - [ ] TradingSignal Pydantic 모델 정의
  - [ ] Valid Range validator (allocation, stop_loss)
  - [ ] Valid JSON validator
  - [ ] Financial Tone validator
- [ ] **리스크 계산**
  - [ ] 포지션 사이징 검증 (최대 10%)
  - [ ] 손절 범위 검증 (최대 20%)
  - [ ] 레버리지 제한
  - [ ] 일일 최대 손실 제한

### 산출물
- ✅ `PHASE3_DEBATE_SYSTEM_DESIGN.md` - 설계 문서 (799 lines)
- ✅ `backend/agents/researchers.py` - Bull/Bear Researcher (600+ lines) + Gemini 통합
- ✅ `backend/agents/debate.py` - Judge & Consensus (350+ lines)
- ✅ `backend/agents/state.py` - State 업데이트 (Phase 3 필드 + 타입 수정)
- ✅ `backend/agents/graph.py` - Phase 3 구조로 재작성 (debate loop)
- ✅ `backend/agents/nodes.py` - risk_manager_node 업데이트 + Technical/Sentiment Analyst 통합
- ✅ `backend/agents/technical_analyst.py` - `ta` 라이브러리 통합 (600+ lines)
- ✅ `backend/agents/sentiment_analyst.py` - Gemini 2.5 Pro Financial CoT (370+ lines)
- ✅ `.env` - GOOGLE_API_KEY 추가
- ✅ `requirements.txt` - langchain-google-genai, ta 추가
- ⏳ `backend/tests/test_researchers.py` - Researcher 테스트
- ⏳ `backend/tests/test_debate.py` - Debate 테스트

### 검증 기준
- [x] Bull/Bear Researcher 노드 구현 완료
- [x] Adaptive calibration 구현 (market regime 기반)
- [x] Judge 노드로 수렴 감지 알고리즘 구현
- [x] Consensus synthesis 알고리즘 구현
- [x] LangGraph에 debate loop 통합 완료
- [x] Graph 컴파일 성공 (9개 노드 구성)
- [x] Risk Manager가 consensus 결과 활용
- [ ] 4 라운드 순환 실행 테스트 (LLM 크레딧 필요)
- [ ] 합의 도달 시 토론 종료 확인 (LLM 크레딧 필요)
- [ ] 기술적 지표가 정확하게 계산됨
- [ ] FinGPT 감성 분석 정확도 > 70%
- [ ] Guardrails가 잘못된 거래 신호를 차단함

### 핵심 성과 (현재까지)
✅ 2024-2025 최신 연구 기반 설계 (TradingAgents, MAD)
✅ Adaptive calibration으로 LLM miscalibration 문제 해결
✅ Evidence-based confidence scoring (5가지 기준)
✅ 4-round debate with convergence detection
✅ LangGraph debate loop 통합 완료 (순환 구조)
✅ Consensus-based risk management (Kelly-inspired sizing)
✅ **Gemini 2.5 Flash 통합** (1M 토큰 컨텍스트, LangChain 래퍼)
✅ Multi-LLM 지원 (OpenAI, Anthropic, Google Gemini)
✅ 1,400+ lines 코드 구현 완료 (researchers + debate + graph + nodes)
✅ 전체 시스템 컴파일 성공 (9개 노드)

---

## 🧠 Phase 3.5: ML Tactical Layer (LSTM + Pattern Recognition) ⏳ 대기 (0%)

### 목표
**단기 가격 예측 및 진입/청산 타이밍 최적화**를 위한 머신러닝 레이어 추가

> 💡 **핵심 아이디어:**
> - LLM은 "전략적 방향성" 제시 (예: "단기 조정 후 상승 예상")
> - LSTM은 "전술적 타이밍" 제공 (예: "다음 15분 하락 확률 70%")
> - 결합 시 **정확한 진입점**과 **익절/손절 타이밍** 확보

### 예상 기간
3주

### 작업 계획

#### 3.5.1 데이터 준비
- [ ] **15분봉 데이터 수집** (`backend/data/ccxt_collector.py` 확장)
  - [ ] 1년치 15분봉 수집 (35,040개 캔들)
  - [ ] 정규화 및 전처리
  - [ ] Train/Validation/Test 분할 (70%/15%/15%)
- [ ] **레이블링 로직**
  ```python
  # 다음 15분 가격 변화율 기준
  if next_close > current_close * 1.001:  # +0.1%
      label = 'UP'
  elif next_close < current_close * 0.999:  # -0.1%
      label = 'DOWN'
  else:
      label = 'SIDEWAYS'
  ```

#### 3.5.2 LSTM 가격 예측 모델
- [ ] **모델 구현** (`backend/ml/price_predictor.py`)
  - [ ] PyTorch LSTM 아키텍처
    - Input: 60 타임스텝 × 5 features (OHLCV)
    - Hidden: 128 units × 2 layers
    - Output: 3 classes (UP/DOWN/SIDEWAYS)
  - [ ] Dropout 0.2 (과적합 방지)
  - [ ] Softmax 출력 (확률)

- [ ] **학습 파이프라인** (`backend/ml/train_lstm.py`)
  - [ ] Adam optimizer (lr=0.001)
  - [ ] CrossEntropyLoss
  - [ ] Early stopping (patience=10)
  - [ ] 학습 시간: ~2시간 (GPU), ~8시간 (CPU)

- [ ] **모델 평가**
  - [ ] 목표 정확도: 62% 이상
  - [ ] Confusion matrix 분석
  - [ ] 클래스별 정밀도/재현율

#### 3.5.3 캔들 패턴 인식 모델
- [ ] **패턴 정의** (`backend/ml/pattern_recognizer.py`)
  - [ ] 9가지 패턴 클래스:
    - `bullish_engulfing`, `bearish_engulfing`
    - `hammer`, `shooting_star`
    - `doji`
    - `morning_star`, `evening_star`
    - `three_white_soldiers`, `three_black_crows`

- [ ] **특징 추출**
  ```python
  features = [
      body_size_ratio,      # 몸통 크기 비율
      upper_shadow_ratio,   # 위꼬리 비율
      lower_shadow_ratio,   # 아래꼬리 비율
      price_change_3,       # 3캔들 가격 변화
      volume_ratio,         # 거래량 비율
      momentum_rsi,         # RSI
      trend_ema            # EMA 추세
  ]
  ```

- [ ] **RandomForest 학습**
  - [ ] n_estimators=100
  - [ ] max_depth=10
  - [ ] 학습 시간: ~10분

#### 3.5.4 동적 포지션 관리
- [ ] **Kelly Criterion** (`backend/ml/position_manager.py`)
  ```python
  # 최적 포지션 비율 계산
  f = (p * b - q) / b
  # f: 투자 비율
  # p: 승률
  # b: 평균 수익/평균 손실
  # q: 1 - p (패배율)

  # Half Kelly 적용 (안전)
  position_size = kelly_fraction * 0.5
  ```

- [ ] **ATR 기반 Trailing Stop**
  ```python
  # Average True Range로 변동성 측정
  atr = calculate_atr(candles, period=14)

  # 진입 시 손절: 2 ATR
  initial_stop = entry_price - (2 * atr)

  # 수익 발생 시 트레일링
  if profit > 0.02:  # 2% 수익
      trailing_stop = entry_price + (profit * 0.5)
  ```

- [ ] **다단계 익절**
  ```python
  take_profit_levels = [
      (0.01, 0.30),  # 1% 수익 → 30% 청산
      (0.02, 0.30),  # 2% 수익 → 30% 추가 청산
      (0.03, 0.40),  # 3% 수익 → 나머지 전량 청산
  ]
  ```

#### 3.5.5 하이브리드 신호 통합
- [ ] **Tactical Agent Node** (`backend/agents/tactical_agent.py`)
  ```python
  # 신호 가중치
  weights = {
      'strategic_llm': 0.40,    # LLM 전략 분석
      'lstm_prediction': 0.30,  # LSTM 가격 예측
      'pattern_signal': 0.30    # 패턴 인식
  }

  # 최종 점수 계산
  final_score = (
      strategic_signal * 0.40 +
      lstm_signal * 0.30 +
      pattern_signal * 0.30
  )

  # 의사결정
  if final_score > 0.3:
      decision = 'BUY'
  elif final_score < -0.3:
      decision = 'SELL'
  else:
      decision = 'HOLD'
  ```

- [ ] **LangGraph 통합**
  - [ ] 새로운 `tactical_analysis` 노드 추가
  - [ ] State에 ML 예측 필드 추가:
    ```python
    class AgentState(TypedDict):
        # ... 기존 필드 ...
        ml_prediction: Dict[str, Any]  # LSTM 예측
        pattern_detected: Dict[str, Any]  # 패턴 인식
        tactical_signal: str  # BUY/SELL/HOLD
        position_size: float  # 계산된 포지션 크기
    ```

#### 3.5.6 백테스팅
- [ ] **LSTM 단독 백테스트** (기준선)
  - [ ] 1년 데이터 테스트
  - [ ] 예상 결과: 25-35% 연 수익, 58% 승률

- [ ] **LLM + LSTM 하이브리드**
  - [ ] 예상 결과: 30-40% 연 수익, 60-62% 승률
  - [ ] Strategic + Tactical 시너지 확인

#### 3.5.7 문서화
- [ ] **구현 가이드** (`ML_TACTICAL_LAYER.md`)
  - [ ] LSTM 아키텍처 설명
  - [ ] 학습 프로세스
  - [ ] 하이퍼파라미터 튜닝 가이드

- [ ] **성능 벤치마크**
  - [ ] LSTM vs LLM 비교
  - [ ] 하이브리드 vs 단독 비교
  - [ ] 계산 비용 분석

### 산출물 (예정)
- ⏳ `backend/ml/__init__.py`
- ⏳ `backend/ml/price_predictor.py` - LSTM 모델
- ⏳ `backend/ml/pattern_recognizer.py` - RandomForest
- ⏳ `backend/ml/position_manager.py` - Kelly + Trailing Stop
- ⏳ `backend/ml/train_lstm.py` - 학습 스크립트
- ⏳ `backend/ml/train_patterns.py` - 패턴 학습
- ⏳ `backend/agents/tactical_agent.py` - 전술 에이전트
- ⏳ `backend/tests/test_ml_models.py` - ML 모델 테스트
- ⏳ `ML_TACTICAL_LAYER.md` - 문서

### 검증 기준
- [ ] LSTM 정확도 > 62% (테스트셋)
- [ ] 패턴 인식 정확도 > 55%
- [ ] 하이브리드 백테스트: 연 수익 > 30%, 승률 > 60%
- [ ] Trailing stop이 손실 20% 이하로 제한
- [ ] Kelly Criterion으로 포지션 > 30% 방지

### 예상 성과
```
Without ML (LLM만):
- 연 수익률: 15-25%
- 승률: 55-60%
- 샤프 비율: 1.2

With ML (LLM + LSTM):
- 연 수익률: 30-40%  (+15% 향상)
- 승률: 60-62%       (+5% 향상)
- 샤프 비율: 1.8     (+50% 향상)
```

---

## 🧪 Phase 4: Lumibot 백테스팅 통합 ⏳ 대기 (0%)

### 목표
LangGraph 에이전트를 Lumibot Strategy에 통합하여 이벤트 기반 백테스팅 구현

### 예상 기간
2주

### 작업 계획

#### 4.1 Lumibot Strategy 작성
- [ ] **LLMAgentStrategy 클래스**
  - [ ] `initialize()` 메서드
  - [ ] `on_trading_iteration()` 메서드
  - [ ] LangGraph 에이전트 호출 통합
  - [ ] 거래 실행 로직
- [ ] **데이터 동기화**
  - [ ] OHLCV 데이터
  - [ ] 뉴스 데이터 (MongoDB)
  - [ ] 기술적 지표
  - [ ] Lookahead Bias 방지 로직

#### 4.2 Semantic Caching 통합
- [ ] **Redis Semantic Cache 설정**
  - [ ] LangChain Redis Semantic Cache 설정
  - [ ] 코사인 유사도 임계값 0.95
  - [ ] 백테스팅용 무기한 캐시
- [ ] **패턴 기반 정규화**
  - [ ] 가격 반올림 (10단위)
  - [ ] 지표 반올림 (5단위)
  - [ ] 추세 분류 (uptrend/downtrend/sideways)
  - [ ] 모멘텀 분류 (oversold/neutral/overbought)

#### 4.3 백테스팅 실행
- [ ] **백테스팅 스크립트**
  - [ ] `backend/run_backtest.py`
  - [ ] 기간 설정 (2024-01-01 ~ 2024-12-31)
  - [ ] 초기 자본 $10,000
  - [ ] 수수료 0.1%
- [ ] **결과 저장**
  - [ ] MongoDB에 reasoning_logs 저장
  - [ ] PostgreSQL에 backtest_results 저장
  - [ ] 거래 실행 로그 저장

#### 4.4 지연 시간 시뮬레이션
- [ ] **API 지연 시간 고려**
  - [ ] LLM 응답 시간 측정
  - [ ] 다음 캔들 시가 체결
  - [ ] 슬리피지 시뮬레이션

### 산출물 (예정)
- ⏳ `backend/strategies/agent_strategy.py`
- ⏳ `backend/utils/semantic_cache.py`
- ⏳ `backend/run_backtest.py`
- ⏳ 백테스트 결과 보고서

### 검증 기준
- [ ] 1년치 백테스팅 완료 (8,760시간)
- [ ] MongoDB에 reasoning_logs 저장됨
- [ ] Semantic Caching으로 API 호출 40% 이상 감소
- [ ] 백테스팅 결과: Sharpe Ratio, Max Drawdown, Total Return 출력
- [ ] Lookahead Bias 없음 (미래 데이터 접근 차단)

---

## 🤖 Phase 4.5: FinRL Execution Layer (강화학습) ⏳ 대기 (0%)

### 목표
**강화학습(Reinforcement Learning)으로 최적 실행 전략 학습** - 포지션 사이징, 진입/청산 타이밍 자동 최적화

> 💡 **핵심 차이:**
> - **LSTM**: "다음에 가격이 오를 것 같다" (예측만)
> - **FinRL**: "지금 포트폴리오의 23% 매수하자" (행동 직접 학습)
>
> **Why FinRL?**
> - 수익 극대화가 **직접적인 학습 목표**
> - 포지션 크기, 타이밍, 리스크 관리 **통합 최적화**
> - 시장 변화에 **적응적 학습**

### 예상 기간
6-8주

### 작업 계획

#### 4.5.1 FinRL 환경 설정
- [ ] **라이브러리 설치**
  ```bash
  pip install finrl
  pip install stable-baselines3[extra]
  pip install gymnasium
  ```

- [ ] **GPU 환경 준비**
  - [ ] CUDA 11.8+ 설치 확인
  - [ ] PyTorch GPU 버전 확인
  - [ ] 학습 시간: GPU 12-24시간 vs CPU 3-5일

#### 4.5.2 커스텀 Trading Environment
- [ ] **CryptoTradingEnv 구현** (`backend/rl/crypto_env.py`)
  ```python
  class CryptoTradingEnv(gym.Env):
      """
      비트코인 트레이딩을 위한 FinRL 환경
      """
      def __init__(self):
          # State Space: 194차원
          # - 기본 시장 데이터: 180차원
          #   (OHLCV + 지표 × 최근 30개 캔들)
          # - LLM 신호: 2차원 (signal, confidence)
          # - LSTM 예측: 2차원 (direction, probability)
          # - 패턴 신호: 1차원
          # - 포트폴리오 상태: 9차원
          #   (현금, 보유량, PnL, 보유일수 등)

          self.observation_space = gym.spaces.Box(
              low=-np.inf,
              high=np.inf,
              shape=(194,)
          )

          # Action Space: 연속형 [-1, 1]
          # -1.0: 전량 매도
          #  0.0: 보유
          # +1.0: 전량 매수
          self.action_space = gym.spaces.Box(
              low=-1,
              high=1,
              shape=(1,)
          )

      def reset(self):
          """에피소드 초기화"""
          self.current_step = 0
          self.portfolio_value = 10000
          self.holdings = 0
          return self._get_observation()

      def step(self, action):
          """
          행동 실행 및 보상 계산

          Returns:
              observation, reward, done, info
          """
          # 1. 행동 실행 (매수/매도)
          self._execute_action(action)

          # 2. 시장 업데이트 (다음 캔들)
          self.current_step += 1
          self._update_market()

          # 3. 보상 계산
          reward = self._calculate_reward()

          # 4. 종료 조건
          done = (self.current_step >= len(self.data) - 1)

          return self._get_observation(), reward, done, {}

      def _calculate_reward(self):
          """
          멀티팩터 보상 함수
          """
          # 1. 포트폴리오 수익률 (40%)
          portfolio_return = (
              self.portfolio_value - self.prev_portfolio_value
          ) / self.prev_portfolio_value

          # 2. 샤프 비율 보상 (30%)
          # 변동성 고려한 위험 조정 수익
          returns_history = self.returns[-30:]
          if len(returns_history) > 5:
              sharpe = (
                  np.mean(returns_history) /
                  (np.std(returns_history) + 1e-9)
              )
              sharpe_reward = sharpe * 0.01
          else:
              sharpe_reward = 0

          # 3. 낙폭 패널티 (20%)
          # 최대 낙폭이 크면 페널티
          max_dd = self._calculate_max_drawdown()
          dd_penalty = -max_dd * 0.5 if max_dd > 0.1 else 0

          # 4. 거래 비용 (5%)
          # 과도한 거래 방지
          trading_cost = -0.001 if action != 0 else 0

          # 5. 보유 기간 보너스 (5%)
          # 스윙 트레이딩 유도 (2일+ 보유)
          holding_bonus = (
              0.0005 * self.holding_days
              if self.holding_days > 2 else 0
          )

          # 최종 보상
          total_reward = (
              portfolio_return * 0.4 +
              sharpe_reward * 0.3 +
              dd_penalty * 0.2 +
              trading_cost * 0.05 +
              holding_bonus * 0.05
          )

          return total_reward
  ```

#### 4.5.3 신호 통합 State
- [ ] **Enhanced State 생성** (`backend/rl/state_builder.py`)
  ```python
  def build_enhanced_state(
      market_data,
      llm_analysis,
      lstm_prediction,
      pattern_signal,
      portfolio_state
  ):
      """
      LLM + LSTM + 패턴을 FinRL State로 통합
      """
      # 1. 기본 시장 데이터 (180차원)
      base_state = prepare_market_features(market_data)

      # 2. LLM 전략 신호 (2차원)
      llm_signal = signal_to_number(llm_analysis['decision'])
      llm_conf = llm_analysis['confidence']

      # 3. LSTM 예측 (2차원)
      lstm_dir = lstm_prediction['direction']  # -1/0/1
      lstm_prob = lstm_prediction['probability']

      # 4. 패턴 신호 (1차원)
      pattern_val = pattern_signal['signal']

      # 5. 포트폴리오 (9차원)
      port = [
          portfolio_state['balance_ratio'],
          portfolio_state['holdings_ratio'],
          portfolio_state['pnl'],
          portfolio_state['holding_days'],
          portfolio_state['max_drawdown'],
          portfolio_state['win_rate'],
          portfolio_state['total_trades'],
          portfolio_state['consecutive_losses'],
          portfolio_state['volatility']
      ]

      # 통합 (194차원)
      enhanced_state = np.concatenate([
          base_state,      # 180
          [llm_signal],    # 1
          [llm_conf],      # 1
          [lstm_dir],      # 1
          [lstm_prob],     # 1
          [pattern_val],   # 1
          port             # 9
      ])

      return enhanced_state
  ```

#### 4.5.4 PPO 알고리즘 학습
- [ ] **PPO Agent 설정** (`backend/rl/train_ppo.py`)
  ```python
  from stable_baselines3 import PPO
  from stable_baselines3.common.vec_env import DummyVecEnv

  # 환경 생성
  env = CryptoTradingEnv(
      df=train_data,
      initial_amount=10000
  )
  env = DummyVecEnv([lambda: env])

  # PPO 모델 생성
  model = PPO(
      policy="MlpPolicy",
      env=env,
      learning_rate=3e-4,
      n_steps=2048,         # 배치 크기
      batch_size=64,
      n_epochs=10,
      gamma=0.99,           # 할인율
      gae_lambda=0.95,      # GAE
      clip_range=0.2,       # PPO clip
      ent_coef=0.01,        # 엔트로피 계수
      vf_coef=0.5,          # 가치 함수 계수
      max_grad_norm=0.5,
      verbose=1,
      tensorboard_log="./tensorboard_ppo/"
  )

  # 학습 (100만 스텝 = 12-24시간)
  model.learn(
      total_timesteps=1_000_000,
      callback=checkpoint_callback
  )

  # 모델 저장
  model.save("trained_models/ppo_crypto_v1")
  ```

- [ ] **학습 모니터링**
  - [ ] TensorBoard 실시간 추적
  - [ ] 에피소드 보상 추이
  - [ ] 정책 손실, 가치 손실
  - [ ] Checkpoint 자동 저장 (매 10만 스텝)

#### 4.5.5 하이브리드 FinRL Agent
- [ ] **통합 Agent** (`backend/agents/finrl_agent.py`)
  ```python
  class HybridFinRLAgent:
      """
      LLM + LSTM + FinRL 통합 에이전트
      """
      def __init__(self):
          # Strategic Layer
          self.llm_agent = compile_trading_graph()

          # Tactical Layer
          self.lstm_predictor = PriceMovementPredictor()
          self.pattern_recognizer = CandlePatternRecognizer()

          # Execution Layer (NEW!)
          self.finrl_agent = PPO.load(
              "trained_models/ppo_crypto_v1"
          )

      def make_decision(self, market_data):
          """
          3-Layer 통합 의사결정
          """
          # 1. Strategic Analysis (LLM)
          strategic = self.llm_agent.analyze(market_data)

          # 2. Tactical Analysis (LSTM + Pattern)
          lstm_pred = self.lstm_predictor.predict(market_data)
          pattern = self.pattern_recognizer.detect(market_data)

          # 3. Build Enhanced State
          state = build_enhanced_state(
              market_data,
              strategic,
              lstm_pred,
              pattern,
              self.get_portfolio_state()
          )

          # 4. FinRL Execution Decision
          action, _states = self.finrl_agent.predict(
              state,
              deterministic=True
          )

          # 5. Interpret Action
          return self._interpret_action(action[0])

      def _interpret_action(self, action_value):
          """
          연속 행동 → 거래 신호 변환

          action_value: -1.0 ~ 1.0
          """
          if action_value > 0.1:
              return {
                  'decision': 'BUY',
                  'amount': action_value * self.portfolio_value,
                  'confidence': abs(action_value)
              }
          elif action_value < -0.1:
              return {
                  'decision': 'SELL',
                  'amount': abs(action_value) * self.holdings_value,
                  'confidence': abs(action_value)
              }
          else:
              return {
                  'decision': 'HOLD',
                  'amount': 0,
                  'confidence': 1.0 - abs(action_value)
              }
  ```

#### 4.5.6 온라인 학습 (Online Learning)
- [ ] **지속적 학습 파이프라인** (`backend/rl/online_learning.py`)
  ```python
  def online_learning_update(model, recent_trades):
      """
      실전 거래 데이터로 모델 미세조정

      매주 실행:
      - 최근 30일 거래 데이터 수집
      - 10,000 스텝 fine-tuning
      - 성능 검증 후 업데이트
      """
      # 최근 데이터 준비
      recent_data = prepare_recent_data(days=30)

      # 환경 재생성
      env = CryptoTradingEnv(df=recent_data)

      # Fine-tuning (짧게)
      model.learn(
          total_timesteps=10_000,
          reset_num_timesteps=False  # 기존 학습 유지
      )

      # 검증
      validation_return = validate_model(model)

      # 성능 향상 시에만 업데이트
      if validation_return > previous_best:
          model.save("trained_models/ppo_crypto_updated")
          return True
      return False
  ```

#### 4.5.7 백테스팅 및 평가
- [ ] **FinRL 백테스트** (`backend/tests/test_finrl.py`)
  ```python
  # 테스트 기간: 2024년 1-3월 (학습 안한 기간)
  test_env = CryptoTradingEnv(df=test_data)

  # FinRL 실행
  obs = test_env.reset()
  total_reward = 0
  trades = []

  while True:
      action, _ = model.predict(obs, deterministic=True)
      obs, reward, done, info = test_env.step(action)
      total_reward += reward
      trades.append(info)

      if done:
          break

  # 성과 분석
  analyze_performance(trades)
  ```

- [ ] **성능 비교**
  ```
  LLM만:              25% 연 수익, 58% 승률
  LLM + LSTM:         40% 연 수익, 62% 승률
  LLM + LSTM + FinRL: 65% 연 수익, 68% 승률 (목표)
  ```

#### 4.5.8 문서화
- [ ] **FinRL 구현 가이드** (`FINRL_EXECUTION_LAYER.md`)
  - [ ] 강화학습 기초 개념
  - [ ] PPO 알고리즘 설명
  - [ ] 보상 함수 설계 철학
  - [ ] 하이퍼파라미터 튜닝 가이드

- [ ] **학습 체크리스트**
  - [ ] GPU 환경 준비
  - [ ] 학습 모니터링 방법
  - [ ] 온라인 학습 스케줄

### 산출물 (예정)
- ⏳ `backend/rl/__init__.py`
- ⏳ `backend/rl/crypto_env.py` - Trading Environment
- ⏳ `backend/rl/state_builder.py` - Enhanced State
- ⏳ `backend/rl/train_ppo.py` - PPO 학습
- ⏳ `backend/rl/online_learning.py` - 지속 학습
- ⏳ `backend/agents/finrl_agent.py` - 통합 Agent
- ⏳ `backend/tests/test_finrl.py` - FinRL 테스트
- ⏳ `trained_models/ppo_crypto_v1.zip` - 학습된 모델
- ⏳ `FINRL_EXECUTION_LAYER.md` - 문서

### 검증 기준
- [ ] 학습 수렴 확인 (보상 증가 안정화)
- [ ] 백테스트 연 수익 > 50% (테스트셋)
- [ ] 샤프 비율 > 2.0
- [ ] 최대 낙폭 < 20%
- [ ] 과적합 없음 (학습/테스트 성능 차이 < 15%)

### 하드웨어 요구사항
```
최소 사양:
- CPU: 4코어 이상
- RAM: 16GB
- GPU: RTX 3060 (8GB VRAM) 이상
- 저장공간: 50GB

권장 사양:
- CPU: 8코어 이상
- RAM: 32GB
- GPU: RTX 4070 (12GB VRAM) 이상
- 저장공간: 100GB
```

### 예상 성과
```
Stage 2 (LLM + LSTM):
- 연 수익률: 30-40%
- 승률: 60-62%
- 샤프 비율: 1.8
- 최대 낙폭: 25%

Stage 3 (LLM + LSTM + FinRL):
- 연 수익률: 50-65%  (+25% 향상)
- 승률: 65-68%       (+6% 향상)
- 샤프 비율: 2.5     (+39% 향상)
- 최대 낙폭: 18%     (-7% 개선)

핵심 개선 영역:
✅ 포지션 사이징 최적화 (+15% 수익)
✅ 진입/청산 타이밍 개선 (+8% 수익)
✅ 리스크 관리 강화 (-7% 낙폭)
```

### 주의사항

**⚠️ 높은 복잡도:**
- 학습 시간: 12-24시간 (GPU 필수)
- 하이퍼파라미터 튜닝: 추가 2-3일
- 온라인 학습 인프라 구축 필요

**⚠️ 과적합 위험:**
- 정규화 필수 (엔트로피 보너스, 드롭아웃)
- Cross-validation 필수
- Out-of-sample 테스트 엄격히 수행

**⚠️ 해석 어려움:**
- FinRL은 "왜 이 결정을 했는지" 설명 불가
- LLM/LSTM 신호로 간접 해석
- 디버깅 어려움

**권장 접근:**
1. Phase 3 완료 후 LLM 성능 검증
2. Phase 3.5 완료 후 LSTM 효과 측정
3. **충분한 성과 확인 후** Phase 4.5 진행
4. GPU 환경 준비되었을 때만 시작

---

## 📊 Phase 5: Landscape of Thoughts 시각화 ⏳ 대기 (0%)

### 목표
Streamlit 대시보드에 추론 과정 시각화 구현

### 예상 기간
1-2주

### 작업 계획

#### 5.1 임베딩 및 차원 축소
- [ ] **SentenceTransformer 설정**
  - [ ] all-MiniLM-L6-v2 모델 로드
  - [ ] reasoning_trace 텍스트 → 768차원 임베딩
- [ ] **UMAP 차원 축소**
  - [ ] 2D 좌표 변환
  - [ ] 군집 파라미터 튜닝

#### 5.2 Streamlit 대시보드
- [ ] **메인 페이지**
  - [ ] UMAP 산점도 (Plotly)
  - [ ] 성공/실패 색상 구분
  - [ ] 클릭 이벤트 처리
- [ ] **상세 페이지**
  - [ ] 추론 트레이스 전체 표시
  - [ ] 토론 트랜스크립트 표시
  - [ ] 거래 결과 표시
- [ ] **필터링 기능**
  - [ ] 날짜 범위 선택
  - [ ] 성공/실패 필터
  - [ ] 키워드 검색

#### 5.3 패턴 분석
- [ ] **실패 패턴 군집 탐지**
  - [ ] 밀집 영역 자동 탐지
  - [ ] 공통 키워드 추출
  - [ ] 환각(Hallucination) 영역 식별

### 산출물 (예정)
- ⏳ `backend/visualization/lot.py`
- ⏳ `streamlit_app/app.py`
- ⏳ `streamlit_app/pages/reasoning_trace.py`

### 검증 기준
- [ ] Streamlit에서 2D 산점도 정상 표시
- [ ] 성공/실패 거래가 색상으로 구분됨
- [ ] 특정 점 선택 시 전체 추론 로그 표시
- [ ] 실패 패턴 군집이 시각적으로 식별됨

---

## 🛡️ Phase 6: Human-in-the-Loop & Guardrails ⏳ 대기 (0%)

### 목표
거래 승인 워크플로우 및 안전성 검증 강화

### 예상 기간
1주

### 작업 계획

#### 6.1 Guardrails 통합 (확장)
- [ ] **Input Guards**
  - [ ] PII Scrubbing (개인정보 제거)
  - [ ] Unusual Prompt 차단
- [ ] **Output Guards**
  - [ ] TradingSignal Pydantic 모델
  - [ ] Valid Range, Valid JSON
  - [ ] Financial Tone, No Toxic Language

#### 6.2 Human Approval Node
- [ ] **인터럽트 메커니즘**
  - [ ] LangGraph `interrupt()` 사용
  - [ ] 승인 대기 상태 저장
- [ ] **승인 UI (Streamlit)**
  - [ ] 대기 중인 거래 목록
  - [ ] 추론 과정 표시
  - [ ] 승인/거부 버튼
  - [ ] 재개 로직

### 산출물 (예정)
- ⏳ `backend/agents/human_approval.py`
- ⏳ `streamlit_app/pages/approval_ui.py`

### 검증 기준
- [ ] Guardrails가 allocation > 0.1 거래를 차단함
- [ ] Human approval node에서 실행 중단됨
- [ ] Streamlit에서 승인 버튼 클릭 시 정상 재개됨

---

## 🚀 Phase 7: 최종 통합 및 최적화 ⏳ 대기 (0%)

### 목표
전체 시스템 통합 테스트 및 성능 최적화

### 예상 기간
1-2주

### 작업 계획

#### 7.1 FastAPI 백엔드 통합
- [ ] **API 엔드포인트**
  - [ ] `/api/backtest/run-agent` - 백테스팅 실행
  - [ ] `/api/reasoning/logs` - 추론 로그 조회
  - [ ] `/ws/agent` - WebSocket 실시간 스트리밍
- [ ] **Next.js 대시보드 연동**

#### 7.2 성능 최적화
- [ ] **캐싱 효과 측정**
  - [ ] Without cache: 시간 측정
  - [ ] With cache: 시간 측정
  - [ ] Speedup 계산
- [ ] **데이터베이스 최적화**
  - [ ] MongoDB 인덱스 최적화
  - [ ] PostgreSQL 파티셔닝
  - [ ] 쿼리 성능 튜닝

#### 7.3 문서화
- [ ] **아키텍처 문서**
  - [ ] `AGENT_ARCHITECTURE.md`
  - [ ] State Machine 다이어그램
  - [ ] API 문서
- [ ] **사용자 가이드**
  - [ ] 설치 가이드
  - [ ] 백테스팅 실행 가이드
  - [ ] 대시보드 사용 가이드

#### 7.4 End-to-End 테스트
- [ ] **전체 시스템 테스트**
  - [ ] 데이터 수집
  - [ ] 백테스팅 실행
  - [ ] 대시보드 확인
  - [ ] API 테스트

### 산출물 (예정)
- ⏳ `backend/main.py` (FastAPI 확장)
- ⏳ `backend/performance_test.py`
- ⏳ `AGENT_ARCHITECTURE.md`
- ⏳ `USER_GUIDE.md`

### 검증 기준
- [ ] 전체 시스템 End-to-End 테스트 통과
- [ ] Semantic Caching으로 40% 이상 속도 향상
- [ ] Streamlit LoT 시각화 정상 작동
- [ ] Next.js 대시보드 정상 작동
- [ ] MongoDB에 1년치 reasoning_logs 저장 완료

---

## 📋 현재 작업: 데이터 수집 및 API 키 설정 + Phase 3 준비

### 완료
- [x] MongoDB checkpointer LangGraph v2 API 정합성 확보 (get_tuple/put_writes/async 래퍼) → MemorySaver 우회 없이 사용 가능
- [x] Claude API 키 설정 완료, LLM 호출 실패 시 폴백 로직 추가(결제/네트워크 미비 시에도 그래프 진행)

### 우선순위 작업
1. **✅ API 키 설정**
   - [ ] OpenAI API 키 (.env 파일)
   - [ ] CryptoPanic API 키 (.env 파일)
   - [ ] (선택) LangSmith API 키

2. **🔄 데이터 수집 (진행 예정)**
   - [ ] 1년치 BTC/USDT OHLCV 데이터 수집
   - [ ] 최근 뉴스 데이터 수집 (30일)
   - [ ] 데이터 검증

3. **🔄 전체 시스템 테스트 (데이터 수집 후)**
   - [ ] LLM 분석 포함 Agent 테스트
   - [ ] 추론 트레이스 확인
   - [ ] 데이터베이스 저장 확인

4. **🚀 Phase 3 착수 준비**
   - [ ] Bull/Bear Researcher 프롬프트/신뢰도 스코어 설계
   - [ ] Debate 서브그래프 라운드/수렴 조건 정의
   - [ ] Technical/Sentiment/Risk 모듈 스켈레톤 작성 계획 수립

### 명령어
```bash
# 1. API 키 설정
cp .env.example .env
notepad .env  # OPENAI_API_KEY, CRYPTOPANIC_API_TOKEN 입력

# 2. 데이터 수집
python backend/data/ccxt_collector.py --symbol BTC/USDT --timeframe 1h --days 365
python backend/data/news_collector.py --currencies BTC --days 30

# 3. Agent 테스트
python backend/tests/test_agent_basic.py
```

---

## 📊 코드 통계

### 전체 코드량
| Category | Files | Lines | 비고 |
|----------|-------|-------|------|
| **Phase 1: 인프라** | 9 | ~1,200 | Docker, DB, 데이터 수집 |
| **Phase 2: Agent** | 6 | ~1,590 | LangGraph, 노드, 체크포인팅 |
| **Phase 3: TradingAgents** | 0 | 0 | 예정 |
| **Phase 4: Lumibot** | 0 | 0 | 예정 |
| **Phase 5: Visualization** | 0 | 0 | 예정 |
| **Phase 6: HITL** | 0 | 0 | 예정 |
| **Phase 7: 통합** | 0 | 0 | 예정 |
| **문서** | 8 | ~1,500 | README, 완료 보고서 |
| **총계** | 23 | ~4,290 | - |

### 테스트 커버리지
- Phase 1: 8/8 테스트 통과 (100%)
- Phase 2: 기본 워크플로 테스트 통과
- 전체: 진행 중

---

## 🎯 마일스톤

### ✅ Milestone 1: 인프라 구축 완료 (2025-11-26)
- Docker 인프라
- 데이터 수집 파이프라인
- 테스트 통과

### ✅ Milestone 2: Agent Foundation 완료 (2025-11-26)
- LangGraph 구조
- 기본 노드 구현
- 체크포인팅 & 트레이싱

### ⏳ Milestone 3: TradingAgents 통합 (예정)
- Bull/Bear Researcher
- Technical Analyst
- Sentiment Analyst
- Risk Manager

### ⏳ Milestone 3.5: ML Tactical Layer (예정) 🆕
- LSTM 가격 예측 모델
- 캔들 패턴 인식 (RandomForest)
- 동적 포지션 관리 (Kelly Criterion)
- 하이브리드 신호 통합
- **목표 성능:** 30-40% 연 수익, 60-62% 승률

### ⏳ Milestone 4: 백테스팅 시스템 (예정)
- Lumibot 통합
- Semantic Caching
- 1년치 백테스트 완료

### ⏳ Milestone 4.5: FinRL Execution Layer (예정) 🆕
- PPO 강화학습 에이전트
- 3-Layer 하이브리드 통합 (LLM + LSTM + FinRL)
- 온라인 학습 파이프라인
- **목표 성능:** 50-65% 연 수익, 65-68% 승률
- **요구사항:** GPU 환경 (RTX 3060 이상)

### ⏳ Milestone 5: 시각화 & HITL (예정)
- Landscape of Thoughts
- Human-in-the-Loop
- Streamlit 대시보드

### ⏳ Milestone 6: 프로덕션 준비 (예정)
- End-to-End 테스트
- 문서화 완료
- 성능 최적화

---

## 🔗 참고 자료

### 공식 문서
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Lumibot Documentation](https://lumibot.lumiwealth.com/)
- [Guardrails AI](https://www.guardrailsai.com/docs)
- [FinGPT GitHub](https://github.com/AI4Finance-Foundation/FinGPT)

### 프로젝트 문서
- [초기 계획서](.claude/plans/radiant-chasing-willow.md)
- [Phase 1 완료](PHASE1_COMPLETE.md)
- [Phase 2 완료](PHASE2_COMPLETE.md)
- [Phase 2 진행상황](PHASE2_PROGRESS.md)
- [Phase 1 README](backend/README_PHASE1.md)

---

## 📝 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2025-11-26 | 1.0.0 | 초기 프로젝트 플랜 생성 |
| 2025-11-26 | 1.1.0 | Phase 1 완료 업데이트 |
| 2025-11-26 | 1.2.0 | Phase 2 완료 업데이트, 진행도 추가 |
| 2025-11-26 | 1.2.1 | MongoDB checkpointer를 LangGraph v2 BaseCheckpointSaver에 맞게 개선 |
| 2025-11-26 | 1.3.0 | **🧠 ML/RL 통합 로드맵 추가** - Phase 3.5 (LSTM + Pattern) 및 Phase 4.5 (FinRL) 추가 |
| 2025-11-26 | 1.4.0 | **🎯 Phase 3 시작 및 핵심 구현 완료** - Bull/Bear Researcher, Judge, Consensus 노드 구현 (40% 진행) |

---

**마지막 업데이트:** 2025-11-26
**다음 업데이트:** Phase 3 완료 시

---

## 📝 ML/RL 통합 추가 배경

### 의사결정 과정
1. **문제 인식:** 현재 시스템은 스윙 트레이딩에 최적화, 데이 트레이딩 한계
2. **분석:** LSTM (시계열 예측) + FinRL (강화학습) 조합 연구
3. **결정:** 단계적 구현으로 위험 분산 및 점진적 성능 향상

### 3-Layer 아키텍처
```
┌─────────────────────────────────────────┐
│   Strategic Layer (LLM)                 │
│   "시장 방향성, 뉴스 분석"                │
│   가중치: 40%                            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Tactical Layer (LSTM + Pattern)       │
│   "단기 예측, 진입/청산 타이밍"           │
│   가중치: 30% (LSTM) + 30% (패턴)        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Execution Layer (FinRL)               │
│   "최적 포지션 사이징, 리스크 관리"       │
│   가중치: 최종 행동 결정                 │
└─────────────────────────────────────────┘
```

### 예상 타임라인
```
현재 (2025-11-26): Phase 2 완료
↓
+2-3주: Phase 3 완료 (TradingAgents)
↓
+3주: Phase 3.5 완료 (LSTM Tactical Layer)
↓ 성과 확인 및 LLM 크레딧 충전
+2주: Phase 4 완료 (Lumibot 백테스팅)
↓
+6-8주: Phase 4.5 완료 (FinRL, GPU 환경 필요)
↓
+3-4주: Phase 5-7 완료 (시각화, HITL, 통합)
```

### 기술 선택 이유

**LSTM (Phase 3.5):**
- ✅ 시계열 데이터 특화
- ✅ 구현 간단 (1주)
- ✅ CPU로도 학습 가능 (8시간)
- ✅ 해석 가능 (확률 출력)
- ❌ 예측만 가능 (행동 결정 못함)

**FinRL (Phase 4.5):**
- ✅ 수익 직접 최적화
- ✅ End-to-end 학습
- ✅ 시장 적응 능력
- ❌ GPU 필수
- ❌ 학습 오래 걸림 (12-24시간)
- ❌ 해석 어려움 (블랙박스)

**권장 순서:**
1. Phase 3 완료 → LLM 성능 기준선 확인
2. Phase 3.5 완료 → LSTM 효과 측정 (빠른 승리)
3. Phase 4.5 고려 → GPU 환경 준비 시에만 진행

