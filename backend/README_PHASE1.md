# HATS Trading System - Phase 1 인프라 구축 가이드

## 개요

Phase 1에서는 HATS 트레이딩 시스템의 기본 인프라를 구축합니다:
- PostgreSQL + TimescaleDB (시계열 OHLCV 데이터)
- MongoDB (비정형 추론 로그, 뉴스)
- Redis (시맨틱 캐싱)
- CCXT 데이터 수집기
- 뉴스 데이터 수집기

## 사전 준비

### 1. 필수 소프트웨어 설치

- **Docker Desktop**: [다운로드](https://www.docker.com/products/docker-desktop/)
- **Python 3.11+**: [다운로드](https://www.python.org/downloads/)
- **Git**: [다운로드](https://git-scm.com/downloads)

### 2. 환경 변수 설정

```bash
# .env.example을 복사하여 .env 파일 생성
cp .env.example .env

# .env 파일을 열어 필수 값 설정
# 최소한 다음 항목은 설정 필요:
# - CRYPTOPANIC_API_TOKEN (https://cryptopanic.com/developers/api/ 에서 무료 발급)
# - OPENAI_API_KEY 또는 ANTHROPIC_API_KEY (나중에 Agent 구현 시 필요)
```

### 3. Python 가상환경 생성

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 의존성 설치
pip install -r backend/requirements_agent.txt
```

## 인프라 실행

### 1. Docker Compose로 데이터베이스 실행

```bash
# 프로젝트 루트 디렉토리에서 실행
docker-compose up -d

# 컨테이너 상태 확인
docker-compose ps

# 예상 출력:
# NAME                 STATUS              PORTS
# hats_postgres        Up                  0.0.0.0:5432->5432/tcp
# hats_mongodb         Up                  0.0.0.0:27017->27017/tcp
# hats_redis           Up                  0.0.0.0:6379->6379/tcp
# hats_adminer         Up                  0.0.0.0:8080->8080/tcp
# hats_mongo_express   Up                  0.0.0.0:8081->8081/tcp
```

### 2. 데이터베이스 초기화 확인

Docker Compose는 자동으로 초기화 스크립트를 실행합니다:
- `backend/db/init_postgres.sql` → PostgreSQL 테이블 생성
- `backend/db/init_mongodb.js` → MongoDB 컬렉션 생성

**PostgreSQL 확인:**
```bash
# psql로 접속
docker exec -it hats_postgres psql -U hats_user -d hats_trading

# 테이블 확인
\dt

# 예상 출력:
#  Schema |        Name         | Type  |   Owner
# --------+---------------------+-------+-----------
#  public | ohlcv_btcusdt_1h    | table | hats_user
#  public | trades              | table | hats_user
#  public | portfolio_snapshots | table | hats_user
#  public | backtest_results    | table | hats_user
#  public | document_embeddings | table | hats_user

# 종료
\q
```

**MongoDB 확인:**
```bash
# mongosh로 접속
docker exec -it hats_mongodb mongosh -u hats_user -p hats_password

# 데이터베이스 선택
use hats_trading

# 컬렉션 확인
show collections

# 예상 출력:
# reasoning_logs
# news
# agent_checkpoints
# backtest_metadata

# 종료
exit
```

### 3. 웹 UI로 데이터베이스 확인

**Adminer (PostgreSQL):**
- URL: http://localhost:8080
- 시스템: PostgreSQL
- 서버: postgres
- 사용자: hats_user
- 암호: hats_password
- 데이터베이스: hats_trading

**Mongo Express (MongoDB):**
- URL: http://localhost:8081
- 자동 로그인 (인증 비활성화됨)

## 데이터 수집 테스트

### 1. CCXT 데이터 수집기 테스트

**1년치 BTC/USDT 1시간 봉 데이터 수집:**

```bash
# 가상환경 활성화 후 실행
cd backend/data
python ccxt_collector.py --symbol BTC/USDT --timeframe 1h --days 365 --exchange binance

# 예상 소요 시간: 약 5-10분 (네트워크 속도에 따라 다름)
# 예상 데이터 양: ~8,760개의 1시간 봉 (365일 × 24시간)
```

**실행 중 출력 예시:**
```
2025-11-26 10:00:00 | INFO     | ✓ Connected to binance exchange
2025-11-26 10:00:00 | INFO     | ✓ Connected to PostgreSQL database: hats_trading
2025-11-26 10:00:00 | INFO     | Starting historical data collection for BTC/USDT
2025-11-26 10:00:00 | INFO     | Period: 365 days, Timeframe: 1h
2025-11-26 10:00:00 | INFO     | Estimated total candles: 8760
2025-11-26 10:00:00 | INFO     | Will fetch in 9 batches
2025-11-26 10:00:02 | SUCCESS  | ✓ Fetched 1000 candles from 2024-11-26 to 2024-12-07
2025-11-26 10:00:03 | SUCCESS  | ✓ Inserted 1000 new candles into ohlcv_btcusdt_1h
2025-11-26 10:00:03 | INFO     | Progress: 11.1% (1/9 batches)
...
2025-11-26 10:08:45 | SUCCESS  | ✓ Historical data collection complete!
2025-11-26 10:08:45 | SUCCESS  | ✓ Total candles collected: 8760
```

**데이터 확인:**
```sql
-- Adminer 또는 psql에서 실행
SELECT COUNT(*) FROM ohlcv_btcusdt_1h;
-- 예상 결과: 8760

SELECT
    MIN(timestamp) as start_date,
    MAX(timestamp) as end_date,
    MIN(low) as all_time_low,
    MAX(high) as all_time_high
FROM ohlcv_btcusdt_1h;
```

### 2. 뉴스 수집기 테스트

**최근 7일 BTC 뉴스 수집:**

```bash
# .env 파일에 CRYPTOPANIC_API_TOKEN 설정 필요
cd backend/data
python news_collector.py --currencies BTC --days 7 --kind news --max-pages 5

# 예상 소요 시간: 약 5-10초
# 예상 데이터 양: ~50-200개의 뉴스 (활동량에 따라 다름)
```

**실행 중 출력 예시:**
```
2025-11-26 10:10:00 | INFO     | ✓ Connected to MongoDB: hats_trading
2025-11-26 10:10:00 | INFO     | ✓ Using collection: news
2025-11-26 10:10:00 | INFO     | Starting historical news collection
2025-11-26 10:10:00 | INFO     | Currencies: BTC, Period: 7 days, Kind: news
2025-11-26 10:10:01 | SUCCESS  | ✓ Fetched 20 news items
2025-11-26 10:10:02 | SUCCESS  | ✓ Inserted: 20, Updated: 0, Skipped: 0
2025-11-26 10:10:02 | INFO     | Progress: Page 1/5, Total collected: 20
...
2025-11-26 10:10:08 | SUCCESS  | ✓ Historical news collection complete!
2025-11-26 10:10:08 | SUCCESS  | ✓ Total items collected: 87
```

**데이터 확인:**
```javascript
// Mongo Express 또는 mongosh에서 실행
use hats_trading

// 전체 뉴스 개수
db.news.countDocuments()

// 최근 뉴스 5개 조회
db.news.find().sort({published_at: -1}).limit(5).pretty()

// 감성 점수 분포
db.news.aggregate([
  {
    $group: {
      _id: null,
      avg_sentiment: { $avg: "$sentiment.score" },
      total_positive: { $sum: "$sentiment.votes_positive" },
      total_negative: { $sum: "$sentiment.votes_negative" }
    }
  }
])
```

## 테스트 스크립트 실행

전체 인프라를 자동으로 테스트하는 스크립트를 실행합니다:

```bash
# Phase 1 검증 스크립트 실행
cd backend
python tests/test_phase1_infrastructure.py
```

## 문제 해결

### Docker 컨테이너가 시작되지 않을 때

```bash
# 로그 확인
docker-compose logs postgres
docker-compose logs mongodb
docker-compose logs redis

# 컨테이너 재시작
docker-compose restart

# 완전히 초기화하고 재시작
docker-compose down -v
docker-compose up -d
```

### PostgreSQL 연결 오류

```bash
# 컨테이너가 정상 실행 중인지 확인
docker-compose ps

# PostgreSQL 로그 확인
docker-compose logs postgres

# 포트 충돌 확인 (5432 포트가 이미 사용 중인지)
# Windows:
netstat -ano | findstr :5432
# macOS/Linux:
lsof -i :5432
```

### MongoDB 연결 오류

```bash
# MongoDB 로그 확인
docker-compose logs mongodb

# 포트 충돌 확인 (27017 포트)
# Windows:
netstat -ano | findstr :27017
# macOS/Linux:
lsof -i :27017
```

### Python 의존성 설치 오류

```bash
# pip 업그레이드
pip install --upgrade pip

# 개별 패키지 설치 시도
pip install ccxt
pip install pymongo
pip install psycopg2-binary

# Windows에서 psycopg2 설치 오류 시
# psycopg2-binary 사용 (이미 requirements에 포함됨)
```

### CryptoPanic API 오류

- API 토큰이 `.env` 파일에 제대로 설정되었는지 확인
- 무료 플랜 제한: 시간당 요청 횟수 제한이 있을 수 있음
- API 키 발급: https://cryptopanic.com/developers/api/

## 다음 단계

Phase 1 인프라가 정상 작동하면 **Phase 2: LangGraph Agent 기반 구축**으로 진행합니다:

- AgentState TypedDict 정의
- LangGraph 기본 그래프 구조 생성
- 기본 Analyst 노드 구현
- 체크포인팅 시스템 구현

자세한 내용은 계획서 참조: `.claude/plans/radiant-chasing-willow.md`

## 참고 자료

- **TimescaleDB 문서**: https://docs.timescale.com/
- **CCXT 문서**: https://docs.ccxt.com/
- **CryptoPanic API**: https://cryptopanic.com/developers/api/
- **LangGraph 문서**: https://langchain-ai.github.io/langgraph/
- **Lumibot 문서**: https://lumibot.lumiwealth.com/

## 지원

문제가 발생하면:
1. 위 문제 해결 섹션 참조
2. Docker 로그 확인
3. 이슈 트래커에 보고
