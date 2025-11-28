# 진행 보고서

## 2025-11-28: Phase 3.5.2 LSTM 실험 완료

### Phase 3.5.2 LSTM 가격 예측 모델 실험
3가지 라벨링 전략 테스트 완료 - 모두 SIDEWAYS 편향 문제로 실패

**실험 결과 요약:**
| 실험 | Threshold | Lookahead | Test Acc | UP | SIDEWAYS | DOWN | 결과 |
|------|-----------|-----------|----------|-----|----------|------|------|
| Baseline | 0.1% | 15분 | 43.46% | 27.49% | 77.13% | 13.06% | SIDEWAYS 편향 |
| Exp 1 | 0.5% | 15분 | N/A | 0.00% | 100.00% | 0.00% | ❌ 재앙 |
| Exp 2 | 0.19% | 30분 | 52.17% | 17.20% | 90.31% | 7.07% | ❌ 편향 악화 |

**구현 완료:**
- ✅ `backend/ml/price_predictor.py` - LSTM 모델 (386 lines, 216,451 params)
- ✅ `backend/ml/train_lstm.py` - 학습 파이프라인 (495 lines)
- ✅ `backend/ml/data_preparation.py` - 데이터 파이프라인 (550 lines)
- ✅ 1년치 데이터 수집 (35,040 candles)
- ✅ 실험 로그 및 모델 저장

**문제점:**
1. 모든 라벨링 전략에서 SIDEWAYS 편향 발생
2. Class weighting, More data 등 해결책 모두 실패
3. 단순 LSTM 아키텍처로는 미세한 UP/DOWN 패턴 학습 실패

**향후 방향:**
- Option A: 이진 분류 (UP/DOWN만)
- Option B: 회귀 모델로 전환
- Option C: Phase 4 진행 (백테스팅 먼저, ML은 보조)

---

## 이전 변경 사항 (2025-11-27)
- 에이전트 상태/흐름
  - 초기 상태 `news_sentiment`를 빈 dict로 설정, 토론 메시지 헬퍼를 구조화 스키마와 레거시 트랜스크립트에 모두 반영.
  - 시장 국면 감지가 가격 필드 누락 시 히스토리 기반으로 백업 계산, 연구자 노드가 `None` 감성 데이터에서도 안전하게 동작.
  - 그래프에서 `should_continue`가 False면 즉시 종료하도록 가드 추가, Debate 시작 정책을 `DebatePolicy`로 중앙집중화.
- 연구자/분석 노드
  - Bull/Bear 연구자 노드를 공통 실행 경로로 리팩토링, Pydantic `model_dump()` 적용, 숫자 안전 헬퍼·RSI `None` 해석 보강.
  - Analyst 노드를 뉴스·가격 요약, LLM 실행 헬퍼로 분리해 가독성 향상; 토론 증거 숫자 감지 정규식 단순화.
- 체크포인터/추적
  - MongoDB 체크포인터에 `get_thread_history` 추가, 클라이언트 주입 옵션으로 테스트/확장 용이성 개선.
- CCXT 수집기
  - 시장 타입(`market_type`) 옵션, CCXT 호출 재시도/백오프 래퍼, 테스트용 DB 연결 스킵/교체 지원, 네트워크 회복 단위 테스트 추가.
- 데이터베이스 스키마
  - Postgres 초기화에 바이낸스 BTC/USDT 무기한 선물(스왑) OHLCV 테이블 추가: 1h, 1m, 5m.

## 테스트
- `pytest backend/tests/test_regressions.py -q`
- `pytest backend/tests/test_ccxt_collector_retry.py backend/tests/test_regressions.py -q`

## 경고/알려진 이슈
- Windows 권한 문제로 `.pytest_cache` 쓰기 경고가 발생하나 테스트 결과에는 영향 없음.
