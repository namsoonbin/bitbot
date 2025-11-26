# 📊 Trading Style Analysis - 스윙 vs 데이 트레이딩

**작성일:** 2025-11-26
**목적:** 현재 시스템의 한계 분석 및 데이 트레이딩 개선 방안

---

## 🎯 현재 시스템 분석

### ✅ 현재 시스템이 잘하는 것 (Swing Trading)

```
타임프레임: 1시간 ~ 1일
보유 기간: 1-7일
의사결정 기반: LLM 추론 + 뉴스 감성 + 기술적 지표

강점:
✓ 펀더멘털 분석 우수 (뉴스 해석)
✓ 복잡한 맥락 이해 (LLM)
✓ 리스크 관리 (다단계 검증)
✓ 변증법적 추론 (Bull vs Bear)

적합한 시나리오:
- "규제 소식으로 3-5일 상승 전망"
- "기관 투자 유입으로 주간 추세 변화"
- "저항선 돌파 후 며칠간 상승 예상"
```

### ❌ 현재 시스템의 한계 (Day Trading)

```
타임프레임: 1분 ~ 15분
보유 기간: 몇 분 ~ 몇 시간
필요한 것: 빠른 패턴 인식 + 정량적 신호

약점:
✗ LLM 응답 속도 느림 (5-10초)
✗ 뉴스는 단기 변동성 설명 못함
✗ 1시간 캔들로는 단타 패턴 부족
✗ 포지션 비중 조절 로직 없음
✗ 실시간 손절/익절 타이밍 부재

부적합한 시나리오:
- "15분 차트에서 쌍바닥 형성"
- "5분 RSI 과매도 → 즉시 반등 매매"
- "변동성 돌파 후 30분 내 익절"
- "손실 2% 도달 시 즉시 손절"
```

---

## 💡 제안: 하이브리드 시스템 아키텍처

### 개념: 2-Layer Trading System

```
┌─────────────────────────────────────────────────────────┐
│                   HATS Trading System v2.0              │
│                                                         │
│  ┌──────────────────────┐  ┌──────────────────────┐  │
│  │   Strategic Layer     │  │   Tactical Layer      │  │
│  │   (Swing Trading)     │  │   (Day Trading)       │  │
│  │                       │  │                       │  │
│  │  • LLM 추론           │  │  • ML 패턴 인식       │  │
│  │  • 뉴스 분석          │  │  • 실시간 신호        │  │
│  │  • 시장 방향 예측     │  │  • 포지션 조절        │  │
│  │                       │  │                       │  │
│  │  타임프레임: 1시간+   │  │  타임프레임: 1-15분   │  │
│  │  보유기간: 1-7일      │  │  보유기간: 몇분-몇시간│  │
│  └──────────┬────────────┘  └────────┬─────────────┘  │
│             │                         │                 │
│             └────────┬────────────────┘                 │
│                      ↓                                   │
│            ┌─────────────────┐                          │
│            │  Risk Manager    │                          │
│            │  (통합 관리)     │                          │
│            └─────────────────┘                          │
└─────────────────────────────────────────────────────────┘

의사결정 흐름:
1. Strategic Layer: "전체적으로 상승 추세, 매수 포지션 유지"
2. Tactical Layer: "15분 차트 과매도, 지금 진입"
3. Tactical Layer: "목표가 도달, 30% 익절"
4. Strategic Layer: "뉴스 악화, 전체 포지션 청산"
```

---

## 🎯 데이 트레이딩을 위한 개선 방안

### 1. ML 기반 패턴 인식 시스템

#### A. 캔들 패턴 분류 모델

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

class CandlePatternRecognizer:
    """
    머신러닝 기반 캔들 패턴 인식기
    """

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.patterns = [
            'bullish_engulfing',
            'bearish_engulfing',
            'hammer',
            'shooting_star',
            'doji',
            'morning_star',
            'evening_star',
            'three_white_soldiers',
            'three_black_crows'
        ]

    def extract_features(self, candles: pd.DataFrame) -> np.ndarray:
        """
        캔들 데이터에서 특징 추출
        """
        features = []

        for i in range(len(candles) - 3):
            window = candles.iloc[i:i+3]

            # 1. 기본 캔들 특징
            body_size = abs(window['close'] - window['open'])
            upper_shadow = window['high'] - window[['open', 'close']].max(axis=1)
            lower_shadow = window[['open', 'close']].min(axis=1) - window['low']
            total_range = window['high'] - window['low']

            # 2. 캔들 간 관계
            price_change = window['close'].pct_change()
            volume_change = window['volume'].pct_change()

            # 3. 정규화된 비율
            body_ratio = body_size / total_range
            upper_shadow_ratio = upper_shadow / total_range
            lower_shadow_ratio = lower_shadow / total_range

            # 4. 모멘텀 지표
            rsi = self.calculate_rsi(window, period=14)

            features.append([
                *body_size.values,
                *body_ratio.values,
                *upper_shadow_ratio.values,
                *lower_shadow_ratio.values,
                *price_change.fillna(0).values,
                *volume_change.fillna(0).values,
                rsi
            ])

        return np.array(features)

    def predict_pattern(self, candles: pd.DataFrame) -> dict:
        """
        패턴 예측 및 신뢰도 반환
        """
        features = self.extract_features(candles)

        # 확률 예측
        probabilities = self.model.predict_proba(features[-1].reshape(1, -1))[0]

        # 가장 높은 확률의 패턴
        max_prob_idx = np.argmax(probabilities)
        pattern = self.patterns[max_prob_idx]
        confidence = probabilities[max_prob_idx]

        return {
            'pattern': pattern,
            'confidence': confidence,
            'signal': self.pattern_to_signal(pattern),
            'all_probabilities': dict(zip(self.patterns, probabilities))
        }

    def pattern_to_signal(self, pattern: str) -> str:
        """패턴 → 거래 신호"""
        bullish = ['bullish_engulfing', 'hammer', 'morning_star', 'three_white_soldiers']
        bearish = ['bearish_engulfing', 'shooting_star', 'evening_star', 'three_black_crows']

        if pattern in bullish:
            return 'BUY'
        elif pattern in bearish:
            return 'SELL'
        else:
            return 'HOLD'
```

**훈련 방법:**
```python
# 1. 과거 데이터에서 패턴 레이블링
historical_data = load_ohlcv(days=365)
patterns_labeled = label_patterns_manually(historical_data)  # 또는 TA-Lib 사용

# 2. 특징 추출
X = recognizer.extract_features(historical_data)
y = patterns_labeled['pattern']

# 3. 학습
recognizer.model.fit(X, y)

# 4. 백테스트로 검증
backtest_accuracy = evaluate_on_test_set(recognizer, test_data)
print(f"Pattern Recognition Accuracy: {backtest_accuracy:.2%}")
```

---

#### B. 가격 움직임 예측 모델 (LSTM)

```python
import torch
import torch.nn as nn

class PriceMovementPredictor(nn.Module):
    """
    LSTM 기반 단기 가격 움직임 예측
    입력: 최근 60개 캔들 (15분 = 15시간 데이터)
    출력: 다음 15분의 가격 방향 (상승/하락/횡보)
    """

    def __init__(self, input_size=5, hidden_size=128, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,  # OHLCV
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 3)  # UP, DOWN, SIDEWAYS
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        lstm_out, _ = self.lstm(x)

        # 마지막 타임스텝 사용
        last_output = lstm_out[:, -1, :]

        x = self.relu(self.fc1(last_output))
        x = self.fc2(x)
        x = self.softmax(x)

        return x

# 사용 예시
model = PriceMovementPredictor()
recent_candles = get_recent_candles(count=60, timeframe='15m')

# 정규화
normalized = (recent_candles - recent_candles.mean()) / recent_candles.std()

# 예측
prediction = model(torch.tensor(normalized).unsqueeze(0))
# [0.1, 0.7, 0.2] → 70% 확률로 하락 예상

if prediction[0][0] > 0.6:  # 60% 이상 확률로 상승
    action = "BUY"
elif prediction[0][1] > 0.6:  # 60% 이상 확률로 하락
    action = "SELL"
else:
    action = "HOLD"
```

**훈련 데이터:**
```python
# 1년치 15분 캔들 = ~35,000 샘플
train_size = int(35000 * 0.8)

X_train = sequences[:train_size]  # (N, 60, 5)
y_train = labels[:train_size]     # (N, 3) - one-hot encoded

# 학습
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# 백테스트 정확도
test_accuracy = evaluate(model, X_test, y_test)
print(f"Price Movement Prediction Accuracy: {test_accuracy:.2%}")
```

---

### 2. 동적 포지션 관리 시스템

#### A. Kelly Criterion 기반 포지션 사이징

```python
class DynamicPositionManager:
    """
    동적 포지션 크기 조절
    """

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        켈리 공식: f = (p * b - q) / b

        f = 투자 비율
        p = 승률
        q = 1 - p (패율)
        b = 평균 수익 / 평균 손실
        """
        if avg_loss == 0:
            return 0

        b = avg_win / avg_loss
        q = 1 - win_rate

        kelly = (win_rate * b - q) / b

        # 안전을 위해 켈리의 50%만 사용 (Half Kelly)
        return max(0, min(kelly * 0.5, 0.3))  # 최대 30%

    def adjust_position_size(
        self,
        base_position: float,
        confidence: float,
        current_drawdown: float,
        volatility: float
    ) -> float:
        """
        여러 요소를 고려한 포지션 조절
        """

        # 1. 신뢰도 기반 조절
        confidence_multiplier = confidence  # 0.0 ~ 1.0

        # 2. 드로다운 기반 조절 (손실 중이면 축소)
        if current_drawdown > 0.05:  # 5% 이상 손실
            drawdown_multiplier = 0.5  # 포지션 절반으로
        else:
            drawdown_multiplier = 1.0

        # 3. 변동성 기반 조절 (변동성 높으면 축소)
        # ATR (Average True Range) 사용
        if volatility > 0.03:  # 3% 이상 변동성
            volatility_multiplier = 0.7
        else:
            volatility_multiplier = 1.0

        # 최종 포지션 크기
        adjusted = base_position * confidence_multiplier * drawdown_multiplier * volatility_multiplier

        return min(adjusted, 0.3)  # 최대 30%

# 사용 예시
manager = DynamicPositionManager()

# 최근 100 거래 통계
win_rate = 0.58  # 58% 승률
avg_win = 0.025  # 평균 2.5% 수익
avg_loss = 0.015  # 평균 1.5% 손실

# Kelly Fraction 계산
kelly_fraction = manager.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
# → 약 0.15 (15% 권장)

# 현재 상황에 맞게 조절
current_confidence = 0.7  # ML 모델 신뢰도
current_drawdown = 0.03   # 현재 3% 손실
current_volatility = 0.02  # 2% 변동성

final_position = manager.adjust_position_size(
    kelly_fraction,
    current_confidence,
    current_drawdown,
    current_volatility
)

print(f"추천 포지션 크기: {final_position:.1%}")
# → 약 10.5%
```

---

#### B. 트레일링 스톱 & 부분 익절 시스템

```python
class TacticalExitManager:
    """
    동적 손절 / 익절 관리
    """

    def __init__(self):
        self.entry_price = None
        self.highest_price = None
        self.positions_closed = 0

    def update_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        atr: float,
        profit_pct: float
    ) -> dict:
        """
        트레일링 스톱 로직
        """

        # 1. ATR 기반 초기 스톱로스
        initial_stop = entry_price - (2 * atr)  # 2 ATR 아래

        # 2. 수익률에 따른 트레일링
        if profit_pct > 0.02:  # 2% 이상 수익
            # 수익의 50%를 보호
            trailing_stop = entry_price + (current_price - entry_price) * 0.5
        elif profit_pct > 0.01:  # 1% 이상 수익
            # 본전 보호
            trailing_stop = entry_price
        else:
            # 초기 스톱 유지
            trailing_stop = initial_stop

        return {
            'stop_loss': max(trailing_stop, initial_stop),
            'should_exit': current_price <= trailing_stop,
            'protection_level': 'breakeven' if trailing_stop >= entry_price else 'initial'
        }

    def partial_profit_taking(
        self,
        profit_pct: float,
        position_size: float
    ) -> dict:
        """
        부분 익절 로직
        """

        actions = []
        remaining_position = position_size

        # 타겟 1: 1% 수익 → 30% 익절
        if profit_pct >= 0.01 and self.positions_closed == 0:
            close_amount = position_size * 0.3
            remaining_position -= close_amount
            actions.append({
                'action': 'PARTIAL_CLOSE',
                'amount': close_amount,
                'reason': 'Target 1 (1% profit)',
                'remaining': remaining_position
            })
            self.positions_closed = 1

        # 타겟 2: 2% 수익 → 추가 30% 익절 (총 60%)
        if profit_pct >= 0.02 and self.positions_closed == 1:
            close_amount = position_size * 0.3
            remaining_position -= close_amount
            actions.append({
                'action': 'PARTIAL_CLOSE',
                'amount': close_amount,
                'reason': 'Target 2 (2% profit)',
                'remaining': remaining_position
            })
            self.positions_closed = 2

        # 타겟 3: 3% 수익 → 전량 익절
        if profit_pct >= 0.03:
            actions.append({
                'action': 'FULL_CLOSE',
                'amount': remaining_position,
                'reason': 'Target 3 (3% profit)',
                'remaining': 0
            })

        return {
            'actions': actions,
            'remaining_position': remaining_position
        }

# 사용 예시
exit_manager = TacticalExitManager()

# 진입: $87,500
entry = 87500
position = 0.15  # 15% 포지션

# 현재: $88,225 (0.83% 수익)
current = 88225
atr = 650  # ATR $650

profit_pct = (current - entry) / entry

# 트레일링 스톱 업데이트
stop_info = exit_manager.update_trailing_stop(current, entry, atr, profit_pct)
print(f"현재 스톱로스: ${stop_info['stop_loss']:,.0f}")
# → $87,500 (본전 보호는 아직 안 됨)

# 부분 익절 체크
profit_actions = exit_manager.partial_profit_taking(profit_pct, position)
if profit_actions['actions']:
    print("익절 신호:", profit_actions['actions'])
# → 아직 1% 도달 안 해서 대기
```

---

### 3. 하이브리드 의사결정 시스템

#### 통합 Agent 구조

```python
class HybridTradingAgent:
    """
    Strategic (LLM) + Tactical (ML) 통합 Agent
    """

    def __init__(self):
        # Strategic Layer (기존)
        self.llm_agent = compile_trading_graph()

        # Tactical Layer (신규)
        self.pattern_recognizer = CandlePatternRecognizer()
        self.price_predictor = PriceMovementPredictor()
        self.position_manager = DynamicPositionManager()
        self.exit_manager = TacticalExitManager()

    def make_decision(
        self,
        timeframe: str = '15m'
    ) -> dict:
        """
        통합 의사결정
        """

        # 1. Strategic Layer: 전반적 시장 방향 (1시간 데이터)
        strategic_state = self.get_strategic_analysis()
        strategic_signal = strategic_state['final_decision']  # BUY/SELL/HOLD
        strategic_confidence = strategic_state['proposed_trade']['confidence']

        # 2. Tactical Layer: 단기 진입 타이밍 (15분 데이터)
        recent_candles = get_recent_candles(60, timeframe)

        # 2-1. 패턴 인식
        pattern = self.pattern_recognizer.predict_pattern(recent_candles)
        pattern_signal = pattern['signal']
        pattern_confidence = pattern['confidence']

        # 2-2. 가격 예측
        price_prediction = self.price_predictor(recent_candles)
        price_signal = self.interpret_prediction(price_prediction)

        # 3. 신호 통합 (가중 평균)
        weights = {
            'strategic': 0.4,  # LLM 분석: 40%
            'pattern': 0.3,    # 패턴 인식: 30%
            'price': 0.3       # 가격 예측: 30%
        }

        # 신호 점수화 (BUY=1, HOLD=0, SELL=-1)
        signal_scores = {
            'strategic': self.signal_to_score(strategic_signal),
            'pattern': self.signal_to_score(pattern_signal),
            'price': self.signal_to_score(price_signal)
        }

        final_score = sum(
            signal_scores[key] * weights[key]
            for key in weights
        )

        # 4. 최종 결정
        if final_score > 0.3:
            final_decision = 'BUY'
        elif final_score < -0.3:
            final_decision = 'SELL'
        else:
            final_decision = 'HOLD'

        # 5. 포지션 크기 계산
        combined_confidence = (
            strategic_confidence * weights['strategic'] +
            pattern_confidence * weights['pattern'] +
            price_prediction.max() * weights['price']
        )

        position_size = self.position_manager.calculate_kelly_fraction(
            win_rate=0.58,  # 최근 통계
            avg_win=0.025,
            avg_loss=0.015
        )

        adjusted_position = self.position_manager.adjust_position_size(
            position_size,
            combined_confidence,
            current_drawdown=0.02,
            volatility=calculate_atr(recent_candles)
        )

        return {
            'decision': final_decision,
            'position_size': adjusted_position,
            'confidence': combined_confidence,
            'reasoning': {
                'strategic': {
                    'signal': strategic_signal,
                    'confidence': strategic_confidence,
                    'weight': weights['strategic']
                },
                'pattern': {
                    'signal': pattern_signal,
                    'pattern_name': pattern['pattern'],
                    'confidence': pattern_confidence,
                    'weight': weights['pattern']
                },
                'price': {
                    'signal': price_signal,
                    'probabilities': price_prediction.tolist(),
                    'weight': weights['price']
                },
                'final_score': final_score
            }
        }

    def get_strategic_analysis(self) -> dict:
        """기존 LLM Agent 실행 (1시간 타임프레임)"""
        market_data = fetch_latest_market_data(timeframe='1h')
        initial_state = create_initial_state(str(uuid.uuid4()), market_data)
        return self.llm_agent.invoke(initial_state)

    def signal_to_score(self, signal: str) -> float:
        """신호를 점수로 변환"""
        return {'BUY': 1.0, 'HOLD': 0.0, 'SELL': -1.0}[signal]

    def interpret_prediction(self, prediction: torch.Tensor) -> str:
        """가격 예측을 신호로 변환"""
        up, down, sideways = prediction[0].tolist()

        if up > 0.6:
            return 'BUY'
        elif down > 0.6:
            return 'SELL'
        else:
            return 'HOLD'
```

---

## 🎯 구현 우선순위

### Phase 3.5: Tactical Layer 추가 (2-3주)

```
Week 1: ML 모델 개발
├─ Day 1-2: 데이터 준비 및 레이블링
├─ Day 3-4: 캔들 패턴 분류 모델 훈련
├─ Day 5-6: LSTM 가격 예측 모델 훈련
└─ Day 7: 백테스트로 모델 검증

Week 2: 포지션 관리 시스템
├─ Day 1-2: Kelly Criterion 구현
├─ Day 3-4: 트레일링 스톱 로직
├─ Day 5-6: 부분 익절 시스템
└─ Day 7: 통합 테스트

Week 3: 하이브리드 통합
├─ Day 1-3: HybridTradingAgent 구현
├─ Day 4-5: 백테스트 (1년 데이터)
├─ Day 6: 성능 비교 (기존 vs 하이브리드)
└─ Day 7: 문서화 및 배포
```

---

## 📊 예상 성과 비교

### 백테스트 시뮬레이션 (가정)

```
기존 시스템 (Swing Only):
─────────────────────────────
타임프레임: 1시간
거래 빈도: 주 2-3회
보유 기간: 평균 3일
예상 승률: 55-60%
예상 연 수익률: 15-25%
최대 낙폭: 20-30%

하이브리드 시스템 (Swing + Day):
─────────────────────────────
타임프레임: 15분 + 1시간
거래 빈도: 일 3-5회
보유 기간: 평균 6시간
예상 승률: 58-65% (ML 보조)
예상 연 수익률: 30-50%
최대 낙폭: 15-25% (빠른 손절)

개선 효과:
✓ 수익률 2배
✓ 승률 5-8% 향상
✓ 낙폭 5-10% 감소
✓ 거래 기회 10배 증가
```

---

## ⚠️ 주의사항 및 트레이드오프

### 1. 복잡도 증가

```
장점:
✓ 더 많은 거래 기회
✓ 빠른 대응
✓ 정교한 포지션 관리

단점:
✗ 시스템 복잡도 2배
✗ 유지보수 부담 증가
✗ 버그 발생 가능성 증가
✗ 모니터링 필요
```

### 2. 데이터 요구사항

```
15분 타임프레임 사용 시:
- 1년 데이터 = 35,040 캔들 (vs 기존 8,760)
- 저장 공간: 4배 증가
- 백테스트 시간: 4배 증가
- DB 쿼리 부하: 증가
```

### 3. 오버피팅 리스크

```
ML 모델의 함정:
✗ 과거 데이터에만 최적화
✗ 시장 구조 변화 시 성능 저하
✗ 정기적 재훈련 필요 (월 1회)

해결 방법:
✓ Walk-forward 테스트
✓ 다양한 시장 상황 포함
✓ 앙상블 모델 사용
✓ LLM과 조합으로 과적합 방지
```

### 4. 실시간 실행 부하

```
현재: 1시간마다 1회 실행 (하루 24회)
하이브리드: 15분마다 1회 (하루 96회)

처리 시간:
- ML 모델 추론: ~100ms
- LLM 호출: ~5초
- 총: ~5.1초 (충분히 실시간 가능)

하지만:
- DB 쿼리 빈도 4배
- API 호출 빈도 4배
- 서버 리소스 2배 필요
```

---

## 💡 최종 추천

### 단계적 접근

#### Step 1: 현재 시스템 최적화 (1주)
```
- LLM 캐싱 강화 (Redis)
- 프롬프트 최적화
- 백테스트 정확도 향상
목표: 스윙 트레이딩 성능 검증
```

#### Step 2: Tactical Layer 프로토타입 (2주)
```
- 15분 데이터 수집
- 단순 패턴 인식 (TA-Lib만)
- 기본 트레일링 스톱
목표: 개념 검증 (Proof of Concept)
```

#### Step 3: ML 모델 개발 (2주)
```
- 캔들 패턴 분류 모델
- LSTM 가격 예측
- 백테스트 검증
목표: 단독 성능 60%+ 승률
```

#### Step 4: 하이브리드 통합 (1주)
```
- Strategic + Tactical 통합
- 가중치 최적화
- 전체 백테스트
목표: 기존 대비 1.5배 수익
```

#### Step 5: 실전 투입 (Paper Trading)
```
- 소액 실전 테스트 ($100)
- 2주간 모니터링
- 문제점 수정
목표: 실전 검증
```

---

## 📚 참고 자료

### ML 기반 트레이딩 논문
1. "Deep Learning for Financial Market Prediction" (2019)
2. "LSTM Networks for Cryptocurrency Price Prediction" (2020)
3. "Ensemble Methods in Algorithmic Trading" (2021)

### 오픈소스 프로젝트
1. **FinRL** - Reinforcement Learning for Trading
2. **Backtrader** - Python Backtesting Library
3. **PyAlgoTrade** - Algorithmic Trading Library

### Kelly Criterion 자료
- "Fortune's Formula" by William Poundstone
- "The Kelly Capital Growth Investment Criterion" by Thorp

---

**결론:**

현재 시스템은 **스윙 트레이딩에 최적화**되어 있고, 이것만으로도 충분히 가치가 있습니다. 하지만 데이 트레이딩을 추가하려면:

1. **ML 모델 필수** (패턴 인식 + 가격 예측)
2. **동적 포지션 관리** (Kelly + 트레일링)
3. **하이브리드 아키텍처** (전략 + 전술)

가 필요하며, 이는 **시스템 복잡도를 2배**로 만들지만, **수익 잠재력도 2배**로 만듭니다.

**추천:** 먼저 현재 시스템으로 스윙 트레이딩 성능을 검증한 후, 단계적으로 데이 트레이딩 기능을 추가하는 것이 안전합니다.
