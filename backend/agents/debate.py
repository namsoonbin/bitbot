"""
Debate System Components: Judge and Consensus
Implements convergence detection and consensus synthesis

Based on Multi-Agent Debate research (2024-2025)
"""

from typing import Dict, Any
from .state import AgentState


# ============================================================================
# Judge Node: Convergence Detection
# ============================================================================

def judge_node(state: AgentState) -> AgentState:
    """
    Judge 노드: 토론 수렴 여부 판단

    수렴 기준:
    1. Bull/Bear 신뢰도 차이 < 0.15
    2. 추천 포지션 차이 < 20%
    3. 양측 모두 상대 논리 일부 인정
    4. 최대 4 라운드 도달
    """
    print("\n=== Judge Node: Convergence Check ===")

    debate_messages = state.get('debate_messages', [])
    round_number = state.get('debate_round', 1)

    if len(debate_messages) < 2:
        # 최소 Bull + Bear 한 쌍 필요
        state['debate_converged'] = False
        state['convergence_reason'] = "Insufficient messages for convergence check"
        return state

    # 최근 Bull/Bear 메시지
    bull_messages = [m for m in debate_messages if m['role'] == 'bull']
    bear_messages = [m for m in debate_messages if m['role'] == 'bear']

    if not bull_messages or not bear_messages:
        state['debate_converged'] = False
        state['convergence_reason'] = "Missing bull or bear message"
        return state

    latest_bull = bull_messages[-1]['content']
    latest_bear = bear_messages[-1]['content']

    # 1. 신뢰도 차이 계산
    bull_confidence = latest_bull.get('confidence', 0.5)
    bear_confidence = latest_bear.get('confidence', 0.5)
    confidence_diff = abs(bull_confidence - bear_confidence)

    # 2. 포지션 추천 차이
    bull_position = latest_bull.get('recommended_position', 0)  # 0-100%
    bear_position = latest_bear.get('recommended_position', 0)  # -100-0%
    position_diff = abs(bull_position - abs(bear_position))

    # 3. 상호 인정 확인 (간단한 키워드 체크)
    bull_counter = latest_bull.get('counter_arguments', '').lower()
    bear_counter = latest_bear.get('counter_arguments', '').lower()

    # "acknowledge", "valid", "merit", "point" 등의 긍정적 표현 확인
    acknowledgment_keywords = ['acknowledge', 'valid', 'merit', 'point', 'fair', 'agree']
    bull_acknowledges = any(kw in bull_counter for kw in acknowledgment_keywords)
    bear_acknowledges = any(kw in bear_counter for kw in acknowledgment_keywords)
    mutual_acknowledgment = bull_acknowledges and bear_acknowledges

    # 4. 수렴 판정
    converged = (
        (confidence_diff < 0.15 and position_diff < 20 and mutual_acknowledgment)
        or round_number >= 4
    )

    # 수렴 이유 기록
    if round_number >= 4:
        reason = f"Maximum rounds reached (4/4)"
    elif converged:
        reason = (
            f"Convergence achieved: "
            f"Confidence diff={confidence_diff:.2f}, "
            f"Position diff={position_diff:.1f}%, "
            f"Mutual acknowledgment={mutual_acknowledgment}"
        )
    else:
        reason = (
            f"Not converged: "
            f"Confidence diff={confidence_diff:.2f} (need <0.15), "
            f"Position diff={position_diff:.1f}% (need <20%), "
            f"Mutual acknowledgment={mutual_acknowledgment}"
        )

    state['debate_converged'] = converged
    state['convergence_reason'] = reason

    print(f"Converged: {converged}")
    print(f"Reason: {reason}")
    print(f"Round: {round_number}/4")

    return state


def should_continue_debate(state: AgentState) -> str:
    """
    Conditional edge 함수: 토론 계속 여부

    Returns:
        "continue": 다음 라운드 계속
        "converged": 합의 도달, consensus로 이동
    """
    if state.get('debate_converged', False):
        return "converged"
    else:
        # 라운드 증가
        state['debate_round'] = state.get('debate_round', 1) + 1
        return "continue"


# ============================================================================
# Consensus Synthesis Node
# ============================================================================

def consensus_synthesis_node(state: AgentState) -> AgentState:
    """
    Consensus Synthesis 노드: 최종 합의 도출

    방법:
    1. Bull/Bear 신뢰도 기반 가중 평균
    2. Evidence 강도 평가
    3. 최종 포지션 및 신뢰도 계산
    """
    print("\n=== Consensus Synthesis Node ===")

    debate_messages = state.get('debate_messages', [])

    # Bull/Bear 메시지 분리
    bull_messages = [m for m in debate_messages if m['role'] == 'bull']
    bear_messages = [m for m in debate_messages if m['role'] == 'bear']

    if not bull_messages or not bear_messages:
        print("Warning: Missing bull or bear messages for consensus")
        state['debate_consensus'] = {
            'position': 0.0,
            'confidence': 0.5,
            'bull_weight': 0.5,
            'bear_weight': 0.5,
            'summary': "Insufficient data for consensus",
            'total_rounds': 0
        }
        return state

    # 최종 Bull/Bear 입장
    final_bull = bull_messages[-1]['content']
    final_bear = bear_messages[-1]['content']

    # 신뢰도 추출
    bull_confidence = final_bull.get('confidence', 0.5)
    bear_confidence = final_bear.get('confidence', 0.5)

    # 가중치 계산 (신뢰도 기반)
    total_confidence = bull_confidence + bear_confidence
    if total_confidence > 0:
        bull_weight = bull_confidence / total_confidence
        bear_weight = bear_confidence / total_confidence
    else:
        bull_weight = bear_weight = 0.5

    # 포지션 추출
    bull_pos = final_bull.get('recommended_position', 0)  # 0-100%
    bear_pos = final_bear.get('recommended_position', 0)  # -100-0%

    # 최종 포지션 계산 (가중 평균)
    consensus_position = (
        bull_pos * bull_weight + bear_pos * bear_weight
    )

    # Evidence 강도 평가
    bull_evidence = final_bull.get('evidence', [])
    bear_evidence = final_bear.get('evidence', [])

    bull_evidence_strength = evaluate_evidence_strength(bull_evidence)
    bear_evidence_strength = evaluate_evidence_strength(bear_evidence)

    # 최종 신뢰도 (evidence 강도 반영)
    consensus_confidence = (
        bull_evidence_strength * bull_weight +
        bear_evidence_strength * bear_weight
    )

    # 합의 요약 생성
    consensus_summary = generate_consensus_summary(
        final_bull, final_bear, consensus_position, consensus_confidence
    )

    # Consensus 결과
    consensus = {
        'position': round(consensus_position, 2),  # -100 ~ 100
        'confidence': round(consensus_confidence, 2),  # 0.0 ~ 1.0
        'bull_weight': round(bull_weight, 2),
        'bear_weight': round(bear_weight, 2),
        'bull_evidence_strength': round(bull_evidence_strength, 2),
        'bear_evidence_strength': round(bear_evidence_strength, 2),
        'summary': consensus_summary,
        'total_rounds': len(bull_messages),
        'final_bull_thesis': final_bull.get('thesis', 'N/A'),
        'final_bear_thesis': final_bear.get('thesis', 'N/A')
    }

    state['debate_consensus'] = consensus

    print(f"Consensus Position: {consensus['position']:.1f}%")
    print(f"Consensus Confidence: {consensus['confidence']:.2f}")
    print(f"Bull Weight: {consensus['bull_weight']:.2f}, Bear Weight: {consensus['bear_weight']:.2f}")
    print(f"Total Rounds: {consensus['total_rounds']}")

    return state


# ============================================================================
# Helper Functions
# ============================================================================

def evaluate_evidence_strength(evidence_list: list) -> float:
    """
    Evidence 강도 평가

    기준:
    - 구체적 숫자 포함 → +0.3
    - 기술적 지표 언급 → +0.2
    - 논리적 연결성 → +0.2
    - 최근 이벤트 참조 → +0.2
    - 반박 가능성 (구체성) → +0.1
    """
    if not evidence_list:
        return 0.5  # Default

    scores = []

    for point in evidence_list:
        score = 0.0

        # 1. 숫자 데이터 포함
        if has_numerical_data(point):
            score += 0.3

        # 2. 기술적 지표 언급
        if mentions_indicator(point):
            score += 0.2

        # 3. 논리적 연결 (because, since, due to 등)
        if has_logical_connection(point):
            score += 0.2

        # 4. 구체적 이벤트 (news, announcement, report 등)
        if mentions_event(point):
            score += 0.2

        # 5. 구체성 (숫자나 날짜가 있으면 반박 가능)
        if is_specific(point):
            score += 0.1

        scores.append(min(score, 1.0))

    return sum(scores) / len(scores)


def has_numerical_data(text: str) -> bool:
    """숫자 데이터 포함 확인"""
    import re
    # $50,000, 3.5%, 1.2M, 65 등
    return bool(re.search(r'[\$€¥£]?[\d,]+\.?\d*[%KMB]?', text))


def mentions_indicator(text: str) -> bool:
    """기술적 지표 언급 확인"""
    indicators = [
        'rsi', 'macd', 'ema', 'sma', 'bollinger',
        'volume', 'support', 'resistance', 'trend',
        'momentum', 'volatility'
    ]
    text_lower = text.lower()
    return any(ind in text_lower for ind in indicators)


def has_logical_connection(text: str) -> bool:
    """논리적 연결어 확인"""
    connectors = [
        'because', 'since', 'due to', 'as a result',
        'therefore', 'thus', 'consequently', 'indicating'
    ]
    text_lower = text.lower()
    return any(conn in text_lower for conn in connectors)


def mentions_event(text: str) -> bool:
    """구체적 이벤트 언급 확인"""
    event_keywords = [
        'news', 'announcement', 'report', 'release',
        'statement', 'decision', 'launch', 'update',
        'breaking', 'recently', 'yesterday', 'today'
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in event_keywords)


def is_specific(text: str) -> bool:
    """구체성 확인 (숫자나 날짜)"""
    import re
    # 숫자 또는 날짜 패턴
    has_number = bool(re.search(r'\d+', text))
    date_patterns = [
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        'today', 'yesterday', 'week', 'month'
    ]
    has_date = any(pat in text.lower() for pat in date_patterns)

    return has_number or has_date


def generate_consensus_summary(
    bull_output: dict,
    bear_output: dict,
    consensus_position: float,
    consensus_confidence: float
) -> str:
    """
    합의 요약 생성

    Returns:
        Human-readable consensus summary
    """
    bull_thesis = bull_output.get('thesis', 'Bullish outlook')
    bear_thesis = bear_output.get('thesis', 'Bearish concerns')

    if consensus_position > 30:
        bias = "net bullish"
        action = "Consider long position"
    elif consensus_position < -30:
        bias = "net bearish"
        action = "Consider short position or exit"
    else:
        bias = "neutral"
        action = "Hold or wait for clearer signals"

    summary = (
        f"After debate, consensus is {bias} with {consensus_confidence:.0%} confidence. "
        f"Bull view: {bull_thesis}. Bear view: {bear_thesis}. "
        f"Recommended action: {action} (position size: {abs(consensus_position):.1f}%)."
    )

    return summary


# ============================================================================
# Debate Round Management
# ============================================================================

def increment_round(state: AgentState) -> AgentState:
    """토론 라운드 증가"""
    current_round = state.get('debate_round', 1)
    state['debate_round'] = current_round + 1
    return state
