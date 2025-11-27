"""
LangGraph Trading Agent
Implements the multi-agent trading workflow using LangGraph

Phase 3: Multi-Agent Debate System
- Bull/Bear Researchers debate for up to 4 rounds
- Judge evaluates convergence
- Consensus synthesis produces final position
"""

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from .state import AgentState, add_reasoning_step
from .nodes import (
    analyst_node,
    risk_manager_node,
    trader_node
)
# Phase 3: Import debate system nodes
from .researchers import (
    bull_researcher_node,
    bear_researcher_node
)
from .debate import (
    judge_node,
    should_continue_debate,
    consensus_synthesis_node
)
from typing import Literal
from loguru import logger


def should_start_debate(state: AgentState) -> Literal["bull_researcher", "end"]:
    """
    Conditional edge: decide if we should start the debate phase

    Returns 'bull_researcher' if analysis indicates potential trade opportunity,
    otherwise returns 'end' to skip debate and go straight to final decision.
    """
    # Check if analyst found any significant signals
    if state.get('fundamental_analysis') or state.get('technical_analysis'):
        logger.info("Analysis complete, starting Bull/Bear debate")
        # Initialize debate state
        state['debate_round'] = 1
        state['debate_messages'] = []
        state['debate_converged'] = False
        return "bull_researcher"
    else:
        logger.warning("No significant analysis results, ending workflow")
        state['final_decision'] = 'HOLD'
        return "end"


def should_execute_trade(state: AgentState) -> Literal["trader", "end"]:
    """
    Conditional edge: decide if trade should be executed

    Returns 'trader' if risk manager approved the trade,
    otherwise returns 'end' to reject the trade.
    """
    risk_assessment = state.get('risk_assessment')

    if risk_assessment and risk_assessment['approved']:
        logger.info(f"Risk manager APPROVED trade: {state['proposed_trade']['action']}")
        return "trader"
    else:
        logger.warning("Risk manager REJECTED trade")
        state['final_decision'] = 'HOLD'
        return "end"


def create_trading_graph() -> StateGraph:
    """
    Create the LangGraph workflow for trading decisions

    Phase 3 Graph Structure:
    START → Analyst → [Debate Loop] → Consensus → Risk Manager → Trader → END

    Debate Loop (max 4 rounds):
    Bull Researcher → Bear Researcher → Judge → (continue or converge)
    ↑                                             |
    └─────────────────(continue)─────────────────┘
                      (converged) → Consensus

    Conditional edges:
    - After Analyst: Check if debate should start
    - After Judge: Check if debate converged (or max rounds reached)
    - After Risk Manager: Check if trade is approved
    """

    # Initialize graph
    workflow = StateGraph(AgentState)

    # ========================================
    # Add nodes
    # ========================================
    workflow.add_node("analyst", analyst_node)

    # Phase 3: Debate System Nodes
    workflow.add_node("bull_researcher", bull_researcher_node)
    workflow.add_node("bear_researcher", bear_researcher_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("consensus", consensus_synthesis_node)

    # Decision nodes
    workflow.add_node("risk_manager", risk_manager_node)
    workflow.add_node("trader", trader_node)

    # ========================================
    # Set entry point
    # ========================================
    workflow.set_entry_point("analyst")

    # ========================================
    # Add edges
    # ========================================

    # Analyst → conditional (start debate or end)
    workflow.add_conditional_edges(
        "analyst",
        should_start_debate,
        {
            "bull_researcher": "bull_researcher",
            "end": END
        }
    )

    # === Debate Loop ===
    # Bull Researcher → Bear Researcher (always)
    workflow.add_edge("bull_researcher", "bear_researcher")

    # Bear Researcher → Judge (always)
    workflow.add_edge("bear_researcher", "judge")

    # Judge → conditional (continue debate or converge to consensus)
    workflow.add_conditional_edges(
        "judge",
        should_continue_debate,
        {
            "continue": "bull_researcher",  # Loop back for next round
            "converged": "consensus"        # Exit loop to consensus
        }
    )

    # Consensus → Risk Manager (always)
    workflow.add_edge("consensus", "risk_manager")

    # Risk Manager → conditional (execute or reject)
    workflow.add_conditional_edges(
        "risk_manager",
        should_execute_trade,
        {
            "trader": "trader",
            "end": END
        }
    )

    # Trader → END (always)
    workflow.add_edge("trader", END)

    logger.info("Phase 3 Trading Graph created successfully")
    logger.info("Graph includes Bull/Bear debate loop with convergence detection")
    return workflow


def compile_trading_graph(checkpointer=None):
    """
    Compile the trading graph into an executable

    Args:
        checkpointer: Optional checkpointer for state persistence
                     (MongoDBSaver for production, MemorySaver for testing)

    Returns:
        Compiled graph ready for execution
    """
    workflow = create_trading_graph()

    if checkpointer:
        logger.info("Compiling graph with checkpointer enabled")
        app = workflow.compile(checkpointer=checkpointer)
    else:
        logger.warning("Compiling graph WITHOUT checkpointer (no state persistence)")
        app = workflow.compile()

    return app


# Example usage (for testing)
if __name__ == "__main__":
    from .state import create_initial_state, MarketData
    from datetime import datetime
    import uuid

    # Create sample market data
    sample_market_data = MarketData(
        timestamp=datetime.now(),
        symbol='BTC/USDT',
        current_price=50000.0,
        open=49500.0,
        high=50500.0,
        low=49000.0,
        volume=1000000.0,
        price_change_24h=500.0,
        price_change_pct_24h=1.01
    )

    # Create initial state
    session_id = str(uuid.uuid4())
    initial_state = create_initial_state(session_id, sample_market_data)

    # Compile graph (without checkpointer for testing)
    app = compile_trading_graph()

    logger.info("Graph compiled successfully")
    logger.info(f"Available nodes: {list(app.get_graph().nodes.keys())}")

    # Note: Actual execution requires implementing the node functions
    # See nodes.py for node implementations
