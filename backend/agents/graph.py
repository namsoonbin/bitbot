"""
LangGraph Trading Agent
Implements the multi-agent trading workflow using LangGraph
"""

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from .state import AgentState, add_reasoning_step
from .nodes import (
    analyst_node,
    bull_researcher_node,
    bear_researcher_node,
    risk_manager_node,
    trader_node
)
from typing import Literal
from loguru import logger


def should_continue_research(state: AgentState) -> Literal["bull_researcher", "end"]:
    """
    Conditional edge: decide if we should continue to research phase

    Returns 'bull_researcher' if analysis indicates potential trade opportunity,
    otherwise returns 'end' to skip research and go straight to final decision.
    """
    # Check if analyst found any significant signals
    if state.get('fundamental_analysis') or state.get('technical_analysis'):
        logger.info("Analysis complete, proceeding to research phase")
        return "bull_researcher"
    else:
        logger.warning("No significant analysis results, ending workflow")
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

    Graph structure:
    START → Analyst → Bull Researcher → Bear Researcher → Risk Manager → Trader → END

    Conditional edges:
    - After Analyst: Check if research is needed
    - After Risk Manager: Check if trade is approved
    """

    # Initialize graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("bull_researcher", bull_researcher_node)
    workflow.add_node("bear_researcher", bear_researcher_node)
    workflow.add_node("risk_manager", risk_manager_node)
    workflow.add_node("trader", trader_node)

    # Set entry point
    workflow.set_entry_point("analyst")

    # Add edges
    # Analyst → conditional (research or end)
    workflow.add_conditional_edges(
        "analyst",
        should_continue_research,
        {
            "bull_researcher": "bull_researcher",
            "end": END
        }
    )

    # Bull Researcher → Bear Researcher (always)
    workflow.add_edge("bull_researcher", "bear_researcher")

    # Bear Researcher → Risk Manager (always)
    workflow.add_edge("bear_researcher", "risk_manager")

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

    logger.info("Trading graph created successfully")
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
