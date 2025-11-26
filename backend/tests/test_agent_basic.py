"""
Basic Agent Test
Tests the LangGraph agent workflow without checkpointing
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.state import create_initial_state, MarketData
from agents.graph import compile_trading_graph
from datetime import datetime
import uuid
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)


def test_agent_workflow():
    """Test the complete agent workflow"""

    logger.info("=" * 80)
    logger.info("TESTING HATS TRADING AGENT - Basic Workflow")
    logger.info("=" * 80)
    logger.info("")

    try:
        # Create sample market data
        sample_market_data = MarketData(
            timestamp=datetime.now(),
            symbol='BTC/USDT',
            current_price=87500.0,
            open=87000.0,
            high=88000.0,
            low=86500.0,
            volume=1000000.0,
            price_change_24h=500.0,
            price_change_pct_24h=0.58
        )

        # Create initial state
        session_id = str(uuid.uuid4())
        logger.info(f"Session ID: {session_id}")

        initial_state = create_initial_state(session_id, sample_market_data)

        logger.info(f"Initial state created:")
        logger.info(f"  - Current price: ${initial_state['market_data']['current_price']:,.2f}")
        logger.info(f"  - 24h change: {initial_state['market_data']['price_change_pct_24h']:+.2f}%")
        logger.info(f"  - Portfolio cash: ${initial_state['portfolio']['cash_balance']:,.2f}")
        logger.info("")

        # Compile graph (without checkpointer for basic test)
        logger.info("Compiling agent graph...")
        app = compile_trading_graph(checkpointer=None)
        logger.success("✓ Graph compiled successfully")
        logger.info("")

        # Run the agent
        logger.info("Executing agent workflow...")
        logger.info("-" * 80)

        final_state = None
        for step_output in app.stream(initial_state):
            # Log each step
            for node_name, node_state in step_output.items():
                logger.info(f"Completed node: {node_name}")

                if node_state.get('error'):
                    logger.error(f"Error in {node_name}: {node_state['error']}")

                final_state = node_state

        logger.info("-" * 80)
        logger.info("")

        # Check results
        if final_state:
            logger.info("=" * 80)
            logger.info("AGENT EXECUTION COMPLETE")
            logger.info("=" * 80)

            logger.info(f"Final Decision: {final_state.get('final_decision', 'UNKNOWN')}")
            logger.info(f"Total Reasoning Steps: {len(final_state['reasoning_trace'])}")
            logger.info(f"API Calls: {final_state['api_calls_count']}")
            logger.info("")

            # Display reasoning trace
            logger.info("Reasoning Trace:")
            for step in final_state['reasoning_trace']:
                logger.info(f"  [{step['role']}] Step {step['step_number']}")
                logger.info(f"    {step['content'][:100]}...")
                if step.get('confidence'):
                    logger.info(f"    Confidence: {step['confidence']:.2f}")
            logger.info("")

            # Display trade details
            if final_state.get('proposed_trade'):
                trade = final_state['proposed_trade']
                logger.info("Proposed Trade:")
                logger.info(f"  Action: {trade['action']}")
                logger.info(f"  Allocation: {trade['allocation']*100:.1f}%")
                logger.info(f"  Confidence: {trade['confidence']:.2f}")
                logger.info(f"  Stop Loss: {trade['stop_loss_pct']}%")
                logger.info(f"  Take Profit: {trade['take_profit_pct']}%")
                logger.info("")

            # Display risk assessment
            if final_state.get('risk_assessment'):
                risk = final_state['risk_assessment']
                logger.info("Risk Assessment:")
                logger.info(f"  Approved: {risk['approved']}")
                logger.info(f"  Risk Score: {risk['risk_score']:.2f}")
                logger.info(f"  Feedback: {risk['feedback']}")
                logger.info("")

            logger.success("✓ Test completed successfully!")
            return True

        else:
            logger.error("✗ No final state received")
            return False

    except Exception as e:
        logger.error(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_agent_workflow()
    sys.exit(0 if success else 1)
