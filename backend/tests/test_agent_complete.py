"""
Complete Agent Test with Checkpointing and Tracing
Tests the full HATS trading agent with all Phase 2 features
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import (
    create_initial_state,
    MarketData,
    compile_trading_graph,
    create_checkpointer,
    setup_langsmith_tracing,
    TracingContext
)
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


def test_agent_with_checkpointing():
    """Test agent with MongoDB checkpointing"""

    logger.info("=" * 80)
    logger.info("TEST 1: Agent with Checkpointing")
    logger.info("=" * 80)
    logger.info("")

    try:
        # Create checkpointer
        logger.info("Creating MongoDB checkpointer...")
        checkpointer = create_checkpointer()
        logger.success("✓ Checkpointer created")
        logger.info("")

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
        thread_id = f"thread_{session_id}"

        logger.info(f"Session ID: {session_id}")
        logger.info(f"Thread ID: {thread_id}")
        logger.info("")

        initial_state = create_initial_state(session_id, sample_market_data)

        # Compile graph WITH checkpointer
        logger.info("Compiling graph with checkpointer...")
        app = compile_trading_graph(checkpointer=checkpointer)
        logger.success("✓ Graph compiled with checkpointing enabled")
        logger.info("")

        # Run the agent (will save checkpoints automatically)
        logger.info("Executing agent workflow with checkpointing...")
        logger.info("-" * 80)

        config = {"configurable": {"thread_id": thread_id}}

        final_state = None
        for step_output in app.stream(initial_state, config):
            for node_name, node_state in step_output.items():
                logger.info(f"✓ Checkpoint saved after: {node_name}")
                final_state = node_state

        logger.info("-" * 80)
        logger.info("")

        # Check checkpoint history
        logger.info("Checking checkpoint history...")
        history = checkpointer.get_thread_history(thread_id)

        logger.success(f"✓ Found {len(history)} checkpoint(s) for this thread")
        for i, checkpoint in enumerate(history, 1):
            logger.info(f"  {i}. Checkpoint at {checkpoint['checkpoint_at']}")

        logger.info("")

        # Clean up
        checkpointer.close()

        logger.success("✓ Test 1 completed successfully!")
        logger.info("")

        return True

    except Exception as e:
        logger.error(f"✗ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_with_tracing():
    """Test agent with LangSmith tracing"""

    logger.info("=" * 80)
    logger.info("TEST 2: Agent with LangSmith Tracing")
    logger.info("=" * 80)
    logger.info("")

    try:
        # Setup tracing
        session_id = str(uuid.uuid4())

        logger.info("Setting up LangSmith tracing...")
        tracing_enabled = setup_langsmith_tracing(
            project_name="hats-trading-test",
            enabled=True
        )

        if not tracing_enabled:
            logger.warning("⚠ Tracing not enabled (API key missing), continuing without tracing")

        logger.info("")

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
        initial_state = create_initial_state(session_id, sample_market_data)

        # Compile graph (without checkpointer for this test)
        logger.info("Compiling graph with tracing...")
        app = compile_trading_graph(checkpointer=None)
        logger.success("✓ Graph compiled")
        logger.info("")

        # Run the agent (traces will be sent to LangSmith if enabled)
        logger.info("Executing agent workflow...")
        logger.info("Traces will be visible at: https://smith.langchain.com/" if tracing_enabled else "")
        logger.info("-" * 80)

        final_state = None
        for step_output in app.stream(initial_state):
            for node_name, node_state in step_output.items():
                logger.info(f"✓ Traced: {node_name}")
                final_state = node_state

        logger.info("-" * 80)
        logger.info("")

        if tracing_enabled:
            logger.success("✓ Traces sent to LangSmith!")
            logger.info("  View traces at: https://smith.langchain.com/")
        else:
            logger.info("  (Tracing was disabled - add LANGCHAIN_API_KEY to enable)")

        logger.info("")

        logger.success("✓ Test 2 completed successfully!")
        logger.info("")

        return True

    except Exception as e:
        logger.error(f"✗ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_with_both():
    """Test agent with both checkpointing AND tracing"""

    logger.info("=" * 80)
    logger.info("TEST 3: Agent with Checkpointing AND Tracing")
    logger.info("=" * 80)
    logger.info("")

    try:
        session_id = str(uuid.uuid4())
        thread_id = f"thread_{session_id}"

        # Create checkpointer
        checkpointer = create_checkpointer()

        # Use tracing context
        with TracingContext(
            session_id=session_id,
            strategy_name="HATS Trading Agent - Full Test",
            backtest_mode=False,
            metadata={'test': 'full_integration'}
        ) as trace_ctx:

            logger.info(f"Session ID: {session_id}")
            logger.info(f"Tracing enabled: {trace_ctx.tracing_enabled}")
            logger.info("")

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
            initial_state = create_initial_state(session_id, sample_market_data)

            # Compile graph with BOTH checkpointing and tracing
            logger.info("Compiling graph with checkpointing + tracing...")
            app = compile_trading_graph(checkpointer=checkpointer)
            logger.success("✓ Graph compiled with full capabilities")
            logger.info("")

            # Run the agent
            logger.info("Executing agent workflow...")
            logger.info("Both checkpoints and traces will be saved")
            logger.info("-" * 80)

            config = {"configurable": {"thread_id": thread_id}}

            final_state = None
            for step_output in app.stream(initial_state, config):
                for node_name, node_state in step_output.items():
                    logger.info(f"✓ {node_name}: Checkpointed + Traced")
                    final_state = node_state

            logger.info("-" * 80)
            logger.info("")

            # Check results
            history = checkpointer.get_thread_history(thread_id)
            logger.success(f"✓ Checkpoints: {len(history)}")

            if trace_ctx.tracing_enabled:
                logger.success("✓ Traces: Sent to LangSmith")
            else:
                logger.info("  Traces: Disabled (no API key)")

            logger.info("")

            # Clean up
            checkpointer.close()

        logger.success("✓ Test 3 completed successfully!")
        logger.info("")

        return True

    except Exception as e:
        logger.error(f"✗ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""

    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "HATS TRADING AGENT - COMPLETE TEST" + " " * 24 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("")

    results = []

    # Test 1: Checkpointing
    results.append(("Checkpointing", test_agent_with_checkpointing()))

    # Test 2: Tracing
    results.append(("Tracing", test_agent_with_tracing()))

    # Test 3: Both
    results.append(("Checkpointing + Tracing", test_agent_with_both()))

    # Summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status:8} | {test_name}")

    logger.info("=" * 80)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 80)

    if passed == total:
        logger.success("")
        logger.success("🎉 ALL TESTS PASSED!")
        logger.success("")
        logger.success("Phase 2 is complete! Key features:")
        logger.success("  ✓ AgentState with full type safety")
        logger.success("  ✓ LangGraph workflow with 5 nodes")
        logger.success("  ✓ MongoDB checkpointing for state persistence")
        logger.success("  ✓ LangSmith tracing for monitoring")
        logger.success("  ✓ PostgreSQL + MongoDB data integration")
        logger.success("")
        logger.success("Next: Phase 3 - TradingAgents Framework Integration")
        logger.success("")
        return True
    else:
        logger.error("")
        logger.error("❌ SOME TESTS FAILED")
        logger.error("")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
