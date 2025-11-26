"""
LangSmith Tracing Configuration for HATS Trading Agent
Enables monitoring, debugging, and optimization of LLM calls
"""

import os
from typing import Optional, Dict, Any
from loguru import logger
from datetime import datetime


def setup_langsmith_tracing(
    project_name: str = "hats-trading",
    enabled: bool = True,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None
) -> bool:
    """
    Configure LangSmith tracing for the agent

    Args:
        project_name: LangSmith project name
        enabled: Enable/disable tracing
        api_key: LangSmith API key (default from env)
        endpoint: LangSmith API endpoint (default from env)

    Returns:
        True if tracing is enabled, False otherwise
    """
    try:
        # Get API key from env if not provided
        api_key = api_key or os.getenv('LANGCHAIN_API_KEY') or os.getenv('LANGSMITH_API_KEY')

        if not enabled:
            logger.info("LangSmith tracing is disabled")
            os.environ['LANGCHAIN_TRACING_V2'] = 'false'
            return False

        if not api_key:
            logger.warning("LangSmith API key not found. Tracing disabled.")
            logger.warning("Get your API key at: https://smith.langchain.com/settings")
            os.environ['LANGCHAIN_TRACING_V2'] = 'false'
            return False

        # Configure LangSmith
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGCHAIN_API_KEY'] = api_key
        os.environ['LANGCHAIN_PROJECT'] = project_name

        if endpoint:
            os.environ['LANGCHAIN_ENDPOINT'] = endpoint
        else:
            os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

        logger.success(f"✓ LangSmith tracing enabled")
        logger.info(f"  Project: {project_name}")
        logger.info(f"  Endpoint: {os.environ['LANGCHAIN_ENDPOINT']}")
        logger.info(f"  View traces at: https://smith.langchain.com/")

        return True

    except Exception as e:
        logger.error(f"✗ Error setting up LangSmith tracing: {e}")
        os.environ['LANGCHAIN_TRACING_V2'] = 'false'
        return False


def create_trace_metadata(
    session_id: str,
    strategy_name: str = "HATS Trading Agent",
    backtest_mode: bool = False,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create metadata for LangSmith traces

    Args:
        session_id: Agent session ID
        strategy_name: Trading strategy name
        backtest_mode: Whether this is a backtest run
        additional_metadata: Additional metadata to include

    Returns:
        Metadata dictionary
    """
    metadata = {
        'session_id': session_id,
        'strategy': strategy_name,
        'mode': 'backtest' if backtest_mode else 'live',
        'timestamp': datetime.now().isoformat(),
        'framework': 'langgraph',
        'version': '1.0.0'
    }

    if additional_metadata:
        metadata.update(additional_metadata)

    return metadata


def get_trace_url(run_id: str, project_name: str = "hats-trading") -> str:
    """
    Get LangSmith trace URL for a run

    Args:
        run_id: LangSmith run ID
        project_name: Project name

    Returns:
        URL to view the trace
    """
    return f"https://smith.langchain.com/o/project/{project_name}/r/{run_id}"


class TracingContext:
    """
    Context manager for LangSmith tracing

    Usage:
        with TracingContext(session_id="abc-123") as ctx:
            # Run agent
            result = agent.run(...)
            # Tracing automatically captured
    """

    def __init__(
        self,
        session_id: str,
        project_name: str = "hats-trading",
        strategy_name: str = "HATS Trading Agent",
        backtest_mode: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize tracing context

        Args:
            session_id: Agent session ID
            project_name: LangSmith project name
            strategy_name: Trading strategy name
            backtest_mode: Whether this is a backtest run
            metadata: Additional metadata
        """
        self.session_id = session_id
        self.project_name = project_name
        self.metadata = create_trace_metadata(
            session_id=session_id,
            strategy_name=strategy_name,
            backtest_mode=backtest_mode,
            additional_metadata=metadata
        )
        self.tracing_enabled = False

    def __enter__(self):
        """Enable tracing on context entry"""
        self.tracing_enabled = setup_langsmith_tracing(
            project_name=self.project_name,
            enabled=True
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up on context exit"""
        if exc_type:
            logger.error(f"Error in tracing context: {exc_val}")
        return False  # Don't suppress exceptions


# Utility functions for trace analysis
def analyze_trace_costs(run_id: str) -> Dict[str, Any]:
    """
    Analyze API costs from a trace (requires langsmith client)

    Args:
        run_id: LangSmith run ID

    Returns:
        Cost analysis dictionary
    """
    try:
        from langsmith import Client

        client = Client()
        run = client.read_run(run_id)

        # Extract cost information
        total_tokens = 0
        total_cost = 0.0

        # Recursively traverse run tree
        def traverse_run(run_data):
            nonlocal total_tokens, total_cost

            # Extract token usage
            if hasattr(run_data, 'outputs'):
                outputs = run_data.outputs or {}
                token_usage = outputs.get('token_usage', {})

                if token_usage:
                    total_tokens += token_usage.get('total_tokens', 0)

                    # Estimate cost (rough estimate for GPT-4o-mini)
                    # Input: $0.15 per 1M tokens, Output: $0.60 per 1M tokens
                    input_tokens = token_usage.get('prompt_tokens', 0)
                    output_tokens = token_usage.get('completion_tokens', 0)

                    cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)
                    total_cost += cost

            # Traverse child runs
            if hasattr(run_data, 'child_runs'):
                for child in run_data.child_runs or []:
                    traverse_run(child)

        traverse_run(run)

        return {
            'total_tokens': total_tokens,
            'estimated_cost_usd': total_cost,
            'run_id': run_id
        }

    except ImportError:
        logger.warning("langsmith package not installed. Cannot analyze trace costs.")
        return {}
    except Exception as e:
        logger.error(f"Error analyzing trace costs: {e}")
        return {}


def get_cache_metrics(run_id: str) -> Dict[str, Any]:
    """
    Extract cache hit/miss metrics from trace

    Args:
        run_id: LangSmith run ID

    Returns:
        Cache metrics dictionary
    """
    try:
        from langsmith import Client

        client = Client()
        run = client.read_run(run_id)

        cache_hits = 0
        cache_misses = 0

        # Look for cache metadata in run
        # This would depend on how caching is implemented
        # Placeholder for now

        return {
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'cache_hit_rate': cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0.0
        }

    except ImportError:
        logger.warning("langsmith package not installed. Cannot get cache metrics.")
        return {}
    except Exception as e:
        logger.error(f"Error getting cache metrics: {e}")
        return {}


# Example usage
if __name__ == "__main__":
    import uuid

    # Setup tracing
    session_id = str(uuid.uuid4())

    # Method 1: Direct setup
    setup_langsmith_tracing(
        project_name="hats-trading",
        enabled=True
    )

    # Method 2: Using context manager
    with TracingContext(
        session_id=session_id,
        strategy_name="HATS Trading Agent v1.0",
        backtest_mode=True,
        metadata={'test_run': True}
    ) as ctx:
        logger.info(f"Tracing context created for session: {session_id}")
        logger.info(f"Tracing enabled: {ctx.tracing_enabled}")

    logger.success("✓ Tracing configuration example completed")
