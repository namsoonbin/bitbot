"""
Rate Limiter for Gemini API

Manages API rate limits for free tier:
- gemini-2.5-flash: 10 requests/minute, 250 requests/day
- gemini-2.5-pro: 2 requests/minute, 50 requests/day

Strategy:
- Add delay between API calls to stay within limits
- 2.5-flash: 6 seconds between calls (safe: 10 calls/min)
- 2.5-pro: 30 seconds between calls (safe: 2 calls/min)

Future upgrade (Paid tier):
- Remove delays or reduce significantly
- Higher quotas available
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Optional
from loguru import logger


# ============================================================================
# Rate Limiter State
# ============================================================================

class RateLimiter:
    """
    Simple rate limiter using time-based delays

    Tracks last API call time and enforces minimum delay
    """

    def __init__(self):
        self.last_call_time: Dict[str, datetime] = {}
        self.call_delays = {
            'gemini-2.5-flash': 6.0,   # 6 seconds (safe for 10/min limit)
            'gemini-2.5-pro': 30.0,    # 30 seconds (safe for 2/min limit)
        }
        self.daily_limits = {
            'gemini-2.5-flash': 250,
            'gemini-2.5-pro': 50,
        }
        self.daily_counts: Dict[str, int] = {}
        self.daily_reset_time: Optional[datetime] = None

    def wait_if_needed(self, model_name: str):
        """
        Wait if necessary to respect rate limits

        Args:
            model_name: Model name (e.g., 'gemini-2.5-flash')
        """
        # Reset daily counts if needed
        self._reset_daily_counts_if_needed()

        # Check daily limit
        if model_name in self.daily_limits:
            current_count = self.daily_counts.get(model_name, 0)
            if current_count >= self.daily_limits[model_name]:
                logger.warning(f"Daily limit reached for {model_name}: {current_count}/{self.daily_limits[model_name]}")
                logger.warning("Consider upgrading to paid tier for higher limits")

        # Get delay for this model
        delay = self.call_delays.get(model_name, 6.0)

        # Check if we need to wait
        if model_name in self.last_call_time:
            elapsed = (datetime.now() - self.last_call_time[model_name]).total_seconds()
            if elapsed < delay:
                wait_time = delay - elapsed
                logger.info(f"Rate limit: waiting {wait_time:.1f}s for {model_name}")
                time.sleep(wait_time)

        # Update last call time and count
        self.last_call_time[model_name] = datetime.now()
        self.daily_counts[model_name] = self.daily_counts.get(model_name, 0) + 1

    def _reset_daily_counts_if_needed(self):
        """Reset daily counts at midnight"""
        now = datetime.now()

        if self.daily_reset_time is None:
            # First call - set reset time to next midnight
            self.daily_reset_time = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return

        if now >= self.daily_reset_time:
            # Reset counts
            logger.info("Resetting daily API call counts")
            self.daily_counts = {}
            self.daily_reset_time = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )

    def get_stats(self) -> Dict[str, int]:
        """
        Get current usage statistics

        Returns:
            Dictionary with current counts and limits
        """
        self._reset_daily_counts_if_needed()

        stats = {}
        for model in ['gemini-2.5-flash', 'gemini-2.5-pro']:
            count = self.daily_counts.get(model, 0)
            limit = self.daily_limits.get(model, 0)
            stats[model] = {
                'count': count,
                'limit': limit,
                'remaining': limit - count
            }

        return stats


# ============================================================================
# Global Rate Limiter Instance
# ============================================================================

# Singleton instance
_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    return _rate_limiter


def wait_for_rate_limit(model_name: str):
    """
    Convenience function to wait for rate limit

    Args:
        model_name: Model name
    """
    _rate_limiter.wait_if_needed(model_name)


def get_rate_limit_stats() -> Dict[str, int]:
    """
    Get current rate limit statistics

    Returns:
        Dictionary with usage stats
    """
    return _rate_limiter.get_stats()


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    """
    Test rate limiter
    """
    print("=" * 80)
    print("Testing Rate Limiter")
    print("=" * 80)

    limiter = get_rate_limiter()

    # Test Flash model (6 second delay)
    print("\n[TEST] Gemini 2.5 Flash (6s delay)")
    for i in range(3):
        print(f"\nCall {i+1}:")
        start = time.time()
        limiter.wait_if_needed('gemini-2.5-flash')
        elapsed = time.time() - start
        print(f"  Waited: {elapsed:.2f}s")

    # Test Pro model (30 second delay)
    print("\n[TEST] Gemini 2.5 Pro (30s delay)")
    print("(Skipping actual wait for demo)")

    # Get stats
    print("\n[STATS] Current usage:")
    stats = limiter.get_stats()
    for model, data in stats.items():
        print(f"  {model}: {data['count']}/{data['limit']} ({data['remaining']} remaining)")

    print("\n" + "=" * 80)
