"""
CCXT OHLCV Data Collector for HATS Trading System
Collects historical and real-time cryptocurrency data from exchanges
"""

import ccxt
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import time
import argparse
from typing import List, Tuple
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")


class CCXTDataCollector:
    """Collects OHLCV data from cryptocurrency exchanges using CCXT"""

    def __init__(
        self,
        exchange_name: str = 'binance',
        db_host: str = 'localhost',
        db_port: int = 5432,
        db_name: str = 'hats_trading',
        db_user: str = 'hats_user',
        db_password: str = 'hats_password',
        *,
        market_type: str = 'spot',
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        exchange=None,
        connect_db: bool = True
    ):
        """
        Initialize CCXT collector

        Args:
            exchange_name: Exchange to collect from (default: binance)
            db_host: PostgreSQL host
            db_port: PostgreSQL port
            db_name: Database name
            db_user: Database user
            db_password: Database password
            market_type: CCXT marketType/defaultType (spot, swap 등)
            max_retries: fetch 실패 시 재시도 횟수
            retry_backoff: 재시도 사이 대기 시간(초)
            exchange: 테스트용 커스텀 익스체인지 인스턴스 주입
        """
        self.exchange_name = exchange_name
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        if exchange:
            self.exchange = exchange
        else:
            self.exchange = getattr(ccxt, exchange_name)({
                'enableRateLimit': True,  # Respect exchange rate limits
                'options': {
                    'defaultType': market_type,
                    'marketType': market_type
                }
            })

        # Database connection (optional for testing)
        self.db_conn = None
        self.db_cursor = None
        if connect_db:
            self.db_conn = psycopg2.connect(
                host=db_host,
                port=db_port,
                dbname=db_name,
                user=db_user,
                password=db_password
            )
            self.db_cursor = self.db_conn.cursor()
            logger.info(f"✓ Connected to PostgreSQL database: {db_name}")

        logger.info(f"✓ Connected to {exchange_name} exchange")

    def _fetch_with_retry(self, func, *args, **kwargs):
        """
        Generic retry wrapper for CCXT calls to handle transient network/exchange errors.
        """
        from ccxt.base.errors import (
            ExchangeNotAvailable,
            NetworkError,
            RequestTimeout,
            DDoSProtection,
        )

        attempts = 0
        while True:
            try:
                return func(*args, **kwargs)
            except (ExchangeNotAvailable, NetworkError, RequestTimeout, DDoSProtection) as e:
                attempts += 1
                if attempts > self.max_retries:
                    logger.error(f"CCXT call failed after {attempts} attempts: {e}")
                    raise
                wait = self.retry_backoff * attempts
                logger.warning(f"CCXT transient error ({e}), retrying in {wait:.1f}s... [{attempts}/{self.max_retries}]")
                time.sleep(wait)
            except Exception:
                raise

    def fetch_ohlcv(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        since: datetime = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('1m', '5m', '1h', '1d')
            since: Start date (if None, fetches most recent)
            limit: Max number of candles to fetch

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            # Convert datetime to milliseconds timestamp
            since_ms = None
            if since:
                since_ms = int(since.timestamp() * 1000)

            logger.info(f"Fetching {symbol} {timeframe} data...")

            # Fetch from exchange
            ohlcv = self._fetch_with_retry(
                self.exchange.fetch_ohlcv,
                symbol=symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=limit
            )

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Convert timestamp from milliseconds to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            logger.success(f"✓ Fetched {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")

            return df

        except Exception as e:
            logger.error(f"✗ Error fetching OHLCV data: {e}")
            raise

    def save_to_db(self, df: pd.DataFrame, table_name: str = 'ohlcv_btcusdt_1h'):
        """
        Save OHLCV data to PostgreSQL TimescaleDB

        Args:
            df: DataFrame with OHLCV data
            table_name: Target table name
        """
        try:
            # Prepare data for insertion
            records = [
                (
                    row['timestamp'],
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume'])
                )
                for _, row in df.iterrows()
            ]

            # Insert with ON CONFLICT DO NOTHING (avoid duplicates)
            insert_query = f"""
                INSERT INTO {table_name} (timestamp, open, high, low, close, volume)
                VALUES %s
                ON CONFLICT (timestamp) DO NOTHING
            """

            execute_values(self.db_cursor, insert_query, records)
            self.db_conn.commit()

            # Get count of inserted rows
            inserted = self.db_cursor.rowcount
            logger.success(f"✓ Inserted {inserted} new candles into {table_name}")

        except Exception as e:
            self.db_conn.rollback()
            logger.error(f"✗ Error saving to database: {e}")
            raise

    def collect_historical_data(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        days: int = 365,
        table_name: str = 'ohlcv_btcusdt_1h'
    ):
        """
        Collect historical OHLCV data for the specified period

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            days: Number of days to collect (backwards from now)
            table_name: Database table name
        """
        logger.info(f"Starting historical data collection for {symbol}")
        logger.info(f"Period: {days} days, Timeframe: {timeframe}")

        # Calculate time range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Determine batch size based on exchange limits
        batch_size = 1000  # Most exchanges limit to 1000 candles per request

        # Calculate timeframe in hours
        timeframe_hours = {
            '1m': 1/60,
            '5m': 5/60,
            '15m': 15/60,
            '1h': 1,
            '4h': 4,
            '1d': 24
        }
        tf_hours = timeframe_hours.get(timeframe, 1)

        # Calculate total batches needed
        total_candles_needed = int((end_date - start_date).total_seconds() / 3600 / tf_hours)
        total_batches = (total_candles_needed // batch_size) + 1

        logger.info(f"Estimated total candles: {total_candles_needed}")
        logger.info(f"Will fetch in {total_batches} batches")

        # Collect in batches
        current_date = start_date
        total_inserted = 0

        for batch_num in range(total_batches):
            try:
                # Fetch batch
                df = self.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_date,
                    limit=batch_size
                )

                if df.empty:
                    logger.warning("No more data available")
                    break

                # Save to database
                self.save_to_db(df, table_name)
                total_inserted += len(df)

                # Update current_date to last fetched timestamp
                current_date = df['timestamp'].max() + timedelta(hours=tf_hours)

                # Progress
                progress_pct = ((batch_num + 1) / total_batches) * 100
                logger.info(f"Progress: {progress_pct:.1f}% ({batch_num + 1}/{total_batches} batches)")

                # Check if we've reached the end
                if current_date >= end_date:
                    logger.success("Reached current date, collection complete!")
                    break

                # Rate limiting (respect exchange limits)
                time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {e}")
                logger.warning("Retrying in 5 seconds...")
                time.sleep(5)
                continue

        logger.success(f"✓ Historical data collection complete!")
        logger.success(f"✓ Total candles collected: {total_inserted}")

        # Show summary statistics
        self.show_data_summary(table_name)

    def show_data_summary(self, table_name: str = 'ohlcv_btcusdt_1h'):
        """Display summary statistics of collected data"""
        try:
            query = f"""
                SELECT
                    COUNT(*) as total_candles,
                    MIN(timestamp) as start_date,
                    MAX(timestamp) as end_date,
                    MIN(low) as all_time_low,
                    MAX(high) as all_time_high,
                    AVG(volume) as avg_volume
                FROM {table_name}
            """

            self.db_cursor.execute(query)
            result = self.db_cursor.fetchone()

            if result:
                total, start, end, low, high, avg_vol = result
                logger.info("=" * 60)
                logger.info("DATA SUMMARY")
                logger.info("=" * 60)
                logger.info(f"Total Candles: {total:,}")
                logger.info(f"Date Range: {start} to {end}")
                logger.info(f"All-Time Low: ${low:,.2f}")
                logger.info(f"All-Time High: ${high:,.2f}")
                logger.info(f"Average Volume: {avg_vol:,.2f}")
                logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Error getting summary: {e}")

    def close(self):
        """Close database connections"""
        self.db_cursor.close()
        self.db_conn.close()
        logger.info("✓ Database connections closed")


def main():
    """Main entry point for CLI usage"""
    parser = argparse.ArgumentParser(
        description='Collect cryptocurrency OHLCV data using CCXT'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
        help='Trading pair (default: BTC/USDT)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
        help='Candle timeframe (default: 1h)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days to collect (default: 365)'
    )
    parser.add_argument(
        '--exchange',
        type=str,
        default='binance',
        help='Exchange name (default: binance)'
    )
    parser.add_argument(
        '--db-host',
        type=str,
        default='localhost',
        help='PostgreSQL host (default: localhost)'
    )

    args = parser.parse_args()

    # Initialize collector
    collector = CCXTDataCollector(
        exchange_name=args.exchange,
        db_host=args.db_host
    )

    try:
        # Determine table name from symbol and timeframe
        table_name = f"ohlcv_{args.symbol.lower().replace('/', '')}_" + args.timeframe

        # Collect historical data
        collector.collect_historical_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            days=args.days,
            table_name=table_name
        )

    except KeyboardInterrupt:
        logger.warning("\n✗ Collection interrupted by user")
    except Exception as e:
        logger.error(f"✗ Fatal error: {e}")
        sys.exit(1)
    finally:
        collector.close()


if __name__ == '__main__':
    main()
