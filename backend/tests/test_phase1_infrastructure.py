"""
Phase 1 Infrastructure Validation Tests
Tests all components of the HATS trading infrastructure
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from pymongo import MongoClient
import redis
import ccxt
from datetime import datetime, timedelta
from loguru import logger
import time

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)


class Phase1InfrastructureTest:
    """Comprehensive infrastructure validation"""

    def __init__(self):
        self.test_results = {
            'postgres': False,
            'mongodb': False,
            'redis': False,
            'ccxt': False,
            'postgres_tables': False,
            'mongodb_collections': False,
            'timescaledb': False,
            'data_insertion': False
        }

    def test_postgres_connection(self):
        """Test PostgreSQL + TimescaleDB connection"""
        logger.info("=" * 60)
        logger.info("TEST 1: PostgreSQL Connection")
        logger.info("=" * 60)

        try:
            conn = psycopg2.connect(
                host='localhost',
                port=5432,
                dbname='hats_trading',
                user='hats_user',
                password='hats_password'
            )
            cursor = conn.cursor()

            # Test basic query
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            logger.success(f"✓ PostgreSQL connected: {version[:50]}...")

            # Check TimescaleDB extension
            cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb';")
            result = cursor.fetchone()

            if result:
                logger.success("✓ TimescaleDB extension enabled")
                self.test_results['timescaledb'] = True
            else:
                logger.warning("✗ TimescaleDB extension not found")

            cursor.close()
            conn.close()

            self.test_results['postgres'] = True
            return True

        except Exception as e:
            logger.error(f"✗ PostgreSQL connection failed: {e}")
            return False

    def test_postgres_tables(self):
        """Test PostgreSQL table creation"""
        logger.info("=" * 60)
        logger.info("TEST 2: PostgreSQL Tables")
        logger.info("=" * 60)

        try:
            conn = psycopg2.connect(
                host='localhost',
                port=5432,
                dbname='hats_trading',
                user='hats_user',
                password='hats_password'
            )
            cursor = conn.cursor()

            # Expected tables
            expected_tables = [
                'ohlcv_btcusdt_1h',
                'trades',
                'portfolio_snapshots',
                'backtest_results',
                'document_embeddings'
            ]

            # Get actual tables
            cursor.execute("""
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
            """)
            actual_tables = [row[0] for row in cursor.fetchall()]

            # Check each expected table
            all_found = True
            for table in expected_tables:
                if table in actual_tables:
                    logger.success(f"✓ Table exists: {table}")
                else:
                    logger.error(f"✗ Table missing: {table}")
                    all_found = False

            # Check hypertable for ohlcv_btcusdt_1h
            cursor.execute("""
                SELECT hypertable_name
                FROM timescaledb_information.hypertables
                WHERE hypertable_name = 'ohlcv_btcusdt_1h'
            """)
            result = cursor.fetchone()

            if result:
                logger.success("✓ ohlcv_btcusdt_1h is a TimescaleDB hypertable")
            else:
                logger.warning("✗ ohlcv_btcusdt_1h is not a hypertable")
                all_found = False

            cursor.close()
            conn.close()

            self.test_results['postgres_tables'] = all_found
            return all_found

        except Exception as e:
            logger.error(f"✗ Table check failed: {e}")
            return False

    def test_mongodb_connection(self):
        """Test MongoDB connection"""
        logger.info("=" * 60)
        logger.info("TEST 3: MongoDB Connection")
        logger.info("=" * 60)

        try:
            client = MongoClient(
                'mongodb://hats_user:hats_password@localhost:27017/'
            )

            # Test connection
            client.admin.command('ping')
            logger.success("✓ MongoDB connected")

            # Check database
            db = client['hats_trading']
            logger.success(f"✓ Database 'hats_trading' accessible")

            client.close()

            self.test_results['mongodb'] = True
            return True

        except Exception as e:
            logger.error(f"✗ MongoDB connection failed: {e}")
            return False

    def test_mongodb_collections(self):
        """Test MongoDB collections"""
        logger.info("=" * 60)
        logger.info("TEST 4: MongoDB Collections")
        logger.info("=" * 60)

        try:
            client = MongoClient(
                'mongodb://hats_user:hats_password@localhost:27017/'
            )
            db = client['hats_trading']

            # Expected collections
            expected_collections = [
                'reasoning_logs',
                'news',
                'agent_checkpoints',
                'backtest_metadata'
            ]

            # Get actual collections
            actual_collections = db.list_collection_names()

            # Check each expected collection
            all_found = True
            for collection in expected_collections:
                if collection in actual_collections:
                    logger.success(f"✓ Collection exists: {collection}")

                    # Check indexes
                    indexes = db[collection].index_information()
                    logger.info(f"  Indexes: {len(indexes)} index(es)")
                else:
                    logger.error(f"✗ Collection missing: {collection}")
                    all_found = False

            client.close()

            self.test_results['mongodb_collections'] = all_found
            return all_found

        except Exception as e:
            logger.error(f"✗ Collection check failed: {e}")
            return False

    def test_redis_connection(self):
        """Test Redis connection"""
        logger.info("=" * 60)
        logger.info("TEST 5: Redis Connection")
        logger.info("=" * 60)

        try:
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)

            # Test connection
            r.ping()
            logger.success("✓ Redis connected")

            # Test set/get
            r.set('test_key', 'test_value')
            value = r.get('test_key')

            if value == 'test_value':
                logger.success("✓ Redis read/write works")
            else:
                logger.error("✗ Redis read/write failed")
                return False

            # Clean up
            r.delete('test_key')

            # Check Redis info
            info = r.info()
            logger.info(f"  Redis version: {info['redis_version']}")
            logger.info(f"  Used memory: {info['used_memory_human']}")

            self.test_results['redis'] = True
            return True

        except Exception as e:
            logger.error(f"✗ Redis connection failed: {e}")
            return False

    def test_ccxt_exchange(self):
        """Test CCXT exchange connectivity"""
        logger.info("=" * 60)
        logger.info("TEST 6: CCXT Exchange Connectivity")
        logger.info("=" * 60)

        try:
            # Initialize Binance exchange
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })

            # Load markets
            markets = exchange.load_markets()
            logger.success(f"✓ CCXT connected to Binance")
            logger.info(f"  Available markets: {len(markets)}")

            # Check BTC/USDT market
            if 'BTC/USDT' in markets:
                logger.success("✓ BTC/USDT market available")
            else:
                logger.error("✗ BTC/USDT market not found")
                return False

            # Test fetching OHLCV (just 1 candle)
            ohlcv = exchange.fetch_ohlcv(
                symbol='BTC/USDT',
                timeframe='1h',
                limit=1
            )

            if len(ohlcv) > 0:
                logger.success(f"✓ Successfully fetched OHLCV data")
                logger.info(f"  Latest candle: {datetime.fromtimestamp(ohlcv[0][0] / 1000)}")
                logger.info(f"  Close price: ${ohlcv[0][4]:,.2f}")
            else:
                logger.error("✗ Failed to fetch OHLCV data")
                return False

            self.test_results['ccxt'] = True
            return True

        except Exception as e:
            logger.error(f"✗ CCXT test failed: {e}")
            return False

    def test_data_insertion(self):
        """Test data insertion and retrieval"""
        logger.info("=" * 60)
        logger.info("TEST 7: Data Insertion & Retrieval")
        logger.info("=" * 60)

        try:
            # Test PostgreSQL insertion
            logger.info("Testing PostgreSQL insertion...")
            conn = psycopg2.connect(
                host='localhost',
                port=5432,
                dbname='hats_trading',
                user='hats_user',
                password='hats_password'
            )
            cursor = conn.cursor()

            # Insert test OHLCV data
            test_timestamp = datetime.now().replace(minute=0, second=0, microsecond=0)
            cursor.execute("""
                INSERT INTO ohlcv_btcusdt_1h (timestamp, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (timestamp) DO NOTHING
            """, (test_timestamp, 50000.0, 51000.0, 49000.0, 50500.0, 100.5))

            conn.commit()
            logger.success("✓ PostgreSQL test data inserted")

            # Retrieve test data
            cursor.execute("""
                SELECT * FROM ohlcv_btcusdt_1h WHERE timestamp = %s
            """, (test_timestamp,))
            result = cursor.fetchone()

            if result:
                logger.success("✓ PostgreSQL test data retrieved")
            else:
                logger.error("✗ PostgreSQL test data not found")
                return False

            cursor.close()
            conn.close()

            # Test MongoDB insertion
            logger.info("Testing MongoDB insertion...")
            client = MongoClient(
                'mongodb://hats_user:hats_password@localhost:27017/'
            )
            db = client['hats_trading']

            # Insert test news
            test_news = {
                'published_at': datetime.now(),
                'title': 'Test News Article',
                'body': 'This is a test news article for infrastructure validation.',
                'source': 'Test Source',
                'url': 'https://example.com/test',
                'currencies': ['BTC'],
                'sentiment': {
                    'score': 0.5,
                    'votes_positive': 10,
                    'votes_negative': 5,
                    'votes_neutral': 3
                },
                'metadata': {
                    'test': True
                }
            }

            result = db.news.insert_one(test_news)
            logger.success("✓ MongoDB test data inserted")

            # Retrieve test data
            retrieved = db.news.find_one({'_id': result.inserted_id})

            if retrieved:
                logger.success("✓ MongoDB test data retrieved")
            else:
                logger.error("✗ MongoDB test data not found")
                return False

            # Clean up test data
            db.news.delete_one({'_id': result.inserted_id})

            client.close()

            self.test_results['data_insertion'] = True
            return True

        except Exception as e:
            logger.error(f"✗ Data insertion test failed: {e}")
            return False

    def run_all_tests(self):
        """Run all infrastructure tests"""
        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║" + " " * 10 + "HATS TRADING - PHASE 1 INFRASTRUCTURE TEST" + " " * 6 + "║")
        logger.info("╚" + "═" * 58 + "╝")
        logger.info("")

        start_time = time.time()

        # Run tests in order
        tests = [
            ('PostgreSQL Connection', self.test_postgres_connection),
            ('PostgreSQL Tables', self.test_postgres_tables),
            ('MongoDB Connection', self.test_mongodb_connection),
            ('MongoDB Collections', self.test_mongodb_collections),
            ('Redis Connection', self.test_redis_connection),
            ('CCXT Exchange', self.test_ccxt_exchange),
            ('Data Insertion', self.test_data_insertion),
        ]

        for test_name, test_func in tests:
            try:
                test_func()
                time.sleep(0.5)  # Brief pause between tests
            except Exception as e:
                logger.error(f"✗ Test '{test_name}' crashed: {e}")

        # Summary
        elapsed_time = time.time() - start_time

        logger.info("")
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)

        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)

        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"{status:8} | {test_name}")

        logger.info("=" * 60)
        logger.info(f"Results: {passed}/{total} tests passed")
        logger.info(f"Time: {elapsed_time:.2f} seconds")
        logger.info("=" * 60)

        if passed == total:
            logger.success("")
            logger.success("🎉 ALL TESTS PASSED! Infrastructure is ready.")
            logger.success("")
            logger.success("Next steps:")
            logger.success("1. Collect historical data: python backend/data/ccxt_collector.py --days 365")
            logger.success("2. Collect news data: python backend/data/news_collector.py --days 7")
            logger.success("3. Proceed to Phase 2: LangGraph Agent Foundation")
            logger.success("")
            return True
        else:
            logger.error("")
            logger.error("❌ SOME TESTS FAILED")
            logger.error("")
            logger.error("Please check:")
            logger.error("1. Docker containers are running: docker-compose ps")
            logger.error("2. Check logs: docker-compose logs")
            logger.error("3. Restart containers: docker-compose restart")
            logger.error("4. See README_PHASE1.md for troubleshooting")
            logger.error("")
            return False


def main():
    """Main entry point"""
    tester = Phase1InfrastructureTest()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
