"""
News Data Collector for HATS Trading System
Collects cryptocurrency news and sentiment data from CryptoPanic API
"""

import requests
from pymongo import MongoClient
from datetime import datetime, timedelta
import time
import argparse
from typing import List, Dict, Optional
from loguru import logger
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")


class NewsCollector:
    """Collects cryptocurrency news from CryptoPanic API and stores in MongoDB"""

    def __init__(
        self,
        api_token: Optional[str] = None,
        mongo_host: str = 'localhost',
        mongo_port: int = 27017,
        mongo_user: str = 'hats_user',
        mongo_password: str = 'hats_password',
        db_name: str = 'hats_trading'
    ):
        """
        Initialize News Collector

        Args:
            api_token: CryptoPanic API token (from env or parameter)
            mongo_host: MongoDB host
            mongo_port: MongoDB port
            mongo_user: MongoDB username
            mongo_password: MongoDB password
            db_name: Database name
        """
        # Get API token from environment or parameter
        self.api_token = api_token or os.getenv('CRYPTOPANIC_API_TOKEN')

        if not self.api_token:
            logger.warning("No CryptoPanic API token provided. Get one at https://cryptopanic.com/developers/api/")
            logger.warning("Set CRYPTOPANIC_API_TOKEN in .env file or pass via --api-token")

        # Use v2 endpoint if specified, otherwise v1
        self.base_url = os.getenv('CRYPTOPANIC_API_ENDPOINT', 'https://cryptopanic.com/api/v1/posts/')
        if not self.base_url.endswith('/'):
            self.base_url += '/'
        self.base_url += 'posts/' if 'v2' in self.base_url else ''

        # MongoDB connection
        connection_string = f'mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/'
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.news_collection = self.db['news']

        logger.info(f"✓ Connected to MongoDB: {db_name}")
        logger.info(f"✓ Using collection: news")

    def fetch_news(
        self,
        currencies: str = 'BTC',
        kind: str = 'news',
        public: bool = True,
        page: int = 1
    ) -> Dict:
        """
        Fetch news from CryptoPanic API

        Args:
            currencies: Comma-separated currency codes (e.g., 'BTC,ETH')
            kind: Type of posts ('news', 'media', 'all')
            public: Include only public posts
            page: Page number for pagination

        Returns:
            API response as dictionary
        """
        try:
            params = {
                'auth_token': self.api_token,
                'currencies': currencies,
                'kind': kind,
                'public': 'true' if public else 'false',
                'page': page
            }

            logger.info(f"Fetching news: currencies={currencies}, kind={kind}, page={page}")

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if 'results' in data:
                logger.success(f"✓ Fetched {len(data['results'])} news items")
            else:
                logger.warning("No results in API response")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Error fetching news: {e}")
            raise

    def process_news_item(self, item: Dict) -> Dict:
        """
        Process and normalize a news item for MongoDB storage

        Args:
            item: Raw news item from API

        Returns:
            Processed news item
        """
        # Extract published_at timestamp
        published_at = item.get('published_at')
        if published_at:
            published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        else:
            published_at = datetime.now()

        # Extract sentiment votes
        votes = item.get('votes', {})

        processed = {
            'published_at': published_at,
            'title': item.get('title', ''),
            'body': item.get('body', ''),
            'source': item.get('source', {}).get('title', 'Unknown'),
            'url': item.get('url', ''),
            'currencies': [c['code'] for c in item.get('currencies', [])],
            'sentiment': {
                'score': None,  # Will be calculated from votes
                'votes_positive': votes.get('positive', 0),
                'votes_negative': votes.get('negative', 0),
                'votes_neutral': votes.get('important', 0)  # 'important' acts as neutral
            },
            'metadata': {
                'kind': item.get('kind', ''),
                'domain': item.get('domain', ''),
                'created_at': item.get('created_at', ''),
                'cryptopanic_id': item.get('id', ''),
                'tags': item.get('tags', [])
            },
            'collected_at': datetime.now()
        }

        # Calculate sentiment score from votes (-1 to 1)
        total_votes = (
            processed['sentiment']['votes_positive'] +
            processed['sentiment']['votes_negative'] +
            processed['sentiment']['votes_neutral']
        )

        if total_votes > 0:
            positive = processed['sentiment']['votes_positive']
            negative = processed['sentiment']['votes_negative']
            neutral = processed['sentiment']['votes_neutral']

            # Score calculation: (positive - negative) / total
            # Neutral votes don't affect direction
            processed['sentiment']['score'] = (positive - negative) / total_votes
        else:
            processed['sentiment']['score'] = 0.0

        return processed

    def save_to_db(self, news_items: List[Dict], update_existing: bool = False):
        """
        Save news items to MongoDB

        Args:
            news_items: List of processed news items
            update_existing: If True, update existing items; if False, skip duplicates
        """
        try:
            if not news_items:
                logger.warning("No news items to save")
                return

            inserted_count = 0
            updated_count = 0
            skipped_count = 0

            for item in news_items:
                # Check if item already exists (by CryptoPanic ID)
                cryptopanic_id = item['metadata'].get('cryptopanic_id')

                if cryptopanic_id:
                    existing = self.news_collection.find_one(
                        {'metadata.cryptopanic_id': cryptopanic_id}
                    )

                    if existing:
                        if update_existing:
                            self.news_collection.update_one(
                                {'metadata.cryptopanic_id': cryptopanic_id},
                                {'$set': item}
                            )
                            updated_count += 1
                        else:
                            skipped_count += 1
                    else:
                        self.news_collection.insert_one(item)
                        inserted_count += 1
                else:
                    # No ID, insert anyway
                    self.news_collection.insert_one(item)
                    inserted_count += 1

            logger.success(f"✓ Inserted: {inserted_count}, Updated: {updated_count}, Skipped: {skipped_count}")

        except Exception as e:
            logger.error(f"✗ Error saving to database: {e}")
            raise

    def collect_historical_news(
        self,
        currencies: str = 'BTC',
        days: int = 7,
        kind: str = 'news',
        max_pages: int = 10
    ):
        """
        Collect historical news for the specified period

        Args:
            currencies: Comma-separated currency codes
            days: Number of days to collect (backwards from now)
            kind: Type of posts ('news', 'media', 'all')
            max_pages: Maximum number of pages to fetch
        """
        if not self.api_token:
            logger.error("✗ Cannot collect news without API token")
            return

        logger.info(f"Starting historical news collection")
        logger.info(f"Currencies: {currencies}, Period: {days} days, Kind: {kind}")

        cutoff_date = datetime.now(datetime.now().astimezone().tzinfo) - timedelta(days=days)
        total_collected = 0
        page = 1

        try:
            while page <= max_pages:
                # Fetch page
                data = self.fetch_news(
                    currencies=currencies,
                    kind=kind,
                    public=True,
                    page=page
                )

                results = data.get('results', [])

                if not results:
                    logger.warning("No more results available")
                    break

                # Process items
                processed_items = []
                reached_cutoff = False

                for item in results:
                    processed = self.process_news_item(item)

                    # Check if we've reached the cutoff date
                    if processed['published_at'] < cutoff_date:
                        reached_cutoff = True
                        break

                    processed_items.append(processed)

                # Save to database
                if processed_items:
                    self.save_to_db(processed_items, update_existing=False)
                    total_collected += len(processed_items)

                # Check if we should stop
                if reached_cutoff:
                    logger.info(f"Reached cutoff date: {cutoff_date}")
                    break

                # Check if there's a next page
                if not data.get('next'):
                    logger.info("No more pages available")
                    break

                # Progress
                logger.info(f"Progress: Page {page}/{max_pages}, Total collected: {total_collected}")

                # Rate limiting (CryptoPanic has limits)
                time.sleep(1)  # 1 request per second to be safe
                page += 1

            logger.success(f"✓ Historical news collection complete!")
            logger.success(f"✓ Total items collected: {total_collected}")

            # Show summary
            self.show_data_summary(currencies)

        except KeyboardInterrupt:
            logger.warning("\n✗ Collection interrupted by user")
            logger.info(f"Collected {total_collected} items before interruption")
        except Exception as e:
            logger.error(f"✗ Error during collection: {e}")
            raise

    def show_data_summary(self, currencies: str = 'BTC'):
        """Display summary statistics of collected news"""
        try:
            currency_list = currencies.split(',')

            # Overall statistics
            total_count = self.news_collection.count_documents({})

            # Statistics for specific currencies
            currency_count = self.news_collection.count_documents({
                'currencies': {'$in': currency_list}
            })

            # Date range
            oldest = self.news_collection.find_one(
                sort=[('published_at', 1)]
            )
            newest = self.news_collection.find_one(
                sort=[('published_at', -1)]
            )

            # Sentiment statistics
            pipeline = [
                {'$match': {'currencies': {'$in': currency_list}}},
                {'$group': {
                    '_id': None,
                    'avg_sentiment': {'$avg': '$sentiment.score'},
                    'total_positive': {'$sum': '$sentiment.votes_positive'},
                    'total_negative': {'$sum': '$sentiment.votes_negative'}
                }}
            ]

            sentiment_stats = list(self.news_collection.aggregate(pipeline))

            logger.info("=" * 60)
            logger.info("NEWS DATA SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total News Items: {total_count:,}")
            logger.info(f"Items for {currencies}: {currency_count:,}")

            if oldest and newest:
                logger.info(f"Date Range: {oldest['published_at']} to {newest['published_at']}")

            if sentiment_stats:
                avg_sentiment = sentiment_stats[0].get('avg_sentiment', 0)
                total_pos = sentiment_stats[0].get('total_positive', 0)
                total_neg = sentiment_stats[0].get('total_negative', 0)

                logger.info(f"Average Sentiment: {avg_sentiment:.3f}")
                logger.info(f"Total Positive Votes: {total_pos:,}")
                logger.info(f"Total Negative Votes: {total_neg:,}")

            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Error getting summary: {e}")

    def close(self):
        """Close MongoDB connection"""
        self.client.close()
        logger.info("✓ MongoDB connection closed")


def main():
    """Main entry point for CLI usage"""
    parser = argparse.ArgumentParser(
        description='Collect cryptocurrency news from CryptoPanic API'
    )
    parser.add_argument(
        '--currencies',
        type=str,
        default='BTC',
        help='Comma-separated currency codes (default: BTC)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to collect (default: 7)'
    )
    parser.add_argument(
        '--kind',
        type=str,
        default='news',
        choices=['news', 'media', 'all'],
        help='Type of posts to collect (default: news)'
    )
    parser.add_argument(
        '--max-pages',
        type=int,
        default=10,
        help='Maximum pages to fetch (default: 10)'
    )
    parser.add_argument(
        '--api-token',
        type=str,
        help='CryptoPanic API token (or set CRYPTOPANIC_API_TOKEN in .env)'
    )
    parser.add_argument(
        '--mongo-host',
        type=str,
        default='localhost',
        help='MongoDB host (default: localhost)'
    )

    args = parser.parse_args()

    # Initialize collector
    collector = NewsCollector(
        api_token=args.api_token,
        mongo_host=args.mongo_host
    )

    try:
        # Collect historical news
        collector.collect_historical_news(
            currencies=args.currencies,
            days=args.days,
            kind=args.kind,
            max_pages=args.max_pages
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
