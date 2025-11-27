"""
Sentiment Analyst Module
Analyzes news sentiment using Gemini 2.5 Pro with Financial Chain-of-Thought

Provides:
- News sentiment analysis (-1.0 to 1.0)
- Sentiment label classification (Very Negative ~ Very Positive)
- Batch processing for efficiency
- Financial domain-specific reasoning

Future Migration Path:
- Option 1: FinGPT local model (Phase 4+)
  - GPU-based inference
  - No API costs
  - Domain-specialized
"""

import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Rate Limiter
from .rate_limiter import wait_for_rate_limit


# ============================================================================
# Pydantic Models
# ============================================================================

class NewsSentiment(BaseModel):
    """Individual news article sentiment"""
    title: str = Field(description="News title")
    sentiment_score: float = Field(
        description="Sentiment score from -1.0 (very negative) to 1.0 (very positive)",
        ge=-1.0,
        le=1.0
    )
    sentiment_label: str = Field(
        description="Sentiment label: Very Negative, Negative, Neutral, Positive, Very Positive"
    )
    reasoning: str = Field(description="Chain-of-thought reasoning for the sentiment")
    impact: str = Field(description="Expected market impact: Bearish, Neutral, Bullish")


class SentimentAnalysisResult(BaseModel):
    """Complete sentiment analysis result"""
    news_sentiments: List[NewsSentiment] = Field(description="Individual news sentiments")
    average_score: float = Field(
        description="Average sentiment score",
        ge=-1.0,
        le=1.0
    )
    overall_label: str = Field(description="Overall sentiment label")
    summary: str = Field(description="Summary of news sentiment trends")


# ============================================================================
# Helper Functions
# ============================================================================

def get_sentiment_label(score: float) -> str:
    """
    Convert sentiment score to label

    Args:
        score: Sentiment score (-1.0 to 1.0)

    Returns:
        Sentiment label
    """
    if score >= 0.6:
        return "Very Positive"
    elif score >= 0.2:
        return "Positive"
    elif score >= -0.2:
        return "Neutral"
    elif score >= -0.6:
        return "Negative"
    else:
        return "Very Negative"


def get_market_impact(score: float) -> str:
    """
    Convert sentiment score to market impact

    Args:
        score: Sentiment score (-1.0 to 1.0)

    Returns:
        Market impact: Bearish, Neutral, Bullish
    """
    if score >= 0.3:
        return "Bullish"
    elif score >= -0.3:
        return "Neutral"
    else:
        return "Bearish"


# ============================================================================
# LLM Setup
# ============================================================================

def get_sentiment_llm() -> ChatGoogleGenerativeAI:
    """
    Get Gemini 2.5 Flash LLM for sentiment analysis

    Uses lower temperature for consistent sentiment scoring

    Note: Using 2.5-Flash instead of 2.5-Pro for free tier quota management
    - Free tier: Flash (250/day, 10/min) vs Pro (50/day, 2/min)
    - Sentiment analysis works well with Flash model
    - Upgrade to Pro available when using paid tier

    Rate Limiting:
    - Automatically applies 6s delay between calls (free tier)
    """
    # Apply rate limiting (free tier: 10 calls/min, 250/day)
    wait_for_rate_limit("gemini-2.5-flash")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,  # Lower temperature for consistency
        google_api_key=api_key,
        convert_system_message_to_human=True
    )


# ============================================================================
# Financial Chain-of-Thought Prompt
# ============================================================================

FINANCIAL_COT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a professional financial sentiment analyst specializing in cryptocurrency markets.

Your task is to analyze news articles and determine their sentiment impact on the crypto market.

**Analysis Framework (Chain-of-Thought):**
1. **Content Understanding**: What is the main topic and key information?
2. **Market Relevance**: How does this relate to crypto prices/adoption/regulation?
3. **Directional Impact**: Is this bullish, bearish, or neutral for the market?
4. **Sentiment Scoring**: Assign a precise score from -1.0 to 1.0

**Scoring Guidelines:**
- **+0.8 to +1.0**: Very Positive (major bullish catalyst - ETF approval, mass adoption, bullish regulation)
- **+0.4 to +0.8**: Positive (good news - institutional investment, positive tech development)
- **+0.2 to +0.4**: Slightly Positive (minor good news)
- **-0.2 to +0.2**: Neutral (informational, mixed signals)
- **-0.4 to -0.2**: Slightly Negative (minor concerns)
- **-0.8 to -0.4**: Negative (bad news - exchange issues, negative regulation)
- **-1.0 to -0.8**: Very Negative (major bearish catalyst - ban, major hack, crash)

Be precise and consistent. Always provide clear reasoning."""),
    ("user", """Analyze the sentiment of these cryptocurrency news articles:

**Symbol**: {symbol}
**Date**: {date}

**News Articles**:
{news_list}

For each article, provide:
1. Chain-of-thought reasoning
2. Sentiment score (-1.0 to 1.0)
3. Sentiment label
4. Expected market impact

Return your analysis in the following JSON format:
{{
  "news_sentiments": [
    {{
      "title": "article title",
      "sentiment_score": 0.5,
      "sentiment_label": "Positive",
      "reasoning": "step-by-step analysis...",
      "impact": "Bullish"
    }}
  ],
  "average_score": 0.3,
  "overall_label": "Positive",
  "summary": "Brief summary of overall sentiment trends"
}}
""")
])


# ============================================================================
# Main Sentiment Analysis Function
# ============================================================================

def analyze_news_sentiment(
    news_data: List[Dict[str, Any]],
    symbol: str = "BTC/USDT"
) -> Dict[str, Any]:
    """
    Analyze sentiment of news articles using Gemini 2.5 Pro

    Args:
        news_data: List of news articles
                   [{'title': str, 'source': str, 'published_at': str, ...}, ...]
        symbol: Trading symbol (for context)

    Returns:
        {
            'news_sentiments': [
                {
                    'title': str,
                    'sentiment_score': float,
                    'sentiment_label': str,
                    'reasoning': str,
                    'impact': str
                },
                ...
            ],
            'average_score': float,
            'overall_label': str,
            'summary': str
        }
    """
    if not news_data or len(news_data) == 0:
        logger.warning("No news data provided for sentiment analysis")
        return {
            'news_sentiments': [],
            'average_score': 0.0,
            'overall_label': 'Neutral',
            'summary': 'No recent news available'
        }

    # Limit to most recent 10 news for efficiency
    news_data = news_data[:10]

    # Format news list for prompt
    from datetime import datetime
    news_list = ""
    for i, news in enumerate(news_data, 1):
        title = news.get('title', 'No title')
        source = news.get('source', 'Unknown')
        published = news.get('published_at', 'Unknown date')
        news_list += f"{i}. [{source}] {title} ({published})\n"

    try:
        # Get LLM
        llm = get_sentiment_llm()

        # Setup parser
        parser = JsonOutputParser(pydantic_object=SentimentAnalysisResult)

        # Build chain
        chain = FINANCIAL_COT_PROMPT | llm | parser

        # Invoke
        logger.info(f"Analyzing sentiment for {len(news_data)} news articles...")
        result = chain.invoke({
            'symbol': symbol,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'news_list': news_list
        })

        logger.info(f"Sentiment analysis complete: Avg score = {result['average_score']:.3f} ({result['overall_label']})")

        return result

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")

        # Fallback: neutral sentiment
        fallback_sentiments = []
        for news in news_data:
            fallback_sentiments.append({
                'title': news.get('title', 'No title'),
                'sentiment_score': 0.0,
                'sentiment_label': 'Neutral',
                'reasoning': f'Analysis failed: {str(e)}',
                'impact': 'Neutral'
            })

        return {
            'news_sentiments': fallback_sentiments,
            'average_score': 0.0,
            'overall_label': 'Neutral',
            'summary': f'Sentiment analysis unavailable: {str(e)}'
        }


# ============================================================================
# Batch Processing (Future Enhancement)
# ============================================================================

def analyze_news_sentiment_batch(
    news_batches: List[List[Dict[str, Any]]],
    symbol: str = "BTC/USDT"
) -> List[Dict[str, Any]]:
    """
    Analyze multiple batches of news (for large datasets)

    Args:
        news_batches: List of news batches
        symbol: Trading symbol

    Returns:
        List of sentiment analysis results
    """
    results = []
    for i, batch in enumerate(news_batches):
        logger.info(f"Processing batch {i+1}/{len(news_batches)}...")
        result = analyze_news_sentiment(batch, symbol)
        results.append(result)

    return results


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    """
    Test sentiment analysis with sample news
    """
    sample_news = [
        {
            'title': 'Bitcoin ETF approval expected next week',
            'source': 'CoinDesk',
            'published_at': '2025-11-27'
        },
        {
            'title': 'Major exchange hacked, $100M stolen',
            'source': 'CryptoNews',
            'published_at': '2025-11-27'
        },
        {
            'title': 'Ethereum upgrade successful',
            'source': 'Ethereum Foundation',
            'published_at': '2025-11-26'
        },
        {
            'title': 'SEC delays decision on crypto regulation',
            'source': 'Bloomberg',
            'published_at': '2025-11-26'
        },
        {
            'title': 'Institutional adoption continues to grow',
            'source': 'Financial Times',
            'published_at': '2025-11-25'
        }
    ]

    print("=" * 80)
    print("Testing Sentiment Analysis with Gemini 2.5 Pro")
    print("=" * 80)

    result = analyze_news_sentiment(sample_news, symbol="BTC/USDT")

    print(f"\n[OVERALL] Sentiment: {result['overall_label']} (Score: {result['average_score']:.3f})")
    print(f"[SUMMARY] {result['summary']}\n")

    print("Individual News Sentiments:")
    print("-" * 80)
    for i, news_sentiment in enumerate(result['news_sentiments'], 1):
        print(f"\n{i}. {news_sentiment['title']}")
        print(f"   Score: {news_sentiment['sentiment_score']:+.2f} | Label: {news_sentiment['sentiment_label']} | Impact: {news_sentiment['impact']}")
        print(f"   Reasoning: {news_sentiment['reasoning'][:150]}...")

    print("\n" + "=" * 80)
