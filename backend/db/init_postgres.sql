-- PostgreSQL initialization script for HATS Trading System

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Enable pgvector extension for RAG
CREATE EXTENSION IF NOT EXISTS vector;

-- Create OHLCV table for BTC/USDT 1-hour candles
CREATE TABLE IF NOT EXISTS ohlcv_btcusdt_1h (
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    PRIMARY KEY (timestamp)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('ohlcv_btcusdt_1h', 'timestamp', if_not_exists => TRUE);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp_desc ON ohlcv_btcusdt_1h (timestamp DESC);

-- Create OHLCV table for BTC/USDT perpetual swap (Binance futures) 1-hour candles
CREATE TABLE IF NOT EXISTS ohlcv_btcusdt_swap_1h (
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('ohlcv_btcusdt_swap_1h', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ohlcv_swap_timestamp_desc ON ohlcv_btcusdt_swap_1h (timestamp DESC);

-- Create OHLCV table for BTC/USDT perpetual swap (Binance futures) 1-minute candles
CREATE TABLE IF NOT EXISTS ohlcv_btcusdt_swap_1m (
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('ohlcv_btcusdt_swap_1m', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ohlcv_swap_1m_timestamp_desc ON ohlcv_btcusdt_swap_1m (timestamp DESC);

-- Create OHLCV table for BTC/USDT perpetual swap (Binance futures) 5-minute candles
CREATE TABLE IF NOT EXISTS ohlcv_btcusdt_swap_5m (
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('ohlcv_btcusdt_swap_5m', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ohlcv_swap_5m_timestamp_desc ON ohlcv_btcusdt_swap_5m (timestamp DESC);

-- Create OHLCV table for BTC/USDT 1-day candles (ML training)
CREATE TABLE IF NOT EXISTS ohlcv_btcusdt_1d (
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('ohlcv_btcusdt_1d', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ohlcv_1d_timestamp_desc ON ohlcv_btcusdt_1d (timestamp DESC);

-- Create OHLCV table for BTC/USDT 4-hour candles (ML training)
CREATE TABLE IF NOT EXISTS ohlcv_btcusdt_4h (
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('ohlcv_btcusdt_4h', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ohlcv_4h_timestamp_desc ON ohlcv_btcusdt_4h (timestamp DESC);

-- Create OHLCV table for BTC/USDT 15-minute candles (ML training)
CREATE TABLE IF NOT EXISTS ohlcv_btcusdt_15m (
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('ohlcv_btcusdt_15m', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ohlcv_15m_timestamp_desc ON ohlcv_btcusdt_15m (timestamp DESC);

-- Create OHLCV table for BTC/USDT 5-minute candles (ML training)
CREATE TABLE IF NOT EXISTS ohlcv_btcusdt_5m (
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('ohlcv_btcusdt_5m', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ohlcv_5m_timestamp_desc ON ohlcv_btcusdt_5m (timestamp DESC);

-- Create table for executed trades
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL CHECK (action IN ('BUY', 'SELL')),
    price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    total_value DECIMAL(20, 8) NOT NULL,
    fee DECIMAL(20, 8) DEFAULT 0,
    reasoning_log_id VARCHAR(100), -- MongoDB ObjectId reference
    strategy_name VARCHAR(50) DEFAULT 'agent',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol);

-- Create table for portfolio snapshots
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    cash_balance DECIMAL(20, 8) NOT NULL,
    btc_balance DECIMAL(20, 8) NOT NULL,
    total_value_usd DECIMAL(20, 8) NOT NULL,
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('portfolio_snapshots', 'timestamp', if_not_exists => TRUE);

-- Create table for backtest results
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) UNIQUE NOT NULL,
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ NOT NULL,
    initial_capital DECIMAL(20, 8) NOT NULL,
    final_capital DECIMAL(20, 8) NOT NULL,
    total_return DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    win_rate DECIMAL(10, 4),
    total_trades INTEGER,
    api_calls INTEGER,
    cache_hit_rate DECIMAL(10, 4),
    reasoning_log_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create table for vector embeddings (RAG)
CREATE TABLE IF NOT EXISTS document_embeddings (
    id SERIAL PRIMARY KEY,
    document_type VARCHAR(50) NOT NULL, -- 'news', 'whitepaper', 'report'
    content TEXT NOT NULL,
    embedding vector(768), -- sentence-transformers all-MiniLM-L6-v2
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embeddings_type ON document_embeddings (document_type);
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON document_embeddings USING ivfflat (embedding vector_cosine_ops);

-- Create view for recent market data
CREATE OR REPLACE VIEW recent_market_data AS
SELECT
    timestamp,
    open,
    high,
    low,
    close,
    volume,
    close - open AS price_change,
    ((close - open) / open * 100) AS price_change_pct
FROM ohlcv_btcusdt_1h
ORDER BY timestamp DESC
LIMIT 100;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO hats_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO hats_user;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'HATS Trading PostgreSQL database initialized successfully!';
    RAISE NOTICE 'TimescaleDB and pgvector extensions enabled.';
    RAISE NOTICE 'Tables created: ohlcv_btcusdt_1h, trades, portfolio_snapshots, backtest_results, document_embeddings';
END $$;
