// MongoDB initialization script for HATS Trading System

// Switch to trading database
db = db.getSiblingDB('hats_trading');

// Create collections with validation schemas

// 1. Reasoning logs collection (Agent's chain-of-thought)
db.createCollection('reasoning_logs', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['timestamp', 'reasoning_trace', 'proposed_trade'],
            properties: {
                timestamp: {
                    bsonType: 'date',
                    description: 'Timestamp of the reasoning session'
                },
                current_price: {
                    bsonType: 'number',
                    description: 'BTC price at the time'
                },
                technical_indicators: {
                    bsonType: 'object',
                    description: 'RSI, MACD, etc.'
                },
                sentiment_score: {
                    bsonType: 'number',
                    minimum: -1.0,
                    maximum: 1.0,
                    description: 'Sentiment score from news analysis'
                },
                reasoning_trace: {
                    bsonType: 'array',
                    description: 'Array of reasoning steps (CoT)',
                    items: {
                        bsonType: 'string'
                    }
                },
                debate_transcript: {
                    bsonType: 'array',
                    description: 'Bull vs Bear debate',
                    items: {
                        bsonType: 'object',
                        required: ['role', 'content'],
                        properties: {
                            role: {
                                bsonType: 'string',
                                enum: ['Bull', 'Bear']
                            },
                            content: {
                                bsonType: 'string'
                            }
                        }
                    }
                },
                proposed_trade: {
                    bsonType: 'object',
                    required: ['action'],
                    properties: {
                        action: {
                            bsonType: 'string',
                            enum: ['BUY', 'SELL', 'HOLD']
                        },
                        allocation: {
                            bsonType: 'number',
                            minimum: 0.0,
                            maximum: 1.0
                        },
                        confidence: {
                            bsonType: 'number',
                            minimum: 0.0,
                            maximum: 1.0
                        },
                        stop_loss_pct: {
                            bsonType: 'number'
                        },
                        take_profit_pct: {
                            bsonType: 'number'
                        },
                        reasoning: {
                            bsonType: 'string'
                        }
                    }
                },
                risk_approved: {
                    bsonType: 'bool',
                    description: 'Risk manager approval'
                },
                risk_feedback: {
                    bsonType: 'string'
                },
                trade_result: {
                    bsonType: 'object',
                    description: 'Actual trade outcome (added later)',
                    properties: {
                        executed: { bsonType: 'bool' },
                        entry_price: { bsonType: 'number' },
                        exit_price: { bsonType: 'number' },
                        pnl: { bsonType: 'number' },
                        pnl_pct: { bsonType: 'number' }
                    }
                }
            }
        }
    }
});

// 2. News collection
db.createCollection('news', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['published_at', 'title', 'source'],
            properties: {
                published_at: {
                    bsonType: 'date',
                    description: 'News publication timestamp'
                },
                title: {
                    bsonType: 'string',
                    description: 'News headline'
                },
                body: {
                    bsonType: 'string',
                    description: 'Full article text'
                },
                source: {
                    bsonType: 'string',
                    description: 'News source (e.g., CryptoPanic, Reuters)'
                },
                url: {
                    bsonType: 'string'
                },
                currencies: {
                    bsonType: 'array',
                    description: 'Related cryptocurrencies',
                    items: {
                        bsonType: 'string'
                    }
                },
                sentiment: {
                    bsonType: 'object',
                    properties: {
                        score: {
                            bsonType: 'number',
                            minimum: -1.0,
                            maximum: 1.0
                        },
                        votes_positive: { bsonType: 'int' },
                        votes_negative: { bsonType: 'int' },
                        votes_neutral: { bsonType: 'int' }
                    }
                },
                metadata: {
                    bsonType: 'object'
                }
            }
        }
    }
});

// 3. Agent state checkpoints (for resuming)
db.createCollection('agent_checkpoints', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['thread_id', 'state', 'checkpoint_at'],
            properties: {
                thread_id: {
                    bsonType: 'string',
                    description: 'LangGraph thread identifier'
                },
                state: {
                    bsonType: 'object',
                    description: 'Full AgentState snapshot'
                },
                checkpoint_at: {
                    bsonType: 'date'
                },
                checkpoint_namespace: {
                    bsonType: 'string'
                }
            }
        }
    }
});

// 4. Backtest analysis metadata
db.createCollection('backtest_metadata', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['run_id', 'parameters', 'created_at'],
            properties: {
                run_id: {
                    bsonType: 'string',
                    description: 'Links to PostgreSQL backtest_results'
                },
                parameters: {
                    bsonType: 'object',
                    description: 'Backtest configuration'
                },
                agent_config: {
                    bsonType: 'object',
                    description: 'LLM model, temperature, etc.'
                },
                created_at: {
                    bsonType: 'date'
                },
                notes: {
                    bsonType: 'string'
                }
            }
        }
    }
});

// Create indexes for performance
db.reasoning_logs.createIndex({ timestamp: -1 });
db.reasoning_logs.createIndex({ 'proposed_trade.action': 1 });
db.reasoning_logs.createIndex({ 'trade_result.pnl_pct': -1 });

db.news.createIndex({ published_at: -1 });
db.news.createIndex({ currencies: 1 });
db.news.createIndex({ source: 1 });
db.news.createIndex({ 'sentiment.score': -1 });

db.agent_checkpoints.createIndex({ thread_id: 1 }, { unique: true });
db.agent_checkpoints.createIndex({ checkpoint_at: -1 });

db.backtest_metadata.createIndex({ run_id: 1 }, { unique: true });
db.backtest_metadata.createIndex({ created_at: -1 });

print('HATS Trading MongoDB initialized successfully!');
print('Collections created: reasoning_logs, news, agent_checkpoints, backtest_metadata');
print('Indexes created for optimal query performance');
