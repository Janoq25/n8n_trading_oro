-- Esquema para gold_trading_recommendations
CREATE TABLE gold_trading_recommendations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    execution_id TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    investment_amount DECIMAL(12,2),
    risk_tolerance TEXT,
    current_price DECIMAL(10,2),
    price_change DECIMAL(10,2),
    price_change_percent DECIMAL(5,2),
    ai_recommendation TEXT CHECK (ai_recommendation IN ('BUY', 'SELL', 'HOLD')),
    confidence_score DECIMAL(3,2),
    predicted_price DECIMAL(10,2),
    risk_level TEXT,
    volatility DECIMAL(5,4),
    chart_data JSONB,
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índices para optimización
CREATE INDEX idx_gold_execution ON gold_trading_recommendations(execution_id);
CREATE INDEX idx_gold_recommendation ON gold_trading_recommendations(ai_recommendation);
CREATE INDEX idx_gold_processed ON gold_trading_recommendations(processed_at);
CREATE INDEX idx_gold_confidence ON gold_trading_recommendations(confidence_score);

-- Vista para analytics
CREATE VIEW gold_trading_analytics AS
SELECT 
    DATE(processed_at) as analysis_date,
    COUNT(*) as total_analyses,
    AVG(confidence_score) as avg_confidence,
    COUNT(*) FILTER (WHERE ai_recommendation = 'BUY') as buy_recommendations,
    COUNT(*) FILTER (WHERE ai_recommendation = 'SELL') as sell_recommendations,
    COUNT(*) FILTER (WHERE ai_recommendation = 'HOLD') as hold_recommendations,
    AVG(current_price) as avg_gold_price
FROM gold_trading_recommendations
GROUP BY DATE(processed_at);
