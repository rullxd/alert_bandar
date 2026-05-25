CREATE EXTENSION IF NOT EXISTS timescaledb;

DROP VIEW IF EXISTS broker_activity_with_price;
DROP VIEW IF EXISTS broker_activity_net;

CREATE TABLE IF NOT EXISTS broker_activity (
    trade_date DATE NOT NULL,
    broker_code TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    broker_type TEXT,
    value NUMERIC(20, 2) NOT NULL DEFAULT 0,
    lot NUMERIC(20, 4) NOT NULL DEFAULT 0,
    avg_price NUMERIC(20, 8),
    freq INTEGER NOT NULL DEFAULT 0,
    icon_url TEXT,
    PRIMARY KEY (trade_date, broker_code, stock_code, side)
);

ALTER TABLE broker_activity DROP COLUMN IF EXISTS downloaded_at;
ALTER TABLE broker_activity DROP COLUMN IF EXISTS source_file;
ALTER TABLE broker_activity ADD COLUMN IF NOT EXISTS icon_url TEXT;

SELECT create_hypertable('broker_activity', 'trade_date', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_broker_activity_broker_date ON broker_activity (broker_code, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_broker_activity_stock_date ON broker_activity (stock_code, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_broker_activity_value ON broker_activity (trade_date DESC, value DESC);

CREATE TABLE IF NOT EXISTS stock_summary (
    trade_date DATE NOT NULL,
    stock_code TEXT NOT NULL,
    stock_name TEXT,
    previous NUMERIC(20, 4),
    open_price NUMERIC(20, 4),
    high NUMERIC(20, 4),
    low NUMERIC(20, 4),
    close_price NUMERIC(20, 4),
    change_value NUMERIC(20, 4),
    volume NUMERIC(20, 2),
    value NUMERIC(20, 2),
    frequency NUMERIC(20, 2),
    foreign_buy NUMERIC(20, 2),
    foreign_sell NUMERIC(20, 2),
    icon_url TEXT,
    PRIMARY KEY (trade_date, stock_code)
);

ALTER TABLE stock_summary DROP COLUMN IF EXISTS source_file;
ALTER TABLE stock_summary ADD COLUMN IF NOT EXISTS icon_url TEXT;

SELECT create_hypertable('stock_summary', 'trade_date', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_stock_summary_stock_date ON stock_summary (stock_code, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_stock_summary_value ON stock_summary (trade_date DESC, value DESC);

CREATE OR REPLACE VIEW broker_activity_net AS
SELECT
    trade_date,
    broker_code,
    stock_code,
    SUM(CASE WHEN side = 'BUY' THEN value ELSE 0 END) AS buy_value,
    SUM(CASE WHEN side = 'SELL' THEN value ELSE 0 END) AS sell_value,
    SUM(CASE WHEN side = 'BUY' THEN lot ELSE 0 END) AS buy_lot,
    SUM(CASE WHEN side = 'SELL' THEN lot ELSE 0 END) AS sell_lot,
    SUM(CASE WHEN side = 'BUY' THEN value ELSE -value END) AS net_value,
    SUM(CASE WHEN side = 'BUY' THEN lot ELSE -lot END) AS net_lot,
    SUM(freq) AS total_freq
FROM broker_activity
GROUP BY trade_date, broker_code, stock_code;

CREATE OR REPLACE VIEW broker_activity_with_price AS
SELECT
    b.*,
    s.close_price,
    s.change_value,
    s.volume AS market_volume,
    s.value AS market_value,
    CASE WHEN s.value > 0 THEN ROUND((b.value / s.value) * 100, 4) END AS market_value_pct
FROM broker_activity b
LEFT JOIN stock_summary s
  ON s.trade_date = b.trade_date AND s.stock_code = b.stock_code;