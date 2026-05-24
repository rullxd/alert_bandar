# TimescaleDB Market Data

Setup ini membuat data broker activity dan stock summary lebih gampang di-query untuk analisa bandar, net buy/sell, akumulasi broker, dan perbandingan ke market value harian.

## 1. Start database

```bash
docker compose up -d
```

Database default:

```text
postgresql://postgres:password@localhost:5433/market_data
```

Web viewer Adminer:

```text
http://localhost:8081
```

Login Adminer:

```text
System: PostgreSQL
Server: timescaledb
Username: postgres
Password: password
Database: market_data
```

## 2. Install dependency migrasi

```bash
pip install "psycopg[binary]"
```

## 3. Import data JSON

Test kecil dulu:

```bash
python migrate_to_timescaledb.py --limit-files 2
```

Import semua data:

```bash
python migrate_to_timescaledb.py
```

Kalau hanya broker activity:

```bash
python migrate_to_timescaledb.py --skip-stock
```

Kalau hanya stock summary:

```bash
python migrate_to_timescaledb.py --skip-broker
```

## Contoh query canggih

Top net buy broker AK:

```sql
SELECT trade_date, stock_code, net_value, net_lot
FROM broker_activity_net
WHERE broker_code = 'AK'
ORDER BY net_value DESC
LIMIT 20;
```

Saham yang nilai transaksi broker-nya dominan terhadap market value:

```sql
SELECT trade_date, broker_code, stock_code, side, value, market_value_pct
FROM broker_activity_with_price
WHERE market_value_pct IS NOT NULL
ORDER BY market_value_pct DESC
LIMIT 20;
```

Akumulasi net buy 20 hari per broker dan saham:

```sql
SELECT broker_code, stock_code, SUM(net_value) AS net_20d
FROM broker_activity_net
WHERE trade_date >= CURRENT_DATE - INTERVAL '20 days'
GROUP BY broker_code, stock_code
ORDER BY net_20d DESC
LIMIT 50;
```
