#!/usr/bin/env python3
"""Import local broker activity and IDX stock summary JSON into TimescaleDB."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


BROKER_DATE_RE = re.compile(r"broker_activity_(\d{4}-\d{2}-\d{2})\.json$")
STOCK_DATE_RE = re.compile(r"idx_stock_(\d{8})\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate JSON market data to TimescaleDB")
    parser.add_argument("--dsn", default=os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5433/market_data"))
    parser.add_argument("--broker-dir", default="BROKER_ACTIVITY_DAILY")
    parser.add_argument("--stock-dir", default="data")
    parser.add_argument("--skip-broker", action="store_true")
    parser.add_argument("--skip-stock", action="store_true")
    parser.add_argument("--limit-files", type=int, default=0, help="Useful for quick test imports")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def broker_rows(path: Path) -> Iterable[tuple[Any, ...]]:
    payload = read_json(path)
    trans = payload.get("response", {}).get("data", {}).get("broker_activity_transaction", {})
    downloaded_at = payload.get("downloaded_at")

    for key, side in (("brokers_buy", "BUY"), ("brokers_sell", "SELL")):
        for item in trans.get(key, []) or []:
            yield (
                item.get("date") or payload.get("from"),
                item.get("broker_code") or payload.get("broker_code"),
                item.get("stock_code"),
                side,
                item.get("type"),
                item.get("value") or 0,
                item.get("lot") or 0,
                item.get("avg_price"),
                int(item.get("freq") or 0),
                downloaded_at,
                str(path),
            )


def stock_rows(path: Path) -> Iterable[tuple[Any, ...]]:
    payload = read_json(path)
    match = STOCK_DATE_RE.search(path.name)
    fallback_date = datetime.strptime(match.group(1), "%Y%m%d").date() if match else None

    for item in payload.get("data", []) or []:
        trade_date = (item.get("Date") or "")[:10] or fallback_date
        yield (
            trade_date,
            item.get("StockCode"),
            item.get("StockName"),
            item.get("Previous"),
            item.get("OpenPrice"),
            item.get("High"),
            item.get("Low"),
            item.get("Close"),
            item.get("Change"),
            item.get("Volume"),
            item.get("Value"),
            item.get("Frequency"),
            item.get("ForeignBuy"),
            item.get("ForeignSell"),
            str(path),
        )


def import_many(conn: Any, sql: str, rows: Iterable[tuple[Any, ...]], batch_size: int = 1000) -> int:
    total = 0
    batch: list[tuple[Any, ...]] = []
    with conn.cursor() as cur:
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                cur.executemany(sql, batch)
                total += len(batch)
                batch.clear()
        if batch:
            cur.executemany(sql, batch)
            total += len(batch)
    conn.commit()
    return total


def iter_files(root: Path, pattern: str, limit: int = 0) -> Iterable[Path]:
    count = 0
    for path in sorted(root.rglob(pattern)):
        if limit and count >= limit:
            break
        count += 1
        yield path


def main() -> None:
    try:
        import psycopg
    except ImportError:
        raise SystemExit("Install dependency dulu: pip install psycopg[binary]")

    args = parse_args()
    broker_sql = """
        INSERT INTO broker_activity
        (trade_date, broker_code, stock_code, side, broker_type, value, lot, avg_price, freq, downloaded_at, source_file)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (trade_date, broker_code, stock_code, side) DO UPDATE SET
          broker_type = EXCLUDED.broker_type, value = EXCLUDED.value, lot = EXCLUDED.lot,
          avg_price = EXCLUDED.avg_price, freq = EXCLUDED.freq, downloaded_at = EXCLUDED.downloaded_at,
          source_file = EXCLUDED.source_file
    """
    stock_sql = """
        INSERT INTO stock_summary
        (trade_date, stock_code, stock_name, previous, open_price, high, low, close_price, change_value,
         volume, value, frequency, foreign_buy, foreign_sell, source_file)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (trade_date, stock_code) DO UPDATE SET
          stock_name = EXCLUDED.stock_name, previous = EXCLUDED.previous, open_price = EXCLUDED.open_price,
          high = EXCLUDED.high, low = EXCLUDED.low, close_price = EXCLUDED.close_price,
          change_value = EXCLUDED.change_value, volume = EXCLUDED.volume, value = EXCLUDED.value,
          frequency = EXCLUDED.frequency, foreign_buy = EXCLUDED.foreign_buy,
          foreign_sell = EXCLUDED.foreign_sell, source_file = EXCLUDED.source_file
    """

    with psycopg.connect(args.dsn) as conn:
        if not args.skip_broker:
            files = list(iter_files(Path(args.broker_dir), "broker_activity_*.json", args.limit_files))
            total = import_many(conn, broker_sql, (row for file in files for row in broker_rows(file)))
            print(f"Imported broker rows: {total:,} from {len(files):,} files")
        if not args.skip_stock:
            files = list(iter_files(Path(args.stock_dir), "idx_stock_*.json", args.limit_files))
            total = import_many(conn, stock_sql, (row for file in files for row in stock_rows(file)))
            print(f"Imported stock rows: {total:,} from {len(files):,} files")


if __name__ == "__main__":
    main()