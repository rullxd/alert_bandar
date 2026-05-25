#!/usr/bin/env python3
"""Import local broker activity and IDX stock summary JSON into TimescaleDB."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, cast


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
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows per database batch commit")
    parser.add_argument("--progress-every", type=int, default=25, help="Log progress every N files")
    parser.add_argument("--log-file", default="migrate_timescaledb.log", help="Write detailed migration log to this file")
    parser.add_argument("--verbose", action="store_true", help="Show per-file logs in console")
    return parser.parse_args()


def setup_logging(log_file: str | None, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("timescaledb_migration")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(file_handler)

    return logger


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def as_dict(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return cast(list[Any], value) if isinstance(value, list) else []


def stockbit_icon_url(stock_code: Any) -> str | None:
    if not stock_code:
        return None
    return f"https://assets.stockbit.com/logos/companies/{str(stock_code).upper()}.png"


def broker_rows(path: Path) -> Iterable[tuple[Any, ...]]:
    payload = read_json(path)
    response = as_dict(payload.get("response"))
    response_data = as_dict(response.get("data"))
    trans = as_dict(response_data.get("broker_activity_transaction"))

    for key, side in (("brokers_buy", "BUY"), ("brokers_sell", "SELL")):
        for raw_item in as_list(trans.get(key)):
            item = as_dict(raw_item)
            company_detail = as_dict(item.get("company_detail"))
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
                company_detail.get("icon_url") or stockbit_icon_url(item.get("stock_code")),
            )


def stock_rows(path: Path) -> Iterable[tuple[Any, ...]]:
    payload = read_json(path)
    match = STOCK_DATE_RE.search(path.name)
    fallback_date = datetime.strptime(match.group(1), "%Y%m%d").date() if match else None

    for raw_item in as_list(payload.get("data")):
        item = as_dict(raw_item)
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
            stockbit_icon_url(item.get("StockCode")),
        )


def import_many(conn: Any, sql: str, rows: Iterable[tuple[Any, ...]], batch_size: int, logger: logging.Logger, label: str) -> int:
    total = 0
    batch: list[tuple[Any, ...]] = []
    started_at = time.time()
    with conn.cursor() as cur:
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                cur.executemany(sql, batch)
                total += len(batch)
                elapsed = max(time.time() - started_at, 0.001)
                logger.debug("%s inserted %d rows (%.0f rows/s)", label, total, total / elapsed)
                batch.clear()
        if batch:
            cur.executemany(sql, batch)
            total += len(batch)
            elapsed = max(time.time() - started_at, 0.001)
            logger.debug("%s inserted %d rows (%.0f rows/s)", label, total, total / elapsed)
    conn.commit()
    return total


def iter_rows_with_progress(
    files: list[Path],
    row_reader: Callable[[Path], Iterable[tuple[Any, ...]]],
    logger: logging.Logger,
    label: str,
    progress_every: int,
) -> Iterable[tuple[Any, ...]]:
    started_at = time.time()
    row_count = 0
    error_count = 0
    total_files = len(files)

    for index, file in enumerate(files, 1):
        file_started_at = time.time()
        file_rows = 0
        try:
            for row in row_reader(file):
                file_rows += 1
                row_count += 1
                yield row
        except Exception as exc:
            error_count += 1
            logger.exception("%s failed reading %s: %s", label, file, exc)
            continue

        elapsed = max(time.time() - started_at, 0.001)
        files_per_second = index / elapsed
        remaining_files = max(total_files - index, 0)
        eta = remaining_files / files_per_second if files_per_second else 0

        if index == 1 or index == total_files or index % max(progress_every, 1) == 0:
            logger.info(
                "%s progress %.0f%% | files %d/%d | rows %d | last %s rows=%d time=%s | speed %.1f files/s | ETA %s | errors %d",
                label,
                (index / total_files) * 100 if total_files else 100,
                index,
                total_files,
                row_count,
                file.name,
                file_rows,
                format_duration(time.time() - file_started_at),
                files_per_second,
                format_duration(eta),
                error_count,
            )

    logger.info(
        "%s read complete | files %d | rows %d | errors %d | elapsed %s",
        label,
        total_files,
        row_count,
        error_count,
        format_duration(time.time() - started_at),
    )


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
    logger = setup_logging(args.log_file, args.verbose)
    logger.info("Starting TimescaleDB migration")
    logger.info("Log file: %s", args.log_file or "disabled")
    logger.info("Batch size: %d rows", args.batch_size)
    broker_sql = """
        INSERT INTO broker_activity
        (trade_date, broker_code, stock_code, side, broker_type, value, lot, avg_price, freq, icon_url)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (trade_date, broker_code, stock_code, side) DO UPDATE SET
          broker_type = EXCLUDED.broker_type, value = EXCLUDED.value, lot = EXCLUDED.lot,
          avg_price = EXCLUDED.avg_price, freq = EXCLUDED.freq, icon_url = EXCLUDED.icon_url
    """
    stock_sql = """
        INSERT INTO stock_summary
        (trade_date, stock_code, stock_name, previous, open_price, high, low, close_price, change_value,
         volume, value, frequency, foreign_buy, foreign_sell, icon_url)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (trade_date, stock_code) DO UPDATE SET
          stock_name = EXCLUDED.stock_name, previous = EXCLUDED.previous, open_price = EXCLUDED.open_price,
          high = EXCLUDED.high, low = EXCLUDED.low, close_price = EXCLUDED.close_price,
          change_value = EXCLUDED.change_value, volume = EXCLUDED.volume, value = EXCLUDED.value,
          frequency = EXCLUDED.frequency, foreign_buy = EXCLUDED.foreign_buy,
          foreign_sell = EXCLUDED.foreign_sell, icon_url = EXCLUDED.icon_url
    """

    with psycopg.connect(args.dsn) as conn:
        if not args.skip_broker:
            files = list(iter_files(Path(args.broker_dir), "broker_activity_*.json", args.limit_files))
            logger.info("Broker import starting | files %d | dir %s", len(files), args.broker_dir)
            rows = iter_rows_with_progress(files, broker_rows, logger, "broker", args.progress_every)
            total = import_many(conn, broker_sql, rows, args.batch_size, logger, "broker")
            logger.info("Broker import done | rows %d | files %d", total, len(files))
        if not args.skip_stock:
            files = list(iter_files(Path(args.stock_dir), "idx_stock_*.json", args.limit_files))
            logger.info("Stock import starting | files %d | dir %s", len(files), args.stock_dir)
            rows = iter_rows_with_progress(files, stock_rows, logger, "stock", args.progress_every)
            total = import_many(conn, stock_sql, rows, args.batch_size, logger, "stock")
            logger.info("Stock import done | rows %d | files %d", total, len(files))
    logger.info("Migration finished successfully")


if __name__ == "__main__":
    main()