from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

from curl_cffi.requests import Session


URL_API = "https://exodus.stockbit.com/order-trade/broker/activity"
URL_REFERER_BASE = "https://exodus.stockbit.com/order-trade/broker/activity"

DEFAULT_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "origin": "https://exodus.stockbit.com",
    "referer": URL_REFERER_BASE,
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def render_progress(current: int, total: int, label: str) -> None:
    width = 24
    if total <= 0:
        bar = "[........................]"
        percent = 0.0
    else:
        filled = min(width, int(width * current / total))
        bar = f"[{('=' * filled).ljust(width, '.')}]"
        percent = (current / total) * 100

    sys.stdout.write(f"\r{bar} {current}/{total} ({percent:5.1f}%) {label}   ")
    sys.stdout.flush()


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download broker activity from Stockbit Exodus endpoint."
    )
    parser.add_argument("--broker-code", default=None, help="Kode broker, contoh: AK")
    parser.add_argument(
        "--transaction-type",
        default="TRANSACTION_TYPE_GROSS",
        help="Transaction type dari endpoint Stockbit.",
    )
    parser.add_argument(
        "--investor-type",
        default="INVESTOR_TYPE_ALL",
        help="Investor type dari endpoint Stockbit.",
    )
    parser.add_argument("--market-board", default="MARKET_TYPE_REGULER")
    parser.add_argument("--date", default=None, help="Tanggal tunggal, format YYYY-MM-DD")
    parser.add_argument("--from-date", default=None, help="Format tanggal YYYY-MM-DD")
    parser.add_argument("--to-date", default=None, help="Format tanggal YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--pages", type=int, default=1, help="Jumlah halaman yang diambil")
    parser.add_argument(
        "--max-broker-workers",
        type=int,
        default=None,
        help="Jumlah broker yang diproses sekaligus. Jika kosong, akan ditanya saat menjalankan.",
    )
    parser.add_argument(
        "--bearer-token",
        default=os.environ.get("EXODUS_BEARER_TOKEN")
        or os.environ.get("STOCKBIT_BEARER_TOKEN")
        or os.environ.get("STOCKBIT_TOKEN"),
        help="Bearer token. Jika tidak diisi, ambil dari environment.",
    )
    parser.add_argument("--delay-min", type=float, default=1.0)
    parser.add_argument("--delay-max", type=float, default=2.5)
    parser.add_argument("--broker-list-file", default="broker_names.txt", help="File list broker format: CODE - NAME")
    return parser.parse_args()


def prompt_if_missing(args: argparse.Namespace) -> argparse.Namespace:
    if not args.date and not args.from_date and not args.to_date:
        use_auto_range = input("Auto download 1 tahun kebelakang? [y/n]: ").strip().lower()
        if use_auto_range in {"", "y", "yes"}:
            today = datetime.now()
            one_year_ago = today - timedelta(days=365)
            args.from_date = one_year_ago.strftime("%Y-%m-%d")
            args.to_date = today.strftime("%Y-%m-%d")
        else:
            args.date = input("Masukkan tanggal (YYYY-MM-DD): ").strip()

    if args.date and (not args.from_date or not args.to_date):
        args.from_date = args.date
        args.to_date = args.date

    if not args.from_date:
        args.from_date = input("Masukkan from-date (YYYY-MM-DD): ").strip()

    if not args.to_date:
        args.to_date = input("Masukkan to-date (YYYY-MM-DD): ").strip()

    # Interaktif: tanya jumlah worker paralel jika belum diberikan
    if args.max_broker_workers is None:
        choice = input("Berapa broker paralel? pilih 1,2,3 (default 3): ").strip()
        if not choice:
            args.max_broker_workers = 3
        else:
            try:
                v = int(choice)
                if v < 1:
                    v = 1
                args.max_broker_workers = v
            except Exception:
                args.max_broker_workers = 3

    return args


def validate_date(value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("Tanggal harus format YYYY-MM-DD") from exc
    return value


def as_dict(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return cast(list[Any], value) if isinstance(value, list) else []


def build_date_range(start_date: str, end_date: str) -> list[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    if start > end:
        raise ValueError("from-date tidak boleh lebih besar dari to-date")

    date_list: list[str] = []
    current_date = start
    while current_date <= end:
        if current_date.weekday() < 5:
            date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    return date_list


def get_broker_activity_section(data: dict[str, Any]) -> dict[str, Any]:
    payload = as_dict(data.get("data", {}))
    return as_dict(payload.get("broker_activity_transaction", {}))


def build_headers(token: str, params: dict[str, Any]) -> dict[str, str]:
    headers = dict(DEFAULT_HEADERS)
    headers["authorization"] = f"Bearer {token}"
    headers["referer"] = (
        f"{URL_REFERER_BASE}?broker_code={params['broker_code']}"
        f"&transaction_type={params['transaction_type']}"
        f"&investor_type={params['investor_type']}"
        f"&limit={params['limit']}"
        f"&market_board={params['market_board']}"
        f"&page={params['page']}"
        f"&from={params['from']}"
        f"&to={params['to']}"
    )
    return headers


def fetch_page(
    session: Any,
    token: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    max_retries = 5

    for attempt in range(1, max_retries + 1):
        response = session.get(
            URL_API,
            params=params,
            headers=build_headers(token, params),
            timeout=60,
        )

        if response.status_code == 429:
            wait_time = random.uniform(10, 20) * attempt
            print(f"\n⚠️  Rate limit page {params['page']}. Retry {attempt}/{max_retries} setelah {wait_time:.1f}s")
            time.sleep(wait_time)
            continue

        if response.status_code >= 500:
            wait_time = random.uniform(3, 6) * attempt
            print(f"\n⚠️  Server error page {params['page']} ({response.status_code}). Retry {attempt}/{max_retries} setelah {wait_time:.1f}s")
            time.sleep(wait_time)
            continue

        response.raise_for_status()
        data = as_dict(json.loads(response.text))

        message = str(data.get("message", "")).lower()
        if "limit" in message:
            wait_time = random.uniform(10, 20) * attempt
            print(f"\n⚠️  Respons limit page {params['page']}. Retry {attempt}/{max_retries} setelah {wait_time:.1f}s")
            time.sleep(wait_time)
            continue

        return data

    raise RuntimeError(f"Gagal mengambil page {params['page']} setelah {max_retries} percobaan")


def load_broker_codes(file_path: Path) -> list[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"File broker list tidak ditemukan: {file_path}")

    broker_codes: list[str] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        code = line.split("-", 1)[0].strip().upper()
        if code and code not in broker_codes:
            broker_codes.append(code)

    return broker_codes


def process_broker(
    args: argparse.Namespace,
    broker_code: str,
    date_range: list[str],
    output_root: Path,
) -> tuple[int, int, int, int]:
    output_dir = output_root / broker_code
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Mulai broker {broker_code} - output: {output_dir.resolve()}")

    success_count = 0
    skip_count = 0
    error_count = 0
    completed_count = 0
    total_days = len(date_range)

    session = cast(Any, Session(impersonate="chrome124"))

    with session as s:
        for date_value in date_range:
            output_name = f"broker_activity_{date_value}.json"
            output_path = output_dir / output_name

            if output_path.exists():
                skip_count += 1
                completed_count += 1
                render_progress(completed_count, total_days, f"{broker_code} skip {date_value}")
                continue

            render_progress(completed_count, total_days, f"{broker_code} ambil {date_value}")
            params: dict[str, Any] = {
                "broker_code": broker_code,
                "transaction_type": args.transaction_type,
                "investor_type": args.investor_type,
                "limit": args.limit,
                "market_board": args.market_board,
                "page": 1,
                "from": date_value,
                "to": date_value,
            }

            try:
                data = fetch_page(s, args.bearer_token, params)
            except Exception as exc:
                error_count += 1
                completed_count += 1
                render_progress(completed_count, total_days, f"{broker_code} gagal {date_value}")
                print(f"\n⚠️  {broker_code} {date_value} gagal: {exc}")
                continue

            activity_section = get_broker_activity_section(data)
            brokers_buy = as_list(activity_section.get("brokers_buy", []))
            brokers_sell = as_list(activity_section.get("brokers_sell", []))
            buy_count = len(brokers_buy)
            sell_count = len(brokers_sell)

            if buy_count == 0 and sell_count == 0:
                # Respons sukses tapi tidak ada data buy/sell — catat log jelas lalu skip
                log(f"{broker_code} {date_value} berhasil tetapi buy/sell kosong, tidak menyimpan file")
                skip_count += 1
                completed_count += 1
                render_progress(completed_count, total_days, f"{broker_code} libur {date_value}")
                continue

            payload: dict[str, Any] = {
                "downloaded_at": datetime.now().isoformat(timespec="seconds"),
                "endpoint": URL_API,
                "broker_code": broker_code,
                "from": date_value,
                "to": date_value,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "total_count": buy_count + sell_count,
                "response": data,
            }

            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)

            success_count += 1
            completed_count += 1
            render_progress(completed_count, total_days, f"{broker_code} selesai {date_value}")

            if completed_count < total_days:
                pause = random.uniform(args.delay_min, args.delay_max)
                time.sleep(pause)

    print()
    log(
        f"Summary {broker_code} selesai={completed_count}/{total_days} berhasil={success_count} skip={skip_count} error={error_count}"
    )
    return success_count, skip_count, error_count, completed_count


def main() -> int:
    load_dotenv_file(Path(".env"))
    args = prompt_if_missing(parse_args())

    if not args.bearer_token:
        log("Bearer token belum ada. Isi EXODUS_BEARER_TOKEN di .env atau pakai --bearer-token.")
        return 1

    from_date = validate_date(args.from_date)
    to_date = validate_date(args.to_date)
    date_range = build_date_range(from_date, to_date)

    output_root = Path(args.output_dir) / "BROKER_ACTIVITY_DAILY"
    output_root.mkdir(parents=True, exist_ok=True)

    if args.broker_code:
        broker_codes = [args.broker_code.upper()]
    else:
        broker_codes = load_broker_codes(Path(args.broker_list_file))

    log("Mulai download broker activity")
    log(f"total_broker={len(broker_codes)} from={from_date} to={to_date} total_hari={len(date_range)} limit={args.limit}")
    log(f"Output root: {output_root.resolve()}")

    total_broker = len(broker_codes)
    processed_broker = 0
    skipped_broker = 0
    total_success = 0
    total_skip = 0
    total_error = 0

    pending_brokers: list[str] = []
    for broker_code in broker_codes:
        broker_folder = output_root / broker_code
        if broker_folder.exists():
            # jika folder ada, cek isinya; jika ada file/entry, anggap sudah selesai dan skip
            try:
                any_entry = any(broker_folder.iterdir())
            except Exception:
                any_entry = False

            if any_entry:
                skipped_broker += 1
                log(f"Skip broker {broker_code} karena folder sudah ada dan tidak kosong: {broker_folder}")
                continue
            # jika folder kosong, kita tetap proses ulang (anggap belum pernah diisi)
            pending_brokers.append(broker_code)
        else:
            pending_brokers.append(broker_code)

    if pending_brokers:
        max_workers = max(1, min(args.max_broker_workers, len(pending_brokers)))
        log(f"Jalankan paralel {max_workers} broker sekaligus")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(process_broker, args, broker_code, date_range, output_root): broker_code
                for broker_code in pending_brokers
            }

            for future in as_completed(future_map):
                broker_code = future_map[future]
                try:
                    success_count, skip_count, error_count, _ = future.result()
                except Exception as exc:
                    processed_broker += 1
                    total_error += 1
                    log(f"Broker {broker_code} gagal total: {exc}")
                    continue

                processed_broker += 1
                total_success += success_count
                total_skip += skip_count
                total_error += error_count

    log(
        "Summary broker "
        f"processed={processed_broker}/{total_broker} "
        f"skip_folder={skipped_broker} "
        f"berhasil={total_success} skip_hari={total_skip} error={total_error}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())