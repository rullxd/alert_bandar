from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--pages", type=int, default=1, help="Jumlah halaman yang diambil")
    parser.add_argument(
        "--bearer-token",
        default=os.environ.get("EXODUS_BEARER_TOKEN")
        or os.environ.get("STOCKBIT_BEARER_TOKEN")
        or os.environ.get("STOCKBIT_TOKEN"),
        help="Bearer token. Jika tidak diisi, ambil dari environment.",
    )
    parser.add_argument("--delay-min", type=float, default=1.0)
    parser.add_argument("--delay-max", type=float, default=2.5)
    return parser.parse_args()


def prompt_if_missing(args: argparse.Namespace) -> argparse.Namespace:
    if not args.broker_code:
        args.broker_code = input("Masukkan code broker: ").strip().upper()

    if not args.date and not args.from_date and not args.to_date:
        args.date = input("Masukkan tanggal (YYYY-MM-DD): ").strip()

    if args.date and (not args.from_date or not args.to_date):
        args.from_date = args.date
        args.to_date = args.date

    if not args.from_date:
        args.from_date = input("Masukkan from-date (YYYY-MM-DD): ").strip()

    if not args.to_date:
        args.to_date = input("Masukkan to-date (YYYY-MM-DD): ").strip()

    return args


def validate_date(value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("Tanggal harus format YYYY-MM-DD") from exc
    return value


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
    session: Session,
    token: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    response = session.get(
        URL_API,
        params=params,
        headers=build_headers(token, params),
        timeout=60,
    )
    response.raise_for_status()
    return json.loads(response.text)


def main() -> int:
    load_dotenv_file(Path(".env"))
    args = prompt_if_missing(parse_args())

    if not args.bearer_token:
        print("❌ Bearer token belum ada. Isi EXODUS_BEARER_TOKEN di .env atau pakai --bearer-token.")
        return 1

    from_date = validate_date(args.from_date)
    to_date = validate_date(args.to_date)

    output_dir = Path(args.output_dir) / args.broker_code
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"broker_activity_{args.broker_code}_{from_date}_to_{to_date}.json"
    output_path = output_dir / output_name

    session = Session(impersonate="chrome124")  # type: ignore[misc]

    results: list[dict[str, Any]] = []
    total_records = None

    with session as s:  # type: ignore[misc]
        for page in range(1, args.pages + 1):
            params = {
                "broker_code": args.broker_code,
                "transaction_type": args.transaction_type,
                "investor_type": args.investor_type,
                "limit": args.limit,
                "market_board": args.market_board,
                "page": page,
                "from": from_date,
                "to": to_date,
            }

            data = fetch_page(s, args.bearer_token, params)
            if total_records is None:
                total_records = data.get("recordsTotal")

            results.append(
                {
                    "page": page,
                    "params": params,
                    "response": data,
                }
            )

            current_records = data.get("recordsTotal", 0)
            print(f"[{page}/{args.pages}] OK - recordsTotal={current_records}")

            if page < args.pages:
                time.sleep(random.uniform(args.delay_min, args.delay_max))

    payload = {
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        "endpoint": URL_API,
        "broker_code": args.broker_code,
        "from": from_date,
        "to": to_date,
        "records_total": total_records,
        "pages": results,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"✅ Tersimpan: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())