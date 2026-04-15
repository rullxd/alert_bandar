"""
Backtest Day Trading Method (Synced with screen_daytrading.py)

Tujuan:
- Menjalankan backtest untuk metode screening + signal terbaru dari screen_daytrading.py
- Menggunakan pipeline yang sama persis agar hasil evaluasi representatif

Usage:
    python backtest_daytrading.py
    python backtest_daytrading.py --days 60
"""

import argparse
import glob
import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from screen_daytrading import (  # type: ignore
    DATA_DIR,
    OUTPUT_DIR,
    LOOKBACK_DAYS,
    load_json,
    prepare,
    calculate_multi_day_metrics,
    screen,
    calculate_entry_signals,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def backtest_signal(
    entry_price: float,
    stop_loss: float,
    target1: float,
    target2: float,
    next_day_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Uji 1 sinyal terhadap data hari berikutnya (OHLC)."""
    if not next_day_data or float(next_day_data.get("High", 0) or 0) == 0:
        return {
            "outcome": "NO_DATA",
            "return_pct": 0.0,
            "exit_price": entry_price,
            "hit_type": "NO_NEXT_DAY",
        }

    next_high = float(next_day_data.get("High", 0) or 0)
    next_low = float(next_day_data.get("Low", 0) or 0)
    next_close = float(next_day_data.get("Close", 0) or 0)

    # Asumsi konservatif: jika low sentuh stop, stop dianggap kena duluan.
    if stop_loss > 0 and next_low <= stop_loss:
        return_pct = ((stop_loss - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
        return {
            "outcome": "LOSS",
            "return_pct": return_pct,
            "exit_price": stop_loss,
            "hit_type": "STOP_LOSS",
        }

    if target2 > 0 and next_high >= target2:
        return_pct = ((target2 - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
        return {
            "outcome": "WIN",
            "return_pct": return_pct,
            "exit_price": target2,
            "hit_type": "TARGET2",
        }

    if target1 > 0 and next_high >= target1:
        return_pct = ((target1 - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
        return {
            "outcome": "WIN",
            "return_pct": return_pct,
            "exit_price": target1,
            "hit_type": "TARGET1",
        }

    return_pct = ((next_close - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
    outcome = "WIN" if next_close >= entry_price else "LOSS"
    return {
        "outcome": outcome,
        "return_pct": return_pct,
        "exit_price": next_close,
        "hit_type": "CLOSE",
    }


def run_backtest(start_days_ago: int = 30, end_days_ago: int = 1) -> pd.DataFrame:
    """
    Jalankan backtest berbasis metode terbaru screen_daytrading.py.

    start_days_ago=30 artinya test 30 hari terakhir (kecuali 1 hari terakhir sebagai next-day pembanding).
    """
    print("=" * 80)
    print("DAY TRADING BACKTEST (SYNCED METHOD)")
    print("=" * 80)

    files_data = sorted(glob.glob(os.path.join(DATA_DIR, "idx_stock_*.json")))
    files_backup = sorted(glob.glob(os.path.join("backup", "idx_stock_*.json")))
    all_files = sorted(files_data + files_backup)

    min_needed = start_days_ago + LOOKBACK_DAYS + 1
    if len(all_files) < min_needed:
        print(f"[ERROR] Data kurang. Ditemukan {len(all_files)}, butuh minimal {min_needed} file.")
        return pd.DataFrame()

    test_files = all_files[-(start_days_ago + 1):-end_days_ago]
    if not test_files:
        print("[ERROR] Tidak ada file periode test.")
        return pd.DataFrame()

    print(f"Backtest files: {len(test_files)}")
    print(f"From: {os.path.basename(test_files[0])}")
    print(f"To  : {os.path.basename(test_files[-1])}")
    print("-" * 80)

    all_trades: List[Dict[str, Any]] = []

    for i, file_path in enumerate(test_files, 1):
        print(f"[{i}/{len(test_files)}] {os.path.basename(file_path)}", end=" -> ")
        try:
            df_today = prepare(load_json(file_path))

            actual_index = all_files.index(file_path)
            hist_start = max(0, actual_index - LOOKBACK_DAYS)
            historical_files = all_files[hist_start:actual_index] if actual_index > 0 else []

            df_historical = pd.DataFrame()
            if len(historical_files) >= 3:
                df_hist_list = [prepare(load_json(f)) for f in historical_files]
                df_historical = pd.concat(df_hist_list, ignore_index=True)
                df_today = calculate_multi_day_metrics(df_today, df_historical)

            df_screened = screen(df_today)
            if len(df_screened) == 0:
                print("no screened stocks")
                continue

            df_hist_arg = df_historical if len(df_historical) > 0 else None
            df_signals = calculate_entry_signals(df_screened, df_historical=df_hist_arg)
            buy_signals = df_signals[df_signals["EntrySignal"].str.contains("BUY", na=False)]
            if len(buy_signals) == 0:
                print("no buy signals")
                continue

            next_file = all_files[actual_index + 1] if actual_index + 1 < len(all_files) else None
            df_next = prepare(load_json(next_file)) if next_file else pd.DataFrame()

            for _, signal in buy_signals.iterrows():
                stock_code = str(signal["StockCode"])
                next_day_data = df_next[df_next["StockCode"] == stock_code]
                next_day_dict = next_day_data.iloc[0].to_dict() if len(next_day_data) > 0 else {}

                entry_price = float(signal.get("Close") or 0)
                stop_loss = float(signal.get("StopLoss") or 0)
                target1 = float(signal.get("Target1") or 0)
                target2 = float(signal.get("Target2") or 0)

                bt = backtest_signal(entry_price, stop_loss, target1, target2, next_day_dict)

                trade = {
                    "Date": df_today["FileDate"].iloc[0],
                    "StockCode": stock_code,
                    "StockName": str(signal.get("StockName", "")),
                    "EntrySignal": str(signal.get("EntrySignal", "")),
                    "SignalGrade": str(signal.get("SignalGrade", "")),
                    "SignalStrength": int(signal.get("SignalStrength") or 0),
                    "SignalReason": str(signal.get("SignalReason", "")),
                    "EntryPrice": entry_price,
                    "StopLoss": stop_loss,
                    "Target1": target1,
                    "Target2": target2,
                    "RiskReward": float(signal.get("RiskReward") or 0),
                    "Outcome": bt["outcome"],
                    "ExitPrice": bt["exit_price"],
                    "ReturnPct": bt["return_pct"],
                    "HitType": bt["hit_type"],
                    "VolumeSurge": float(signal.get("VolumeSurge") or 0),
                    "MarketPhase": str(signal.get("MarketPhase", "Unknown")),
                }
                all_trades.append(trade)

            print(f"{len(buy_signals)} buy signals")

        except Exception as e:
            print(f"error: {str(e)[:80]}")

    if not all_trades:
        print("\n[ERROR] Tidak ada trade yang dihasilkan pada periode backtest.")
        return pd.DataFrame()

    return pd.DataFrame(all_trades)


def analyze_results(df: pd.DataFrame) -> None:
    """Tampilkan ringkasan performa backtest dan simpan CSV hasil."""
    if len(df) == 0:
        print("\n[ERROR] No results to analyze")
        return

    valid = df[df["Outcome"].isin(["WIN", "LOSS"])].copy()
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    print(f"Total Trades: {len(df)}")
    print(f"Valid Trades: {len(valid)}")
    if len(valid) == 0:
        print("No valid trades (all NO_DATA).")
        return

    wins = valid[valid["Outcome"] == "WIN"]
    losses = valid[valid["Outcome"] == "LOSS"]

    win_rate = (len(wins) / len(valid) * 100) if len(valid) > 0 else 0.0
    avg_win = float(wins["ReturnPct"].mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses["ReturnPct"].mean()) if len(losses) > 0 else 0.0
    avg_return = float(valid["ReturnPct"].mean())
    cumulative = float(valid["ReturnPct"].sum())

    profit_factor = abs(wins["ReturnPct"].sum() / losses["ReturnPct"].sum()) if len(losses) > 0 and losses["ReturnPct"].sum() != 0 else 0.0
    expectancy = (win_rate / 100.0 * avg_win) + ((1 - win_rate / 100.0) * avg_loss)

    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg Win: {avg_win:+.2f}%")
    print(f"Avg Loss: {avg_loss:+.2f}%")
    print(f"Avg Return: {avg_return:+.2f}%")
    print(f"Cumulative Return: {cumulative:+.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Expectancy: {expectancy:+.2f}% per trade")

    print("\nBy EntrySignal:")
    for signal_name in sorted(valid["EntrySignal"].dropna().unique()):
        subset = valid[valid["EntrySignal"] == signal_name]
        if len(subset) == 0:
            continue
        wr = len(subset[subset["Outcome"] == "WIN"]) / len(subset) * 100
        avg_ret = float(subset["ReturnPct"].mean())
        print(f"  {signal_name:18s} | n={len(subset):3d} | WR={wr:5.1f}% | Avg={avg_ret:+6.2f}%")

    print("\nBy SignalGrade:")
    grade_order = ["A", "B", "C", ""]
    for g in grade_order:
        subset = valid[valid["SignalGrade"] == g]
        if len(subset) == 0:
            continue
        wr = len(subset[subset["Outcome"] == "WIN"]) / len(subset) * 100
        avg_ret = float(subset["ReturnPct"].mean())
        label = g if g else "(empty)"
        print(f"  Grade {label:7s} | n={len(subset):3d} | WR={wr:5.1f}% | Avg={avg_ret:+6.2f}%")

    print("\nBy Exit Type:")
    for hit_type in ["TARGET2", "TARGET1", "CLOSE", "STOP_LOSS"]:
        subset = valid[valid["HitType"] == hit_type]
        if len(subset) == 0:
            continue
        pct = len(subset) / len(valid) * 100
        avg_ret = float(subset["ReturnPct"].mean())
        print(f"  {hit_type:10s} | n={len(subset):3d} ({pct:5.1f}%) | Avg={avg_ret:+6.2f}%")

    out_file = os.path.join(OUTPUT_DIR, f"backtest_method_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(out_file, index=False)
    print(f"\nSaved: {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest for latest day trading method")
    parser.add_argument("--days", type=int, default=30, help="Jumlah hari backtest (default: 30)")
    args = parser.parse_args()

    df_results = run_backtest(start_days_ago=args.days)
    analyze_results(df_results)


if __name__ == "__main__":
    main()
