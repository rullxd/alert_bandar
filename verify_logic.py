"""
Quick verification: Compare signals between main_screener and backtest logic
"""
import os
import glob
import json
import pandas as pd
from datetime import datetime

# Import from backtest
import sys
sys.path.insert(0, '.')

DATA_DIR = "data"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw["data"])
    name = os.path.basename(path)
    digits = "".join([c for c in name if c.isdigit()])
    df["FileDate"] = datetime.strptime(digits[-8:], "%Y%m%d").date()
    return df

# Load latest file
files = sorted(glob.glob(os.path.join(DATA_DIR, "idx_stock_*.json")))
latest = files[-1]

print("Loading:", os.path.basename(latest))
print("\n" + "="*80)

# Test 1: Import and run main_screener logic
print("TEST 1: Running main_screener logic...")
from main_screener import prepare, screen, calculate_multi_day_metrics, calculate_entry_signals, LOOKBACK_DAYS

df_today = load_json(latest)
df_today = prepare(df_today)

# Load historical
historical_files = files[-(LOOKBACK_DAYS+1):-1]
df_hist_list = [prepare(load_json(f)) for f in historical_files]
df_historical = pd.concat(df_hist_list, ignore_index=True)
df_today = calculate_multi_day_metrics(df_today, df_historical)

df_screened = screen(df_today)
df_signals = calculate_entry_signals(df_screened)
buy_signals_main = df_signals[df_signals["EntrySignal"].str.contains("BUY", na=False)]

print(f"✓ Main screener: {len(buy_signals_main)} BUY signals")

# Test 2: Import and run backtest logic
print("\nTEST 2: Running backtest logic...")
from backtest_daytrading import prepare as bt_prepare, screen as bt_screen
from backtest_daytrading import calculate_multi_day_metrics as bt_multi
from backtest_daytrading import calculate_entry_signals as bt_signals

df_today2 = load_json(latest)
df_today2 = bt_prepare(df_today2)

df_hist_list2 = [bt_prepare(load_json(f)) for f in historical_files]
df_historical2 = pd.concat(df_hist_list2, ignore_index=True)
df_today2 = bt_multi(df_today2, df_historical2)

df_screened2 = bt_screen(df_today2)
df_signals2 = bt_signals(df_screened2)
buy_signals_bt = df_signals2[df_signals2["EntrySignal"].str.contains("BUY", na=False)]

print(f"✓ Backtest: {len(buy_signals_bt)} BUY signals")

# Compare
print("\n" + "="*80)
print("COMPARISON:")
print("="*80)

if len(buy_signals_main) == len(buy_signals_bt):
    print("✅ Signal COUNT: MATCH")
else:
    print(f"❌ Signal COUNT: MISMATCH (main={len(buy_signals_main)}, backtest={len(buy_signals_bt)})")

# Compare stock codes
stocks_main = set(buy_signals_main["StockCode"].values)
stocks_bt = set(buy_signals_bt["StockCode"].values)

if stocks_main == stocks_bt:
    print("✅ Stock CODES: MATCH")
else:
    print(f"❌ Stock CODES: MISMATCH")
    only_main = stocks_main - stocks_bt
    only_bt = stocks_bt - stocks_main
    if only_main:
        print(f"   Only in main: {only_main}")
    if only_bt:
        print(f"   Only in backtest: {only_bt}")

# Compare signal details for matching stocks
print("\nDETAILS COMPARISON:")
print("-"*80)
for stock in sorted(stocks_main & stocks_bt):
    signal_main = buy_signals_main[buy_signals_main["StockCode"] == stock].iloc[0]
    signal_bt = buy_signals_bt[buy_signals_bt["StockCode"] == stock].iloc[0]
    
    print(f"\n{stock}:")
    print(f"  Signal:   Main='{signal_main['EntrySignal']}' | BT='{signal_bt['EntrySignal']}'")
    print(f"  Strength: Main={signal_main['SignalStrength']} | BT={signal_bt['SignalStrength']}")
    print(f"  StopLoss: Main={signal_main['StopLoss']:.0f} | BT={signal_bt['StopLoss']:.0f}")
    print(f"  Target1:  Main={signal_main['Target1']:.0f} | BT={signal_bt['Target1']:.0f}")
    print(f"  Target2:  Main={signal_main['Target2']:.0f} | BT={signal_bt['Target2']:.0f}")
    print(f"  R/R:      Main={signal_main['RiskReward']:.2f} | BT={signal_bt['RiskReward']:.2f}")
    
    # Check if all match
    matches = (
        signal_main['EntrySignal'] == signal_bt['EntrySignal'] and
        signal_main['SignalStrength'] == signal_bt['SignalStrength'] and
        abs(signal_main['StopLoss'] - signal_bt['StopLoss']) < 1 and
        abs(signal_main['Target1'] - signal_bt['Target1']) < 1 and
        abs(signal_main['Target2'] - signal_bt['Target2']) < 1 and
        abs(signal_main['RiskReward'] - signal_bt['RiskReward']) < 0.01
    )
    
    if matches:
        print("  ✅ EXACT MATCH")
    else:
        print("  ❌ MISMATCH")

print("\n" + "="*80)
if len(buy_signals_main) == len(buy_signals_bt) and stocks_main == stocks_bt:
    print("✅ VERIFICATION PASSED: Logika main_screener dan backtest SAMA!")
else:
    print("❌ VERIFICATION FAILED: Ada perbedaan!")
print("="*80)
