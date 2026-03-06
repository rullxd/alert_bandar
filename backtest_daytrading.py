"""
Backtest Day Trading Strategy
Test entry signals against real market data
"""

import os
import glob
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import settings from main_screener
DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Screening parameters (same as main_screener)
MIN_VALUE = 2_000_000_000
MIN_VOLUME = 500_000
MIN_FREQUENCY = 200
MAX_SPREAD_PCT = 2.0
MIN_PRICE = 100
MAX_PRICE = 50_000
MIN_VOLATILITY = 0.8
MAX_VOLATILITY = 20.0
MIN_VOLUME_VS_SHARES = 0.000005
AVG_TRADE_SIZE_MIN = 3_000_000
MIN_VALUE_RANK_PERCENTILE = 0.5
MIN_NET_FOREIGN = -1_000_000
MIN_NONREG_RATIO = 0.01
LOOKBACK_DAYS = 7
MIN_LIQUID_DAYS = 5


def load_json(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw["data"])
    name = os.path.basename(path)
    digits = "".join([c for c in name if c.isdigit()])
    df["FileDate"] = datetime.strptime(digits[-8:], "%Y%m%d").date()
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "Previous","Close","Volume","Value","Frequency","High","Low",
        "ForeignBuy","ForeignSell","NonRegularVolume",
        "TradebleShares","Bid","Offer"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    
    df["NetForeign"] = df["ForeignBuy"] - df["ForeignSell"]
    df["NonRegRatio"] = (df["NonRegularVolume"] / df["Volume"].replace(0, float('nan'))).fillna(0)
    df["ChangePct"] = ((df["Close"] - df["Previous"]) / df["Previous"] * 100).replace([float("inf"), -float("inf")], 0)
    df["Volatility"] = ((df["High"] - df["Low"]) / df["Previous"] * 100).replace([float("inf"), -float("inf")], 0)
    df["SpreadPct"] = df.apply(
        lambda r: abs(r["Offer"] - r["Bid"]) / r["Close"] * 100 if r["Close"] > 0 else 0,
        axis=1
    )
    df["AvgTradeSize"] = df.apply(
        lambda r: r["Value"] / r["Frequency"] if r["Frequency"] > 0 else 0,
        axis=1
    )
    df["VolumeVsShares"] = df.apply(
        lambda r: r["Volume"] / r["TradebleShares"] if r["TradebleShares"] > 0 else 0,
        axis=1
    )
    df["ValueRank"] = df["Value"].rank(pct=True)
    df["CandleStrength"] = df.apply(
        lambda r: (r["Close"] - r["Low"]) / (r["High"] - r["Low"]) if r["High"] > r["Low"] else 0.5,
        axis=1
    )
    
    return df


def screen(df: pd.DataFrame) -> pd.DataFrame:
    """Filter stocks for day trading (same as main_screener)"""
    
    has_multiday = "ConsistencyScore" in df.columns
    
    base_filter = (
        (df["Value"] >= MIN_VALUE) &
        (df["Volume"] >= MIN_VOLUME) &
        (df["Frequency"] >= MIN_FREQUENCY) &
        (df["SpreadPct"] <= MAX_SPREAD_PCT) &
        (df["Close"] >= MIN_PRICE) &
        (df["Close"] <= MAX_PRICE) &
        (df["Volatility"] >= MIN_VOLATILITY) &
        (df["Volatility"] <= MAX_VOLATILITY) &
        (df["VolumeVsShares"] >= MIN_VOLUME_VS_SHARES) &
        (df["AvgTradeSize"] >= AVG_TRADE_SIZE_MIN) &
        (df["ValueRank"] >= MIN_VALUE_RANK_PERCENTILE) &
        (df["NetForeign"] >= MIN_NET_FOREIGN) &
        (df["NonRegRatio"] >= MIN_NONREG_RATIO)
    )
    
    if has_multiday:
        multiday_filter = (
            (df["LiquidDays30d"] >= MIN_LIQUID_DAYS) &
            (df["ConsistencyScore"] >= 40)
        )
        filtered = df[base_filter & multiday_filter].copy()
    else:
        filtered = df[base_filter].copy()
    
    return filtered


def calculate_multi_day_metrics(df_today: pd.DataFrame, df_historical: pd.DataFrame) -> pd.DataFrame:
    """Calculate multi-day metrics"""
    result = df_today.copy()
    
    df_hist = df_historical.copy()
    df_hist["PrevClose"] = df_hist.groupby("StockCode")["Close"].shift(1)
    df_hist["TR1"] = df_hist["High"] - df_hist["Low"]
    df_hist["TR2"] = (df_hist["High"] - df_hist["PrevClose"]).abs()
    df_hist["TR3"] = (df_hist["Low"] - df_hist["PrevClose"]).abs()
    df_hist["TrueRange"] = df_hist[["TR1", "TR2", "TR3"]].max(axis=1)
    
    hist_grouped = df_hist.groupby("StockCode")
    
    avg_volume = hist_grouped["Volume"].mean()
    avg_value = hist_grouped["Value"].mean()
    
    liquid_days = hist_grouped.apply(
        lambda g: ((g["Value"] >= MIN_VALUE * 0.5) & (g["Volume"] >= MIN_VOLUME * 0.5)).sum()
    )
    
    result["AvgVolume30d"] = result["StockCode"].map(avg_volume).fillna(0)
    result["AvgValue30d"] = result["StockCode"].map(avg_value).fillna(0)
    result["LiquidDays30d"] = result["StockCode"].map(liquid_days).fillna(0)
    result["ConsistencyScore"] = (result["LiquidDays30d"] / LOOKBACK_DAYS * 100).clip(0, 100)
    
    result["VolumeSurge"] = (result["Volume"] / result["AvgVolume30d"].replace(0, float('nan'))).fillna(1)
    
    support_7d = hist_grouped["Low"].min()
    resistance_7d = hist_grouped["High"].max()
    
    result["Support7d"] = result["StockCode"].map(support_7d).fillna(0)
    result["Resistance7d"] = result["StockCode"].map(resistance_7d).fillna(0)
    
    price_range = result["Resistance7d"] - result["Support7d"]
    result["PricePosition"] = ((result["Close"] - result["Support7d"]) / price_range.replace(0, float('nan')) * 100).fillna(50).clip(0, 100)
    
    def get_phase(group: pd.DataFrame) -> str:
        if len(group) < 5:
            return "Unknown"
        highs = group["High"].values
        lows = group["Low"].values
        if highs[-1] > highs[0] and lows[-1] > lows[0]:
            return "Trending Up"
        elif highs[-1] < highs[0] and lows[-1] < lows[0]:
            return "Trending Down"
        return "Sideways"
    
    phase = hist_grouped.apply(get_phase)
    result["MarketPhase"] = result["StockCode"].map(phase).fillna("Unknown")
    
    def calc_atr(group: pd.DataFrame) -> float:
        if "TrueRange" in group.columns and len(group) >= 3:
            return group["TrueRange"].tail(7).mean()
        if len(group) > 0:
            return (group["High"] - group["Low"]).mean()
        return 0.0
    
    atr_values = hist_grouped.apply(calc_atr)
    result["ATR7"] = result["StockCode"].map(atr_values).fillna(result["Close"] * 0.03)
    
    def calc_sma7(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return group["Close"].mean() if len(group) > 0 else 0
        return group["Close"].tail(7).mean()
    
    sma_values = hist_grouped.apply(calc_sma7)
    result["SMA7"] = result["StockCode"].map(sma_values).fillna(result["Close"])
    
    result["TrendStrength"] = ((result["Close"] - result["SMA7"]) / result["SMA7"] * 100).fillna(0)
    
    return result


def calculate_entry_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate entry signals (same logic as main_screener)"""
    result = df.copy()
    
    result["EntrySignal"] = "HOLD"
    result["SignalStrength"] = 0
    result["StopLoss"] = 0.0
    result["Target1"] = 0.0
    result["Target2"] = 0.0
    result["RiskReward"] = 0.0
    
    has_multiday = "PricePosition" in df.columns
    
    if not has_multiday:
        result["VolumeSurge"] = 1.0
        result["MarketPhase"] = "Unknown"
        result["PricePosition"] = 50.0
        result["Resistance7d"] = result["High"]
        result["SMA7"] = result["Close"]
        result["ATR7"] = result["Close"] * 0.03
        result["TrendStrength"] = 0.0
    
    for idx, row in result.iterrows():
        close = row["Close"]
        high = row["High"]
        resistance = row.get("Resistance7d", close)
        change_pct = row["ChangePct"]
        position = row.get("PricePosition", 50)
        phase = row.get("MarketPhase", "Unknown")
        surge = row.get("VolumeSurge", 1)
        atr = row.get("ATR7", close * 0.03)
        candle_strength = row.get("CandleStrength", 0.5)
        trend_strength = row.get("TrendStrength", 0)
        
        # STRATEGY 1: BREAKOUT
        breakout_distance = (high - resistance) / resistance * 100 if resistance > 0 else 0
        is_breakout = breakout_distance >= -2.0
        
        if (is_breakout and
            surge >= 1.2 and
            candle_strength >= 0.6 and
            change_pct >= 1.0 and
            phase != "Trending Down"):
            
            result.at[idx, "EntrySignal"] = "BUY - BREAKOUT"
            strength = min(30, change_pct * 15) + min(30, (surge - 1) * 20) + min(20, candle_strength * 25) + min(20, abs(trend_strength) * 5)
            result.at[idx, "SignalStrength"] = int(min(100, strength))
            
            stop = close - (2 * atr)
            result.at[idx, "StopLoss"] = round(max(stop, close * 0.92))
            result.at[idx, "Target1"] = round(close + (1.5 * atr))
            result.at[idx, "Target2"] = round(close + (3 * atr))
            
            risk = close - result.at[idx, "StopLoss"]
            reward = result.at[idx, "Target2"] - close
            result.at[idx, "RiskReward"] = round(reward / risk, 2) if risk > 0 else 0
        
        # STRATEGY 2: PULLBACK
        elif (15 <= position <= 50 and
              surge >= 1.1 and
              candle_strength >= 0.55 and
              -3.0 <= change_pct <= -0.3 and
              phase != "Trending Down"):
            
            result.at[idx, "EntrySignal"] = "BUY - PULLBACK"
            strength = min(25, (40 - abs(change_pct)) * 2) + min(25, (surge - 1) * 20) + min(25, candle_strength * 35) + min(25, trend_strength * 5)
            result.at[idx, "SignalStrength"] = int(min(100, strength))
            
            stop = close - (2.5 * atr)
            result.at[idx, "StopLoss"] = round(stop)
            result.at[idx, "Target1"] = round(close + (2 * atr))
            result.at[idx, "Target2"] = round(close + (3 * atr))
            
            risk = close - stop
            reward = result.at[idx, "Target2"] - close
            result.at[idx, "RiskReward"] = round(reward / risk, 2) if risk > 0 else 0
        
        # STRATEGY 3: MOMENTUM
        elif (change_pct >= 2.0 and
              surge >= 1.3 and
              candle_strength >= 0.6 and
              position < 80 and
              phase != "Trending Down"):
            
            result.at[idx, "EntrySignal"] = "BUY - MOMENTUM"
            strength = min(30, change_pct * 10) + min(30, (surge - 1) * 20) + min(20, candle_strength * 30) + min(20, abs(trend_strength) * 5)
            result.at[idx, "SignalStrength"] = int(min(100, strength))
            
            stop = close * 0.96
            result.at[idx, "StopLoss"] = round(stop)
            result.at[idx, "Target1"] = round(close * 1.03)
            result.at[idx, "Target2"] = round(close * 1.05)
            
            risk = close - stop
            reward = (close * 1.05) - close
            result.at[idx, "RiskReward"] = round(reward / risk, 2) if risk > 0 else 0
        
        # STRATEGY 4: RISKY BOUNCE
        elif (change_pct >= 5.0 and
              candle_strength >= 0.7 and
              position < 60):
            
            result.at[idx, "EntrySignal"] = "BUY - RISKY BOUNCE"
            strength = min(25, change_pct * 8) + min(25, candle_strength * 30) + 20
            result.at[idx, "SignalStrength"] = int(min(70, strength))
            
            stop = close * 0.95
            result.at[idx, "StopLoss"] = round(stop)
            result.at[idx, "Target1"] = round(close * 1.025)
            result.at[idx, "Target2"] = round(close * 1.04)
            
            risk = close - stop
            reward = (close * 1.04) - close
            result.at[idx, "RiskReward"] = round(reward / risk, 2) if risk > 0 else 0
    
    return result


def backtest_signal(entry_price: float, stop_loss: float, target1: float, target2: float,
                   next_day_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backtest a signal against next day's actual data
    Returns result with outcome, return%, and how it was hit
    """
    if not next_day_data or next_day_data.get("High", 0) == 0:
        return {
            "outcome": "NO_DATA",
            "return_pct": 0,
            "exit_price": entry_price,
            "hit_type": "NO_NEXT_DAY"
        }
    
    next_high = next_day_data["High"]
    next_low = next_day_data["Low"]
    next_close = next_day_data["Close"]
    
    # Check if stop loss was hit first (conservatively)
    if next_low <= stop_loss:
        return_pct = ((stop_loss - entry_price) / entry_price) * 100
        return {
            "outcome": "LOSS",
            "return_pct": return_pct,
            "exit_price": stop_loss,
            "hit_type": "STOP_LOSS"
        }
    
    # Check if Target2 was hit
    if next_high >= target2:
        return_pct = ((target2 - entry_price) / entry_price) * 100
        return {
            "outcome": "WIN",
            "return_pct": return_pct,
            "exit_price": target2,
            "hit_type": "TARGET2"
        }
    
    # Check if Target1 was hit
    if next_high >= target1:
        return_pct = ((target1 - entry_price) / entry_price) * 100
        return {
            "outcome": "WIN",
            "return_pct": return_pct,
            "exit_price": target1,
            "hit_type": "TARGET1"
        }
    
    # Neither target nor stop hit - exit at close
    return_pct = ((next_close - entry_price) / entry_price) * 100
    outcome = "WIN" if next_close >= entry_price else "LOSS"
    
    return {
        "outcome": outcome,
        "return_pct": return_pct,
        "exit_price": next_close,
        "hit_type": "CLOSE"
    }


def run_backtest(start_days_ago: int = 30, end_days_ago: int = 1) -> pd.DataFrame:
    """
    Run backtest for specified period
    Returns DataFrame with all trades
    """
    print("="*80)
    print("🧪 DAY TRADING STRATEGY BACKTEST")
    print("="*80)
    
    # Load all files
    files_data = sorted(glob.glob(os.path.join(DATA_DIR, "idx_stock_*.json")))
    files_backup = sorted(glob.glob(os.path.join("backup", "idx_stock_*.json")))
    all_files = sorted(files_data + files_backup)
    
    if len(all_files) < start_days_ago + 10:
        print(f"❌ Not enough data files. Found {len(all_files)}, need {start_days_ago + 10}")
        return pd.DataFrame()
    
    # Select testing period
    test_files = all_files[-(start_days_ago+1):-end_days_ago]
    
    print(f"\n📊 Backtest Period:")
    print(f"   Files: {len(test_files)} days")
    print(f"   From: {os.path.basename(test_files[0])}")
    print(f"   To: {os.path.basename(test_files[-1])}")
    print()
    
    all_trades: List[Dict[str, Any]] = []
    
    for i, file_path in enumerate(test_files):
        date_str = os.path.basename(file_path)
        print(f"[{i+1}/{len(test_files)}] Processing {date_str}...", end=" ")
        
        try:
            # Load today's data
            df_today = load_json(file_path)
            df_today = prepare(df_today)
            
            # Load historical for multi-day metrics
            # FIX: Find actual index in all_files, not test_files
            actual_index = all_files.index(file_path)
            hist_start = max(0, actual_index - LOOKBACK_DAYS)
            historical_files = all_files[hist_start:actual_index] if actual_index > 0 else []
            
            if len(historical_files) >= 3:
                df_hist_list = [prepare(load_json(f)) for f in historical_files]
                df_historical = pd.concat(df_hist_list, ignore_index=True)
                df_today = calculate_multi_day_metrics(df_today, df_historical)
            
            # Screen and generate signals
            df_screened = screen(df_today)
            df_signals = calculate_entry_signals(df_screened)
            buy_signals = df_signals[df_signals["EntrySignal"].str.contains("BUY", na=False)]
            
            if len(buy_signals) == 0:
                print("No signals")
                continue
            
            # Load next day's data for validation
            # FIX: Use actual_index, not i
            next_file = all_files[actual_index + 1] if actual_index + 1 < len(all_files) else None
            df_next = prepare(load_json(next_file)) if next_file else pd.DataFrame()
            
            # Backtest each signal
            for _, signal in buy_signals.iterrows():
                stock_code = signal["StockCode"]
                entry_price = signal["Close"]
                stop_loss = signal["StopLoss"]
                target1 = signal["Target1"]
                target2 = signal["Target2"]
                
                # Get next day's data for this stock
                next_day_data = df_next[df_next["StockCode"] == stock_code]
                if len(next_day_data) > 0:
                    next_day_dict = next_day_data.iloc[0].to_dict()
                else:
                    next_day_dict = {}
                
                # Run backtest
                result = backtest_signal(entry_price, stop_loss, target1, target2, next_day_dict)
                
                # Record trade
                trade = {
                    "Date": df_today["FileDate"].iloc[0],
                    "StockCode": stock_code,
                    "StockName": signal.get("StockName", ""),
                    "EntrySignal": signal["EntrySignal"],
                    "SignalStrength": signal["SignalStrength"],
                    "EntryPrice": entry_price,
                    "StopLoss": stop_loss,
                    "Target1": target1,
                    "Target2": target2,
                    "Outcome": result["outcome"],
                    "ExitPrice": result["exit_price"],
                    "ReturnPct": result["return_pct"],
                    "HitType": result["hit_type"],
                    "VolumeSurge": signal.get("VolumeSurge", 0),
                    "MarketPhase": signal.get("MarketPhase", "Unknown")
                }
                all_trades.append(trade)
            
            print(f"{len(buy_signals)} signals")
            
        except Exception as e:
            print(f"Error: {str(e)[:50]}")
            continue
    
    if not all_trades:
        print("\n❌ No trades generated in backtest period")
        return pd.DataFrame()
    
    return pd.DataFrame(all_trades)


def analyze_results(df: pd.DataFrame) -> None:
    """Analyze and display backtest results"""
    
    if len(df) == 0:
        print("\n❌ No results to analyze")
        return
    
    print("\n" + "="*80)
    print("📈 BACKTEST RESULTS")
    print("="*80)
    
    total_trades = len(df)
    valid_trades = df[df["Outcome"].isin(["WIN", "LOSS"])]
    
    if len(valid_trades) == 0:
        print("\n⚠️  No valid trades (all NO_DATA)")
        return
    
    wins = valid_trades[valid_trades["Outcome"] == "WIN"]
    losses = valid_trades[valid_trades["Outcome"] == "LOSS"]
    
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / len(valid_trades) * 100) if len(valid_trades) > 0 else 0
    
    avg_win = wins["ReturnPct"].mean() if len(wins) > 0 else 0
    avg_loss = losses["ReturnPct"].mean() if len(losses) > 0 else 0
    avg_return = valid_trades["ReturnPct"].mean()
    
    total_return = valid_trades["ReturnPct"].sum()
    
    print(f"\n📊 OVERVIEW:")
    print(f"   Total Trades: {total_trades}")
    print(f"   Valid Trades: {len(valid_trades)}")
    print(f"   Wins: {win_count} ({win_rate:.1f}%)")
    print(f"   Losses: {loss_count}")
    
    print(f"\n💰 RETURNS:")
    print(f"   Average Win: {avg_win:+.2f}%")
    print(f"   Average Loss: {avg_loss:+.2f}%")
    print(f"   Average Return: {avg_return:+.2f}%")
    print(f"   Cumulative Return: {total_return:+.2f}%")
    
    # Expectancy
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    print(f"   Expectancy: {expectancy:+.2f}% per trade")
    
    # By signal type
    print(f"\n📡 BY SIGNAL TYPE:")
    for signal_type in df["EntrySignal"].unique():
        if "BUY" not in signal_type:
            continue
        subset = valid_trades[valid_trades["EntrySignal"] == signal_type]
        if len(subset) == 0:
            continue
        wins_subset = subset[subset["Outcome"] == "WIN"]
        wr = len(wins_subset) / len(subset) * 100
        avg_ret = subset["ReturnPct"].mean()
        print(f"   {signal_type:20s}: {len(subset):3d} trades, WR: {wr:5.1f}%, Avg: {avg_ret:+6.2f}%")
    
    # By exit type
    print(f"\n🎯 BY EXIT TYPE:")
    for hit_type in ["TARGET2", "TARGET1", "CLOSE", "STOP_LOSS"]:
        subset = valid_trades[valid_trades["HitType"] == hit_type]
        if len(subset) == 0:
            continue
        pct = len(subset) / len(valid_trades) * 100
        avg_ret = subset["ReturnPct"].mean()
        print(f"   {hit_type:15s}: {len(subset):3d} ({pct:5.1f}%), Avg: {avg_ret:+6.2f}%")
    
    # Best and worst
    print(f"\n🏆 BEST TRADES:")
    top5 = valid_trades.nlargest(5, "ReturnPct")
    for _, trade in top5.iterrows():
        print(f"   {trade['Date']} {trade['StockCode']:6s} {trade['EntrySignal']:20s} {trade['ReturnPct']:+6.2f}% ({trade['HitType']})")
    
    print(f"\n💔 WORST TRADES:")
    bottom5 = valid_trades.nsmallest(5, "ReturnPct")
    for _, trade in bottom5.iterrows():
        print(f"   {trade['Date']} {trade['StockCode']:6s} {trade['EntrySignal']:20s} {trade['ReturnPct']:+6.2f}% ({trade['HitType']})")
    
    # Save to CSV
    output_file = os.path.join(OUTPUT_DIR, f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n" + "="*80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Backtest Day Trading Strategy')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest (default: 30)')
    args = parser.parse_args()
    
    df_results = run_backtest(start_days_ago=args.days)
    
    if len(df_results) > 0:
        analyze_results(df_results)
    else:
        print("\n❌ Backtest failed - no results generated")


if __name__ == "__main__":
    main()
