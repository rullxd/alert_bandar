"""
Unified Day Trading Screener - IMPROVED VERSION
Combines: Basic Screening + Broker Analysis + AI Insight

Key Improvements:
- Separated liquidity screening from signal generation
- Independent strategy scoring (all strategies evaluated)
- Best strategy selection with quality threshold
- Fixed RISKY BOUNCE logic (only for reversals)
- Optimized strategy priority
- Added top N signal filtering
- Consolidated scoring system

Usage:
    python main_screener_improved.py                    # Basic only
    python main_screener_improved.py --broker           # + Broker data
    python main_screener_improved.py --ai               # + AI analysis  
    python main_screener_improved.py --broker --ai      # Full features (recommended)
    python main_screener_improved.py --top 10           # Limit to top 10 signals
"""

import os
import glob
import time
import html
import json
import argparse
import pandas as pd
import requests
from datetime import datetime, date
from dotenv import load_dotenv
from typing import Any, Dict, List, cast, Tuple

# Load environment variables
load_dotenv()

# =========================
# CONFIGURATION
# =========================
DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Telegram Config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# GOAPI Config (for broker data)
_goapi_keys: List[str] = []

# Try loading from ket.txt first
if os.path.exists("ket.txt"):
    try:
        with open("ket.txt", "r") as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip first line "GOAPI KEYS:"
                key = line.strip()
                if key and not key.startswith("GOAPI") and not key.startswith("#"):
                    _goapi_keys.append(key)
    except:
        pass

# Fallback: Load from .env
if not _goapi_keys:
    for i in range(1, 10):
        key = os.getenv(f"GOAPI_KEY_{i}", "").strip()
        if key:
            _goapi_keys.append(key)
    
    if not _goapi_keys:
        key = os.getenv("GOAPI_KEY", "").strip()
        if key:
            _goapi_keys.append(key)

_current_key_index = 0
GOAPI_BASE = "https://api.goapi.io"

# Gemini Config (for AI)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# =========================
# PARAMETER DAY TRADING
# =========================

# STAGE 1: Liquidity Screening (Strict)
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

MIN_NET_FOREIGN = -1_000_000
MIN_NONREG_RATIO = 0.01
MIN_VALUE_RANK_PERCENTILE = 0.5

# Multi-day requirements (RELAXED)
LOOKBACK_DAYS = 7
MIN_LIQUID_DAYS = 4              # Relaxed from 5 to 4
MIN_CONSISTENCY_SCORE = 40       # 40% consistency minimum

# STAGE 2: Strategy Parameters (TUNED)
MOMENTUM_THRESHOLD = 2.0         # Raised from 1.5 (clearer momentum)
BREAKOUT_VOLUME_SURGE = 1.8      # Raised from 1.5 (stricter)
PULLBACK_VOLUME_SURGE = 1.3      # Relaxed for pullback
BOUNCE_VOLUME_SURGE = 1.2        # Relaxed for bounce

# Stop loss & targets
ATR_MULTIPLIER_STOP = 2.0
ATR_MULTIPLIER_TARGET = 3.0

# STAGE 3: Signal Quality Gate
MIN_SIGNAL_SCORE = 60            # Minimum score to generate BUY signal
MAX_SIGNALS = 15                 # Maximum signals to send (top N)

# Broker settings
MIN_TOP_BUYERS = 5
MIN_TOP_SELLERS = 5


# =========================
# CORE FUNCTIONS
# =========================

def load_json(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw["data"])
    df["FileDate"] = extract_date(path)
    return df


def extract_date(path: str) -> date:
    name = os.path.basename(path)
    digits = "".join([c for c in name if c.isdigit()])
    return datetime.strptime(digits[-8:], "%Y%m%d").date()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "Previous","Close","Volume","Value","Frequency","High","Low",
        "ForeignBuy","ForeignSell","NonRegularVolume",
        "TradebleShares","Bid","Offer"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)  # type: ignore
    
    df["NetForeign"] = df["ForeignBuy"] - df["ForeignSell"]  # type: ignore
    df["NonRegRatio"] = (df["NonRegularVolume"] / df["Volume"].replace(0, float('nan'))).fillna(0)  # type: ignore
    df["ChangePct"] = ((df["Close"] - df["Previous"]) / df["Previous"] * 100).replace([float("inf"), -float("inf")], 0)  # type: ignore
    df["Volatility"] = ((df["High"] - df["Low"]) / df["Previous"] * 100).replace([float("inf"), -float("inf")], 0)  # type: ignore
    df["SpreadPct"] = df.apply(  # type: ignore
        lambda r: abs(r["Offer"] - r["Bid"]) / r["Close"] * 100 if r["Close"] > 0 else 0,  # type: ignore
        axis=1
    )
    df["AvgTradeSize"] = df.apply(  # type: ignore
        lambda r: r["Value"] / r["Frequency"] if r["Frequency"] > 0 else 0,  # type: ignore
        axis=1
    )
    df["VolumeVsShares"] = df.apply(  # type: ignore
        lambda r: r["Volume"] / r["TradebleShares"] if r["TradebleShares"] > 0 else 0,  # type: ignore
        axis=1
    )
    df["ValueRank"] = df["Value"].rank(pct=True)  # type: ignore
    
    # Calculate Candle Strength (bullish/bearish)
    # 1.0 = close at high (very bullish), 0.0 = close at low (very bearish)
    df["CandleStrength"] = df.apply(  # type: ignore
        lambda r: (r["Close"] - r["Low"]) / (r["High"] - r["Low"]) if r["High"] > r["Low"] else 0.5,  # type: ignore
        axis=1
    )
    
    return df


def calculate_multi_day_metrics(df_today: pd.DataFrame, df_historical: pd.DataFrame) -> pd.DataFrame:
    """Calculate multi-day comparison metrics"""
    result = df_today.copy()
    
    # Calculate True Range for historical data (needed for ATR)
    df_hist = df_historical.copy()
    df_hist["PrevClose"] = df_hist.groupby("StockCode")["Close"].shift(1)
    df_hist["TR1"] = df_hist["High"] - df_hist["Low"]
    df_hist["TR2"] = (df_hist["High"] - df_hist["PrevClose"]).abs()
    df_hist["TR3"] = (df_hist["Low"] - df_hist["PrevClose"]).abs()
    df_hist["TrueRange"] = df_hist[["TR1", "TR2", "TR3"]].max(axis=1)  # type: ignore
    
    hist_grouped = df_hist.groupby("StockCode")
    
    avg_volume = hist_grouped["Volume"].mean()
    avg_value = hist_grouped["Value"].mean()
    avg_frequency = hist_grouped["Frequency"].mean()
    
    liquid_days = hist_grouped.apply(
        lambda g: ((g["Value"] >= MIN_VALUE * 0.5) & (g["Volume"] >= MIN_VOLUME * 0.5)).sum()
    )
    
    def get_trend(group: pd.DataFrame) -> str:
        if len(group) < 5:
            return "Unknown"
        recent_avg = group.tail(5)["Close"].mean()
        older_avg = group.head(5)["Close"].mean()
        if recent_avg > older_avg * 1.02:
            return "Up"
        elif recent_avg < older_avg * 0.98:
            return "Down"
        return "Sideways"
    
    trend = hist_grouped.apply(get_trend)
    
    result["AvgVolume30d"] = result["StockCode"].map(avg_volume).fillna(0)  # type: ignore
    result["AvgValue30d"] = result["StockCode"].map(avg_value).fillna(0)  # type: ignore
    result["AvgFreq30d"] = result["StockCode"].map(avg_frequency).fillna(0)  # type: ignore
    result["LiquidDays30d"] = result["StockCode"].map(liquid_days).fillna(0)  # type: ignore
    result["Trend30d"] = result["StockCode"].map(trend).fillna("Unknown")  # type: ignore
    
    result["VolumeSurge"] = (result["Volume"] / result["AvgVolume30d"].replace(0, float('nan'))).fillna(1)  # type: ignore
    result["ValueSurge"] = (result["Value"] / result["AvgValue30d"].replace(0, float('nan'))).fillna(1)  # type: ignore
    result["ConsistencyScore"] = (result["LiquidDays30d"] / LOOKBACK_DAYS * 100).clip(0, 100)  # type: ignore
    
    support_7d = hist_grouped["Low"].min()
    resistance_7d = hist_grouped["High"].max()
    
    result["Support7d"] = result["StockCode"].map(support_7d).fillna(0)  # type: ignore
    result["Resistance7d"] = result["StockCode"].map(resistance_7d).fillna(0)  # type: ignore
    result["DistFromSupport"] = ((result["Close"] - result["Support7d"]) / result["Support7d"] * 100).fillna(0)  # type: ignore
    result["DistFromResistance"] = ((result["Resistance7d"] - result["Close"]) / result["Close"] * 100).fillna(0)  # type: ignore
    
    price_range = result["Resistance7d"] - result["Support7d"]
    result["PricePosition"] = ((result["Close"] - result["Support7d"]) / price_range.replace(0, float('nan')) * 100).fillna(50).clip(0, 100)  # type: ignore
    
    def get_phase(group: pd.DataFrame) -> str:
        if len(group) < 3:
            return "Unknown"
        highs = group["High"].values
        lows = group["Low"].values
        if highs[-1] > highs[0] and lows[-1] > lows[0]:
            return "Trending Up"
        elif highs[-1] < highs[0] and lows[-1] < lows[0]:
            return "Trending Down"
        return "Sideways"
    
    phase = hist_grouped.apply(get_phase)
    result["MarketPhase"] = result["StockCode"].map(phase).fillna("Unknown")  # type: ignore
    
    # Calculate ATR (Average True Range) from historical data
    def calc_atr(group: pd.DataFrame) -> float:
        if "TrueRange" in group.columns and len(group) >= 3:
            return group["TrueRange"].tail(7).mean()
        if len(group) > 0:
            return (group["High"] - group["Low"]).mean()
        return 0.0
    
    atr_values = hist_grouped.apply(calc_atr)
    result["ATR7"] = result["StockCode"].map(atr_values).fillna(result["Close"] * 0.03)  # type: ignore
    
    # Calculate SMA7 (Simple Moving Average 7-day)
    def calc_sma7(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return group["Close"].mean() if len(group) > 0 else 0
        return group["Close"].tail(7).mean()
    
    sma_values = hist_grouped.apply(calc_sma7)
    result["SMA7"] = result["StockCode"].map(sma_values).fillna(result["Close"])  # type: ignore
    
    # Trend Strength (distance from SMA in %)
    result["TrendStrength"] = ((result["Close"] - result["SMA7"]) / result["SMA7"] * 100).fillna(0)  # type: ignore
    
    return result


# =========================
# STAGE 1: LIQUIDITY SCREENING
# =========================

def screen_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 1: Filter for liquidity only
    This is strict - we only want tradeable stocks
    """
    
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
            (df["ConsistencyScore"] >= MIN_CONSISTENCY_SCORE)
        )
        filtered = df[base_filter & multiday_filter].copy()
    else:
        filtered = df[base_filter].copy()
    
    return filtered


# =========================
# STAGE 2: STRATEGY SCORING
# =========================

def calculate_momentum_score(df: pd.DataFrame) -> pd.Series:  # type: ignore
    """
    Score MOMENTUM strategy (0-100)
    Strong upward move with volume confirmation
    Best for: Riding existing strong moves
    """
    score = pd.Series(0.0, index=df.index)
    
    for idx, row in df.iterrows():
        change_pct = row["ChangePct"]
        surge = row.get("VolumeSurge", 1.0)
        candle = row.get("CandleStrength", 0.5)
        position = row.get("PricePosition", 50)
        phase = row.get("MarketPhase", "Unknown")
        trend_str = row.get("TrendStrength", 0)
        
        # Requirements
        if (change_pct >= MOMENTUM_THRESHOLD and
            surge >= 1.3 and
            candle >= 0.6 and
            position < 80 and
            phase != "Trending Down"):
            
            s = 0.0
            # Change momentum (max 30)
            s += min(30, change_pct * 10)
            # Volume surge (max 30)
            s += min(30, (surge - 1) * 20)
            # Candle strength (max 20)
            s += min(20, candle * 30)
            # Trend strength (max 20)
            s += min(20, abs(trend_str) * 5)
            
            score.loc[idx] = min(100, s)  # type: ignore
    
    return score


def calculate_breakout_score(df: pd.DataFrame) -> pd.Series:  # type: ignore
    """
    Score BREAKOUT strategy (0-100)
    Price breaking resistance with strong volume
    Best for: Catching fresh momentum at resistance break
    """
    score = pd.Series(0.0, index=df.index)
    
    for idx, row in df.iterrows():
        close = row["Close"]
        high = row["High"]
        resistance = row.get("Resistance7d", close)
        change_pct = row["ChangePct"]
        surge = row.get("VolumeSurge", 1.0)
        candle = row.get("CandleStrength", 0.5)
        phase = row.get("MarketPhase", "Unknown")
        trend_str = row.get("TrendStrength", 0)
        
        # Breakout distance
        breakout_dist = (high - resistance) / resistance * 100 if resistance > 0 else -10
        is_breakout = breakout_dist >= -2.0  # Within 2% of resistance
        
        # Requirements (STRICTER volume)
        if (is_breakout and
            surge >= BREAKOUT_VOLUME_SURGE and  # 1.8x (stricter)
            candle >= 0.6 and
            change_pct >= 1.0 and
            phase != "Trending Down"):
            
            s = 0.0
            # Change momentum (max 30)
            s += min(30, change_pct * 15)
            # Volume surge (max 35) - higher weight for breakout
            s += min(35, (surge - 1) * 25)
            # Candle strength (max 20)
            s += min(20, candle * 25)
            # Trend strength (max 15)
            s += min(15, abs(trend_str) * 5)
            
            score.loc[idx] = min(100, s)  # type: ignore
    
    return score


def calculate_pullback_score(df: pd.DataFrame) -> pd.Series:  # type: ignore
    """
    Score PULLBACK strategy (0-100)
    Buying dip in uptrend
    Best for: Better entry timing on established uptrends
    """
    score = pd.Series(0.0, index=df.index)
    
    for idx, row in df.iterrows():
        position = row.get("PricePosition", 50)
        change_pct = row["ChangePct"]
        surge = row.get("VolumeSurge", 1.0)
        candle = row.get("CandleStrength", 0.5)
        phase = row.get("MarketPhase", "Unknown")
        trend_str = row.get("TrendStrength", 0)
        
        # Requirements
        if (15 <= position <= 50 and  # In lower half of range
            surge >= PULLBACK_VOLUME_SURGE and  # 1.3x
            candle >= 0.55 and
            -3.0 <= change_pct <= -0.3 and  # Small pullback
            phase != "Trending Down"):
            
            s = 0.0
            # Pullback quality (max 30) - smaller pullback = better
            pullback_quality = max(0, 3 - abs(change_pct))
            s += min(30, pullback_quality * 10)
            # Volume surge (max 25)
            s += min(25, (surge - 1) * 20)
            # Candle strength (max 25)
            s += min(25, candle * 35)
            # Trend strength (max 20)
            s += min(20, trend_str * 5)  # Positive trend strength
            
            score.loc[idx] = min(100, s)  # type: ignore
    
    return score


def calculate_bounce_score(df: pd.DataFrame) -> pd.Series:  # type: ignore
    """
    Score BOUNCE strategy (0-70) - RISKY!
    Reversal play on oversold stocks
    Best for: Catching dead cat bounce (HIGH RISK)
    
    FIXED: Only triggers for DOWN moves that are bouncing
    """
    score = pd.Series(0.0, index=df.index)
    
    for idx, row in df.iterrows():
        close = row["Close"]
        low = row["Low"]
        change_pct = row["ChangePct"]
        candle = row.get("CandleStrength", 0.5)
        position = row.get("PricePosition", 50)
        surge = row.get("VolumeSurge", 1.0)
        
        # FIXED: Only for stocks that FELL then bounced
        # Check if we're bouncing from a low (not a strong up move!)
        bounce_from_low = (close > low * 1.02)  # Closed 2% above low
        had_down_move = change_pct <= -1.5  # Was down at some point
        
        # Requirements (CORRECTED LOGIC)
        if (had_down_move and  # Stock went DOWN
            bounce_from_low and  # Now bouncing from low
            candle >= 0.7 and  # Strong reversal candle
            position < 60 and  # In lower range
            surge >= BOUNCE_VOLUME_SURGE):  # 1.2x volume
            
            s = 0.0
            # Bounce strength (max 25)
            bounce_str = (close - low) / (close - low + 0.01) * 100
            s += min(25, bounce_str * 0.3)
            # Candle strength (max 25)
            s += min(25, candle * 30)
            # Volume (max 20)
            s += min(20, (surge - 1) * 20)
            
            # Cap at 70 (indicate high risk)
            score.loc[idx] = min(70, s)  # type: ignore
    
    return score


# =========================
# STAGE 3: SIGNAL GENERATION
# =========================

def generate_signals(df: pd.DataFrame, min_score: int = MIN_SIGNAL_SCORE) -> pd.DataFrame:
    """
    Stage 3: Generate BUY signals from strategy scores
    
    Process:
    1. Calculate all strategy scores
    2. Pick best strategy per stock
    3. Only signal if score >= min_score (quality gate)
    4. Calculate stops/targets based on strategy
    """
    result = df.copy()
    
    # Ensure multi-day columns exist
    has_multiday = "PricePosition" in df.columns
    if not has_multiday:
        result["VolumeSurge"] = 1.0
        result["MarketPhase"] = "Unknown"
        result["PricePosition"] = 50.0
        result["Resistance7d"] = result["High"]
        result["Support7d"] = result["Low"]
        result["SMA7"] = result["Close"]
        result["ATR7"] = result["Close"] * 0.03
        result["TrendStrength"] = 0.0
        result["CandleStrength"] = 0.5
    
    # Calculate all strategy scores
    print("   Scoring MOMENTUM strategy...")
    result["MomentumScore"] = calculate_momentum_score(result)
    
    print("   Scoring BREAKOUT strategy...")
    result["BreakoutScore"] = calculate_breakout_score(result)
    
    print("   Scoring PULLBACK strategy...")
    result["PullbackScore"] = calculate_pullback_score(result)
    
    print("   Scoring BOUNCE strategy...")
    result["BounceScore"] = calculate_bounce_score(result)
    
    # Find best strategy for each stock
    score_cols = ["MomentumScore", "BreakoutScore", "PullbackScore", "BounceScore"]
    result["BestScore"] = result[score_cols].max(axis=1)  # type: ignore
    result["BestStrategy"] = result[score_cols].idxmax(axis=1)  # type: ignore
    
    # Map strategy names
    strategy_map = {
        "MomentumScore": "MOMENTUM",
        "BreakoutScore": "BREAKOUT",
        "PullbackScore": "PULLBACK",
        "BounceScore": "RISKY BOUNCE"
    }
    result["BestStrategy"] = result["BestStrategy"].map(strategy_map)  # type: ignore
    
    # Initialize signal columns
    result["EntrySignal"] = "NO SIGNAL"
    result["SignalStrength"] = 0
    result["StopLoss"] = 0.0
    result["Target1"] = 0.0
    result["Target2"] = 0.0
    result["RiskReward"] = 0.0
    
    # Generate signals only for stocks with score >= min_score
    for idx, row in result.iterrows():
        if row["BestScore"] >= min_score:
            strategy = row["BestStrategy"]
            close = row["Close"]
            atr = row["ATR7"]
            
            result.at[idx, "EntrySignal"] = f"BUY - {strategy}"
            result.at[idx, "SignalStrength"] = int(row["BestScore"])
            
            # Calculate stops/targets based on strategy
            if strategy == "MOMENTUM":
                # Fixed % stops for momentum
                stop = close * 0.96  # 4% stop
                result.at[idx, "StopLoss"] = round(stop)
                result.at[idx, "Target1"] = round(close * 1.03)  # 3%
                result.at[idx, "Target2"] = round(close * 1.05)  # 5%
                
            elif strategy == "BREAKOUT":
                # ATR-based stops for breakout
                stop = close - (2 * atr)
                result.at[idx, "StopLoss"] = round(max(stop, close * 0.92))
                result.at[idx, "Target1"] = round(close + (1.5 * atr))
                result.at[idx, "Target2"] = round(close + (3 * atr))
                
            elif strategy == "PULLBACK":
                # ATR-based stops for pullback (wider)
                stop = close - (2.5 * atr)
                result.at[idx, "StopLoss"] = round(stop)
                result.at[idx, "Target1"] = round(close + (2 * atr))
                result.at[idx, "Target2"] = round(close + (3 * atr))
                
            elif strategy == "RISKY BOUNCE":
                # Fixed % stops for bounce (tighter)
                stop = close * 0.95  # 5% stop
                result.at[idx, "StopLoss"] = round(stop)
                result.at[idx, "Target1"] = round(close * 1.025)  # 2.5%
                result.at[idx, "Target2"] = round(close * 1.04)   # 4%
            
            # Calculate Risk/Reward
            risk = close - result.at[idx, "StopLoss"]
            reward = result.at[idx, "Target2"] - close
            rr = reward / risk if risk > 0 else 0
            result.at[idx, "RiskReward"] = round(rr, 2)
    
    return result


# =========================
# BROKER FUNCTIONS (Optional)
# =========================

def get_current_api_key() -> str:
    global _current_key_index
    if not _goapi_keys:
        raise RuntimeError("No GOAPI keys configured")
    return _goapi_keys[_current_key_index]


def rotate_api_key() -> None:
    global _current_key_index
    if not _goapi_keys:
        return
    _current_key_index = (_current_key_index + 1) % len(_goapi_keys)


def http_get_json(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return cast(Dict[str, Any], r.json())


def fetch_broker_summary(stock_code: str, trade_date: str, max_retries: int = 3) -> Dict[str, Any]:
    if not _goapi_keys:
        raise RuntimeError("No GOAPI keys available")
    
    for attempt in range(max_retries):
        try:
            key = get_current_api_key()
            url = f"{GOAPI_BASE}/stock/idx/{stock_code}/broker"
            headers = {"X-API-KEY": key}
            params = {"date": trade_date}
            
            data = http_get_json(url, headers, params)
            if data and isinstance(data, dict):
                return data
            
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                rotate_api_key()
                time.sleep(1)
                continue
            raise
        except Exception:
            raise
    
    raise RuntimeError(f"Failed to fetch broker data after {max_retries} retries")


def extract_top_buyers(broker_data: Dict[str, Any], min_top: int = MIN_TOP_BUYERS) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    data_list = broker_data.get("data", [])
    
    if not isinstance(data_list, list):
        return results
    
    for item in data_list:
        if not isinstance(item, dict):
            continue
        top_buy = item.get("top_buy", [])
        if isinstance(top_buy, list) and len(top_buy) >= min_top:
            for broker_entry in top_buy[:min_top]:
                if isinstance(broker_entry, dict):
                    results.append({
                        "broker": broker_entry.get("broker", ""),
                        "value": broker_entry.get("value", 0),
                        "percentage": broker_entry.get("percentage", 0)
                    })
            break
    
    return results


def extract_top_sellers(broker_data: Dict[str, Any], min_top: int = MIN_TOP_SELLERS) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    data_list = broker_data.get("data", [])
    
    if not isinstance(data_list, list):
        return results
    
    for item in data_list:
        if not isinstance(item, dict):
            continue
        top_sell = item.get("top_sell", [])
        if isinstance(top_sell, list) and len(top_sell) >= min_top:
            for broker_entry in top_sell[:min_top]:
                if isinstance(broker_entry, dict):
                    results.append({
                        "broker": broker_entry.get("broker", ""),
                        "value": broker_entry.get("value", 0),
                        "percentage": broker_entry.get("percentage", 0)
                    })
            break
    
    return results


# =========================
# TELEGRAM FORMATTING
# =========================

def format_telegram_basic(signals: pd.DataFrame) -> str:
    """Format for basic mode (no broker/AI)"""
    lines: List[str] = []
    
    for i, (_, row) in enumerate(signals.iterrows(), 1):
        lines.append(f"<b>📊 {row['StockCode']}</b> - {row['StockName']}")
        lines.append(f"💰 Price: Rp {int(row['Close']):,}")
        lines.append(f"📈 Change: {row['ChangePct']:.2f}%")
        lines.append(f"📊 Volume: {int(row['Volume']):,}")
        lines.append(f"💵 Value: Rp {int(row['Value']/1e9):.2f}B")
        lines.append("")
        lines.append(f"🎯 <b>{row['EntrySignal']}</b>")
        lines.append(f"⭐ Score: {row['SignalStrength']}/100")
        lines.append(f"🛡️ Stop: Rp {int(row['StopLoss']):,}")
        lines.append(f"🎯 T1: Rp {int(row['Target1']):,} | T2: Rp {int(row['Target2']):,}")
        lines.append(f"📊 R/R: {row['RiskReward']:.2f}")
        
        if i < len(signals):
            lines.append("━━━━━━━━━━━━━━━━━━")
    
    return "\n".join(lines)


def format_telegram_with_broker(
    row: pd.Series,  # type: ignore
    buyers: List[Dict[str, Any]],
    sellers: List[Dict[str, Any]],
    trade_date: str
) -> str:
    """Format with broker data"""
    lines: List[str] = []
    
    lines.append(f"<b>📊 {row['StockCode']}</b> - {row['StockName']}")
    lines.append(f"💰 Price: Rp {int(row['Close']):,}")
    lines.append(f"📈 Change: {row['ChangePct']:.2f}%")
    lines.append(f"📊 Volume: {int(row['Volume']):,}")
    lines.append(f"💵 Value: Rp {int(row['Value']/1e9):.2f}B")
    lines.append("")
    lines.append(f"🎯 <b>{row['EntrySignal']}</b>")
    lines.append(f"⭐ Score: {row['SignalStrength']}/100")
    lines.append(f"🛡️ Stop: Rp {int(row['StopLoss']):,}")
    lines.append(f"🎯 T1: Rp {int(row['Target1']):,} | T2: Rp {int(row['Target2']):,}")
    lines.append(f"📊 R/R: {row['RiskReward']:.2f}")
    
    # Broker data
    if buyers:
        lines.append("")
        lines.append("<b>🟢 TOP BUYERS:</b>")
        for b in buyers:
            val_b = b['value'] / 1e9
            lines.append(f"  • {b['broker']}: Rp {val_b:.2f}B ({b['percentage']:.1f}%)")
    
    if sellers:
        lines.append("")
        lines.append("<b>🔴 TOP SELLERS:</b>")
        for s in sellers:
            val_s = s['value'] / 1e9
            lines.append(f"  • {s['broker']}: Rp {val_s:.2f}B ({s['percentage']:.1f}%)")
    
    return "\n".join(lines)


def format_telegram_with_ai(
    row: pd.Series,  # type: ignore
    buyers: List[Dict[str, Any]],
    sellers: List[Dict[str, Any]],
    trade_date: str,
    ai_result: Dict[str, Any]
) -> str:
    """Format with AI analysis"""
    lines: List[str] = []
    
    lines.append(f"<b>📊 {row['StockCode']}</b> - {row['StockName']}")
    lines.append(f"💰 Price: Rp {int(row['Close']):,}")
    lines.append(f"📈 Change: {row['ChangePct']:.2f}%")
    lines.append(f"📊 Volume: {int(row['Volume']):,}")
    lines.append(f"💵 Value: Rp {int(row['Value']/1e9):.2f}B")
    lines.append("")
    lines.append(f"🎯 <b>{row['EntrySignal']}</b>")
    lines.append(f"⭐ Score: {row['SignalStrength']}/100")
    lines.append(f"🛡️ Stop: Rp {int(row['StopLoss']):,}")
    lines.append(f"🎯 T1: Rp {int(row['Target1']):,} | T2: Rp {int(row['Target2']):,}")
    lines.append(f"📊 R/R: {row['RiskReward']:.2f}")
    
    # Broker data
    if buyers:
        lines.append("")
        lines.append("<b>🟢 TOP BUYERS:</b>")
        for b in buyers:
            val_b = b['value'] / 1e9
            lines.append(f"  • {b['broker']}: Rp {val_b:.2f}B ({b['percentage']:.1f}%)")
    
    if sellers:
        lines.append("")
        lines.append("<b>🔴 TOP SELLERS:</b>")
        for s in sellers:
            val_s = s['value'] / 1e9
            lines.append(f"  • {s['broker']}: Rp {val_s:.2f}B ({s['percentage']:.1f}%)")
    
    # AI analysis
    if ai_result.get('success'):
        analysis = ai_result.get('analysis', {})
        lines.append("")
        lines.append("<b>🤖 AI ANALYSIS:</b>")
        
        if 'recommendation' in analysis:
            rec = analysis['recommendation']
            lines.append(f"<b>Recommendation:</b> {rec}")
        
        if 'key_points' in analysis:
            points = analysis['key_points']
            if isinstance(points, list):
                for p in points[:3]:
                    lines.append(f"  • {p}")
        
        if 'risk_level' in analysis:
            lines.append(f"<b>Risk:</b> {analysis['risk_level']}")
    else:
        lines.append("")
        lines.append(f"<b>🤖 AI:</b> {ai_result.get('error', 'Analysis failed')}")
    
    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    
    try:
        r = requests.post(url, json=data, timeout=30)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"[ERROR] Telegram failed: {e}")
        return False


# =========================
# MAIN FUNCTION
# =========================

def main():
    parser = argparse.ArgumentParser(description='Unified Day Trading Screener - IMPROVED')
    parser.add_argument('--broker', action='store_true', help='Include broker analysis')
    parser.add_argument('--ai', action='store_true', help='Include AI analysis')
    parser.add_argument('--top', type=int, default=MAX_SIGNALS, help=f'Max signals to send (default: {MAX_SIGNALS})')
    parser.add_argument('--min-score', type=int, default=MIN_SIGNAL_SCORE, help=f'Min signal score (default: {MIN_SIGNAL_SCORE})')
    args = parser.parse_args()
    
    use_broker = args.broker
    use_ai = args.ai
    max_signals = args.top
    min_score = args.min_score
    
    # Auto-detect features
    if not use_broker and not use_ai:
        if _goapi_keys:
            print("[i] GOAPI keys detected. Use --broker to enable broker analysis")
        if GEMINI_API_KEY:
            print("[i] Gemini API key detected. Use --ai to enable AI analysis")
    
    print("="*70)
    print("🚀 UNIFIED DAY TRADING SCREENER - IMPROVED")
    print("="*70)
    print(f"📊 Mode: {'Basic' if not use_broker and not use_ai else ''}")
    if use_broker:
        print("   ✅ Broker Analysis: ENABLED")
    if use_ai:
        print("   ✅ AI Analysis: ENABLED")
    print(f"   🎯 Min Signal Score: {min_score}")
    print(f"   📊 Max Signals: {max_signals}")
    print("="*70)
    
    # Load data
    files_data = glob.glob(os.path.join(DATA_DIR, "idx_stock_*.json"))
    files_backup = glob.glob(os.path.join("backup", "idx_stock_*.json"))
    files = sorted(files_data + files_backup)
    
    if not files:
        print("[ERROR] No data files found")
        return
    
    latest_file = files[-1]
    print(f"\n📅 Processing: {os.path.basename(latest_file)}")
    
    df_today = load_json(latest_file)
    df_today = prepare(df_today)
    
    # Load historical data
    historical_files = files[-(LOOKBACK_DAYS+1):-1] if len(files) > LOOKBACK_DAYS else files[:-1]
    
    if len(historical_files) >= 3:
        print(f"📊 Loading {len(historical_files)} days historical data...")
        df_hist_list: List[pd.DataFrame] = []
        for f in historical_files:
            try:
                df_temp = load_json(f)
                df_temp = prepare(df_temp)
                df_hist_list.append(df_temp)
            except Exception:
                continue
        
        if df_hist_list:
            df_historical = pd.concat(df_hist_list, ignore_index=True)
            df_today = calculate_multi_day_metrics(df_today, df_historical)
            print("✅ Multi-day analysis complete\n")
    
    # STAGE 1: Liquidity Screening
    print("🔍 STAGE 1: Liquidity screening...")
    df_liquid = screen_liquidity(df_today)
    print(f"✅ {len(df_liquid)} stocks passed liquidity filter\n")
    
    if len(df_liquid) == 0:
        print("No stocks passed liquidity screening")
        return
    
    # STAGE 2 & 3: Strategy Scoring + Signal Generation
    print("🎯 STAGE 2-3: Scoring strategies & generating signals...")
    df_signals = generate_signals(df_liquid, min_score=min_score)
    
    # Filter only BUY signals
    buy_signals = df_signals[df_signals["EntrySignal"].str.contains("BUY", na=False)].copy()
    
    # Sort by SignalStrength and limit to top N
    buy_signals = buy_signals.sort_values("SignalStrength", ascending=False).head(max_signals)
    
    print(f"✅ Generated {len(buy_signals)} BUY signals (top {max_signals})\n")
    
    if len(buy_signals) == 0:
        print("No BUY signals generated with score >= {}".format(min_score))
        return
    
    # Show strategy breakdown
    print("📊 Strategy Breakdown:")
    strategy_counts = buy_signals["BestStrategy"].value_counts()
    for strategy, count in strategy_counts.items():
        print(f"   {strategy}: {count}")
    print()
    
    trade_date = df_today['FileDate'].iloc[0]
    trade_date_str = trade_date.strftime("%Y-%m-%d")
    
    # Process signals based on mode
    results: List[str] = []
    
    if not use_broker and not use_ai:
        # Basic mode - single message
        msg = format_telegram_basic(buy_signals)
        results.append(msg)
    
    elif use_broker and not use_ai:
        # Broker mode
        print("="*70)
        print("📊 FETCHING BROKER DATA")
        print("="*70 + "\n")
        
        for i, (_, row) in enumerate(buy_signals.iterrows(), 1):
            stock_code = row['StockCode']
            print(f"[{i}/{len(buy_signals)}] Fetching {stock_code}...")
            
            try:
                broker_data = fetch_broker_summary(stock_code, trade_date_str)
                buyers = extract_top_buyers(broker_data, MIN_TOP_BUYERS)
                sellers = extract_top_sellers(broker_data, MIN_TOP_SELLERS)
                
                if buyers or sellers:
                    print(f"  ✓ Buyers: {len(buyers)}, Sellers: {len(sellers)}")
                    msg = format_telegram_with_broker(row, buyers, sellers, trade_date_str)
                    results.append(msg)
                
                time.sleep(0.5)
            except Exception as e:
                print(f"  ✗ Error: {str(e)[:100]}")
                continue
    
    elif use_ai:
        # AI mode (with or without broker)
        try:
            from ai_stock_analyzer import analyze_stock_with_ai
        except ImportError:
            print("[ERROR] ai_stock_analyzer.py not found!")
            print("Please ensure ai_stock_analyzer.py is in the same directory")
            return
        
        print("="*70)
        print("🤖 PROCESSING WITH AI")
        print("="*70 + "\n")
        
        for i, (_, row) in enumerate(buy_signals.iterrows(), 1):
            stock_code = row['StockCode']
            print(f"[{i}/{len(buy_signals)}] Processing {stock_code}...")
            
            try:
                # Fetch broker if enabled
                buyers = []
                sellers = []
                if use_broker:
                    broker_data = fetch_broker_summary(stock_code, trade_date_str)
                    buyers = extract_top_buyers(broker_data, MIN_TOP_BUYERS)
                    sellers = extract_top_sellers(broker_data, MIN_TOP_SELLERS)
                    print(f"  ✓ Broker: {len(buyers)} buyers, {len(sellers)} sellers")
                
                # AI analysis
                print(f"  🤖 Analyzing with AI...")
                stock_dict = dict(row.to_dict())  # type: ignore[arg-type]
                broker_dict = {'buyers': buyers, 'sellers': sellers} if use_broker else None
                
                ai_result = analyze_stock_with_ai(stock_dict, broker_dict)  # type: ignore[arg-type]
                
                if ai_result['success']:
                    print(f"  ✓ AI analysis complete")
                else:
                    print(f"  ⚠️  AI failed: {ai_result.get('error', 'Unknown')[:50]}")
                
                msg = format_telegram_with_ai(row, buyers, sellers, trade_date_str, ai_result)
                results.append(msg)
                
                time.sleep(1)
            except Exception as e:
                print(f"  ✗ Error: {str(e)[:100]}")
                continue
    
    # Export to Excel
    export_path = os.path.join(OUTPUT_DIR, f"signals_{trade_date_str}.xlsx")
    try:
        buy_signals.to_excel(export_path, index=False)
        print(f"\n💾 Exported to: {export_path}")
    except Exception as e:
        print(f"\n⚠️  Export failed: {e}")
    
    # Send to Telegram
    if results:
        print("\n" + "="*70)
        print("📱 SENDING TO TELEGRAM")
        print("="*70 + "\n")
        
        mode_str: list[str] = []
        if use_broker:
            mode_str.append("Broker")
        if use_ai:
            mode_str.append("AI")
        mode_text = " + ".join(mode_str) if mode_str else "Basic"
        
        header = f"🎯 <b>DAY TRADING SIGNALS - IMPROVED ({mode_text})</b>\n" + \
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n" + \
                f"<b>Date:</b> {trade_date_str}\n" + \
                f"<b>Signals:</b> {len(results)}\n" + \
                f"<b>Min Score:</b> {min_score}\n" + \
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        send_telegram(header)
        time.sleep(1)
        
        for i, msg in enumerate(results, 1):
            formatted = f"<b>[{i}/{len(results)}]</b>\n\n{msg}"
            if send_telegram(formatted):
                print(f"  ✓ [{i}/{len(results)}] Sent")
            else:
                print(f"  ✗ [{i}/{len(results)}] Failed")
            time.sleep(1)
        
        print(f"\n✅ Sent {len(results)} signals to Telegram")
    
    print("\n" + "="*70)
    print("✅ COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()