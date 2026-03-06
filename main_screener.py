"""
Unified Day Trading Screener
Combines: Basic Screening + Broker Analysis + AI Insight
Usage:
    python main_screener.py                    # Basic only
    python main_screener.py --broker           # + Broker data
    python main_screener.py --ai               # + AI analysis  
    python main_screener.py --broker --ai      # Full features (recommended)
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
from typing import Any, Dict, List, cast

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

# NEW STRATEGY PARAMETERS
MOMENTUM_THRESHOLD = 1.5      # Raised from 1.0
REVERSAL_THRESHOLD = -2.5     # More conservative
MOMENTUM_WEIGHT = 0.7         # Favor momentum over reversal
REVERSAL_WEIGHT = 0.3

LOOKBACK_DAYS = 7
MIN_LIQUID_DAYS = 5           # Raised from 4
VOLUME_SURGE_THRESHOLD = 1.5  # Keep strict
CONSISTENCY_WEIGHT = 0.15

# NEW: Breakout strategy parameters
MIN_BREAKOUT_VOLUME = 1.5     # Min volume surge for breakout
MIN_CANDLE_STRENGTH = 0.7     # Min bullish candle strength (0-1)
MIN_PULLBACK_SUPPORT = 20     # Min % from low for pullback entry
MAX_PULLBACK_SUPPORT = 40     # Max % from low for pullback entry
ATR_MULTIPLIER_STOP = 2.0     # Stop loss = Entry - (ATR * 2)
ATR_MULTIPLIER_TARGET = 3.0   # Target = Entry + (ATR * 3)

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
    
    df["DayTradingScore"] = calculate_daytrading_score(df)  # type: ignore
    
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
    result["MarketPhase"] = result["StockCode"].map(phase).fillna("Unknown")  # type: ignore
    
    # NEW: Calculate ATR (Average True Range) from historical data
    def calc_atr(group: pd.DataFrame) -> float:
        if "TrueRange" in group.columns and len(group) >= 3:
            return group["TrueRange"].tail(7).mean()
        # Fallback: use average of (High - Low)
        if len(group) > 0:
            return (group["High"] - group["Low"]).mean()
        return 0.0
    
    atr_values = hist_grouped.apply(calc_atr)
    result["ATR7"] = result["StockCode"].map(atr_values).fillna(result["Close"] * 0.03)  # type: ignore
    
    # NEW: Calculate SMA7 (Simple Moving Average 7-day)
    def calc_sma7(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return group["Close"].mean() if len(group) > 0 else 0
        return group["Close"].tail(7).mean()
    
    sma_values = hist_grouped.apply(calc_sma7)
    result["SMA7"] = result["StockCode"].map(sma_values).fillna(result["Close"])  # type: ignore
    
    # NEW: Trend Strength (distance from SMA in %)
    result["TrendStrength"] = ((result["Close"] - result["SMA7"]) / result["SMA7"] * 100).fillna(0)  # type: ignore
    
    return result


def calculate_daytrading_score(df: pd.DataFrame) -> pd.Series:  # type: ignore
    """Calculate day trading score 0-100"""
    score = pd.Series(0.0, index=df.index)
    
    score += (df["Value"] / df["Value"].max() * 20).fillna(0)  # type: ignore
    
    volatility_normalized = 15 * (1 - (df["Volatility"] - 5).abs() / 10)
    volatility_score = volatility_normalized.where(
        (df["Volatility"] >= 1) & (df["Volatility"] <= 15), 0
    )
    score += volatility_score.clip(0, 15)  # type: ignore
    
    score += (df["Frequency"] / df["Frequency"].max() * 15).fillna(0)  # type: ignore
    
    spread_score = (1 - df["SpreadPct"] / MAX_SPREAD_PCT) * 10
    score += spread_score.clip(0, 10)  # type: ignore
    
    score += (df["AvgTradeSize"] / df["AvgTradeSize"].max() * 10).fillna(0)  # type: ignore
    
    momentum_score = pd.Series(0.0, index=df.index)
    
    momentum_mask = (df["ChangePct"] >= 1) & (df["ChangePct"] <= 10)
    momentum_score[momentum_mask] = (df["ChangePct"][momentum_mask] / 10 * 15).clip(0, 15)  # type: ignore
    
    extreme_mask = df["ChangePct"] > 10
    momentum_score[extreme_mask] = 10
    
    reversal_mask = (df["ChangePct"] >= -5) & (df["ChangePct"] <= -2)
    momentum_score[reversal_mask] = (5 + df["ChangePct"][reversal_mask]) / 3 * 10  # type: ignore
    
    extreme_reversal_mask = df["ChangePct"] < -5
    momentum_score[extreme_reversal_mask] = 5
    
    score += momentum_score  # type: ignore
    
    if "ConsistencyScore" in df.columns:
        score += (df["ConsistencyScore"] / 100 * 15).fillna(0)  # type: ignore
    
    return score


def calculate_entry_signals(df: pd.DataFrame) -> pd.DataFrame:
    """NEW STRATEGY: Breakout + ATR-based stops + Multiple confirmations"""
    result = df.copy()
    
    result["EntrySignal"] = "HOLD"
    result["SignalStrength"] = 0
    result["StopLoss"] = 0.0
    result["Target1"] = 0.0
    result["Target2"] = 0.0
    result["RiskReward"] = 0.0
    
    has_multiday = "PricePosition" in df.columns
    
    # If no multi-day data, add defaults
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
        support = row.get("Support7d", close)  # Add support
        change_pct = row["ChangePct"]
        position = row.get("PricePosition", 50)
        phase = row.get("MarketPhase", "Unknown")
        surge = row.get("VolumeSurge", 1)
        sma7 = row.get("SMA7", close)
        atr = row.get("ATR7", close * 0.03)  # Default 3% if no ATR
        candle_strength = row.get("CandleStrength", 0.5)
        trend_strength = row.get("TrendStrength", 0)
        
        # ======================
        # STRATEGY 1: BREAKOUT (RELAXED)
        # ======================
        # Relaxed conditions to generate more signals
        
        breakout_distance = (high - resistance) / resistance * 100 if resistance > 0 else 0
        is_breakout = breakout_distance >= -2.0  # Within 2% of resistance
        
        if (is_breakout and
            surge >= 1.2 and          # Lowered from 1.5
            candle_strength >= 0.6 and  # Lowered from 0.7
            change_pct >= 1.0 and     # Lowered from 1.5
            phase != "Trending Down"):  # Only avoid downtrend
            
            result.at[idx, "EntrySignal"] = "BUY - BREAKOUT"
            
            # Signal strength based on multiple factors
            strength = 0
            strength += min(30, change_pct * 15)  # Max 30 from momentum
            strength += min(30, (surge - 1) * 20)  # Max 30 from volume
            strength += min(20, candle_strength * 25)  # Max 20 from candle
            strength += min(20, abs(trend_strength) * 5)  # Max 20 from trend
            result.at[idx, "SignalStrength"] = int(min(100, strength))
            
            # ATR-based stops (2x ATR below entry)
            stop = close - (2 * atr)
            result.at[idx, "StopLoss"] = round(max(stop, close * 0.92))  # Min 8% stop
            
            # Targets based on ATR (1.5x and 3x ATR)
            result.at[idx, "Target1"] = round(close + (1.5 * atr))
            result.at[idx, "Target2"] = round(close + (3 * atr))
            
            risk = close - result.at[idx, "StopLoss"]
            reward = result.at[idx, "Target2"] - close
            rr = reward / risk if risk > 0 else 0
            result.at[idx, "RiskReward"] = round(rr, 2)
        
        # ======================
        # STRATEGY 2: PULLBACK (RELAXED)
        # ======================
        
        elif (15 <= position <= 50 and    # Wider range
              surge >= 1.1 and            # Lowered from 1.3
              candle_strength >= 0.55 and  # Lowered from 0.65
              -3.0 <= change_pct <= -0.3 and  # Wider range
              phase != "Trending Down"):  # Only avoid downtrend
            
            result.at[idx, "EntrySignal"] = "BUY - PULLBACK"
            
            # Conservative signal strength
            strength = 0
            strength += min(25, (40 - abs(change_pct)) * 2)  # Small pullback = better
            strength += min(25, (surge - 1) * 20)
            strength += min(25, candle_strength * 35)
            strength += min(25, trend_strength * 5)
            result.at[idx, "SignalStrength"] = int(min(100, strength))
            
            # ATR-based stops
            stop = close - (2.5 * atr)  # Wider stop for pullback
            result.at[idx, "StopLoss"] = round(stop)
            
            # Conservative targets
            result.at[idx, "Target1"] = round(close + (2 * atr))
            result.at[idx, "Target2"] = round(close + (3 * atr))
            
            risk = close - stop
            reward = result.at[idx, "Target2"] - close
            rr = reward / risk if risk > 0 else 0
            result.at[idx, "RiskReward"] = round(rr, 2)
        
        # ======================
        # STRATEGY 3: MOMENTUM (NEW - EASY)
        # ======================
        # Simple momentum play for strong moves
        
        elif (change_pct >= 2.0 and      # Strong move
              surge >= 1.3 and           # Good volume
              candle_strength >= 0.6 and  # Bullish candle
              position < 80 and          # Not overbought
              phase != "Trending Down"):
            
            result.at[idx, "EntrySignal"] = "BUY - MOMENTUM"
            
            strength = 0
            strength += min(30, change_pct * 10)
            strength += min(30, (surge - 1) * 20)
            strength += min(20, candle_strength * 30)
            strength += min(20, abs(trend_strength) * 5)
            result.at[idx, "SignalStrength"] = int(min(100, strength))
            
            # Fixed % stops for momentum
            stop = close * 0.96  # 4% stop
            result.at[idx, "StopLoss"] = round(stop)
            
            # Simple % targets
            result.at[idx, "Target1"] = round(close * 1.03)  # 3%
            result.at[idx, "Target2"] = round(close * 1.05)  # 5%
            
            risk = close - stop
            reward = (close * 1.05) - close
            rr = reward / risk if risk > 0 else 0
            result.at[idx, "RiskReward"] = round(rr, 2)
        
        # ======================        # STRATEGY 4: RISKY BOUNCE (New - for weak markets)
        # ======================
        # Dead cat bounce / weak bounce - RISKY!
        # Only trigger when better signals don't exist
        
        elif (change_pct >= 5.0 and       # Big move (desperation bounce)
              candle_strength >= 0.7 and   # Strong candle (conviction)
              position < 60):              # Price in lower range (room to move)
            
            result.at[idx, "EntrySignal"] = "BUY - RISKY BOUNCE"
            
            # Lower strength score (indicate higher risk) 
            strength = 0
            strength += min(25, change_pct * 8)
            strength += min(25, candle_strength * 30)
            strength += 20  # Base for big move
            result.at[idx, "SignalStrength"] = int(min(70, strength))  # Cap at 70 (high risk)
            
            # Tighter stops for risky trades
            stop = close * 0.95  # 5% stop (wider due to volatility)
            result.at[idx, "StopLoss"] = round(stop)
            
            # Conservative targets
            result.at[idx, "Target1"] = round(close * 1.025)  # 2.5%
            result.at[idx, "Target2"] = round(close * 1.04)   # 4%
            
            risk = close - stop
            reward = (close * 1.04) - close
            rr = reward / risk if risk > 0 else 0
            result.at[idx, "RiskReward"] = round(rr, 2)
        
        # ======================        # AVOID CONDITIONS
        # ======================
        elif position > 75 or change_pct > 7:
            result.at[idx, "EntrySignal"] = "AVOID - OVERBOUGHT"
            result.at[idx, "SignalStrength"] = 0
        
        elif close < sma7 or phase == "Trending Down":
            result.at[idx, "EntrySignal"] = "AVOID - DOWNTREND"
            result.at[idx, "SignalStrength"] = 0
        
        elif surge < 1.2:
            result.at[idx, "EntrySignal"] = "AVOID - LOW VOLUME"
            result.at[idx, "SignalStrength"] = 0
    
    return result


def screen(df: pd.DataFrame) -> pd.DataFrame:
    """Filter stocks for day trading"""
    
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
    
    filtered["Category"] = "Neutral"  # type: ignore
    filtered.loc[filtered["ChangePct"] >= MOMENTUM_THRESHOLD, "Category"] = "Momentum"  # type: ignore
    filtered.loc[filtered["ChangePct"] <= REVERSAL_THRESHOLD, "Category"] = "Reversal"  # type: ignore
    
    if has_multiday:
        filtered["VolumeSurgeFlag"] = filtered["VolumeSurge"] >= VOLUME_SURGE_THRESHOLD  # type: ignore
        filtered["BreakoutFlag"] = (
            (filtered["VolumeSurge"] >= VOLUME_SURGE_THRESHOLD) &
            (filtered["ChangePct"] > 2) &
            (filtered["Trend30d"] == "Up")
        )  # type: ignore
    
    return filtered.sort_values(by="DayTradingScore", ascending=False)


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


def fetch_broker_summary(symbol: str, trade_date: str) -> Dict[str, Any]:
    """Fetch broker summary with fallback API keys"""
    url = f"{GOAPI_BASE}/stock/idx/{symbol}/broker_summary"
    params = {"date": trade_date, "investor": "ALL"}
    
    max_attempts = len(_goapi_keys) if _goapi_keys else 1
    attempt = 0
    
    while attempt < max_attempts:
        try:
            api_key = get_current_api_key()
            headers = {"X-API-KEY": api_key}
            raw = http_get_json(url, headers, params)
            
            if raw.get("status") != "success":
                raise RuntimeError(f"API returned non-success status: {raw}")
            
            return raw
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                rotate_api_key()
                attempt += 1
                wait_time = min(2 ** attempt, 30)
                print(f"    Rate limit, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            elif e.response.status_code == 401:
                raise RuntimeError(f"Unauthorized: Invalid API key")
            else:
                raise RuntimeError(f"HTTP {e.response.status_code}: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error fetching {symbol}: {str(e)}")
    
    raise RuntimeError(f"All API keys exceeded rate limit")


def extract_brokers_by_side(broker_data: Dict[str, Any], side: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Extract top brokers by side"""
    brokers: List[Dict[str, Any]] = []
    
    if not broker_data or not broker_data.get("data") or not broker_data.get("data", {}).get("results"):
        return []
    
    for item in broker_data["data"]["results"]:
        try:
            if item.get("side") == side:
                broker: Dict[str, Any] = {
                    "broker_code": str(item.get("code", "")),
                    "broker_name": str(item.get("broker", {}).get("name", "Unknown")),
                    "lot": int(item.get("lot", 0)),
                    "value": int(item.get("value", 0)),
                    "avg": float(item.get("avg", 0)),
                }
                brokers.append(broker)
        except (KeyError, TypeError, ValueError):
            continue
    
    brokers.sort(key=lambda x: int(x["value"]), reverse=True)
    return brokers[:limit]


def extract_top_buyers(broker_data: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    return extract_brokers_by_side(broker_data, "BUY", limit)


def extract_top_sellers(broker_data: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    return extract_brokers_by_side(broker_data, "SELL", limit)


def rupiah(n: int) -> str:
    return f"{n:,}".replace(",", ".")


# =========================
# FORMATTING FUNCTIONS
# =========================

def format_telegram_basic(df: pd.DataFrame) -> str:
    """Format basic screening results"""
    msg = "📊 <b>DAY TRADING ALERT</b>\n"
    msg += "=" * 35 + "\n\n"
    
    if len(df) == 0:
        msg += "❌ No stocks passed screening\n"
        return msg
    
    has_signals = "EntrySignal" in df.columns
    
    if has_signals:
        buy_stocks = df[df["EntrySignal"].str.contains("BUY", na=False)].sort_values(by="SignalStrength", ascending=False)
        
        if len(buy_stocks) > 0:
            msg += "🎯 <b>BUY SIGNALS</b>\n"
            msg += "─" * 35 + "\n"
            
            for idx, row in enumerate(buy_stocks.head(5).itertuples(), 1):
                stock_code = html.escape(str(row.StockCode))
                signal = html.escape(str(row.EntrySignal))  # type: ignore
                msg += f"{idx}. <b>{stock_code}</b> - {signal}\n"
                msg += f"   Price: Rp {row.Close:,.0f} ({row.ChangePct:+.2f}&#37;)\n"  # type: ignore
                msg += f"   🎯 T1: {row.Target1:,.0f} | T2: {row.Target2:,.0f}\n"  # type: ignore
                msg += f"   🛑 SL: {row.StopLoss:,.0f} | R:R 1:{row.RiskReward:.1f}\n\n"  # type: ignore
    
    msg += f"📈 Total: {len(df)} stocks\n"
    msg += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    
    return msg


def format_telegram_with_broker(row: Any, buyers: List[Dict[str, Any]], 
                                sellers: List[Dict[str, Any]], trade_date: str) -> str:
    """Format with broker data"""
    stock_code = html.escape(str(row['StockCode']))
    stock_name = html.escape(str(row['StockName'][:20]))
    
    msg = "=" * 30 + "\n"
    msg += f"🎯 <b>{stock_code}</b> - {stock_name}\n"
    msg += "=" * 30 + "\n\n"
    
    signal = row.get('EntrySignal', 'N/A')
    strength = row.get('SignalStrength', 0)
    
    msg += f"📡 <b>SIGNAL:</b> {signal} ({strength}/100)\n\n"
    
    price = row['Close']
    msg += f"💰 <b>Price:</b> Rp {price:,.0f} ({row['ChangePct']:+.2f}%)\n"
    msg += f"🛑 <b>Stop Loss:</b> {row['StopLoss']:,.0f}\n"
    msg += f"🎯 <b>Target 1:</b> {row['Target1']:,.0f} | T2: {row['Target2']:,.0f}\n"
    msg += f"⚖️ <b>R/R:</b> 1:{row['RiskReward']:.2f}\n\n"
    
    if 'VolumeSurge' in row.index:
        msg += f"📊 Vol Surge: {row['VolumeSurge']:.2f}x | Phase: {row['MarketPhase']}\n\n"
    
    total_buy_value = sum(b["value"] for b in buyers) if buyers else 0
    total_sell_value = sum(s["value"] for s in sellers) if sellers else 0
    
    if buyers or sellers:
        msg += "─" * 30 + "\n"
        msg += "🏦 <b>BROKER ACTIVITY</b>\n"
        msg += "─" * 30 + "\n"
        
        if buyers:
            msg += f"🟢 <b>Top Buyer:</b> {buyers[0]['broker_code']}\n"
            msg += f"   Val: Rp {rupiah(int(buyers[0]['value']))} ({buyers[0]['lot']:,.0f} lot)\n"
        
        if sellers:
            msg += f"🔴 <b>Top Seller:</b> {sellers[0]['broker_code']}\n"
            msg += f"   Val: Rp {rupiah(int(sellers[0]['value']))} ({sellers[0]['lot']:,.0f} lot)\n"
        
        if total_buy_value > 0 and total_sell_value > 0:
            ratio = total_buy_value / total_sell_value
            sentiment = "🟢 BULLISH" if ratio > 1.2 else "🔴 BEARISH" if ratio < 0.8 else "⚪ NEUTRAL"
            msg += f"\n⚖️ B/S Ratio: {ratio:.2f}x {sentiment}\n"
    
    msg += "\n" + "=" * 30
    return msg


def format_telegram_with_ai(row: Any, buyers: List[Dict[str, Any]], 
                            sellers: List[Dict[str, Any]], 
                            trade_date: str, ai_result: Dict[str, Any]) -> str:
    """Format with AI analysis"""
    stock_code = html.escape(str(row['StockCode']))
    stock_name = html.escape(str(row['StockName'][:20]))
    
    msg = "━" * 35 + "\n"
    msg += f"🎯 <b>{stock_code}</b> - {stock_name}\n"
    msg += "━" * 35 + "\n\n"
    
    signal = row.get('EntrySignal', 'N/A')
    strength = row.get('SignalStrength', 0)
    msg += f"📡 <b>SIGNAL:</b> {signal} ({strength}/100)\n\n"
    
    price = row['Close']
    msg += f"💰 <b>Price:</b> Rp {price:,.0f} ({row['ChangePct']:+.2f}%)\n"
    msg += f"🛑 <b>Stop Loss:</b> {row['StopLoss']:,.0f}\n"
    msg += f"🎯 <b>Target 1:</b> {row['Target1']:,.0f} | T2: {row['Target2']:,.0f}\n"
    msg += f"⚖️ <b>R/R:</b> 1:{row['RiskReward']:.2f}\n\n"
    
    if 'VolumeSurge' in row.index:
        msg += f"📊 Vol Surge: {row['VolumeSurge']:.2f}x | Phase: {row['MarketPhase']}\n\n"
    
    if buyers or sellers:
        total_buy = sum(b['value'] for b in buyers) if buyers else 0
        total_sell = sum(s['value'] for s in sellers) if sellers else 0
        
        msg += "─" * 35 + "\n"
        msg += "🏦 <b>BROKER ACTIVITY</b>\n"
        msg += "─" * 35 + "\n"
        
        if buyers:
            msg += f"🟢 <b>Top Buyer:</b> {buyers[0]['broker_code']}\n"
            msg += f"   Val: Rp {rupiah(int(buyers[0]['value']))} ({buyers[0]['lot']:,.0f} lot)\n"
        
        if sellers:
            msg += f"🔴 <b>Top Seller:</b> {sellers[0]['broker_code']}\n"
            msg += f"   Val: Rp {rupiah(int(sellers[0]['value']))} ({sellers[0]['lot']:,.0f} lot)\n"
        
        if total_buy > 0 and total_sell > 0:
            ratio = total_buy / total_sell
            sentiment = "🟢 BULLISH" if ratio > 1.2 else "🔴 BEARISH" if ratio < 0.8 else "⚪ NEUTRAL"
            msg += f"\n⚖️ B/S Ratio: {ratio:.2f}x {sentiment}\n"
        
        msg += "\n"
    
    if ai_result and ai_result.get('success'):
        msg += "━" * 35 + "\n"
        msg += "🤖 <b>AI INSIGHT</b>\n"
        msg += "━" * 35 + "\n"
        msg += ai_result['analysis']
        msg += "\n"
    
    msg += "━" * 35
    return msg


def send_telegram(message: str) -> bool:
    """Send message to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARNING] Telegram not configured")
        return False
    
    MAX_LENGTH = 4096
    if len(message) > MAX_LENGTH:
        message = message[:MAX_LENGTH-50] + "\n\n... (truncated)"
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    try:
        response = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            },
            timeout=20
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"[ERROR] Telegram failed: {e}")
        return False


# =========================
# MAIN FUNCTION
# =========================

def main():
    parser = argparse.ArgumentParser(description='Unified Day Trading Screener')
    parser.add_argument('--broker', action='store_true', help='Include broker analysis')
    parser.add_argument('--ai', action='store_true', help='Include AI analysis')
    args = parser.parse_args()
    
    use_broker = args.broker
    use_ai = args.ai
    
    # Auto-detect features
    if not use_broker and not use_ai:
        if _goapi_keys:
            print("[i] GOAPI keys detected. Use --broker to enable broker analysis")
        if GEMINI_API_KEY:
            print("[i] Gemini API key detected. Use --ai to enable AI analysis")
    
    print("="*70)
    print("🚀 UNIFIED DAY TRADING SCREENER")
    print("="*70)
    print(f"📊 Mode: {'Basic' if not use_broker and not use_ai else ''}")
    if use_broker:
        print("   ✅ Broker Analysis: ENABLED")
    if use_ai:
        print("   ✅ AI Analysis: ENABLED")
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
            df_today["DayTradingScore"] = calculate_daytrading_score(df_today)
            print("✅ Multi-day analysis complete\n")
    
    # Screen
    print("🔍 Running screener...")
    df_screened = screen(df_today)
    print(f"✅ {len(df_screened)} stocks passed screening\n")
    
    if len(df_screened) == 0:
        print("No stocks passed screening")
        return
    
    # Generate signals
    print("🎯 Generating entry signals...")
    df_signals = calculate_entry_signals(df_screened)
    buy_signals = df_signals[df_signals["EntrySignal"].str.contains("BUY", na=False)]
    print(f"✅ Found {len(buy_signals)} BUY signals\n")
    
    if len(buy_signals) == 0:
        print("No BUY signals generated")
        return
    
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
        
        header = f"🎯 <b>DAY TRADING SIGNALS ({mode_text})</b>\n" + \
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n" + \
                f"<b>Date:</b> {trade_date_str}\n" + \
                f"<b>Signals:</b> {len(results)}\n" + \
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
