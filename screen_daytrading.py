import os
import json
import glob
import time
import argparse
import pandas as pd
import requests
from datetime import datetime, date
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, cast
import html

# Load environment variables
load_dotenv()

DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Telegram Config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# GOAPI Config (untuk broker data)
_goapi_keys: List[str] = []
if os.path.exists("key.txt"):
    try:
        with open("key.txt", "r") as _f:
            for _line in _f.readlines()[1:]:  # skip header "GOAPI KEYS:"
                _k = _line.strip()
                if _k and not _k.startswith("GOAPI") and not _k.startswith("#"):
                    _goapi_keys.append(_k)
    except Exception:
        pass
if not _goapi_keys:
    for _i in range(1, 10):
        _k = os.getenv(f"GOAPI_KEY_{_i}", "").strip()
        if _k:
            _goapi_keys.append(_k)
    if not _goapi_keys:
        _k = os.getenv("GOAPI_KEY", "").strip()
        if _k:
            _goapi_keys.append(_k)
_current_key_index = 0
GOAPI_BASE = "https://api.goapi.io"

# =========================
# PARAMETER DAY TRADING - HYBRID STRATEGY
# =========================
# Filter Likuiditas (PENTING untuk day trading)
MIN_VALUE = 2_000_000_000      # Turunkan dari 3B ke 2B (lebih banyak kandidat)
MIN_VOLUME = 500_000           # Turunkan dari 1M ke 500K lot
MIN_FREQUENCY = 200            # Turunkan dari 300 ke 200x transaksi
MAX_SPREAD_PCT = 2.0           # Naikkan dari 1.5% ke 2% (lebih toleran)

# Filter Harga & Volatilitas
MIN_PRICE = 100                # Rp 100 - hindari penny stocks
MAX_PRICE = 50_000             # Rp 50K - untuk affordability
MIN_VOLATILITY = 0.8           # Turunkan dari 1.0% ke 0.8%
MAX_VOLATILITY = 20.0          # Naikkan dari 15% ke 20% (terima lebih volatile)

# Filter Aktivitas
MIN_VOLUME_VS_SHARES = 0.000005 # 0.0005% - volume vs shares
AVG_TRADE_SIZE_MIN = 3_000_000  # 3 juta - institusi/whale activity

# Filter Momentum & Big Money (NEW)
MIN_NET_FOREIGN = -1_000_000   # Net foreign (sedikit selling ok)
MIN_NONREG_RATIO = 0.01        # Min 1% non-regular (ada bandar)
MIN_VALUE_RANK_PERCENTILE = 0.5  # Top 50% by value

# Hybrid Strategy Split (NEW)
MOMENTUM_THRESHOLD = 1.0       # ChangePct >= 1% = momentum play
REVERSAL_THRESHOLD = -2.0      # ChangePct <= -2% = reversal play
MOMENTUM_WEIGHT = 0.6          # 60% portfolio momentum
REVERSAL_WEIGHT = 0.4          # 40% portfolio reversal

# Multi-Day Analysis (NEW)
LOOKBACK_DAYS = 7              # Analyze 7 hari terakhir
MIN_LIQUID_DAYS = 4            # Minimal 4/7 hari liquid (57%)
VOLUME_SURGE_THRESHOLD = 1.5   # Volume today > 150% avg = surge
CONSISTENCY_WEIGHT = 0.15      # 15% weight untuk consistency

# Broker Settings
MIN_TOP_BROKERS = 5            # Ambil 5 broker teratas
BROKER_CALL_SLEEP = 1.5        # Jeda antar request (detik)
BROKER_BIAS_THRESHOLD = 0.6    # top-buyer/total >= 60% = NET BUY

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
    # numeric safety
    num_cols = [
        "Previous","OpenPrice","FirstTrade","Close","Volume","Value","Frequency","High","Low",
        "ForeignBuy","ForeignSell","NonRegularVolume",
        "TradebleShares","Bid","Offer"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)  # type: ignore
    
    # Big Money Flow (NEW)
    df["NetForeign"] = df["ForeignBuy"] - df["ForeignSell"]  # type: ignore
    df["NonRegRatio"] = (df["NonRegularVolume"] / df["Volume"].replace(0, float('nan'))).fillna(0)  # type: ignore

    # Fitur Day Trading
    df["ChangePct"] = ((df["Close"] - df["Previous"]) / df["Previous"] * 100).replace([float("inf"), -float("inf")], 0)  # type: ignore
    
    # Volatilitas (High-Low range)
    df["Volatility"] = ((df["High"] - df["Low"]) / df["Previous"] * 100).replace([float("inf"), -float("inf")], 0)  # type: ignore
    
    # Spread Analysis (liquidity)
    df["SpreadPct"] = df.apply(  # type: ignore
        lambda r: abs(r["Offer"] - r["Bid"]) / r["Close"] * 100 if r["Close"] > 0 else 0,  # type: ignore
        axis=1
    )
    
    # Average Trade Size
    df["AvgTradeSize"] = df.apply(  # type: ignore
        lambda r: r["Value"] / r["Frequency"] if r["Frequency"] > 0 else 0,  # type: ignore
        axis=1
    )
    
    # Volume vs Shares Outstanding
    df["VolumeVsShares"] = df.apply(  # type: ignore
        lambda r: r["Volume"] / r["TradebleShares"] if r["TradebleShares"] > 0 else 0,  # type: ignore
        axis=1
    )
    
    # Value Percentile Rank
    df["ValueRank"] = df["Value"].rank(pct=True)  # type: ignore
    
    # Day Trading Score (0-100) - will be recalculated after multi-day if available
    df["DayTradingScore"] = calculate_daytrading_score(df)  # type: ignore
    
    return df


def calculate_multi_day_metrics(df_today: pd.DataFrame, df_historical: pd.DataFrame) -> pd.DataFrame:
    """Calculate 30-day comparison metrics"""
    result = df_today.copy()
    
    # Group historical by StockCode
    hist_grouped = df_historical.groupby("StockCode")
    
    # Calculate 30-day averages
    avg_volume = hist_grouped["Volume"].mean()
    avg_value = hist_grouped["Value"].mean()
    avg_frequency = hist_grouped["Frequency"].mean()
    
    # Calculate consistency (berapa hari liquid?)
    liquid_days = hist_grouped.apply(
        lambda g: ((g["Value"] >= MIN_VALUE * 0.5) & (g["Volume"] >= MIN_VOLUME * 0.5)).sum()
    )
    
    # Calculate trend (price direction)
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
    
    # Merge back to today's data
    result["AvgVolume30d"] = result["StockCode"].map(avg_volume).fillna(0)  # type: ignore
    result["AvgValue30d"] = result["StockCode"].map(avg_value).fillna(0)  # type: ignore
    result["AvgFreq30d"] = result["StockCode"].map(avg_frequency).fillna(0)  # type: ignore
    result["LiquidDays30d"] = result["StockCode"].map(liquid_days).fillna(0)  # type: ignore
    result["Trend30d"] = result["StockCode"].map(trend).fillna("Unknown")  # type: ignore
    
    # Calculate surge ratios
    result["VolumeSurge"] = (result["Volume"] / result["AvgVolume30d"].replace(0, float('nan'))).fillna(1)  # type: ignore
    result["ValueSurge"] = (result["Value"] / result["AvgValue30d"].replace(0, float('nan'))).fillna(1)  # type: ignore
    
    # Consistency Score (0-100) - FIXED: divide by LOOKBACK_DAYS not stock count!
    result["ConsistencyScore"] = (result["LiquidDays30d"] / LOOKBACK_DAYS * 100).clip(0, 100)  # type: ignore
    
    # Support/Resistance Levels (NEW FOR DAYTRADING)
    support_7d = hist_grouped["Low"].min()
    resistance_7d = hist_grouped["High"].max()
    
    result["Support7d"] = result["StockCode"].map(support_7d).fillna(0)  # type: ignore
    result["Resistance7d"] = result["StockCode"].map(resistance_7d).fillna(0)  # type: ignore
    
    # Distance from Support/Resistance (for entry timing)
    result["DistFromSupport"] = ((result["Close"] - result["Support7d"]) / result["Support7d"] * 100).fillna(0)  # type: ignore
    result["DistFromResistance"] = ((result["Resistance7d"] - result["Close"]) / result["Close"] * 100).fillna(0)  # type: ignore
    
    # Price Position (0-100): dimana posisi harga di range 7 hari?
    price_range = result["Resistance7d"] - result["Support7d"]
    result["PricePosition"] = ((result["Close"] - result["Support7d"]) / price_range.replace(0, float('nan')) * 100).fillna(50).clip(0, 100)  # type: ignore
    
    # Market Phase Detection
    def get_phase(group: pd.DataFrame) -> str:
        if len(group) < 5:
            return "Unknown"
        highs = group["High"].values
        lows = group["Low"].values
        # Trending Up: Higher highs and higher lows
        if highs[-1] > highs[0] and lows[-1] > lows[0]:
            return "Trending Up"
        # Trending Down: Lower highs and lower lows
        elif highs[-1] < highs[0] and lows[-1] < lows[0]:
            return "Trending Down"
        # Sideways: range bound
        return "Sideways"
    
    phase = hist_grouped.apply(get_phase)
    result["MarketPhase"] = result["StockCode"].map(phase).fillna("Unknown")  # type: ignore

    # EMA5 / EMA10 dari close historis (untuk MA cross detection)
    def calc_ema(group: pd.DataFrame, span: int) -> float:
        closes = group["Close"].values
        if len(closes) < 2:
            return float(closes[-1]) if len(closes) == 1 else 0.0
        s = pd.Series(closes, dtype=float)
        return float(s.ewm(span=span, adjust=False).mean().iloc[-1])

    ema5  = hist_grouped.apply(lambda g: calc_ema(g, 5))
    ema10 = hist_grouped.apply(lambda g: calc_ema(g, 10))
    prev_ema5  = hist_grouped.apply(lambda g: calc_ema(g.iloc[:-1] if len(g) > 1 else g, 5))
    prev_ema10 = hist_grouped.apply(lambda g: calc_ema(g.iloc[:-1] if len(g) > 1 else g, 10))

    result["EMA5_hist"]      = result["StockCode"].map(ema5).fillna(0)   # type: ignore
    result["EMA10_hist"]     = result["StockCode"].map(ema10).fillna(0)  # type: ignore
    result["PrevEMA5_hist"]  = result["StockCode"].map(prev_ema5).fillna(0)  # type: ignore
    result["PrevEMA10_hist"] = result["StockCode"].map(prev_ema10).fillna(0)  # type: ignore

    return result


# =========================
# TECHNICAL PATTERN DETECTION
# =========================

def _get_effective_open(row: pd.Series) -> float:  # type: ignore
    open_price = float(row.get("OpenPrice", 0) or 0)
    if open_price > 0:
        return open_price
    return float(row.get("Previous", 0) or 0)

def _candle_body(o: float, c: float) -> float:
    return abs(c - o)

def _upper_shadow(o: float, h: float, c: float) -> float:
    return h - max(o, c)

def _lower_shadow(o: float, l: float, c: float) -> float:
    return min(o, c) - l

def detect_candlestick_patterns(row: pd.Series) -> List[str]:  # type: ignore
    """
    Deteksi pola candlestick single-bar dari OHLC harian.
    Mengembalikan list nama pola yang terdeteksi.
    """
    patterns: List[str] = []

    o = _get_effective_open(row)
    h = float(row.get("High",     0) or 0)
    l = float(row.get("Low",      0) or 0)
    c = float(row.get("Close",    0) or 0)

    if o <= 0 or h <= 0 or l <= 0 or c <= 0:
        return patterns

    total_range = h - l
    if total_range <= 0:
        return patterns

    body   = _candle_body(o, c)
    upper  = _upper_shadow(o, h, c)
    lower  = _lower_shadow(o, l, c)
    body_ratio  = body / total_range
    upper_ratio = upper / total_range
    lower_ratio = lower / total_range
    is_bull = c >= o

    # --- Doji (ketidakpastian / potensi reversal) ---
    if body_ratio <= 0.10:
        patterns.append("Doji")

    # --- Hammer (bullish reversal dari bawah) ---
    # Body kecil di atas, ekor bawah panjang (≥ 2× body), ekor atas pendek
    if (lower >= body * 2.0) and (upper <= body * 0.5) and body_ratio <= 0.35:
        patterns.append("Hammer")

    # --- Inverted Hammer / Shooting Star ---
    # Body kecil di bawah, ekor atas panjang
    if (upper >= body * 2.0) and (lower <= body * 0.5) and body_ratio <= 0.35:
        if is_bull:
            patterns.append("Inverted Hammer")
        else:
            patterns.append("Shooting Star")

    # --- Marubozu (tren kuat, nyaris tanpa shadow) ---
    if body_ratio >= 0.90:
        if is_bull:
            patterns.append("Bullish Marubozu")
        else:
            patterns.append("Bearish Marubozu")

    # --- Bullish Spinning Top (konsolidasi, range besar tapi body kecil) ---
    if 0.10 < body_ratio <= 0.30 and upper_ratio >= 0.25 and lower_ratio >= 0.25:
        patterns.append("Spinning Top")

    # --- Pin Bar (long wick satu sisi, sinyal reversal tajam) ---
    if lower_ratio >= 0.60 and body_ratio <= 0.20:
        patterns.append("Bullish Pin Bar")
    elif upper_ratio >= 0.60 and body_ratio <= 0.20:
        patterns.append("Bearish Pin Bar")

    # --- Long Upper Wick (tekanan jual di atas, waspada) ---
    if upper_ratio >= 0.50 and is_bull and body_ratio <= 0.30:
        patterns.append("Long Upper Wick")

    return patterns


def detect_price_patterns(row: pd.Series, hist_rows: Optional[pd.DataFrame]) -> List[str]:  # type: ignore
    """
    Deteksi pola multi-bar (butuh data historis).
    hist_rows: DataFrame baris historis untuk saham ini (sorted ascending by date, terakhir = kemarin).
    """
    patterns: List[str] = []

    if hist_rows is None or len(hist_rows) < 2:
        return patterns

    prev = hist_rows.iloc[-1]  # bar kemarin
    p_o  = _get_effective_open(prev)
    p_h  = float(prev.get("High",     0) or 0)
    p_l  = float(prev.get("Low",      0) or 0)
    p_c  = float(prev.get("Close",    0) or 0)

    c_o  = _get_effective_open(row)
    c_h  = float(row.get("High",     0) or 0)
    c_l  = float(row.get("Low",      0) or 0)
    c_c  = float(row.get("Close",    0) or 0)

    if p_h <= 0 or p_l <= 0 or p_c <= 0 or c_h <= 0 or c_l <= 0 or c_c <= 0:
        return patterns

    p_body = _candle_body(p_o, p_c)
    c_body = _candle_body(c_o, c_c)

    # --- Bullish Engulfing ---
    if (p_c < p_o and c_c > c_o           # kemarin merah, hari ini hijau
            and c_o <= p_c                 # open hari ini ≤ close kemarin
            and c_c >= p_o                 # close hari ini ≥ open kemarin
            and c_body > p_body * 1.1):    # badan hari ini lebih besar
        patterns.append("Bullish Engulfing")

    # --- Bearish Engulfing ---
    if (p_c > p_o and c_c < c_o
            and c_o >= p_c
            and c_c <= p_o
            and c_body > p_body * 1.1):
        patterns.append("Bearish Engulfing")

    # --- Inside Bar (konsolidasi sebelum breakout) ---
    if c_h <= p_h and c_l >= p_l:
        patterns.append("Inside Bar")

    # --- Outside Bar (volatilitas meledak) ---
    if c_h > p_h and c_l < p_l:
        patterns.append("Outside Bar")

    # --- Tweezer Bottom (support kuat, bullish reversal) ---
    if abs(p_l - c_l) / max(p_l, 1) <= 0.005 and p_c < p_o and c_c > c_o:
        patterns.append("Tweezer Bottom")

    # --- Tweezer Top (resistance kuat, bearish) ---
    if abs(p_h - c_h) / max(p_h, 1) <= 0.005 and p_c > p_o and c_c < c_o:
        patterns.append("Tweezer Top")

    return patterns


def detect_volume_patterns(row: pd.Series, avg_volume: float) -> List[str]:  # type: ignore
    """Deteksi pola volume."""
    patterns: List[str] = []
    volume = float(row.get("Volume", 0) or 0)
    if avg_volume <= 0:
        return patterns
    ratio = volume / avg_volume

    if ratio >= 3.0:
        patterns.append("Volume Climax")
    elif ratio >= 2.0:
        patterns.append("Volume Surge 2x")
    elif ratio <= 0.3:
        patterns.append("Volume Dry-up")

    # Value vs Volume mismatch → institutional block trade
    value = float(row.get("Value", 0) or 0)
    avg_val = float(row.get("AvgValue30d", 0) or 0)
    if avg_val > 0 and value / avg_val >= 2.5 and ratio < 2.0:
        patterns.append("Block Trade")

    return patterns


def detect_ma_patterns(row: pd.Series) -> List[str]:  # type: ignore
    """Deteksi pola Moving Average dari EMA5/EMA10 historis."""
    patterns: List[str] = []

    close     = float(row.get("Close",         0) or 0)
    ema5      = float(row.get("EMA5_hist",      0) or 0)
    ema10     = float(row.get("EMA10_hist",     0) or 0)
    prev_ema5 = float(row.get("PrevEMA5_hist",  0) or 0)
    prev_ema10= float(row.get("PrevEMA10_hist", 0) or 0)

    if ema5 <= 0 or ema10 <= 0:
        return patterns

    # Golden Cross: EMA5 baru saja naik melewati EMA10
    if prev_ema5 <= prev_ema10 and ema5 > ema10:
        patterns.append("Golden Cross (EMA5/10)")

    # Death Cross
    if prev_ema5 >= prev_ema10 and ema5 < ema10:
        patterns.append("Death Cross (EMA5/10)")

    # Harga ≥ EMA5 dan EMA5 ≥ EMA10 → tren naik terkonfirmasi
    if close >= ema5 >= ema10:
        patterns.append("Above MA Stack")

    # Price menembus EMA5 ke atas (dari bawah)
    prev_close = float(row.get("Previous", 0) or 0)
    if prev_close > 0 and prev_close < ema5 and close >= ema5:
        patterns.append("EMA5 Breakout")

    return patterns


def detect_all_patterns(
    row: pd.Series,  # type: ignore
    hist_rows: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Jalankan semua detektor dan kembalikan dict lengkap:
    {
        "patterns": List[str],        # semua pola terdeteksi
        "bullish_patterns": List[str],
        "bearish_patterns": List[str],
        "pattern_score": float,       # 0-30 bonus score
        "pattern_str": str,           # string ringkas untuk display
    }
    """
    BEARISH_TAGS = {"Doji", "Shooting Star", "Bearish Marubozu", "Bearish Pin Bar",
                    "Long Upper Wick", "Bearish Engulfing", "Tweezer Top", "Death Cross (EMA5/10)"}
    NEUTRAL_TAGS = {"Inside Bar", "Outside Bar", "Spinning Top", "Volume Dry-up"}

    candle_pats  = detect_candlestick_patterns(row)
    volume_pats  = detect_volume_patterns(
        row,
        avg_volume=float(row.get("AvgVolume30d", 0) or 0),
    )
    ma_pats      = detect_ma_patterns(row)
    price_pats   = detect_price_patterns(row, hist_rows)

    all_pats = candle_pats + volume_pats + ma_pats + price_pats

    bullish = [p for p in all_pats if p not in BEARISH_TAGS and p not in NEUTRAL_TAGS]
    bearish = [p for p in all_pats if p in BEARISH_TAGS]

    # Score: setiap pola bullish +5, bearish -3, max 30
    score = max(0.0, min(30.0, len(bullish) * 5.0 - len(bearish) * 3.0))

    pattern_str = " + ".join(all_pats) if all_pats else ""

    return {
        "patterns":        all_pats,
        "bullish_patterns": bullish,
        "bearish_patterns": bearish,
        "pattern_score":   score,
        "pattern_str":     pattern_str,
    }


def calculate_daytrading_score(df: pd.DataFrame) -> pd.Series:  # type: ignore
    """Hitung score day trading 0-100"""
    score = pd.Series(0.0, index=df.index)

    # 1. Likuiditas gabungan — Value + Volume + Frequency (25 pts)
    val_norm  = (df["Value"]     / df["Value"].max()).fillna(0)  # type: ignore
    vol_norm  = (df["Volume"]    / df["Volume"].max()).fillna(0)  # type: ignore
    freq_norm = (df["Frequency"] / df["Frequency"].max()).fillna(0)  # type: ignore
    score += (val_norm * 0.5 + vol_norm * 0.25 + freq_norm * 0.25) * 25  # type: ignore

    # 2. Volatilitas optimal untuk scalping (15 pts) — ideal 2-8%
    vs = pd.Series(0.0, index=df.index)
    in_range  = (df["Volatility"] >= 2) & (df["Volatility"] <= 8)
    low_range = (df["Volatility"] >= 1) & (df["Volatility"]  < 2)
    hi_range  = (df["Volatility"]  > 8) & (df["Volatility"] <= 12)
    vs[in_range]  = 15.0
    vs[low_range] = ((df["Volatility"][low_range] - 1) * 10).clip(0, 10)  # type: ignore
    vs[hi_range]  = ((1 - (df["Volatility"][hi_range] - 8) / 4) * 15).clip(0, 15)  # type: ignore
    score += vs  # type: ignore

    # 3. Big Money / Bandar Activity (20 pts)
    nonreg_score = (df["NonRegRatio"] / 0.1 * 15).clip(0, 15)  # type: ignore
    trade_score  = (df["AvgTradeSize"] / df["AvgTradeSize"].max() * 5).fillna(0)  # type: ignore
    score += nonreg_score + trade_score  # type: ignore

    # 4. Spread sempit = mudah entry/exit (10 pts)
    score += ((1 - df["SpreadPct"] / MAX_SPREAD_PCT) * 10).clip(0, 10)  # type: ignore

    # 5. Kualitas Momentum (15 pts)
    ms = pd.Series(0.0, index=df.index)
    up_mask       = (df["ChangePct"] >= 0) & (df["ChangePct"] <= 8)
    overbought    = df["ChangePct"] > 8
    reversal_mask = (df["ChangePct"] >= -4) & (df["ChangePct"] < 0)
    ms[up_mask]       = (df["ChangePct"][up_mask] / 8 * 15).clip(0, 15)  # type: ignore
    ms[overbought]    = 8.0
    ms[reversal_mask] = ((4 + df["ChangePct"][reversal_mask]) / 4 * 8).clip(0, 8)  # type: ignore
    score += ms  # type: ignore

    # 6. Volume Surge Bonus (10 pts)
    if "VolumeSurge" in df.columns:
        score += ((df["VolumeSurge"] - 1) * 5).clip(0, 10).fillna(0)  # type: ignore

    # 7. Konsistensi Likuiditas (5 pts)
    if "ConsistencyScore" in df.columns:
        score += (df["ConsistencyScore"] / 100 * 5).fillna(0)  # type: ignore

    return score.clip(0, 100)  # type: ignore


def calculate_entry_signals(
    df: pd.DataFrame,
    df_historical: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Generate entry signals untuk day trading — v3 (+ Technical Pattern Detection)

    4 Tipe Sinyal:
      BUY - BREAKOUT   : Volume meledak + harga break high (aggressive)
      BUY - AKUMULASI  : Big money / bandar diam-diam masuk (swing)
      BUY - MOMENTUM   : Trend following dengan volume backup
      BUY - REVERSAL   : Bounce dari area support (counter-trend)

    Pattern bonus: deteksi candlestick, multi-bar, volume & MA patterns menambah
    score dan alasan sinyal.

    Grading: A (konfirmasi kuat ≥70), B (memadai ≥45), C (spekulatif ≥25)
    """
    result = df.copy()
    has_md = "PricePosition" in df.columns

    result["EntrySignal"]    = "HOLD"  # type: ignore
    result["SignalGrade"]    = ""      # type: ignore
    result["SignalStrength"] = 0       # type: ignore
    result["SignalReason"]   = ""      # type: ignore
    result["TechPatterns"]   = ""      # type: ignore
    result["StopLoss"]       = 0.0     # type: ignore
    result["Target1"]        = 0.0     # type: ignore
    result["Target2"]        = 0.0     # type: ignore
    result["RiskReward"]     = 0.0     # type: ignore

    # Pre-build historical lookup per StockCode (sorted ascending)
    hist_lookup: Dict[str, pd.DataFrame] = {}
    if df_historical is not None and len(df_historical) > 0:
        for code, grp in df_historical.groupby("StockCode"):
            hist_lookup[str(code)] = grp.sort_values("FileDate")

    for idx, row in result.iterrows():
        close        = float(row["Close"])
        change_pct   = float(row["ChangePct"])
        volatility   = float(row["Volatility"])
        net_foreign  = float(row["NetForeign"])
        non_reg      = float(row["NonRegRatio"])
        avg_trade    = float(row["AvgTradeSize"])
        volume_surge = float(row.get("VolumeSurge", 1.0)) if has_md else 1.0
        value_surge  = float(row.get("ValueSurge",  1.0)) if has_md else 1.0
        position     = float(row.get("PricePosition", 50.0)) if has_md else 50.0
        phase        = str(row.get("MarketPhase", "Unknown")) if has_md else "Unknown"
        resistance   = float(row.get("Resistance7d", 0.0)) if has_md else 0.0
        stock_code   = str(row["StockCode"])

        atr_pct = max(volatility, 1.0)  # proxy ATR harian

        # ------------------------------------------------------------------
        # Technical Pattern Detection
        # ------------------------------------------------------------------
        hist_rows = hist_lookup.get(stock_code)
        pat_info  = detect_all_patterns(row, hist_rows)
        pat_score = pat_info["pattern_score"]          # 0-30
        pat_str   = pat_info["pattern_str"]            # "Hammer + Golden Cross..."
        bullish_pats: List[str] = pat_info["bullish_patterns"]
        bearish_pats: List[str] = pat_info["bearish_patterns"]

        # Pattern context flags
        has_bullish_candle  = any(p in bullish_pats for p in
                                  {"Hammer", "Bullish Marubozu", "Bullish Engulfing",
                                   "Bullish Pin Bar", "Inverted Hammer", "Tweezer Bottom"})
        has_bearish_warning = len(bearish_pats) > 0
        has_golden_cross    = "Golden Cross (EMA5/10)" in bullish_pats
        has_above_ma        = "Above MA Stack" in bullish_pats
        has_volume_climax   = "Volume Climax" in bullish_pats
        has_ema_breakout    = "EMA5 Breakout" in bullish_pats
        bearish_count       = len(bearish_pats)
        bullish_count       = len(bullish_pats)
        strong_accumulation = (
            non_reg >= 0.08 and
            avg_trade >= 8_000_000 and
            net_foreign > 0 and
            value_surge >= 1.0
        )

        # ------------------------------------------------------------------
        # Scoring setiap tipe sinyal (0-100 base, +pattern bonus)
        # ------------------------------------------------------------------
        candidates: dict[str, float] = {}

        # --- BREAKOUT ---
        if change_pct >= 1.0 and volume_surge >= 2.0:
            pts  = min(30, change_pct * 5)
            pts += min(30, (volume_surge - 1) * 12)
            pts += min(20, (value_surge  - 1) * 10)
            pts += min(20, max(0.0, position - 50) * 0.5)
            # Pattern boost
            if has_bullish_candle and position >= 55:  pts += 4
            if has_volume_climax:                      pts += 4
            if has_ema_breakout:                       pts += 4
            if has_bearish_warning:                    pts -= 12
            candidates["BUY - BREAKOUT"] = max(0.0, pts)

        # --- AKUMULASI (Bandar) ---
        if (
            non_reg >= 0.03 and
            avg_trade >= 5_000_000 and
            (net_foreign > 0 or value_surge >= 1.1 or volume_surge >= 1.0)
        ):
            pts  = min(35, non_reg * 500)
            pts += min(25, avg_trade / 800_000)
            pts += min(20, net_foreign / 1e8) if net_foreign > 0 else 0
            pts += 20 if value_surge >= 1.2 else (10 if value_surge >= 0.9 else 0)
            # Pattern boost
            if has_bullish_candle and value_surge >= 1.0:  pts += 3
            if has_above_ma:                               pts += 3
            if has_bearish_warning:                        pts -= 10
            if phase == "Trending Down":                  pts -= 12
            candidates["BUY - AKUMULASI"] = max(0.0, pts)

        # --- MOMENTUM ---
        if change_pct >= 0.5 and volume_surge >= 1.2 and phase != "Trending Down":
            pts  = min(30, change_pct * 5)
            pts += min(25, (volume_surge - 0.8) * 20)
            pts += min(20, (value_surge  - 0.8) * 15)
            pts += 15 if phase == "Trending Up" else (8 if phase == "Sideways" else 0)
            pts += 10 if 35 <= position <= 80 else 0
            # Pattern boost
            if has_golden_cross:                     pts += 6
            if has_above_ma:                         pts += 4
            if has_bullish_candle and position >= 45: pts += 4
            if "Bullish Marubozu" in bullish_pats:  pts += 3
            if has_bearish_warning:                  pts -= 10
            candidates["BUY - MOMENTUM"] = max(0.0, pts)

        # --- REVERSAL ---
        if -5.0 <= change_pct <= -1.0 and volume_surge >= 0.8 and phase != "Trending Down":
            pts  = min(30, abs(change_pct) * 5)
            pts += min(25, max(0.0, (50 - position)) * 0.8)
            pts += min(20, (volume_surge - 0.5) * 20)
            pts += min(15, net_foreign / 5e7) if net_foreign > 0 else 0
            pts += 10 if phase in ["Sideways", "Trending Up"] else 0
            # Pattern boost: hammer/pin bar di area support = sinyal kuat
            if ("Hammer" in bullish_pats or "Bullish Pin Bar" in bullish_pats) and position <= 20: pts += 8
            if "Tweezer Bottom" in bullish_pats and position <= 25:  pts += 6
            if "Bullish Engulfing" in bullish_pats:                  pts += 6
            if has_bearish_warning and bearish_count >= 2:            pts -= 8
            candidates["BUY - REVERSAL"] = max(0.0, pts)

        # ------------------------------------------------------------------
        # Pilih sinyal terkuat
        # ------------------------------------------------------------------
        if candidates:
            best_signal = max(candidates, key=lambda k: candidates[k])
            best_score  = candidates[best_signal]
        else:
            best_signal, best_score = "HOLD", 0.0

        # Quality gates: suppress BUY jika bertentangan dengan trend/pattern context.
        if best_signal == "BUY - BREAKOUT" and (phase == "Trending Down" or bearish_count >= 1):
            best_signal, best_score = "HOLD", 0.0
        elif best_signal == "BUY - MOMENTUM" and (phase == "Trending Down" or bearish_count >= 1):
            best_signal, best_score = "HOLD", 0.0
        elif best_signal == "BUY - AKUMULASI":
            if phase == "Trending Down" and not strong_accumulation:
                best_signal, best_score = "HOLD", 0.0
            elif bearish_count >= 2 and bullish_count == 0:
                best_signal, best_score = "HOLD", 0.0
        elif best_signal == "BUY - REVERSAL":
            if position > 35 and bullish_count == 0:
                best_signal, best_score = "HOLD", 0.0
            elif bearish_count >= 2 and bullish_count == 0:
                best_signal, best_score = "HOLD", 0.0

        # Override AVOID untuk kondisi ekstrem
        if position > 92 or change_pct > 12:
            best_signal, best_score = "AVOID - OVERBOUGHT", 0.0
        elif phase == "Trending Down" and change_pct < -3:
            best_signal, best_score = "AVOID - DOWNTREND", 0.0
        elif change_pct < -7:
            best_signal, best_score = "AVOID - CRASH", 0.0

        # Grade kualitas sinyal
        grade = ""
        if "BUY" in best_signal:
            if best_score >= 70:   grade = "A"
            elif best_score >= 45: grade = "B"
            elif best_score >= 25: grade = "C"
            else:
                best_signal, best_score = "HOLD", 0.0

        # ------------------------------------------------------------------
        # Risk management per tipe sinyal
        # ------------------------------------------------------------------
        stop = tgt1 = tgt2 = 0.0
        reasons: list[str] = []

        if best_signal == "BUY - BREAKOUT":
            reasons = [f"Naik {change_pct:+.1f}%", f"Vol {volume_surge:.1f}x"]
            if phase not in ("", "Unknown"): reasons.append(f"Fase: {phase}")
            stop = close * (1 - min(atr_pct, 4.0) / 100)
            tgt1 = close * 1.04
            tgt2 = (min(resistance * 1.01, close * 1.08)
                    if resistance > close * 1.04 else close * 1.07)

        elif best_signal == "BUY - AKUMULASI":
            reasons = [f"NonReg {non_reg:.1%}", f"Avg Rp{avg_trade/1e6:.0f}jt"]
            if net_foreign > 0: reasons.append("Asing: NET BUY")
            stop = close * 0.97
            tgt1 = close * 1.03
            tgt2 = close * 1.06

        elif best_signal == "BUY - MOMENTUM":
            reasons = [f"Naik {change_pct:+.1f}%", f"Vol {volume_surge:.1f}x"]
            if phase not in ("", "Unknown"): reasons.append(f"Fase: {phase}")
            stop = close * (1 - min(atr_pct * 0.8, 4.0) / 100)
            tgt1 = close * 1.03
            tgt2 = (min(resistance * 0.98, close * 1.06)
                    if resistance > close * 1.03 else close * 1.05)

        elif best_signal == "BUY - REVERSAL":
            reasons = [f"Dip {change_pct:.1f}%", f"Pos {position:.0f}/100"]
            if volume_surge > 1: reasons.append(f"Vol {volume_surge:.1f}x")
            stop = close * (1 - min(atr_pct, 5.0) / 100)
            tgt1 = close * 1.03
            tgt2 = close * 1.06

        # Append top bullish patterns to reasons
        if "BUY" in best_signal and bullish_pats:
            top_pats = bullish_pats[:2]  # max 2 pola di reason
            reasons.append(" + ".join(top_pats))

        risk   = close - stop if stop > 0 else 1.0
        reward = tgt2 - close if tgt2 > 0 else 0.0
        rr     = round(reward / risk, 2) if risk > 0 else 0.0

        result.at[idx, "EntrySignal"]    = best_signal           # type: ignore
        result.at[idx, "SignalGrade"]    = grade                 # type: ignore
        result.at[idx, "SignalStrength"] = int(min(best_score, 100))  # type: ignore
        result.at[idx, "SignalReason"]   = " | ".join(reasons)  # type: ignore
        result.at[idx, "TechPatterns"]   = pat_str              # type: ignore
        result.at[idx, "StopLoss"]       = round(stop)          # type: ignore
        result.at[idx, "Target1"]        = round(tgt1)          # type: ignore
        result.at[idx, "Target2"]        = round(tgt2)          # type: ignore
        result.at[idx, "RiskReward"]     = rr                   # type: ignore

    return result


def screen(df: pd.DataFrame) -> pd.DataFrame:
    """Filter saham untuk day trading - HYBRID STRATEGY + MULTI-DAY"""
    
    # Check if multi-day metrics exist
    has_multiday = "ConsistencyScore" in df.columns
    
  
    base_filter = (
        # Filter Likuiditas
        (df["Value"] >= MIN_VALUE) &
        (df["Volume"] >= MIN_VOLUME) &
        (df["Frequency"] >= MIN_FREQUENCY) &
        (df["SpreadPct"] <= MAX_SPREAD_PCT) &
        
        # Filter Harga
        (df["Close"] >= MIN_PRICE) &
        (df["Close"] <= MAX_PRICE) &
        
        # Filter Volatilitas
        (df["Volatility"] >= MIN_VOLATILITY) &
        (df["Volatility"] <= MAX_VOLATILITY) &
        
        # Filter Aktivitas
        (df["VolumeVsShares"] >= MIN_VOLUME_VS_SHARES) &
        (df["AvgTradeSize"] >= AVG_TRADE_SIZE_MIN) &
        
        # Filter Value Rank
        (df["ValueRank"] >= MIN_VALUE_RANK_PERCENTILE) &
        
        # Filter Big Money
        (df["NetForeign"] >= MIN_NET_FOREIGN) &
        (df["NonRegRatio"] >= MIN_NONREG_RATIO)
    )
    
    # Add multi-day filters if available
    if has_multiday:
        
        multiday_filter = (
            (df["LiquidDays30d"] >= MIN_LIQUID_DAYS) &  # Consistently liquid
            (df["ConsistencyScore"] >= 40)  # At least 40% consistency (kualitas)
        )
        filtered = df[base_filter & multiday_filter].copy()
    else:
        filtered = df[base_filter].copy()
    
    # Categorize: Momentum vs Reversal
    filtered["Category"] = "Neutral"  # type: ignore
    filtered.loc[filtered["ChangePct"] >= MOMENTUM_THRESHOLD, "Category"] = "Momentum"  # type: ignore
    filtered.loc[filtered["ChangePct"] <= REVERSAL_THRESHOLD, "Category"] = "Reversal"  # type: ignore
    
    # Add quality flags if multi-day available
    if has_multiday:
        filtered["VolumeSurgeFlag"] = filtered["VolumeSurge"] >= VOLUME_SURGE_THRESHOLD  # type: ignore
        filtered["BreakoutFlag"] = (
            (filtered["VolumeSurge"] >= VOLUME_SURGE_THRESHOLD) &
            (filtered["ChangePct"] > 2) &
            (filtered["Trend30d"] == "Up")
        )  # type: ignore
    
    # Sort by DayTradingScore
    return filtered.sort_values(by="DayTradingScore", ascending=False)


def send_telegram(message: str) -> bool:
    """Kirim pesan ke Telegram, split jika terlalu panjang"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARNING] Telegram credentials not configured. Message not sent.")
        return False
    
    MAX_LENGTH = 4000  # Telegram limit 4096, pakai 4000 untuk safety
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Split message jika terlalu panjang
    messages: List[str] = []
    if len(message) <= MAX_LENGTH:
        messages = [message]
    else:
        # Split by lines untuk maintain formatting
        lines = message.split('\n')
        current_msg = ""
        
        for line in lines:
            if len(current_msg) + len(line) + 1 <= MAX_LENGTH:
                current_msg += line + '\n'
            else:
                if current_msg:
                    messages.append(current_msg.rstrip())
                current_msg = line + '\n'
        
        if current_msg:
            messages.append(current_msg.rstrip())
    
    # Send semua messages
    try:
        for idx, msg in enumerate(messages, 1):
            if len(messages) > 1:
                # Tambahkan part indicator
                header = f"📄 <b>Part {idx}/{len(messages)}</b>\n\n"
                msg_to_send = header + msg if idx > 1 else msg
            else:
                msg_to_send = msg
            
            response = requests.post(
                url,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": msg_to_send,
                    "parse_mode": "HTML"
                },
                timeout=20
            )
            
            if response.status_code != 200:
                error_detail = response.json().get('description', 'Unknown error')
                print(f"[ERROR] Telegram API error: {error_detail}")
                print(f"[DEBUG] Message preview: {msg_to_send[:200]}...")
                
                # Fallback: Try sending without HTML parse mode
                plain_text = msg_to_send.replace('<b>', '').replace('</b>', '').replace('<i>', '').replace('</i>', '').replace('<code>', '').replace('</code>', '')
                response_fallback = requests.post(
                    url,
                    json={
                        "chat_id": TELEGRAM_CHAT_ID,
                        "text": plain_text
                    },
                    timeout=20
                )
                response_fallback.raise_for_status()
                print(f"[OK] Message {idx} sent (fallback to plain text)")
            else:
                print(f"[OK] Message {idx}/{len(messages)} sent")
        
        print(f"[OK] Total {len(messages)} message(s) sent to Telegram")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram: {e}")
        if hasattr(e, 'response') and e.response is not None:  # type: ignore
            try:
                error_data = e.response.json()  # type: ignore
                print(f"[ERROR] Telegram response: {error_data}")
            except:
                print(f"[ERROR] Response text: {e.response.text[:500]}")  # type: ignore
        return False


# =========================
# BROKER FUNCTIONS
# =========================

def _get_current_api_key() -> str:
    global _current_key_index
    if not _goapi_keys:
        raise RuntimeError("Tidak ada GOAPI key yang dikonfigurasi")
    return _goapi_keys[_current_key_index]


def _rotate_api_key() -> None:
    global _current_key_index
    if _goapi_keys:
        _current_key_index = (_current_key_index + 1) % len(_goapi_keys)


def _http_get_json(url: str, headers: Dict[str, str], params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return cast(Dict[str, Any], r.json())


def fetch_broker_summary(stock_code: str, trade_date: str, max_retries: int = 5) -> Dict[str, Any]:
    if not _goapi_keys:
        raise RuntimeError("Tidak ada GOAPI key")
    url = f"{GOAPI_BASE}/stock/idx/{stock_code}/broker_summary"
    params: Dict[str, Any] = {"date": trade_date, "investor": "ALL"}
    attempt = 0
    while attempt < max_retries:
        try:
            key = _get_current_api_key()
            raw = _http_get_json(url, {"X-API-KEY": key}, params)
            if raw.get("status") != "success":
                raise RuntimeError(f"API status bukan success: {raw.get('status')}")
            return raw
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                _rotate_api_key()
                wait = min(2 ** attempt, 30)
                print(f"    Rate limit, tunggu {wait}s...")
                time.sleep(wait)
                attempt += 1
                continue
            raise
    raise RuntimeError(f"Gagal fetch broker {stock_code} setelah {max_retries} percobaan")


def _extract_brokers_by_side(broker_data: Dict[str, Any], side: str, top_n: int) -> List[Dict[str, Any]]:
    """Parse broker list dari response /broker_summary (data.results[])."""
    results: List[Dict[str, Any]] = []
    data_block = broker_data.get("data") or {}  # type: ignore[assignment]
    if not isinstance(data_block, dict):
        return results
    items = data_block.get("results") or []  # type: ignore[assignment]
    if not isinstance(items, list):
        return results
    for raw in items:  # type: ignore[union-attr]
        if not isinstance(raw, dict):
            continue
        item: Dict[str, Any] = raw  # type: ignore[assignment]
        if str(item.get("side") or "").upper() != side.upper():
            continue
        code = str(item.get("code") or "")
        results.append({
            "broker":     code,
            "value":      float(item.get("value") or 0),
            "lot":        int(item.get("lot") or 0),
            "avg":        float(item.get("avg") or 0),
            "percentage": 0.0,  # dihitung di fetch_broker_map
        })
    results.sort(key=lambda x: x["value"], reverse=True)
    return results[:top_n]


def extract_top_buyers(broker_data: Dict[str, Any], top_n: int = MIN_TOP_BROKERS) -> List[Dict[str, Any]]:
    return _extract_brokers_by_side(broker_data, "BUY", top_n)


def extract_top_sellers(broker_data: Dict[str, Any], top_n: int = MIN_TOP_BROKERS) -> List[Dict[str, Any]]:
    return _extract_brokers_by_side(broker_data, "SELL", top_n)


def fetch_broker_map(
    buy_signals: pd.DataFrame,
    trade_date_str: str
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch broker data untuk semua BUY signal stocks.
    Return: {stock_code: {"buyers": [...], "sellers": [...], "bias": str}}
    """
    broker_map: Dict[str, Dict[str, Any]] = {}
    total = len(buy_signals)
    print(f"\n📊 Fetching broker data untuk {total} saham...")

    for i, (_, row) in enumerate(buy_signals.iterrows(), 1):
        stock_code = str(row["StockCode"])
        print(f"  [{i}/{total}] {stock_code}...", end=" ")
        try:
            data = fetch_broker_summary(stock_code, trade_date_str)
            buyers  = extract_top_buyers(data)
            sellers = extract_top_sellers(data)

            # Hitung total value semua transaksi dari API (untuk % per broker)
            all_items = (data.get("data") or {})  # type: ignore[assignment]
            if isinstance(all_items, dict):
                all_results = all_items.get("results") or []  # type: ignore[assignment]
            else:
                all_results = []
            total_all_val = sum(float((r.get("value") or 0)) for r in all_results if isinstance(r, dict))  # type: ignore[union-attr]

            # Isi percentage per entry
            for entry in buyers + sellers:
                entry["percentage"] = (entry["value"] / total_all_val * 100) if total_all_val > 0 else 0.0

            # Hitung bias: NET BUY / NET SELL / NEUTRAL
            total_buy_val  = sum(b["value"] for b in buyers)
            total_sell_val = sum(s["value"] for s in sellers)
            grand_total    = total_buy_val + total_sell_val
            if grand_total > 0:
                buy_ratio = total_buy_val / grand_total
                if buy_ratio >= BROKER_BIAS_THRESHOLD:
                    bias = "NET BUY"
                elif buy_ratio <= (1 - BROKER_BIAS_THRESHOLD):
                    bias = "NET SELL"
                else:
                    bias = "NEUTRAL"
            else:
                bias = "NEUTRAL"

            broker_map[stock_code] = {"buyers": buyers, "sellers": sellers, "bias": bias}
            print(f"✓ Buyers:{len(buyers)} Sellers:{len(sellers)} [{bias}]")
        except Exception as e:
            print(f"✗ {str(e)[:60]}")
        time.sleep(BROKER_CALL_SLEEP)

    return broker_map


# =========================
# TELEGRAM FORMATTING
# =========================

def format_signal_card(
    row: pd.Series,  # type: ignore
    broker_data: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """
    Format satu BUY signal menjadi card Telegram — mengikuti style main_screener.
    broker_data: entry dari broker_map (dict berisi buyers, sellers, bias)
    """
    lines: List[str] = []

    stock_code = html.escape(str(row["StockCode"]))
    stock_name = html.escape(str(row["StockName"]))
    price      = int(row["Close"])
    change_pct = float(row["ChangePct"])
    volume     = int(row["Volume"])
    value_b    = float(row["Value"]) / 1e9
    signal     = str(row["EntrySignal"])
    grade      = str(row.get("SignalGrade",    "") or "")
    strength   = int(row["SignalStrength"])
    reason     = str(row.get("SignalReason",   "") or "")
    stop_loss  = int(row["StopLoss"])
    target1    = int(row["Target1"])
    target2    = int(row["Target2"])
    rr         = float(row["RiskReward"])
    volatility = float(row["Volatility"])

    # Optional multi-day fields
    surge      = float(row["VolumeSurge"])  if "VolumeSurge"     in row.index else 1.0  # type: ignore[operator]
    trend      = str(row["Trend30d"])       if "Trend30d"        in row.index else ""    # type: ignore[operator]
    phase      = str(row["MarketPhase"])    if "MarketPhase"     in row.index else ""    # type: ignore[operator]
    position   = float(row["PricePosition"]) if "PricePosition" in row.index else 50.0  # type: ignore[operator]
    support    = int(row["Support7d"])      if "Support7d"       in row.index else 0     # type: ignore[operator]
    resistance = int(row["Resistance7d"])   if "Resistance7d"    in row.index else 0     # type: ignore[operator]
    tech_pats  = str(row["TechPatterns"])   if "TechPatterns"    in row.index else ""    # type: ignore[operator]
    tech_pats  = str(row["TechPatterns"])   if "TechPatterns"    in row.index else ""    # type: ignore[operator]

    if   "BREAKOUT"  in signal: sig_emoji = "💥"
    elif "AKUMULASI" in signal: sig_emoji = "🐋"
    elif "MOMENTUM"  in signal: sig_emoji = "🚀"
    elif "REVERSAL"  in signal: sig_emoji = "🔄"
    else:                        sig_emoji = "📌"

    grade_str = f" [Grade <b>{html.escape(grade)}</b>]" if grade else ""

    # --- Header saham ---
    lines.append(f"<b>📊 {stock_code}</b> - {stock_name}")
    lines.append(f"💰 Price: Rp {price:,}")
    lines.append(f"📈 Change: {change_pct:+.2f}%")
    lines.append(f"📊 Volume: {volume:,}")
    lines.append(f"💵 Value: Rp {value_b:.2f}B")
    lines.append(f"🕯️ Volatility: {volatility:.2f}%")
    lines.append("")

    # --- Signal ---
    lines.append(f"{sig_emoji} <b>{html.escape(signal)}</b>{grade_str}")
    lines.append(f"⭐ Score: {strength}/100")
    if reason:
        lines.append(f"📝 {html.escape(reason)}")
    lines.append(f"🛡️ Stop: Rp {stop_loss:,}")
    lines.append(f"🎯 T1: Rp {target1:,} | T2: Rp {target2:,}")
    lines.append(f"📊 R/R: 1:{rr:.2f}")

    # --- Multi-day context ---
    if surge > 0 or trend or phase:
        surge_emoji = "🔥" if surge >= 2 else "⬆️" if surge >= 1.5 else ""
        ctx_parts: List[str] = []
        if surge > 0:
            ctx_parts.append(f"Vol Surge: {surge:.1f}x {surge_emoji}")
        if trend:
            ctx_parts.append(f"Trend: {trend}")
        if phase:
            ctx_parts.append(f"Fase: {phase}")
        if ctx_parts:
            lines.append(f"📈 {' | '.join(ctx_parts)}")

    if support > 0 and resistance > 0:
        lines.append(f"📐 Support: {support:,} | Resistance: {resistance:,} | Pos: {position:.0f}/100")

    # --- Technical Patterns ---
    if tech_pats and tech_pats.strip():
        lines.append("")
        lines.append(f"🔮 <b>Pola Teknikal:</b> {html.escape(tech_pats)}")

    # --- Broker data ---
    if broker_data:
        bias: str = str(broker_data.get("bias") or "NEUTRAL")
        bias_emoji = "🟢" if bias == "NET BUY" else "🔴" if bias == "NET SELL" else "⚪"
        lines.append("")
        lines.append(f"{bias_emoji} <b>Broker Bias: {bias}</b>")

        buyers: List[Dict[str, Any]] = list(broker_data.get("buyers") or [])  # type: ignore[assignment]
        if buyers:
            lines.append("")
            lines.append("<b>🟢 TOP BUYERS:</b>")
            for b in buyers:
                b_val = b["value"] / 1e9
                b_avg = float(b.get("avg") or 0)
                lines.append(f"  • {html.escape(str(b['broker']))}: {b_val:.2f}M ({b['percentage']:.1f}%) | avg {b_avg:,.0f}")

        sellers: List[Dict[str, Any]] = list(broker_data.get("sellers") or [])  # type: ignore[assignment]
        if sellers:
            lines.append("")
            lines.append("<b>🔴 TOP SELLERS:</b>")
            for s in sellers:
                s_val = s["value"] / 1e9
                s_avg = float(s.get("avg") or 0)
                lines.append(f"  • {html.escape(str(s['broker']))}: {s_val:.2f}M ({s['percentage']:.1f}%) | avg {s_avg:,.0f}")

    return "\n".join(lines)


def format_summary_header(df: pd.DataFrame, buy_count: int, use_broker: bool, trade_date_str: str) -> str:
    """Header message — market context + screening summary."""
    has_multiday = "ConsistencyScore" in df.columns

    momentum_stocks = df[df["Category"] == "Momentum"]
    reversal_stocks = df[df["Category"] == "Reversal"]
    neutral_stocks  = df[df["Category"] == "Neutral"]

    mode = "Basic + Broker" if use_broker else "Basic"

    lines: List[str] = []
    lines.append(f"🎯 <b>DAY TRADING SIGNALS ({mode})</b>")
    lines.append("━" * 30)
    lines.append(f"<b>Date:</b> {trade_date_str}")
    lines.append(f"<b>Lolos Screening:</b> {len(df)} saham")
    lines.append(f"<b>BUY Signals:</b> {buy_count}")
    lines.append("")

    lines.append("📊 <b>CATEGORY BREAKDOWN</b>")
    lines.append(f"• 📈 Momentum: {len(momentum_stocks)}")
    lines.append(f"• 🔄 Reversal: {len(reversal_stocks)}")
    lines.append(f"• ➖ Neutral:  {len(neutral_stocks)}")
    lines.append("")

    avg_score = df["DayTradingScore"].mean()
    avg_vol   = df["Volatility"].mean()
    avg_spread = df["SpreadPct"].mean()
    lines.append("📈 <b>MARKET SUMMARY</b>")
    lines.append(f"• Avg Score: {avg_score:.1f}")
    lines.append(f"• Avg Volatility: {avg_vol:.2f}%")
    lines.append(f"• Avg Spread: {avg_spread:.3f}%")

    if has_multiday:
        avg_consistency = df["ConsistencyScore"].mean()
        avg_surge       = df["VolumeSurge"].mean()
        lines.append(f"• Avg Consistency: {avg_consistency:.1f}% (7d)")
        lines.append(f"• Avg Vol Surge: {avg_surge:.2f}x")

    lines.append("")

    # Market bias
    if len(momentum_stocks) >= len(reversal_stocks) * 2:
        lines.append("📈 <b>Bullish Market</b> — Fokus momentum")
    elif len(reversal_stocks) >= len(momentum_stocks) * 2:
        lines.append("📉 <b>Bearish Market</b> — Fokus reversal")
    else:
        lines.append("⚖️ <b>Balanced Market</b> — Mixed signal")

    # Risk profile
    high_vol = len(df[df["Volatility"] >= 5])
    med_vol  = len(df[(df["Volatility"] >= 2) & (df["Volatility"] < 5)])
    low_vol  = len(df[df["Volatility"] < 2])
    lines.append("")
    lines.append("📊 <b>RISK PROFILE</b>")
    lines.append(f"• 🔥 Aggressive (≥5%): {high_vol}")
    lines.append(f"• ⚖️ Balanced  (2-5%): {med_vol}")
    lines.append(f"• 🛡️ Conservative (&lt;2%): {low_vol}")

    lines.append("━" * 30)
    lines.append(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Day Trading Screener")
    parser.add_argument("--broker", action="store_true", help="Sertakan analisis top broker (butuh GOAPI key)")
    args = parser.parse_args()
    use_broker = args.broker

    if use_broker and not _goapi_keys:
        print("[WARNING] --broker dinyalakan tapi tidak ada GOAPI key. Broker dinonaktifkan.")
        use_broker = False

    # Load dari data/ dan backup/
    files_data = glob.glob(os.path.join(DATA_DIR, "idx_stock_*.json"))
    files_backup = glob.glob(os.path.join("backup", "idx_stock_*.json"))
    
    # Gabungkan dan sort berdasarkan nama file (tanggal)
    files = sorted(files_data + files_backup)
    
    if not files:
        print("[ERROR] No data files found")
        return
    
    print("="*70)
    print("🚀 DAY TRADING SCREENER - MULTI-DAY ANALYSIS")
    print("="*70)
    if use_broker:
        print("   ✅ Broker Analysis: ENABLED")
    else:
        if _goapi_keys:
            print("   ℹ️  Broker: tersedia (gunakan --broker untuk mengaktifkan)")
        else:
            print("   ⚠️  Broker: tidak ada GOAPI key")
    print(f"📁 Total {len(files)} files ditemukan (data/ + backup/)")
    
    # Load today's data (file terbaru)
    latest_file = files[-1]
    print(f"\n📅 Data Hari Ini: {os.path.basename(latest_file)}")
    
    df_today = load_json(latest_file)
    df_today = prepare(df_today)
    
    df_hist_list: List[pd.DataFrame] = []  # initialize before conditional block
    # Load historical data (7 hari)
    historical_files = files[-(LOOKBACK_DAYS+1):-1] if len(files) > LOOKBACK_DAYS else files[:-1]
    
    if len(historical_files) >= 3:  # Minimal 3 hari untuk analisis
        print(f"📊 Loading {len(historical_files)} hari historical data...")
        
        df_hist_list = []
        for f in historical_files:
            try:
                df_temp = load_json(f)
                df_temp = prepare(df_temp)
                df_hist_list.append(df_temp)
            except Exception as e:
                print(f"  ⚠️  Skip {os.path.basename(f)}: {e}")
                continue
        
        if df_hist_list:
            df_historical = pd.concat(df_hist_list, ignore_index=True)
            print(f"✅ Loaded {len(df_hist_list)} days, {len(df_historical)} records total")
            
            # Calculate multi-day metrics
            print("🔄 Calculating multi-day metrics...")
            df_today = calculate_multi_day_metrics(df_today, df_historical)
            
            # Recalculate score dengan consistency
            df_today["DayTradingScore"] = calculate_daytrading_score(df_today)
            
            print("✅ Multi-day analysis complete!\n")
        else:
            print("⚠️  Insufficient historical data. Using single-day analysis.\n")
    else:
        print(f"⚠️  Only {len(historical_files)} days found. Need 10+ for multi-day analysis.")
        print("   Using single-day mode.\n")
    
    # Screen
    res = screen(df_today)
    
    # Generate Entry Signals + Technical Pattern Detection
    if len(res) > 0:
        print("🎯 Generating entry/exit signals + technical patterns...")
        df_hist_for_signals: Optional[pd.DataFrame] = None
        if df_hist_list:
            df_hist_for_signals = pd.concat(df_hist_list, ignore_index=True)
        res = calculate_entry_signals(res, df_historical=df_hist_for_signals)
        
        buy_signals = res[res["EntrySignal"].str.contains("BUY", na=False)]
        print(f"✅ Found {len(buy_signals)} BUY signals!\n")

    cols = [
        "FileDate","StockCode","StockName",
        "Close","ChangePct","Volatility",
        "Volume","Value","Frequency",
        "High","Low","SpreadPct",
        "AvgTradeSize","VolumeVsShares","ValueRank",
        "DayTradingScore","Category",
        # Entry signals & risk management
        "EntrySignal","SignalGrade","SignalStrength","SignalReason",
        "TechPatterns",
        "StopLoss","Target1","Target2","RiskReward",
        # Support/Resistance & multi-day
        "Support7d","Resistance7d","PricePosition","MarketPhase"
    ]

    # Save hasil
    out_file = os.path.join(
        OUTPUT_DIR,
        f"daytrading_{res['FileDate'].iloc[0]}.csv" if len(res) > 0 else "daytrading_no_results.csv"
    )

    if len(res) > 0:
        res[cols].to_csv(out_file, index=False)
        
        avg_score = res["DayTradingScore"].mean()
        max_score = res["DayTradingScore"].max()
        top_stock = res.iloc[0]["StockCode"]
        avg_vol = res["Volatility"].mean()
        
        print(f"[RESULT] {len(res)} saham lolos screening day trading")
        print(f"  • Top Stock: {top_stock} (Score: {max_score:.1f})")
        print(f"  • Avg Score: {avg_score:.1f}")
        print(f"  • Avg Volatility: {avg_vol:.2f}%")
        print(f"  • Avg Spread: {res['SpreadPct'].mean():.3f}%")
        
        # Top 10 saham by score
        top10 = res.nlargest(10, "DayTradingScore")
        print(f"\n[TOP 10 DAY TRADING STOCKS]")
        for idx, row in enumerate(top10.itertuples(), 1):
            emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx}."
            print(f"  {emoji} {row.StockCode}: Rp {row.Close:,.0f} ({row.ChangePct:+.2f}%) | Vol {row.Volatility:.2f}% | Score {row.DayTradingScore:.1f}")  # type: ignore
        
        # Grouping by volatility
        high_vol = res[res["Volatility"] >= 5]
        med_vol = res[(res["Volatility"] >= 2) & (res["Volatility"] < 5)]
        low_vol = res[res["Volatility"] < 2]
        
        print(f"\n[VOLATILITY DISTRIBUTION]")
        print(f"  • High (≥5%): {len(high_vol)} saham - untuk aggressive trading")
        print(f"  • Medium (2-5%): {len(med_vol)} saham - balanced risk/reward")
        print(f"  • Low (<2%): {len(low_vol)} saham - conservative trading")
        
        # Fetch broker data (opsional)
        broker_data_map: Optional[Dict[str, Dict[str, Any]]] = None
        buy_df = res[res["EntrySignal"].str.contains("BUY", na=False)].sort_values(
            by="SignalStrength", ascending=False
        )
        trade_date_str = str(res["FileDate"].iloc[0])

        if use_broker and len(buy_df) > 0:
            broker_data_map = fetch_broker_map(buy_df, trade_date_str)

        # =========================
        # KIRIM KE TELEGRAM
        # =========================
        print("\n" + "="*70)
        print("📱 SENDING TO TELEGRAM")
        print("="*70 + "\n")

        # 1. Header / summary
        header_msg = format_summary_header(res, len(buy_df), use_broker, trade_date_str)
        send_telegram(header_msg)
        time.sleep(1)

        # 2. Satu pesan per BUY signal — ikuti style main_screener
        total_signals = len(buy_df)
        for i, (_, row) in enumerate(buy_df.iterrows(), 1):
            stock_code = str(row["StockCode"])
            bd = broker_data_map.get(stock_code) if broker_data_map else None
            card = format_signal_card(row, bd)  # type: ignore[arg-type]
            full_msg = f"<b>[{i}/{total_signals}]</b>\n\n{card}"
            if send_telegram(full_msg):
                print(f"  ✓ [{i}/{total_signals}] {stock_code} sent")
            else:
                print(f"  ✗ [{i}/{total_signals}] {stock_code} failed")
            time.sleep(1)

        print(f"\n✅ Sent {total_signals} signal card(s) to Telegram")
    else:
        print(f"[OK] Tidak ada saham yang lolos screening hari ini")
        res[cols].to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
