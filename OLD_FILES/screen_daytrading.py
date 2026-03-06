import os
import json
import glob
import pandas as pd
import requests
from datetime import datetime, date
from dotenv import load_dotenv
from typing import List
import html

# Load environment variables
load_dotenv()

DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Telegram Config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

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
        "Previous","Close","Volume","Value","Frequency","High","Low",
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
    
    return result


def calculate_daytrading_score(df: pd.DataFrame) -> pd.Series:  # type: ignore
    """Hitung score day trading 0-100 - HYBRID STRATEGY + MULTI-DAY"""
    score = pd.Series(0.0, index=df.index)
    
    # 1. Likuiditas - Value (20 poin) - reduced for multi-day
    score += (df["Value"] / df["Value"].max() * 20).fillna(0)  # type: ignore
    
    # 2. Volatilitas optimal (15 poin)
    volatility_normalized = 15 * (1 - (df["Volatility"] - 5).abs() / 10)
    volatility_score = volatility_normalized.where(
        (df["Volatility"] >= 1) & (df["Volatility"] <= 15), 0
    )
    score += volatility_score.clip(0, 15)  # type: ignore
    
    # 3. Frequency - aktivitas transaksi (15 poin)
    score += (df["Frequency"] / df["Frequency"].max() * 15).fillna(0)  # type: ignore
    
    # 4. Spread rendah (10 poin)
    spread_score = (1 - df["SpreadPct"] / MAX_SPREAD_PCT) * 10
    score += spread_score.clip(0, 10)  # type: ignore
    
    # 5. Average Trade Size (10 poin)
    score += (df["AvgTradeSize"] / df["AvgTradeSize"].max() * 10).fillna(0)  # type: ignore
    
    # 6. Momentum Bias (15 poin)
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
    
    # 7. Consistency Score (15 poin) - NEW!
    if "ConsistencyScore" in df.columns:
        score += (df["ConsistencyScore"] / 100 * 15).fillna(0)  # type: ignore
    
    return score


def calculate_entry_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate konkret entry/exit signals untuk day trading"""
    result = df.copy()
    
    # Initialize signal columns
    result["EntrySignal"] = "HOLD"  # type: ignore
    result["SignalStrength"] = 0  # type: ignore
    result["StopLoss"] = 0.0  # type: ignore
    result["Target1"] = 0.0  # type: ignore
    result["Target2"] = 0.0  # type: ignore
    result["RiskReward"] = 0.0  # type: ignore
    
    if "PricePosition" not in df.columns:
        return result  # Skip if no multi-day data
    
    for idx, row in result.iterrows():
        close = row["Close"]
        support = row.get("Support7d", 0)
        resistance = row.get("Resistance7d", 0)
        change_pct = row["ChangePct"]
        position = row.get("PricePosition", 50)
        phase = row.get("MarketPhase", "Unknown")
        surge = row.get("VolumeSurge", 1)
        
        # BUY SIGNAL - Momentum/Breakout (OPTIMIZED & FIXED)
        if (change_pct >= 0.3 and  # Lebih sensitif: 0.3% (ada movement)
            surge >= 0.7 and       # Lebih toleran: 0.7x (volume wajar)
            position < 95 and      # Hampir semua position OK
            phase != "Trending Down"):  # SKIP downtrend
            
            result.at[idx, "EntrySignal"] = "BUY - MOMENTUM"  # type: ignore
            result.at[idx, "SignalStrength"] = min(100, int(change_pct * 15 + surge * 15))  # type: ignore
            
            # Stop loss: FIXED 4% untuk momentum (lebih longgar)
            # Momentum butuh ruang untuk bergerak
            stop = close * 0.96  # Fixed 4% risk
            result.at[idx, "StopLoss"] = round(stop)  # type: ignore
            
            # Target 1: +3% (quick profit)
            result.at[idx, "Target1"] = round(close * 1.03)  # type: ignore
            
            # Target 2: +5% or resistance-1% (whichever is closer)
            target2_by_pct = close * 1.05
            if resistance > close:
                resistance_distance = (resistance - close) / close * 100
                # Only use resistance if within 3-8% range
                if 3 <= resistance_distance <= 8:
                    target2 = min(resistance * 0.99, target2_by_pct)
                else:
                    target2 = target2_by_pct
            else:
                target2 = target2_by_pct
            
            result.at[idx, "Target2"] = round(target2)  # type: ignore
            
            # Risk/Reward Ratio
            risk = close - stop
            reward = target2 - close
            rr = reward / risk if risk > 0 else 0
            result.at[idx, "RiskReward"] = round(rr, 2)  # type: ignore
        
        # BUY SIGNAL - Reversal/Bounce (LEBIH REALISTIS)
        elif (change_pct <= -1.0 and  # Ubah ke -1.0% (mild dip)
              change_pct >= -3.5 and  # Max -3.5% (bukan crash)
              position < 50 and       # Relaksasi ke 50
              surge >= 1.0 and        # Turunkan ke 1.0x (normal volume OK)
              phase != "Trending Down"):  # JANGAN BUY kalau downtrend!
            
            result.at[idx, "EntrySignal"] = "BUY - REVERSAL"  # type: ignore
            result.at[idx, "SignalStrength"] = min(100, int(abs(change_pct) * 12 + surge * 15))  # type: ignore
            
            # Stop loss: FIXED 5% untuk reversal (butuh lebih banyak ruang)
            # Reversal lebih volatile, perlu cushion lebih
            stop = close * 0.95  # Fixed 5% risk
            
            result.at[idx, "StopLoss"] = round(stop)  # type: ignore
            
            # Target 1: +3% (better reward)
            result.at[idx, "Target1"] = round(close * 1.03)  # type: ignore
            
            # Target 2: +5% (better R:R)
            result.at[idx, "Target2"] = round(close * 1.05)  # type: ignore
            
            risk = close - stop
            reward = close * 1.05 - close
            rr = reward / risk if risk > 0 else 0
            result.at[idx, "RiskReward"] = round(rr, 2)  # type: ignore
        
        # SELL/AVOID - Overbought
        elif position > 85 or change_pct > 8:
            result.at[idx, "EntrySignal"] = "AVOID - OVERBOUGHT"  # type: ignore
            result.at[idx, "SignalStrength"] = 0  # type: ignore
        
        # SELL/AVOID - Weak
        elif change_pct < -6 or surge < 0.8:
            result.at[idx, "EntrySignal"] = "AVOID - WEAK"  # type: ignore
            result.at[idx, "SignalStrength"] = 0  # type: ignore
    
    return result


def screen(df: pd.DataFrame) -> pd.DataFrame:
    """Filter saham untuk day trading - HYBRID STRATEGY + MULTI-DAY"""
    
    # Check if multi-day metrics exist
    has_multiday = "ConsistencyScore" in df.columns
    
    print(f"\n🔍 FILTER DEBUG:")
    print(f"   Total stocks: {len(df)}")
    
    # Debug: Check each filter individually
    print(f"   Pass Value >= {MIN_VALUE/1e9:.0f}B: {(df['Value'] >= MIN_VALUE).sum()}")
    print(f"   Pass Volume >= {MIN_VOLUME/1e6:.0f}M: {(df['Volume'] >= MIN_VOLUME).sum()}")
    print(f"   Pass Frequency >= {MIN_FREQUENCY}: {(df['Frequency'] >= MIN_FREQUENCY).sum()}")
    print(f"   Pass Volatility {MIN_VOLATILITY}-{MAX_VOLATILITY}%: {((df['Volatility'] >= MIN_VOLATILITY) & (df['Volatility'] <= MAX_VOLATILITY)).sum()}")
    print(f"   Pass AvgTradeSize >= {AVG_TRADE_SIZE_MIN/1e6:.0f}M: {(df['AvgTradeSize'] >= AVG_TRADE_SIZE_MIN).sum()}")
    
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
    
    print(f"   ✅ Pass base filters: {base_filter.sum()}")
    
    # Add multi-day filters if available
    if has_multiday:
        print(f"   Pass LiquidDays >= {MIN_LIQUID_DAYS}: {(df['LiquidDays30d'] >= MIN_LIQUID_DAYS).sum()}")
        print(f"   Pass Consistency >= 40%: {(df['ConsistencyScore'] >= 40).sum()}")
        
        multiday_filter = (
            (df["LiquidDays30d"] >= MIN_LIQUID_DAYS) &  # Consistently liquid
            (df["ConsistencyScore"] >= 40)  # At least 40% consistency (kualitas)
        )
        filtered = df[base_filter & multiday_filter].copy()
        print(f"   ✅ Pass all filters (with multi-day): {len(filtered)}")
    else:
        filtered = df[base_filter].copy()
        print(f"   ✅ Pass all filters: {len(filtered)}")
    
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


def send_telegram(message: str) -> None:
    """Kirim pesan ke Telegram, split jika terlalu panjang"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARNING] Telegram credentials not configured. Message not sent.")
        return
    
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
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram: {e}")
        if hasattr(e, 'response') and e.response is not None:  # type: ignore
            try:
                error_data = e.response.json()  # type: ignore
                print(f"[ERROR] Telegram response: {error_data}")
            except:
                print(f"[ERROR] Response text: {e.response.text[:500]}")  # type: ignore


def format_telegram_message_single(df: pd.DataFrame) -> str:
    """Format hasil screening untuk Telegram - MULTI-DAY ANALYSIS"""
    msg = "📊 <b>DAY TRADING ALERT - MULTI-DAY</b>\n"
    msg += "=" * 35 + "\n\n"
    
    if len(df) == 0:
        msg += "❌ Tidak ada saham yang lolos screening\n"
        return msg
    
    # Check if multi-day data available
    has_multiday = "ConsistencyScore" in df.columns
    has_signals = "EntrySignal" in df.columns
    
    # 🎯 BUY SIGNALS SECTION (PALING PENTING UNTUK DAY TRADING!)
    if has_signals:
        buy_stocks = df[df["EntrySignal"].str.contains("BUY", na=False)].sort_values(by="SignalStrength", ascending=False)
        
        if len(buy_stocks) > 0:
            msg += "🎯 <b>ACTIONABLE BUY SIGNALS</b>\n"
            msg += "─" * 35 + "\n"
            
            for idx, row in enumerate(buy_stocks.head(5).itertuples(), 1):
                stock_code = html.escape(str(row.StockCode))
                stock_name = html.escape(str(row.StockName)[:15])  # type: ignore
                price = row.Close  # type: ignore
                change_pct = row.ChangePct  # type: ignore
                signal = html.escape(str(row.EntrySignal))  # type: ignore
                strength = row.SignalStrength  # type: ignore
                stop_loss = row.StopLoss  # type: ignore
                target1 = row.Target1  # type: ignore
                target2 = row.Target2  # type: ignore
                rr = row.RiskReward  # type: ignore
                
                emoji = "🚀" if idx == 1 else "⭐" if idx == 2 else "💎" if idx == 3 else f"{idx}."
                
                msg += f"{emoji} <b>{stock_code}</b> - {stock_name}\n"
                msg += f"   {signal} (Strength: {strength}/100)\n"
                msg += f"   Price: Rp {price:,.0f} ({change_pct:+.2f}&#37;)\n"
                msg += f"   🛑 SL: {stop_loss:,.0f} | 🎯 T1: {target1:,.0f} | T2: {target2:,.0f}\n"
                msg += f"   ⚖️ Risk/Reward: 1:{rr:.1f}\n\n"
            
            msg += "\n"
    
    # Summary
    avg_score = df["DayTradingScore"].mean()
    avg_volatility = df["Volatility"].mean()
    avg_spread = df["SpreadPct"].mean()
    
    msg += f"📈 <b>MARKET SUMMARY</b>\n"
    msg += f"• Total Saham: {len(df)}\n"
    msg += f"• Avg Score: {avg_score:.1f}\n"
    msg += f"• Avg Volatility: {avg_volatility:.2f}&#37;\n"
    msg += f"• Avg Spread: {avg_spread:.3f}&#37;\n"
    
    if has_multiday:
        avg_consistency = df["ConsistencyScore"].mean()
        avg_surge = df["VolumeSurge"].mean()
        msg += f"• Avg Consistency: {avg_consistency:.1f}&#37; (7d)\n"
        msg += f"• Avg Volume Surge: {avg_surge:.2f}x\n"
    
    msg += "\n"
    
    # Split by category
    momentum_stocks = df[df["Category"] == "Momentum"].sort_values(by="DayTradingScore", ascending=False)
    reversal_stocks = df[df["Category"] == "Reversal"].sort_values(by="DayTradingScore", ascending=False)
    neutral_stocks = df[df["Category"] == "Neutral"].sort_values(by="DayTradingScore", ascending=False)
    
    msg += f"📊 <b>CATEGORY BREAKDOWN</b>\n"
    msg += f"• 📈 Momentum (Trend): {len(momentum_stocks)} stocks\n"
    msg += f"• 🔄 Reversal (Bounce): {len(reversal_stocks)} stocks\n"
    msg += f"• ➖ Neutral: {len(neutral_stocks)} stocks\n\n"
    
    # Momentum Stocks
    if len(momentum_stocks) > 0:
        msg += f"\n📈 <b>MOMENTUM PLAYS ({len(momentum_stocks)})</b>\n"
        msg += "─" * 35 + "\n"
        
        for idx, row in enumerate(momentum_stocks.head(10).itertuples(), 1):
            stock_code = html.escape(str(row.StockCode))
            stock_name = html.escape(str(row.StockName)[:15])  # type: ignore
            score = row.DayTradingScore  # type: ignore
            change_pct = row.ChangePct  # type: ignore
            volatility = row.Volatility  # type: ignore
            price = row.Close  # type: ignore
            
            emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx}."
            
            msg += f"{emoji} <b>{stock_code}</b> - {stock_name}\n"
            msg += f"   Rp {price:,.0f} ({change_pct:+.2f}&#37;) | Vol: {volatility:.2f}&#37;\n"
            
            # Add multi-day info if available
            if has_multiday:
                surge = row.VolumeSurge if hasattr(row, 'VolumeSurge') else 1  # type: ignore
                surge_float = float(surge)  # type: ignore
                consistency = row.ConsistencyScore if hasattr(row, 'ConsistencyScore') else 0  # type: ignore
                trend = row.Trend30d if hasattr(row, 'Trend30d') else "Unknown"  # type: ignore
                
                surge_emoji = "🔥" if surge_float >= 2 else "⬆️" if surge_float >= 1.5 else ""
                msg += f"   Score: {score:.1f} | Surge: {surge_float:.1f}x {surge_emoji} | Trend: {trend}\n\n"
            else:
                msg += f"   Score: {score:.1f}\n\n"
    
    # Reversal Stocks
    if len(reversal_stocks) > 0:
        msg += f"\n🔄 <b>REVERSAL PLAYS ({len(reversal_stocks)})</b>\n"
        msg += "─" * 35 + "\n"
        
        for idx, row in enumerate(reversal_stocks.head(10).itertuples(), 1):
            stock_code = html.escape(str(row.StockCode))
            stock_name = html.escape(str(row.StockName)[:15])  # type: ignore
            score = row.DayTradingScore  # type: ignore
            change_pct = row.ChangePct  # type: ignore
            volatility = row.Volatility  # type: ignore
            price = row.Close  # type: ignore
            
            emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx}."
            
            msg += f"{emoji} <b>{stock_code}</b> - {stock_name}\n"
            msg += f"   Rp {price:,.0f} ({change_pct:+.2f}&#37;) | Vol: {volatility:.2f}&#37;\n"
            
            # Add multi-day info if available
            if has_multiday:
                surge = row.VolumeSurge if hasattr(row, 'VolumeSurge') else 1  # type: ignore
                surge_float = float(surge)  # type: ignore
                consistency = row.ConsistencyScore if hasattr(row, 'ConsistencyScore') else 0  # type: ignore
                trend = row.Trend30d if hasattr(row, 'Trend30d') else "Unknown"  # type: ignore
                
                surge_emoji = "🔥" if surge_float >= 2 else "⬆️" if surge_float >= 1.5 else ""
                msg += f"   Score: {score:.1f} | Surge: {surge_float:.1f}x {surge_emoji} | Trend: {trend}\n\n"
            else:
                msg += f"   Score: {score:.1f}\n\n"
    
    # Strategy suggestion
    msg += f"\n💡 <b>STRATEGY SUGGESTION</b>\n"
    msg += f"─" * 35 + "\n"
    
    if len(momentum_stocks) >= len(reversal_stocks) * 2:
        msg += "📈 <b>Bullish Market</b>\n"
        msg += f"Momentum dominan! Fokus: 80&#37; momentum, 20&#37; reversal\n"
    elif len(reversal_stocks) >= len(momentum_stocks) * 2:
        msg += "📉 <b>Bearish Market</b>\n"
        msg += f"Reversal dominan! Fokus: 70&#37; reversal, 30&#37; momentum\n"
    else:
        msg += "⚖️ <b>Balanced Market</b>\n"
        msg += f"Mixed signal. Stick to plan: 60&#37; momentum, 40&#37; reversal\n"
    
    msg += f"\n📊 <b>RISK PROFILE</b>\n"
    high_vol = df[df["Volatility"] >= 5]
    med_vol = df[(df["Volatility"] >= 2) & (df["Volatility"] < 5)]
    low_vol = df[df["Volatility"] < 2]
    
    msg += f"• 🔥 Aggressive (≥5&#37;): {len(high_vol)} stocks\n"
    msg += f"• ⚖️ Balanced (2-5&#37;): {len(med_vol)} stocks\n"
    msg += f"• 🛡️ Conservative (&lt;2&#37;): {len(low_vol)} stocks\n"
    
    msg += "\n" + "=" * 35 + "\n"
    msg += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    msg += "💡 <i>Ready for scalping/day trading!</i>"
    
    return msg


def main() -> None:
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
    print(f"📁 Total {len(files)} files ditemukan (data/ + backup/)")
    
    # Load today's data (file terbaru)
    latest_file = files[-1]
    print(f"\n📅 Data Hari Ini: {os.path.basename(latest_file)}")
    
    df_today = load_json(latest_file)
    df_today = prepare(df_today)
    
    # Load historical data (7 hari)
    historical_files = files[-(LOOKBACK_DAYS+1):-1] if len(files) > LOOKBACK_DAYS else files[:-1]
    
    if len(historical_files) >= 3:  # Minimal 3 hari untuk analisis
        print(f"📊 Loading {len(historical_files)} hari historical data...")
        
        df_hist_list: List[pd.DataFrame] = []
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
    
    # Generate Entry Signals (NEW FOR DAYTRADING)
    if len(res) > 0:
        print("🎯 Generating entry/exit signals...")
        res = calculate_entry_signals(res)
        
        buy_signals = res[res["EntrySignal"].str.contains("BUY", na=False)]
        print(f"✅ Found {len(buy_signals)} BUY signals!\n")

    cols = [
        "FileDate","StockCode","StockName",
        "Close","ChangePct","Volatility",
        "Volume","Value","Frequency",
        "High","Low","SpreadPct",
        "AvgTradeSize","VolumeVsShares","ValueRank",
        "DayTradingScore","Category",
        # NEW: Entry signals & risk management
        "EntrySignal","SignalStrength","StopLoss","Target1","Target2","RiskReward",
        # NEW: Support/Resistance
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
        
        # Kirim ke Telegram
        telegram_msg = format_telegram_message_single(res)
        send_telegram(telegram_msg)
    else:
        print(f"[OK] Tidak ada saham yang lolos screening hari ini")
        res[cols].to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
