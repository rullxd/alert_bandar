"""
Day Trading Screener + Broker Analysis
Kombinasi screen_daytrading.py dengan broker top buyer/seller analysis
"""

import os
import glob
import time
import html
import pandas as pd
import requests
from dotenv import load_dotenv
from typing import Any, Dict, List, cast
from screen_daytrading import (
    load_json, prepare, calculate_multi_day_metrics, 
    calculate_daytrading_score, screen, calculate_entry_signals,
    LOOKBACK_DAYS
)

# Load environment variables
load_dotenv()

DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Telegram Config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# GOAPI Config
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

# Broker settings
MIN_TOP_BUYERS = 5
MIN_TOP_SELLERS = 5


def get_current_api_key() -> str:
    global _current_key_index
    if not _goapi_keys:
        raise RuntimeError("No API keys configured in .env")
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
    """Fetch broker summary dengan fallback API keys"""
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
                # Exponential backoff: 2^attempt seconds
                wait_time = min(2 ** attempt, 30)  # Max 30 seconds
                print(f"    Rate limit hit, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            elif e.response.status_code == 401:
                raise RuntimeError(f"Unauthorized: Invalid API key for {symbol}")
            else:
                raise RuntimeError(f"HTTP {e.response.status_code}: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error fetching {symbol}: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error fetching {symbol}: {str(e)}")
    
    raise RuntimeError(f"All {max_attempts} API keys exceeded rate limit for {symbol}")


def extract_brokers_by_side(broker_data: Dict[str, Any], side: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Extract top brokers by side (BUY or SELL)"""
    brokers: List[Dict[str, Any]] = []
    
    # Handle None or empty broker_data
    if not broker_data:
        return []
    
    if not broker_data.get("data") or not broker_data.get("data", {}).get("results"):
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
    """Extract top buyers"""
    return extract_brokers_by_side(broker_data, "BUY", limit)


def extract_top_sellers(broker_data: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    """Extract top sellers"""
    return extract_brokers_by_side(broker_data, "SELL", limit)


def rupiah(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def format_telegram_with_broker(row: Any, buyers: List[Dict[str, Any]], 
                                sellers: List[Dict[str, Any]], trade_date: str) -> str:
    """Format output untuk Telegram dengan info broker"""
    
    stock_code = html.escape(str(row['StockCode']))
    stock_name = html.escape(str(row['StockName'][:20]))
    
    # Header
    msg = "="*30 + "\n"
    msg += f"🎯 <b>{stock_code}</b> - {stock_name}\n"
    msg += "="*30 + "\n\n"
    
    # Signal Info
    signal = row.get('EntrySignal', 'N/A')
    strength = row.get('SignalStrength', 0)
    
    msg += f"📡 <b>SIGNAL:</b> {signal}\n"
    msg += f"💪 <b>Strength:</b> {strength}/100 "
    
    if strength >= 80:
        msg += "🔥 Strong"
    elif strength >= 60:
        msg += "✅ Good"
    elif strength >= 40:
        msg += "○ Fair"
    else:
        msg += "⚠️ Weak"
    msg += "\n\n"
    
    # Price & Entry Setup
    price = row['Close']
    change = row['ChangePct']
    stop_loss = row.get('StopLoss', 0)
    target1 = row.get('Target1', 0)
    target2 = row.get('Target2', 0)
    rr = row.get('RiskReward', 0)
    
    msg += f"💰 <b>Price:</b> Rp {price:,.0f} ({change:+.2f}%)\n"
    msg += f"🛑 <b>Stop Loss:</b> {stop_loss:,.0f} ({((stop_loss-price)/price*100):+.1f}%)\n"
    msg += f"🎯 <b>Target 1:</b> {target1:,.0f} (+{((target1-price)/price*100):.1f}%)\n"
    msg += f"🎯 <b>Target 2:</b> {target2:,.0f} (+{((target2-price)/price*100):.1f}%)\n"
    msg += f"⚖️ <b>Risk/Reward:</b> 1:{rr:.2f}\n\n"
    
    # Multi-day metrics
    if 'VolumeSurge' in row.index:
        surge = row['VolumeSurge']
        position = row.get('PricePosition', 0)
        phase = row.get('MarketPhase', 'Unknown')
        
        msg += f"📊 <b>TECHNICAL</b>\n"
        msg += f"• Volume Surge: {surge:.2f}x "
        if surge >= 2.0:
            msg += "🔥"
        elif surge >= 1.5:
            msg += "⬆️"
        msg += "\n"
        msg += f"• Price Position: {position:.0f}% "
        if position < 30:
            msg += "(Oversold)"
        elif position > 70:
            msg += "(Overbought)"
        else:
            msg += "(Mid-range)"
        msg += "\n"
        msg += f"• Market Phase: {phase}\n\n"
    
    # Calculate totals once for reuse
    total_buy_lot = sum(b["lot"] for b in buyers) if buyers else 0
    total_buy_value = sum(b["value"] for b in buyers) if buyers else 0
    total_sell_lot = sum(s["lot"] for s in sellers) if sellers else 0
    total_sell_value = sum(s["value"] for s in sellers) if sellers else 0
    
    # Buyers Section
    msg += "\n" + "-"*30 + "\n"
    msg += "🟢 <b>TOP BUYERS</b>\n"
    msg += "-"*30 + "\n"
    
    if buyers:
        avg_buy = sum(b["avg"] * b["lot"] for b in buyers) / total_buy_lot if total_buy_lot > 0 else 0
        
        msg += f"<b>Total:</b> {rupiah(int(total_buy_lot))} lot | Rp {rupiah(int(total_buy_value))}\n"
        msg += f"<b>Avg Buy:</b> Rp {avg_buy:,.0f}\n\n"
        
        for i, b in enumerate(buyers[:5], 1):
            diff = price - b["avg"]
            diff_pct = (diff / b["avg"] * 100) if b["avg"] > 0 else 0
            
            delta_emoji = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
            broker_code = html.escape(str(b['broker_code']))
            broker_name = html.escape(str(b['broker_name'][:18]))
            
            msg += f"#{i}. <code>{broker_code}</code> {broker_name}\n"
            msg += f"   Avg: Rp {b['avg']:,.0f} {delta_emoji} ({diff_pct:+.1f}%)\n"
            msg += f"   Vol: {rupiah(int(b['lot']))} lot | Val: Rp {rupiah(int(b['value']))}\n"
        
        msg += "\n"
    else:
        msg += "⚠️ No buyer data available\n\n"
    
    # Sellers Section
    msg += "\n" + "-"*30 + "\n"
    msg += "🔴 <b>TOP SELLERS</b>\n"
    msg += "-"*30 + "\n"
    
    if sellers:
        avg_sell = sum(s["avg"] * s["lot"] for s in sellers) / total_sell_lot if total_sell_lot > 0 else 0
        
        msg += f"<b>Total:</b> {rupiah(int(total_sell_lot))} lot | Rp {rupiah(int(total_sell_value))}\n"
        msg += f"<b>Avg Sell:</b> Rp {avg_sell:,.0f}\n\n"
        
        for i, s in enumerate(sellers[:5], 1):
            diff = price - s["avg"]
            diff_pct = (diff / s["avg"] * 100) if s["avg"] > 0 else 0
            
            delta_emoji = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
            broker_code = html.escape(str(s['broker_code']))
            broker_name = html.escape(str(s['broker_name'][:18]))
            
            msg += f"#{i}. <code>{broker_code}</code> {broker_name}\n"
            msg += f"   Avg: Rp {s['avg']:,.0f} {delta_emoji} ({diff_pct:+.1f}%)\n"
            msg += f"   Vol: {rupiah(int(s['lot']))} lot | Val: Rp {rupiah(int(s['value']))}\n"
        
        msg += "\n"
    else:
        msg += "⚠️ No seller data available\n\n"
    
    # Market Analysis (using pre-calculated totals)
    if buyers and sellers:
        buy_pressure = (total_buy_lot / (total_buy_lot + total_sell_lot) * 100) if (total_buy_lot + total_sell_lot) > 0 else 50
        
        msg += "\n" + "-"*30 + "\n"
        msg += "📊 <b>ANALYSIS</b>\n"
        msg += "-"*30 + "\n"
        msg += f"<b>Buy Pressure:</b> {buy_pressure:.1f}% "
        
        if buy_pressure >= 60:
            msg += "🟢 Strong Buyers"
        elif buy_pressure >= 55:
            msg += "✅ Buyers Dominate"
        elif buy_pressure >= 45:
            msg += "⚖️ Balanced"
        elif buy_pressure >= 40:
            msg += "⚠️ Sellers Edge"
        else:
            msg += "🔴 Strong Sellers"
        
        msg += "\n\n"
    
    msg += f"📅 {trade_date}\n"
    msg += "="*30 + "\n"
    
    return msg


def generate_summary_statistics(buy_signals: pd.DataFrame, screened_count: int, 
                               broker_success_count: int, trade_date: str) -> str:
    """Generate comprehensive summary statistics"""
    
    total_signals = len(buy_signals)
    
    msg = "="*30 + "\n"
    msg += "📊 <b>SUMMARY STATISTICS</b>\n"
    msg += "="*30 + "\n\n"
    
    # Signal Overview
    msg += "<b>📈 SIGNAL OVERVIEW</b>\n"
    msg += "-"*30 + "\n"
    msg += f"• Total Signals: <b>{total_signals}</b>\n"
    msg += f"• Stocks Screened: {screened_count}\n"
    msg += f"• With Broker Data: {broker_success_count}\n"
    msg += f"• Date: {trade_date}\n\n"
    
    # Signal Type Breakdown
    momentum_signals = buy_signals[buy_signals['EntrySignal'].str.contains('MOMENTUM', na=False)]
    reversal_signals = buy_signals[buy_signals['EntrySignal'].str.contains('REVERSAL', na=False)]
    
    msg += "<b>🎯 SIGNAL TYPE</b>\n"
    msg += "-"*30 + "\n"
    msg += f"• MOMENTUM: {len(momentum_signals)} ({len(momentum_signals)/total_signals*100:.0f}%)\n"
    msg += f"• REVERSAL: {len(reversal_signals)} ({len(reversal_signals)/total_signals*100:.0f}%)\n\n"
    
    # Signal Strength Distribution
    avg_strength = 0.0  # Default value
    if 'SignalStrength' in buy_signals.columns:
        avg_strength = buy_signals['SignalStrength'].mean()
        max_strength = buy_signals['SignalStrength'].max()
        min_strength = buy_signals['SignalStrength'].min()
        
        strong_signals = len(buy_signals[buy_signals['SignalStrength'] >= 80])
        good_signals = len(buy_signals[(buy_signals['SignalStrength'] >= 60) & (buy_signals['SignalStrength'] < 80)])
        fair_signals = len(buy_signals[buy_signals['SignalStrength'] < 60])
        
        msg += "<b>💪 SIGNAL STRENGTH</b>\n"
        msg += "-"*30 + "\n"
        msg += f"• Average: <b>{avg_strength:.0f}/100</b>\n"
        msg += f"• Range: {min_strength:.0f} - {max_strength:.0f}\n"
        msg += f"• 🔥 Strong (&gt;=80): {strong_signals}\n"
        msg += f"• ✅ Good (60-79): {good_signals}\n"
        msg += f"• ○ Fair (&lt;60): {fair_signals}\n\n"
    
    # Price Analysis
    if 'Close' in buy_signals.columns:
        avg_price = buy_signals['Close'].mean()
        median_price = buy_signals['Close'].median()
        min_price = buy_signals['Close'].min()
        max_price = buy_signals['Close'].max()
        
        msg += "<b>💰 PRICE ANALYSIS</b>\n"
        msg += "-"*30 + "\n"
        msg += f"• Average: Rp {avg_price:,.0f}\n"
        msg += f"• Median: Rp {median_price:,.0f}\n"
        msg += f"• Range: Rp {min_price:,.0f} - Rp {max_price:,.0f}\n\n"
    
    # Risk/Reward Statistics
    if 'RiskReward' in buy_signals.columns:
        avg_rr = buy_signals['RiskReward'].mean()
        max_rr = buy_signals['RiskReward'].max()
        min_rr = buy_signals['RiskReward'].min()
        
        msg += "<b>⚖️ RISK/REWARD</b>\n"
        msg += "-"*30 + "\n"
        msg += f"• Average R:R: <b>1:{avg_rr:.2f}</b>\n"
        msg += f"• Best R:R: 1:{max_rr:.2f}\n"
        msg += f"• Range: 1:{min_rr:.2f} - 1:{max_rr:.2f}\n\n"
    
    # Technical Metrics
    if 'VolumeSurge' in buy_signals.columns:
        avg_surge = buy_signals['VolumeSurge'].mean()
        max_surge = buy_signals['VolumeSurge'].max()
        high_surge = len(buy_signals[buy_signals['VolumeSurge'] >= 2.0])
        
        msg += "<b>📊 TECHNICAL METRICS</b>\n"
        msg += "-"*30 + "\n"
        msg += f"• Avg Volume Surge: {avg_surge:.2f}x\n"
        msg += f"• Max Volume Surge: {max_surge:.2f}x\n"
        msg += f"• High Surge (&gt;=2x): {high_surge} signals\n\n"
    
    # Market Phase Distribution
    if 'MarketPhase' in buy_signals.columns:
        phase_counts = buy_signals['MarketPhase'].value_counts()
        
        msg += "<b>🌊 MARKET PHASE</b>\n"
        msg += "-"*30 + "\n"
        for phase, count in phase_counts.items():
            pct = (count / total_signals * 100)
            phase_escaped = html.escape(str(phase))
            msg += f"• {phase_escaped}: {count} ({pct:.0f}%)\n"
        msg += "\n"
    
    # Top Performers by Signal Strength
    if 'SignalStrength' in buy_signals.columns and total_signals > 0:
        top_n = min(3, total_signals)
        top_signals = buy_signals.nlargest(top_n, 'SignalStrength')[['StockCode', 'SignalStrength', 'Close', 'EntrySignal']]
        
        msg += f"<b>🏆 TOP {top_n} SIGNALS</b>\n"
        msg += "-"*30 + "\n"
        for i, (_, row) in enumerate(top_signals.iterrows(), 1):
            signal_type = 'MOMENTUM' if 'MOMENTUM' in row['EntrySignal'] else 'REVERSAL'
            stock_code_escaped = html.escape(str(row['StockCode']))
            msg += f"{i}. <b>{stock_code_escaped}</b> - Strength {row['SignalStrength']:.0f}/100\n"
            msg += f"   {signal_type} @ Rp {row['Close']:,.0f}\n"
        msg += "\n"
    
    # Expected Targets
    if 'Target1' in buy_signals.columns and 'Target2' in buy_signals.columns:
        avg_t1_pct = ((buy_signals['Target1'] - buy_signals['Close']) / buy_signals['Close'] * 100).mean()
        avg_t2_pct = ((buy_signals['Target2'] - buy_signals['Close']) / buy_signals['Close'] * 100).mean()
        
        msg += "<b>🎯 TARGET EXPECTATIONS</b>\n"
        msg += "-"*30 + "\n"
        msg += f"• Average Target 1: +{avg_t1_pct:.1f}%\n"
        msg += f"• Average Target 2: +{avg_t2_pct:.1f}%\n\n"
    
    # Trading Recommendations
    msg += "<b>💡 TRADING RECOMMENDATIONS</b>\n"
    msg += "-"*30 + "\n"
    
    if total_signals >= 5:
        msg += "• 🟢 Multiple opportunities available\n"
        msg += "• Focus on signals with strength &gt;=70\n"
    elif total_signals >= 3:
        msg += "• 🟡 Moderate opportunities\n"
        msg += "• Select best risk/reward ratios\n"
    else:
        msg += "• 🟠 Limited opportunities\n"
        msg += "• Be selective with entries\n"
    
    if 'SignalStrength' in buy_signals.columns:
        if avg_strength >= 70:
            msg += "• 💪 Strong signal quality overall\n"
        elif avg_strength >= 60:
            msg += "• ✅ Good signal quality\n"
        else:
            msg += "• ⚠️ Exercise caution - lower quality\n"
    
    if len(momentum_signals) > len(reversal_signals):
        msg += "• 📈 Momentum dominant - trend following\n"
    elif len(reversal_signals) > len(momentum_signals):
        msg += "• 🔄 Reversal dominant - counter-trend\n"
    else:
        msg += "• ⚖️ Balanced signals - diversified\n"
    
    msg += "\n" + "-"*30 + "\n"
    msg += "⚠️ Always use stop loss and manage risk\n"
    
    return msg


def send_telegram(message: str) -> bool:
    """Kirim pesan ke Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARNING] Telegram not configured")
        return False
    
    # Telegram message length limit is 4096 characters
    MAX_LENGTH = 4096
    if len(message) > MAX_LENGTH:
        print(f"[WARNING] Message too long ({len(message)} chars), truncating to {MAX_LENGTH}")
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
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] Telegram HTTP error {e.response.status_code}: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Telegram network error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")
        return False


def main() -> None:
    """Main: Day trading screening + broker analysis"""
    
    print("="*70)
    print("🎯 DAY TRADING SCREENER + BROKER ANALYSIS")
    print("="*70)
    
    # Load data
    files_data = glob.glob(os.path.join(DATA_DIR, "idx_stock_*.json"))
    files_backup = glob.glob(os.path.join("backup", "idx_stock_*.json"))
    files = sorted(files_data + files_backup)
    
    if not files:
        print("[ERROR] No data files found")
        return
    
    # Load today's data
    latest_file = files[-1]
    print(f"\n📅 Processing: {os.path.basename(latest_file)}")
    
    df_today = load_json(latest_file)
    df_today = prepare(df_today)
    
    # Load historical for multi-day analysis
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
    print("🔍 Running day trading screener...")
    df_screened = screen(df_today)
    print(f"✅ {len(df_screened)} stocks passed screening\n")
    
    if len(df_screened) == 0:
        print("No stocks passed screening criteria")
        return
    
    # Generate signals
    print("🎯 Generating entry signals...")
    df_signals = calculate_entry_signals(df_screened)
    
    # Filter BUY signals
    buy_signals = df_signals[df_signals["EntrySignal"].str.contains("BUY", na=False)]
    print(f"✅ Found {len(buy_signals)} BUY signals\n")
    
    if len(buy_signals) == 0:
        print("No BUY signals generated")
        return
    
    # Get trade date with validation
    if len(df_today) == 0 or 'FileDate' not in df_today.columns:
        print("[ERROR] Cannot extract trade date from empty data")
        return
    
    trade_date = df_today['FileDate'].iloc[0]
    trade_date_str = trade_date.strftime("%Y-%m-%d")
    
    # Process each BUY signal with broker data
    print("="*70)
    print("📊 FETCHING BROKER DATA")
    print("="*70 + "\n")
    
    results: List[str] = []
    broker_success_count = 0
    
    for idx, row in buy_signals.iterrows():
        stock_code = row['StockCode']
        
        print(f"[{idx+1}/{len(buy_signals)}] Fetching {stock_code}...")
        
        try:
            # Fetch broker data
            broker_data = fetch_broker_summary(stock_code, trade_date_str)
            buyers = extract_top_buyers(broker_data, MIN_TOP_BUYERS)
            sellers = extract_top_sellers(broker_data, MIN_TOP_SELLERS)
            
            if buyers or sellers:
                print(f"  ✓ Buyers: {len(buyers)}, Sellers: {len(sellers)}")
                broker_success_count += 1
                
                # Format message
                msg = format_telegram_with_broker(row, buyers, sellers, trade_date_str)
                results.append(msg)
                
            else:
                print(f"  ⚠️ No broker data available")
            
            time.sleep(0.5)  # Rate limiting
            
        except RuntimeError as e:
            print(f"  ✗ Error: {str(e)}")
            time.sleep(1)
            continue
        except Exception as e:
            print(f"  ✗ Unexpected error: {type(e).__name__}: {str(e)[:100]}")
            time.sleep(1)
            continue
    
    # Send to Telegram
    if results:
        print("\n" + "="*70)
        print("📱 SENDING TO TELEGRAM")
        print("="*70 + "\n")
        
        # Send header
        header = f"🎯 <b>DAY TRADING SIGNALS + BROKER</b>\n" + \
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n" + \
                f"<b>Date:</b> {trade_date_str}\n" + \
                f"<b>Total Signals:</b> {len(results)}\n" + \
                f"<b>Screened:</b> {len(df_screened)} stocks\n" + \
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        send_telegram(header)
        time.sleep(1)
        
        # Send each signal
        for i, msg in enumerate(results, 1):
            formatted = f"<b>[Signal {i}/{len(results)}]</b>\n\n{msg}"
            if send_telegram(formatted):
                print(f"  ✓ [{i}/{len(results)}] Sent")
            else:
                print(f"  ✗ [{i}/{len(results)}] Failed")
            time.sleep(0.5)
        
        print(f"\n✅ Sent {len(results)} signals to Telegram")
        
        # Generate and send summary statistics
        print("\n📊 Generating summary statistics...")
        summary = generate_summary_statistics(buy_signals, len(df_screened), 
                                             broker_success_count, trade_date_str)
        
        time.sleep(1)
        if send_telegram(summary):
            print("✅ Summary statistics sent")
        else:
            print("✗ Summary statistics failed")
    
    print("\n" + "="*70)
    print("✅ COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
