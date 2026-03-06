"""
Day Trading Screener + Broker Analysis + AI Insight
Enhanced version dengan Gemini AI analysis
"""

import os
import sys
import time
import pandas as pd
from typing import Any, Dict, List
from dotenv import load_dotenv

# Import existing modules
from daytrading_with_broker import (
    load_json, prepare, calculate_multi_day_metrics,
    calculate_daytrading_score, screen, calculate_entry_signals,
    fetch_broker_summary, extract_top_buyers, extract_top_sellers,
    send_telegram, rupiah, LOOKBACK_DAYS, DATA_DIR,
    MIN_TOP_BUYERS, MIN_TOP_SELLERS
)

# Import AI analyzer
from ai_stock_analyzer import analyze_stock_with_ai, format_ai_analysis_for_telegram

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
USE_AI = bool(GEMINI_API_KEY)  # Auto-detect if AI enabled


def format_enhanced_telegram(row: Any, buyers: List[Dict[str, Any]], 
                             sellers: List[Dict[str, Any]], 
                             trade_date: str,
                             ai_analysis: Dict[str, Any] = None) -> str:
    """Format message dengan AI insight"""
    
    stock_code = row['StockCode']
    stock_name = row['StockName'][:20]
    
    # Header
    msg = "━"*35 + "\n"
    msg += f"🎯 <b>{stock_code}</b> - {stock_name}\n"
    msg += "━"*35 + "\n\n"
    
    # Signal Info
    signal = row.get('EntrySignal', 'N/A')
    strength = row.get('SignalStrength', 0)
    
    msg += f"📡 <b>SIGNAL:</b> {signal} | "
    if strength >= 80:
        msg += "🔥 Strong"
    elif strength >= 60:
        msg += "✅ Good"
    else:
        msg += "○ Fair"
    msg += f" ({strength}/100)\n\n"
    
    # Price & Setup
    price = row['Close']
    change = row['ChangePct']
    msg += f"💰 <b>Price:</b> Rp {price:,.0f} ({change:+.2f}%)\n"
    msg += f"🛑 <b>Stop Loss:</b> {row['StopLoss']:,.0f}\n"
    msg += f"🎯 <b>Target 1:</b> {row['Target1']:,.0f} | T2: {row['Target2']:,.0f}\n"
    msg += f"⚖️ <b>R/R:</b> 1:{row['RiskReward']:.2f}\n\n"
    
    # Technical
    if 'VolumeSurge' in row.index:
        msg += f"📊 <b>Vol Surge:</b> {row['VolumeSurge']:.2f}x | "
        msg += f"<b>Position:</b> {row['PricePosition']:.0f}%\n"
        msg += f"<b>Phase:</b> {row['MarketPhase']}\n\n"
    
    # Broker Summary (simplified)
    if buyers or sellers:
        total_buy = sum(b['value'] for b in buyers) if buyers else 0
        total_sell = sum(s['value'] for s in sellers) if sellers else 0
        
        msg += "─"*35 + "\n"
        msg += "🏦 <b>BROKER ACTIVITY</b>\n"
        msg += "─"*35 + "\n"
        
        if buyers:
            msg += f"🟢 <b>Top Buyer:</b> {buyers[0]['broker_code']}\n"
            msg += f"   Val: Rp {rupiah(int(buyers[0]['value']))} "
            msg += f"({buyers[0]['lot']:,.0f} lot)\n"
        
        if sellers:
            msg += f"🔴 <b>Top Seller:</b> {sellers[0]['broker_code']}\n"
            msg += f"   Val: Rp {rupiah(int(sellers[0]['value']))} "
            msg += f"({sellers[0]['lot']:,.0f} lot)\n"
        
        if total_buy > 0 and total_sell > 0:
            ratio = total_buy / total_sell
            sentiment = "🟢 BULLISH" if ratio > 1.2 else "🔴 BEARISH" if ratio < 0.8 else "⚪ NEUTRAL"
            msg += f"\n⚖️ <b>B/S Ratio:</b> {ratio:.2f}x {sentiment}\n"
        
        msg += "\n"
    
    # AI Analysis (if available)
    if ai_analysis and ai_analysis.get('success'):
        msg += "━"*35 + "\n"
        msg += "🤖 <b>AI INSIGHT</b>\n"
        msg += "━"*35 + "\n"
        msg += ai_analysis['analysis']
        msg += "\n"
    
    msg += "━"*35
    
    return msg


def main():
    """Main execution with AI enhancement"""
    
    print("="*70)
    print("🤖 DAY TRADING SCREENER + BROKER + AI")
    print("="*70)
    
    if USE_AI:
        print("✅ AI Analysis: ENABLED (Gemini)")
    else:
        print("⚠️  AI Analysis: DISABLED (set GEMINI_API_KEY to enable)")
    
    print("\n🔍 Loading latest data files...")
    
    # Get latest N+1 files (N historical + 1 today)
    all_files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("idx_stock")], reverse=True)
    
    if len(all_files) < (LOOKBACK_DAYS + 1):
        print(f"[ERROR] Need at least {LOOKBACK_DAYS + 1} data files")
        return
    
    files_to_load = all_files[:(LOOKBACK_DAYS + 1)]  # Load N historical + 1 today
    print(f"Found {len(files_to_load)} files ({LOOKBACK_DAYS} historical + 1 today)")
    
    # Load data
    dfs = []
    for fname in reversed(files_to_load):
        path = os.path.join(DATA_DIR, fname)
        df = load_json(path)
        df = prepare(df)
        dfs.append(df)
        print(f"  ✓ {fname}")
    
    if not dfs:
        print("[ERROR] No data loaded")
        return
    
    # Separate today and historical data
    df_today = dfs[-1]  # Last file is today
    
    # Multi-day analysis
    if len(dfs) > 1:
        print("\n📊 Calculating multi-day metrics...")
        df_historical = pd.concat(dfs[:-1], ignore_index=True)  # All except today
        df_today = calculate_multi_day_metrics(df_today, df_historical)
        print("✅ Multi-day analysis complete")
    else:
        print("\n⚠️  Only 1 day data, skipping multi-day analysis")
    
    # Calculate scores
    print("🎯 Calculating day trading scores...")
    df_today["DayTradingScore"] = calculate_daytrading_score(df_today)
    
    # Screen
    print("🔍 Screening stocks...")
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
    
    # Get trade date
    trade_date = df_today['FileDate'].iloc[0]
    trade_date_str = trade_date.strftime("%Y-%m-%d")
    
    # Process each BUY signal
    print("="*70)
    print("📊 PROCESSING SIGNALS WITH AI")
    print("="*70 + "\n")
    
    results: List[str] = []
    success_count = 0
    
    for idx, row in buy_signals.iterrows():
        stock_code = row['StockCode']
        
        print(f"[{idx+1}/{len(buy_signals)}] Processing {stock_code}...")
        
        try:
            # Fetch broker data
            broker_data = fetch_broker_summary(stock_code, trade_date_str)
            buyers = extract_top_buyers(broker_data, MIN_TOP_BUYERS)
            sellers = extract_top_sellers(broker_data, MIN_TOP_SELLERS)
            
            print(f"  ✓ Broker: {len(buyers)} buyers, {len(sellers)} sellers")
            
            # AI Analysis (if enabled)
            ai_result = None
            if USE_AI:
                print(f"  🤖 Analyzing with AI...")
                
                # Prepare data for AI
                stock_dict = row.to_dict()
                broker_dict = {'buyers': buyers, 'sellers': sellers}
                
                ai_result = analyze_stock_with_ai(stock_dict, broker_dict)
                
                if ai_result['success']:
                    print(f"  ✓ AI analysis complete")
                else:
                    print(f"  ⚠️  AI analysis failed: {ai_result.get('error', 'Unknown')[:50]}")
            
            # Format message
            msg = format_enhanced_telegram(row, buyers, sellers, trade_date_str, ai_result)
            results.append(msg)
            success_count += 1
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:100]}")
            continue
    
    # Send to Telegram
    if results:
        print("\n" + "="*70)
        print("📱 SENDING TO TELEGRAM")
        print("="*70 + "\n")
        
        # Header
        ai_status = "ENABLED ✅" if USE_AI else "DISABLED"
        header = f"🤖 <b>AI-POWERED TRADING SIGNALS</b>\n" + \
                 f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n" + \
                 f"<b>Date:</b> {trade_date_str}\n" + \
                 f"<b>Signals:</b> {len(results)}\n" + \
                 f"<b>AI Analysis:</b> {ai_status}\n" + \
                 f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        send_telegram(header)
        time.sleep(1)
        
        # Send each signal
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
