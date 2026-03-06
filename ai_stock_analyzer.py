"""
AI Stock Analyzer using Google Gemini
Analyze screened stocks and broker data with AI insight
"""

import os
import warnings
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv

# Suppress ALL FutureWarnings (including from google.generativeai)
warnings.filterwarnings('ignore', category=FutureWarning)

# Import Gemini (warning suppressed above)
try:
    import google.generativeai as genai  # type: ignore
    gemini_available = True
except ImportError:
    genai = None  # type: ignore
    gemini_available = False

load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if GEMINI_API_KEY and gemini_available:
    try:
        genai.configure(api_key=GEMINI_API_KEY)  # type: ignore
    except Exception as e:
        print(f"[WARNING] Failed to configure Gemini API: {e}")

# Model configuration  
# Available models dari list_models():
# - gemini-flash-latest (⚡ fastest, recommended)
# - gemini-pro-latest (🧠 more powerful)
# - gemini-2.5-flash (📦 specific version)
MODEL_NAME = "gemini-flash-latest"  # Always use latest flash version


def create_analysis_prompt(stock_data: Dict[str, Any], broker_data: Optional[Dict[str, Any]] = None) -> str:
    """Create detailed prompt for AI analysis"""
    
    prompt = f"""Anda adalah seorang expert technical analyst untuk pasar saham Indonesia (IDX).
Analisis data saham berikut dan berikan insight trading yang actionable.

📊 DATA SAHAM:
- Ticker: {stock_data.get('StockCode', 'N/A')}
- Nama: {stock_data.get('StockName', 'N/A')}
- Harga: Rp {stock_data.get('Close', 0):,.0f}
- Perubahan: {stock_data.get('ChangePct', 0):+.2f}%
- Volume: {stock_data.get('Volume', 0):,.0f} lot
- Value: Rp {stock_data.get('Value', 0):,.0f}
- Frequency: {stock_data.get('Frequency', 0):,.0f}x

📈 TECHNICAL INDICATORS:
- Signal: {stock_data.get('EntrySignal', 'N/A')}
- Signal Strength: {stock_data.get('SignalStrength', 0)}/100
- Volatility: {stock_data.get('Volatility', 0):.2f}%
- Day Trading Score: {stock_data.get('DayTradingScore', 0):.1f}
- Stop Loss: Rp {stock_data.get('StopLoss', 0):,.0f}
- Target 1: Rp {stock_data.get('Target1', 0):,.0f}
- Target 2: Rp {stock_data.get('Target2', 0):,.0f}
- Risk/Reward: 1:{stock_data.get('RiskReward', 0):.2f}

📊 MULTI-DAY METRICS:
- Volume Surge: {stock_data.get('VolumeSurge', 0):.2f}x
- Price Position: {stock_data.get('PricePosition', 0):.1f}%
- Liquid Days (7d): {stock_data.get('LiquidDays', 0)}/7
- Market Phase: {stock_data.get('MarketPhase', 'Unknown')}
"""

    # Add broker data if available
    if broker_data and 'buyers' in broker_data and 'sellers' in broker_data:
        buyers = broker_data['buyers']
        sellers = broker_data['sellers']
        
        total_buy = 0
        total_sell = 0
        
        if buyers:
            total_buy = sum(b['value'] for b in buyers)
            prompt += f"\n🟢 TOP BUYERS:\n"
            prompt += f"- Total Value: Rp {total_buy:,.0f}\n"
            for i, b in enumerate(buyers[:3], 1):
                prompt += f"  {i}. {b['broker_code']} - {b['broker_name']}: {b['lot']:,.0f} lot, Avg: Rp {b['avg']:,.0f}\n"
        
        if sellers:
            total_sell = sum(s['value'] for s in sellers)
            prompt += f"\n🔴 TOP SELLERS:\n"
            prompt += f"- Total Value: Rp {total_sell:,.0f}\n"
            for i, s in enumerate(sellers[:3], 1):
                prompt += f"  {i}. {s['broker_code']} - {s['broker_name']}: {s['lot']:,.0f} lot, Avg: Rp {s['avg']:,.0f}\n"
        
        if buyers and sellers and total_sell > 0:
            ratio = total_buy / total_sell
            prompt += f"\n⚖️ Buy/Sell Ratio: {ratio:.2f}x\n"

    prompt += """

🎯 TUGAS ANDA:
Berikan analisis singkat (max 250 kata) yang mencakup:

1. **Market Sentiment** (1-2 kalimat)
   - Apakah bullish/bearish/neutral?
   - Kenapa?

2. **Key Insight** (2-3 kalimat)
   - Pattern atau pola yang terdeteksi
   - Aktivitas broker (jika ada data)
   - Volume dan likuiditas

3. **Risk Assessment** (1-2 kalimat)
   - Risk level: Low/Medium/High
   - Faktor risiko utama

4. **Trading Recommendation** (2-3 kalimat)
   - Apakah layak entry?
   - Strategy (momentum/reversal/hold)
   - Key level to watch

5. **Confidence Score** (1 angka)
   - Berikan skor 0-100 untuk confidence level analisis ini

Format output dengan struktur jelas dan emoji untuk readability.
Fokus pada actionable insight, bukan teori.
"""

    return prompt


def analyze_stock_with_ai(stock_data: Dict[str, Any], broker_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Analyze stock using Gemini AI"""
    
    if not GEMINI_API_KEY:
        return {
            "success": False,
            "error": "GEMINI_API_KEY not configured in .env",
            "analysis": None
        }
    
    if not gemini_available or genai is None:
        return {
            "success": False,
            "error": "google-generativeai package not installed. Run: pip install google-generativeai",
            "analysis": None
        }
    
    try:
        # Create model
        model = genai.GenerativeModel(MODEL_NAME)  # type: ignore
        
        # Generate prompt
        prompt = create_analysis_prompt(stock_data, broker_data)
        
        # Generate response
        response = model.generate_content(prompt)  # type: ignore
        
        # Extract text
        analysis_text = response.text
        
        return {
            "success": True,
            "analysis": analysis_text,
            "stock_code": stock_data.get('StockCode', 'N/A'),
            "model": MODEL_NAME
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "analysis": None,
            "stock_code": stock_data.get('StockCode', 'N/A')
        }


def format_ai_analysis_for_telegram(stock_data: Dict[str, Any], ai_result: Dict[str, Any]) -> str:
    """Format AI analysis for Telegram"""
    
    stock_code = stock_data.get('StockCode', 'N/A')
    
    msg = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"🤖 <b>AI ANALYSIS: {stock_code}</b>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    if ai_result['success']:
        msg += ai_result['analysis']
        msg += f"\n\n<i>🤖 Powered by {ai_result['model']}</i>"
    else:
        msg += f"⚠️ <i>AI analysis failed: {ai_result.get('error', 'Unknown error')}</i>"
    
    return msg


def batch_analyze_stocks(stocks_list: List[Dict[str, Any]], 
                         broker_data_list: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Analyze multiple stocks in batch"""
    
    results: List[Dict[str, Any]] = []
    
    for i, stock in enumerate(stocks_list):
        print(f"[{i+1}/{len(stocks_list)}] Analyzing {stock.get('StockCode', 'N/A')}...")
        
        # Get corresponding broker data if available
        broker_data = None
        if broker_data_list and i < len(broker_data_list):
            broker_data = broker_data_list[i]
        
        # Analyze
        result = analyze_stock_with_ai(stock, broker_data)
        results.append(result)
        
        if result['success']:
            print(f"  ✓ Analysis complete")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown')}")
    
    return results


# ==================================================
# TEST STANDALONE
# ==================================================
def test_ai_analyzer():
    """Test AI analyzer with sample data"""
    
    print("="*60)
    print("🧪 TESTING AI STOCK ANALYZER")
    print("="*60)
    
    # Sample stock data
    sample_stock: Dict[str, Any] = {
        'StockCode': 'BBCA',
        'StockName': 'Bank Central Asia Tbk',
        'Close': 9500,
        'ChangePct': 2.35,
        'Volume': 25000000,
        'Value': 237500000000,
        'Frequency': 5234,
        'EntrySignal': 'BUY',
        'SignalStrength': 78,
        'Volatility': 1.85,
        'DayTradingScore': 82.5,
        'StopLoss': 9300,
        'Target1': 9700,
        'Target2': 9850,
        'RiskReward': 2.15,
        'VolumeSurge': 1.85,
        'PricePosition': 68.5,
        'LiquidDays': 6,
        'MarketPhase': 'Accumulation'
    }
    
    # Sample broker data
    sample_broker: Dict[str, List[Dict[str, Union[str, int, float]]]] = {
        'buyers': [
            {'broker_code': 'YP', 'broker_name': 'Yuanta Sekuritas', 'lot': 5000000, 'value': 47500000000, 'avg': 9500},
            {'broker_code': 'AK', 'broker_name': 'Asiatrust Securities', 'lot': 3200000, 'value': 30400000000, 'avg': 9500},
            {'broker_code': 'MG', 'broker_name': 'Mandiri Sekuritas', 'lot': 2800000, 'value': 26600000000, 'avg': 9500}
        ],
        'sellers': [
            {'broker_code': 'NG', 'broker_name': 'NISP Sekuritas', 'lot': 3500000, 'value': 33250000000, 'avg': 9500},
            {'broker_code': 'UB', 'broker_name': 'UOB Kay Hian', 'lot': 2900000, 'value': 27550000000, 'avg': 9500}
        ]
    }
    
    print("\n📊 Analyzing sample stock...")
    result = analyze_stock_with_ai(sample_stock, sample_broker)
    
    print("\n" + "="*60)
    print("📋 RESULTS")
    print("="*60)
    
    if result['success']:
        print("\n✅ Analysis successful!\n")
        print(result['analysis'])
    else:
        print(f"\n❌ Analysis failed: {result['error']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    test_ai_analyzer()
