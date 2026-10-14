import os
from datetime import datetime
import pytz
import requests
import pandas as pd
import yfinance as yf

# --- CONFIGURATION ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Top FnO Stocks & Major Indices List
FNO_SYMBOLS = [
    "^NSEI", "^NSEBANK", "^CNXIT",
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS",
    "AXISBANK.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ASIANPAINT.NS", "NTPC.NS", "POWERGRID.NS", "TATASTEEL.NS",
    "BAJAJ-AUTO.NS", "M&M.NS", "HCLTECH.NS", "WIPRO.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "GRASIM.NS", "JSWSTEEL.NS", "TECHM.NS", "INDUSINDBK.NS",
    "TATAMOTORS.NS", "ONGC.NS", "COALINDIA.NS", "BPCL.NS", "HEROMOTOCO.NS",
    "EICHERMOT.NS", "BRITANNIA.NS", "CIPLA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "APOLLOHOSP.NS", "TATACONSUM.NS", "SBILIFE.NS", "HDFCLIFE.NS", "BAJAJFINSV.NS",
    "UPL.NS", "VEDL.NS", "HINDALCO.NS", "SRF.NS", "CHOLAFIN.NS", "DLF.NS",
    "INDIGO.NS", "PIDILITIND.NS", "NAUKRI.NS", "GODREJCP.NS", "DABUR.NS",
    "MARICO.NS", "COLPAL.NS", "SBICARD.NS", "MUTHOOTFIN.NS", "LUPIN.NS",
    "AUROPHARMA.NS", "BIOCON.NS", "CANBK.NS", "PNB.NS", "BANKBARODA.NS",
    "JINDALSTEL.NS", "SAIL.NS", "TATACOMM.NS", "HINDPETRO.NS", "IOC.NS",
    "MCDOWELL-N.NS", "ZYDUSLIFE.NS", "IPCALAB.NS", "ALKEM.NS", "LALPATHLAB.NS",
    "AMBUJACEM.NS", "ACC.NS", "SHREECEM.NS", "ICICIGI.NS", "ICICIPRULI.NS",
    "MAXHEALTH.NS", "POLYCAB.NS", "TRENT.NS", "PERSISTENT.NS", "COFORGE.NS",
    "MPHASIS.NS", "LTIM.NS", "HAVELLS.NS", "ABFRL.NS", "PFC.NS", "RECLTD.NS"
]

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Error: Telegram credentials missing!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": False
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Telegram Response: {response.text}")
    except Exception as e:
        print(f"Telegram Error: {e}")

def scan_symbol(symbol):
    try:
        df = yf.download(symbol, period="2d", interval="15m", progress=False)
        if df.empty or len(df) < 25:
            return

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        candle_time = df.index[-2]
        if candle_time.tzinfo is None:
            candle_time = pytz.utc.localize(candle_time)
        ist = pytz.timezone('Asia/Kolkata')
        c_time_ist = candle_time.astimezone(ist)

        lookback = 20
        recent_df = df.iloc[-26:-2]
        resistance = recent_df['High'].max()
        support = recent_df['Low'].min()
        vol_ma = recent_df['Volume'].mean()
        
        close_price = float(df['Close'].iloc[-2])
        volume = float(df['Volume'].iloc[-2])
        is_high_volume = volume > (vol_ma * 1.3)

        signal = None
        if close_price > resistance and is_high_volume:
            signal = "🚀 BUY CE (Resistance Breakout)"
            level_val = resistance
        elif close_price < support and is_high_volume:
            signal = "📉 BUY PE (Support Breakdown)"
            level_val = support
        else:
            return

        clean_symbol = symbol.replace(".NS", "").replace("^", "")
        tv_url = f"https://in.tradingview.com/chart/?symbol=NSE:{clean_symbol}"
        
        date_str = c_time_ist.strftime("%d-%b-%Y")
        time_str = c_time_ist.strftime("%I:%M %p")

        alert_text = (
            f"⚡ <b>15-MIN MOMENTUM SIGNAL (TEST)</b>\n\n"
            f"📌 <b>Symbol:</b> {clean_symbol}\n"
            f"🎯 <b>Signal:</b> {signal}\n"
            f"💰 <b>Price:</b> ₹{close_price:.2f}\n"
            f"📊 <b>Level Hit:</b> ₹{level_val:.2f}\n"
            f"⏱ <b>Candle Time:</b> {time_str} ({date_str})\n\n"
            f"🔗 <a href='{tv_url}'>Open Chart in TradingView</a>"
        )
        
        send_telegram_message(alert_text)
        print(f"Alert sent for {clean_symbol}")

    except Exception as e:
        print(f"Error scanning {symbol}: {e}")

def main():
    print("Force scanning stocks for testing...")
    
    # Send a startup test message to confirm Telegram is connected
    send_telegram_message("🤖 <b>Scanner Bot Started Successfully!</b> Checking latest closed candles...")

    # Scan a few top stocks or all
    for symbol in FNO_SYMBOLS[:10]: # Testing with top 10 first to save time
        scan_symbol(symbol)
        
    print("Scan completed.")

if __name__ == "__main__":
    main()
