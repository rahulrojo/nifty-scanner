import os
import requests
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import pytz

# --- TELEGRAM BOT CONFIGURATION ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_msg(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Error: Telegram credentials not set in environment variables.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

# --- F&O SYMBOLS (INDICES + TOP F&O STOCKS) ---
SYMBOLS = [
    # Major Indices
    "^NSEI", "^NSEBANK", "^FINNIFTY",
    # Top F&O Stocks (NSE)
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
    "AXISBANK.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "BAJFINANCE.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
    "HCLTECH.NS", "M&M.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS",
    "WIPRO.NS", "ULTRACEMCO.NS", "JSWSTEEL.NS", "GRASIM.NS", "HINDALCO.NS"
]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_symbol(symbol):
    try:
        # Download 15-minute data (last 5 days to cover 75 candles lookback)
        df = yf.download(symbol, period="5d", interval="15m", progress=False)
        if df.empty or len(df) < 75:
            return

        # Flatten multi-index columns if returned by yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # --- Indicator Parameters ---
        pd_val = 15
        bbl = 20
        mult = 1.8
        lb = 75
        ph = 0.92
        ema_len = 9

        # --- WVF Bottom (Buy CE Base) ---
        df['HighestClose'] = df['Close'].rolling(window=pd_val).max()
        df['WVF'] = ((df['HighestClose'] - df['Low']) / df['HighestClose']) * 100.0
        df['MidLine'] = df['WVF'].rolling(window=bbl).mean()
        df['sDev'] = df['WVF'].rolling(window=bbl).std() * mult
        df['UpperBand'] = df['MidLine'] + df['sDev']
        df['RangeHigh'] = df['WVF'].rolling(window=lb).max() * ph
        df['isPanic'] = (df['WVF'] >= df['UpperBand']) | (df['WVF'] >= df['RangeHigh'])

        # --- WVF Top (Buy PE Base) ---
        df['LowestClose'] = df['Close'].rolling(window=pd_val).min()
        df['WVFTop'] = ((df['High'] - df['LowestClose']) / df['LowestClose']) * 100.0
        df['MidLineTop'] = df['WVFTop'].rolling(window=bbl).mean()
        df['sDevTop'] = df['WVFTop'].rolling(window=bbl).std() * mult
        df['UpperBandTop'] = df['MidLineTop'] + df['sDevTop']
        df['RangeHighTop'] = df['WVFTop'].rolling(window=lb).max() * ph
        df['isGreed'] = (df['WVFTop'] >= df['UpperBandTop']) | (df['WVFTop'] >= df['RangeHighTop'])

        # --- Trend & Filter Indicators ---
        df['EMA9'] = df['Close'].ewm(span=ema_len, adjust=False).mean()
        
        # VWAP calculation
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (tp * df['Volume']).groupby(df.index.date).cumsum() / df['Volume'].groupby(df.index.date).cumsum()
        
        # RSI 14
        df['RSI'] = calculate_rsi(df['Close'], 14)

        # Focus on the last completed 15-minute candle
        curr = df.iloc[-2]
        prev = df.iloc[-3]

        # IST Time conversion
        ist = pytz.timezone('Asia/Kolkata')
        candle_time = curr.name.tz_convert(ist).strftime('%Y-%m-%d %H:%M IST')

        # Clean display name
        display_symbol = symbol.replace(".NS", "").replace("^", "")

        # --- Signal Checks ---
        wvf_buy_exit = prev['isPanic'] and (curr['WVF'] < prev['WVF'])
        buy_ce_signal = wvf_buy_exit and (curr['Close'] > curr['Open']) and (curr['Close'] > curr['EMA9']) and (curr['Close'] > curr['VWAP'])

        wvf_sell_exit = prev['isGreed'] and (curr['WVFTop'] < prev['WVFTop'])
        buy_pe_signal = wvf_sell_exit and (curr['Close'] < curr['Open']) and (curr['Close'] < curr['EMA9']) and (curr['Close'] < curr['VWAP'])

        if buy_ce_signal:
            msg = (
                f"🟡 <b>BUY CE ALERT (15m)</b> 🟡\n\n"
                f"<b>Symbol:</b> {display_symbol}\n"
                f"<b>Candle Time:</b> {candle_time}\n"
                f"<b>Price:</b> ₹{curr['Close']:.2f}\n"
                f"<b>RSI (14):</b> {curr['RSI']:.2f}\n"
                f"<b>Target (1.5%):</b> ₹{(curr['Close'] * 1.015):.2f}\n"
                f"<b>StopLoss (0.8%):</b> ₹{(curr['Close'] * 0.992):.2f}"
            )
            send_telegram_msg(msg)

        elif buy_pe_signal:
            msg = (
                f"🔵 <b>BUY PE ALERT (15m)</b> 🔵\n\n"
                f"<b>Symbol:</b> {display_symbol}\n"
                f"<b>Candle Time:</b> {candle_time}\n"
                f"<b>Price:</b> ₹{curr['Close']:.2f}\n"
                f"<b>RSI (14):</b> {curr['RSI']:.2f}\n"
                f"<b>Target (1.5%):</b> ₹{(curr['Close'] * 0.985):.2f}\n"
                f"<b>StopLoss (0.8%):</b> ₹{(curr['Close'] * 1.008):.2f}"
            )
            send_telegram_msg(msg)

    except Exception as e:
        print(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    # Startup Test Message
    now_ist = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S IST')
    send_telegram_msg(f"🚀 <b>WVF Strategy Bot Started Successfully!</b>\n⏰ <b>Execution Time:</b> {now_ist}\n📊 Scanning 15-min F&O charts...")

    # Run scanner across symbols
    for sym in SYMBOLS:
        analyze_symbol(sym)
