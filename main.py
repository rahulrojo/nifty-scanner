import os
import time
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
import requests

# ==============================================================================
# SETTING: Sirf pehli baar 2 din ke signal lene ke liye ise True rakha hai.
# Telegram par ek baar message aane ke baad ise CHANGE karke False kar dena.
# ==============================================================================
SCAN_PAST_2_DAYS = True  

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
    # Top F&O Stocks
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
    "AXISBANK.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "BAJFINANCE.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
    "HCLTECH.NS", "M&M.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS",
    "WIPRO.NS", "ULTRACEMCO.NS", "JSWSTEEL.NS", "GRASIM.NS", "HINDALCO.NS"
]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_symbol(symbol):
    try:
        # Fetch 7 days of 15m data to accurately calculate 200 EMA
        df = yf.download(symbol, period="7d", interval="15m", progress=False)
        if df.empty or len(df) < 205:
            return

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # --- EXACT SCALPING STRATEGY FORMULAS ---
        df['src'] = df['Low']  # Source is Low as per Pine Script

        # SMA 25 & EMA 200 of Low
        df['out_sma'] = df['src'].rolling(window=25).mean()
        df['out_ema'] = df['src'].ewm(span=200, adjust=False).mean()

        # Keltner Channel (Length 10, Mult 2.0, ATR 14)
        df['ma_k'] = df['src'].rolling(window=10).mean()
        high_low = df['High'] - df['Low']
        high_cp = (df['High'] - df['Close'].shift(1)).abs()
        low_cp = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()
        df['kelt_upper'] = df['ma_k'] + (df['atr'] * 2.0)
        df['kelt_lower'] = df['ma_k'] - (df['atr'] * 2.0)

        # Stochastic %K (10, 1, 1)
        low_10 = df['Low'].rolling(window=10).min()
        high_10 = df['High'].rolling(window=10).max()
        df['stoch_k'] = 100 * ((df['Close'] - low_10) / (high_10 - low_10))

        # MACD Fast (4, 34, 5) on Low
        df['fast_ma'] = df['src'].ewm(span=4, adjust=False).mean()
        df['slow_ma'] = df['src'].ewm(span=34, adjust=False).mean()
        df['macd'] = df['fast_ma'] - df['slow_ma']
        df['macd_signal'] = df['macd'].ewm(span=5, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # RSI 14
        df['RSI'] = calculate_rsi(df['Close'], 14)

        # --- SIGNAL TRIGGERS ---
        df['long_signal'] = (
            (df['Close'] > df['out_sma']) &
            (df['Close'] < df['kelt_upper']) &
            (df['Close'] > df['kelt_lower']) &
            (df['macd_hist'] < 0) &
            (df['stoch_k'] < 50) &
            (df['Close'] > df['out_ema'])
        )

        df['short_signal'] = (
            (df['Close'] < df['out_sma']) &
            (df['Close'] < df['kelt_upper']) &
            (df['Close'] > df['kelt_lower']) &
            (df['macd_hist'] > 0) &
            (df['stoch_k'] > 50) &
            (df['Close'] < df['out_ema'])
        )

        ist = pytz.timezone('Asia/Kolkata')
        display_symbol = symbol.replace(".NS", "").replace("^", "")

        now_ist = datetime.datetime.now(ist)
        two_days_ago = now_ist - datetime.timedelta(days=2)

        # Ignore unclosed active bar
        df_completed = df.iloc[:-1]

        if SCAN_PAST_2_DAYS:
            # Check all candles of last 2 days
            df_scan = df_completed[df_completed.index >= two_days_ago]
        else:
            # Check ONLY the last completed 15m candle
            df_scan = df_completed.iloc[-1:]

        for idx, row in df_scan.iterrows():
            candle_time = idx.tz_convert(ist).strftime('%Y-%m-%d %H:%M IST')

            if row['long_signal']:
                msg = (
                    f"🔹 <b>SCALP LONG ALERT (15m)</b> 🔹\n\n"
                    f"<b>Symbol:</b> {display_symbol}\n"
                    f"<b>Candle Time:</b> {candle_time}\n"
                    f"<b>Price:</b> ₹{row['Close']:.2f}\n"
                    f"<b>RSI (14):</b> {row['RSI']:.2f}"
                )
                send_telegram_msg(msg)
                time.sleep(0.3)

            elif row['short_signal']:
                msg = (
                    f"🔻 <b>SCALP SHORT ALERT (15m)</b> 🔻\n\n"
                    f"<b>Symbol:</b> {display_symbol}\n"
                    f"<b>Candle Time:</b> {candle_time}\n"
                    f"<b>Price:</b> ₹{row['Close']:.2f}\n"
                    f"<b>RSI (14):</b> {row['RSI']:.2f}"
                )
                send_telegram_msg(msg)
                time.sleep(0.3)

    except Exception as e:
        print(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    now_ist = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S IST')
    
    mode_desc = "Historical 2 Days Scan" if SCAN_PAST_2_DAYS else "Live 15m Signal Scan"
    send_telegram_msg(
        f"🚀 <b>Scalping System Bot Started!</b>\n"
        f"⏰ <b>Execution Time:</b> {now_ist}\n"
        f"📊 <b>Mode:</b> {mode_desc}\n"
        f"🎯 Scanning 15m F&O (LONG & SHORT)..."
    )

    for sym in SYMBOLS:
        analyze_symbol(sym)
