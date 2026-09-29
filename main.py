import warnings
warnings.filterwarnings("ignore")

import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================
# TEST SETTING (Aap ise False kar dena baad mein)
# ==========================================
SEND_TEST_MESSAGE = True  

# Telegram Config (GitHub Secrets)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOLS = {
    "^NSEI": "NIFTY 50",
    "^NSEBANK": "BANK NIFTY",
    "NIFTY_FIN_SERVICE.NS": "FIN NIFTY",
    "RELIANCE.NS": "RELIANCE",
    "SBIN.NS": "SBIN",
    "HDFCBANK.NS": "HDFCBANK",
    "ICICIBANK.NS": "ICICIBANK",
    "INFY.NS": "INFY",
    "TATAMOTORS.NS": "TATAMOTORS",
    "TATASTEEL.NS": "TATASTEEL",
    "ADANIENT.NS": "ADANIENT",
    "BHARTIARTL.NS": "BHARTIARTL",
    "AXISBANK.NS": "AXISBANK",
    "BAJFINANCE.NS": "BAJFINANCE",
    "LT.NS": "LT",
    "MARUTI.NS": "MARUTI",
    "SUNPHARMA.NS": "SUNPHARMA",
    "TITAN.NS": "TITAN",
    "TCS.NS": "TCS",
    "MCX.NS": "MCX",
    "CIPLA.NS": "CIPLA"
}

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        print(f"Telegram status: {r.status_code}")
    except Exception as e:
        print(f"Error sending telegram message: {e}")

def calculate_squeeze(df):
    length = 20
    multBB = 2.0
    multKC = 1.5

    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')

    # Bollinger Bands
    df['bbMid'] = df['Close'].rolling(window=length).mean()
    df['bbStd'] = df['Close'].rolling(window=length).std()
    df['bbUpper'] = df['bbMid'] + (multBB * df['bbStd'])
    df['bbLower'] = df['bbMid'] - (multBB * df['bbStd'])

    # Keltner Channels
    df['kcEma'] = df['Close'].ewm(span=length, adjust=False).mean()
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = (df['High'] - df['Close'].shift(1)).abs()
    df['tr3'] = (df['Low'] - df['Close'].shift(1)).abs()
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=length).mean()

    df['kcUpper'] = df['kcEma'] + (df['atr'] * multKC)
    df['kcLower'] = df['kcEma'] - (df['atr'] * multKC)

    # Is Squeezed Condition
    df['isSqueezed'] = (df['bbUpper'] < df['kcUpper']) & (df['bbLower'] > df['kcLower'])

    # Calculate Continuous Squeeze Count
    squeeze_counts = []
    count = 0
    for sq in df['isSqueezed']:
        if sq:
            count += 1
        else:
            count = 0
        squeeze_counts.append(count)

    df['squeezeCount'] = squeeze_counts

    # Signal Calculation
    sqStartHigh = None
    sqStartLow = None
    buy_signals = []
    sell_signals = []

    for i in range(len(df)):
        row = df.iloc[i]

        if row['squeezeCount'] == 1:
            sqStartHigh = row['High']
            sqStartLow = row['Low']

        buy = False
        sell = False

        if sqStartHigh is not None and not row['isSqueezed']:
            is_green = row['Close'] > row['Open']
            if is_green and (row['Close'] > sqStartHigh):
                buy = True
                sqStartHigh = None
                sqStartLow = None

        if sqStartLow is not None and not buy and not row['isSqueezed']:
            is_red = row['Close'] < row['Open']
            if is_red and (row['Close'] < sqStartLow):
                sell = True
                sqStartHigh = None
                sqStartLow = None

        buy_signals.append(buy)
        sell_signals.append(sell)

    df['sqBuySignal'] = buy_signals
    df['sqSellSignal'] = sell_signals
    return df

def scan_markets():
    signals_sent = 0
    for ticker, name in SYMBOLS.items():
        try:
            data = yf.download(ticker, period="10d", interval="30m", progress=False, auto_adjust=True)
            if data.empty or len(data) < 30:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                df = data.xs(ticker, level=1, axis=1).copy() if ticker in data.columns.get_level_values(1) else data.droplevel(0, axis=1)
            else:
                df = data.copy()

            df = df.dropna().copy()
            df = calculate_squeeze(df)

            last_bar = df.iloc[-1]
            bar_time = last_bar.name.strftime('%d-%b %H:%M')
            sq_count = int(last_bar['squeezeCount'])

            # 1. Continuous Squeeze Live Update Message
            if sq_count > 0:
                msg = (f"⏳ <b>SQUEEZE IN PROGRESS</b>\n\n"
                       f"<b>Symbol:</b> {name}\n"
                       f"<b>Squeeze Count:</b> Squeeze {sq_count}\n"
                       f"<b>Timeframe:</b> 30m\n"
                       f"<b>Close Price:</b> ₹{last_bar['Close']:.2f}\n"
                       f"<b>Time:</b> {bar_time}")
                send_telegram(msg)
                signals_sent += 1
                print(f"Squeeze Count {sq_count} sent for {name}")

            # 2. Breakout Signal Messages
            elif last_bar['sqBuySignal']:
                msg = (f"🟡 <b>SQUEEZE BUY BREAKOUT</b>\n\n"
                       f"<b>Symbol:</b> {name}\n"
                       f"<b>Timeframe:</b> 30m\n"
                       f"<b>Close Price:</b> ₹{last_bar['Close']:.2f}\n"
                       f"<b>Time:</b> {bar_time}")
                send_telegram(msg)
                signals_sent += 1
                print(f"BUY Signal sent for {name}")

            elif last_bar['sqSellSignal']:
                msg = (f"🖤 <b>SQUEEZE SELL BREAKOUT</b>\n\n"
                       f"<b>Symbol:</b> {name}\n"
                       f"<b>Timeframe:</b> 30m\n"
                       f"<b>Close Price:</b> ₹{last_bar['Close']:.2f}\n"
                       f"<b>Time:</b> {bar_time}")
                send_telegram(msg)
                signals_sent += 1
                print(f"SELL Signal sent for {name}")

        except Exception as e:
            print(f"Error scanning {name}: {e}")

    print(f"Scanning completed. Total messages sent: {signals_sent}")

if __name__ == "__main__":
    if SEND_TEST_MESSAGE:
        send_telegram("🧪 <b>SYSTEM TEST</b>\n\nSqueeze Counter (30m Timeframe) active!")
        print("Test message sent to Telegram.")

    scan_markets()
