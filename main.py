import os
import requests
import pandas as pd
import yfinance as yf

# ==========================================
# TEST SETTING (Badd mein isse False kar dena)
# ==========================================
SEND_TEST_MESSAGE = False  

# Telegram Config (GitHub Secrets)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Tickers List (Major Indexes + High Volatility FnO Stocks)
SYMBOLS = {
    # Major Indexes
    "^NSEI": "NIFTY 50",
    "^NSEBANK": "BANK NIFTY",
    "NIFTY_FIN_SERVICE.NS": "FIN NIFTY",
    # High Volume & Volatility FnO Stocks
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
    "MCX.NS": "MCX"
}

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error sending telegram message: {e}")

def calculate_squeeze_signals(df):
    length = 20
    multBB = 2.0
    multKC = 1.5

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

    # Squeeze Conditions
    df['isSqueezed'] = (df['bbUpper'] < df['kcUpper']) & (df['bbLower'] > df['kcLower'])
    df['squeezeStart'] = df['isSqueezed'] & (~df['isSqueezed'].shift(1).fillna(False))

    # Signal Logic
    sqStartHigh = None
    sqStartLow = None
    buy_signals = []
    sell_signals = []

    for i in range(len(df)):
        row = df.iloc[i]

        if row['squeezeStart']:
            sqStartHigh = row['High']
            sqStartLow = row['Low']

        buy = False
        sell = False

        # Green Candle Close > Squeeze High
        if sqStartHigh is not None:
            is_green = row['Close'] > row['Open']
            if is_green and (row['Close'] > sqStartHigh):
                buy = True
                sqStartHigh = None
                sqStartLow = None

        # Red Candle Close < Squeeze Low
        if sqStartLow is not None and not buy:
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
    for ticker, name in SYMBOLS.items():
        try:
            data = yf.download(ticker, period="5d", interval="15m", progress=False)
            if data.empty or len(data) < 30:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            df = calculate_squeeze_signals(data)
            
            # Check last closed candle
            last_bar = df.iloc[-2]
            prev_time = df.index[-2].strftime('%d-%b %H:%M')

            if last_bar['sqBuySignal']:
                msg = f"🟡 <b>SQUEEZE BUY SIGNAL</b>\n\n<b>Symbol:</b> {name}\n<b>Timeframe:</b> 15m\n<b>Close Price:</b> ₹{last_bar['Close']:.2f}\n<b>Time:</b> {prev_time}"
                send_telegram(msg)
                print(f"BUY Signal sent for {name}")

            elif last_bar['sqSellSignal']:
                msg = f"🖤 <b>SQUEEZE SELL SIGNAL</b>\n\n<b>Symbol:</b> {name}\n<b>Timeframe:</b> 15m\n<b>Close Price:</b> ₹{last_bar['Close']:.2f}\n<b>Time:</b> {prev_time}"
                send_telegram(msg)
                print(f"SELL Signal sent for {name}")

        except Exception as e:
            print(f"Error scanning {name}: {e}")

if __name__ == "__main__":
    # Test Message Alert
    if SEND_TEST_MESSAGE:
        send_telegram("🧪 <b>SYSTEM TEST</b>\n\nSqueeze Signal Scanner script successfully run ho gayi hai!")
        print("Test message sent to Telegram.")
        
    scan_markets()
