import os
import sys
import pytz
import datetime
import requests
import numpy as np
import pandas as pd
import yfinance as yf

# --- TELEGRAM CONFIGURATION ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- UNIVERSE: MAJOR INDEXES + TOP 100 VOLATILE F&O STOCKS (NSE) ---
SYMBOLS = [
    # Major Indexes
    "^NSEI", "^NSEBANK", "^CNXIT", "^NSEMDCP50",
    # Top 100 Volatile F&O Stocks
    "RELIANCE.NS", "TATAMOTORS.NS", "SBIN.NS", "ICICIBANK.NS", "HDFCBANK.NS",
    "INFY.NS", "TCS.NS", "BAJFINANCE.NS", "TATASTEEL.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "AXISBANK.NS", "KOTAKBANK.NS", "MARUTI.NS", "MM.NS",
    "SUNPHARMA.NS", "HINDALCO.NS", "VEDL.NS", "BHARATFORG.NS", "DLF.NS",
    "BANDHANBNK.NS", "ZEEL.NS", "IDEA.NS", "CANBK.NS", "PNB.NS",
    "COALINDIA.NS", "CHOLAFIN.NS", "INDUSINDBK.NS", "JSWSTEEL.NS", "TATACHEM.NS",
    "TATAPOWER.NS", "BEL.NS", "HAL.NS", "RECLTD.NS", "PFC.NS",
    "DIXON.NS", "PERSISTENT.NS", "COFORGE.NS", "LT.NS", "SRF.NS",
    "ASTRAL.NS", "JUBLFOOD.NS", "POLYCAB.NS", "MCX.NS", "MOTHERSON.NS",
    "NMDC.NS", "NATIONALUM.NS", "SAIL.NS", "IRCTC.NS", "RVNL.NS",
    "HDFCLIFE.NS", "SBILIFE.NS", "BALKRISIND.NS", "ESCORTS.NS", "TIINDIA.NS",
    "TRENT.NS", "AUROPHARMA.NS", "LUPIN.NS", "CIPLA.NS", "GLENMARK.NS",
    "GRANULES.NS", "BIOCON.NS", "SYNGENE.NS", "TATACOMM.NS", "INDIGO.NS", "AMBUJACEM.NS",
    "GODREJPROP.NS", "OBEROIRLTY.NS", "BSOFT.NS", "KPITTECH.NS", "LTTS.NS",
    "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "PIDILITIND.NS", "BERGEPAINT.NS",
    "ASIANPAINT.NS", "TITAN.NS", "TATAELXSI.NS", "MUTHOOTFIN.NS", "MANAPPURAM.NS",
    "PEL.NS", "GNFC.NS", "METROPOLIS.NS", "DRREDDY.NS", "ABFRL.NS",
    "HINDPETRO.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "ONGC.NS",
    "PETRONET.NS", "BHEL.NS", "CONCOR.NS", "VOLTAS.NS", "UPL.NS",
    "AARTIIND.NS", "APOLLOTYRE.NS", "LICHSGFIN.NS", "EXIDEIND.NS", "IPCALAB.NS"
]

# --- SQUEEZE CALCULATIONS ---
def calculate_squeeze(df, length=20, mult_bb=2.0, mult_kc=1.5):
    # Bollinger Bands
    df['sma'] = df['Close'].rolling(window=length).mean()
    df['std'] = df['Close'].rolling(window=length).std(ddof=0)
    df['bb_upper'] = df['sma'] + (mult_bb * df['std'])
    df['bb_lower'] = df['sma'] - (mult_bb * df['std'])

    # Keltner Channels
    df['kc_ema'] = df['Close'].ewm(span=length, adjust=False).mean()
    
    # ATR Calculation
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=length).mean()

    df['kc_upper'] = df['kc_ema'] + (df['atr'] * mult_kc)
    df['kc_lower'] = df['kc_ema'] - (df['atr'] * mult_kc)

    # Squeeze Logic
    df['is_squeezed'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
    df['squeeze_start'] = df['is_squeezed'] & (~df['is_squeezed'].shift(1).fillna(False))
    
    return df

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram Token/Chat ID missing. Message:", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

def main():
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.datetime.now(ist)
    today_date = now_ist.date()

    # Mode: Command line argument `--all-today` check karne ke liye
    run_all_today = "--all-today" in sys.argv

    print(f"Starting Scan at {now_ist.strftime('%Y-%m-%d %H:%M:%S')} IST")

    for ticker in SYMBOLS:
        try:
            # 30m candles for last 5 days
            data = yf.download(ticker, period="5d", interval="30m", progress=False)
            if data.empty or len(data) < 30:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            df = calculate_squeeze(data)

            # IST timezone fix
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert(ist)
            else:
                df.index = df.index.tz_convert(ist)

            # Aaj ki candles filter karein
            df_today = df[df.index.date == today_date].copy()
            if df_today.empty:
                continue

            # Aaj ke kitne total squeezes aaye
            squeeze_count = 0
            
            for timestamp, row in df_today.iterrows():
                if row['squeeze_start']:
                    squeeze_count += 1
                    
                    # Diff in minutes check karein (Live mode me last 35 mins ki candle check hogi)
                    time_diff = (now_ist - timestamp).total_seconds() / 60.0

                    # Conditions to send message:
                    # 1. Force send all today's squeezes (--all-today flag active)
                    # 2. Live Market mode: Candle trigger last 35 min ke andar ka ho
                    if run_all_today or (time_diff <= 35 and time_diff >= 0):
                        clean_symbol = ticker.replace(".NS", "").replace("^", "")
                        candle_time_str = timestamp.strftime('%I:%M %p')
                        close_price = round(row['Close'], 2)

                        msg = (
                            f"🟡 *SQUEEZE STARTED ALERT* 🟡\n\n"
                            f"📌 *Stock/Index:* `{clean_symbol}`\n"
                            f"⏱ *Timeframe:* 30 Min\n"
                            f"🕒 *Candle Time:* `{candle_time_str}`\n"
                            f"🔢 *Today's Squeeze No:* #{squeeze_count}\n"
                            f"💰 *Close Price:* ₹{close_price}\n\n"
                            f"⚡ *Status:* Bollinger Bands inside Keltner Channel!"
                        )
                        send_telegram_message(msg)
                        print(f"Alert sent for {clean_symbol} at {candle_time_str}")

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

if __name__ == "__main__":
    main()
