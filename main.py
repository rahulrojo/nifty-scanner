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

# --- UNIVERSE: MAJOR INDEXES + TOP VOLATILE F&O STOCKS (NSE) ---
SYMBOLS = [
    # Major Indexes
    "^NSEI", "^NSEBANK", "^CNXIT", "^NSEMDCP50",
    # Top Volatile F&O Stocks
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

# --- SQUEEZE CALCULATIONS (EXACT TRADINGVIEW LOGIC) ---
def calculate_squeeze(df, length=20, mult_bb=2.0, mult_kc=1.5):
    # 1. Bollinger Bands (ta.bb)
    df['sma'] = df['Close'].rolling(window=length).mean()
    df['std'] = df['Close'].rolling(window=length).std(ddof=0)
    df['bb_upper'] = df['sma'] + (mult_bb * df['std'])
    df['bb_lower'] = df['sma'] - (mult_bb * df['std'])

    # 2. Keltner Channels (ta.ema + ta.atr)
    df['kc_ema'] = df['Close'].ewm(span=length, adjust=False).mean()
    
    # Exact TradingView ATR Calculation (RMA / Wilder's Smoothing)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # RMA formula: ewm with alpha = 1 / length
    df['atr'] = tr.ewm(alpha=1/length, adjust=False).mean()

    df['kc_upper'] = df['kc_ema'] + (df['atr'] * mult_kc)
    df['kc_lower'] = df['kc_ema'] - (df['atr'] * mult_kc)

    # 3. Squeeze Condition
    df['is_squeezed'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
    df['squeeze_start'] = df['is_squeezed'] & (~df['is_squeezed'].shift(1).fillna(False))
    
    return df

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[LOG] Telegram Token/Chat ID Missing! Msg: {message[:30]}...")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code != 200:
            print(f"[ERROR] Telegram API Error: {res.text}")
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram message: {e}")

def main():
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.datetime.now(ist)
    today_date = now_ist.date()

    run_all_today = "--all-today" in sys.argv

    print(f"--- Scan Started at {now_ist.strftime('%Y-%m-%d %H:%M:%S')} IST ---")
    signals_found = 0

    for ticker in SYMBOLS:
        try:
            # Fetch data using Ticker object
            tk = yf.Ticker(ticker)
            data = tk.history(period="7d", interval="30m")
            
            if data.empty or len(data) < 25:
                continue

            df = calculate_squeeze(data)

            # Timezone conversion to IST
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert(ist)
            else:
                df.index = df.index.tz_convert(ist)

            # Aaj ki candles filter karein
            df_today = df[df.index.date == today_date].copy()
            if df_today.empty:
                continue

            # Count total squeezes today for this symbol
            squeeze_count = 0
            
            for i, (timestamp, row) in enumerate(df_today.iterrows()):
                if row['squeeze_start']:
                    squeeze_count += 1
                    
                    # Live mode: Check if signal was in last 2 candles (to handle GH Actions delay)
                    is_recent = (len(df_today) - i) <= 2

                    if run_all_today or is_recent:
                        signals_found += 1
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
                            f"⚡ *Status:* BB inside Keltner Channel!"
                        )
                        send_telegram_message(msg)
                        print(f"✅ Alert sent: {clean_symbol} at {candle_time_str}")

        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")

    print(f"--- Scan Finished. Total Signals Sent: {signals_found} ---")

if __name__ == "__main__":
    main()
