import os
import time
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
import requests

# ==============================================================================
# SCAN_PAST_2_DAYS = True  -> Pehli baar pichhle 2 din ke EXACT TradingView signals aayenge.
# Signals Telegram par aane ke baad ise Change karke False kar dena.
# ==============================================================================
SCAN_PAST_2_DAYS = True  

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_msg(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Error: TELEGRAM_BOT_TOKEN ya TELEGRAM_CHAT_ID missing hai.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

# Complete NSE F&O List
SYMBOLS = [
    "^NSEI", "^NSEBANK", "^FINNIFTY", "^NIFTYSMLCAP50",
    "AARTIIND.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", 
    "ADANIENT.NS", "ADANIPORTS.NS", "ALKEM.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", 
    "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTRAL.NS", "ATUL.NS", 
    "AUBANK.NS", "AUROPHARMA.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", 
    "BAJFINANCE.NS", "BALKRISIND.NS", "BALRAMCHIN.NS", "BANDHANBNK.NS", "BANKBARODA.NS", 
    "BATAINDIA.NS", "BEL.NS", "BHARATFORG.NS", "BHARTIARTL.NS", "BHEL.NS", 
    "BIOCON.NS", "BPCL.NS", "BRITANNIA.NS", "BSOFT.NS", "CANBK.NS", 
    "CANFINHOME.NS", "CHAMBLFERT.NS", "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", 
    "COFORGE.NS", "COLPAL.NS", "CONCOR.NS", "COROMANDEL.NS", "CROMPTON.NS", 
    "CUMMINSIND.NS", "DABUR.NS", "DALBHARAT.NS", "DEEPAKNTR.NS", "DIVISLAB.NS", 
    "DIXON.NS", "DLF.NS", "DRREDDY.NS", "EICHERMOT.NS", "ESCORTS.NS", 
    "EXIDEIND.NS", "FEDERALBNK.NS", "GAIL.NS", "GLENMARK.NS", "GODREJCP.NS", 
    "GODREJPROP.NS", "GRANULES.NS", "GRASIM.NS", "GUJGASLTD.NS", "HAL.NS", 
    "HAVELLS.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", 
    "HINDALCO.NS", "HINDCOPPER.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", 
    "ICICIGI.NS", "ICICIPRULI.NS", "IDEA.NS", "IDFCFIRSTB.NS", "IEX.NS", 
    "IGL.NS", "INDHOTEL.NS", "INDIACEM.NS", "INDUSINDBK.NS", "INDUSTOWER.NS", 
    "INFY.NS", "IOC.NS", "IPCALAB.NS", "IRCTC.NS", "IREDA.NS", 
    "IRFC.NS", "ITC.NS", "JINDALSTEL.NS", "JIOFIN.NS", "JKCEMENT.NS", 
    "JSWSTEEL.NS", "JUBLFOOD.NS", "KALYANKJIL.NS", "KEI.NS", "KOTAKBANK.NS", 
    "LALPATHLAB.NS", "LICHSGFIN.NS", "LT.NS", "LTIM.NS", "LTTS.NS", 
    "LUPIN.NS", "M&M.NS", "M&MFIN.NS", "MANAPPURAM.NS", "MARICO.NS", 
    "MARUTI.NS", "MCDOWELL-N.NS", "MCX.NS", "METROPOLIS.NS", "MFSL.NS", 
    "MGL.NS", "MOTHERSON.NS", "MPHASIS.NS", "MRF.NS", "MUTHOOTFIN.NS", 
    "NATIONALUM.NS", "NAUKRI.NS", "NAVINFLUOR.NS", "NESTLEIND.NS", "NMDC.NS", 
    "NTPC.NS", "OBEROIRLTY.NS", "OFSS.NS", "ONGC.NS", "PAGEIND.NS", 
    "PERSISTENT.NS", "PETRONET.NS", "PFC.NS", "PIDILITIND.NS", "PIIND.NS", 
    "PNB.NS", "POLYCAB.NS", "POWERGRID.NS", "PVRINOX.NS", "RAMCOCEM.NS", 
    "RBLBANK.NS", "RECLTD.NS", "RELIANCE.NS", "SAIL.NS", "SBICARD.NS", 
    "SBILIFE.NS", "SBIN.NS", "SHREECEM.NS", "SHRIRAMFIN.NS", "SIEMENS.NS", 
    "SOLARINDS.NS", "SONACOMS.NS", "SRF.NS", "SUNPHARMA.NS", "SUNTV.NS", 
    "SYNGENE.NS", "TATACHEM.NS", "TATACOMM.NS", "TATACONSUM.NS", "TATAMOTORS.NS", 
    "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", 
    "TORNTPHARM.NS", "TORNTPOWER.NS", "TRENT.NS", "TVSMOTOR.NS", "UBL.NS", 
    "ULTRACEMCO.NS", "UPL.NS", "VBL.NS", "VEDL.NS", "VOLTAS.NS", 
    "WIPRO.NS", "ZYDUSLIFE.NS"
]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_symbol(symbol):
    count = 0
    try:
        # Fetch 60d of 15m data to properly align EMA 200 and long-term position state
        df = yf.download(symbol, period="60d", interval="15m", progress=False)
        if df.empty or len(df) < 205:
            return 0

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # --- EXACT PINE SCRIPT FORMULAS ---
        src = df['Low']

        # Moving Averages
        df['out'] = src.rolling(window=25).mean()              # SMA(low, 25)
        df['out2'] = src.ewm(span=200, adjust=False).mean()    # EMA(low, 200)

        # Keltner Channel
        ma_k = src.rolling(window=10).mean()                    # SMA(low, 10)
        high_low = df['High'] - df['Low']
        high_cp = (df['High'] - df['Close'].shift(1)).abs()
        low_cp = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        rangema = tr.ewm(alpha=1/14, adjust=False).mean()       # ta.atr(14) via Wilder RMA
        df['upper'] = ma_k + rangema * 2.0
        df['lower'] = ma_k - rangema * 2.0

        # Stochastic (10, 1, 1)
        low_10 = df['Low'].rolling(window=10).min()
        high_10 = df['High'].rolling(window=10).max()
        df['stoch_k'] = 100 * ((df['Close'] - low_10) / (high_10 - low_10))

        # MACD (4, 34, 5) on Low
        fast_ma = src.ewm(span=4, adjust=False).mean()
        slow_ma = src.ewm(span=34, adjust=False).mean()
        macd = fast_ma - slow_ma
        signal = macd.ewm(span=5, adjust=False).mean()
        df['hist'] = macd - signal

        df['RSI'] = calculate_rsi(df['Close'], 14)

        # Raw Strategy Signal Conditions (Pine Script exact match)
        df['long_cond'] = (
            (df['Close'] > df['out']) &
            (df['Close'] < df['upper']) &
            (df['Close'] > df['lower']) &
            (df['hist'] < 0) &
            (df['stoch_k'] < 50) &
            (df['Close'] > df['out2'])
        )

        df['short_cond'] = (
            (df['Close'] < df['out']) &
            (df['Close'] < df['upper']) &
            (df['Close'] > df['lower']) &
            (df['hist'] > 0) &
            (df['stoch_k'] > 50) &
            (df['Close'] < df['out2'])
        )

        # --- TRADINGVIEW STRATEGY ENGINE SIMULATOR ---
        # Simulates strategy.entry position switching behavior
        position = 0  # 0: Flat, 1: Long, -1: Short
        df['trade_entry_long'] = False
        df['trade_entry_short'] = False

        for i in range(len(df)):
            if df['long_cond'].iloc[i]:
                if position != 1:
                    df['trade_entry_long'].iloc[i] = True
                    position = 1
            elif df['short_cond'].iloc[i]:
                if position != -1:
                    df['trade_entry_short'].iloc[i] = True
                    position = -1

        ist = pytz.timezone('Asia/Kolkata')
        display_symbol = symbol.replace(".NS", "").replace("^", "")

        df_completed = df.iloc[:-1]

        # 1 day = 25 candles (15m). 2 days = 50 candles.
        if SCAN_PAST_2_DAYS:
            df_scan = df_completed.tail(50)
        else:
            df_scan = df_completed.tail(1)

        for idx, row in df_scan.iterrows():
            if idx.tzinfo is None:
                candle_dt = pytz.utc.localize(idx).astimezone(ist)
            else:
                candle_dt = idx.astimezone(ist)
                
            candle_time = candle_dt.strftime('%Y-%m-%d %H:%M IST')

            if row['trade_entry_long']:
                count += 1
                msg = (
                    f"🔹 <b>SCALP LONG ALERT (15m)</b> 🔹\n\n"
                    f"<b>Symbol:</b> {display_symbol}\n"
                    f"<b>Candle Time:</b> {candle_time}\n"
                    f"<b>Price:</b> ₹{row['Close']:.2f}\n"
                    f"<b>RSI (14):</b> {row['RSI']:.2f}"
                )
                send_telegram_msg(msg)
                time.sleep(0.3)

            elif row['trade_entry_short']:
                count += 1
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

    return count

if __name__ == "__main__":
    now_ist = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S IST')
    
    mode_desc = "Historical 2 Days Scan" if SCAN_PAST_2_DAYS else "Live 15m Signal Scan"
    send_telegram_msg(
        f"🚀 <b>Scalping System Bot Started!</b>\n"
        f"⏰ <b>Execution Time:</b> {now_ist}\n"
        f"📊 <b>Mode:</b> {mode_desc}\n"
        f"🎯 Scanning 180+ F&O Stocks & Indices (15m)..."
    )

    total_found = 0
    for sym in SYMBOLS:
        total_found += analyze_symbol(sym)

    send_telegram_msg(f"✅ <b>Scan Complete!</b>\nTotal Signals Found: <b>{total_found}</b>")
