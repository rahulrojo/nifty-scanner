import os
import time
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
import requests

# ==============================================================================
# SCAN_PAST_2_DAYS = True  -> Sirf ek baar run karke pichhle 2 din ke signals dekhein.
# Uske baad ise False karke save kar dein live automated scanning ke liye.
# ==============================================================================
SCAN_PAST_2_DAYS = False  

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

# Verified Correct NSE F&O Symbol List
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
    "LUPIN.NS", "M&M.NS", "M&MFIN.NS", "MANAPPURM.NS", "MARICO.NS", 
    "MARUTI.NS", "UNITDSPR.NS", "MCX.NS", "METROPOLIS.NS", "MFSL.NS", 
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
        df = yf.download(symbol, period="60d", interval="15m", progress=False)
        if df.empty or len(df) < 205:
            return 0

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # --- EXACT PINE SCRIPT FORMULAS MATCH ---
        src = df['Low']

        # Moving Averages
        out = src.rolling(window=25).mean()              # ta.sma(low, 25)
        out2 = src.ewm(span=200, adjust=False).mean()    # ta.ema(low, 200)

        # Keltner Channel (Low based SMA 10 + ATR 14 * 2.0)
        ma_k = src.rolling(window=10).mean()
        high_low = df['High'] - df['Low']
        high_cp = (df['High'] - df['Close'].shift(1)).abs()
        low_cp = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        rangema = tr.ewm(alpha=1/14, adjust=False).mean()  # ta.atr(14)
        upper = ma_k + rangema * 2.0
        lower = ma_k - rangema * 2.0

        # Stochastic (10, 1, 1)
        low_10 = df['Low'].rolling(window=10).min()
        high_10 = df['High'].rolling(window=10).max()
        stoch_k = 100 * ((df['Close'] - low_10) / (high_10 - low_10))

        # MACD (4, 34, 5) on Low
        fast_ma = src.ewm(span=4, adjust=False).mean()
        slow_ma = src.ewm(span=34, adjust=False).mean()
        macd = fast_ma - slow_ma
        signal = macd.ewm(span=5, adjust=False).mean()
        hist = macd - signal

        rsi = calculate_rsi(df['Close'], 14)

        # Raw Pine Script Entry Conditions
        long_cond = (
            (df['Close'] > out) &
            (df['Close'] < upper) &
            (df['Close'] > lower) &
            (hist < 0) &
            (stoch_k < 50) &
            (df['Close'] > out2)
        )

        short_cond = (
            (df['Close'] < out) &
            (df['Close'] < upper) &
            (df['Close'] > lower) &
            (hist > 0) &
            (stoch_k > 50) &
            (df['Close'] < out2)
        )

        # --- EXACT TRADINGVIEW STRATEGY STATE MACHINE ---
        # State tracking prevents multiple continuous alerts for same signal
        trade_entry_long = [False] * len(df)
        trade_entry_short = [False] * len(df)
        position = 0  # 0: Flat, 1: Long, -1: Short

        for i in range(len(df)):
            if long_cond.iloc[i]:
                if position != 1:
                    trade_entry_long[i] = True
                    position = 1
            elif short_cond.iloc[i]:
                if position != -1:
                    trade_entry_short[i] = True
                    position = -1

        df['trade_entry_long'] = trade_entry_long
        df['trade_entry_short'] = trade_entry_short
        df['RSI'] = rsi

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
            tag = "📜 <b>[PAST 2-DAYS SIGNAL]</b>\n" if SCAN_PAST_2_DAYS else ""

            if row['trade_entry_long']:
                count += 1
                msg = (
                    f"{tag}🔹 <b>SCALP LONG ALERT (15m)</b> 🔹\n\n"
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
                    f"{tag}🔻 <b>SCALP SHORT ALERT (15m)</b> 🔻\n\n"
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
