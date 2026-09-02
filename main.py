import os
import time
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
import requests

# ==============================================================================
# SCAN_PAST_2_DAYS = True -> Pehli baar run par past 2 days ke signals aayenge.
# Signals Telegram par aane ke baad ise False karke save kar dena.
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

# --- ALL NSE F&O INDICES & COMPLETE F&O STOCKS LIST ---
SYMBOLS = [
    # Major Indices
    "^NSEI", "^NSEBANK", "^FINNIFTY", "^NIFTYSMLCAP50",
    
    # Complete F&O Stocks (NSE)
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
        df = yf.download(symbol, period="10d", interval="15m", progress=False)
        if df.empty or len(df) < 205:
            return 0

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['src'] = df['Low']
        df['out_sma'] = df['src'].rolling(window=25).mean()
        df['out_ema'] = df['src'].ewm(span=200, adjust=False).mean()

        df['ma_k'] = df['src'].rolling(window=10).mean()
        high_low = df['High'] - df['Low']
        high_cp = (df['High'] - df['Close'].shift(1)).abs()
        low_cp = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()
        df['kelt_upper'] = df['ma_k'] + (df['atr'] * 2.0)
        df['kelt_lower'] = df['ma_k'] - (df['atr'] * 2.0)

        low_10 = df['Low'].rolling(window=10).min()
        high_10 = df['High'].rolling(window=10).max()
        df['stoch_k'] = 100 * ((df['Close'] - low_10) / (high_10 - low_10))

        df['fast_ma'] = df['src'].ewm(span=4, adjust=False).mean()
        df['slow_ma'] = df['src'].ewm(span=34, adjust=False).mean()
        df['macd'] = df['fast_ma'] - df['slow_ma']
        df['macd_signal'] = df['macd'].ewm(span=5, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        df['RSI'] = calculate_rsi(df['Close'], 14)

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

            if row['long_signal']:
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

            elif row['short_signal']:
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
