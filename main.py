import os
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import pytz

# Telegram Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
STATE_FILE = "signaled_candles.json"

# Indexes + F&O Stocks List
FNO_STOCKS = [
    # Main Indexes
    "^NSEI", "^NSEBANK", "NIFTY_FIN_SERVICE.NS",
    
    # F&O Stocks
    "AARTIIND.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "ALKEM.NS", "AMBUJACEMENT.NS", "APOLLOHOSP.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTRAL.NS", "ATUL.NS",
    "AUBANK.NS", "AUROPHARMA.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BALKRISIND.NS",
    "BALRAMCHIN.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "BATAINDIA.NS", "BEL.NS", "BERGEPAINT.NS", "BHARATFORG.NS",
    "BHARTIARTL.NS", "BHEL.NS", "BIOCON.NS", "BSOFT.NS", "BPCL.NS", "BRITANNIA.NS", "CANBK.NS", "CANFINHOME.NS",
    "CHAMBLFERT.NS", "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", "CONCOR.NS", "COROMANDEL.NS",
    "CROMPTON.NS", "CUB.NS", "CUMMINSIND.NS", "DABUR.NS", "DALBHARAT.NS", "DEEPAKNTR.NS", "DIVISLAB.NS", "DIXON.NS",
    "DLF.NS", "DRREDDY.NS", "EICHERMOT.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "GAIL.NS", "GLENMARK.NS",
    "GMRINFRA.NS", "GNFC.NS", "GODREJCP.NS", "GODREJPROP.NS", "GRANULES.NS", "GRASIM.NS", "GUJGASLTD.NS", "HAL.NS",
    "HAVELLS.NS", "HCLTECH.NS", "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDCOPPER.NS",
    "HINDPETRO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDEA.NS", "IDFC.NS", "IDFCFIRSTB.NS",
    "IEX.NS", "IGL.NS", "INDHOTEL.NS", "INDIACEM.NS", "INDIAMART.NS", "INDIGO.NS", "INDUSINDBK.NS", "INDUSTOWER.NS",
    "INFY.NS", "IOC.NS", "IPCALAB.NS", "IRCTC.NS", "ITC.NS", "JINDALSTEL.NS", "JKCEMENT.NS", "JSWSTEEL.NS",
    "JUBLFOOD.NS", "KOTAKBANK.NS", "LALPATHLAB.NS", "LAURUSLABS.NS", "LICHSGFIN.NS", "LT.NS", "LTIM.NS", "LTTS.NS",
    "LUPIN.NS", "M&M.NS", "M&MFIN.NS", "MANAPPURAM.NS", "MARICO.NS", "MARUTI.NS", "MCDOWELL-N.NS", "MCX.NS",
    "METROPOLIS.NS", "MFSL.NS", "MGL.NS", "MOTHERSON.NS", "MPHASIS.NS", "MRF.NS", "MUTHOOTFIN.NS", "NATIONALUM.NS",
    "NAVINFLUOR.NS", "NESTLEIND.NS", "NMDC.NS", "NTPC.NS", "OBEROIRLTY.NS", "OFSS.NS", "ONGC.NS", "PAGEIND.NS",
    "PERSISTENT.NS", "PETRONET.NS", "PFC.NS", "PIDILITIND.NS", "PIIND.NS", "PNB.NS", "POLYCAB.NS", "POWERGRID.NS",
    "PVRINOX.NS", "RAMCOCEM.NS", "RBLBANK.NS", "RECLTD.NS", "RELIANCE.NS", "SAIL.NS", "SBICARD.NS", "SBILIFE.NS",
    "SBIN.NS", "SHREECEM.NS", "SHRIRAMFIN.NS", "SIEMENS.NS", "SRF.NS", "SUNPHARMA.NS", "SUNTV.NS", "SYNGENE.NS",
    "TATACHEMICAL.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS",
    "TITAN.NS", "TORNTPHARM.NS", "TRENT.NS", "TVSMOTOR.NS", "UBL.NS", "ULTRACEMCO.NS", "UPL.NS", "VEDL.NS",
    "VOLTAS.NS", "WIPRO.NS", "ZEEL.NS"
]

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")

def load_signaled_history():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()

def save_signaled_history(history):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(list(history), f)
    except Exception as e:
        print(f"History Save Error: {e}")

def calculate_pivots(df, left=5, right=5):
    df['pivot_low'] = np.nan
    df['pivot_high'] = np.nan

    for i in range(left, len(df) - right):
        # Pivot Low Calculation
        current_low = df['Low'].iloc[i]
        if all(current_low < df['Low'].iloc[i - left:i]) and all(current_low < df['Low'].iloc[i + 1:i + right + 1]):
            df.iloc[i, df.columns.get_loc('pivot_low')] = current_low

        # Pivot High Calculation
        current_high = df['High'].iloc[i]
        if all(current_high > df['High'].iloc[i - left:i]) and all(current_high > df['High'].iloc[i + 1:i + right + 1]):
            df.iloc[i, df.columns.get_loc('pivot_high')] = current_high

    return df

def process_strategy(df):
    df = calculate_pivots(df, left=5, right=5)
    
    lastLow1, lastLow2 = None, None
    lastHigh1, lastHigh2 = None, None
    resistance, support = None, None
    
    longTriggered = False
    shortTriggered = False
    
    signals = []

    for i in range(len(df)):
        # Pivot Low Tracking
        pl = df['pivot_low'].iloc[i]
        if not np.isnan(pl):
            lastLow2 = lastLow1
            lastLow1 = pl
            support = pl

        # Pivot High Tracking
        ph = df['pivot_high'].iloc[i]
        if not np.isnan(ph):
            lastHigh2 = lastHigh1
            lastHigh1 = ph
            resistance = ph

        validLows = (lastLow1 is not None) and (lastLow2 is not None)
        validHighs = (lastHigh1 is not None) and (lastHigh2 is not None)

        isHL = validLows and (lastLow1 > lastLow2)
        isLH = validHighs and (lastHigh1 < lastHigh2)

        close_price = df['Close'].iloc[i]

        longSignal = isHL and (resistance is not None) and (close_price > resistance) and not longTriggered
        shortSignal = isLH and (support is not None) and (close_price < support) and not shortTriggered

        if longSignal:
            longTriggered = True
            shortTriggered = False
            signals.append(("LONG", close_price, df.index[i]))

        if shortSignal:
            shortTriggered = True
            longTriggered = False
            signals.append(("SHORT", close_price, df.index[i]))

    return signals

def run_scanner():
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(ist)
    today_str = now_ist.strftime("%Y-%m-%d")

    signaled_history = load_signaled_history()

    try:
        data = yf.download(FNO_STOCKS, period="5d", interval="15m", group_by='ticker', threads=True, progress=False)
    except Exception as e:
        print(f"Data Fetch Error: {e}")
        return

    for symbol in FNO_STOCKS:
        try:
            df = data[symbol].dropna().copy() if symbol in data else None
            if df is None or df.empty or len(df) < 20:
                continue

            signals = process_strategy(df)

            for sig_type, price, candle_time in signals:
                # Convert time to IST
                if candle_time.tzinfo is None:
                    candle_time = ist.localize(candle_time)
                else:
                    candle_time = candle_time.astimezone(ist)

                candle_day = candle_time.strftime("%Y-%m-%d")
                
                # Filter: Aaj ke bane hue signals hi consider karenge
                if candle_day == today_str:
                    signal_key = f"{symbol}_{sig_type}_{candle_time.strftime('%Y%m%d_%H%M')}"

                    if signal_key not in signaled_history:
                        signaled_history.add(signal_key)

                        # Clean Name for Indexes
                        clean_symbol = symbol.replace(".NS", "").replace("^NSEI", "NIFTY 50").replace("^NSEBANK", "BANKNIFTY")
                        time_str = candle_time.strftime("%d-%b %H:%M")

                        emoji = "🟩" if sig_type == "LONG" else "🟥"
                        msg = (f"{emoji} *PIVOT BREAKOUT SIGNAL*\n\n"
                               f"*Symbol:* {clean_symbol}\n"
                               f"*Signal:* {sig_type}\n"
                               f"*Entry Price:* ₹{price:.2f}\n"
                               f"*Time:* {time_str}\n"
                               f"*Timeframe:* 15 Min")

                        send_telegram(msg)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue

    save_signaled_history(signaled_history)

if __name__ == "__main__":
    run_scanner()
