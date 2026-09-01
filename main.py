import os
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time
import pytz

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
STATE_FILE = "signaled_candles.json"

FNO_STOCKS = [
    "^NSEI", "^NSEBANK", "NIFTY_FIN_SERVICE.NS",
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
        print("Telegram Secrets (TOKEN/CHAT_ID) Missing in GitHub Settings!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        print(f"Telegram Post Status: {r.status_code}")
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

def check_latest_candle_breakout(df, left=5, right=5):
    n = len(df)
    if n < (left + right + 10):
        return None

    lows = df['Low'].values
    highs = df['High'].values
    closes = df['Close'].values

    lastLow1, lastLow2 = None, None
    lastHigh1, lastHigh2 = None, None
    resistance, support = None, None

    for i in range(left, n - right - 1):
        c_low = lows[i]
        if all(c_low < lows[i-left:i]) and all(c_low < lows[i+1:i+right+1]):
            lastLow2 = lastLow1
            lastLow1 = c_low
            support = c_low

        c_high = highs[i]
        if all(c_high > highs[i-left:i]) and all(c_high > highs[i+1:i+right+1]):
            lastHigh2 = lastHigh1
            lastHigh1 = c_high
            resistance = c_high

    validLows = (lastLow1 is not None) and (lastLow2 is not None)
    validHighs = (lastHigh1 is not None) and (lastHigh2 is not None)

    isHL = validLows and (lastLow1 > lastLow2)
    isLH = validHighs and (lastHigh1 < lastHigh2)

    last_idx = -1
    last_close = closes[last_idx]
    last_time = df.index[last_idx]
    last_support = support if support is not None else lows[last_idx]
    last_resistance = resistance if resistance is not None else highs[last_idx]

    longSignal = isHL and (resistance is not None) and (last_close > resistance)
    shortSignal = isLH and (support is not None) and (last_close < support)

    if longSignal:
        sl = last_support
        target = last_close + (last_close - sl) * 1.5
        return ("LONG", last_close, sl, target, last_time)
    elif shortSignal:
        sl = last_resistance
        target = last_close - (sl - last_close) * 1.5
        return ("SHORT", last_close, sl, target, last_time)

    return None

def run_scanner():
    # 1. Immediate Test Message on Trigger
    send_telegram("🔔 *Bot Connection Active:* Scanner execution started successfully!")

    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(ist)
    current_time = now_ist.time()

    # 2. Market Hours Check (9:15 AM to 3:30 PM IST)
    market_open = time(9, 15)
    market_close = time(3, 30)

    if not (market_open <= current_time <= market_close):
        print(f"Market is closed ({current_time.strftime('%H:%M IST')}). Stock scanning skipped.")
        return

    signaled_history = load_signaled_history()

    try:
        data = yf.download(FNO_STOCKS, period="5d", interval="15m", progress=False)
    except Exception as e:
        print(f"Data Fetch Error: {e}")
        return

    sent_count = 0
    for symbol in FNO_STOCKS:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if symbol not in data['Close'].columns:
                    continue
                df = pd.DataFrame({
                    'Open': data['Open'][symbol],
                    'High': data['High'][symbol],
                    'Low': data['Low'][symbol],
                    'Close': data['Close'][symbol]
                }).dropna()
            else:
                df = data.dropna().copy()

            if df.empty or len(df) < 20:
                continue

            res = check_latest_candle_breakout(df)
            if res is not None:
                sig_type, price, sl, target, candle_time = res

                if candle_time.tzinfo is None:
                    candle_time = ist.localize(candle_time)
                else:
                    candle_time = candle_time.astimezone(ist)

                signal_key = f"{symbol}_{sig_type}_{candle_time.strftime('%Y%m%d_%H%M')}"

                if signal_key not in signaled_history:
                    signaled_history.add(signal_key)

                    clean_symbol = symbol.replace(".NS", "").replace("^NSEI", "NIFTY 50").replace("^NSEBANK", "BANKNIFTY")
                    time_str = candle_time.strftime("%d-%b %H:%M")

                    emoji = "🟩" if sig_type == "LONG" else "🟥"
                    msg = (f"{emoji} *PIVOT BREAKOUT SIGNAL*\n\n"
                           f"*Symbol:* {clean_symbol}\n"
                           f"*Signal:* {sig_type}\n"
                           f"*Entry Price:* ₹{price:.2f}\n"
                           f"*Stoploss:* ₹{sl:.2f}\n"
                           f"*Target:* ₹{target:.2f}\n"
                           f"*Time:* {time_str}\n"
                           f"*Timeframe:* 15 Min")

                    send_telegram(msg)
                    sent_count += 1
                    print(f"LIVE SIGNAL SENT: {clean_symbol} {sig_type}")
        except Exception as e:
            continue

    print(f"Scan Done. Signals Sent: {sent_count}")
    save_signaled_history(signaled_history)

if __name__ == "__main__":
    run_scanner()
