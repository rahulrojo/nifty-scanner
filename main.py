import os
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
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
        print("Telegram Credentials missing!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        print(f"Telegram status: {r.status_code}")
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

def process_strategy_exact_pine(df, left=5, right=5):
    n = len(df)
    if n < (left + right + 10):
        return []

    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    times = df.index

    pivot_lows = [None] * n
    pivot_highs = [None] * n

    for i in range(left, n - right):
        c_low = lows[i]
        is_pl = True
        for k in range(i - left, i):
            if lows[k] <= c_low:
                is_pl = False
                break
        if is_pl:
            for k in range(i + 1, i + right + 1):
                if lows[k] <= c_low:
                    is_pl = False
                    break
        if is_pl:
            pivot_lows[i + right] = c_low

        c_high = highs[i]
        is_ph = True
        for k in range(i - left, i):
            if highs[k] >= c_high:
                is_ph = False
                break
        if is_ph:
            for k in range(i + 1, i + right + 1):
                if highs[k] >= c_high:
                    is_ph = False
                    break
        if is_ph:
            pivot_highs[i + right] = c_high

    lastLow1, lastLow2 = None, None
    lastHigh1, lastHigh2 = None, None
    resistance, support = None, None
    longTriggered, shortTriggered = False, False

    signals = []

    for i in range(n):
        pl = pivot_lows[i]
        ph = pivot_highs[i]

        if pl is not None:
            lastLow2 = lastLow1
            lastLow1 = pl
            support = pl

        if ph is not None:
            lastHigh2 = lastHigh1
            lastHigh1 = ph
            resistance = ph

        validLows = (lastLow1 is not None) and (lastLow2 is not None)
        validHighs = (lastHigh1 is not None) and (lastHigh2 is not None)

        isHL = validLows and (lastLow1 > lastLow2)
        isLH = validHighs and (lastHigh1 < lastHigh2)

        close_price = closes[i]

        longSignal = isHL and (resistance is not None) and (close_price > resistance) and not longTriggered
        shortSignal = isLH and (support is not None) and (close_price < support) and not shortTriggered

        if longSignal:
            longTriggered = True
            shortTriggered = False
            signals.append(("LONG", close_price, times[i]))

        if shortSignal:
            shortTriggered = True
            longTriggered = False
            signals.append(("SHORT", close_price, times[i]))

    return signals

def run_scanner():
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(ist)
    cutoff_time = now_ist - timedelta(hours=24) # Allow recent 24h signals

    signaled_history = load_signaled_history()

    try:
        data = yf.download(FNO_STOCKS, period="10d", interval="15m", group_by='ticker', threads=True, progress=False)
    except Exception as e:
        print(f"Data Fetch Error: {e}")
        return

    sent_count = 0
    for symbol in FNO_STOCKS:
        try:
            df = data[symbol].dropna().copy() if symbol in data else None
            if df is None or df.empty or len(df) < 30:
                continue

            signals = process_strategy_exact_pine(df)

            for sig_type, price, candle_time in signals:
                if candle_time.tzinfo is None:
                    candle_time = ist.localize(candle_time)
                else:
                    candle_time = candle_time.astimezone(ist)

                # Process if signal happened within the last 24 hours
                if candle_time >= cutoff_time:
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
                               f"*Time:* {time_str}\n"
                               f"*Timeframe:* 15 Min")

                        send_telegram(msg)
                        sent_count += 1
                        print(f"SENT: {clean_symbol} {sig_type} at {time_str}")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue

    print(f"Completed. Signals sent in this run: {sent_count}")
    save_signaled_history(signaled_history)

if __name__ == "__main__":
    run_scanner()
