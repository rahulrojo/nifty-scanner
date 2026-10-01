import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION SETTINGS
# ==========================================
SEND_TEST_MSG = False  # Test ho gaya ho toh False rakhein

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

SYMBOLS = [
    "^NSEI", "^NSEBANK", "NIFTY_FIN_SERVICE.NS", "NIFTY_MID_SELECT.NS",
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "INFY.NS", "TCS.NS", "LT.NS", "BHARTIARTL.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "MARUTI.NS", "BAJFINANCE.NS", "HINDUNILVR.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "CHOLAFIN.NS", "INDUSINDBK.NS", "BANKBARODA.NS",
    "HDFCLIFE.NS", "SBILIFE.NS", "CANBK.NS", "PNB.NS", "JINDALSTEL.NS", "HINDALCO.NS",
    "VEDL.NS", "BPCL.NS", "IOC.NS", "HPCL.NS", "TATAPOWER.NS", "POWERGRID.NS",
    "NTPC.NS", "COALINDIA.NS", "DLF.NS", "GODREJPROP.NS", "METROPOLIS.NS", "APOLLOHOSP.NS",
    "MAXHEALTH.NS", "DIXON.NS", "POLYCAB.NS", "HAL.NS", "BEL.NS", "RECLTD.NS",
    "PFC.NS", "SHREECEM.NS", "ULTRACEMCO.NS", "GRASIM.NS", "ASIANPAINT.NS", "BERGEPAINT.NS",
    "PIDILITIND.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS", "TRENT.NS", "PAGEIND.NS",
    "COFORGE.NS", "PERSISTENT.NS", "LTTS.NS", "MPHASIS.NS", "TECHM.NS", "WIPRO.NS",
    "HCLTECH.NS", "EICHERMOT.NS", "TVSMOTOR.NS", "HEROMOTOCO.NS", "BATAINDIA.NS", "TITAN.NS",
    "VOLTAS.NS", "UPL.NS", "CONCOR.NS", "AUROPHARMA.NS", "LUPIN.NS", "MANAPPURAM.NS"
]

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram Secrets Missing!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram Error: {e}")

def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def analyze_symbol(symbol):
    try:
        # Fetch 30m data
        df = yf.download(symbol, period="7d", interval="30m", progress=False)
        if df.empty or len(df) < 60:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['EMA_Fast'] = calculate_ema(df['Close'], 9)
        df['EMA_Slow'] = calculate_ema(df['Close'], 21)
        df['EMA_Trend'] = calculate_ema(df['Close'], 50)

        df['CandleBody'] = (df['Close'] - df['Open']).abs()
        df['CandleRange'] = df['High'] - df['Low']
        df['IsStrong'] = df['CandleBody'] > (df['CandleRange'] * 0.4)

        df['EMACrossOver'] = (df['EMA_Fast'] > df['EMA_Slow']) & (df['EMA_Fast'].shift(1) <= df['EMA_Slow'].shift(1))
        df['EMACrossUnder'] = (df['EMA_Fast'] < df['EMA_Slow']) & (df['EMA_Fast'].shift(1) >= df['EMA_Slow'].shift(1))

        df['CE_Condition'] = df['EMACrossOver'] & (df['Close'] > df['EMA_Trend']) & (df['Close'] > df['Open']) & df['IsStrong']
        df['PE_Condition'] = df['EMACrossUnder'] & (df['Close'] < df['EMA_Trend']) & (df['Close'] < df['Open']) & df['IsStrong']

        # State Machine Simulation
        waitingCE = False
        waitingPE = False
        rangeHigh = 0.0
        rangeLow = 0.0
        waitCount = 0
        maxWaitBars = 4
        in_position = False

        # Target index: -2 (Strictly LAST COMPLETED CANDLE, ignoring live running candle)
        target_idx = len(df) - 2

        for i in range(50, len(df) - 1):
            row = df.iloc[i]

            # Trigger Wait States (Only if not already in position)
            if row['CE_Condition'] and not waitingCE and not in_position:
                rangeHigh = float(row['High'])
                rangeLow = float(row['Low'])
                waitingCE = True
                waitingPE = False
                waitCount = 0

            elif row['PE_Condition'] and not waitingPE and not in_position:
                rangeHigh = float(row['High'])
                rangeLow = float(row['Low'])
                waitingPE = True
                waitingCE = False
                waitCount = 0

            if waitingCE or waitingPE:
                waitCount += 1

            if waitCount > maxWaitBars:
                waitingCE = False
                waitingPE = False

            # Confirmations
            confirmedCE = waitingCE and (float(row['Close']) > rangeHigh) and not in_position
            confirmedPE = waitingPE and (float(row['Close']) < rangeLow) and not in_position

            # Invalidation
            if waitingCE and (float(row['Close']) < rangeLow):
                waitingCE = False
            if waitingPE and (float(row['Close']) > rangeHigh):
                waitingPE = False

            # Check if alert belongs strictly to the last closed 30M bar
            if i == target_idx:
                clean_symbol = symbol.replace(".NS", "").replace("^", "")
                last_price = round(float(row['Close']), 2)

                # Format IST Time
                raw_time = df.index[i]
                try:
                    if raw_time.tzinfo is not None:
                        ist_time = raw_time.tz_convert('Asia/Kolkata')
                    else:
                        ist_time = raw_time.tz_localize('UTC').tz_convert('Asia/Kolkata')
                    candle_time_str = ist_time.strftime('%d-%b %I:%M %p')
                except Exception:
                    candle_time_str = str(raw_time)

                if confirmedCE:
                    sl = round(rangeLow, 2)
                    risk = last_price - sl
                    tp = round(last_price + (risk * 1.5), 2)
                    msg = (
                        f"🚀 *ROMY 9.5: CE BUY NOW*\n\n"
                        f"📌 *Symbol:* {clean_symbol}\n"
                        f"⏰ *Timeframe:* 30M (Closed Bar)\n"
                        f"🕐 *Candle Time:* {candle_time_str}\n"
                        f"💵 *Entry Price:* ₹{last_price}\n"
                        f"🛑 *Stop-Loss (SL):* ₹{sl}\n"
                        f"🎯 *Target (TP 1:1.5):* ₹{tp}\n"
                    )
                    return msg

                elif confirmedPE:
                    sl = round(rangeHigh, 2)
                    risk = sl - last_price
                    tp = round(last_price - (risk * 1.5), 2)
                    msg = (
                        f"💥 *ROMY 9.5: PE BUY NOW*\n\n"
                        f"📌 *Symbol:* {clean_symbol}\n"
                        f"⏰ *Timeframe:* 30M (Closed Bar)\n"
                        f"🕐 *Candle Time:* {candle_time_str}\n"
                        f"💵 *Entry Price:* ₹{last_price}\n"
                        f"🛑 *Stop-Loss (SL):* ₹{sl}\n"
                        f"🎯 *Target (TP 1:1.5):* ₹{tp}\n"
                    )
                    return msg

            if confirmedCE or confirmedPE:
                in_position = True
                waitingCE = False
                waitingPE = False

        return None
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None

def main():
    if SEND_TEST_MSG:
        send_telegram_message("🤖 *Romy 9.5 Scanner Active (Strict Closed Candle Mode)*")

    print("Scanning Top 80 Option Symbols...")
    for symbol in SYMBOLS:
        alert_msg = analyze_symbol(symbol)
        if alert_msg:
            send_telegram_message(alert_msg)
            print(f"Alert Sent for {symbol}!")

if __name__ == "__main__":
    main()
