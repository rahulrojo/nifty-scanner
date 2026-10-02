import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# ==========================================
# CONFIGURATION SETTINGS
# ==========================================
SEND_TEST_MSG = False  # Set to True for testing setup, False for regular runs

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

def analyze_symbol_for_today(symbol):
    try:
        # Fetch native 30M candles directly (60 days period for full EMA warmup)
        df = yf.download(symbol, period="60d", interval="30m", auto_adjust=False, progress=False)
        if df.empty:
            return []

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        if len(df) < 100:
            return []

        # Timezone localization to IST
        if df.index.tzinfo is not None:
            df.index = df.index.tz_convert('Asia/Kolkata')
        else:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')

        # EMA Calculations
        df['EMA_Fast'] = calculate_ema(df['Close'], 9)
        df['EMA_Slow'] = calculate_ema(df['Close'], 21)
        df['EMA_Trend'] = calculate_ema(df['Close'], 50)

        df['CandleBody'] = (df['Close'] - df['Open']).abs()
        df['CandleRange'] = df['High'] - df['Low']
        df['IsStrong'] = df['CandleBody'] > (df['CandleRange'] * 0.4)

        # Crossover Detection
        df['EMACrossOver'] = (df['EMA_Fast'] > df['EMA_Slow']) & (df['EMA_Fast'].shift(1) <= df['EMA_Slow'].shift(1))
        df['EMACrossUnder'] = (df['EMA_Fast'] < df['EMA_Slow']) & (df['EMA_Fast'].shift(1) >= df['EMA_Slow'].shift(1))

        df['CE_Condition'] = df['EMACrossOver'] & (df['Close'] > df['EMA_Trend']) & (df['Close'] > df['Open']) & df['IsStrong']
        df['PE_Condition'] = df['EMACrossUnder'] & (df['Close'] < df['EMA_Trend']) & (df['Close'] < df['Open']) & df['IsStrong']

        waitingCE = False
        waitingPE = False
        rangeHigh = 0.0
        rangeLow = 0.0
        waitCount = 0
        maxWaitBars = 4

        in_position = False
        pos_type = None
        sl_price = 0.0
        tp_price = 0.0

        ist_tz = pytz.timezone('Asia/Kolkata')
        today_date = datetime.now(ist_tz).date()

        alerts = []

        for i in range(50, len(df)):
            row = df.iloc[i]
            ist_time = df.index[i]
            candle_date = ist_time.date()
            prev_date = df.index[i-1].date() if i > 0 else candle_date

            # Reset state at market open
            if candle_date != prev_date:
                in_position = False
                pos_type = None
                waitingCE = False
                waitingPE = False

            # Position SL/TP check
            if in_position:
                if pos_type == 'CE':
                    if float(row['Low']) <= sl_price or float(row['High']) >= tp_price:
                        in_position = False
                        pos_type = None
                elif pos_type == 'PE':
                    if float(row['High']) >= sl_price or float(row['Low']) <= tp_price:
                        in_position = False
                        pos_type = None

            # Setup trigger
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

            confirmedCE = waitingCE and (float(row['Close']) > rangeHigh) and not in_position
            confirmedPE = waitingPE and (float(row['Close']) < rangeLow) and not in_position

            if waitingCE and (float(row['Close']) < rangeLow):
                waitingCE = False
            if waitingPE and (float(row['Close']) > rangeHigh):
                waitingPE = False

            # Send Alert only for TODAY's signals
            if candle_date == today_date:
                clean_symbol = symbol.replace(".NS", "").replace("^", "")
                last_price = round(float(row['Close']), 2)
                candle_time_str = ist_time.strftime('%d-%b %I:%M %p')

                if confirmedCE:
                    sl = round(rangeLow, 2)
                    risk = last_price - sl
                    if risk > 0:
                        tp = round(last_price + (risk * 1.5), 2)
                        msg = (
                            f"🚀 *ROMY 9.5: TODAY'S CE BUY SIGNAL*\n\n"
                            f"📌 *Symbol:* {clean_symbol}\n"
                            f"⏰ *Timeframe:* 30M (NSE Session)\n"
                            f"🕐 *Candle Time:* {candle_time_str}\n"
                            f"💵 *Entry Price:* ₹{last_price}\n"
                            f"🛑 *Stop-Loss (SL):* ₹{sl}\n"
                            f"🎯 *Target (TP 1:1.5):* ₹{tp}\n"
                        )
                        alerts.append(msg)

                elif confirmedPE:
                    sl = round(rangeHigh, 2)
                    risk = sl - last_price
                    if risk > 0:
                        tp = round(last_price - (risk * 1.5), 2)
                        msg = (
                            f"💥 *ROMY 9.5: TODAY'S PE BUY SIGNAL*\n\n"
                            f"📌 *Symbol:* {clean_symbol}\n"
                            f"⏰ *Timeframe:* 30M (NSE Session)\n"
                            f"🕐 *Candle Time:* {candle_time_str}\n"
                            f"💵 *Entry Price:* ₹{last_price}\n"
                            f"🛑 *Stop-Loss (SL):* ₹{sl}\n"
                            f"🎯 *Target (TP 1:1.5):* ₹{tp}\n"
                        )
                        alerts.append(msg)

            if confirmedCE:
                in_position = True
                pos_type = 'CE'
                sl_price = rangeLow
                sl_risk = float(row['Close']) - sl_price
                tp_price = float(row['Close']) + (sl_risk * 1.5)
                waitingCE = False

            elif confirmedPE:
                in_position = True
                pos_type = 'PE'
                sl_price = rangeHigh
                sl_risk = sl_price - float(row['Close'])
                tp_price = float(row['Close']) - (sl_risk * 1.5)
                waitingPE = False

        return alerts
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return []

def main():
    if SEND_TEST_MSG:
        send_telegram_message("🤖 *Scanning All 30M Signals (Direct Native 30M Fix)...*")

    print("Scanning All 80 Stocks For Today's Signals...")
    total_alerts = 0
    for symbol in SYMBOLS:
        alerts = analyze_symbol_for_today(symbol)
        for alert_msg in alerts:
            send_telegram_message(alert_msg)
            total_alerts += 1
            print(f"Today's Alert Sent for {symbol}!")

    print(f"Done! Total signals sent for today: {total_alerts}")

if __name__ == "__main__":
    main()
