import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION SETTINGS
# ==========================================
# Test message flag (Test hone ke baad isko False kar dena)
SEND_TEST_MSG = True  

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Top 80 High Volatility Option Trading Stocks & Indices (Yahoo Finance Tickers)
SYMBOLS = [
    "^NSEI", "^NSEBANK", "NIFTY_FIN_SERVICE.NS", "NIFTY_MID_SELECT.NS", # Indices
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
    """Telegram par text alert bhejne ke liye function"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram Token ya Chat ID missing hai Secrets me!")
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
        print(f"Telegram Message Error: {e}")

def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def analyze_symbol(symbol):
    """Romy 9.5 Master Breakout Engine (30M) Logic"""
    try:
        # Download last 5 days data in 30M timeframe
        df = yf.download(symbol, period="5d", interval="30m", progress=False)
        if df.empty or len(df) < 60:
            return None

        # Multi-index fix if yfinance returns multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['EMA_Fast'] = calculate_ema(df['Close'], 9)
        df['EMA_Slow'] = calculate_ema(df['Close'], 21)
        df['EMA_Trend'] = calculate_ema(df['Close'], 50)

        df['CandleBody'] = (df['Close'] - df['Open']).abs()
        df['CandleRange'] = df['High'] - df['Low']
        df['IsStrong'] = df['CandleBody'] > (df['CandleRange'] * 0.4)

        # Crossover Detection
        df['EMACrossOver'] = (df['EMA_Fast'] > df['EMA_Slow']) & (df['EMA_Fast'].shift(1) <= df['EMA_Slow'].shift(1))
        df['EMACrossUnder'] = (df['EMA_Fast'] < df['EMA_Slow']) & (df['EMA_Fast'].shift(1) >= df['EMA_Slow'].shift(1))

        # Raw Conditions
        df['CE_Condition'] = df['EMACrossOver'] & (df['Close'] > df['EMA_Trend']) & (df['Close'] > df['Open']) & df['IsStrong']
        df['PE_Condition'] = df['EMACrossUnder'] & (df['Close'] < df['EMA_Trend']) & (df['Close'] < df['Open']) & df['IsStrong']

        # State Engine simulation over candles
        waitingCE = False
        waitingPE = False
        rangeHigh = 0.0
        rangeLow = 0.0
        waitCount = 0
        maxWaitBars = 4

        alerts = []

        for i in range(50, len(df)):
            row = df.iloc[i]

            # CE Wait Trigger
            if row['CE_Condition'] and not waitingCE:
                rangeHigh = row['High']
                rangeLow = row['Low']
                waitingCE = True
                waitingPE = False
                waitCount = 0

            # PE Wait Trigger
            elif row['PE_Condition'] and not waitingPE:
                rangeHigh = row['High']
                rangeLow = row['Low']
                waitingPE = True
                waitingCE = False
                waitCount = 0

            if waitingCE or waitingPE:
                waitCount += 1

            if waitCount > maxWaitBars:
                waitingCE = False
                waitingPE = False

            # Confirmations
            confirmedCE = waitingCE and row['Close'] > rangeHigh
            confirmedPE = waitingPE and row['Close'] < rangeLow

            # Invalidation
            if waitingCE and row['Close'] < rangeLow:
                waitingCE = False
            if waitingPE and row['Close'] > rangeHigh:
                waitingPE = False

            # Trigger Alert on the latest completed candle
            if i == len(df) - 1:
                clean_symbol = symbol.replace(".NS", "").replace("^", "")
                last_price = round(float(row['Close']), 2)

                # Time formatting to IST (Indian Standard Time)
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
                        f"⏰ *Timeframe:* 30M\n"
                        f"🕐 *Candle Time:* {candle_time_str}\n"
                        f"💵 *Entry Price:* ₹{last_price}\n"
                        f"🛑 *Stop-Loss (SL):* ₹{sl}\n"
                        f"🎯 *Target (TP 1:1.5):* ₹{tp}\n"
                    )
                    alerts.append(msg)
                    waitingCE = False

                elif confirmedPE:
                    sl = round(rangeHigh, 2)
                    risk = sl - last_price
                    tp = round(last_price - (risk * 1.5), 2)
                    msg = (
                        f"💥 *ROMY 9.5: PE BUY NOW*\n\n"
                        f"📌 *Symbol:* {clean_symbol}\n"
                        f"⏰ *Timeframe:* 30M\n"
                        f"🕐 *Candle Time:* {candle_time_str}\n"
                        f"💵 *Entry Price:* ₹{last_price}\n"
                        f"🛑 *Stop-Loss (SL):* ₹{sl}\n"
                        f"🎯 *Target (TP 1:1.5):* ₹{tp}\n"
                    )
                    alerts.append(msg)
                    waitingPE = False

        return alerts
    except Exception as e:
        print(f"Error checking {symbol}: {e}")
        return None

def main():
    if SEND_TEST_MSG:
        send_telegram_message("🤖 *Romy 9.5 Engine Test Alert*\n\nBot start ho chuka hai aur 30M Timeframe Scan kar raha hai!")

    print("Scanning Top 80 Option Stocks & Indices...")
    for symbol in SYMBOLS:
        alerts = analyze_symbol(symbol)
        if alerts:
            for alert_msg in alerts:
                send_telegram_message(alert_msg)
                print(f"Alert Sent for {symbol}!")

if __name__ == "__main__":
    main()
