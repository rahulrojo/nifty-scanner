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
SEND_TEST_MSG = False

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

def resample_nse_strict_30m(df_5m):
    """5M data ko strictly NSE 30M Bins (09:15, 09:45, 10:15...) me map karta hai"""
    if df_5m.empty:
        return pd.DataFrame()

    df = df_5m.copy()
    if df.index.tzinfo is not None:
        df.index = df.index.tz_convert('Asia/Kolkata')
    else:
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')

    # Keep strictly NSE session hours
    df = df.between_time('09:15', '15:29')

    def assign_nse_bin(ts):
        t = ts.time()
        minutes_from_open = (t.hour - 9) * 60 + (t.minute - 15)
        if minutes_from_open < 0:
            return None
        
        bin_idx = minutes_from_open // 30
        bin_start_mins = 9 * 60 + 15 + (bin_idx * 30)
        
        h = bin_start_mins // 60
        m = bin_start_mins % 60
        
        if h > 15 or (h == 15 and m > 15):
            h, m = 15, 15
            
        return f"{h:02d}:{m:02d}"

    df['Date'] = df.index.date
    df['BinTime'] = [assign_nse_bin(ts) for ts in df.index]
    df = df.dropna(subset=['BinTime'])

    grouped = df.groupby(['Date', 'BinTime']).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).reset_index()

    grouped['Datetime'] = pd.to_datetime(grouped['Date'].astype(str) + ' ' + grouped['BinTime'])
    grouped = grouped.set_index('Datetime').sort_index()

    return grouped[['Open', 'High', 'Low', 'Close', 'Volume']]

def analyze_symbol_for_today(symbol):
    try:
        # Fetch 5-minute data (1 month) for exact candle mapping
        df_5m = yf.download(symbol, period="1mo", interval="5m", auto_adjust=False, progress=False)
        if df_5m.empty:
            return []

        if isinstance(df_5m.columns, pd.MultiIndex):
            df_5m.columns = df_5m.columns.get_level_values(0)

        df = resample_nse_strict_30m(df_5m)
        if len(df) < 50:
            return []

        # EMA Indicators
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

        # Auto-target last active trading date
        last_data_date = df.index[-1].date()
        target_date = last_data_date

        alerts = []

        for i in range(30, len(df)):
            row = df.iloc[i]
            ist_time = df.index[i]
            candle_date = ist_time.date()
            prev_date = df.index[i-1].date() if i > 0 else candle_date

            if candle_date != prev_date:
                in_position = False
                pos_type = None
                waitingCE = False
                waitingPE = False

            if in_position:
                if pos_type == 'CE':
                    if float(row['Low']) <= sl_price or float(row['High']) >= tp_price:
                        in_position = False
                        pos_type = None
                elif pos_type == 'PE':
                    if float(row['High']) >= sl_price or float(row['Low']) <= tp_price:
                        in_position = False
                        pos_type = None

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

            if candle_date == target_date:
                clean_symbol = symbol.replace(".NS", "").replace("^", "")
                last_price = round(float(row['Close']), 2)
                candle_time_str = ist_time.strftime('%d-%b %I:%M %p')

                if confirmedCE:
                    sl = round(rangeLow, 2)
                    risk = last_price - sl
                    if risk > 0:
                        tp = round(last_price + (risk * 1.5), 2)
                        msg = (
                            f"🚀 *ROMY 9.5: CE BUY SIGNAL*\n\n"
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
                            f"💥 *ROMY 9.5: PE BUY SIGNAL*\n\n"
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
    print("Scanning All Stocks with Strict NSE 30M Bins...")
    total_alerts = 0
    for symbol in SYMBOLS:
        alerts = analyze_symbol_for_today(symbol)
        for alert_msg in alerts:
            send_telegram_message(alert_msg)
            total_alerts += 1
            print(f"Alert Sent for {symbol}!")

    print(f"\nScan Finished! Total valid signals: {total_alerts}")

if __name__ == "__main__":
    main()
