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
SEND_TEST_MSG = False  # Set to True for testing setup, False for production

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

def create_nse_30m_candles(df_15m):
    """15-min data (60 days) ko exact TradingView NSE 30M bars mein group karta hai"""
    if df_15m.empty:
        return pd.DataFrame()
    
    df = df_15m.copy()
    if df.index.tzinfo is not None:
        df.index = df.index.tz_convert('Asia/Kolkata')
    else:
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')

    def get_30m_label(ts):
        t = ts.strftime('%H:%M')
        if t in ['09:15', '09:30']: return '09:15'
        elif t in ['09:45', '10:00']: return '09:45'
        elif t in ['10:15', '10:30']: return '10:15'
        elif t in ['10:45', '11:00']: return '10:45'
        elif t in ['11:15', '11:30']: return '11:15'
        elif t in ['11:45', '12:00']: return '11:45'
        elif t in ['12:15', '12:30']: return '12:15'
        elif t in ['12:45', '13:00']: return '12:45'
        elif t in ['13:15', '13:30']: return '13:15'
        elif t in ['13:45', '14:00']: return '13:45'
        elif t in ['14:15', '14:30']: return '14:15'
        elif t in ['14:45', '15:00']: return '14:45'
        elif t == '15:15': return '15:15'
        return None

    df['Date'] = df.index.date
    df['BarTime'] = [get_30m_label(ts) for ts in df.index]
    df = df.dropna(subset=['BarTime'])

    grouped = df.groupby(['Date', 'BarTime']).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).reset_index()

    grouped['Datetime'] = pd.to_datetime(grouped['Date'].astype(str) + ' ' + grouped['BarTime'])
    grouped = grouped.set_index('Datetime').sort_index()

    return grouped[['Open', 'High', 'Low', 'Close', 'Volume']]

def analyze_symbol_for_today(symbol):
    try:
        # Fetch 60 days of 15m data with raw prices (auto_adjust=False)
        df_15m = yf.download(symbol, period="60d", interval="15m", auto_adjust=False, progress=False)
        if df_15m.empty:
            return []

        if isinstance(df_15m.columns, pd.MultiIndex):
            df_15m.columns = df_15m.columns.get_level_values(0)

        df = create_nse_30m_candles(df_15m)
        if len(df) < 150:
            return []

        # EMA Calculation over 750+ 30M candles
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

        ist_tz = pytz.timezone('Asia/Kolkata')
        today_date = datetime.now(ist_tz).date()

        alerts = []

        # Skip first 100 bars for EMA convergence
        for i in range(100, len(df)):
            row = df.iloc[i]
            ist_time = df.index[i]
            candle_date = ist_time.date()
            prev_date = df.index[i-1].date() if i > 0 else candle_date

            if candle_date != prev_date and in_position:
                in_position = False
                pos_type = None

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
        send_telegram_message("🤖 *Scanning All 30M Signals (60D Warmup Fixed)...*")

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
