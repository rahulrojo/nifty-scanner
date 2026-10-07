import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# ==========================================
# CONFIGURATION & SECRETS
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Top 100 Volatile F&O Stocks + Indices
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
    "VOLTAS.NS", "UPL.NS", "CONCOR.NS", "AUROPHARMA.NS", "LUPIN.NS", "MANAPPURAM.NS",
    "NMDC.NS", "NATIONALUM.NS", "GLENMARK.NS", "MCX.NS", "SAIL.NS", "MOTHERSON.NS",
    "SHRIRAMFIN.NS", "INDUSTOWER.NS", "OFSS.NS", "MUTHOOTFIN.NS", "TATACOMM.NS", "LICHSGFIN.NS",
    "BALKRISIND.NS", "ITC.NS", "DABUR.NS", "ESCORTS.NS", "AARTIIND.NS", "AMBUJACEM.NS"
]

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram Secrets Missing!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": False
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram Error: {e}")

def get_tradingview_url(clean_symbol):
    tv_symbol = clean_symbol
    if clean_symbol == "NSEI":
        tv_symbol = "NIFTY"
    elif clean_symbol == "NSEBANK":
        tv_symbol = "BANKNIFTY"
    elif clean_symbol == "NIFTY_FIN_SERVICE":
        tv_symbol = "FINNIFTY"
    elif clean_symbol == "NIFTY_MID_SELECT":
        tv_symbol = "MIDCPNIFTY"
    return f"https://in.tradingview.com/chart/?symbol=NSE%3A{tv_symbol}"

def resample_nse_strict_30m(df_5m):
    if df_5m.empty:
        return pd.DataFrame()

    df = df_5m.copy()
    if df.index.tzinfo is not None:
        df.index = df.index.tz_convert('Asia/Kolkata')
    else:
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')

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

def calculate_atr(df):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    return df

def analyze_mother_candle(symbol):
    try:
        df_5m = yf.download(symbol, period="1mo", interval="5m", auto_adjust=False, progress=False)
        if df_5m.empty:
            return []

        if isinstance(df_5m.columns, pd.MultiIndex):
            df_5m.columns = df_5m.columns.get_level_values(0)

        df = resample_nse_strict_30m(df_5m)
        if len(df) < 20:
            return []

        df = calculate_atr(df)

        minAtrMult = 1.0
        maxAtrMult = 2.5
        minBodyPct = 0.50
        maxBars = 8

        mHigh = None
        mLow = None
        mBarIdx = None
        mActive = False

        last_data_date = df.index[-1].date()
        target_date = last_data_date
        alerts = []

        for i in range(15, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            ist_time = df.index[i]
            candle_date = ist_time.date()
            prev_date = df.index[i-1].date() if i > 0 else candle_date

            # Reset / Expiry condition
            if mActive and ((i - mBarIdx > maxBars) or (candle_date != prev_date)):
                mActive = False

            # Inside Bar & Mother Candle Rules
            insideBar = (row['High'] <= prev_row['High']) and (row['Low'] >= prev_row['Low'])
            cRange = prev_row['High'] - prev_row['Low']
            cBody = abs(prev_row['Close'] - prev_row['Open'])
            cAtr = prev_row['ATR'] if not np.isnan(prev_row['ATR']) else 1.0

            motherTimeOk = prev_row.name.strftime('%H:%M') <= '13:30'
            cOk = (cRange >= minAtrMult * cAtr) and (cRange <= maxAtrMult * cAtr) and (cBody >= minBodyPct * cRange) and motherTimeOk

            # Detect Mother Candle
            if not mActive and insideBar and cOk:
                mHigh = float(prev_row['High'])
                mLow = float(prev_row['Low'])
                mBarIdx = i - 1
                mActive = True

            if not mActive or (i <= mBarIdx + 1):
                continue

            close = float(row['Close'])

            # Simple High / Low Breakout Check
            brkUp = close > mHigh
            brkDn = close < mLow

            if (brkUp or brkDn) and (candle_date == target_date):
                clean_symbol = symbol.replace(".NS", "").replace("^", "")
                close_price = round(close, 2)
                candle_time_str = ist_time.strftime('%d-%b %I:%M %p')
                tv_link = get_tradingview_url(clean_symbol)

                if brkUp:
                    signal_type = "CE WAIT"
                    icon = "⚡"
                else:
                    signal_type = "PE WAIT"
                    icon = "🔻"

                msg = (
                    f"{icon} *MOTHER CANDLE BREAKOUT: {signal_type}*\n\n"
                    f"📌 *Symbol:* {clean_symbol}\n"
                    f"⏰ *Timeframe:* 30M\n"
                    f"🕐 *Candle Time:* {candle_time_str}\n"
                    f"💵 *Breakout Close:* ₹{close_price}\n\n"
                    f"📊 [Open Chart on TradingView]({tv_link})"
                )
                alerts.append(msg)
                mActive = False  # Deactivate after breakout signal

        return alerts
    except Exception as e:
        print(f"Error scanning {symbol}: {e}")
        return []

def main():
    print("Scanning Top 100 Stocks for Mother Candle Breakouts (CE/PE WAIT)...")
    total_alerts = 0
    for symbol in SYMBOLS:
        alerts = analyze_mother_candle(symbol)
        for alert_msg in alerts:
            send_telegram_message(alert_msg)
            total_alerts += 1
            print(f"Signal sent for {symbol}!")

    print(f"\nScan Complete. Total signals: {total_alerts}")

if __name__ == "__main__":
    main()
