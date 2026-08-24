import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# Telegram Credentials (GitHub Secrets se aayenge)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram Send Error: {e}")

# Script run hote hi sabse pehle ye message jayega
send_telegram("🚀 *Vix_Mix Bot Run ho gaya hai!* Market analysis shuru ho raha hai...")

def calculate_vix_mix(symbol="^NSEI", period="60d", interval="1d"):
    # Stock / Index Data download (Default: Nifty 50)
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        send_telegram("⚠️ Market data load nahi ho paya.")
        return

    # Pine script parameters
    pd_val = 22
    bbl = 20
    mult = 2.0
    lb = 50
    ph = 0.99
    pl = 1.01

    close = df['Close']
    low = df['Low']
    high = df['High']

    # Williams Vix Fix Formula
    highest_close = close.rolling(pd_val).max()
    wvf = ((highest_close - low) / highest_close) * 100

    # Bollinger Bands for WVF
    mid_line = wvf.rolling(bbl).mean()
    s_dev = mult * wvf.rolling(bbl).std()
    upper_band = mid_line + s_dev
    lower_band = mid_line - s_dev

    # Percentile High / Low
    range_high = wvf.rolling(lb).max() * ph
    range_low = wvf.rolling(lb).min() * pl

    df['isGreenVix'] = (wvf >= upper_band) | (wvf >= range_high)
    df['isRedVix']   = (wvf <= lower_band) | (wvf <= range_low)
    df['wvf'] = wvf

    # Cluster & Signal Logic
    box_highs = []
    min_red_wvf = None
    min_red_high = None
    in_cluster = False

    for i in range(len(df)):
        is_green = df['isGreenVix'].iloc[i]
        is_red = df['isRedVix'].iloc[i]
        c_high = high.iloc[i]
        c_low = low.iloc[i]
        c_wvf = df['wvf'].iloc[i]

        if is_green:
            if in_cluster and min_red_high is not None:
                box_highs.append(min_red_high)
                if len(box_highs) > 4:
                    box_highs.pop(0)
                min_red_wvf = None
                min_red_high = None
            in_cluster = True

        if in_cluster and is_red:
            if min_red_wvf is None or c_wvf < min_red_wvf:
                min_red_wvf = c_wvf
                min_red_high = c_high

    # Signal Check on latest candle
    if len(box_highs) >= 2:
        breakout_level = max(box_highs[-1], box_highs[-2])
        last_close = close.iloc[-1]
        prev_close = close.iloc[-2]

        # Breakout condition check
        if prev_close <= breakout_level and last_close > breakout_level:
            send_telegram(f"🟢 *GREEN BUY SIGNAL GENERATED!*\nSymbol: {symbol}\nPrice: {last_close}")
        else:
            send_telegram(f"📊 *Analysis Complete:* Abhi koi naya Buy Signal nahi hai.\nLatest Price: {last_close}")

if __name__ == "__main__":
    calculate_vix_mix("^NSEI") # Yahan apna favorite stock symbol daal sakte hain
