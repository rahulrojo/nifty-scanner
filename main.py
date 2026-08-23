import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# Telegram Configuration (GitHub Secrets se load hoga)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# NSE F&O Stock List (Aap isme aur tickers add kar sakte hain)
FNO_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
    "AXISBANK.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "BAJFINANCE.NS", "HCLTECH.NS", "ASIANPAINT.NS", "NTPC.NS",
    "ULTRACATE.NS", "POWERGRID.NS", "INDUSINDBK.NS", "M&M.NS", "TATACONSUM.NS"
]

def send_telegram_msg(message):
    """Telegram par notification bhejta hai."""
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        requests.post(url, json=payload)

def calculate_vix_mix_signals(df):
    """Pine Script ke Vix_Mix logic ke hisab se signals calculate karta hai."""
    pd_val, bbl, mult, lb, ph, pl = 22, 20, 2.0, 50, 0.99, 1.01

    # Williams Vix Fix Calculation
    highest_close = df['Close'].rolling(pd_val).max()
    wvf = ((highest_close - df['Low']) / highest_close) * 100
    mid_line = wvf.rolling(bbl).mean()
    s_dev = mult * wvf.rolling(bbl).std(ddof=0)
    
    upper_band = mid_line + s_dev
    lower_band = mid_line - s_dev
    
    range_high = wvf.rolling(lb).max() * ph
    range_low = wvf.rolling(lb).min() * pl

    is_green_vix = (wvf >= upper_band) | (wvf >= range_high)
    is_red_vix = (wvf <= lower_band) | (wvf <= range_low)

    box_highs = []
    min_red_wvf = None
    min_red_high = None
    in_cluster = False
    green_buy_triggered = False
    
    green_buy_signals = [False] * len(df)

    # Stateful Loop - Pine Script ke logic ke sath match karne ke liye
    for i in range(len(df)):
        c_wvf = wvf.iloc[i]
        c_is_green = is_green_vix.iloc[i]
        c_is_red = is_red_vix.iloc[i]
        c_high = df['High'].iloc[i]
        c_close = df['Close'].iloc[i]
        p_close = df['Close'].iloc[i-1] if i > 0 else None

        if pd.isna(c_wvf):
            continue

        # Green VIX & Cluster Reset
        if c_is_green:
            if in_cluster and min_red_wvf is not None:
                box_highs.append(min_red_high)
                if len(box_highs) > 4:
                    box_highs.pop(0)
                min_red_wvf = None
                min_red_high = None
            in_cluster = True

        # Red VIX tracking in cluster
        if in_cluster and c_is_red:
            if min_red_wvf is None or c_wvf < min_red_wvf:
                min_red_wvf = c_wvf
                min_red_high = c_high

        # Buy Signal Verification
        can_buy = len(box_highs) >= 2
        green_buy = False

        if can_buy:
            ref_high1 = box_highs[-1]
            ref_high2 = box_highs[-2]
            breakout_level = max(ref_high1, ref_high2)

            if p_close is not None:
                crossover = (p_close <= breakout_level) and (c_close > breakout_level)
                if not green_buy_triggered and crossover:
                    green_buy = True

        if green_buy:
            green_buy_triggered = True

        green_buy_signals[i] = green_buy

    return green_buy_signals

def main():
    alerts = []
    print("F&O Stocks scan ho rahe hain...")

    for ticker in FNO_STOCKS:
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if len(df) < 60:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            signals = calculate_vix_mix_signals(df)

            # Latest bar check
            if signals[-1]:
                latest_price = round(float(df['Close'].iloc[-1]), 2)
                stock_name = ticker.replace('.NS', '')
                alerts.append(f"🟢 *GREEN BUY SIGNAL*\n*Stock:* `{stock_name}`\n*Price:* ₹{latest_price}")
        except Exception as e:
            print(f"Error scanning {ticker}: {e}")

    if alerts:
        msg = "🚀 *VIX_MIX STRATEGY ALERTS*\n\n" + "\n\n---\n\n".join(alerts)
        send_telegram_msg(msg)
        print("Telegram alert bhej diya gaya!")
    else:
        print("Aaj koi Green Buy signal nahi mila.")

if __name__ == "__main__":
    main()
