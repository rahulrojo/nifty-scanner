import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta

# Telegram Settings
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = "-1003921675472"

# FnO Stocks List
FNO_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "TATAMOTORS.NS", "AXISBANK.NS", "BAJFINANCE.NS",
    "LTIM.NS", "MARUTI.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "SUNPHARMA.NS",
    "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "VOLTAS.NS",
    "ASTRAL.NS", "INDHOTEL.NS", "GRASIM.NS", "TORNTPHARM.NS", "BHARATFORG.NS",
    "ABCAPITAL.NS", "HINDZINC.NS", "CGPOWER.NS", "SHRIRAMFIN.NS", "JINDALSTEL.NS",
    "KOTAKBANK.NS", "LT.NS", "M&M.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "HAL.NS",
    "BEL.NS", "COALINDIA.NS", "BPCL.NS", "IOC.NS", "DLF.NS", "GODREJPROP.NS"
]

def send_telegram(message):
    if not BOT_TOKEN:
        print("❌ ERROR: BOT_TOKEN secret is missing!")
        return False
        
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"Telegram API Error [{response.status_code}]: {response.text}")
        else:
            print("Signal sent successfully to Telegram!")
        return response.status_code == 200
    except Exception as e:
        print(f"Exception while sending Telegram message: {e}")
        return False

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def scan_vix_mix_balanced(ticker):
    try:
        data = yf.download(ticker, period="5d", interval="15m", progress=False)
        if data.empty or len(data) < 60:
            return

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        df = data.copy()

        # 1. Price Bollinger Bands
        df['P_Mid'] = df['Close'].rolling(20).mean()
        df['P_Std'] = df['Close'].rolling(20).std()
        df['P_Upper'] = df['P_Mid'] + (2.0 * df['P_Std'])
        df['P_Lower'] = df['P_Mid'] - (2.0 * df['P_Std'])

        # 2. RSI Calculation
        df['RSI'] = calculate_rsi(df['Close'], 14)

        # 3. VIX Mix Calculations
        highest_close_22 = df['Close'].rolling(22).max()
        df['WVF_Buy'] = ((highest_close_22 - df['Low']) / highest_close_22) * 100
        wvf_buy_mid = df['WVF_Buy'].rolling(20).mean()
        wvf_buy_std = df['WVF_Buy'].rolling(20).std()
        df['WVF_Buy_Upper'] = wvf_buy_mid + (2.0 * wvf_buy_std)
        df['WVF_Buy_High'] = df['WVF_Buy'].rolling(50).max() * 0.88

        lowest_close_22 = df['Close'].rolling(22).min()
        df['WVF_Sell'] = ((df['High'] - lowest_close_22) / lowest_close_22) * 100
        wvf_sell_mid = df['WVF_Sell'].rolling(20).mean()
        wvf_sell_std = df['WVF_Sell'].rolling(20).std()
        df['WVF_Sell_Upper'] = wvf_sell_mid + (2.0 * wvf_sell_std)
        df['WVF_Sell_High'] = df['WVF_Sell'].rolling(50).max() * 0.88

        df['Is_Panic'] = (df['WVF_Buy'].shift(1) >= df['WVF_Buy_Upper'].shift(1)) | (df['WVF_Buy'].shift(1) >= df['WVF_Buy_High'].shift(1))
        df['Is_Euphoria'] = (df['WVF_Sell'].shift(1) >= df['WVF_Sell_Upper'].shift(1)) | (df['WVF_Sell'].shift(1) >= df['WVF_Sell_High'].shift(1))

        curr = df.iloc[-2]
        prev = df.iloc[-3]
        recent_window = df.iloc[-5:-2]

        panic_recent = recent_window['Is_Panic'].any() or curr['Is_Panic']
        panic_turning = panic_recent and (curr['WVF_Buy'] < prev['WVF_Buy'])

        euphoria_recent = recent_window['Is_Euphoria'].any() or curr['Is_Euphoria']
        euphoria_turning = euphoria_recent and (curr['WVF_Sell'] < prev['WVF_Sell'])

        buy_reentry = (curr['Low'] <= curr['P_Lower'] or prev['Low'] <= prev['P_Lower']) and (curr['Close'] > curr['Open']) and (curr['Close'] > curr['P_Lower'])
        sell_reentry = (curr['High'] >= curr['P_Upper'] or prev['High'] >= prev['P_Upper']) and (curr['Close'] < curr['Open']) and (curr['Close'] < curr['P_Upper'])

        # Strict RSI Filters
        rsi_buy_ok = curr['RSI'] <= 42
        rsi_sell_ok = curr['RSI'] >= 58

        stock_name = ticker.replace(".NS", "")
        close_price = round(float(curr['Close']), 2)
        low_price = round(float(curr['Low']), 2)
        high_price = round(float(curr['High']), 2)

        # Current IST Timestamp
        ist_tz = timezone(timedelta(hours=5, minutes=30))
        signal_time = datetime.now(ist_tz).strftime("%I:%M %p (%d %b)")

        if panic_turning and buy_reentry and rsi_buy_ok:
            msg = f"🟢 *VIX MIX BALANCED BUY SIGNAL (15m)* 🟢\n\n*Stock:* {stock_name}\n*Price:* ₹{close_price}\n*StopLoss Zone:* ₹{low_price}\n*RSI:* {round(float(curr['RSI']), 1)}\n*Time:* ⏰ {signal_time}"
            send_telegram(msg)

        elif euphoria_turning and sell_reentry and rsi_sell_ok:
            msg = f"🔴 *VIX MIX BALANCED SELL SIGNAL (15m)* 🔴\n\n*Stock:* {stock_name}\n*Price:* ₹{close_price}\n*StopLoss Zone:* ₹{high_price}\n*RSI:* {round(float(curr['RSI']), 1)}\n*Time:* ⏰ {signal_time}"
            send_telegram(msg)

    except Exception as e:
        print(f"Error scanning {ticker}: {e}")

if __name__ == "__main__":
    print("🚀 VIX Mix Strict Scanner Started...")
    
    for stock in FNO_STOCKS:
        scan_vix_mix_balanced(stock)
        
    print("✅ Market scan finished successfully.")
