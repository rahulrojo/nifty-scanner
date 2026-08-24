import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# Config Settings
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = "-1003921675472"

# Top FnO Stocks List (Add more tickers as needed with .NS extension)
FNO_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "TATAMOTORS.NS", "AXISBANK.NS", "BAJFINANCE.NS",
    "LTIM.NS", "MARUTI.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "SUNPHARMA.NS",
    "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "VOLTAS.NS"
]

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Failed to send alert: {e}")

def calculate_vix_mix_signals(ticker):
    try:
        # Fetch 15-minute intraday data
        data = yf.download(ticker, period="5d", interval="15m", progress=False)
        if data.empty or len(data) < 50:
            return

        # Flatten multi-index columns if returned by yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        df = data.copy()

        # 1. Price Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (2.0 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Mid'] - (2.0 * df['BB_Std'])

        # 2. VIX Fix Bottom Logic (BUY)
        highest_close = df['Close'].rolling(window=22).max()
        df['WVF_Buy'] = ((highest_close - df['Low']) / highest_close) * 100
        wvf_buy_mid = df['WVF_Buy'].rolling(window=20).mean()
        wvf_buy_std = df['WVF_Buy'].rolling(window=20).std()
        df['WVF_Buy_Upper'] = wvf_buy_mid + (2.0 * wvf_buy_std)
        df['WVF_Buy_High'] = df['WVF_Buy'].rolling(window=50).max() * 0.90

        df['Is_Buy_Panic'] = (df['WVF_Buy'] >= df['WVF_Buy_Upper']) & (df['WVF_Buy'] >= df['WVF_Buy_High'])

        # 3. Inverted VIX Fix Top Logic (SELL)
        lowest_close = df['Close'].rolling(window=22).min()
        df['WVF_Sell'] = ((df['High'] - lowest_close) / lowest_close) * 100
        wvf_sell_mid = df['WVF_Sell'].rolling(window=20).mean()
        wvf_sell_std = df['WVF_Sell'].rolling(window=20).std()
        df['WVF_Sell_Upper'] = wvf_sell_mid + (2.0 * wvf_sell_std)
        df['WVF_Sell_High'] = df['WVF_Sell'].rolling(window=50).max() * 0.90

        df['Is_Sell_Euphoria'] = (df['WVF_Sell'] >= df['WVF_Sell_Upper']) & (df['WVF_Sell'] >= df['WVF_Sell_High'])

        # Latest completed candle check
        last_row = df.iloc[-2]  # Using previous closed candle to avoid live repainting
        prev_rows = df.iloc[-5:-2]

        buy_panic_recent = prev_rows['Is_Buy_Panic'].any() or last_row['Is_Buy_Panic']
        sell_euphoria_recent = prev_rows['Is_Sell_Euphoria'].any() or last_row['Is_Sell_Euphoria']

        stock_name = ticker.replace(".NS", "")
        close_price = round(float(last_row['Close']), 2)
        low_price = round(float(last_row['Low']), 2)
        high_price = round(float(last_row['High']), 2)

        # Signal Trigger Conditions
        if buy_panic_recent and (last_row['Close'] > last_row['BB_Lower']) and (last_row['Close'] > last_row['Open']):
            msg = f"🟢 *FnO BUY DIP SIGNAL (15m)* 🟢\n\n*Stock:* {stock_name}\n*Price:* ₹{close_price}\n*StopLoss Zone:* ₹{low_price}"
            send_telegram(msg)

        elif sell_euphoria_recent and (last_row['Close'] < last_row['BB_Upper']) and (last_row['Close'] < last_row['Open']):
            msg = f"🔴 *FnO SELL TOP SIGNAL (15m)* 🔴\n\n*Stock:* {stock_name}\n*Price:* ₹{close_price}\n*StopLoss Zone:* ₹{high_price}"
            send_telegram(msg)

    except Exception as e:
        print(f"Error scanning {ticker}: {e}")

if __name__ == "__main__":
    for ticker in FNO_STOCKS:
        calculate_vix_mix_signals(ticker)
