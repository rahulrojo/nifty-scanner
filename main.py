import os
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz

# ==========================================
# 1. CONFIGURATION & STOCKS LIST
# ==========================================
STOCKS = [
    "CONCOR.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS", "TATAMOTORS.NS",
    "ICICIBANK.NS", "HDFCBANK.NS", "SBIN.NS", "AXISBANK.NS", "BHARTIARTL.NS"
]  # Aap yahan apni pasand ke NSE stocks (.NS extension ke sath) add kar sakte hain

TIMEFRAME = "15m"
DAYS_LOOKBACK = 2
SENT_SIGNALS_FILE = "sent_signals.json"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Load existing sent signals to prevent duplicates
if os.path.exists(SENT_SIGNALS_FILE):
    try:
        with open(SENT_SIGNALS_FILE, "r") as f:
            sent_signals = json.load(f)
    except Exception:
        sent_signals = []
else:
    sent_signals = []

# ==========================================
# 2. PINE SCRIPT STRATEGY LOGIC CONVERSION
# ==========================================
def calculate_strategy(df):
    if len(df) < 100:
        return []

    # Parameters from Pine Script
    left, right = 5, 5
    pd_val, bbl, mult, lb, ph = 22, 20, 2.0, 50, 0.90
    price_bbl, price_mult, rsi_len = 20, 2.0, 14
    slPct = 1.5

    # 1. RSI (RMA Method like TradingView)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/rsi_len, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/rsi_len, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. Price Bollinger Bands
    df['p_mid'] = df['Close'].rolling(price_bbl).mean()
    df['p_std'] = df['Close'].rolling(price_bbl).std(ddof=0)
    df['p_upper'] = df['p_mid'] + price_mult * df['p_std']
    df['p_lower'] = df['p_mid'] - price_mult * df['p_std']

    # 3. VIX Mix Logic
    highest_close = df['Close'].rolling(pd_val).max()
    df['wvf_buy'] = ((highest_close - df['Low']) / highest_close) * 100
    df['wvf_buy_upper'] = df['wvf_buy'].rolling(bbl).mean() + mult * df['wvf_buy'].rolling(bbl).std(ddof=0)
    df['wvf_buy_high'] = df['wvf_buy'].rolling(lb).max() * ph
    df['is_buy_panic'] = (df['wvf_buy'] >= df['wvf_buy_upper']) & (df['wvf_buy'] >= df['wvf_buy_high'])

    lowest_close = df['Close'].rolling(pd_val).min()
    df['wvf_sell'] = ((df['High'] - lowest_close) / lowest_close) * 100
    df['wvf_sell_upper'] = df['wvf_sell'].rolling(bbl).mean() + mult * df['wvf_sell'].rolling(bbl).std(ddof=0)
    df['wvf_sell_high'] = df['wvf_sell'].rolling(lb).max() * ph
    df['is_sell_euphoria'] = (df['wvf_sell'] >= df['wvf_sell_upper']) & (df['wvf_sell'] >= df['wvf_sell_high'])

    # Active condition in last 5 bars
    df['vix_buy_active'] = df['is_buy_panic'].rolling(6).max() > 0
    df['vix_sell_active'] = df['is_sell_euphoria'].rolling(6).max() > 0

    # 4. Pivots Structure Logic
    lows = df['Low'].values
    highs = df['High'].values
    n = len(df)

    p_low = np.full(n, np.nan)
    p_high = np.full(n, np.nan)

    for i in range(left, n - right):
        if lows[i] == np.min(lows[i - left : i + right + 1]):
            p_low[i + right] = lows[i]
        if highs[i] == np.max(highs[i - left : i + right + 1]):
            p_high[i + right] = highs[i]

    df['pivotLow'] = p_low
    df['pivotHigh'] = p_high

    # Signal Engine Simulation
    lastLow1, lastLow2 = np.nan, np.nan
    lastHigh1, lastHigh2 = np.nan, np.nan
    resistance, support = np.nan, np.nan

    longTriggered = False
    shortTriggered = False

    signals = []

    for i in range(n):
        pl = df['pivotLow'].iloc[i]
        ph_val = df['pivotHigh'].iloc[i]

        if not np.isnan(pl):
            lastLow2 = lastLow1
            lastLow1 = pl
            support = pl

        if not np.isnan(ph_val):
            lastHigh2 = lastHigh1
            lastHigh1 = ph_val
            resistance = ph_val

        validLows = not np.isnan(lastLow1) and not np.isnan(lastLow2)
        validHighs = not np.isnan(lastHigh1) and not np.isnan(lastHigh2)

        isHL = validLows and (lastLow1 > lastLow2)
        isLH = validHighs and (lastHigh1 < lastHigh2)

        close = df['Close'].iloc[i]
        open_p = df['Open'].iloc[i]
        p_low_b = df['p_lower'].iloc[i]
        p_high_b = df['p_upper'].iloc[i]
        rsi = df['RSI'].iloc[i]

        vix_b_active = df['vix_buy_active'].iloc[i]
        vix_s_active = df['vix_sell_active'].iloc[i]

        prev_close = df['Close'].iloc[i-1] if i > 0 else np.nan
        prev_p_lower = df['p_lower'].iloc[i-1] if i > 0 else np.nan
        prev_p_upper = df['p_upper'].iloc[i-1] if i > 0 else np.nan

        crossover_lower = (prev_close <= prev_p_lower) and (close > p_low_b)
        crossunder_upper = (prev_close >= prev_p_upper) and (close < p_high_b)

        longCondition = vix_b_active and isHL and (close > resistance or crossover_lower) and (close > open_p) and (rsi < 50)
        shortCondition = vix_s_active and isLH and (close < support or crossunder_upper) and (close < open_p) and (rsi > 50)

        longSignal = longCondition and not longTriggered
        shortSignal = shortCondition and not shortTriggered

        if longSignal:
            longTriggered = True
            shortTriggered = False
            sl_price = round(close * (1 - slPct / 100), 2)
            signals.append({
                'type': 'BUY',
                'timestamp': df.index[i],
                'price': round(close, 2),
                'sl': sl_price,
                'rsi': round(rsi, 1)
            })
        elif shortSignal:
            shortTriggered = True
            longTriggered = False
            sl_price = round(close * (1 + slPct / 100), 2)
            signals.append({
                'type': 'SELL',
                'timestamp': df.index[i],
                'price': round(close, 2),
                'sl': sl_price,
                'rsi': round(rsi, 1)
            })

    return signals

# ==========================================
# 3. TELEGRAM SENDER & MAIN SCANNER
# ==========================================
def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram Credentials missing!")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    res = requests.post(url, json=payload)
    return res.status_code == 200

def main():
    global sent_signals
    new_signals_count = 0

    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    cutoff_time = now - timedelta(days=DAYS_LOOKBACK)

    for symbol in STOCKS:
        clean_symbol = symbol.replace(".NS", "")
        print(f"Scanning {clean_symbol}...")

        try:
            df = yf.download(symbol, period="5d", interval=TIMEFRAME, progress=False)
            if df.empty:
                continue

            # Flatten MultiIndex Columns if returned by yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index = df.index.tz_convert(ist)
            signals = calculate_strategy(df)

            for sig in signals:
                sig_time = sig['timestamp']

                # Filter signals within the last 2 days
                if sig_time >= cutoff_time:
                    sig_id = f"{clean_symbol}_{sig['type']}_{sig_time.strftime('%Y%m%d_%H%M')}"

                    if sig_id not in sent_signals:
                        time_str = sig_time.strftime("%I:%M %p")

                        if sig['type'] == 'BUY':
                            msg = (
                                f"🟢 <b>VIX MIX BALANCED BUY SIGNAL ({TIMEFRAME})</b> 🟢\n\n"
                                f"<b>Stock:</b> {clean_symbol}\n"
                                f"<b>Price:</b> ₹{sig['price']}\n"
                                f"<b>StopLoss Zone:</b> ₹{sig['sl']}\n"
                                f"<b>RSI:</b> {sig['rsi']}\n"
                                f"<b>Chart Candle:</b> 📊 {time_str}"
                            )
                        else:
                            msg = (
                                f"🔴 <b>VIX MIX BALANCED SELL SIGNAL ({TIMEFRAME})</b> 🔴\n\n"
                                f"<b>Stock:</b> {clean_symbol}\n"
                                f"<b>Price:</b> ₹{sig['price']}\n"
                                f"<b>StopLoss Zone:</b> ₹{sig['sl']}\n"
                                f"<b>RSI:</b> {sig['rsi']}\n"
                                f"<b>Chart Candle:</b> 📊 {time_str}"
                            )

                        print(f"Sending signal for {clean_symbol}...")
                        if send_telegram_message(msg):
                            sent_signals.append(sig_id)
                            new_signals_count += 1

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # Save updated sent signals list
    with open(SENT_SIGNALS_FILE, "w") as f:
        json.dump(sent_signals, f, indent=4)

    print(f"Scan complete. New signals sent: {new_signals_count}")

if __name__ == "__main__":
    main()
