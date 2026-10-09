import os
import sys
import pytz
import datetime
import requests
import numpy as np
import pandas as pd
import yfinance as yf

# --- TELEGRAM CONFIGURATION ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- UNIVERSE: MAJOR INDEXES + TOP 100 VOLATILE F&O STOCKS (NSE) ---
SYMBOLS = [
    # Major Indexes
    "^NSEI", "^NSEBANK", "^CNXIT", "^NSEMDCP50",
    # Top Volatile F&O Stocks
    "RELIANCE.NS", "TATAMOTORS.NS", "SBIN.NS", "ICICIBANK.NS", "HDFCBANK.NS",
    "INFY.NS", "TCS.NS", "BAJFINANCE.NS", "TATASTEEL.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "AXISBANK.NS", "KOTAKBANK.NS", "MARUTI.NS", "MM.NS",
    "SUNPHARMA.NS", "HINDALCO.NS", "VEDL.NS", "BHARATFORG.NS", "DLF.NS",
    "BANDHANBNK.NS", "ZEEL.NS", "IDEA.NS", "CANBK.NS", "PNB.NS",
    "COALINDIA.NS", "CHOLAFIN.NS", "INDUSINDBK.NS", "JSWSTEEL.NS", "TATACHEM.NS",
    "TATAPOWER.NS", "BEL.NS", "HAL.NS", "RECLTD.NS", "PFC.NS",
    "DIXON.NS", "PERSISTENT.NS", "COFORGE.NS", "LT.NS", "SRF.NS",
    "ASTRAL.NS", "JUBLFOOD.NS", "POLYCAB.NS", "MCX.NS", "MOTHERSON.NS",
    "NMDC.NS", "NATIONALUM.NS", "SAIL.NS", "IRCTC.NS", "RVNL.NS",
    "HDFCLIFE.NS", "SBILIFE.NS", "BALKRISIND.NS", "ESCORTS.NS", "TIINDIA.NS",
    "TRENT.NS", "AUROPHARMA.NS", "LUPIN.NS", "CIPLA.NS", "GLENMARK.NS",
    "GRANULES.NS", "BIOCON.NS", "SYNGENE.NS", "TATACOMM.NS", "INDIGO.NS", "AMBUJACEM.NS",
    "GODREJPROP.NS", "OBEROIRLTY.NS", "BSOFT.NS", "KPITTECH.NS", "LTTS.NS",
    "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "PIDILITIND.NS", "BERGEPAINT.NS",
    "ASIANPAINT.NS", "TITAN.NS", "TATAELXSI.NS", "MUTHOOTFIN.NS", "MANAPPURAM.NS",
    "PEL.NS", "GNFC.NS", "METROPOLIS.NS", "DRREDDY.NS", "ABFRL.NS",
    "HINDPETRO.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "ONGC.NS",
    "PETRONET.NS", "BHEL.NS", "CONCOR.NS", "VOLTAS.NS", "UPL.NS",
    "AARTIIND.NS", "APOLLOTYRE.NS", "LICHSGFIN.NS", "EXIDEIND.NS", "IPCALAB.NS"
]

def get_tradingview_url(ticker):
    mapping = {
        "^NSEI": "NSE:NIFTY",
        "^NSEBANK": "NSE:BANKNIFTY",
        "^CNXIT": "NSE:CNXIT",
        "^NSEMDCP50": "NSE:NIFTY_MID_SELECT"
    }
    if ticker in mapping:
        tv_symbol = mapping[ticker]
    else:
        clean = ticker.replace(".NS", "").replace("^", "")
        tv_symbol = f"NSE:{clean}"
    return f"https://in.tradingview.com/chart/?symbol={tv_symbol}"

def resample_to_tradingview_30m(df, ist):
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert(ist)
    else:
        df.index = df.index.tz_convert(ist)

    # Market Hours Only (09:15 to 15:30 IST)
    df = df.between_time('09:15', '15:30')

    # Resample 5m -> 30m starting at 09:15 AM IST (Exact TradingView Candles)
    resampled = df.resample('30min', offset='15min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    return resampled

def calculate_signals(df, length=20, mult_bb=2.0, mult_kc=1.5):
    # 1. Bollinger Bands
    df['sma'] = df['Close'].rolling(window=length).mean()
    df['std'] = df['Close'].rolling(window=length).std(ddof=0)
    df['bb_upper'] = df['sma'] + (mult_bb * df['std'])
    df['bb_lower'] = df['sma'] - (mult_bb * df['std'])

    # 2. Keltner Channels (EMA + RMA ATR)
    df['kc_ema'] = df['Close'].ewm(span=length, adjust=False).mean()
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/length, adjust=False).mean()

    df['kc_upper'] = df['kc_ema'] + (df['atr'] * mult_kc)
    df['kc_lower'] = df['kc_ema'] - (df['atr'] * mult_kc)

    # 3. Squeeze Conditions
    df['is_squeezed'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
    df['squeeze_start'] = df['is_squeezed'] & (~df['is_squeezed'].shift(1).fillna(False))
    df['squeeze_release'] = (~df['is_squeezed']) & (df['is_squeezed'].shift(1).fillna(False))

    # 4. Momentum Oscillator
    highest_h = df['High'].rolling(window=length).max()
    lowest_l = df['Low'].rolling(window=length).min()
    avg_hl = (highest_h + lowest_l) / 2.0
    avg_val = (avg_hl + df['kc_ema']) / 2.0
    val = df['Close'] - avg_val

    x = np.arange(length)
    x_mean = (length - 1) / 2.0
    x_var = np.sum((x - x_mean) ** 2)

    def calc_linreg(window):
        y_mean = np.mean(window)
        slope = np.sum((x - x_mean) * (window - y_mean)) / x_var
        return y_mean + slope * (length - 1 - x_mean)

    df['mom'] = val.rolling(window=length).apply(calc_linreg, raw=True)

    # Signal Columns
    df['buy_signal'] = False
    df['sell_signal'] = False
    df['sq_buy_signal'] = False
    df['sq_sell_signal'] = False

    sq_start_high = None
    sq_start_low = None

    for i in range(len(df)):
        # Squeeze Release Signals
        if df['squeeze_release'].iloc[i]:
            if df['mom'].iloc[i] > 0 and df['Close'].iloc[i] > df['kc_ema'].iloc[i]:
                df.iloc[i, df.columns.get_loc('buy_signal')] = True
            elif df['mom'].iloc[i] < 0 and df['Close'].iloc[i] < df['kc_ema'].iloc[i]:
                df.iloc[i, df.columns.get_loc('sell_signal')] = True

        # Track Squeeze High/Low Level
        if df['squeeze_start'].iloc[i]:
            sq_start_high = df['High'].iloc[i]
            sq_start_low = df['Low'].iloc[i]

        # Squeeze High/Low Breakout Signals
        if sq_start_high is not None and sq_start_low is not None:
            c = df['Close'].iloc[i]
            o = df['Open'].iloc[i]

            if c > o and c > sq_start_high:
                df.iloc[i, df.columns.get_loc('sq_buy_signal')] = True
                sq_start_high = None
                sq_start_low = None
            elif c < o and c < sq_start_low:
                df.iloc[i, df.columns.get_loc('sq_sell_signal')] = True
                sq_start_high = None
                sq_start_low = None

    return df

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[LOG] Telegram Token/Chat ID Missing!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": False
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code != 200:
            print(f"[ERROR] Telegram API Error: {res.text}")
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram message: {e}")

def main():
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.datetime.now(ist)
    today_date = now_ist.date()

    run_all_today = "--all-today" in sys.argv

    print(f"--- Scan Started at {now_ist.strftime('%Y-%m-%d %H:%M:%S')} IST ---")
    signals_found = 0

    for ticker in SYMBOLS:
        try:
            tk = yf.Ticker(ticker)
            # Fetch 5m data for accurate resampling
            raw_data = tk.history(period="7d", interval="5m")
            
            if raw_data.empty or len(raw_data) < 100:
                continue

            # Resample to TV 30m candles
            df_30m = resample_to_tradingview_30m(raw_data, ist)
            df = calculate_signals(df_30m)

            df_today = df[df.index.date == today_date].copy()
            if df_today.empty:
                continue

            squeeze_count = 0
            
            for i, (timestamp, row) in enumerate(df_today.iterrows()):
                if row['squeeze_start']:
                    squeeze_count += 1

                is_recent = (len(df_today) - i) <= 2

                if not (run_all_today or is_recent):
                    continue

                clean_symbol = ticker.replace(".NS", "").replace("^", "")
                candle_time_str = timestamp.strftime('%I:%M %p')
                close_price = round(row['Close'], 2)
                tv_link = get_tradingview_url(ticker)

                msg = None

                if row['squeeze_start']:
                    msg = (
                        f"🟡 <b>SQUEEZE STARTED ALERT</b> 🟡\n\n"
                        f"📌 <b>Stock/Index:</b> {clean_symbol}\n"
                        f"⏱ <b>Timeframe:</b> 30 Min\n"
                        f"🕒 <b>Candle Time:</b> {candle_time_str}\n"
                        f"🔢 <b>Today's Squeeze No:</b> #{squeeze_count}\n"
                        f"💰 <b>Close Price:</b> ₹{close_price}\n\n"
                        f"⚡ <i>Status: BB inside Keltner Channel!</i>\n"
                        f"📈 <a href='{tv_link}'>Open Chart in TradingView</a>"
                    )
                elif row['buy_signal']:
                    msg = (
                        f"🟢 <b>BUY SIGNAL (Squeeze Blast)</b> 🟢\n\n"
                        f"📌 <b>Stock/Index:</b> {clean_symbol}\n"
                        f"⏱ <b>Timeframe:</b> 30 Min\n"
                        f"🕒 <b>Candle Time:</b> {candle_time_str}\n"
                        f"💰 <b>Close Price:</b> ₹{close_price}\n\n"
                        f"🚀 <i>Status: Squeeze Released + Bullish Momentum!</i>\n"
                        f"📈 <a href='{tv_link}'>Open Chart in TradingView</a>"
                    )
                elif row['sell_signal']:
                    msg = (
                        f"🔴 <b>SELL SIGNAL (Squeeze Blast)</b> 🔴\n\n"
                        f"📌 <b>Stock/Index:</b> {clean_symbol}\n"
                        f"⏱ <b>Timeframe:</b> 30 Min\n"
                        f"🕒 <b>Candle Time:</b> {candle_time_str}\n"
                        f"💰 <b>Close Price:</b> ₹{close_price}\n\n"
                        f"🔻 <i>Status: Squeeze Released + Bearish Momentum!</i>\n"
                        f"📈 <a href='{tv_link}'>Open Chart in TradingView</a>"
                    )
                elif row['sq_buy_signal']:
                    msg = (
                        f"🚀 <b>BUY SIGNAL (High Breakout)</b> 🚀\n\n"
                        f"📌 <b>Stock/Index:</b> {clean_symbol}\n"
                        f"⏱ <b>Timeframe:</b> 30 Min\n"
                        f"🕒 <b>Candle Time:</b> {candle_time_str}\n"
                        f"💰 <b>Close Price:</b> ₹{close_price}\n\n"
                        f"🔥 <i>Status: Closed above Squeeze Candle High!</i>\n"
                        f"📈 <a href='{tv_link}'>Open Chart in TradingView</a>"
                    )
                elif row['sq_sell_signal']:
                    msg = (
                        f"🔻 <b>SELL SIGNAL (Low Breakout)</b> 🔻\n\n"
                        f"📌 <b>Stock/Index:</b> {clean_symbol}\n"
                        f"⏱ <b>Timeframe:</b> 30 Min\n"
                        f"🕒 <b>Candle Time:</b> {candle_time_str}\n"
                        f"💰 <b>Close Price:</b> ₹{close_price}\n\n"
                        f"💥 <i>Status: Closed below Squeeze Candle Low!</i>\n"
                        f"📈 <a href='{tv_link}'>Open Chart in TradingView</a>"
                    )

                if msg:
                    signals_found += 1
                    send_telegram_message(msg)
                    print(f"✅ Alert sent for {clean_symbol} at {candle_time_str}")

        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")

    print(f"--- Scan Finished. Total Signals Sent: {signals_found} ---")

if __name__ == "__main__":
    main()
