import os
import json
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import requests

# ==============================================================================
# 1. CONFIGURATION & STOCKS / INDEXES LIST
# ==============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")

ENABLE_TELEGRAM_ALERTS = True

SYMBOLS = [
    # --- MAJOR INDEXES ---
    "^NSEI",           # NIFTY 50
    "^NSEBANK",        # BANK NIFTY
    "^CNXFIN",         # FIN NIFTY

    # --- TOP 50 HIGH VOLATILITY / BETA F&O STOCKS ---
    "MCX.NS", "DIXON.NS", "TRENT.NS", "VOLTAS.NS", "COFORGE.NS",
    "PERSISTENT.NS", "OFSS.NS", "BSOFT.NS", "HAL.NS", "BEL.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "MOTHERSON.NS", "POLYCAB.NS", "GLENMARK.NS",
    "ESCORTS.NS", "LTIM.NS", "NATIONALUM.NS", "NMDC.NS", "SAIL.NS",
    "TATASTEEL.NS", "VEDL.NS", "HINDPETRO.NS", "BPCL.NS", "INDUSTOWER.NS",
    "CHAMBLFERT.NS", "METROPOLIS.NS", "BALRAMCHIN.NS", "AUBANK.NS", "BANDHANBNK.NS",
    "IDFCFIRSTB.NS", "RBLBANK.NS", "PFC.NS", "REC.NS", "BHEL.NS",
    "HINDCOPPER.NS", "EXIDEIND.NS", "TATAPOWER.NS", "ABCAPITAL.NS", "JUBLFOOD.NS",
    "M&MFIN.NS", "PEL.NS", "GRANULES.NS", "LAURUSLABS.NS", "DEEPAKNTR.NS",
    "CROMPTON.NS", "AUROPHARMA.NS", "IDEA.NS", "MANAPPURAM.NS", "ONGC.NS"
]

TIMEFRAME = "15m"             # 15 Minute Timeframe

# Squeeze Engine Parameters
BB_LENGTH = 20
BB_MULT = 2.0
KC_LENGTH = 20
KC_MULT = 1.5
EMA_BASE_LEN = 200
USE_TREND = True
RR_RATIO = 1.8

# SIRF ABHI KE LIVE TIME SE 35 MINUTE KI CANDLES KO ALLOW KAREGA (Purane Sab Block)
MAX_CANDLE_AGE_MINUTES = 35

STATE_FILE = "sent_signals.json"

# ==============================================================================
# 2. STATE MANAGEMENT & TELEGRAM SENDER
# ==============================================================================
def load_sent_signals():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return set(data)
                elif isinstance(data, dict):
                    return set(data.keys())
        except Exception:
            return set()
    return set()

def save_sent_signals(sent_set):
    with open(STATE_FILE, "w") as f:
        json.dump(list(sent_set), f, indent=4)

def send_telegram_alert(message: str):
    if not ENABLE_TELEGRAM_ALERTS:
        print(f"[MUTED] Alert generated: {message}")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            print(f"[SENT] {message}")
        else:
            print(f"[TELEGRAM ERROR] {res.text}")
    except Exception as e:
        print(f"[EXCEPTION] {e}")

# ==============================================================================
# 3. TRADINGVIEW EXACT MATH
# ==============================================================================
def calc_linreg(series, length=20):
    x = np.arange(length)
    x_mean = x.mean()
    x_dev = x - x_mean
    var_x = (x_dev**2).sum()

    def get_linreg_val(window):
        if len(window) < length or np.isnan(window).any():
            return np.nan
        y_mean = window.mean()
        slope = np.dot(x_dev, window - y_mean) / var_x
        return y_mean + slope * (length - 1 - x_mean)

    return series.rolling(window=length).apply(get_linreg_val, raw=True)

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['sma'] = df['Close'].rolling(window=BB_LENGTH).mean()
    df['std'] = df['Close'].rolling(window=BB_LENGTH).std()
    df['bb_upper'] = df['sma'] + (BB_MULT * df['std'])
    df['bb_lower'] = df['sma'] - (BB_MULT * df['std'])

    df['tr0'] = abs(df['High'] - df['Low'])
    df['tr1'] = abs(df['High'] - df['Close'].shift(1))
    df['tr2'] = abs(df['Low'] - df['Close'].shift(1))
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=KC_LENGTH).mean()

    df['kc_upper'] = df['sma'] + (df['atr'] * KC_MULT)
    df['kc_lower'] = df['sma'] - (df['atr'] * KC_MULT)

    df['sqzOn'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
    df['sqzOff'] = (df['bb_upper'] > df['kc_upper']) & (df['bb_lower'] < df['kc_lower'])

    df['sqzStart'] = df['sqzOn'] & (~df['sqzOn'].shift(1).fillna(False))
    df['sqzFired'] = df['sqzOff'] & df['sqzOn'].shift(1).fillna(False)

    df['highest_20'] = df['High'].rolling(20).max()
    df['lowest_20'] = df['Low'].rolling(20).min()
    df['hl_avg'] = (df['highest_20'] + df['lowest_20']) / 2
    df['mid_avg'] = (df['hl_avg'] + df['sma']) / 2
    df['mom_raw'] = df['Close'] - df['mid_avg']
    df['momentum'] = calc_linreg(df['mom_raw'], length=20)

    df['ema200'] = df['Close'].ewm(span=EMA_BASE_LEN, adjust=False).mean()

    return df

# ==============================================================================
# 4. SCANNER EXECUTION (STRICT REAL-TIME FILTERING)
# ==============================================================================
def run_scanner():
    sent_signals = load_sent_signals()

    # Current IST Time
    ist_tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    now_ist = datetime.datetime.now(ist_tz)

    for symbol in SYMBOLS:
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="60d", interval=TIMEFRAME)

            if df.empty or len(df) < 200:
                continue

            df = calculate_indicators(df)

            # Check Last 2 Candles
            for i in [-2, -1]:
                candle_dt = df.index[i]
                if candle_dt.tzinfo is None:
                    candle_dt = candle_dt.tz_localize('UTC').tz_convert(ist_tz)
                else:
                    candle_dt = candle_dt.tz_convert(ist_tz)

                # Time Difference in Minutes
                candle_age_minutes = (now_ist - candle_dt).total_seconds() / 60.0

                # STRICT CHECK: Candle agar 35 minute se purani hai, toh IGNORE karo!
                if candle_age_minutes > MAX_CANDLE_AGE_MINUTES or candle_age_minutes < -5:
                    continue

                candle_time = candle_dt.strftime('%Y-%m-%d %H:%M')
                close_p = df['Close'].iloc[i]
                high_p = df['High'].iloc[i]
                low_p = df['Low'].iloc[i]
                ema200 = df['ema200'].iloc[i]
                mom = df['momentum'].iloc[i]

                sqz_start = df['sqzStart'].iloc[i]
                sqz_fired = df['sqzFired'].iloc[i]

                display_name = symbol.replace("^NSEI", "NIFTY 50").replace("^NSEBANK", "BANK NIFTY").replace("^CNXFIN", "FIN NIFTY")

                # 1. LIVE WAIT ALERT
                if sqz_start:
                    if mom > 0 and (not USE_TREND or close_p > ema200):
                        sig_id = f"WAIT_CE_{symbol}_{candle_time}"
                        if sig_id not in sent_signals:
                            msg = (f"⏳ *WAIT CE (Live Squeeze)* | {display_name}\n"
                                   f"Time: {candle_time}\n"
                                   f"Current Price: ₹{close_p:.2f}\n"
                                   f"Status: Squeeze active, prepare for Call breakout!")
                            send_telegram_alert(msg)
                            sent_signals.add(sig_id)

                    elif mom < 0 and (not USE_TREND or close_p < ema200):
                        sig_id = f"WAIT_PE_{symbol}_{candle_time}"
                        if sig_id not in sent_signals:
                            msg = (f"⏳ *WAIT PE (Live Squeeze)* | {display_name}\n"
                                   f"Time: {candle_time}\n"
                                   f"Current Price: ₹{close_p:.2f}\n"
                                   f"Status: Squeeze active, prepare for Put breakdown!")
                            send_telegram_alert(msg)
                            sent_signals.add(sig_id)

                # 2. LIVE BUY CE / BUY PE ALERT
                if sqz_fired:
                    if mom > 0 and (not USE_TREND or close_p > ema200):
                        sl = min(low_p, df['Low'].iloc[i-1])
                        risk = max(close_p - sl, close_p * 0.005)
                        tp = close_p + (risk * RR_RATIO)

                        sig_id = f"BUY_CE_{symbol}_{candle_time}"
                        if sig_id not in sent_signals:
                            msg = (f"🚀 *BUY CE NOW (Live Call Entry)* | {display_name}\n"
                                   f"Time: {candle_time}\n"
                                   f"Entry Price: ₹{close_p:.2f}\n"
                                   f"Stop Loss: ₹{sl:.2f}\n"
                                   f"Target (1:{RR_RATIO}): ₹{tp:.2f}")
                            send_telegram_alert(msg)
                            sent_signals.add(sig_id)

                    elif mom < 0 and (not USE_TREND or close_p < ema200):
                        sl = max(high_p, df['High'].iloc[i-1])
                        risk = max(sl - close_p, close_p * 0.005)
                        tp = close_p - (risk * RR_RATIO)

                        sig_id = f"BUY_PE_{symbol}_{candle_time}"
                        if sig_id not in sent_signals:
                            msg = (f"💥 *BUY PE NOW / SELL (Live Put Entry)* | {display_name}\n"
                                   f"Time: {candle_time}\n"
                                   f"Entry Price: ₹{close_p:.2f}\n"
                                   f"Stop Loss: ₹{sl:.2f}\n"
                                   f"Target (1:{RR_RATIO}): ₹{tp:.2f}")
                            send_telegram_alert(msg)
                            sent_signals.add(sig_id)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    save_sent_signals(sent_signals)

if __name__ == "__main__":
    run_scanner()
