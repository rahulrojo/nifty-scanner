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

# Master Toggle: Set to False to disable Telegram alerts
ENABLE_TELEGRAM_ALERTS = True

# Major Indexes + Top 50 High Volatility / Beta F&O Stocks
SYMBOLS = [
    # --- MAJOR INDEXES ---
    "^NSEI",           # NIFTY 50
    "^NSEBANK",        # BANK NIFTY
    "^CNXFIN",         # FIN NIFTY

    # --- TOP 50 HIGH VOLATILITY / BETA F&O STOCKS ---
    "MCX.NS",
    "DIXON.NS",
    "TRENT.NS",
    "VOLTAS.NS",
    "COFORGE.NS",
    "PERSISTENT.NS",
    "OFSS.NS",
    "BSOFT.NS",
    "HAL.NS",
    "BEL.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "MOTHERSON.NS",
    "POLYCAB.NS",
    "GLENMARK.NS",
    "ESCORTS.NS",
    "LTIM.NS",
    "NATIONALUM.NS",
    "NMDC.NS",
    "SAIL.NS",
    "TATASTEEL.NS",
    "VEDL.NS",
    "HINDPETRO.NS",
    "BPCL.NS",
    "INDUSTOWER.NS",
    "CHAMBLFERT.NS",
    "METROPOLIS.NS",
    "BALRAMCHIN.NS",
    "AUBANK.NS",
    "BANDHANBNK.NS",
    "IDFCFIRSTB.NS",
    "RBLBANK.NS",
    "PFC.NS",
    "REC.NS",
    "BHEL.NS",
    "HINDCOPPER.NS",
    "EXIDEIND.NS",
    "TATAPOWER.NS",
    "ABCAPITAL.NS",
    "JUBLFOOD.NS",
    "M&MFIN.NS",
    "PEL.NS",
    "GRANULES.NS",
    "LAURUSLABS.NS",
    "DEEPAKNTR.NS",
    "CROMPTON.NS",
    "AUROPHARMA.NS",
    "IDEA.NS",
    "MANAPPURAM.NS",
    "ONGC.NS"
]

TIMEFRAME = "15m"             # 15 Minute Timeframe
FETCH_DAYS = 3                # Process Signals for Last 3 Days

# Squeeze Engine Parameters
BB_LENGTH = 20
BB_MULT = 2.0
KC_LENGTH = 20
KC_MULT = 1.5
EMA_BASE_LEN = 200
USE_TREND = True
MAX_WAIT_BARS = 3
RR_RATIO = 1.8

STATE_FILE = "sent_signals.json"

# ==============================================================================
# 2. STATE MANAGEMENT & TELEGRAM SENDER
# ==============================================================================
def load_sent_signals():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_sent_signals(sent_list):
    with open(STATE_FILE, "w") as f:
        json.dump(sent_list, f, indent=4)

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
# 3. TRADINGVIEW EXACT MATH (LINREG + INDICATORS)
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
    # Bollinger Bands
    df['sma'] = df['Close'].rolling(window=BB_LENGTH).mean()
    df['std'] = df['Close'].rolling(window=BB_LENGTH).std()
    df['bb_upper'] = df['sma'] + (BB_MULT * df['std'])
    df['bb_lower'] = df['sma'] - (BB_MULT * df['std'])

    # Keltner Channels
    df['tr0'] = abs(df['High'] - df['Low'])
    df['tr1'] = abs(df['High'] - df['Close'].shift(1))
    df['tr2'] = abs(df['Low'] - df['Close'].shift(1))
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=KC_LENGTH).mean()

    df['kc_upper'] = df['sma'] + (df['atr'] * KC_MULT)
    df['kc_lower'] = df['sma'] - (df['atr'] * KC_MULT)

    # Squeeze States
    df['sqzOn'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
    df['sqzOff'] = (df['bb_upper'] > df['kc_upper']) & (df['bb_lower'] < df['kc_lower'])
    df['sqzFired'] = df['sqzOff'] & df['sqzOn'].shift(1)

    # Pine Script Momentum Formula
    df['highest_20'] = df['High'].rolling(20).max()
    df['lowest_20'] = df['Low'].rolling(20).min()
    df['hl_avg'] = (df['highest_20'] + df['lowest_20']) / 2
    df['mid_avg'] = (df['hl_avg'] + df['sma']) / 2
    df['mom_raw'] = df['Close'] - df['mid_avg']
    df['momentum'] = calc_linreg(df['mom_raw'], length=20)

    # 200 EMA Baseline
    df['ema200'] = df['Close'].ewm(span=EMA_BASE_LEN, adjust=False).mean()

    # Filters
    trend_ce = (~USE_TREND) | (df['Close'] > df['ema200'])
    trend_pe = (~USE_TREND) | (df['Close'] < df['ema200'])

    df['rawCE'] = df['sqzFired'] & (df['momentum'] > 0) & trend_ce
    df['rawPE'] = df['sqzFired'] & (df['momentum'] < 0) & trend_pe

    return df

# ==============================================================================
# 4. MULTI-STOCK SCANNER EXECUTION
# ==============================================================================
def run_scanner():
    sent_signals = load_sent_signals()
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=FETCH_DAYS)

    for symbol in SYMBOLS:
        print(f"\n--- Scanning {symbol} ---")
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="60d", interval=TIMEFRAME)

            if df.empty or len(df) < 200:
                print(f"Skipping {symbol}: Insufficient historical data")
                continue

            df = calculate_indicators(df)
            df_slice = df[df.index >= cutoff_date.strftime('%Y-%m-%d')]

            waiting_ce = False
            waiting_pe = False
            wait_count = 0
            range_high = 0.0
            range_low = 0.0

            for i in range(1, len(df_slice)):
                candle_time = df_slice.index[i].strftime('%Y-%m-%d %H:%M')
                close_p = df_slice['Close'].iloc[i]
                prev_close = df_slice['Close'].iloc[i-1]
                high_p = df_slice['High'].iloc[i]
                low_p = df_slice['Low'].iloc[i]

                raw_ce = df_slice['rawCE'].iloc[i]
                raw_pe = df_slice['rawPE'].iloc[i]

                display_name = symbol.replace("^NSEI", "NIFTY 50").replace("^NSEBANK", "BANK NIFTY").replace("^CNXFIN", "FIN NIFTY")

                # --------------------------------------------------------------
                # STEP 1: CHECK ENTRY CONFIRMATION FOR EXISTING WAIT SETUP
                # --------------------------------------------------------------
                if waiting_ce:
                    wait_count += 1
                    # Breakout confirmation (Close crosses above range_high)
                    if close_p > range_high and prev_close <= range_high:
                        sl = range_low
                        risk = close_p - sl
                        tp = close_p + (risk * RR_RATIO)
                        
                        sig_id = f"BUY_CE_{symbol}_{candle_time}"
                        if sig_id not in sent_signals:
                            msg = (f"🚀 *BUY CE NOW (Call Entry)* | {display_name}\n"
                                   f"Time: {candle_time}\n"
                                   f"Entry Price: ₹{close_p:.2f}\n"
                                   f"Stop Loss: ₹{sl:.2f}\n"
                                   f"Target (1:{RR_RATIO}): ₹{tp:.2f}")
                            send_telegram_alert(msg)
                            sent_signals.append(sig_id)
                        waiting_ce = False

                    # Invalidation or Timeout
                    elif close_p < range_low or wait_count > MAX_WAIT_BARS:
                        waiting_ce = False

                elif waiting_pe:
                    wait_count += 1
                    # Breakdown confirmation (Close crosses below range_low)
                    if close_p < range_low and prev_close >= range_low:
                        sl = range_high
                        risk = sl - close_p
                        tp = close_p - (risk * RR_RATIO)
                        
                        sig_id = f"BUY_PE_{symbol}_{candle_time}"
                        if sig_id not in sent_signals:
                            msg = (f"💥 *BUY PE NOW / SELL (Put Entry)* | {display_name}\n"
                                   f"Time: {candle_time}\n"
                                   f"Entry Price: ₹{close_p:.2f}\n"
                                   f"Stop Loss: ₹{sl:.2f}\n"
                                   f"Target (1:{RR_RATIO}): ₹{tp:.2f}")
                            send_telegram_alert(msg)
                            sent_signals.append(sig_id)
                        waiting_pe = False

                    # Invalidation or Timeout
                    elif close_p > range_high or wait_count > MAX_WAIT_BARS:
                        waiting_pe = False

                # --------------------------------------------------------------
                # STEP 2: CHECK NEW SQUEEZE BREAKOUT TRIGGER (WAIT ALERT)
                # --------------------------------------------------------------
                if not waiting_ce and not waiting_pe:
                    if raw_ce:
                        range_high = max(high_p, df_slice['High'].iloc[i-1])
                        range_low = min(low_p, df_slice['Low'].iloc[i-1])
                        waiting_ce = True
                        wait_count = 0
                        
                        sig_id = f"WAIT_CE_{symbol}_{candle_time}"
                        if sig_id not in sent_signals:
                            msg = (f"⏳ *WAIT CE (Call Setup)* | {display_name}\n"
                                   f"Time: {candle_time}\n"
                                   f"Buy Trigger: Above ₹{range_high:.2f}\n"
                                   f"SL Level: ₹{range_low:.2f}")
                            send_telegram_alert(msg)
                            sent_signals.append(sig_id)

                    elif raw_pe:
                        range_high = max(high_p, df_slice['High'].iloc[i-1])
                        range_low = min(low_p, df_slice['Low'].iloc[i-1])
                        waiting_pe = True
                        wait_count = 0
                        
                        sig_id = f"WAIT_PE_{symbol}_{candle_time}"
                        if sig_id not in sent_signals:
                            msg = (f"⏳ *WAIT PE (Put Setup)* | {display_name}\n"
                                   f"Time: {candle_time}\n"
                                   f"Sell Trigger: Below ₹{range_low:.2f}\n"
                                   f"SL Level: ₹{range_high:.2f}")
                            send_telegram_alert(msg)
                            sent_signals.append(sig_id)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    save_sent_signals(sent_signals)

if __name__ == "__main__":
    run_scanner()
