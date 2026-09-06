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

# Top 50 High-Volume F&O Stocks + Major NSE Indexes
SYMBOLS = [
    # --- MAJOR INDEXES ---
    "^NSEI",           # NIFTY 50
    "^NSEBANK",        # BANK NIFTY
    "^CNXFIN",         # FIN NIFTY

    # --- TOP 50 HIGH VOLUME / LIQUID F&O STOCKS ---
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "DIXON.NS",
    "TATAMOTORS.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TATASTEEL.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "COALINDIA.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "JSWSTEEL.NS",
    "HINDALCO.NS",
    "TRENT.NS",
    "BEL.NS",
    "HAL.NS",
    "BHEL.NS",
    "REC.NS",
    "PFC.NS",
    "ZOMATO.NS",
    "JIOFIN.NS",
    "DLF.NS",
    "VEDL.NS",
    "PERSISTENT.NS",
    "COFORGE.NS",
    "POLYCAB.NS",
    "CHOLAFIN.NS",
    "MUTHOOTFIN.NS",
    "CANBK.NS",
    "INDUSINDBK.NS",
    "HEROMOTOCO.NS",
    "TVSMOTOR.NS",
    "EICHERMOT.NS",
    "MOTHERSON.NS"
]

TIMEFRAME = "15m"             # 15 Minute Timeframe
FETCH_DAYS = 3                # Last 3 Days Historical Signals

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
# 2. STATE MANAGEMENT (JSON)
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

# ==============================================================================
# 3. TELEGRAM ALERT SENDER
# ==============================================================================
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
# 4. SQUEEZE ENGINE INDICATORS
# ==============================================================================
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
    df['sqzFired'] = df['sqzOff'] & df['sqzOn'].shift(1)

    df['hl_avg'] = (df['High'].rolling(20).max() + df['Low'].rolling(20).min()) / 2
    df['mid_avg'] = (df['hl_avg'] + df['sma']) / 2
    df['momentum'] = df['Close'] - df['mid_avg']

    df['ema200'] = df['Close'].ewm(span=EMA_BASE_LEN, adjust=False).mean()

    trend_ce = (~USE_TREND) | (df['Close'] > df['ema200'])
    trend_pe = (~USE_TREND) | (df['Close'] < df['ema200'])

    df['rawCE'] = df['sqzFired'] & (df['momentum'] > 0) & trend_ce
    df['rawPE'] = df['sqzFired'] & (df['momentum'] < 0) & trend_pe

    return df

# ==============================================================================
# 5. MULTI-STOCK SCANNER EXECUTION
# ==============================================================================
def run_scanner():
    sent_signals = load_sent_signals()
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=FETCH_DAYS)

    for symbol in SYMBOLS:
        print(f"\n--- Scanning {symbol} ---")
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="5d", interval=TIMEFRAME)
            df = df[df.index >= cutoff_date.strftime('%Y-%m-%d')]

            if df.empty or len(df) < 20:
                print(f"Skipping {symbol}: Insufficient data")
                continue

            df = calculate_indicators(df)

            waiting_ce = False
            waiting_pe = False
            wait_count = 0
            range_high = 0.0
            range_low = 0.0

            for i in range(1, len(df)):
                candle_time = df.index[i].strftime('%Y-%m-%d %H:%M')
                close_p = df['Close'].iloc[i]
                high_p = df['High'].iloc[i]
                low_p = df['Low'].iloc[i]

                raw_ce = df['rawCE'].iloc[i]
                raw_pe = df['rawPE'].iloc[i]

                # Clean display name for indexes
                display_name = symbol.replace("^NSEI", "NIFTY 50").replace("^NSEBANK", "BANK NIFTY").replace("^CNXFIN", "FIN NIFTY")

                # --------------------------------------------------------------
                # STEP 1: WARNING MESSAGES (WAIT CE / WAIT PE)
                # --------------------------------------------------------------
                if raw_ce and not waiting_ce and not waiting_pe:
                    range_high = max(high_p, df['High'].iloc[i-1])
                    range_low = min(low_p, df['Low'].iloc[i-1])
                    waiting_ce, waiting_pe, wait_count = True, False, 0
                    
                    sig_id = f"WAIT_CE_{symbol}_{candle_time}"
                    if sig_id not in sent_signals:
                        msg = (f"⏳ *WAIT CE (Call Setup)* | {display_name}\n"
                               f"Time: {candle_time}\n"
                               f"Buy Trigger: Above ₹{range_high:.2f}\n"
                               f"SL Level: ₹{range_low:.2f}")
                        send_telegram_alert(msg)
                        sent_signals.append(sig_id)

                elif raw_pe and not waiting_pe and not waiting_ce:
                    range_high = max(high_p, df['High'].iloc[i-1])
                    range_low = min(low_p, df['Low'].iloc[i-1])
                    waiting_pe, waiting_ce, wait_count = True, False, 0
                    
                    sig_id = f"WAIT_PE_{symbol}_{candle_time}"
                    if sig_id not in sent_signals:
                        msg = (f"⏳ *WAIT PE (Put Setup)* | {display_name}\n"
                               f"Time: {candle_time}\n"
                               f"Sell Trigger: Below ₹{range_low:.2f}\n"
                               f"SL Level: ₹{range_high:.2f}")
                        send_telegram_alert(msg)
                        sent_signals.append(sig_id)

                if waiting_ce or waiting_pe:
                    wait_count += 1

                if wait_count > MAX_WAIT_BARS:
                    waiting_ce, waiting_pe = False, False

                # --------------------------------------------------------------
                # STEP 2: ENTRY CONFIRMATION MESSAGES (BUY CE / BUY PE)
                # --------------------------------------------------------------
                if waiting_ce and close_p > range_high:
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

                if waiting_pe and close_p < range_low:
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

                # Invalidation
                if waiting_ce and close_p < range_low:
                    waiting_ce = False
                if waiting_pe and close_p > range_high:
                    waiting_pe = False

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    save_sent_signals(sent_signals)

if __name__ == "__main__":
    run_scanner()
