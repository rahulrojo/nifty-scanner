import os
import json
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import requests

# Pandas future warning hide karne ke liye
pd.set_option('future.no_silent_downcasting', True)

# ==============================================================================
# 1. CONFIGURATION & ALL INDEXES + ALL F&O STOCKS
# ==============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")

ENABLE_TELEGRAM_ALERTS = True
TIMEFRAME = "15m"

# Major Indexes + High Liquid F&O Stocks
SYMBOLS = [
    # --- INDEXES ---
    "^NSEI", "^NSEBANK", "^CNXFIN", "^NSEMDCP50",

    # --- TOP F&O STOCKS ---
    "AARTIIND.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABFRL.NS", 
    "ACC.NS", "ADANIENT.NS", "ADANIPORTS.NS", "ALKEM.NS", "AMBUJACEM.NS", 
    "APOLLOHOSP.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTRAL.NS", 
    "ATUL.NS", "AUBANK.NS", "AUROPHARMA.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", 
    "BAJAJFINSV.NS", "BAJFINANCE.NS", "BALKRISIND.NS", "BALRAMCHIN.NS", "BANDHANBNK.NS", 
    "BANKBARODA.NS", "BATAINDIA.NS", "BEL.NS", "BERGEPAINT.NS", "BHARATFORG.NS", 
    "BHARTIARTL.NS", "BHEL.NS", "BIOCON.NS", "BSOFT.NS", "BPCL.NS", 
    "BRITANNIA.NS", "CANBK.NS", "CANFINHOME.NS", "CHAMBLFERT.NS", "CHOLAFIN.NS", 
    "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", "CONCOR.NS", 
    "COROMANDEL.NS", "CROMPTON.NS", "CUB.NS", "CUMMINSIND.NS", "DABUR.NS", 
    "DALBHARAT.NS", "DEEPAKNTR.NS", "DIVISLAB.NS", "DIXON.NS", "DLF.NS", 
    "DRREDDY.NS", "EICHERMOT.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", 
    "GAIL.NS", "GLENMARK.NS", "GMRINFRA.NS", "GNFC.NS", "GODREJCP.NS", 
    "GODREJPROP.NS", "GRANULES.NS", "GRASIM.NS", "GUJGASLTD.NS", "HAL.NS", 
    "HAVELLS.NS", "HCLTECH.NS", "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", 
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDCOPPER.NS", "HINDPETRO.NS", "HINDUNILVR.NS", 
    "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDEA.NS", "IDFCFIRSTB.NS", 
    "IEX.NS", "IGL.NS", "INDHOTEL.NS", "INDIACEM.NS", "INDIAMART.NS", 
    "INDIGO.NS", "INDUSINDBK.NS", "INDUSTOWER.NS", "INFY.NS", "IOC.NS", 
    "IPCALAB.NS", "IRCTC.NS", "ITC.NS", "JINDALSTEL.NS", "JKCEMENT.NS", 
    "JSWSTEEL.NS", "JUBLFOOD.NS", "KOTAKBANK.NS", "LALPATHLAB.NS", "LAURUSLABS.NS", 
    "LICHSGFIN.NS", "LTIM.NS", "LT.NS", "LTTS.NS", "LUPIN.NS", 
    "M&M.NS", "M&MFIN.NS", "MANAPPURAM.NS", "MARICO.NS", "MARUTI.NS", 
    "MCDOWELL-N.NS", "MCX.NS", "METROPOLIS.NS", "MFSL.NS", "MGL.NS", 
    "MOTHERSON.NS", "MPHASIS.NS", "MRF.NS", "MUTHOOTFIN.NS", "NATIONALUM.NS", 
    "NAUKRI.NS", "NAVINFLUOR.NS", "NESTLEIND.NS", "NMDC.NS", "NTPC.NS", 
    "OBEROIRLTY.NS", "OFSS.NS", "ONGC.NS", "PAGEIND.NS", "PERSISTENT.NS", 
    "PETRONET.NS", "PFC.NS", "PIDILITIND.NS", "PIIND.NS", "PNB.NS", 
    "POLYCAB.NS", "POWERGRID.NS", "PVRINOX.NS", "RAMCOCEM.NS", "RBLBANK.NS", 
    "REC.NS", "RELIANCE.NS", "SAIL.NS", "SBICARD.NS", "SBILIFE.NS", 
    "SBIN.NS", "SHREECEM.NS", "SHRIRAMFIN.NS", "SIEMENS.NS", "SRF.NS", 
    "SUNPHARMA.NS", "SUNTV.NS", "SYNGENE.NS", "TATACHEMICALS.NS", "TATACOMM.NS", 
    "TATACONSUM.NS", "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", 
    "TECHM.NS", "TITAN.NS", "TORNTPHARM.NS", "TRENT.NS", "TVSMOTOR.NS", 
    "UBL.NS", "ULTRACEMCO.NS", "UPL.NS", "VEDL.NS", "VOLTAS.NS", 
    "WIPRO.NS", "ZEEL.NS"
]

# Indicator Inputs (TradingView Code Matching)
LENGTH = 20
MULT_BB = 2.0
MULT_KC = 1.5

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
        print(f"[MUTED] Alert: {message}")
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
            print(f"[SENT TO TELEGRAM] {message}")
        else:
            print(f"[TELEGRAM ERROR] {res.text}")
    except Exception as e:
        print(f"[EXCEPTION] {e}")

# ==============================================================================
# 3. PINE SCRIPT MATH EXACT MATCHING
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
    # 1. Bollinger Bands
    df['bbMid'] = df['Close'].rolling(window=LENGTH).mean()
    df['std'] = df['Close'].rolling(window=LENGTH).std()
    df['bbUpper'] = df['bbMid'] + (MULT_BB * df['std'])
    df['bbLower'] = df['bbMid'] - (MULT_BB * df['std'])

    # 2. Keltner Channels (EMA + Wilder's ATR)
    df['kcEma'] = df['Close'].ewm(span=LENGTH, adjust=False).mean()
    
    df['tr0'] = abs(df['High'] - df['Low'])
    df['tr1'] = abs(df['High'] - df['Close'].shift(1))
    df['tr2'] = abs(df['Low'] - df['Close'].shift(1))
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df['kcRange'] = df['tr'].ewm(alpha=1/LENGTH, adjust=False).mean() # Pine ta.atr

    df['kcUpper'] = df['kcEma'] + (df['kcRange'] * MULT_KC)
    df['kcLower'] = df['kcEma'] - (df['kcRange'] * MULT_KC)

    # 3. Squeeze Conditions
    df['isSqueezed'] = (df['bbUpper'] < df['kcUpper']) & (df['bbLower'] > df['kcLower'])
    
    # PEHLA BLACK SQUARE (Squeeze Start)
    df['squeezeStart'] = df['isSqueezed'] & (~df['isSqueezed'].shift(1).fillna(False))
    
    # Squeeze Release
    df['squeezeRelease'] = df['isSqueezed'].shift(1).fillna(False) & (~df['isSqueezed'])

    # 4. Pine Script Momentum
    df['highest_20'] = df['High'].rolling(LENGTH).max()
    df['lowest_20'] = df['Low'].rolling(LENGTH).min()
    df['hl_avg'] = (df['highest_20'] + df['lowest_20']) / 2
    df['mid_avg'] = (df['hl_avg'] + df['kcEma']) / 2
    df['mom_raw'] = df['Close'] - df['mid_avg']
    df['momentum'] = calc_linreg(df['mom_raw'], length=LENGTH)

    # 5. Buy / Sell Signals (Pine Script Rules)
    df['buySignal'] = df['squeezeRelease'] & (df['momentum'] > 0) & (df['Close'] > df['kcEma'])
    df['sellSignal'] = df['squeezeRelease'] & (df['momentum'] < 0) & (df['Close'] < df['kcEma'])

    return df

# ==============================================================================
# 4. SCANNER EXECUTION (TODAY'S REALTIME CANDLES)
# ==============================================================================
def run_scanner():
    sent_signals = load_sent_signals()

    ist_tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    now_ist = datetime.datetime.now(ist_tz)
    today_date = now_ist.date()

    print(f"=== Running Indicator Mapper Scanner | IST: {now_ist.strftime('%Y-%m-%d %H:%M:%S')} ===")

    signals_found = 0

    for symbol in SYMBOLS:
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="5d", interval=TIMEFRAME)

            if df.empty or len(df) < 50:
                continue

            df = calculate_indicators(df)

            # Scan last 2 candles
            for i in [-2, -1]:
                candle_dt = df.index[i]
                if candle_dt.tzinfo is None:
                    candle_dt = candle_dt.tz_localize('UTC').tz_convert(ist_tz)
                else:
                    candle_dt = candle_dt.tz_convert(ist_tz)

                # Today filter
                if candle_dt.date() != today_date:
                    continue

                candle_time = candle_dt.strftime('%Y-%m-%d %H:%M')
                close_p = df['Close'].iloc[i]
                high_p = df['High'].iloc[i]
                low_p = df['Low'].iloc[i]

                sqz_start = df['squeezeStart'].iloc[i]
                buy_sig = df['buySignal'].iloc[i]
                sell_sig = df['sellSignal'].iloc[i]

                display_name = (symbol.replace("^NSEI", "NIFTY 50")
                                      .replace("^NSEBANK", "BANK NIFTY")
                                      .replace("^CNXFIN", "FIN NIFTY")
                                      .replace("^NSEMDCP50", "MIDCAP NIFTY")
                                      .replace(".NS", ""))

                # 1. BLACK SQUARE ALERT (SQUEEZE BUILDING)
                if sqz_start:
                    sig_id = f"SQZ_BUILD_{symbol}_{candle_time}"
                    if sig_id not in sent_signals:
                        signals_found += 1
                        msg = (f"⬛ *SQUEEZE BUILDING (Black Square)* | {display_name}\n"
                               f"Time: {candle_time}\n"
                               f"Current Price: ₹{close_p:.2f}\n"
                               f"Status: Squeeze start hua hai! Breakout ke liye ready rahein.")
                        send_telegram_alert(msg)
                        sent_signals.add(sig_id)

                # 2. YELLOW LABEL ALERT (BUY SIGNAL)
                if buy_sig:
                    sig_id = f"BUY_SIG_{symbol}_{candle_time}"
                    if sig_id not in sent_signals:
                        signals_found += 1
                        msg = (f"🟡 *BUY SIGNAL (Yellow Label)* | {display_name}\n"
                               f"Time: {candle_time}\n"
                               f"Entry Price: ₹{close_p:.2f}\n"
                               f"High Price Level: ₹{high_p:.2f}")
                        send_telegram_alert(msg)
                        sent_signals.add(sig_id)

                # 3. BLUE LABEL ALERT (SELL SIGNAL)
                if sell_sig:
                    sig_id = f"SELL_SIG_{symbol}_{candle_time}"
                    if sig_id not in sent_signals:
                        signals_found += 1
                        msg = (f"🔵 *SELL SIGNAL (Blue Label)* | {display_name}\n"
                               f"Time: {candle_time}\n"
                               f"Entry Price: ₹{close_p:.2f}\n"
                               f"Low Price Level: ₹{low_p:.2f}")
                        send_telegram_alert(msg)
                        sent_signals.add(sig_id)

        except Exception as e:
            print(f"Error scanning {symbol}: {e}")

    save_sent_signals(sent_signals)
    print(f"=== Scan Finish. Total Signals Sent: {signals_found} ===")

if __name__ == "__main__":
    run_scanner()
