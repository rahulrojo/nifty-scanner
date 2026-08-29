import os
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz

# ==========================================
# 1. INDEXES & NSE F&O STOCKS LIST
# ==========================================
INDEX_MAP = {
    "^NSEI": "NIFTY",
    "^NSEBANK": "BANKNIFTY",
    "^CNXFIN": "FINNIFTY",
    "^NSEMDCP50": "MIDCPNIFTY"
}

STOCKS = [
    # Major Indexes
    "^NSEI", "^NSEBANK", "^CNXFIN", "^NSEMDCP50",
    
    # F&O Stocks
    "AARTIIND.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", 
    "ADANIENT.NS", "ADANIPORTS.NS", "ALKEM.NS", "AMBUJACEMENT.NS", "APOLLOHOSP.NS", 
    "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTRAL.NS", "ATUL.NS", "AUBANK.NS", 
    "AUROPHARMA.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", 
    "BALKRISIND.NS", "BALRAMCHIN.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "BATAINDIA.NS", 
    "BEL.NS", "BERGEPAINT.NS", "BHARATFORG.NS", "BHARTIARTL.NS", "BHEL.NS", "BIOCON.NS", 
    "BOSCHLTD.NS", "BPCL.NS", "BRITANNIA.NS", "BSOFT.NS", "CANBK.NS", "CANFINHOME.NS", 
    "CHAMBLFERT.NS", "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", 
    "CONCOR.NS", "COROMANDEL.NS", "CROMPTON.NS", "CUB.NS", "CUMMINSIND.NS", "DABUR.NS", 
    "DALBHARAT.NS", "DEEPAKNTR.NS", "DIVISLAB.NS", "DIXON.NS", "DLF.NS", "DRREDDY.NS", 
    "EICHERMOT.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "GAIL.NS", "GLENMARK.NS", 
    "GMMPFAUDLR.NS", "GNFC.NS", "GODREJCP.NS", "GODREJPROP.NS", "GRANULES.NS", "GRASIM.NS", 
    "GUJGASLTD.NS", "HAL.NS", "HAVELLS.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", 
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDCOPPER.NS", "HINDPETRO.NS", "HINDUNILVR.NS", 
    "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDEA.NS", "IDFCFIRSTB.NS", "IEX.NS", 
    "IGL.NS", "INDHOTEL.NS", "INDIAMART.NS", "INDIGO.NS", "INDUSINDBK.NS", "INDUSTOWER.NS", 
    "INFY.NS", "IOC.NS", "IPCALAB.NS", "IRCTC.NS", "ITC.NS", "JINDALSTEL.NS", "JKCEMENT.NS", 
    "JSWSTEEL.NS", "JUBLFOOD.NS", "KOTAKBANK.NS", "LALPATHLAB.NS", "LAURUSLABS.NS", 
    "LICHSGFIN.NS", "LTIM.NS", "LT.NS", "LTF.NS", "LUPIN.NS", "M&M.NS", "M&MFIN.NS", 
    "MANAPPURAM.NS", "MARICO.NS", "MARUTI.NS", "MCDOWELL-N.NS", "MCX.NS", "METROPOLIS.NS", 
    "MFSL.NS", "MGL.NS", "MOTHERSON.NS", "MPHASIS.NS", "MRF.NS", "MUTHOOTFIN.NS", 
    "NATIONALUM.NS", "NAVINFLUOR.NS", "NESTLEIND.NS", "NMDC.NS", "NTPC.NS", "OBEROIRTY.NS", 
    "OFSS.NS", "OIL.NS", "ONGC.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", 
    "PFC.NS", "PIDILITIND.NS", "PIIND.NS", "PNB.NS", "POLYCAB.NS", "POWERGRID.NS", 
    "PVRINOX.NS", "RAMCOCEM.NS", "RBLBANK.NS", "RECLTD.NS", "RELIANCE.NS", "SAIL.NS", 
    "SBICARD.NS", "SBILIFE.NS", "SBIN.NS", "SHREECEM.NS", "SHRIRAMFIN.NS", "SIEMENS.NS", 
    "SRF.NS", "SUNPHARMA.NS", "SUNTV.NS", "SYNGENE.NS", "TATACONSUM.NS", 
    "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", 
    "TORNTPHARM.NS", "TRENT.NS", "TVSMOTOR.NS", "UBL.NS", "ULTRACETECH.NS", "UNIONBANK.NS", 
    "UPL.NS", "VEDL.NS", "VOLTAS.NS", "WIPRO.NS", "ZEEL.NS", "ZYDUSLIFE.NS"
]

TIMEFRAME = "15m"
SENT_SIGNALS_FILE = "sent_signals.json"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Load past execution state
sent_ids = []
last_run_time = None

if os.path.exists(SENT_SIGNALS_FILE):
    try:
        with open(SENT_SIGNALS_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                sent_ids = data.get("sent_ids", [])
                last_run_str = data.get("last_run_time", None)
                if last_run_str:
                    last_run_time = datetime.fromisoformat(last_run_str)
            elif isinstance(data, list):
                sent_ids = data
    except Exception as e:
        print(f"Error loading state file: {e}")

# ==========================================
# 2. EXACT TRADINGVIEW MATCHED STRATEGY
# ==========================================
def calculate_strategy(df):
    if len(df) < 60:
        return []

    left, right = 5, 5
    pd_val, bbl, mult, lb, ph = 22, 20, 2.0, 50, 0.90
    price_bbl, price_mult, rsi_len = 20, 2.0, 14
    slPct, tpPct = 1.5, 3.0

    df['p_mid'] = df['Close'].rolling(price_bbl).mean()
    df['p_std'] = df['Close'].rolling(price_bbl).std(ddof=1)
    df['p_upper'] = df['p_mid'] + price_mult * df['p_std']
    df['p_lower'] = df['p_mid'] - price_mult * df['p_std']

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/rsi_len, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/rsi_len, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    highest_close = df['Close'].rolling(pd_val).max()
    df['wvf_buy'] = ((highest_close - df['Low']) / highest_close) * 100.0
    df['wvf_buy_upper'] = df['wvf_buy'].rolling(bbl).mean() + mult * df['wvf_buy'].rolling(bbl).std(ddof=1)
    df['wvf_buy_high'] = df['wvf_buy'].rolling(lb).max() * ph
    df['is_buy_panic'] = (df['wvf_buy'] >= df['wvf_buy_upper']) & (df['wvf_buy'] >= df['wvf_buy_high'])

    lowest_close = df['Close'].rolling(pd_val).min()
    df['wvf_sell'] = ((df['High'] - lowest_close) / lowest_close) * 100.0
    df['wvf_sell_upper'] = df['wvf_sell'].rolling(bbl).mean() + mult * df['wvf_sell'].rolling(bbl).std(ddof=1)
    df['wvf_sell_high'] = df['wvf_sell'].rolling(lb).max() * ph
    df['is_sell_euphoria'] = (df['wvf_sell'] >= df['wvf_sell_upper']) & (df['wvf_sell'] >= df['wvf_sell_high'])

    df['vix_buy_active'] = df['is_buy_panic'].astype(int).rolling(6).max() > 0
    df['vix_sell_active'] = df['is_sell_euphoria'].astype(int).rolling(6).max() > 0

    lows = df['Low'].values
    highs = df['High'].values
    closes = df['Close'].values
    opens = df['Open'].values
    p_lowers = df['p_lower'].values
    p_uppers = df['p_upper'].values
    rsis = df['RSI'].values
    vix_b_actives = df['vix_buy_active'].values
    vix_s_actives = df['vix_sell_active'].values

    n = len(df)
    p_low = np.full(n, np.nan)
    p_high = np.full(n, np.nan)

    for i in range(left, n - right):
        if lows[i] == np.min(lows[i - left : i + right + 1]):
            p_low[i + right] = lows[i]
        if highs[i] == np.max(highs[i - left : i + right + 1]):
            p_high[i + right] = highs[i]

    lastLow1, lastLow2 = None, None
    lastHigh1, lastHigh2 = None, None
    resistance, support = None, None

    longTriggered = False
    shortTriggered = False
    position = 0
    sl_price, tp_price = 0.0, 0.0

    signals = []

    for i in range(n):
        if position == 1:
            if lows[i] <= sl_price or highs[i] >= tp_price:
                position = 0
                longTriggered = False
                shortTriggered = False
        elif position == -1:
            if highs[i] >= sl_price or lows[i] <= tp_price:
                position = 0
                longTriggered = False
                shortTriggered = False

        pl = p_low[i]
        ph_val = p_high[i]

        if not np.isnan(pl):
            lastLow2 = lastLow1
            lastLow1 = pl
            support = pl

        if not np.isnan(ph_val):
            lastHigh2 = lastHigh1
            lastHigh1 = ph_val
            resistance = ph_val

        validLows = (lastLow1 is not None) and (lastLow2 is not None)
        validHighs = (lastHigh1 is not None) and (lastHigh2 is not None)

        isHL = validLows and (lastLow1 > lastLow2)
        isLH = validHighs and (lastHigh1 < lastHigh2)

        c = closes[i]
        o = opens[i]
        p_low_b = p_lowers[i]
        p_high_b = p_uppers[i]
        rsi = rsis[i]

        vix_b_active = vix_b_actives[i]
        vix_s_active = vix_s_actives[i]

        prev_c = closes[i-1] if i > 0 else np.nan
        prev_p_lower = p_lowers[i-1] if i > 0 else np.nan
        prev_p_upper = p_uppers[i-1] if i > 0 else np.nan

        crossover_lower = (prev_c < prev_p_lower) and (c > p_low_b)
        crossunder_upper = (prev_c > prev_p_upper) and (c < p_high_b)

        res_cond = (resistance is not None) and (c > resistance)
        sup_cond = (support is not None) and (c < support)

        longCondition = vix_b_active and isHL and (res_cond or crossover_lower) and (c > o) and (rsi < 50)
        shortCondition = vix_s_active and isLH and (sup_cond or crossunder_upper) and (c < o) and (rsi > 50)

        longSignal = longCondition and not longTriggered
        shortSignal = shortCondition and not shortTriggered

        if longSignal:
            position = 1
            entry_p = round(c, 2)
            sl_price = round(entry_p * (1 - slPct / 100), 2)
            tp_price = round(entry_p * (1 + tpPct / 100), 2)
            longTriggered = True
            shortTriggered = False

            signals.append({
                'type': 'BUY',
                'timestamp': df.index[i],
                'price': entry_p,
                'sl': sl_price,
                'rsi': round(rsi, 1)
            })

        elif shortSignal:
            position = -1
            entry_p = round(c, 2)
            sl_price = round(entry_p * (1 + slPct / 100), 2)
            tp_price = round(entry_p * (1 - tpPct / 100), 2)
            shortTriggered = True
            longTriggered = False

            signals.append({
                'type': 'SELL',
                'timestamp': df.index[i],
                'price': entry_p,
                'sl': sl_price,
                'rsi': round(rsi, 1)
            })

    return signals

# ==========================================
# 3. TELEGRAM SENDER ENGINE
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
    global sent_ids, last_run_time
    new_signals_count = 0

    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)

    # If first execution ever, default filter start to last 24 hours
    filter_start_time = last_run_time if last_run_time is not None else (now - timedelta(hours=24))

    for symbol in STOCKS:
        clean_symbol = INDEX_MAP.get(symbol, symbol.replace(".NS", "").replace("-", ""))
        print(f"Scanning {clean_symbol}...")

        try:
            df = yf.download(symbol, period="5d", interval=TIMEFRAME, progress=False)
            if df.empty or len(df) < 60:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index = df.index.tz_convert(ist)

            # Exclude live running candle (only process completed 15m candles)
            df_closed = df[df.index + pd.Timedelta(minutes=15) <= now]

            if df_closed.empty:
                continue

            signals = calculate_strategy(df_closed)

            for sig in signals:
                sig_start_time = sig['timestamp']
                sig_close_time = sig_start_time + pd.Timedelta(minutes=15)

                if sig_close_time > filter_start_time:
                    sig_id = f"{clean_symbol}_{sig['type']}_{sig_start_time.strftime('%Y%m%d_%H%M')}"

                    if sig_id not in sent_ids:
                        time_str = sig_start_time.strftime("%I:%M %p")
                        label_type = "INDEX" if symbol in INDEX_MAP else "STOCK"

                        if sig['type'] == 'BUY':
                            msg = (
                                f"🟢 <b>VIX MIX BUY SIGNAL ({TIMEFRAME})</b> 🟢\n\n"
                                f"<b>{label_type}:</b> {clean_symbol}\n"
                                f"<b>Price:</b> {sig['price']}\n"
                                f"<b>StopLoss Zone:</b> {sig['sl']}\n"
                                f"<b>RSI:</b> {sig['rsi']}\n"
                                f"<b>Chart Candle:</b> 📊 {time_str}"
                            )
                        else:
                            msg = (
                                f"🔴 <b>VIX MIX SELL SIGNAL ({TIMEFRAME})</b> 🔴\n\n"
                                f"<b>{label_type}:</b> {clean_symbol}\n"
                                f"<b>Price:</b> {sig['price']}\n"
                                f"<b>StopLoss Zone:</b> {sig['sl']}\n"
                                f"<b>RSI:</b> {sig['rsi']}\n"
                                f"<b>Chart Candle:</b> 📊 {time_str}"
                            )

                        print(f"Sending signal for {clean_symbol} ({time_str})...")
                        if send_telegram_message(msg):
                            sent_ids.append(sig_id)
                            new_signals_count += 1

        except Exception as e:
            print(f"Error scanning {symbol}: {e}")

    # Send Execution Status/Confirmation Message to Telegram
    time_now_str = now.strftime("%I:%M %p")
    status_msg = (
        f"🤖 <b>Scanner Execution Status</b>\n\n"
        f"✅ Scan completed successfully at <b>{time_now_str}</b>\n"
        f"📊 <b>New Signals Sent:</b> {new_signals_count}"
    )
    send_telegram_message(status_msg)

    # Save state file
    save_data = {
        "last_run_time": now.isoformat(),
        "sent_ids": sent_ids
    }
    with open(SENT_SIGNALS_FILE, "w") as f:
        json.dump(save_data, f, indent=4)

    print(f"Scan complete. New signals sent: {new_signals_count}")

if __name__ == "__main__":
    main()
