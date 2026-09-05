from datetime import datetime, time, timedelta, timezone
import os
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
import yfinance as yf

# ==============================================================================
# 1. CONFIGURATION & GITHUB SECRETS
# ==============================================================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# TEST_MODE:
# - True  : Pichle 1 din ke sabhi CE/PE signals Telegram par bhejega (Verification ke liye).
# - False : Live mode — sirf fresh 15-min candle wale naye signals bhejega.
TEST_MODE = True

MARKET_START = time(9, 15)
MARKET_END = time(15, 30)
IST = timezone(timedelta(hours=5, minutes=30))

# ==============================================================================
# FULL NSE FnO WATCHLIST (Indices + All FnO Stocks)
# ==============================================================================
WATCHLIST = [
    # --- INDICES ---
    "^NSEI",        # Nifty 50
    "^NSEBANK",     # Bank Nifty
    "^FINNIFTY",    # Fin Nifty
    "^MIDCPNIFTY",  # Midcap Nifty
    
    # --- STOCKS (A-Z) ---
    "AARTIIND.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABFRL.NS", 
    "ACC.NS", "ADANIENT.NS", "ADANIPORTS.NS", "ALKEM.NS", "AMBUJACEM.NS", 
    "APOLLOHOSP.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTRAL.NS", 
    "ATUL.NS", "AUBANK.NS", "AUROPHARMA.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", 
    "BAJAJFINSV.NS", "BAJFINANCE.NS", "BALKRISIND.NS", "BALRAMCHIN.NS", "BANDHANBNK.NS", 
    "BANKBARODA.NS", "BATAINDIA.NS", "BEL.NS", "BERGEPAINT.NS", "BHARATFORG.NS", 
    "BHARTIARTL.NS", "BHEL.NS", "BIOCON.NS", "BPCL.NS", "BRITANNIA.NS", 
    "BSOFT.NS", "CANBK.NS", "CANFINHOME.NS", "CHAMBLFERT.NS", "CHOLAFIN.NS", 
    "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", "CONCOR.NS", 
    "COROMANDEL.NS", "CROMPTON.NS", "CUMMINSIND.NS", "DABUR.NS", "DALBHARAT.NS", 
    "DEEPAKNTR.NS", "DIVISLAB.NS", "DIXON.NS", "DLF.NS", "DRREDDY.NS", 
    "EICHERMOT.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "GAIL.NS", 
    "GLENMARK.NS", "GMMPFAUDLR.NS", "GNFC.NS", "GODREJCP.NS", "GODREJPROP.NS", 
    "GRANULES.NS", "GRASIM.NS", "GUJGASLTD.NS", "HAL.NS", "HAVELLS.NS", 
    "HCLTECH.NS", "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", 
    "HINDALCO.NS", "HINDCOPPER.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", 
    "ICICIGI.NS", "ICICIPRULI.NS", "IDEA.NS", "IDFCFIRSTB.NS", "IEX.NS", 
    "IGL.NS", "INDHOTEL.NS", "INDIACEM.NS", "INDIAMART.NS", "INDIGO.NS", 
    "INDUSINDBK.NS", "INDUSTOWER.NS", "INFY.NS", "IOC.NS", "IPCALAB.NS", 
    "IRCTC.NS", "ITC.NS", "JINDALSTEL.NS", "JKCEMENT.NS", "JSWSTEEL.NS", 
    "JUBLFOOD.NS", "KOTAKBANK.NS", "LALPATHLAB.NS", "LAURUSLABS.NS", "LICHSGFIN.NS", 
    "LTIM.NS", "LT.NS", "LTF.NS", "LUPIN.NS", "M&M.NS", 
    "M&MFIN.NS", "MANAPPURAM.NS", "MARICO.NS", "MARUTI.NS", "MCDOWELL-N.NS", 
    "MCX.NS", "METROPOLIS.NS", "MFSL.NS", "MGL.NS", "MOTHERSON.NS", 
    "MPHASIS.NS", "MRF.NS", "MUTHOOTFIN.NS", "NATIONALUM.NS", "NAVINFLUOR.NS", 
    "NESTLEIND.NS", "NMDC.NS", "NTPC.NS", "OBEROIRLTY.NS", "OFSS.NS", 
    "ONGC.NS", "PAGEIND.NS", "PEL.NS", "PERSISTENT.NS", "PETRONET.NS", 
    "PFC.NS", "PIDILITIND.NS", "PIIND.NS", "PNB.NS", "POLYCAB.NS", 
    "POWERGRID.NS", "PVRINOX.NS", "RAMCOCEM.NS", "RBLBANK.NS", "RECLTD.NS", 
    "RELIANCE.NS", "SAIL.NS", "SBICARD.NS", "SBILIFE.NS", "SBIN.NS", 
    "SHREECEM.NS", "SHRIRAMFIN.NS", "SIEMENS.NS", "SRF.NS", "SUNPHARMA.NS", 
    "SUNTV.NS", "SYNGENE.NS", "TATACHEM.NS", "TATACONSUM.NS", "TATAMOTORS.NS", 
    "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", 
    "TORNTPHARM.NS", "TRENT.NS", "TVSMOTOR.NS", "UBL.NS", "ULTRACEMCO.NS", 
    "UPL.NS", "VEDL.NS", "VOLTAS.NS", "WIPRO.NS", "ZEEL.NS", "ZYDUSLIFE.NS"
]


# ==============================================================================
# 2. TELEGRAM SENDER FUNCTION
# ==============================================================================
def send_telegram_alert(message: str):
    """Telegram Bot API se alert bhejta hai."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Error: TELEGRAM_BOT_TOKEN ya TELEGRAM_CHAT_ID missing hai!")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload)
        with urllib.request.urlopen(req) as response:
            if response.getcode() == 200:
                print("📲 Telegram Alert Sent Successfully!")
    except Exception as e:
        print(f"❌ Telegram Sending Error: {e}")


# ==============================================================================
# 3. PINE SCRIPT INDICATOR FUNCTIONS
# ==============================================================================
def pine_rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1.0 / length, adjust=False).mean()

def pine_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return pine_rma(tr, length)


# ==============================================================================
# 4. ROMYO BIG MOVE BREAKOUT STRATEGY ENGINE (15M)
# ==============================================================================
def run_romyo_breakout_engine(
    df: pd.DataFrame, 
    donchian_len=50, 
    use_volume=True, 
    vol_len=20, 
    vol_mult=1.1, 
    use_vol_expansion=True, 
    atr_len=14, 
    atr_avg_len=50, 
    trail_atr_mult=2.0
):
    
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']

    # 1. Donchian High/Low (Current bar excluding: [1])
    donchian_high = high.shift(1).rolling(window=donchian_len).max()
    donchian_low = low.shift(1).rolling(window=donchian_len).min()

    # 2. Volatility Expansion Filter
    atr_val = pine_atr(df, atr_len)
    atr_avg = atr_val.rolling(window=atr_avg_len).mean()
    vol_expand_ok = (not use_vol_expansion) or (atr_val > atr_avg)

    # 3. Volume Filter
    vol_avg = volume.rolling(window=vol_len).mean()
    vol_ok = (not use_volume) or (volume > (vol_avg * vol_mult))

    # 4. Raw Signals
    raw_buy = (close > donchian_high) & vol_ok & vol_expand_ok
    raw_sell = (close < donchian_low) & vol_ok & vol_expand_ok

    # 5. Trailing Stop & Position Lock Logic
    n = len(df)
    pos_state = ["NONE"] * n
    trail_sl = [np.nan] * n
    valid_buy = [False] * n
    valid_sell = [False] * n

    current_pos = "NONE"
    current_sl = 0.0

    for i in range(n):
        prev_pos = current_pos
        
        r_buy = raw_buy.iloc[i]
        r_sell = raw_sell.iloc[i]
        c_price = close.iloc[i]
        h_price = high.iloc[i]
        l_price = low.iloc[i]
        a_val = atr_val.iloc[i]

        if pd.isna(a_val) or pd.isna(donchian_high.iloc[i]):
            pos_state[i] = current_pos
            continue

        # Strategy Entry condition (Only enter when FLAT)
        if r_buy and current_pos == "NONE":
            current_pos = "LONG"
            current_sl = c_price - (a_val * trail_atr_mult)

        elif r_sell and current_pos == "NONE":
            current_pos = "SHORT"
            current_sl = c_price + (a_val * trail_atr_mult)

        # Trailing SL update & Position exit check
        if current_pos == "LONG":
            current_sl = max(current_sl, c_price - (a_val * trail_atr_mult))
            if l_price <= current_sl:
                current_pos = "NONE"

        elif current_pos == "SHORT":
            current_sl = min(current_sl, c_price + (a_val * trail_atr_mult))
            if h_price >= current_sl:
                current_pos = "NONE"

        v_buy = r_buy and (current_pos == "LONG") and (prev_pos == "NONE")
        v_sell = r_sell and (current_pos == "SHORT") and (prev_pos == "NONE")

        pos_state[i] = current_pos
        trail_sl[i] = current_sl if current_pos != "NONE" else np.nan
        valid_buy[i] = v_buy
        valid_sell[i] = v_sell

    df["Pos_State"] = pos_state
    df["Trail_SL"] = trail_sl
    df["Valid_Buy"] = valid_buy
    df["Valid_Sell"] = valid_sell

    return df

def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return MARKET_START <= now.time() <= MARKET_END


# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
def main():
    now = datetime.now(IST)
    print(f"\n==================================================================")
    print(f"🚀 ROMYO 15M BIG MOVE SCANNER RUNNING | {now.strftime('%Y-%m-%d %H:%M:%S IST')}")
    print(f"==================================================================")

    if not is_market_open() and not TEST_MODE:
        print("⏸ Market is CLOSED. Skipping scanner.")
        return

    signals_found = 0

    for ticker in WATCHLIST:
        try:
            df = yf.download(ticker, period="5d", interval="15m", progress=False)
            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.dropna()
            df = run_romyo_breakout_engine(df)

            # --- TEST MODE vs LIVE MODE FILTER ---
            if TEST_MODE:
                if df.index.tz is not None:
                    one_day_ago = pd.Timestamp.now(tz=df.index.tz) - pd.Timedelta(days=1)
                else:
                    one_day_ago = pd.Timestamp.now() - pd.Timedelta(days=1)

                scan_df = df[df.index >= one_day_ago]
            else:
                scan_df = df.iloc[-2:]

            target_signals = scan_df[scan_df["Valid_Buy"] | scan_df["Valid_Sell"]]

            for idx, row in target_signals.iterrows():
                signals_found += 1
                candle_time = idx.strftime('%Y-%m-%d %H:%M IST')

                if row["Valid_Buy"]:
                    msg = (
                        f"🚀 *ROMYO BIG MOVE: CE BUY (15M)*\n\n"
                        f"📈 *Asset:* `{ticker}`\n"
                        f"⏰ *Time:* `{candle_time}`\n"
                        f"💰 *Entry Close:* `{row['Close']:.2f}`\n"
                        f"🛡 *Trail SL:* `{row['Trail_SL']:.2f}`"
                    )
                    print(f"[CE BUY] Asset: {ticker} | Time: {candle_time} | Entry: {row['Close']:.2f}")
                    send_telegram_alert(msg)

                elif row["Valid_Sell"]:
                    msg = (
                        f"💥 *ROMYO BIG MOVE: PE BUY (15M)*\n\n"
                        f"📉 *Asset:* `{ticker}`\n"
                        f"⏰ *Time:* `{candle_time}`\n"
                        f"💰 *Entry Close:* `{row['Close']:.2f}`\n"
                        f"🛡 *Trail SL:* `{row['Trail_SL']:.2f}`"
                    )
                    print(f"[PE BUY] Asset: {ticker} | Time: {candle_time} | Entry: {row['Close']:.2f}")
                    send_telegram_alert(msg)

        except Exception as e:
            print(f"Error scanning {ticker}: {e}")

    print(f"\n✅ Scan Complete. Sent {signals_found} signals to Telegram channel.")

if __name__ == "__main__":
    main()
