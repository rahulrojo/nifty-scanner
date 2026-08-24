import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# Telegram Credentials
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram Send Error: {e}")

# Major Indices + FnO Stocks List
INDICES = [
    "^NSEI",         # Nifty 50
    "^NSEBANK",      # Nifty Bank
    "^CNXIT",        # Nifty IT
    "^CNXAUTO",      # Nifty Auto
    "^CNXPHARMA",    # Nifty Pharma
    "^CNXFMCG",      # Nifty FMCG
    "^CNXMETAL",     # Nifty Metal
    "^CNXENERGY",    # Nifty Energy
    "^CNXREALTY",    # Nifty Realty
    "^CNX100",       # Nifty 100
    "^BSESN"         # Sensex
]

FNO_STOCKS = [
    "AARTIIND.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABFRL.NS", 
    "ACC.NS", "ADANIENT.NS", "ADANIPORTS.NS", "ALKEM.NS", "AMBUJACEMENT.NS", "APOLLOHOSP.NS", 
    "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTRAL.NS", "ATUL.NS", "AUBANK.NS", 
    "AUROPHARMA.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", 
    "BALKRISIND.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "BATAINDIA.NS", "BEL.NS", "BERGEPAINT.NS", 
    "BHARATFORG.NS", "BHARTIARTL.NS", "BHEL.NS", "BIOCON.NS", "BPCL.NS", "BRITANNIA.NS", 
    "BSOFT.NS", "CANBK.NS", "CANFINHOME.NS", "CHAMBLFERT.NS", "CHOLAFIN.NS", "CIPLA.NS", 
    "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", "CONCOR.NS", "COROMANDEL.NS", "CROMPTON.NS", 
    "CUMMINSIND.NS", "DABUR.NS", "DALBHARAT.NS", "DEEPAKNTR.NS", "DIVISLAB.NS", "DIXON.NS", 
    "DLF.NS", "DRREDDY.NS", "EICHERMOT.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", 
    "GAIL.NS", "GLENMARK.NS", "GMRAIRPORT.NS", "GNFC.NS", "GODREJCP.NS", "GODREJPROP.NS", 
    "GRANULES.NS", "GRASIM.NS", "HAL.NS", "HAVELLS.NS", "HCLTECH.NS", 
    "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDCOPPER.NS", 
    "HINDPETRO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDEA.NS", 
    "IDFCFIRSTB.NS", "IEX.NS", "IGL.NS", "INDHOTEL.NS", "INDIAMART.NS", 
    "INDIGO.NS", "INDUSINDBK.NS", "INDUSTOWER.NS", "INFY.NS", "IOC.NS", "IPCALAB.NS", 
    "IRCTC.NS", "ITC.NS", "JINDALSTEL.NS", "JKCEMENT.NS", "JSWSTEEL.NS", "JUBLFOOD.NS", 
    "KALYANKJIL.NS", "KEI.NS", "KOTAKBANK.NS", "LALPATHLAB.NS", "LAURUSLABS.NS", "LICHSGFIN.NS", 
    "LTIM.NS", "LT.NS", "LUPIN.NS", "M&M.NS", "M&MFIN.NS", "MANAPPURAM.NS", 
    "MARICO.NS", "MARUTI.NS", "MCX.NS", "METROPOLIS.NS", "MFSL.NS", "MGL.NS", "MOTHERSON.NS", 
    "MPHASIS.NS", "MRF.NS", "MUTHOOTFIN.NS", "NATIONALUM.NS", "NAVINFLUOR.NS", "NESTLEIND.NS", 
    "NMDC.NS", "NTPC.NS", "OBEROIRLTY.NS", "OFSS.NS", "OIL.NS", "ONGC.NS", "PAGEIND.NS", 
    "PERSISTENT.NS", "PETRONET.NS", "PFC.NS", "PIDILITIND.NS", "PIIND.NS", "PNB.NS", 
    "POLYCAB.NS", "POWERGRID.NS", "PVRINOX.NS", "RAMCOCEM.NS", "RBLBANK.NS", "RECLTD.NS", 
    "RELIANCE.NS", "SAIL.NS", "SBICARD.NS", "SBILIFE.NS", "SBIN.NS", "SHREECEM.NS", 
    "SHRIRAMFIN.NS", "SIEMENS.NS", "SRF.NS", "SUNPHARMA.NS", "SUNTV.NS", "SYNGENE.NS", 
    "TATACHEMICALS.NS", "TATACONSUM.NS", "TATELXSI.NS", "TATAMOTORS.NS", "TATAPOWER.NS", 
    "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "TORNTPHARM.NS", "TORNTPOWER.NS", 
    "TRENT.NS", "TVSMOTOR.NS", "UBL.NS", "ULTRACEMCO.NS", "UNIONBANK.NS", "UPL.NS", 
    "VEDL.NS", "VOLTAS.NS", "WIPRO.NS", "ZEEL.NS"
]

ALL_WATCHLIST = INDICES + FNO_STOCKS

def check_vix_mix(symbol):
    df = yf.download(symbol, period="60d", interval="1d", progress=False)
    if df.empty or len(df) < 50:
        return

    # MultiIndex dataframe handling
    if isinstance(df.columns, pd.MultiIndex):
        close = df['Close'][symbol]
        low = df['Low'][symbol]
        high = df['High'][symbol]
    else:
        close = df['Close']
        low = df['Low']
        high = df['High']

    pd_val, bbl, mult, lb, ph, pl = 22, 20, 2.0, 50, 0.99, 1.01

    highest_close = close.rolling(pd_val).max()
    wvf = ((highest_close - low) / highest_close) * 100

    mid_line = wvf.rolling(bbl).mean()
    s_dev = mult * wvf.rolling(bbl).std()
    upper_band = mid_line + s_dev
    lower_band = mid_line - s_dev

    range_high = wvf.rolling(lb).max() * ph
    range_low = wvf.rolling(lb).min() * pl

    is_green_vix = (wvf >= upper_band) | (wvf >= range_high)
    is_red_vix   = (wvf <= lower_band) | (wvf <= range_low)

    box_highs = []
    min_red_wvf = None
    min_red_high = None
    in_cluster = False

    for i in range(len(df)):
        is_green = is_green_vix.iloc[i]
        is_red = is_red_vix.iloc[i]
        c_high = high.iloc[i]
        c_wvf = wvf.iloc[i]

        if is_green:
            if in_cluster and min_red_high is not None:
                box_highs.append(min_red_high)
                if len(box_highs) > 4:
                    box_highs.pop(0)
                min_red_wvf = None
                min_red_high = None
            in_cluster = True

        if in_cluster and is_red:
            if min_red_wvf is None or c_wvf < min_red_wvf:
                min_red_wvf = c_wvf
                min_red_high = c_high

    if len(box_highs) >= 2:
        breakout_level = max(box_highs[-1], box_highs[-2])
        last_close = close.iloc[-1]
        prev_close = close.iloc[-2]

        # Breakout alert trigger
        if prev_close <= breakout_level and last_close > breakout_level:
            is_index = symbol.startswith("^")
            label_type = "📊 INDEX SIGNAL" if is_index else "📈 STOCK SIGNAL"
            send_telegram(f"🟢 *GREEN BUY SIGNAL DETECTED!*\n\n{label_type}: `{symbol}`\n💰 *Current Price:* {last_close:.2f}\n🎯 *Breakout Level:* {breakout_level:.2f}")

if __name__ == "__main__":
    send_telegram(f"🚀 *Vix_Mix Scanner Started!*\nScanning {len(INDICES)} Indices & {len(FNO_STOCKS)} FnO Stocks...")
    for item in ALL_WATCHLIST:
        try:
            check_vix_mix(item)
        except Exception as e:
            print(f"Error scanning {item}: {e}")
    send_telegram("✅ *Scan Complete!* All Indices & FnO stocks scanned.")
