
import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Complete Nifty F&O Stocks List
FNO_STOCKS = [
    "AARTIIND.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "ALKEM.NS", "AMBUJACEMENT.NS", "APOLLOHOSP.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTRAL.NS", "ATUL.NS",
    "AUBANK.NS", "AUROPHARMA.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BALKRISIND.NS",
    "BALRAMCHIN.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "BATAINDIA.NS", "BEL.NS", "BERGEPAINT.NS", "BHARATFORG.NS",
    "BHARTIARTL.NS", "BHEL.NS", "BIOCON.NS", "BSOFT.NS", "BPCL.NS", "BRITANNIA.NS", "CANBK.NS", "CANFINHOME.NS",
    "CHAMBLFERT.NS", "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", "CONCOR.NS", "COROMANDEL.NS",
    "CROMPTON.NS", "CUB.NS", "CUMMINSIND.NS", "DABUR.NS", "DALBHARAT.NS", "DEEPAKNTR.NS", "DIVISLAB.NS", "DIXON.NS",
    "DLF.NS", "DRREDDY.NS", "EICHERMOT.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "GAIL.NS", "GLENMARK.NS",
    "GMRINFRA.NS", "GNFC.NS", "GODREJCP.NS", "GODREJPROP.NS", "GRANULES.NS", "GRASIM.NS", "GUJGASLTD.NS", "HAL.NS",
    "HAVELLS.NS", "HCLTECH.NS", "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDCOPPER.NS",
    "HINDPETRO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDEA.NS", "IDFC.NS", "IDFCFIRSTB.NS",
    "IEX.NS", "IGL.NS", "INDHOTEL.NS", "INDIACEM.NS", "INDIAMART.NS", "INDIGO.NS", "INDUSINDBK.NS", "INDUSTOWER.NS",
    "INFY.NS", "IOC.NS", "IPCALAB.NS", "IRCTC.NS", "ITC.NS", "JINDALSTEL.NS", "JKCEMENT.NS", "JSWSTEEL.NS",
    "JUBLFOOD.NS", "KOTAKBANK.NS", "LALPATHLAB.NS", "LAURUSLABS.NS", "LICHSGFIN.NS", "LT.NS", "LTIM.NS", "LTTS.NS",
    "LUPIN.NS", "M&M.NS", "M&MFIN.NS", "MANAPPURAM.NS", "MARICO.NS", "MARUTI.NS", "MCDOWELL-N.NS", "MCX.NS",
    "METROPOLIS.NS", "MFSL.NS", "MGL.NS", "MOTHERSON.NS", "MPHASIS.NS", "MRF.NS", "MUTHOOTFIN.NS", "NATIONALUM.NS",
    "NAVINFLUOR.NS", "NESTLEIND.NS", "NMDC.NS", "NTPC.NS", "OBEROIRLTY.NS", "OFSS.NS", "ONGC.NS", "PAGEIND.NS",
    "PERSISTENT.NS", "PETRONET.NS", "PFC.NS", "PIDILITIND.NS", "PIIND.NS", "PNB.NS", "POLYCAB.NS", "POWERGRID.NS",
    "PVRINOX.NS", "RAMCOCEM.NS", "RBLBANK.NS", "RECLTD.NS", "RELIANCE.NS", "SAIL.NS", "SBICARD.NS", "SBILIFE.NS",
    "SBIN.NS", "SHREECEM.NS", "SHRIRAMFIN.NS", "SIEMENS.NS", "SRF.NS", "SUNPHARMA.NS", "SUNTV.NS", "SYNGENE.NS",
    "TATACHEMICAL.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS",
    "TITAN.NS", "TORNTPHARM.NS", "TRENT.NS", "TVSMOTOR.NS", "UBL.NS", "ULTRACEMCO.NS", "UPL.NS", "VEDL.NS",
    "VOLTAS.NS", "WIPRO.NS", "ZEEL.NS"
]

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def calculate_black_buy_exact(df):
    pd_val = 22
    bbl = 20
    mult = 2.0
    lb = 50
    ph = 0.99
    pl = 1.01
    max_red_gap = 15

    # Williams Vix Fix Calculations
    df['highestClose'] = df['Close'].rolling(window=pd_val).max()
    df['wvf'] = np.where(df['highestClose'] != 0, ((df['highestClose'] - df['Low']) / df['highestClose']) * 100, 0)
    df['sDev'] = mult * df['wvf'].rolling(window=bbl).std()
    df['midLine'] = df['wvf'].rolling(window=bbl).mean()
    df['lowerBand'] = df['midLine'] - df['sDev']
    df['rangeLow'] = df['wvf'].rolling(window=lb).min() * pl

    df['isRed'] = (df['wvf'] <= df['lowerBand']) | (df['wvf'] <= df['rangeLow'])

    cluster_active = False
    cluster_min_body = None
    cluster_min_high = None
    cluster_min_low = None
    last_red_bar = None

    active_boxes = []

    watched_level = None
    watched_start_idx = None
    watching_black_buy = False

    black_buy_triggered = False
    signal_price = None

    for i in range(len(df)):
        row = df.iloc[i]
        bar_index = i
        is_red = row['isRed']

        if is_red:
            body_size = row['wvf']
            gap_too_big = cluster_active and (last_red_bar is not None) and ((bar_index - last_red_bar) > max_red_gap)

            if (not cluster_active) or gap_too_big:
                cluster_active = True
                cluster_min_body = body_size
                cluster_min_high = row['High']
                cluster_min_low = row['Low']

                active_boxes.append({
                    'low': cluster_min_low,
                    'high': cluster_min_high,
                    'broken': False
                })

                if len(active_boxes) > 4:
                    active_boxes.pop(0)
            else:
                if body_size < cluster_min_body:
                    cluster_min_body = body_size
                    cluster_min_high = row['High']
                    cluster_min_low = row['Low']

                    if len(active_boxes) > 0:
                        active_boxes[-1]['low'] = cluster_min_low
                        active_boxes[-1]['high'] = cluster_min_high

            last_red_bar = bar_index

        buy1_signal = False
        entry_box_high = None

        if len(active_boxes) >= 2:
            sorted_boxes = sorted(active_boxes, key=lambda x: x['low'])
            box2 = sorted_boxes[1]

            if not box2['broken'] and row['Close'] > box2['high']:
                box2['broken'] = True
                buy1_signal = True
                entry_box_high = box2['high']

        if buy1_signal:
            watched_level = entry_box_high
            watched_start_idx = bar_index
            watching_black_buy = True

        if watching_black_buy:
            after_buy1 = bar_index > watched_start_idx
            green_candle = row['Close'] > row['Open']
            above_level = row['Close'] > watched_level if watched_level is not None else False

            if after_buy1 and green_candle and above_level:
                if i == len(df) - 1:
                    black_buy_triggered = True
                    signal_price = row['Close']

                watching_black_buy = False
                watched_level = None
                watched_start_idx = None

    return black_buy_triggered, signal_price

def run_scanner():
    for symbol in FNO_STOCKS:
        try:
            df = yf.download(symbol, period="20d", interval="15m", progress=False)
            if len(df) < 60:
                continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            signal, price = calculate_black_buy_exact(df)
            if signal:
                msg = f"🚀 *ROMYO BLACK BUY SIGNAL*\n\n*Stock:* {symbol.replace('.NS','')}\n*Price:* ₹{price:.2f}\n*Timeframe:* 15 Min"
                send_telegram(msg)
        except Exception:
            continue

if __name__ == "__main__":
    run_scanner()
