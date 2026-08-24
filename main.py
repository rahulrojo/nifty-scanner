import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

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
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram Credentials Missing!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    res = requests.post(url, json=payload)
    print(f"Telegram Response: {res.status_code}")

def process_strategy(df, lookback_bars=25):
    if df.empty or len(df) < 60:
        return []

    pd_val, bbl, mult, lb, ph, pl = 22, 20, 2.0, 50, 0.99, 1.01

    df['highestClose'] = df['Close'].rolling(window=pd_val).max()
    df['wvf'] = np.where(df['highestClose'] != 0, ((df['highestClose'] - df['Low']) / df['highestClose']) * 100, 0)
    df['sDev'] = mult * df['wvf'].rolling(window=bbl).std()
    df['midLine'] = df['wvf'].rolling(window=bbl).mean()
    df['lowerBand'] = df['midLine'] - df['sDev']
    df['upperBand'] = df['midLine'] + df['sDev']
    df['rangeHigh'] = df['wvf'].rolling(window=lb).max() * ph
    df['rangeLow'] = df['wvf'].rolling(window=lb).min() * pl

    df['isGreenVix'] = (df['wvf'] >= df['upperBand']) | (df['wvf'] >= df['rangeHigh'])
    df['isRedVix'] = (df['wvf'] <= df['lowerBand']) | (df['wvf'] <= df['rangeLow'])

    inCluster = False
    minRedWvf = None
    minRedHigh = None

    boxHighs = []
    greenBuyTriggered = False
    blackBuyTriggered = False

    signals = []

    for i in range(len(df)):
        row = df.iloc[i]
        isGreen = row['isGreenVix']
        isRed = row['isRedVix']

        if isGreen:
            if inCluster and minRedHigh is not None:
                boxHighs.append(minRedHigh)
                if len(boxHighs) > 4:
                    boxHighs.pop(0)
                minRedWvf = None
                minRedHigh = None
            inCluster = True

        if inCluster and isRed:
            if minRedWvf is None or row['wvf'] < minRedWvf:
                minRedWvf = row['wvf']
                minRedHigh = row['High']

        canBuy = len(boxHighs) >= 2
        refHigh1 = boxHighs[-1] if canBuy else None
        refHigh2 = boxHighs[-2] if canBuy else None
        breakoutLevel = max(refHigh1, refHigh2) if canBuy else None

        prev_close = df.iloc[i-1]['Close'] if i > 0 else row['Close']
        greenBuy = canBuy and (not greenBuyTriggered) and (prev_close <= breakoutLevel) and (row['Close'] > breakoutLevel)

        if greenBuy:
            greenBuyTriggered = True
            blackBuyTriggered = False
            if i >= len(df) - lookback_bars:
                signals.append(("BUY (GREEN)", row['Close'], df.index[i]))

        secondRedHigh = refHigh2
        isGreenCandle = row['Close'] > row['Open']
        blackBuy = greenBuyTriggered and (not blackBuyTriggered) and isGreenCandle and (secondRedHigh is not None) and (row['Low'] > secondRedHigh)

        if blackBuy:
            blackBuyTriggered = True
            greenBuyTriggered = False
            if i >= len(df) - lookback_bars:
                signals.append(("BUY (BLACK)", row['Close'], df.index[i]))

    return signals

def run_scanner():
    send_telegram("🔍 *Vix_Mix Scanner Running...*\nScanning 180+ F&O Stocks for Signals...")

    try:
        data = yf.download(FNO_STOCKS, period="10d", interval="15m", group_by='ticker', threads=True, progress=False)
    except Exception as e:
        print(f"Data Fetch Error: {e}")
        return

    signals_found = 0
    for symbol in FNO_STOCKS:
        try:
            df = data[symbol].dropna().copy() if symbol in data else None
            if df is None or df.empty:
                continue

            signals = process_strategy(df, lookback_bars=25)
            for signal_type, price, signal_time in signals:
                signals_found += 1
                time_str = signal_time.strftime("%d-%b %H:%M")
                msg = f"🚀 *VIX MIX SIGNAL ALERT*\n\n*Stock:* {symbol.replace('.NS','')}\n*Signal:* {signal_type}\n*Price:* ₹{price:.2f}\n*Time:* {time_str}\n*Timeframe:* 15 Min"
                send_telegram(msg)
        except Exception:
            continue

    if signals_found == 0:
        send_telegram("✅ *Scan Complete*: No Buy signals found in recent bars.")

if __name__ == "__main__":
    run_scanner()
