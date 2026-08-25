import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta

# Telegram Settings
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = "-1003921675472"

# Complete NSE FnO Stocks List (~180 Stocks)
FNO_STOCKS = [
    "AARTIIND.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", "ADANIENT.NS", 
    "ADANIPORTS.NS", "ALKEM.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", 
    "ASTRAL.NS", "ATUL.NS", "AUBANK.NS", "AUROPHARMA.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", 
    "BAJAJFINSV.NS", "BAJFINANCE.NS", "BALKRISIND.NS", "BALRAMCHIN.NS", "BANDHANBNK.NS", 
    "BANKBARODA.NS", "BANKINDIA.NS", "BATAINDIA.NS", "BEL.NS", "BHARATFORG.NS", "BHARTIARTL.NS", 
    "BHEL.NS", "BIOCON.NS", "BPCL.NS", "BRITANNIA.NS", "BSOFT.NS", "CANBK.NS", "CANFINHOME.NS", 
    "CHAMBLFERT.NS", "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", 
    "CONCOR.NS", "COROMANDEL.NS", "CROMPTON.NS", "CUMMINSIND.NS", "DABUR.NS", "DALBHARAT.NS", 
    "DEEPAKNTR.NS", "DIVISLAB.NS", "DIXON.NS", "DLF.NS", "DRREDDY.NS", "EICHERMOT.NS", 
    "ESCORTS.NS", "EXIDEIND.NS", "GAIL.NS", "GLENMARK.NS", "GODREJCP.NS", "GODREJPROP.NS", 
    "GRANULES.NS", "GRASIM.NS", "GUJGASLTD.NS", "HAL.NS", "HAVELLS.NS", "HCLTECH.NS", 
    "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDCOPPER.NS", 
    "HINDPETRO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDEA.NS", 
    "IDFCFIRSTB.NS", "IEX.NS", "IGL.NS", "INDHOTEL.NS", "INDIACEM.NS", "INDIAMART.NS", 
    "INDIGO.NS", "INDUSINDBK.NS", "INDUSTOWER.NS", "INFY.NS", "IOC.NS", "IPCALAB.NS", 
    "IRCTC.NS", "ITC.NS", "JINDALSTEL.NS", "JIOFIN.NS", "JKCEMENT.NS", "JSWSTEEL.NS", 
    "JUBLFOOD.NS", "KOTAKBANK.NS", "LALPATHLAB.NS", "LAURUSLABS.NS", "LICHSGFIN.NS", 
    "LT.NS", "LTIM.NS", "LTTS.NS", "LUPIN.NS", "M&M.NS", "M&MFIN.NS", "MANAPPURAM.NS", 
    "MARICO.NS", "MARUTI.NS", "MCDOWELL-N.NS", "MCX.NS", "METROPOLIS.NS", "MFSL.NS", 
    "MGL.NS", "MOTHERSON.NS", "MPHASIS.NS", "MRF.NS", "MUTHOOTFIN.NS", "NATIONALUM.NS", 
    "NAUKRI.NS", "NAVINFLUOR.NS", "NESTLEIND.NS", "NMDC.NS", "NTPC.NS", "OBEROIRLTY.NS", 
    "OFSS.NS", "ONGC.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFC.NS", 
    "PIDILITIND.NS", "PIIND.NS", "PNB.NS", "POLYCAB.NS", "POWERGRID.NS", "PVRINOX.NS", 
    "RAMCOCEM.NS", "RBLBANK.NS", "REC.NS", "RELIANCE.NS", "SAIL.NS", "SBICARD.NS", 
    "SBILIFE.NS", "SBIN.NS", "SHREECEM.NS", "SHRIRAMFIN.NS", "SIEMENS.NS", "SRF.NS", 
    "SUNPHARMA.NS", "SUNTV.NS", "SYNGENE.NS", "TATACHEMICALS.NS", "TATACOMM.NS", 
    "TATACONSUM.NS", "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", 
    "TITAN.NS", "TORNTPHARM.NS", "TRENT.NS", "TVSMOTOR.NS", "UBL.NS", "ULTRACEMCO.NS", 
    "UPL.NS", "VEDL.NS", "VOLTAS.NS", "WIPRO.NS", "ZEEL.NS", "ZYDUSLIFE.NS"
]

def is_market_open():
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist_tz)
    
    # Check Weekend (Saturday = 5, Sunday = 6)
    if now_ist.weekday() >= 5:
        return False
        
    # Indian Market Timing (9:15 AM to 3:30 PM IST)
    market_start = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_start <= now_ist <= market_end

def send_telegram(message):
    if not BOT_TOKEN:
        print("❌ ERROR: BOT_TOKEN secret is missing!")
        return False
        
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"Telegram API Error [{response.status_code}]: {response.text}")
        else:
            print("Signal sent successfully to Telegram!")
        return response.status_code == 200
    except Exception as e:
        print(f"Exception while sending Telegram message: {e}")
        return False

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def scan_vix_mix_balanced(ticker):
    try:
        data = yf.download(ticker, period="5d", interval="15m", progress=False)
        if data.empty or len(data) < 60:
            return

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        df = data.copy()

        # 1. Price Bollinger Bands
        df['P_Mid'] = df['Close'].rolling(20).mean()
        df['P_Std'] = df['Close'].rolling(20).std()
        df['P_Upper'] = df['P_Mid'] + (2.0 * df['P_Std'])
        df['P_Lower'] = df['P_Mid'] - (2.0 * df['P_Std'])

        # 2. RSI Calculation
        df['RSI'] = calculate_rsi(df['Close'], 14)

        # 3. VIX Mix Calculations
        highest_close_22 = df['Close'].rolling(22).max()
        df['WVF_Buy'] = ((highest_close_22 - df['Low']) / highest_close_22) * 100
        wvf_buy_mid = df['WVF_Buy'].rolling(20).mean()
        wvf_buy_std = df['WVF_Buy'].rolling(20).std()
        df['WVF_Buy_Upper'] = wvf_buy_mid + (2.0 * wvf_buy_std)
        df['WVF_Buy_High'] = df['WVF_Buy'].rolling(50).max() * 0.88

        lowest_close_22 = df['Close'].rolling(22).min()
        df['WVF_Sell'] = ((df['High'] - lowest_close_22) / lowest_close_22) * 100
        wvf_sell_mid = df['WVF_Sell'].rolling(20).mean()
        wvf_sell_std = df['WVF_Sell'].rolling(20).std()
        df['WVF_Sell_Upper'] = wvf_sell_mid + (2.0 * wvf_sell_std)
        df['WVF_Sell_High'] = df['WVF_Sell'].rolling(50).max() * 0.88

        df['Is_Panic'] = (df['WVF_Buy'].shift(1) >= df['WVF_Buy_Upper'].shift(1)) | (df['WVF_Buy'].shift(1) >= df['WVF_Buy_High'].shift(1))
        df['Is_Euphoria'] = (df['WVF_Sell'].shift(1) >= df['WVF_Sell_Upper'].shift(1)) | (df['WVF_Sell'].shift(1) >= df['WVF_Sell_High'].shift(1))

        curr = df.iloc[-2]
        prev = df.iloc[-3]
        recent_window = df.iloc[-5:-2]

        panic_recent = recent_window['Is_Panic'].any() or curr['Is_Panic']
        panic_turning = panic_recent and (curr['WVF_Buy'] < prev['WVF_Buy'])

        euphoria_recent = recent_window['Is_Euphoria'].any() or curr['Is_Euphoria']
        euphoria_turning = euphoria_recent and (curr['WVF_Sell'] < prev['WVF_Sell'])

        buy_reentry = (curr['Low'] <= curr['P_Lower'] or prev['Low'] <= prev['P_Lower']) and (curr['Close'] > curr['Open']) and (curr['Close'] > curr['P_Lower'])
        sell_reentry = (curr['High'] >= curr['P_Upper'] or prev['High'] >= prev['P_Upper']) and (curr['Close'] < curr['Open']) and (curr['Close'] < curr['P_Upper'])

        # Strict RSI Filters
        rsi_buy_ok = curr['RSI'] <= 42
        rsi_sell_ok = curr['RSI'] >= 58

        # Candle Timestamp Validation
        candle_dt = pd.to_datetime(df.index[-2])
        ist_tz = timezone(timedelta(hours=5, minutes=30))
        if candle_dt.tzinfo is None:
            candle_dt = candle_dt.tz_localize('UTC').tz_convert(ist_tz)
        else:
            candle_dt = candle_dt.tz_convert(ist_tz)

        now_ist = datetime.now(ist_tz)
        time_diff = (now_ist - candle_dt).total_seconds() / 60

        # Reject if candle is older than 25 minutes
        if time_diff > 25:
            return

        stock_name = ticker.replace(".NS", "")
        close_price = round(float(curr['Close']), 2)
        low_price = round(float(curr['Low']), 2)
        high_price = round(float(curr['High']), 2)
        candle_time_str = candle_dt.strftime("%I:%M %p")

        if panic_turning and buy_reentry and rsi_buy_ok:
            msg = f"🟢 *VIX MIX BALANCED BUY SIGNAL (15m)* 🟢\n\n*Stock:* {stock_name}\n*Price:* ₹{close_price}\n*StopLoss Zone:* ₹{low_price}\n*RSI:* {round(float(curr['RSI']), 1)}\n*Chart Candle:* 📊 {candle_time_str}"
            send_telegram(msg)

        elif euphoria_turning and sell_reentry and rsi_sell_ok:
            msg = f"🔴 *VIX MIX BALANCED SELL SIGNAL (15m)* 🔴\n\n*Stock:* {stock_name}\n*Price:* ₹{close_price}\n*StopLoss Zone:* ₹{high_price}\n*RSI:* {round(float(curr['RSI']), 1)}\n*Chart Candle:* 📊 {candle_time_str}"
            send_telegram(msg)

    except Exception as e:
        print(f"Error scanning {ticker}: {e}")

if __name__ == "__main__":
    if not is_market_open():
        print("⏸️ Market is currently CLOSED (Outside 9:15 AM - 3:30 PM IST). Skipping scan.")
    else:
        print("🚀 VIX Mix Strict Scanner Started...")
        for stock in FNO_STOCKS:
            scan_vix_mix_balanced(stock)
        print("✅ Market scan finished successfully.")
