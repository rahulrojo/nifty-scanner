import os
from datetime import datetime
import pandas as pd
import pandas_ta as ta
import requests
import yfinance as yf

# ===== CONFIGURATION =====
TIMEFRAME = "30m"
PERIOD = "5d"

# Major F&O Indexes + Top Liquid F&O Stocks List
SYMBOLS = [
    # Top Major Indexes
    "^NSEI",
    "^NSEBANK",
    "NIFTY_FIN_SERVICE.NS",
    "NIFTY_MID_SELECT.NS",
    # Top F&O Stocks
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "AXISBANK.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "ITC.NS",
    "HINDUNILVR.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "MARUTI.NS",
    "M&M.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "HCLTECH.NS",
    "SUNPHARMA.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "COALINDIA.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "BEL.NS",
    "HAL.NS",
    "BHEL.NS",
    "JIOFIN.NS",
    "REC.NS",
    "PFC.NS",
    "ZOMATO.NS",
    "DLF.NS",
    "VEDL.NS",
    "IOC.NS",
    "BPCL.NS",
    "HPCL.NS",
    "GAIL.NS",
    "TATA-POWER.NS",
    "TATACONSUM.NS",
    "INDUSINDBK.NS",
    "BANKBARODA.NS",
    "CANBK.NS",
    "PNB.NS",
    "SHRIRAMFIN.NS",
    "CHOLAFIN.NS",
    "DABUR.NS",
    "GODREJCP.NS",
    "BRITANNIA.NS",
    "NESTLEIND.NS",
    "CIPLA.NS",
    "DRREDDY.NS",
    "DIVISLAB.NS",
    "LTIM.NS",
    "TECHM.NS",
    "WIPRO.NS",
    "COFORGE.NS",
    "PERSISTENT.NS",
    "MPHASIS.NS",
    "EICHERMOT.NS",
    "HEROMOTOCO.NS",
    "BAJAJ-AUTO.NS",
    "TVSMOTOR.NS",
    "ASHOKLEY.NS",
    "BHARATFORG.NS",
    "COMMINSIND.NS",
    "SIEMENS.NS",
    "ABB.NS",
    "HAVELLS.NS",
    "VOLTAS.NS",
    "POLYCAB.NS",
    "DIXON.NS",
    "PIDILITIND.NS",
    "SRF.NS",
    "GRASIM.NS",
    "AMBUJACEM.NS",
    "DALBHARAT.NS",
    "APOLLOHOSP.NS",
    "LALPATHLAB.NS",
    "MUTHOOTFIN.NS",
    "MANAPPURAM.NS",
    "NATIONALUM.NS",
    "HINDALCO.NS",
    "NMDC.NS",
    "SAIL.NS",
    "JINDALSTEL.NS",
    "JSWSTEEL.NS",
    "IRCTC.NS",
    "IRFC.NS",
    "RVNL.NS",
    "AUBANK.NS",
    "FEDERALBNK.NS",
    "BANDHANBNK.NS",
    "IDFCFIRSTB.NS",
]

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def send_telegram_message(message):
  if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("Telegram credentials missing!")
    return
  url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
  payload = {
      "chat_id": TELEGRAM_CHAT_ID,
      "text": message,
      "parse_mode": "Markdown",
      "disable_web_page_preview": True,
  }
  try:
    requests.post(url, json=payload, timeout=10)
  except Exception as e:
    print(f"Telegram error: {e}")


def get_tv_link(ticker):
  symbol_map = {
      "^NSEI": "NIFTY",
      "^NSEBANK": "BANKNIFTY",
      "NIFTY_FIN_SERVICE.NS": "FINNIFTY",
      "NIFTY_MID_SELECT.NS": "MIDCPNIFTY",
  }
  tv_symbol = symbol_map.get(ticker, ticker.replace(".NS", ""))
  return f"https://in.tradingview.com/chart/?symbol=NSE:{tv_symbol}&interval=30"


def scan_symbol(ticker):
  try:
    df = yf.download(
        ticker, period=PERIOD, interval=TIMEFRAME, progress=False
    )
    if df is None or df.empty or len(df) < 30:
      return

    # Safely handle MultiIndex columns if returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = [col[0] for col in df.columns]

    # Ensure required columns exist
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
      if col not in df.columns:
        return

    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["Body"] = (df["Close"] - df["Open"]).abs()
    df["Range"] = df["High"] - df["Low"]

    df = df.dropna()
    if len(df) < 15:
      return

    # Check latest completed candles
    for i in range(len(df) - 2, len(df)):
      if i < 2:
        continue

      row = df.iloc[i]
      prev = df.iloc[i - 1]
      prev_prev = df.iloc[i - 2]
      curr_date = df.index[i]

      # Mother Candle Check (Previous Candle)
      inside_bar = prev["High"] <= prev_prev["High"] and prev["Low"] >= prev_prev["Low"]
      c_range = prev["Range"]
      c_body = prev["Body"]
      c_atr = prev["ATR"]

      if pd.isna(c_atr) or c_atr == 0:
        continue

      c_ok = (
          c_range >= 1.0 * c_atr
          and c_range <= 2.5 * c_atr
          and (c_body / c_range) >= 0.50
      )

      if inside_bar and c_ok:
        m_high = prev["High"]
        m_low = prev["Low"]
        m_mid = (m_high + m_low) / 2
        tv_link = get_tv_link(ticker)
        clean_name = ticker.replace(".NS", "").replace("^", "")

        # Breakout Up (BUY CE)
        if row["Close"] > m_high and row["Close"] > row["Open"]:
          entry = row["Close"]
          sl = m_mid
          risk = entry - sl
          if risk <= 0:
            continue
          t1 = entry + (risk * 1.0)
          t2 = entry + (risk * 2.0)

          msg = (
              f"🚀 *BUY CE SIGNAL* 🚀\n\n"
              f"📊 *Symbol:* `{clean_name}`\n"
              f"⏰ *Candle Time:* `{curr_date.strftime('%d %b %Y, %H:%M IST')}`\n"
              f"📈 *Timeframe:* `30 min`\n\n"
              f"🔹 *Entry Price:* `{entry:.2f}`\n"
              f"🛑 *Stop Loss:* `{sl:.2f}`\n"
              f"🎯 *Target 1:* `{t1:.2f}`\n"
              f"🎯 *Target 2:* `{t2:.2f}`\n\n"
              f"🔗 [Open Chart on TradingView]({tv_link})"
          )
          send_telegram_message(msg)

        # Breakout Down (BUY PE)
        elif row["Close"] < m_low and row["Close"] < row["Open"]:
          entry = row["Close"]
          sl = m_mid
          risk = sl - entry
          if risk <= 0:
            continue
          t1 = entry - (risk * 1.0)
          t2 = entry - (risk * 2.0)

          msg = (
              f"🔻 *BUY PE SIGNAL* 🔻\n\n"
              f"📊 *Symbol:* `{clean_name}`\n"
              f"⏰ *Candle Time:* `{curr_date.strftime('%d %b %Y, %H:%M IST')}`\n"
              f"📈 *Timeframe:* `30 min`\n\n"
              f"🔹 *Entry Price:* `{entry:.2f}`\n"
              f"🛑 *Stop Loss:* `{sl:.2f}`\n"
              f"🎯 *Target 1:* `{t1:.2f}`\n"
              f"🎯 *Target 2:* `{t2:.2f}`\n\n"
              f"🔗 [Open Chart on TradingView]({tv_link})"
          )
          send_telegram_message(msg)

  except Exception as e:
    print(f"Error scanning {ticker}: {e}")


if __name__ == "__main__":
  for sym in SYMBOLS:
    scan_symbol(sym)
