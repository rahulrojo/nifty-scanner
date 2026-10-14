import os
from datetime import datetime
import pandas as pd
import pandas_ta as ta
import requests
import yfinance as yf

# ===== CONFIGURATION =====
TIMEFRAME = "30m"
PERIOD = "5d"  # 30m data ke liye yfinance 5-7 days ka data support karta hai

# Major F&O Indexes + Top 100 Liquid F&O Stocks List
SYMBOLS = [
    # Top Major Indexes
    "^NSEI",
    "^NSEBANK",
    "FINNIFTY.NS",
    "NIFTY_MID_SELECT.NS",
    # Top F&O Stocks (Liquid & High Volume)
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
    print("Telegram credentials missing! Printing message:")
    print(message)
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
  """TradingView chart URL generator."""
  symbol_map = {
      "^NSEI": "NIFTY",
      "^NSEBANK": "BANKNIFTY",
      "FINNIFTY.NS": "FINNIFTY",
      "NIFTY_MID_SELECT.NS": "MIDCPNIFTY",
  }

  if ticker in symbol_map:
    tv_symbol = symbol_map[ticker]
  else:
    tv_symbol = ticker.replace(".NS", "")

  return f"https://in.tradingview.com/chart/?symbol=NSE:{tv_symbol}&interval=30"


def scan_symbol(ticker):
  try:
    print(f"Scanning {ticker}...")
    df = yf.download(
        ticker, period=PERIOD, interval=TIMEFRAME, progress=False
    )
    if df.empty or len(df) < 30:
      return

    # MultiIndex column flatten kar rahe hain
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_0

    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["EMA"] = ta.ema(df["Close"], length=21)
    df["Body"] = (df["Close"] - df["Open"]).abs()
    df["Range"] = df["High"] - df["Low"]

    # Recent candles check (latest completed candle)
    for i in range(len(df) - 2, len(df)):
      if i < 15:
        continue

      row = df.iloc[i]
      prev = df.iloc[i - 1]
      curr_date = df.index[i]

      # Step 1: Mother Candle Check (Previous Candle)
      inside_bar = (
          prev["High"] <= df.iloc[i - 2]["High"]
          and prev["Low"] >= df.iloc[i - 2]["Low"]
      )
      c_range = prev["Range"]
      c_body = prev["Body"]
      c_atr = prev["ATR"]

      # Pine Script conditions match
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

        # Step 2: Confirmation Candle (Breakout Above Mother High)
        if row["Close"] > m_high and row["Close"] > row["Open"]:
          entry = row["Close"]
          sl = m_mid
          risk = entry - sl
          t1 = entry + (risk * 1.0)
          t2 = entry + (risk * 2.0)

          msg = (
              f"🚀 *BUY CE SIGNAL* 🚀\n\n"
              f"📊 *Symbol:* `{clean_name}`\n"
              f"⏰ *Candle Date & Time:* `{curr_date.strftime('%d %b %Y, %H:%M IST')}`\n"
              f"📈 *Timeframe:* `30 min`\n\n"
              f"🔹 *Entry Price:* `{entry:.2f}`\n"
              f"🛑 *Stop Loss:* `{sl:.2f}`\n"
              f"🎯 *Target 1 (1:1):* `{t1:.2f}`\n"
              f"🎯 *Target 2 (1:2):* `{t2:.2f}`\n\n"
              f"🔗 [Open Chart on TradingView]({tv_link})"
          )
          send_telegram_message(msg)

        # Step 2: Confirmation Candle (Breakout Below Mother Low)
        elif row["Close"] < m_low and row["Close"] < row["Open"]:
          entry = row["Close"]
          sl = m_mid
          risk = sl - entry
          t1 = entry - (risk * 1.0)
          t2 = entry - (risk * 2.0)

          msg = (
              f"🔻 *BUY PE SIGNAL* 🔻\n\n"
              f"📊 *Symbol:* `{clean_name}`\n"
              f"⏰ *Candle Date & Time:* `{curr_date.strftime('%d %b %Y, %H:%M IST')}`\n"
              f"📈 *Timeframe:* `30 min`\n\n"
              f"🔹 *Entry Price:* `{entry:.2f}`\n"
              f"🛑 *Stop Loss:* `{sl:.2f}`\n"
              f"🎯 *Target 1 (1:1):* `{t1:.2f}`\n"
              f"🎯 *Target 2 (1:2):* `{t2:.2f}`\n\n"
              f"🔗 [Open Chart on TradingView]({tv_link})"
          )
          send_telegram_message(msg)

  except Exception as e:
    print(f"Error scanning {ticker}: {e}")


if __name__ == "__main__":
  for sym in SYMBOLS:
    scan_symbol(sym)
