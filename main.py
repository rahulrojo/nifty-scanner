import os
import json
import time
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.parse import quote

import requests
import pandas as pd
import numpy as np
import yfinance as yf

# =========================================================
# ROMYO MOTHER CANDLE SCANNER
# NSE TOP 100 + MAJOR INDIAN INDICES
# TIMEFRAME: 30 MINUTES
# =========================================================

IST = ZoneInfo("Asia/Kolkata")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID", "1003921675472"
).strip()

# Scan completed 30-minute candles only.
INTERVAL = "30m"
PERIOD = "60d"

EMA_LENGTH = 21
ATR_LENGTH = 14
MAX_MOTHER_BARS = 8

MIN_ATR_MULT = 1.0
MAX_ATR_MULT = 2.5
MIN_BODY_PCT = 50

MIN_INSIDE_CANDLES = 1
BREAKOUT_BODY_PCT = 40
MAX_BREAKOUT_ATR = 2.0
MAX_RISK_ATR = 2.0

RR1 = 1.0
RR2 = 2.0

MIN_QUALITY = 6
REQUEST_DELAY = 1.0

STATE_FILE = Path("signal_state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =========================================================
# NSE TOP 100 WATCHLIST
# This is a practical large-cap watchlist, not a live
# ranking of the current Nifty 100 constituents.
# Review/update constituents periodically.
# =========================================================

STOCKS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "BHARTIARTL",
    "INFY", "TCS", "ITC", "SBIN", "LICI", "HINDUNILVR",
    "LT", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA",
    "KOTAKBANK", "AXISBANK", "M&M", "ULTRACEMCO", "NTPC",
    "ONGC", "TITAN", "ADANIENT", "ADANIPORTS", "BEL",
    "POWERGRID", "COALINDIA", "TATAMOTORS", "TATASTEEL",
    "JSWSTEEL", "WIPRO", "TECHM", "TRENT", "ETERNAL",
    "BAJAJFINSV", "GRASIM", "ASIANPAINT", "HINDALCO",
    "NESTLEIND", "SBILIFE", "HDFCLIFE", "INDUSINDBK",
    "DRREDDY", "CIPLA", "APOLLOHOSP", "EICHERMOT",
    "HEROMOTOCO", "BAJAJ-AUTO", "TVSMOTOR", "DIVISLAB",
    "BRITANNIA", "TATACONSUM", "PIDILITIND", "DABUR",
    "GODREJCP", "COLPAL", "MARICO", "BERGEPAINT",
    "BPCL", "IOC", "GAIL", "HINDPETRO", "VEDL",
    "SHREECEM", "AMBUJACEM", "ACC", "DLF", "LODHA",
    "PHOENIXLTD", "OBEROIRLTY", "INDIGO", "HAL",
    "MAZDOCK", "BHEL", "IRCTC", "IRFC", "RECLTD",
    "PFC", "CHOLAFIN", "SHRIRAMFIN", "BANKBARODA",
    "PNB", "CANBK", "UNIONBANK", "IDFCFIRSTB",
    "FEDERALBNK", "YESBANK", "AUROPHARMA", "ZYDUSLIFE",
    "TORNTPHARM", "LUPIN", "MAXHEALTH", "FORTIS",
    "DMART", "JIOFIN", "PAYTM", "POLICYBZR",
    "PERSISTENT", "COFORGE"
]

# Major indices. Yahoo Finance symbols are used here.
INDICES = {
    "^NSEI": "NIFTY 50",
    "^NSEBANK": "NIFTY BANK",
    "^CNXIT": "NIFTY IT",
    "^CNXPHARMA": "NIFTY PHARMA",
    "^CNXAUTO": "NIFTY AUTO",
    "^CNXMETAL": "NIFTY METAL",
    "^CNXFMCG": "NIFTY FMCG",
    "^CNXENERGY": "NIFTY ENERGY",
    "^CNXREALTY": "NIFTY REALTY",
    "^CNXPSUBANK": "NIFTY PSU BANK",
    "^INDIAVIX": "INDIA VIX",
}

# Yahoo Finance tickers for NSE equities.
SYMBOLS = {
    f"{stock}.NS": stock for stock in STOCKS
}

# Add index symbols.
for ticker, name in INDICES.items():
    SYMBOLS[ticker] = name


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error(
            "Telegram credentials missing. "
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."
        )
        return False

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_TOKEN}/sendMessage"
    )

    try:
        response = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "disable_web_page_preview": True
            },
            timeout=20
        )

        response.raise_for_status()
        result = response.json()

        if not result.get("ok"):
            logging.error("Telegram API error: %s", result)
            return False

        return True

    except Exception as exc:
        logging.error("Telegram send failed: %s", exc)
        return False


# =========================================================
# STATE / DUPLICATE PROTECTION
# =========================================================

def load_state():
    try:
        if STATE_FILE.exists():
            with STATE_FILE.open("r", encoding="utf-8") as file:
                return json.load(file)
    except Exception as exc:
        logging.error("Could not read state file: %s", exc)

    return {"sent": []}


def save_state(state):
    temp_file = STATE_FILE.with_suffix(".tmp")

    with temp_file.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)

    temp_file.replace(STATE_FILE)


def signal_key(symbol, candle_time, side):
    return f"{symbol}|{candle_time}|{side}"


# =========================================================
# INDICATORS
# =========================================================

def calculate_indicators(df):
    df = df.copy()

    df["EMA"] = df["Close"].ewm(
        span=EMA_LENGTH, adjust=False
    ).mean()

    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.rolling(
        ATR_LENGTH, min_periods=ATR_LENGTH
    ).mean()

    # Intraday VWAP: reset every Indian trading day.
    typical_price = (
        df["High"] + df["Low"] + df["Close"]
    ) / 3.0

    volume = df["Volume"].fillna(0)

    trading_day = pd.Series(
        df.index.date, index=df.index
    )

    pv = typical_price * volume

    cumulative_pv = pv.groupby(trading_day).cumsum()
    cumulative_volume = volume.groupby(trading_day).cumsum()

    df["VWAP"] = (
        cumulative_pv /
        cumulative_volume.replace(0, np.nan)
    )

    df["AVG_VOL"] = volume.rolling(20).mean()

    return df


# =========================================================
# DATA DOWNLOAD
# =========================================================

def download_data(ticker):
    try:
        df = yf.download(
            ticker,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            logging.warning("No data for %s", ticker)
            return None

        # Recent yfinance versions can return MultiIndex columns.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = [
            "Open", "High", "Low", "Close", "Volume"
        ]

        if not all(col in df.columns for col in required):
            logging.warning("Missing OHLCV columns: %s", ticker)
            return None

        df = df[required].copy()

        index = pd.DatetimeIndex(df.index)

        if index.tz is None:
            # Yahoo intraday timestamps should be timezone-aware.
            # If naive, treat them as UTC rather than silently
            # interpreting them as Indian local time.
            index = index.tz_localize("UTC")

        df.index = index.tz_convert(IST)
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        for col in required:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

        df = df.dropna(
            subset=["Open", "High", "Low", "Close"]
        )

        # Discard the currently forming candle.
        now = pd.Timestamp.now(tz=IST)
        df = df[df.index + pd.Timedelta(minutes=30) <= now]

        if len(df) < 50:
            logging.warning("Not enough candles for %s", ticker)
            return None

        return calculate_indicators(df)

    except Exception as exc:
        logging.error("Download error %s: %s", ticker, exc)
        return None


# =========================================================
# QUALITY SCORE: 0 TO 10
# =========================================================

def quality_score(
    df, i, mother_high, mother_low, inside_count,
    side, breakout_idx
):
    score = 0

    candle = df.iloc[i]
    breakout = df.iloc[breakout_idx]

    mother_range = mother_high - mother_low
    candle_range = candle["High"] - candle["Low"]

    if mother_range <= 0 or candle_range <= 0:
        return 0

    # 1. Strong mother candle body.
    mother = df.iloc[i - inside_count - 1]
    m_range = mother["High"] - mother["Low"]

    if m_range > 0:
        m_body_pct = abs(
            mother["Close"] - mother["Open"]
        ) / m_range * 100

        if m_body_pct >= 60:
            score += 1

    # 2. Compression.
    if inside_count >= 2:
        score += 1

    # 3. Confirmation candle body.
    body_pct = abs(
        candle["Close"] - candle["Open"]
    ) / candle_range * 100

    if body_pct >= 60:
        score += 1

    # 4. Breakout candle body.
    b_range = breakout["High"] - breakout["Low"]

    if b_range > 0:
        b_body_pct = abs(
            breakout["Close"] - breakout["Open"]
        ) / b_range * 100

        if b_body_pct >= 60:
            score += 1

    # 5. Trend.
    if side == "BUY CE":
        if candle["Close"] > candle["EMA"]:
            score += 1
    else:
        if candle["Close"] < candle["EMA"]:
            score += 1

    # 6. VWAP.
    if pd.notna(candle["VWAP"]):
        if side == "BUY CE" and candle["Close"] > candle["VWAP"]:
            score += 1
        elif side == "BUY PE" and candle["Close"] < candle["VWAP"]:
            score += 1

    # 7. Volume.
    if (
        pd.notna(candle["AVG_VOL"])
        and candle["AVG_VOL"] > 0
        and candle["Volume"] >= candle["AVG_VOL"]
    ):
        score += 1

    # 8. Confirmation candle must be clean.
    if side == "BUY CE" and candle["Low"] > breakout["High"]:
        score += 1

    if side == "BUY PE" and candle["High"] < breakout["Low"]:
        score += 1

    # 9. Breakout close beyond the mother.
    if side == "BUY CE" and breakout["Close"] > mother_high:
        score += 1

    if side == "BUY PE" and breakout["Close"] < mother_low:
        score += 1

    # 10. Controlled risk.
    atr = candle["ATR"]

    if pd.notna(atr) and atr > 0:
        entry = candle["Close"]

        stop = (
            (mother_high + mother_low) / 2
            if side == "BUY CE"
            else (mother_high + mother_low) / 2
        )

        risk = (
            entry - stop
            if side == "BUY CE"
            else stop - entry
        )

        if 0 < risk <= MAX_RISK_ATR * atr:
            score += 1

    return min(score, 10)


# =========================================================
# SIGNAL SCANNER
# =========================================================

def scan_symbol(ticker, display_name, df):
    signals = []

    if df is None or len(df) < 50:
        return signals

    # We scan historical candles as well as the latest completed
    # candle. Duplicate protection prevents repeat Telegram alerts.
    for i in range(30, len(df) - 1):
        row = df.iloc[i]

        if pd.isna(row["ATR"]) or row["ATR"] <= 0:
            continue

        # Mother candle candidate is the previous candle.
        mother_idx = i - 2
        mother = df.iloc[mother_idx]

        m_high = float(mother["High"])
        m_low = float(mother["Low"])
        m_range = m_high - m_low

        if m_range <= 0:
            continue

        m_atr = float(mother["ATR"]) if pd.notna(
            mother["ATR"]
        ) else 0

        if m_atr <= 0:
            continue

        m_body = abs(
            float(mother["Close"]) - float(mother["Open"])
        )

        # Mother candle quality.
        if not (
            MIN_ATR_MULT * m_atr
            <= m_range
            <= MAX_ATR_MULT * m_atr
        ):
            continue

        if m_body < MIN_BODY_PCT / 100 * m_range:
            continue

        # Compression candles must stay inside mother.
        inside_count = 0
        valid_compression = True

        for j in range(mother_idx + 1, i):
            child = df.iloc[j]

            if (
                child["High"] <= m_high
                and child["Low"] >= m_low
            ):
                inside_count += 1
            else:
                valid_compression = False
                break

        if not valid_compression:
            continue

        if inside_count < MIN_INSIDE_CANDLES:
            continue

        # Breakout must occur after compression.
        breakout_idx = i - 1
        breakout = df.iloc[breakout_idx]

        # Do not allow a setup older than the allowed window.
        if i - mother_idx > MAX_MOTHER_BARS + 1:
            continue

        # Breakout candle filters.
        b_range = float(
            breakout["High"] - breakout["Low"]
        )

        if b_range <= 0:
            continue

        b_body = abs(
            float(breakout["Close"] - breakout["Open"])
        )

        if b_body < BREAKOUT_BODY_PCT / 100 * b_range:
            continue

        if b_range > MAX_BREAKOUT_ATR * float(row["ATR"]):
            continue

        # Confirmation candle must be fully outside the
        # breakout candle, without touching it.
        side = None

        if (
            breakout["Close"] > m_high
            and breakout["Close"] > breakout["Open"]
            and row["Close"] > row["Open"]
            and row["Low"] > breakout["High"]
            and row["Close"] > row["EMA"]
            and (
                pd.isna(row["VWAP"])
                or row["Close"] > row["VWAP"]
            )
        ):
            side = "BUY CE"

        elif (
            breakout["Close"] < m_low
            and breakout["Close"] < breakout["Open"]
            and row["Close"] < row["Open"]
            and row["High"] < breakout["Low"]
            and row["Close"] < row["EMA"]
            and (
                pd.isna(row["VWAP"])
                or row["Close"] < row["VWAP"]
            )
        ):
            side = "BUY PE"

        if side is None:
            continue

        entry = float(row["Close"])
        sl = (m_high + m_low) / 2

        risk = entry - sl if side == "BUY CE" else sl - entry

        if risk <= 0 or risk > MAX_RISK_ATR * float(row["ATR"]):
            continue

        target1 = (
            entry + risk * RR1
            if side == "BUY CE"
            else entry - risk * RR1
        )

        target2 = (
            entry + risk * RR2
            if side == "BUY CE"
            else entry - risk * RR2
        )

        score = quality_score(
            df, i, m_high, m_low, inside_count,
            side, breakout_idx
        )

        if score < MIN_QUALITY:
            continue

        candle_time = df.index[i]

        # The alert is for the completed confirmation candle.
        signals.append({
            "ticker": ticker,
            "name": display_name,
            "side": side,
            "entry": entry,
            "sl": sl,
            "target1": target1,
            "target2": target2,
            "score": score,
            "candle_time": candle_time.isoformat(),
            "mother_high": m_high,
            "mother_low": m_low
        })

    return signals


# =========================================================
# MESSAGE FORMAT
# =========================================================

def format_message(signal):
    side = signal["side"]

    emoji = "🟢" if side == "BUY CE" else "🔴"

    candle_time = pd.Timestamp(signal["candle_time"])
    candle_time = candle_time.tz_convert(IST)

    date_str = candle_time.strftime("%d-%m-%Y")
    time_str = candle_time.strftime("%I:%M %p")

    tv_symbol = quote(signal["ticker"], safe="")
    tv_link = (
        "https://www.tradingview.com/chart/"
        f"?symbol={tv_symbol}"
    )

    return (
        f"{emoji} ROMYO {side} SIGNAL\n\n"
        f"📊 Stock/Index: {signal['name']}\n"
        f"🔎 Symbol: {signal['ticker']}\n"
        f"⏱️ Timeframe: 30 Minutes\n"
        f"📅 Candle Date: {date_str}\n"
        f"🕒 Candle Time: {time_str} IST\n\n"
        f"💰 Entry: {signal['entry']:.2f}\n"
        f"🛑 Stop Loss: {signal['sl']:.2f}\n"
        f"🎯 Target 1: {signal['target1']:.2f}\n"
        f"🎯 Target 2: {signal['target2']:.2f}\n\n"
        f"⭐ Quality: {signal['score']}/10\n"
        f"📦 Setup: Mother Candle Breakout\n"
        f"✅ Confirmation: PASSED\n\n"
        f"📈 TradingView Chart:\n{tv_link}\n\n"
        f"⚠️ Signal is based on the underlying chart. "
        f"CE/PE option prices are not calculated."
    )


# =========================================================
# MAIN
# =========================================================

def main():
    logging.info("ROMYO scanner starting")

    if not TELEGRAM_TOKEN:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN GitHub Secret"
        )

    state = load_state()
    sent = set(state.get("sent", []))

    new_count = 0

    for ticker, display_name in SYMBOLS.items():
        logging.info("Scanning %s", ticker)

        df = download_data(ticker)

        if df is None:
            time.sleep(REQUEST_DELAY)
            continue

        try:
            signals = scan_symbol(ticker, display_name, df)

            for signal in signals:
                key = signal_key(
                    ticker,
                    signal["candle_time"],
                    signal["side"]
                )

                if key in sent:
                    continue

                # Avoid sending old historical signals on first run.
                signal_time = pd.Timestamp(
                    signal["candle_time"]
                )

                now = pd.Timestamp.now(tz=IST)

                # Only alert on candles completed within the
                # previous 90 minutes.
                age_minutes = (
                    now - signal_time - pd.Timedelta(minutes=30)
                ).total_seconds() / 60

                if age_minutes > 90 or age_minutes < -5:
                    continue

                message = format_message(signal)

                if send_telegram(message):
                    sent.add(key)
                    new_count += 1

                    # Persist immediately so an interrupted run
                    # is less likely to resend the same signal.
                    state["sent"] = list(sent)[-5000:]
                    save_state(state)

                    logging.info(
                        "Sent %s for %s",
                        signal["side"], ticker
                    )

        except Exception:
            logging.exception("Scanning failed for %s", ticker)

        time.sleep(REQUEST_DELAY)

    state["sent"] = list(sent)[-5000:]
    save_state(state)

    logging.info("Scan complete. New signals: %d", new_count)


if __name__ == "__main__":
    main()
