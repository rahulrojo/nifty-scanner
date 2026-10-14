import os
import json
import time
import logging
from pathlib import Path
from datetime import time as dt_time
from zoneinfo import ZoneInfo
from urllib.parse import quote

import requests
import pandas as pd
import numpy as np
import yfinance as yf


# =====================================================
# ROMYO SCANNER
# 30-MINUTE MOTHER CANDLE
# BUY CE / BUY PE
# =====================================================

IST = ZoneInfo("Asia/Kolkata")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1003921675472").strip()

TEST_MODE = os.getenv("SEND_TEST_MESSAGE", "false").lower() == "true"

INTERVAL = "30m"
PERIOD = "60d"

EMA_LEN = 21
ATR_LEN = 14

MIN_ATR_MULT = 1.0
MAX_ATR_MULT = 2.5
MIN_BODY_PCT = 50

MIN_INSIDE = 1
MAX_BARS = 8

BREAKOUT_BODY_PCT = 40
MAX_BREAKOUT_ATR = 2.0

MIN_SCORE = 2
MAX_RISK_ATR = 2.0

RR1 = 1.0
RR2 = 2.0
TIME_STOP = 6

# Original Pine Script session settings
MOTHER_START = dt_time(9, 15)
MOTHER_END = dt_time(13, 30)

SIGNAL_START = dt_time(9, 15)
SIGNAL_END = dt_time(14, 30)

STATE_FILE = Path("signal_state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


# =====================================================
# STOCK WATCHLIST
# =====================================================

STOCKS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "BHARTIARTL",
    "INFY", "TCS", "ITC", "SBIN", "LICI", "HINDUNILVR",
    "LT", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA",
    "KOTAKBANK", "AXISBANK", "M&M", "ULTRACEMCO", "NTPC",
    "ONGC", "TITAN", "ADANIENT", "ADANIPORTS", "BEL",
    "POWERGRID", "COALINDIA", "TMPV", "TATASTEEL",
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

SYMBOLS = {f"{s}.NS": s for s in STOCKS}
SYMBOLS.update(INDICES)


# =====================================================
# TELEGRAM
# =====================================================

def telegram(message):
    if not TOKEN or not CHAT_ID:
        logging.error("Telegram credentials missing")
        return False

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message,
                "disable_web_page_preview": True,
            },
            timeout=20,
        )

        response.raise_for_status()
        result = response.json()

        if not result.get("ok"):
            logging.error("Telegram API error: %s", result)
            return False

        return True

    except Exception:
        logging.exception("Telegram message failed")
        return False


def send_test():
    now = pd.Timestamp.now(tz=IST)

    message = (
        "🧪 ROMYO 30M SCANNER TEST\n\n"
        "✅ GitHub Actions is running.\n"
        "✅ Telegram connection is working.\n\n"
        f"📅 Date: {now.strftime('%d-%m-%Y')}\n"
        f"🕒 Time: {now.strftime('%I:%M:%S %p')} IST\n"
        "⏱️ Timeframe: 30 Minutes\n\n"
        "This is a test message, not a trading signal."
    )

    if not telegram(message):
        raise RuntimeError("Telegram test failed")

    logging.info("Test message sent")


# =====================================================
# STATE / DUPLICATE PROTECTION
# =====================================================

def load_state():
    try:
        if STATE_FILE.exists():
            return json.loads(
                STATE_FILE.read_text(encoding="utf-8")
            )
    except Exception:
        logging.exception("State file read failed")

    return {"sent": []}


def save_state(state):
    temp = STATE_FILE.with_suffix(".tmp")

    temp.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8"
    )

    temp.replace(STATE_FILE)


# =====================================================
# DOWNLOAD 30-MINUTE DATA
# =====================================================

def download_data(symbol):
    try:
        df = yf.download(
            symbol,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=25,
        )

        if df is None or df.empty:
            logging.warning("%s: no data", symbol)
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]

        if not all(c in df.columns for c in required):
            logging.warning("%s: missing OHLCV", symbol)
            return None

        df = df[required].copy()

        idx = pd.DatetimeIndex(df.index)

        if idx.tz is None:
            idx = idx.tz_localize("UTC")

        df.index = idx.tz_convert(IST)
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        for col in required:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

        df.dropna(
            subset=["Open", "High", "Low", "Close"],
            inplace=True
        )

        # Yahoo timestamps generally label the candle start.
        # Never use a candle which has not finished yet.
        now = pd.Timestamp.now(tz=IST)

        df = df[
            df.index + pd.Timedelta(minutes=30) <= now
        ].copy()

        if len(df) < 50:
            logging.warning(
                "%s: insufficient completed candles (%d)",
                symbol, len(df)
            )
            return None

        # EMA
        df["EMA"] = df["Close"].ewm(
            span=EMA_LEN,
            adjust=False
        ).mean()

        # ATR, using the original Pine calculation.
        previous_close = df["Close"].shift(1)

        true_range = pd.concat(
            [
                df["High"] - df["Low"],
                (df["High"] - previous_close).abs(),
                (df["Low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        df["ATR"] = true_range.rolling(
            ATR_LEN,
            min_periods=ATR_LEN
        ).mean()

        # Intraday VWAP
        typical = (
            df["High"] + df["Low"] + df["Close"]
        ) / 3

        volume = df["Volume"].fillna(0)
        session_key = pd.Series(
            df.index.date,
            index=df.index
        )

        pv = (typical * volume).groupby(session_key).cumsum()
        vv = volume.groupby(session_key).cumsum()

        df["VWAP"] = pv / vv.replace(0, np.nan)

        df["AVG_VOL"] = volume.rolling(
            20,
            min_periods=20
        ).mean()

        return df

    except Exception:
        logging.exception("Download failed for %s", symbol)
        return None


# =====================================================
# SESSION FILTERS
# =====================================================

def in_session(timestamp, start, end):
    t = timestamp.timetz().replace(tzinfo=None)
    return start <= t <= end


# =====================================================
# PINE-STYLE SCANNER
# =====================================================

def scan_symbol(symbol, name, df):
    signals = []

    if df is None or len(df) < 50:
        return signals

    # Persistent state, analogous to Pine var variables.
    mother = None
    pending_ce = False
    pending_pe = False
    pending_ce_high = None
    pending_pe_low = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        previous = df.iloc[i - 1]

        timestamp = df.index[i]

        if pd.isna(row["ATR"]) or row["ATR"] <= 0:
            continue

        # The Pine script resets an automatic mother candle
        # at a new trading day.
        if timestamp.date() != df.index[i - 1].date():
            mother = None

        # ---------------------------------------------
        # MOTHER CANDLE EXPIRES
        # ---------------------------------------------

        if mother is not None:
            age = i - mother["index"]

            if age > MAX_BARS:
                mother = None

        # ---------------------------------------------
        # SELECT AUTOMATIC MOTHER CANDLE
        # ---------------------------------------------

        inside_previous = (
            row["High"] <= previous["High"]
            and row["Low"] >= previous["Low"]
        )

        previous_range = previous["High"] - previous["Low"]
        previous_body = abs(
            previous["Close"] - previous["Open"]
        )

        previous_atr = df.iloc[i - 1]["ATR"]

        previous_in_session = in_session(
            df.index[i - 1],
            MOTHER_START,
            MOTHER_END
        )

        mother_candidate_ok = (
            pd.notna(previous_atr)
            and previous_atr > 0
            and previous_range >= MIN_ATR_MULT * previous_atr
            and previous_range <= MAX_ATR_MULT * previous_atr
            and previous_body >= MIN_BODY_PCT / 100 * previous_range
            and previous_in_session
        )

        # Match Pine: the previous candle becomes the mother
        # when the current candle is inside it.
        if mother is None and inside_previous and mother_candidate_ok:
            mother = {
                "index": i - 1,
                "high": float(previous["High"]),
                "low": float(previous["Low"]),
                "volume": float(previous["Volume"]),
                "inside_count": 1,
                "last_child": float(row["High"] - row["Low"]),
            }

        elif mother is not None and i > mother["index"]:
            mh = mother["high"]
            ml = mother["low"]

            if row["High"] <= mh and row["Low"] >= ml:
                mother["inside_count"] += 1
                mother["last_child"] = float(
                    row["High"] - row["Low"]
                )

        if mother is None:
            continue

        # ---------------------------------------------
        # SESSION / MATURITY
        # ---------------------------------------------

        signal_session = in_session(
            timestamp,
            SIGNAL_START,
            SIGNAL_END
        )

        armed = mother["inside_count"] >= MIN_INSIDE

        if not signal_session or not armed:
            continue

        mh = mother["high"]
        ml = mother["low"]
        mr = mh - ml

        if mr <= 0:
            continue

        # ---------------------------------------------
        # QUALITY SCORE: ORIGINAL Q/4
        # ---------------------------------------------

        has_volume = float(row["Volume"]) > 0

        sc1 = (
            not has_volume
            or mother["volume"] >= row["AVG_VOL"]
            if pd.notna(row["AVG_VOL"])
            else not has_volume
        )

        sc2 = mother["inside_count"] >= 2

        sc3 = (
            mother["last_child"] <= 0.6 * mr
        )

        sc4 = (
            not has_volume
            or (
                pd.notna(row["AVG_VOL"])
                and float(row["Volume"]) >= float(row["AVG_VOL"])
            )
        )

        score = int(sc1) + int(sc2) + int(sc3) + int(sc4)

        if score < MIN_SCORE:
            continue

        # ---------------------------------------------
        # BREAKOUT RULES
        # ---------------------------------------------

        body_range = float(row["High"] - row["Low"])

        if body_range <= 0:
            continue

        body_ok = (
            abs(float(row["Close"] - row["Open"]))
            >= BREAKOUT_BODY_PCT / 100 * body_range
        )

        size_ok = (
            body_range <= MAX_BREAKOUT_ATR * float(row["ATR"])
        )

        if not body_ok or not size_ok:
            continue

        trend_ce = row["Close"] > row["EMA"]
        trend_pe = row["Close"] < row["EMA"]

        vwap_ce = (
            pd.isna(row["VWAP"])
            or row["Close"] > row["VWAP"]
        )

        vwap_pe = (
            pd.isna(row["VWAP"])
            or row["Close"] < row["VWAP"]
        )

        breakout_up = (
            row["Close"] > mh
            and row["Close"] > row["Open"]
            and trend_ce
            and vwap_ce
        )

        breakout_down = (
            row["Close"] < ml
            and row["Close"] < row["Open"]
            and trend_pe
            and vwap_pe
        )

        # ---------------------------------------------
        # PENDING CONFIRMATION
        # ---------------------------------------------

        if breakout_up:
            pending_ce = True
            pending_ce_high = float(row["High"])

        if breakout_down:
            pending_pe = True
            pending_pe_low = float(row["Low"])

        # Same-color candle, fully beyond breakout candle.
        confirm_ce = (
            pending_ce
            and row["Close"] > row["Open"]
            and row["Low"] > pending_ce_high
        )

        confirm_pe = (
            pending_pe
            and row["Close"] < row["Open"]
            and row["High"] < pending_pe_low
        )

        # ---------------------------------------------
        # RISK FILTER
        # ---------------------------------------------

        midpoint = (mh + ml) / 2

        entry = float(row["Close"])

        risk_ce = entry - midpoint
        risk_pe = midpoint - entry

        risk_ok_ce = (
            risk_ce > 0
            and risk_ce <= MAX_RISK_ATR * float(row["ATR"])
        )

        risk_ok_pe = (
            risk_pe > 0
            and risk_pe <= MAX_RISK_ATR * float(row["ATR"])
        )

        ce_signal = confirm_ce and risk_ok_ce
        pe_signal = confirm_pe and risk_ok_pe

        if ce_signal:
            pending_ce = False

        if pe_signal:
            pending_pe = False

        # ---------------------------------------------
        # BUILD SIGNAL
        # ---------------------------------------------

        if ce_signal or pe_signal:
            side = "BUY CE" if ce_signal else "BUY PE"

            stop = midpoint

            risk = (
                entry - stop
                if side == "BUY CE"
                else stop - entry
            )

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

            signals.append({
                "symbol": symbol,
                "name": name,
                "side": side,
                "entry": entry,
                "sl": stop,
                "t1": target1,
                "t2": target2,
                "score": score,
                "quality": int(round(score * 10 / 4)),
                "time": timestamp.isoformat(),
            })

            # Original Pine logic deactivates the mother
            # after a breakout signal is generated.
            mother = None

        elif breakout_up or breakout_down:
            # Breakout occurred, so the mother is no longer active.
            mother = None

    return signals


# =====================================================
# FORMAT TELEGRAM SIGNAL
# =====================================================

def format_signal(signal):
    timestamp = pd.Timestamp(signal["time"])

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(IST)
    else:
        timestamp = timestamp.tz_convert(IST)

    tv_symbol = quote(signal["symbol"], safe="")
    tv_link = (
        "https://www.tradingview.com/chart/"
        f"?symbol={tv_symbol}"
    )

    icon = "🟢" if signal["side"] == "BUY CE" else "🔴"

    return (
        f"{icon} ROMYO {signal['side']}\n\n"
        f"📊 Stock/Index: {signal['name']}\n"
        f"Symbol: {signal['symbol']}\n"
        f"⏱️ Timeframe: 30 Minutes\n"
        f"📅 Date: {timestamp.strftime('%d-%m-%Y')}\n"
        f"🕒 Candle Time: {timestamp.strftime('%I:%M %p')} IST\n\n"
        f"💰 Entry: {signal['entry']:.2f}\n"
        f"🛑 Stop Loss: {signal['sl']:.2f}\n"
        f"🎯 Target 1: {signal['t1']:.2f}\n"
        f"🎯 Target 2: {signal['t2']:.2f}\n"
        f"⏳ Time Stop: {TIME_STOP} candles\n\n"
        f"⭐ Quality: {signal['quality']}/10\n"
        f"📋 Original Score: Q{signal['score']}/4\n\n"
        f"📈 TradingView Chart:\n{tv_link}\n\n"
        "⚠️ Signal is based on the underlying chart. "
        "Option strike and premium are not calculated."
    )


# =====================================================
# MAIN
# =====================================================

def main():
    logging.info("ROMYO 30m scanner started")

    if not TOKEN or not CHAT_ID:
        raise RuntimeError(
            "Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID "
            "to GitHub repository Secrets."
        )

    if TEST_MODE:
        send_test()

    state = load_state()
    sent = set(state.get("sent", []))

    now = pd.Timestamp.now(tz=IST)

    total_data = 0
    total_signals = 0
    errors = 0

    for symbol, name in SYMBOLS.items():
        logging.info("Scanning %s", symbol)

        df = download_data(symbol)

        if df is None:
            errors += 1
            continue

        total_data += 1

        try:
            signals = scan_symbol(symbol, name, df)

            for signal in signals:
                candle_time = pd.Timestamp(signal["time"])

                if candle_time.tzinfo is None:
                    candle_time = candle_time.tz_localize(IST)
                else:
                    candle_time = candle_time.tz_convert(IST)

                # Scan recent completed signals, including the
                # previous trading day. No 90-minute cutoff.
                age = (now - candle_time).total_seconds()

                if age < 0:
                    continue

                if age > 3 * 24 * 60 * 60:
                    continue

                key = (
                    f"{signal['symbol']}|"
                    f"{signal['time']}|"
                    f"{signal['side']}"
                )

                if key in sent:
                    continue

                if telegram(format_signal(signal)):
                    sent.add(key)
                    total_signals += 1

                    state["sent"] = sorted(sent)[-5000:]
                    save_state(state)

                    logging.info(
                        "SIGNAL SENT: %s %s Q%d/4",
                        signal["symbol"],
                        signal["side"],
                        signal["score"]
                    )
                else:
                    logging.error(
                        "Could not send signal: %s",
                        key
                    )

        except Exception:
            errors += 1
            logging.exception("Scan failed for %s", symbol)

        time.sleep(0.25)

    state["sent"] = sorted(sent)[-5000:]
    save_state(state)

    logging.info(
        "FINISHED: data_symbols=%d, new_signals=%d, errors=%d",
        total_data,
        total_signals,
        errors
    )


if __name__ == "__main__":
    main()
