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
# ROMYO SCANNER - MATCHES THE PROVIDED PINE LOGIC
# Timeframe: 30 minutes. Pine Script itself is unchanged.
# =====================================================

IST = ZoneInfo("Asia/Kolkata")
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1003921675472").strip()
TEST_MODE = os.getenv("SEND_TEST_MESSAGE", "false").lower() == "true"

INTERVAL = "30m"
PERIOD = "60d"
ATR_LEN = 14
EMA_LEN = 21

# Mirrors the Pine inputs/defaults supplied by the user.
MODE = os.getenv("MOTHER_MODE", "Auto").strip().title()  # Auto / Manual
MANUAL_TIME = pd.Timestamp("2026-10-01 09:15:00", tz=IST)

MIN_ATR_MULT = 1.0
MAX_ATR_MULT = 2.5
MIN_BODY_PCT = 50
MIN_INSIDE = 1
MAX_BARS = 8

USE_SESSION = True
MOTHER_START = dt_time(9, 15)
MOTHER_END = dt_time(13, 30)
SIGNAL_START = dt_time(9, 15)
SIGNAL_END = dt_time(14, 30)

USE_CLOSE = True
BUFFER_PCT = 0.0
BREAKOUT_BODY_PCT = 40
MAX_BREAKOUT_ATR = 2.0
USE_TREND = True
USE_VWAP = True
MIN_SCORE = 2
FULLY_OUTSIDE = False
KEEP_ALIVE = True

SL_MODE = "Mother Midpoint (50%)"  # or "Opposite end of mother"
RR1 = 1.0
RR2 = 2.0
MAX_RISK_ATR = 2.0
TIME_STOP = 6

STATE_FILE = Path("signal_state.json")
MAX_SIGNAL_AGE_HOURS = 72
TRADINGVIEW_BASE = "https://www.tradingview.com/chart/?symbol="

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
    "^NSEI": ("NIFTY 50", "NSE:NIFTY"),
    "^NSEBANK": ("NIFTY BANK", "NSE:BANKNIFTY"),
    "^CNXIT": ("NIFTY IT", "NSE:CNXIT"),
    "^CNXPHARMA": ("NIFTY PHARMA", "NSE:CNXPHARMA"),
    "^CNXAUTO": ("NIFTY AUTO", "NSE:CNXAUTO"),
    "^CNXMETAL": ("NIFTY METAL", "NSE:CNXMETAL"),
    "^CNXFMCG": ("NIFTY FMCG", "NSE:CNXFMCG"),
    "^CNXENERGY": ("NIFTY ENERGY", "NSE:CNXENERGY"),
    "^CNXREALTY": ("NIFTY REALTY", "NSE:CNXREALTY"),
    "^CNXPSUBANK": ("NIFTY PSU BANK", "NSE:CNXPSUBANK"),
    "^INDIAVIX": ("INDIA VIX", "NSE:INDIAVIX"),
}

SYMBOLS = {f"{s}.NS": {"name": s, "tv": f"NSE:{s}"} for s in STOCKS}
SYMBOLS.update({
    symbol: {"name": info[0], "tv": info[1]}
    for symbol, info in INDICES.items()
})


def telegram(message):
    if not TOKEN or not CHAT_ID:
        logging.error("Telegram credentials missing.")
        return False
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        response = requests.post(
            url,
            data={"chat_id": CHAT_ID, "text": message, "disable_web_page_preview": True},
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
    msg = (
        "🧪 ROMYO 30M SCANNER TEST\n\n"
        "✅ GitHub Actions is running.\n"
        "✅ Telegram connection is working.\n\n"
        f"📅 Date: {now:%d-%m-%Y}\n"
        f"🕒 Time: {now:%I:%M:%S %p} IST\n"
        "⏱️ Timeframe: 30 Minutes\n\n"
        "This is a test message, not a trading signal."
    )
    if not telegram(msg):
        raise RuntimeError("Telegram test failed")
    logging.info("Test message sent")


def load_state():
    try:
        if STATE_FILE.exists():
            obj = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and isinstance(obj.get("sent", []), list):
                return obj
    except Exception:
        logging.exception("State file read failed")
    return {"sent": []}


def save_state(state):
    temp = STATE_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp.replace(STATE_FILE)


def pine_rma(series, length):
    """Pine ta.rma equivalent: seed with SMA, then Wilder recurrence."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) < length:
        return pd.Series(out, index=series.index)
    valid_count = 0
    seed_sum = 0.0
    prev = np.nan
    for i, value in enumerate(values):
        if np.isnan(value):
            out[i] = prev
            continue
        if np.isnan(prev):
            valid_count += 1
            seed_sum += value
            if valid_count == length:
                prev = seed_sum / length
                out[i] = prev
        else:
            prev = (prev * (length - 1) + value) / length
            out[i] = prev
    return pd.Series(out, index=series.index)


def download_data(symbol):
    try:
        df = yf.download(
            symbol, period=PERIOD, interval=INTERVAL, auto_adjust=False,
            progress=False, threads=False, timeout=25
        )
        if df is None or df.empty:
            logging.warning("%s: no data", symbol)
            return None
        if isinstance(df.columns, pd.MultiIndex):
            # yfinance may return (Price, Ticker) or (Ticker, Price).
            if "Open" in df.columns.get_level_values(0):
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = df.columns.get_level_values(-1)

        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required):
            logging.warning("%s: missing OHLCV columns", symbol)
            return None
        df = df[required].copy()
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is None:
            # Yahoo intraday indexes are normally exchange-local/UTC-aware;
            # naive indexes are treated as UTC as in the original script.
            idx = idx.tz_localize("UTC")
        df.index = idx.tz_convert(IST)
        df = df[~df.index.duplicated(keep="last")].sort_index()

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

        # Yahoo labels intraday bars by their start. Only use fully closed bars.
        now = pd.Timestamp.now(tz=IST)
        df = df[df.index + pd.Timedelta(minutes=30) <= now].copy()
        if len(df) < 50:
            logging.warning("%s: insufficient completed candles (%d)", symbol, len(df))
            return None

        # Pine ta.ema uses alpha=2/(length+1), adjust=False is the matching recurrence.
        df["EMA"] = df["Close"].ewm(span=EMA_LEN, adjust=False, min_periods=EMA_LEN).mean()

        prev_close = df["Close"].shift(1)
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        # Pine ta.atr(14) = ta.rma(ta.tr(true), 14).
        df["ATR"] = pine_rma(tr, ATR_LEN)

        typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
        vol = df["Volume"].fillna(0.0)
        session = pd.Series(df.index.date, index=df.index)
        pv = (typical * vol).groupby(session).cumsum()
        vv = vol.groupby(session).cumsum()
        df["VWAP"] = (pv / vv.replace(0, np.nan)).where(vv > 0)

        # Pine ta.sma(volume, 20): rolling arithmetic mean.
        df["AVG_VOL"] = vol.rolling(20, min_periods=20).mean()
        return df
    except Exception:
        logging.exception("Download failed for %s", symbol)
        return None


def in_session(timestamp, start, end):
    t = timestamp.timetz().replace(tzinfo=None)
    return start <= t <= end


def pine_bool(value):
    return bool(value) if pd.notna(value) else False


def scan_symbol(symbol, meta, df):
    """Sequentially mirror the supplied Pine script's state transitions."""
    signals = []
    if df is None or len(df) < 50:
        return signals

    mother = None
    pending_ce = False
    pending_pe = False
    pending_ce_high = np.nan
    pending_pe_low = np.nan
    previous_date = None

    for i in range(len(df)):
        row = df.iloc[i]
        timestamp = df.index[i]
        if pd.isna(row["ATR"]) or row["ATR"] <= 0:
            continue

        day_change = previous_date is not None and timestamp.date() != previous_date
        previous_date = timestamp.date()

        # Pine expiry runs on confirmed bars; automatic mothers reset on new day.
        if mother is not None:
            age = i - mother["index"]
            if age > MAX_BARS or (day_change and MODE == "Auto"):
                mother = None

        # Candidate is the prior candle; current candle must be inside it.
        inside_bar = False
        if i > 0:
            prev = df.iloc[i - 1]
            inside_bar = row["High"] <= prev["High"] and row["Low"] >= prev["Low"]
            prev_range = float(prev["High"] - prev["Low"])
            prev_body = abs(float(prev["Close"] - prev["Open"]))
            prev_atr = df.iloc[i - 1]["ATR"]
            prev_time_ok = (not USE_SESSION) or in_session(df.index[i - 1], MOTHER_START, MOTHER_END)
            candidate_ok = (
                pd.notna(prev_atr) and prev_atr > 0
                and prev_range >= MIN_ATR_MULT * float(prev_atr)
                and prev_range <= MAX_ATR_MULT * float(prev_atr)
                and prev_body >= MIN_BODY_PCT / 100.0 * prev_range
                and prev_time_ok
            )
        else:
            prev = None
            candidate_ok = False

        new_auto = MODE == "Auto" and mother is None and inside_bar and candidate_ok
        new_manual = (
            MODE == "Manual"
            and timestamp >= MANUAL_TIME
            and (i == 0 or df.index[i - 1] < MANUAL_TIME)
        )

        if new_manual:
            mother = {
                "index": i,
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "volume": float(row["Volume"]) if pd.notna(row["Volume"]) else 0.0,
                "inside_count": 0,
                "last_child": np.nan,
            }
        elif new_auto:
            mother = {
                "index": i - 1,
                "high": float(prev["High"]),
                "low": float(prev["Low"]),
                "volume": float(prev["Volume"]) if pd.notna(prev["Volume"]) else 0.0,
                "inside_count": 1,
                "last_child": float(row["High"] - row["Low"]),
            }
        elif mother is not None and i > mother["index"]:
            if row["High"] <= mother["high"] and row["Low"] >= mother["low"]:
                mother["inside_count"] += 1
                mother["last_child"] = float(row["High"] - row["Low"])

        if mother is None:
            # Pending confirmation deliberately survives mother deactivation, as in Pine.
            continue

        mh = mother["high"]
        ml = mother["low"]
        mr = mh - ml
        if mr <= 0:
            continue

        # Pine session filter uses the bar's opening timestamp.
        in_sig = (not USE_SESSION) or in_session(timestamp, SIGNAL_START, SIGNAL_END)
        base = i > mother["index"]
        armed = MODE == "Manual" or mother["inside_count"] >= MIN_INSIDE

        vol_now = float(row["Volume"]) if pd.notna(row["Volume"]) else 0.0
        avg_vol = row["AVG_VOL"]
        has_vol = vol_now > 0
        sc1 = (not has_vol) or (pd.notna(avg_vol) and mother["volume"] >= float(avg_vol))
        sc2 = mother["inside_count"] >= 2
        sc3 = pd.notna(mother["last_child"]) and mother["last_child"] <= 0.6 * mr
        sc4 = (not has_vol) or (pd.notna(avg_vol) and vol_now >= float(avg_vol))
        score = int(sc1) + int(sc2) + int(sc3) + int(sc4)

        candle_range = float(row["High"] - row["Low"])
        body_ok = candle_range > 0 and abs(float(row["Close"] - row["Open"])) >= BREAKOUT_BODY_PCT / 100.0 * candle_range
        size_ok = candle_range <= MAX_BREAKOUT_ATR * float(row["ATR"])
        trend_ce = (not USE_TREND) or float(row["Close"]) > float(row["EMA"])
        trend_pe = (not USE_TREND) or float(row["Close"]) < float(row["EMA"])
        vwap = row["VWAP"]
        vwap_ce = (not USE_VWAP) or pd.isna(vwap) or float(row["Close"]) > float(vwap)
        vwap_pe = (not USE_VWAP) or pd.isna(vwap) or float(row["Close"]) < float(vwap)

        buf = mr * BUFFER_PCT / 100.0
        up_level = mh + buf
        dn_level = ml - buf
        close_up = float(row["Close"]) > up_level
        close_dn = float(row["Close"]) < dn_level
        if FULLY_OUTSIDE:
            brk_up = float(row["Low"]) > up_level
            brk_dn = float(row["High"]) < dn_level
        elif USE_CLOSE:
            brk_up, brk_dn = close_up, close_dn
        else:
            brk_up = float(row["High"]) > up_level
            brk_dn = float(row["Low"]) < dn_level

        common = base and armed and in_sig and body_ok and size_ok and score >= MIN_SCORE
        breakout_up = common and brk_up and not brk_dn and float(row["Close"]) > float(row["Open"]) and trend_ce and vwap_ce
        breakout_dn = common and brk_dn and not brk_up and float(row["Close"]) < float(row["Open"]) and trend_pe and vwap_pe

        # Pine sets pending first, then evaluates confirmation on the same bar.
        # Same-bar confirmation cannot occur because Low > High / High < Low is impossible.
        if breakout_up:
            pending_ce = True
            pending_ce_high = float(row["High"])
        if breakout_dn:
            pending_pe = True
            pending_pe_low = float(row["Low"])

        confirm_ce = pending_ce and float(row["Close"]) > float(row["Open"]) and float(row["Low"]) > pending_ce_high
        confirm_pe = pending_pe and float(row["Close"]) < float(row["Open"]) and float(row["High"]) < pending_pe_low

        if SL_MODE == "Mother Midpoint (50%)":
            sl_ce = sl_pe = (mh + ml) / 2.0
        else:
            sl_ce, sl_pe = ml, mh

        entry = float(row["Close"])
        risk_ce = entry - sl_ce
        risk_pe = sl_pe - entry
        risk_ok_ce = risk_ce > 0 and risk_ce <= MAX_RISK_ATR * float(row["ATR"])
        risk_ok_pe = risk_pe > 0 and risk_pe <= MAX_RISK_ATR * float(row["ATR"])
        ce_signal = confirm_ce and risk_ok_ce
        pe_signal = confirm_pe and risk_ok_pe

        if ce_signal:
            pending_ce = False
        if pe_signal:
            pending_pe = False

        # Pine deactivates the mother at breakout; rejected filter breaks can also
        # deactivate it when keepAlive=false. Pending confirmation survives.
        broke_out = base and (
            breakout_up or breakout_dn or
            ((not KEEP_ALIVE) and (close_up or close_dn))
        )

        if ce_signal or pe_signal:
            is_ce = ce_signal
            side = "BUY CE" if is_ce else "BUY PE"
            sl = sl_ce if is_ce else sl_pe
            risk = abs(entry - sl)
            t1 = entry + risk * RR1 if is_ce else entry - risk * RR1
            t2 = entry + risk * RR2 if is_ce else entry - risk * RR2
            signals.append({
                "symbol": symbol,
                "name": meta["name"],
                "tv_symbol": meta["tv"],
                "side": side,
                "entry": entry,
                "sl": sl,
                "t1": t1,
                "t2": t2,
                "score": score,
                "quality": int(round(score * 10 / 4)),
                "time": timestamp.isoformat(),
                "bar_start": timestamp.isoformat(),
                "bar_close": (timestamp + pd.Timedelta(minutes=30)).isoformat(),
            })
            mother = None
        elif broke_out:
            mother = None

    return signals


def format_signal(signal):
    start = pd.Timestamp(signal["bar_start"]).tz_convert(IST)
    close = pd.Timestamp(signal["bar_close"]).tz_convert(IST)
    tv_link = TRADINGVIEW_BASE + quote(signal["tv_symbol"], safe="")
    icon = "🟢" if signal["side"] == "BUY CE" else "🔴"
    return (
        f"{icon} ROMYO {signal['side']}\n\n"
        f"📊 Stock/Index: {signal['name']}\n"
        f"Symbol: {signal['symbol']}\n"
        f"⏱️ Timeframe: 30 Minutes\n"
        f"📅 Date: {start:%d-%m-%Y}\n"
        f"🕒 Candle start: {start:%I:%M %p} IST\n"
        f"🔔 Candle close: {close:%I:%M %p} IST\n\n"
        f"💰 Underlying entry: {signal['entry']:.2f}\n"
        f"🛑 Stop Loss: {signal['sl']:.2f}\n"
        f"🎯 Target 1: {signal['t1']:.2f}\n"
        f"🎯 Target 2: {signal['t2']:.2f}\n"
        f"⏳ Time Stop: {TIME_STOP} candles\n\n"
        f"⭐ Quality: {signal['quality']}/10\n"
        f"📋 Original Score: Q{signal['score']}/4\n\n"
        f"📈 TradingView Chart:\n{tv_link}\n\n"
        "⚠️ Signal underlying chart par based hai. Option strike/premium calculate nahi kiya gaya."
    )


def main():
    logging.info("ROMYO 30m scanner started | mode=%s", MODE)
    if MODE not in {"Auto", "Manual"}:
        raise RuntimeError("MOTHER_MODE must be Auto or Manual")
    if not TOKEN or not CHAT_ID:
        raise RuntimeError("Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to GitHub Secrets.")
    if TEST_MODE:
        send_test()

    state = load_state()
    sent = set(state.get("sent", []))
    now = pd.Timestamp.now(tz=IST)
    total_data = total_signals = errors = 0

    for symbol, meta in SYMBOLS.items():
        logging.info("Scanning %s", symbol)
        df = download_data(symbol)
        if df is None:
            errors += 1
            continue
        total_data += 1
        try:
            for signal in scan_symbol(symbol, meta, df):
                candle_time = pd.Timestamp(signal["time"]).tz_convert(IST)
                age_seconds = (now - candle_time).total_seconds()
                if age_seconds < 0 or age_seconds > MAX_SIGNAL_AGE_HOURS * 3600:
                    continue

                key = f"{signal['symbol']}|{signal['time']}|{signal['side']}"
                if key in sent:
                    continue
                if telegram(format_signal(signal)):
                    sent.add(key)
                    total_signals += 1
                    state["sent"] = sorted(sent)[-5000:]
                    save_state(state)
                    logging.info("SIGNAL SENT: %s %s Q%d/4", symbol, signal["side"], signal["score"])
                else:
                    logging.error("Could not send signal: %s", key)
        except Exception:
            errors += 1
            logging.exception("Scan failed for %s", symbol)
        time.sleep(0.2)

    state["sent"] = sorted(sent)[-5000:]
    save_state(state)
    logging.info("FINISHED: data_symbols=%d, new_signals=%d, errors=%d", total_data, total_signals, errors)


if __name__ == "__main__":
    main()
