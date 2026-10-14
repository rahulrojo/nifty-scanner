import os
import json
import time
import logging
from pathlib import Path
from datetime import timedelta
from zoneinfo import ZoneInfo
from urllib.parse import quote

import requests
import pandas as pd
import numpy as np
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID", "1003921675472"
).strip()

TEST_MODE = os.getenv("SEND_TEST_MESSAGE", "false").lower() == "true"

INTERVAL = "30m"
PERIOD = "60d"

EMA_LEN = 21
ATR_LEN = 14

MIN_ATR = 1.0
MAX_ATR = 2.5
MIN_BODY = 50

MIN_INSIDE = 1
MAX_MOTHER_AGE = 8

BREAKOUT_BODY = 40
MAX_BREAKOUT_ATR = 2.0
MAX_RISK_ATR = 2.0

MIN_QUALITY = 6
RR1 = 1.0
RR2 = 2.0

STATE_FILE = Path("signal_state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Large-cap NSE watchlist. Review constituents periodically.
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


def telegram(message):
    if not TOKEN or not CHAT_ID:
        logging.error("Telegram token/chat ID missing")
        return False

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    try:
        r = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message,
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
        r.raise_for_status()

        result = r.json()
        if not result.get("ok"):
            logging.error("Telegram API: %s", result)
            return False

        return True

    except Exception:
        logging.exception("Telegram message failed")
        return False


def load_state():
    try:
        if STATE_FILE.exists():
            return json.loads(
                STATE_FILE.read_text(encoding="utf-8")
            )
    except Exception:
        logging.exception("Cannot load state")

    return {"sent": []}


def save_state(state):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )
    tmp.replace(STATE_FILE)


def send_test():
    now = pd.Timestamp.now(tz=IST)

    msg = (
        "🧪 ROMYO SCANNER TEST MESSAGE\n\n"
        "✅ GitHub Actions started successfully.\n"
        "✅ Telegram Bot connection is working.\n\n"
        f"📅 Date: {now.strftime('%d-%m-%Y')}\n"
        f"🕒 Time: {now.strftime('%I:%M:%S %p')} IST\n"
        "⏱️ Scanner timeframe: 30 minutes\n\n"
        "⚠️ This is a TEST, not a trading signal."
    )

    if telegram(msg):
        logging.info("Test message sent")
    else:
        raise RuntimeError("Telegram test message failed")


def download_data(symbol):
    try:
        df = yf.download(
            symbol,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            logging.warning("%s: no Yahoo data", symbol)
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        needed = ["Open", "High", "Low", "Close", "Volume"]

        if not all(c in df.columns for c in needed):
            logging.warning("%s: OHLCV missing", symbol)
            return None

        df = df[needed].copy()
        idx = pd.DatetimeIndex(df.index)

        if idx.tz is None:
            idx = idx.tz_localize("UTC")

        df.index = idx.tz_convert(IST)
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        for c in needed:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

        # Remove currently forming 30-minute candle.
        now = pd.Timestamp.now(tz=IST)
        df = df[df.index + pd.Timedelta(minutes=30) <= now]

        if len(df) < 50:
            logging.warning("%s: insufficient candles", symbol)
            return None

        # Indicators
        df["EMA"] = df["Close"].ewm(
            span=EMA_LEN, adjust=False
        ).mean()

        prev_close = df["Close"].shift(1)

        tr = pd.concat(
            [
                df["High"] - df["Low"],
                (df["High"] - prev_close).abs(),
                (df["Low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        df["ATR"] = tr.rolling(ATR_LEN).mean()

        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        vol = df["Volume"].fillna(0)
        days = pd.Series(df.index.date, index=df.index)

        pv_sum = (typical * vol).groupby(days).cumsum()
        vol_sum = vol.groupby(days).cumsum()

        df["VWAP"] = pv_sum / vol_sum.replace(0, np.nan)

        df["AVG_VOL"] = vol.rolling(20).mean()

        return df

    except Exception:
        logging.exception("Download failed: %s", symbol)
        return None


def score_signal(df, mother_idx, breakout_idx, confirm_idx, side):
    mother = df.iloc[mother_idx]
    breakout = df.iloc[breakout_idx]
    confirm = df.iloc[confirm_idx]

    score = 0

    # 1. Mother candle body quality
    mr = mother["High"] - mother["Low"]
    if mr > 0 and abs(mother["Close"] - mother["Open"]) / mr >= 0.60:
        score += 1

    # 2. Trend confirmation
    if side == "BUY CE" and confirm["Close"] > confirm["EMA"]:
        score += 1
    if side == "BUY PE" and confirm["Close"] < confirm["EMA"]:
        score += 1

    # 3. VWAP confirmation
    if pd.notna(confirm["VWAP"]):
        if side == "BUY CE" and confirm["Close"] > confirm["VWAP"]:
            score += 1
        if side == "BUY PE" and confirm["Close"] < confirm["VWAP"]:
            score += 1

    # 4. Breakout body strength
    br = breakout["High"] - breakout["Low"]
    if br > 0 and abs(breakout["Close"] - breakout["Open"]) / br >= 0.60:
        score += 1

    # 5. Confirmation candle body strength
    cr = confirm["High"] - confirm["Low"]
    if cr > 0 and abs(confirm["Close"] - confirm["Open"]) / cr >= 0.60:
        score += 1

    # 6. Volume confirmation
    if (
        pd.notna(confirm["AVG_VOL"])
        and confirm["AVG_VOL"] > 0
        and confirm["Volume"] >= confirm["AVG_VOL"]
    ):
        score += 1

    # 7. Clean separation from breakout candle
    if side == "BUY CE" and confirm["Low"] > breakout["High"]:
        score += 1
    if side == "BUY PE" and confirm["High"] < breakout["Low"]:
        score += 1

    # 8. Breakout close outside mother
    if side == "BUY CE" and breakout["Close"] > mother["High"]:
        score += 1
    if side == "BUY PE" and breakout["Close"] < mother["Low"]:
        score += 1

    # 9. Mother candle has controlled size
    if pd.notna(mother["ATR"]) and mother["ATR"] > 0:
        if mr <= MAX_ATR * mother["ATR"]:
            score += 1

    # 10. Controlled risk
    entry = float(confirm["Close"])
    stop = float((mother["High"] + mother["Low"]) / 2)
    risk = entry - stop if side == "BUY CE" else stop - entry

    if (
        pd.notna(confirm["ATR"])
        and confirm["ATR"] > 0
        and 0 < risk <= MAX_RISK_ATR * confirm["ATR"]
    ):
        score += 1

    return score


def scan_symbol(symbol, name, df):
    signals = []

    if df is None or len(df) < 50:
        return signals

    # The last index is included: this fixes the old off-by-one bug.
    for confirm_idx in range(30, len(df)):
        confirm = df.iloc[confirm_idx]

        if pd.isna(confirm["ATR"]) or confirm["ATR"] <= 0:
            continue

        # Confirmation must immediately follow breakout.
        breakout_idx = confirm_idx - 1
        breakout = df.iloc[breakout_idx]

        # Search for a valid mother candle behind the breakout.
        # Only completed inside candles may be between them.
        earliest = max(
            0,
            breakout_idx - MAX_MOTHER_AGE - 1
        )

        for mother_idx in range(earliest, breakout_idx - MIN_INSIDE):
            mother = df.iloc[mother_idx]

            if pd.isna(mother["ATR"]) or mother["ATR"] <= 0:
                continue

            mh = float(mother["High"])
            ml = float(mother["Low"])
            mr = mh - ml

            if mr <= 0:
                continue

            mbody = abs(float(mother["Close"] - mother["Open"]))

            if not (
                MIN_ATR * float(mother["ATR"])
                <= mr
                <= MAX_ATR * float(mother["ATR"])
            ):
                continue

            if mbody < MIN_BODY / 100 * mr:
                continue

            # Every candle between mother and breakout must
            # be inside the mother candle.
            inside = df.iloc[mother_idx + 1:breakout_idx]

            if len(inside) < MIN_INSIDE:
                continue

            if not (
                (inside["High"] <= mh)
                & (inside["Low"] >= ml)
            ).all():
                continue

            if len(inside) > MAX_MOTHER_AGE:
                continue

            br = float(breakout["High"] - breakout["Low"])

            if br <= 0:
                continue

            bbody = abs(float(breakout["Close"] - breakout["Open"]))

            if bbody < BREAKOUT_BODY / 100 * br:
                continue

            if br > MAX_BREAKOUT_ATR * float(confirm["ATR"]):
                continue

            side = None

            # CE: bullish breakout and next candle completely
            # above the breakout candle.
            if (
                breakout["Close"] > mh
                and breakout["Close"] > breakout["Open"]
                and confirm["Close"] > confirm["Open"]
                and confirm["Low"] > breakout["High"]
                and confirm["Close"] > confirm["EMA"]
                and (
                    pd.isna(confirm["VWAP"])
                    or confirm["Close"] > confirm["VWAP"]
                )
            ):
                side = "BUY CE"

            # PE: bearish breakout and next candle completely
            # below the breakout candle.
            elif (
                breakout["Close"] < ml
                and breakout["Close"] < breakout["Open"]
                and confirm["Close"] < confirm["Open"]
                and confirm["High"] < breakout["Low"]
                and confirm["Close"] < confirm["EMA"]
                and (
                    pd.isna(confirm["VWAP"])
                    or confirm["Close"] < confirm["VWAP"]
                )
            ):
                side = "BUY PE"

            if side is None:
                continue

            entry = float(confirm["Close"])
            stop = (mh + ml) / 2

            risk = entry - stop if side == "BUY CE" else stop - entry

            if not (0 < risk <= MAX_RISK_ATR * float(confirm["ATR"])):
                continue

            quality = score_signal(
                df, mother_idx, breakout_idx, confirm_idx, side
            )

            if quality < MIN_QUALITY:
                continue

            target1 = (
                entry + RR1 * risk
                if side == "BUY CE"
                else entry - RR1 * risk
            )

            target2 = (
                entry + RR2 * risk
                if side == "BUY CE"
                else entry - RR2 * risk
            )

            signals.append({
                "symbol": symbol,
                "name": name,
                "side": side,
                "entry": entry,
                "sl": stop,
                "t1": target1,
                "t2": target2,
                "quality": quality,
                "time": df.index[confirm_idx].isoformat(),
            })

            # One mother candle / setup per confirmation candle.
            break

    return signals


def format_signal(s):
    candle_time = pd.Timestamp(s["time"]).tz_convert(IST)
    tv_symbol = quote(s["symbol"], safe="")
    tv_link = f"https://www.tradingview.com/chart/?symbol={tv_symbol}"

    icon = "🟢" if s["side"] == "BUY CE" else "🔴"

    return (
        f"{icon} ROMYO {s['side']}\n\n"
        f"📊 Stock/Index: {s['name']}\n"
        f"Symbol: {s['symbol']}\n"
        f"⏱️ Timeframe: 30 Minutes\n"
        f"📅 Date: {candle_time.strftime('%d-%m-%Y')}\n"
        f"🕒 Candle time: {candle_time.strftime('%I:%M %p')} IST\n\n"
        f"💰 Entry: {s['entry']:.2f}\n"
        f"🛑 Stop Loss: {s['sl']:.2f}\n"
        f"🎯 Target 1: {s['t1']:.2f}\n"
        f"🎯 Target 2: {s['t2']:.2f}\n\n"
        f"⭐ Quality: {s['quality']}/10\n"
        f"📦 Mother Candle Breakout\n"
        f"✅ Confirmation passed\n\n"
        f"📈 TradingView:\n{tv_link}\n\n"
        "Note: CE/PE is a directional signal on the underlying. "
        "Option premium and strike are not calculated."
    )


def main():
    logging.info("ROMYO scanner started")

    if not TOKEN or not CHAT_ID:
        raise RuntimeError(
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in GitHub Secrets"
        )

    # A manual Run workflow sends a test message if enabled.
    if TEST_MODE:
        send_test()

    state = load_state()
    sent = set(state.get("sent", []))

    now = pd.Timestamp.now(tz=IST)
    new_count = 0
    data_count = 0

    for symbol, name in SYMBOLS.items():
        logging.info("Scanning %s", symbol)

        df = download_data(symbol)

        if df is None:
            time.sleep(0.3)
            continue

        data_count += 1

        try:
            signals = scan_symbol(symbol, name, df)

            for s in signals:
                candle_time = pd.Timestamp(s["time"])
                age = (
                    now - candle_time - pd.Timedelta(minutes=30)
                ).total_seconds() / 60

                # Only recent completed confirmations are sent.
                if age < -5 or age > 90:
                    continue

                key = f"{symbol}|{s['time']}|{s['side']}"

                if key in sent:
                    continue

                if telegram(format_signal(s)):
                    sent.add(key)
                    new_count += 1

                    state["sent"] = sorted(sent)[-5000:]
                    save_state(state)

                    logging.info(
                        "SIGNAL SENT: %s %s quality=%s",
                        symbol, s["side"], s["quality"]
                    )

        except Exception:
            logging.exception("Scan error: %s", symbol)

        time.sleep(0.3)

    state["sent"] = sorted(sent)[-5000:]
    save_state(state)

    logging.info(
        "Finished. Data symbols=%d, new signals=%d",
        data_count, new_count
    )

    if data_count == 0:
        raise RuntimeError(
            "No symbols returned usable Yahoo data. "
            "Check Actions logs and Yahoo Finance availability."
        )


if __name__ == "__main__":
    main()
