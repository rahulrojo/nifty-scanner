import os
import json
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ==========================================
# ROMYO MOTHER CANDLE BREAKOUT SCANNER
# TIMEFRAME: 30 MINUTES
# ==========================================

IST = ZoneInfo("Asia/Kolkata")

INTERVAL = "30m"
PERIOD = "60d"

EMA_LEN = 21
ATR_LEN = 14

MIN_MOTHER_ATR = 1.0
MAX_MOTHER_ATR = 2.5
MIN_BODY_PCT = 50

MIN_INSIDE = 1
MAX_WAIT_BARS = 8

BREAK_BODY_PCT = 40
MAX_BREAK_ATR = 2.0

MIN_SCORE = 2
MAX_RISK_ATR = 2.0

RR1 = 1.0
RR2 = 2.0

TIME_STOP = 6

STATE_FILE = "signal_state.json"

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not CHAT_ID:
    CHAT_ID = "1003921675472"

# ==========================================
# STOCK UNIVERSE
# ==========================================

STOCKS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK",
    "SBIN", "AXISBANK", "KOTAKBANK",
    "INDUSINDBK", "BANKBARODA", "PNB",
    "CANBK", "UNIONBANK", "IDFCFIRSTB",
    "FEDERALBNK", "AUBANK",

    "BAJFINANCE", "BAJAJFINSV", "SHRIRAMFIN",
    "CHOLAFIN", "MUTHOOTFIN", "RECLTD",
    "PFC", "IRFC", "HDFCLIFE", "SBILIFE",
    "ICICIPRULI", "LICI",

    "INFY", "TCS", "HCLTECH", "WIPRO",
    "TECHM", "LTIM", "PERSISTENT",
    "COFORGE", "MPHASIS",

    "LT", "SIEMENS", "ABB", "BEL",
    "HAL", "BHEL", "RVNL", "IRCTC",

    "ONGC", "OIL", "IOC", "BPCL",
    "HINDPETRO", "GAIL", "NTPC",
    "POWERGRID", "TATAPOWER", "ADANIPOWER",
    "ADANIENT", "ADANIPORTS",

    "JSWSTEEL", "TATASTEEL", "HINDALCO",
    "VEDL", "SAIL", "COALINDIA",
    "NMDC", "JINDALSTEL",

    "TATAMOTORS", "M&M", "MARUTI",
    "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO",
    "TVSMOTOR", "ASHOKLEY",

    "APOLLOHOSP", "SUNPHARMA", "DRREDDY",
    "CIPLA", "DIVISLAB", "ZYDUSLIFE",
    "LUPIN", "AUROPHARMA", "ALKEM",
    "MAXHEALTH",

    "HDFCAMC", "CDSL", "BSE", "MCX",

    "DLF", "GODREJPROP", "OBEROIRLTY",
    "LODHA", "PHOENIX",

    "ULTRACEMCO", "GRASIM", "AMBUJACEM",
    "SHREECEM",

    "ASIANPAINT", "TITAN", "TRENT",
    "ETERNAL", "ITC", "HINDUNILVR",
    "NESTLEIND", "BRITANNIA", "TATACONSUM",

    "PIDILITIND", "DABUR", "GODREJCP",
    "COLPAL", "MARICO", "BERGEPAINT",

    "INDIGO", "CONCOR", "DELHIVERY"
]

STOCKS = list(dict.fromkeys(STOCKS))

INDEXES = [
    "^NSEI",
    "^NSEBANK",
    "^CNXIT",
    "^CNXPHARMA",
    "^CNXAUTO",
    "^CNXMETAL",
    "^CNXFMCG",
    "^CNXENERGY",
    "^CNXREALTY",
    "^CNXPSUBANK",
    "^CNXINFRA",
    "^CNXMEDIA"
]

SYMBOLS = STOCKS + INDEXES

# ==========================================
# TELEGRAM
# ==========================================

def send_telegram(message):

    if not BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN missing")
        return False

    url = (
        f"https://api.telegram.org/"
        f"bot{BOT_TOKEN}/sendMessage"
    )

    try:
        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            },
            timeout=20
        )

        print("Telegram:", response.text)

        return response.ok

    except Exception as e:
        print("Telegram error:", e)
        return False


# ==========================================
# STATE MANAGEMENT
# ==========================================

def load_state():

    if os.path.exists(STATE_FILE):

        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)

        except Exception:
            pass

    return {}


def save_state(state):

    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ==========================================
# INDICATORS
# ==========================================

def calculate_indicators(df):

    df = df.copy()

    df["EMA"] = df["Close"].ewm(
        span=EMA_LEN,
        adjust=False
    ).mean()

    previous_close = df["Close"].shift(1)

    tr = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - previous_close),
        abs(df["Low"] - previous_close)
    ], axis=1).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1 / ATR_LEN,
        adjust=False
    ).mean()

    typical = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    date_group = pd.Series(
        df.index.date,
        index=df.index
    )

    pv = typical * df["Volume"]

    df["VWAP"] = (
        pv.groupby(date_group).cumsum() /
        df["Volume"].groupby(date_group).cumsum()
    )

    df["AVG_VOL"] = df["Volume"].rolling(20).mean()

    return df


# ==========================================
# QUALITY SCORE OUT OF 10
# ==========================================

def quality_score(
    mother_vol,
    avg_vol,
    inside_count,
    last_child,
    mother_range,
    volume,
    body_pct,
    atr,
    candle_range
):

    score = 0

    # Mother volume quality: 2 points
    if mother_vol >= avg_vol:
        score += 2

    # Inside candle compression: 2 points
    if inside_count >= 2:
        score += 2

    # Child candle contraction: 2 points
    if (
        last_child is not None
        and last_child <= 0.6 * mother_range
    ):
        score += 2

    # Breakout volume: 2 points
    if volume >= avg_vol:
        score += 2

    # Breakout strength: 2 points
    if body_pct >= 60:
        score += 2
    elif body_pct >= 50:
        score += 1

    return min(score, 10)


# ==========================================
# SESSION FILTER
# ==========================================

def in_session(timestamp, start, end):

    t = timestamp.hour * 60 + timestamp.minute

    return start <= t <= end


# ==========================================
# SCANNER LOGIC
# ==========================================

def scan_symbol(symbol, state):

    ticker = symbol if symbol.startswith("^") else symbol + ".NS"

    try:

        df = yf.download(
            ticker,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df is None or len(df) < 50:
            return

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[
            ["Open", "High", "Low", "Close", "Volume"]
        ].copy()

        df = df.dropna()

        if df.empty:
            return

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        df.index = df.index.tz_convert(IST)

        df = calculate_indicators(df)

        # Incomplete candle remove
        now = datetime.now(IST)

        completed = (
            df.index + pd.Timedelta(minutes=30)
            <= now - pd.Timedelta(minutes=3)
        )

        df = df.loc[completed]

        if len(df) < 30:
            return

        # Initial scan: recent trading session only
        if symbol not in state:

            cutoff = now - timedelta(days=2)

        else:

            cutoff = pd.Timestamp(
                state[symbol]
            ).to_pydatetime()

        active = False

        m_high = None
        m_low = None
        m_vol = None
        m_bar = None

        inside_count = 0
        last_child = None

        for i in range(1, len(df)):

            row = df.iloc[i]
            prev = df.iloc[i - 1]

            ts = df.index[i]

            if ts.to_pydatetime() <= cutoff:
                continue

            # Session
            mother_session = in_session(
                df.index[i - 1],
                9 * 60 + 15,
                13 * 60 + 30
            )

            signal_session = in_session(
                ts,
                9 * 60 + 15,
                14 * 60 + 30
            )

            # =================================
            # EXPIRE MOTHER CANDLE
            # =================================

            if active:

                if i - m_bar > MAX_WAIT_BARS:

                    active = False

            # =================================
            # DETECT MOTHER + INSIDE BAR
            # =================================

            inside = (
                row["High"] <= prev["High"]
                and row["Low"] >= prev["Low"]
            )

            prev_range = prev["High"] - prev["Low"]

            prev_body = abs(
                prev["Close"] - prev["Open"]
            )

            prev_atr = prev["ATR"]

            mother_ok = (
                pd.notna(prev_atr)
                and prev_range >= MIN_MOTHER_ATR * prev_atr
                and prev_range <= MAX_MOTHER_ATR * prev_atr
                and prev_body >= MIN_BODY_PCT / 100 * prev_range
                and mother_session
            )

            if not active and inside and mother_ok:

                active = True

                m_high = float(prev["High"])
                m_low = float(prev["Low"])
                m_vol = float(prev["Volume"])

                m_bar = i - 1

                inside_count = 1

                last_child = float(
                    row["High"] - row["Low"]
                )

            elif active:

                if (
                    row["High"] <= m_high
                    and row["Low"] >= m_low
                ):

                    inside_count += 1

                    last_child = float(
                        row["High"] - row["Low"]
                    )

            if not active:
                continue

            # =================================
            # BREAKOUT CONDITIONS
            # =================================

            mother_range = m_high - m_low

            up_level = m_high
            down_level = m_low

            close_up = row["Close"] > up_level
            close_down = row["Close"] < down_level

            candle_range = row["High"] - row["Low"]

            if candle_range <= 0:
                continue

            body = abs(
                row["Close"] - row["Open"]
            )

            body_pct = body / candle_range * 100

            body_ok = body_pct >= BREAK_BODY_PCT

            size_ok = (
                candle_range <= MAX_BREAK_ATR * row["ATR"]
            )

            armed = inside_count >= MIN_INSIDE

            # Quality components
            avg_vol = row["AVG_VOL"]

            if pd.isna(avg_vol):
                continue

            q = quality_score(
                m_vol,
                avg_vol,
                inside_count,
                last_child,
                mother_range,
                row["Volume"],
                body_pct,
                row["ATR"],
                candle_range
            )

            # Original Pine score out of 4
            original_score = (
                int(m_vol >= avg_vol) +
                int(inside_count >= 2) +
                int(
                    last_child is not None
                    and last_child <= 0.6 * mother_range
                ) +
                int(row["Volume"] >= avg_vol)
            )

            common = (
                signal_session
                and armed
                and body_ok
                and size_ok
                and original_score >= MIN_SCORE
            )

            # =================================
            # EMA / VWAP
            # =================================

            ema = row["EMA"]
            vwap = row["VWAP"]

            trend_ce = row["Close"] > ema
            trend_pe = row["Close"] < ema

            vwap_ce = (
                pd.isna(vwap)
                or row["Close"] > vwap
            )

            vwap_pe = (
                pd.isna(vwap)
                or row["Close"] < vwap
            )

            # =================================
            # STOPLOSS AND RISK
            # =================================

            sl_ce = (m_high + m_low) / 2
            sl_pe = (m_high + m_low) / 2

            risk_ce = row["Close"] - sl_ce
            risk_pe = sl_pe - row["Close"]

            risk_ok_ce = (
                risk_ce > 0
                and risk_ce <= MAX_RISK_ATR * row["ATR"]
            )

            risk_ok_pe = (
                risk_pe > 0
                and risk_pe <= MAX_RISK_ATR * row["ATR"]
            )

            # =================================
            # FINAL SIGNAL
            # =================================

            ce_signal = (
                common
                and close_up
                and not close_down
                and row["Close"] > row["Open"]
                and trend_ce
                and vwap_ce
                and risk_ok_ce
            )

            pe_signal = (
                common
                and close_down
                and not close_up
                and row["Close"] < row["Open"]
                and trend_pe
                and vwap_pe
                and risk_ok_pe
            )

            if not (ce_signal or pe_signal):
                continue

            # =================================
            # SIGNAL DETAILS
            # =================================

            side = "BUY CE" if ce_signal else "BUY PE"

            entry = float(row["Close"])

            sl = float(
                sl_ce if ce_signal else sl_pe
            )

            risk = abs(entry - sl)

            if ce_signal:

                target1 = entry + risk * RR1
                target2 = entry + risk * RR2

            else:

                target1 = entry - risk * RR1
                target2 = entry - risk * RR2

            candle_start = ts.strftime(
                "%d-%m-%Y %I:%M %p"
            )

            candle_end = (
                ts + timedelta(minutes=30)
            ).strftime("%I:%M %p")

            name = symbol.replace("^", "")

            # =================================
            # TELEGRAM MESSAGE
            # =================================

            message = f"""
<b>🚀 ROMYO MOTHER CANDLE SIGNAL</b>

<b>{side}</b>

━━━━━━━━━━━━━━━━━━

📊 Symbol: <b>{name}</b>

⏱ Timeframe: 30 MIN

📅 Candle Date: {ts.strftime('%d-%m-%Y')}

🕒 Candle Time:
{candle_start} - {candle_end} IST

━━━━━━━━━━━━━━━━━━

🎯 Entry: {entry:.2f}

🛑 Stoploss: {sl:.2f}

✅ Target 1: {target1:.2f}

🎯 Target 2: {target2:.2f}

━━━━━━━━━━━━━━━━━━

⭐ Signal Quality: <b>{q}/10</b>

📈 Inside Candles: {inside_count}

📊 Breakout Body: {body_pct:.1f}%

📦 Mother Range: {mother_range:.2f}

📉 ATR: {row['ATR']:.2f}

━━━━━━━━━━━━━━━━━━

⏳ Time Stop: {TIME_STOP} candles

⚠️ Signal is based on completed 30m candle.

<b>ROMYO SIGNALS</b>
"""

            if send_telegram(message):

                print(
                    f"SIGNAL SENT: {symbol} {side} {ts}"
                )

                state[symbol] = ts.isoformat()

                save_state(state)

                # Mother candle consumed after signal
                active = False

                return

        # No signal: save latest processed candle
        if symbol not in state:

            state[symbol] = df.index[-1].isoformat()

            save_state(state)

    except Exception as e:

        print(
            f"ERROR {symbol}: {str(e)}"
        )


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 50)

    print("ROMYO 30 MIN MOTHER CANDLE SCANNER")

    print("Started:", datetime.now(IST))

    print("Total Symbols:", len(SYMBOLS))

    print("=" * 50)

    state = load_state()

    for i, symbol in enumerate(SYMBOLS, 1):

        print(
            f"[{i}/{len(SYMBOLS)}] Scanning {symbol}"
        )

        scan_symbol(symbol, state)

        time.sleep(0.5)

    save_state(state)

    print("=" * 50)

    print("SCAN COMPLETED")

    print("Finished:", datetime.now(IST))

    print("=" * 50)


if __name__ == "__main__":

    main()
