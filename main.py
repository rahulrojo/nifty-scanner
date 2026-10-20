import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import yfinance as yf


# =========================================================
# ROMYO 30-MINUTE CONFIRMED MOTHER CANDLE SCANNER
# BUY CE / BUY PE
# First run: signals from previous trading session only
# Later runs: newly confirmed signals only
# =========================================================

IST = ZoneInfo("Asia/Kolkata")

INTERVAL = "30m"
PERIOD = "60d"

EMA_LEN = 21
ATR_LEN = 14
VOL_LEN = 20

MIN_MOTHER_ATR = 1.0
MAX_MOTHER_ATR = 2.5
MIN_BODY_PCT = 50.0

VOL_MULT = 1.0
MAX_WAIT_BARS = 8

STATE_FILE = Path("signal_state.json")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =========================================================
# 100 NSE F&O STOCKS
# Symbols use Yahoo Finance format
# =========================================================

STOCKS = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "INDUSINDBK.NS",
    "BANKBARODA.NS",
    "PNB.NS",
    "CANBK.NS",
    "UNIONBANK.NS",
    "IDFCFIRSTB.NS",
    "FEDERALBNK.NS",
    "AUBANK.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "SHRIRAMFIN.NS",
    "CHOLAFIN.NS",
    "MUTHOOTFIN.NS",
    "RECLTD.NS",
    "PFC.NS",
    "IRFC.NS",
    "HDFCLIFE.NS",
    "SBILIFE.NS",
    "ICICIPRULI.NS",
    "LICI.NS",
    "INFY.NS",
    "TCS.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "TECHM.NS",
    "LTIM.NS",
    "PERSISTENT.NS",
    "COFORGE.NS",
    "MPHASIS.NS",
    "LT.NS",
    "SIEMENS.NS",
    "ABB.NS",
    "BEL.NS",
    "HAL.NS",
    "BHEL.NS",
    "RVNL.NS",
    "IRCTC.NS",
    "RECLTD.NS",
    "ONGC.NS",
    "OIL.NS",
    "IOC.NS",
    "BPCL.NS",
    "HINDPETRO.NS",
    "GAIL.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "TATAPOWER.NS",
    "ADANIPOWER.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "JSWSTEEL.NS",
    "TATASTEEL.NS",
    "HINDALCO.NS",
    "VEDL.NS",
    "SAIL.NS",
    "COALINDIA.NS",
    "NMDC.NS",
    "JINDALSTEL.NS",
    "TATAMOTORS.NS",
    "M&M.NS",
    "MARUTI.NS",
    "HEROMOTOCO.NS",
    "BAJAJ-AUTO.NS",
    "EICHERMOT.NS",
    "TVSMOTOR.NS",
    "ASHOKLEY.NS",
    "BHARTIARTL.NS",
    "IDEA.NS",
    "TATACOMM.NS",
    "INDIGO.NS",
    "SUNPHARMA.NS",
    "DRREDDY.NS",
    "CIPLA.NS",
    "DIVISLAB.NS",
    "APOLLOHOSP.NS",
    "MAXHEALTH.NS",
    "ZYDUSLIFE.NS",
    "LUPIN.NS",
    "AUROPHARMA.NS",
    "TORNTPHARM.NS",
    "ITC.NS",
    "HINDUNILVR.NS",
    "NESTLEIND.NS",
    "BRITANNIA.NS",
    "TATACONSUM.NS",
    "ASIANPAINT.NS",
    "TITAN.NS",
    "TRENT.NS",
    "DMART.NS",
    "ULTRACEMCO.NS",
    "GRASIM.NS",
    "AMBUJACEM.NS",
    "DLF.NS",
    "GODREJPROP.NS",
]

# Remove accidental duplicates while preserving order.
STOCKS = list(dict.fromkeys(STOCKS))

# Major NSE indices supported by Yahoo Finance.
# Index availability can vary by data source.
INDEXES = [
    "^NSEI",          # Nifty 50
    "^NSEBANK",       # Bank Nifty
    "^CNXIT",         # Nifty IT
    "^CNXPHARMA",     # Nifty Pharma
    "^CNXAUTO",       # Nifty Auto
    "^CNXMETAL",      # Nifty Metal
    "^CNXFMCG",       # Nifty FMCG
    "^CNXENERGY",     # Nifty Energy
    "^CNXREALTY",     # Nifty Realty
    "^CNXPSUBANK",    # Nifty PSU Bank
    "^CNXINFRA",      # Nifty Infrastructure
    "^CNXMEDIA",      # Nifty Media
]

SYMBOLS = list(dict.fromkeys(STOCKS + INDEXES))


# =========================================================
# TIME / DATA HELPERS
# =========================================================

def now_ist():
    return datetime.now(IST)


def normalize_dataframe(df):
    """Normalize Yahoo Finance dataframe columns."""
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.loc[:, ~df.columns.duplicated()]
    required = ["Open", "High", "Low", "Close", "Volume"]

    if any(col not in df.columns for col in required):
        return pd.DataFrame()

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # Yahoo timestamps normally represent the beginning of a candle.
    idx = pd.to_datetime(df.index)

    if idx.tz is None:
        # If the provider supplies naive timestamps, interpret them
        # in the exchange timezone rather than silently treating
        # them as UTC.
        idx = idx.tz_localize(IST)
    else:
        idx = idx.tz_convert(IST)

    df.index = idx
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # Never evaluate a candle that has not had 30 minutes to close.
    cutoff = pd.Timestamp(now_ist())
    df = df[(df.index + pd.Timedelta(minutes=30)) <= cutoff]

    return df


def download_batch(symbols):
    """Download a batch with limited retries."""
    for attempt in range(3):
        try:
            raw = yf.download(
                tickers=symbols,
                period=PERIOD,
                interval=INTERVAL,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
                timeout=25
            )

            if raw is None or raw.empty:
                raise ValueError("Yahoo returned empty data")

            result = {}

            if isinstance(raw.columns, pd.MultiIndex):
                first_level = set(raw.columns.get_level_values(0))

                for symbol in symbols:
                    if symbol not in first_level:
                        continue

                    part = raw[symbol].copy()
                    part = normalize_dataframe(part)

                    if not part.empty:
                        result[symbol] = part

            elif len(symbols) == 1:
                part = normalize_dataframe(raw)

                if not part.empty:
                    result[symbols[0]] = part

            return result

        except Exception as exc:
            logging.warning(
                "Download attempt %s failed for batch: %s",
                attempt + 1,
                exc
            )
            time.sleep(2 * (attempt + 1))

    return {}


def download_all():
    """
    Use batches to reduce the chance of Yahoo Finance throttling
    a request for more than 100 instruments.
    """
    all_data = {}

    batch_size = 15

    for start in range(0, len(SYMBOLS), batch_size):
        batch = SYMBOLS[start:start + batch_size]

        logging.info(
            "Downloading batch %s/%s",
            start // batch_size + 1,
            (len(SYMBOLS) + batch_size - 1) // batch_size
        )

        data = download_batch(batch)
        all_data.update(data)

        # Small pause to reduce request pressure.
        time.sleep(1)

    return all_data


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df):
    df = df.copy()

    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    # Pine ta.atr() uses Wilder's RMA.
    df["ATR"] = true_range.ewm(
        alpha=1 / ATR_LEN,
        min_periods=ATR_LEN,
        adjust=False
    ).mean()

    df["EMA"] = df["Close"].ewm(
        span=EMA_LEN,
        min_periods=EMA_LEN,
        adjust=False
    ).mean()

    df["VOL_MA"] = df["Volume"].rolling(
        VOL_LEN,
        min_periods=VOL_LEN
    ).mean()

    return df


# =========================================================
# CONFIRMED MOTHER CANDLE SIGNAL LOGIC
# Matches the TradingView rules as closely as practical.
# =========================================================

def find_signals(original_df):
    if original_df.empty or len(original_df) < 60:
        return []

    df = add_indicators(original_df)

    signals = []

    mother_high = None
    mother_low = None
    mother_bar = None

    waiting_breakout = False
    waiting_confirmation = False

    breakout_dir = 0
    breakout_bar = None
    breakout_level = None
    breakout_sl = None

    for i in range(len(df)):
        row = df.iloc[i]

        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        open_price = float(row["Open"])
        volume = float(row["Volume"])

        atr = row["ATR"]
        ema = row["EMA"]
        vol_ma = row["VOL_MA"]

        timestamp = df.index[i]

        if pd.isna(atr) or pd.isna(ema) or pd.isna(vol_ma):
            continue

        candle_range = high - low

        if candle_range <= 0:
            continue

        body_size = abs(close - open_price)
        body_pct = body_size / candle_range * 100.0

        mother_candidate = (
            candle_range >= float(atr) * MIN_MOTHER_ATR
            and candle_range <= float(atr) * MAX_MOTHER_ATR
            and body_pct >= MIN_BODY_PCT
        )

        # Expire an old setup.
        if (
            waiting_breakout
            and mother_bar is not None
            and i - mother_bar > MAX_WAIT_BARS
        ):
            waiting_breakout = False
            waiting_confirmation = False
            breakout_dir = 0

        # Create a mother candle only when no setup is active.
        if (
            mother_candidate
            and not waiting_breakout
            and not waiting_confirmation
        ):
            mother_high = high
            mother_low = low
            mother_bar = i
            waiting_breakout = True

        # Breakout candle: it must CLOSE outside the mother range.
        if i > 0:
            prev_close = float(df["Close"].iloc[i - 1])

            volume_ok = volume >= float(vol_ma) * VOL_MULT

            bull_trend_ok = close > float(ema)
            bear_trend_ok = close < float(ema)

            break_up = (
                waiting_breakout
                and i > mother_bar
                and close > mother_high
                and prev_close <= mother_high
            )

            break_down = (
                waiting_breakout
                and i > mother_bar
                and close < mother_low
                and prev_close >= mother_low
            )

            if break_up and volume_ok and bull_trend_ok:
                waiting_breakout = False
                waiting_confirmation = True
                breakout_dir = 1
                breakout_bar = i
                breakout_level = mother_high
                breakout_sl = mother_low

            elif break_down and volume_ok and bear_trend_ok:
                waiting_breakout = False
                waiting_confirmation = True
                breakout_dir = -1
                breakout_bar = i
                breakout_level = mother_low
                breakout_sl = mother_high

        # Confirmation candle must be the very next 30m candle.
        if (
            waiting_confirmation
            and breakout_bar is not None
            and i == breakout_bar + 1
        ):
            if (
                breakout_dir == 1
                and close > breakout_level
                and low > breakout_level
            ):
                signals.append({
                    "symbol": "",
                    "side": "BUY CE",
                    "time": timestamp.isoformat(),
                    "time_text": timestamp.strftime("%d-%m-%Y %H:%M"),
                    "close": close,
                    "level": float(breakout_level),
                    "sl": float(breakout_sl),
                })

            elif (
                breakout_dir == -1
                and close < breakout_level
                and high < breakout_level
            ):
                signals.append({
                    "symbol": "",
                    "side": "BUY PE",
                    "time": timestamp.isoformat(),
                    "time_text": timestamp.strftime("%d-%m-%Y %H:%M"),
                    "close": close,
                    "level": float(breakout_level),
                    "sl": float(breakout_sl),
                })

            waiting_confirmation = False
            breakout_dir = 0
            breakout_bar = None
            breakout_level = None
            breakout_sl = None

    return signals


# =========================================================
# STATE MANAGEMENT / DEDUPLICATION
# =========================================================

def load_state():
    if not STATE_FILE.exists():
        return None

    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            state = json.load(f)

        if not isinstance(state, dict):
            return None

        return state

    except Exception as exc:
        logging.error("Could not read state file: %s", exc)
        return None


def save_state(state):
    temp_file = STATE_FILE.with_suffix(".tmp")

    with temp_file.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    temp_file.replace(STATE_FILE)


def latest_bar_time(df):
    if df.empty:
        return None

    return df.index[-1].isoformat()


def previous_session_date(all_data):
    """
    Return the latest available trading date strictly before today.
    This handles weekends and most exchange holidays better than
    simply subtracting one calendar day.
    """
    today = now_ist().date()
    dates = set()

    for df in all_data.values():
        for ts in df.index:
            if ts.date() < today:
                dates.add(ts.date())

    if not dates:
        return None

    return max(dates)


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        raise RuntimeError(
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in GitHub Secrets."
        )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    response = requests.post(
        url,
        json={
            "chat_id": CHAT_ID,
            "text": message,
            "disable_web_page_preview": True,
        },
        timeout=20
    )

    response.raise_for_status()

    result = response.json()

    if not result.get("ok"):
        raise RuntimeError(f"Telegram API error: {result}")


def format_price(value):
    return f"{value:,.2f}"


def is_index(symbol):
    return symbol.startswith("^")


def tradingview_url(symbol):
    if is_index(symbol):
        mapping = {
            "^NSEI": "NSE:NIFTY",
            "^NSEBANK": "NSE:BANKNIFTY",
            "^CNXIT": "NSE:CNXIT",
            "^CNXPHARMA": "NSE:CNXPHARMA",
            "^CNXAUTO": "NSE:CNXAUTO",
            "^CNXMETAL": "NSE:CNXMETAL",
            "^CNXFMCG": "NSE:CNXFMCG",
            "^CNXENERGY": "NSE:CNXENERGY",
            "^CNXREALTY": "NSE:CNXREALTY",
            "^CNXPSUBANK": "NSE:CNXPSUBANK",
            "^CNXINFRA": "NSE:CNXINFRA",
            "^CNXMEDIA": "NSE:CNXMEDIA",
        }

        return (
            "https://www.tradingview.com/chart/?symbol="
            + requests.utils.quote(mapping.get(symbol, symbol), safe="")
        )

    ticker = symbol.removesuffix(".NS")
    tv_symbol = f"NSE:{ticker}"

    return (
        "https://www.tradingview.com/chart/?symbol="
        + requests.utils.quote(tv_symbol, safe="")
    )


def send_signal(symbol, signal):
    market_type = "INDEX" if is_index(symbol) else "F&O STOCK"

    message = (
        f"🔔 ROMYO CONFIRMED SIGNAL\n\n"
        f"📌 Symbol: {symbol.replace('.NS', '')}\n"
        f"📂 Type: {market_type}\n"
        f"🚦 Signal: {signal['side']}\n"
        f"⏱ Timeframe: 30 minutes\n"
        f"🕒 Confirmation candle start: {signal['time_text']} IST\n"
        f"💰 Confirmation close: ₹{format_price(signal['close'])}\n"
        f"📍 Breakout level: ₹{format_price(signal['level'])}\n"
        f"🛑 Chart-based stop loss: ₹{format_price(signal['sl'])}\n\n"
        f"📊 Chart: {tradingview_url(symbol)}\n\n"
        f"⚠️ Confirm the live price and option liquidity in Kite. "
        f"This is a chart signal, not a guaranteed profitable trade."
    )

    send_telegram(message)


# =========================================================
# MAIN
# =========================================================

def main():
    if not BOT_TOKEN or not CHAT_ID:
        raise RuntimeError(
            "Missing GitHub Secrets: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID"
        )

    logging.info("ROMYO scanner starting.")
    logging.info("Instruments: %s", len(SYMBOLS))

    state = load_state()
    first_run = state is None

    if first_run:
        state = {
            "version": 1,
            "initialized_at": now_ist().isoformat(),
            "last_processed": {},
        }

        logging.info(
            "First run: previous available trading session will be scanned."
        )

    last_processed = state.setdefault("last_processed", {})

    all_data = download_all()

    if not all_data:
        raise RuntimeError(
            "No market data downloaded. State was not advanced."
        )

    prior_session = previous_session_date(all_data)

    if first_run and prior_session is None:
        raise RuntimeError(
            "Could not identify a previous trading session. "
            "State was not initialized."
        )

    messages_sent = 0
    updated_watermarks = {}

    for symbol in SYMBOLS:
        df = all_data.get(symbol)

        if df is None or df.empty:
            logging.warning("No data for %s; will retry on next run.", symbol)
            continue

        signals = find_signals(df)

        for signal in signals:
            signal["symbol"] = symbol

        latest_ts = latest_bar_time(df)
        old_watermark = last_processed.get(symbol)

        if first_run:
            # First run sends ONLY signals confirmed during the
            # previous available trading session.
            eligible = [
                s for s in signals
                if datetime.fromisoformat(s["time"]).astimezone(IST).date()
                == prior_session
            ]
        else:
            # Subsequent runs send only signals newer than the
            # previously processed completed candle.
            if old_watermark:
                eligible = [
                    s for s in signals
                    if datetime.fromisoformat(s["time"])
                    > datetime.fromisoformat(old_watermark)
                ]
            else:
                # A symbol that previously had no data should not
                # dump old history into Telegram.
                eligible = []

        for signal in eligible:
            try:
                send_signal(symbol, signal)
                messages_sent += 1
                logging.info(
                    "Sent %s %s at %s",
                    symbol,
                    signal["side"],
                    signal["time_text"]
                )
                time.sleep(0.15)

            except Exception as exc:
                logging.error(
                    "Telegram send failed for %s: %s",
                    symbol,
                    exc
                )
                # Do not advance this symbol's watermark if a
                # message could not be delivered. It can retry.
                latest_ts = old_watermark
                break

        if latest_ts:
            updated_watermarks[symbol] = latest_ts

    # Update state only after processing. A failed instrument keeps
    # its old watermark so it can retry.
    last_processed.update(updated_watermarks)

    state["last_run_at"] = now_ist().isoformat()
    state["last_run_sent_count"] = messages_sent

    if first_run:
        state["first_run_previous_session"] = str(prior_session)

    save_state(state)

    logging.info(
        "Finished. First run=%s, signals sent=%s, instruments downloaded=%s",
        first_run,
        messages_sent,
        len(all_data)
    )


if __name__ == "__main__":
    main()
