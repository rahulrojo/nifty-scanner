"""
ROMYO Confirmed Mother Candle Breakout  ->  Python port (30 min) + Telegram alerts

Pine logic (exact port):
  1. Mother candle : range between MIN/MAX ATR, body >= MIN_BODY_PCT, only when no setup is active
  2. Breakout      : first candle that CLOSES outside mother (close[1] must still be inside),
                     with volume filter + EMA21 trend filter
  3. Confirmation  : the NEXT candle must close AND stay (low/high) fully beyond the breakout level
                     -> only this candle prints BUY CE / BUY PE
  Stop loss = opposite side of the mother candle.

Run behaviour:
  * First run (no state file)  -> sends signals of the PREVIOUS WORKING DAY only
  * Next runs                  -> sends only NEW signals of today (live)

ENV:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   (required)
  DEBUG=1            -> prints every signal of the last 3 days (no Telegram needed for compare)
  ONLY=SIEMENS,TCS   -> scan only these symbols (for testing against TradingView)
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from html import escape
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ===================== SETTINGS (same as Pine) =====================
EMA_LEN = 21
ATR_LEN = 14
MIN_MOTHER_ATR = 1.0
MAX_MOTHER_ATR = 2.5
MIN_BODY_PCT = 50.0
USE_VOLUME = True
VOL_LEN = 20
VOL_MULT = 1.0
USE_EMA = True
MAX_WAIT = 8

# ===================== BOT SETTINGS =====================
INTERVAL = "30m"          # chart timeframe (built by us from 5m, anchored at 9:15 like TradingView)
FETCH_INTERVAL = "5m"
PERIOD = "45d"                    # yfinance limit for 30m is 60 days
TARGET_RRS = (1.0, 1.5, 2.0)      # targets as multiple of risk (entry - SL)
STATE_FILE = "romyo_state_v2.json"
BATCH = 25
IST = ZoneInfo("Asia/Kolkata")
CANDLE_MIN = 30
SESSION_END = (15, 30)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DEBUG = os.getenv("DEBUG", "0") == "1"
ONLY = {x.strip().upper() for x in os.getenv("ONLY", "").split(",") if x.strip()}

# ===================== SYMBOLS =====================
# (yahoo symbol, display name, tradingview symbol)
INDICES = [
    ("^NSEI", "NIFTY 50", "NSE:NIFTY"),
    ("^NSEBANK", "BANK NIFTY", "NSE:BANKNIFTY"),
    ("^CNXFIN", "FIN NIFTY", "NSE:CNXFINANCE"),
    ("NIFTY_MID_SELECT.NS", "MIDCAP NIFTY", "NSE:NIFTY_MID_SELECT"),
    ("^BSESN", "SENSEX", "BSE:SENSEX"),
]

STOCKS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK",
    "BANKBARODA", "PNB", "CANBK", "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "AUBANK",
    "YESBANK", "UNIONBANK", "INFY", "TCS", "WIPRO", "HCLTECH", "TECHM", "PIIND",
    "COFORGE", "PERSISTENT", "MPHASIS", "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN",
    "SHRIRAMFIN", "MUTHOOTFIN", "MANAPPURAM", "LICHSGFIN", "PFC", "RECLTD", "SBILIFE",
    "HDFCLIFE", "ICICIPRULI", "ICICIGI", "SBICARD", "JIOFIN", "TATASTEEL", "JSWSTEEL",
    "HINDALCO", "VEDL", "JINDALSTEL", "SAIL", "NMDC", "COALINDIA", "ONGC", "BPCL", "IOC",
    "HINDPETRO", "GAIL", "NTPC", "POWERGRID", "TATAPOWER", "ADANIENT", "ADANIPORTS",
    "ADANIGREEN", "LT", "BEL", "HAL", "BHEL", "SIEMENS", "POLYCAB", "HAVELLS", "DIXON",
    "VOLTAS", "MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "TVSMOTOR",
    "ASHOKLEY", "SUNPHARMA", "CIPLA", "DRREDDY", "LUPIN", "AUROPHARMA", "BIOCON",
    "TORNTPHARM", "APOLLOHOSP", "TITAN", "TRENT", "ITC", "HINDUNILVR", "BRITANNIA",
    "TATACONSUM", "NESTLEIND", "BHARTIARTL", "INDIGO", "IRCTC", "DLF", "GODREJPROP",
    "ULTRACEMCO", "GRASIM", "AMBUJACEM", "UPL", "ETERNAL",
]


def build_universe():
    uni = []
    for ysym, name, tv in INDICES:
        uni.append({"y": ysym, "name": name, "tv": tv, "index": True})
    for s in STOCKS:
        tv = "NSE:" + s.replace("&", "_").replace("-", "_")
        uni.append({"y": s + ".NS", "name": s, "tv": tv, "index": False})
    if ONLY:
        uni = [u for u in uni if u["name"].upper() in ONLY]
    return uni


# ===================== PINE-STYLE INDICATORS =====================
def pine_rma(values, length):
    out = np.full(len(values), np.nan)
    if len(values) < length:
        return out
    out[length - 1] = np.mean(values[:length])
    a = 1.0 / length
    for i in range(length, len(values)):
        out[i] = a * values[i] + (1 - a) * out[i - 1]
    return out


def pine_ema(values, length):
    out = np.full(len(values), np.nan)
    if len(values) < length:
        return out
    out[length - 1] = np.mean(values[:length])
    a = 2.0 / (length + 1)
    for i in range(length, len(values)):
        out[i] = a * values[i] + (1 - a) * out[i - 1]
    return out


def pine_atr(h, l, c, length):
    tr = np.empty(len(h))
    tr[0] = h[0] - l[0]
    for i in range(1, len(h)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    return pine_rma(tr, length)


# ===================== DATA =====================
def clean(df):
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        return None
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["Volume"] = df["Volume"].fillna(0)
    return to_30m(df)


def to_30m(df):
    """Resample 5m -> 30m bars at 09:15, 09:45 ... 15:15 (same grid as TradingView NSE)."""
    df = df.between_time("09:15", "15:29")
    if df.empty:
        return None
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = df.resample("30min", origin="start_day", offset="9h15min", label="left", closed="left").agg(agg)
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out if not out.empty else None


def drop_open_candle(df, now):
    """Keep only fully closed candles (Pine: barstate.isconfirmed)."""
    keep = []
    for ts in df.index:
        close_t = ts + timedelta(minutes=CANDLE_MIN)
        sess_end = ts.replace(hour=SESSION_END[0], minute=SESSION_END[1], second=0, microsecond=0)
        close_t = min(close_t, sess_end)
        keep.append(close_t <= now)
    return df[np.array(keep, dtype=bool)]


def fetch_all(universe):
    data = {}
    for i in range(0, len(universe), BATCH):
        chunk = universe[i:i + BATCH]
        tickers = [u["y"] for u in chunk]
        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(tickers, period=PERIOD, interval=FETCH_INTERVAL, group_by="ticker",
                                  auto_adjust=False, threads=False, progress=False)
                break
            except Exception as e:  # noqa
                print(f"batch download error ({attempt + 1}/3): {e}")
                time.sleep(3)
        for u in chunk:
            df = None
            try:
                if raw is not None and not raw.empty:
                    if isinstance(raw.columns, pd.MultiIndex):
                        if u["y"] in raw.columns.get_level_values(0):
                            df = clean(raw[u["y"]])
                    else:
                        df = clean(raw)
            except Exception as e:
                print(f"parse error {u['name']}: {e}")
                df = None
            if df is None:  # single-ticker fallback
                try:
                    df = clean(yf.Ticker(u["y"]).history(period=PERIOD, interval=FETCH_INTERVAL, auto_adjust=False))
                except Exception:
                    df = None
            if df is None or len(df) < ATR_LEN + 5:
                print(f"SKIP (no data): {u['name']} [{u['y']}]")
                continue
            data[u["y"]] = df
    return data


# ===================== STRATEGY (exact Pine port) =====================
def scan_symbol(df, index_symbol):
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    v = df["Volume"].to_numpy(float)
    times = list(df.index)
    n = len(df)

    ema = pine_ema(c, EMA_LEN)
    atr = pine_atr(h, l, c, ATR_LEN)
    vol_ma = pd.Series(v).rolling(VOL_LEN).mean().to_numpy()

    # indices have no volume on Yahoo -> volume filter cannot work, so skip it for them
    use_vol = USE_VOLUME and not (index_symbol or v.sum() == 0)

    mother_high = mother_low = np.nan
    mother_bar = None
    waiting_breakout = False
    waiting_conf = False
    breakout_dir = 0
    breakout_bar = None
    breakout_level = np.nan
    breakout_sl = np.nan
    bo_mother = None
    bo_vr = None

    signals = []

    for i in range(n):
        # expire old setup
        if waiting_breakout and mother_bar is not None and i - mother_bar > MAX_WAIT:
            waiting_breakout = False
            waiting_conf = False
            breakout_dir = 0

        rng = h[i] - l[i]
        body_pct = abs(c[i] - o[i]) / rng * 100 if rng > 0 else 0.0
        mother_cand = (not np.isnan(atr[i])) and rng >= atr[i] * MIN_MOTHER_ATR \
            and rng <= atr[i] * MAX_MOTHER_ATR and body_pct >= MIN_BODY_PCT

        if mother_cand and not waiting_breakout and not waiting_conf:
            mother_high, mother_low, mother_bar = h[i], l[i], i
            waiting_breakout = True

        vol_ok = (not use_vol) or ((not np.isnan(vol_ma[i])) and v[i] >= vol_ma[i] * VOL_MULT)
        bull_ok = (not USE_EMA) or (c[i] > ema[i])
        bear_ok = (not USE_EMA) or (c[i] < ema[i])
        prev_close = c[i - 1] if i > 0 else np.nan

        break_up = waiting_breakout and i > mother_bar and c[i] > mother_high and prev_close <= mother_high
        break_down = waiting_breakout and i > mother_bar and c[i] < mother_low and prev_close >= mother_low

        if break_up and vol_ok and bull_ok:
            waiting_breakout = False
            waiting_conf = True
            breakout_dir = 1
            breakout_bar = i
            breakout_level = mother_high
            breakout_sl = mother_low
            bo_mother = (mother_bar, mother_high, mother_low)
            bo_vr = (v[i] / vol_ma[i]) if (not np.isnan(vol_ma[i]) and vol_ma[i] > 0) else None

        if break_down and vol_ok and bear_ok:
            waiting_breakout = False
            waiting_conf = True
            breakout_dir = -1
            breakout_bar = i
            breakout_level = mother_low
            breakout_sl = mother_high
            bo_mother = (mother_bar, mother_high, mother_low)
            bo_vr = (v[i] / vol_ma[i]) if (not np.isnan(vol_ma[i]) and vol_ma[i] > 0) else None

        buy_ce = buy_pe = False
        if waiting_conf and i == breakout_bar + 1:
            if breakout_dir == 1 and c[i] > breakout_level and l[i] > breakout_level:
                buy_ce = True
            elif breakout_dir == -1 and c[i] < breakout_level and h[i] < breakout_level:
                buy_pe = True
            waiting_conf = False

        if buy_ce or buy_pe:
            vr = bo_vr
            signals.append({
                "side": "CE" if buy_ce else "PE",
                "time": times[i],
                "close": float(c[i]),
                "sl": float(breakout_sl),
                "level": float(breakout_level),
                "bo_time": times[breakout_bar],
                "bo_close": float(c[breakout_bar]),
                "m_time": times[bo_mother[0]],
                "m_high": float(bo_mother[1]),
                "m_low": float(bo_mother[2]),
                "vol_ratio": vr,
            })
    return signals


# ===================== TELEGRAM =====================
def fmt_price(x):
    return f"{x:,.2f}"


def fmt_t(ts, ref=None):
    """'10:15' if same day as ref else '17 Oct 10:15'."""
    if ref is not None and ts.date() == ref.date():
        return ts.strftime("%H:%M")
    return ts.strftime("%d %b %H:%M")


def build_message(u, s):
    is_ce = s["side"] == "CE"
    icon = "🟢" if is_ce else "🔴"
    risk = abs(s["close"] - s["sl"])
    sign = 1 if is_ce else -1
    tg = []
    for rr in TARGET_RRS:
        tg.append(f"T{len(tg) + 1} ({rr:g}R) {fmt_price(s['close'] + sign * rr * risk)}")
    risk_pct = risk / s["close"] * 100 if s["close"] else 0
    link = f"https://in.tradingview.com/chart/?symbol={quote(u['tv'], safe='')}&interval=30"

    lines = [
        f"{icon} <b>BUY {s['side']}</b> — <b>{escape(u['name'])}</b>",
        f"🕒 <b>Candle:</b> {s['time'].strftime('%d %b %Y, %H:%M')} IST (30m)",
        f"💰 Entry (candle close): {fmt_price(s['close'])}",
        f"🛑 SL: {fmt_price(s['sl'])}  (risk {fmt_price(risk)} | {risk_pct:.2f}%)",
        "🎯 " + " | ".join(tg),
        f"👩 Mother: {fmt_t(s['m_time'], s['time'])}  H {fmt_price(s['m_high'])} / L {fmt_price(s['m_low'])}",
        f"⚡ Breakout candle: {fmt_t(s['bo_time'], s['time'])} (close {fmt_price(s['bo_close'])})",
    ]
    if s["vol_ratio"] is not None:
        lines.append(f"📊 Breakout candle volume: {s['vol_ratio']:.1f}x of {VOL_LEN}-bar avg")
    lines.append(f"📈 <a href=\"{link}\">TradingView chart</a>")
    lines.append("<i>Target/SL underlying price ke hain, option premium ke nahi.</i>")
    return "\n".join(lines)


def send_telegram(text):
    if not TOKEN or not CHAT_ID:
        print("Telegram secrets missing — message not sent:\n" + text)
        return False
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    for _ in range(4):
        try:
            r = requests.post(url, json=payload, timeout=20)
            if r.status_code == 200:
                time.sleep(1.2)
                return True
            if r.status_code == 429:
                wait = r.json().get("parameters", {}).get("retry_after", 5)
                time.sleep(wait + 1)
                continue
            print("Telegram error:", r.status_code, r.text[:200])
            return False
        except Exception as e:  # noqa
            print("Telegram exception:", e)
            time.sleep(3)
    return False


# ===================== STATE =====================
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(state, today):
    cutoff = (today - timedelta(days=10)).isoformat()
    state["sent"] = sorted(k for k in state.get("sent", []) if k.split("|")[1][:10] >= cutoff)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=1)


def sig_key(u, s):
    return f"{u['name']}|{s['time'].isoformat()}|{s['side']}"


# ===================== MAIN =====================
def main():
    now = datetime.now(IST)
    today = now.date()
    universe = build_universe()
    print(f"Now (IST): {now:%Y-%m-%d %H:%M} | symbols: {len(universe)}")

    data = fetch_all(universe)
    print(f"Fetched data for {len(data)}/{len(universe)} symbols")
    if not data:
        print("No data fetched at all — exiting without touching state.")
        send_telegram("⚠️ <b>ROMYO scanner</b>: Yahoo se data nahi mila (rate-limit/blocked ho sakta hai). Agli run mein dobara try hoga.")
        sys.exit(1)

    all_signals = []  # (universe item, signal)
    all_dates = set()
    for u in universe:
        df = data.get(u["y"])
        if df is None:
            continue
        df = drop_open_candle(df, now)
        if len(df) < ATR_LEN + 5:
            continue
        all_dates.update(d for d in df.index.date)
        for s in scan_symbol(df, u["index"]):
            all_signals.append((u, s))

    all_signals.sort(key=lambda x: (x[1]["time"], x[0]["name"]))

    if all_dates:
        print(f"Latest closed candle date in data: {max(all_dates)} | trading days loaded: {len(all_dates)}")
    by_day = {}
    for _, s_ in all_signals:
        by_day[s_["time"].date()] = by_day.get(s_["time"].date(), 0) + 1
    print("Signals found per day (last 5):", dict(sorted(by_day.items())[-5:]))

    if DEBUG:
        d3 = sorted(all_dates)[-3:]
        print("---- DEBUG: signals of last 3 trading days ----")
        for u, s in all_signals:
            if s["time"].date() in d3:
                print(f"{u['name']:<12} BUY {s['side']} candle {s['time']:%d-%b %H:%M} close {s['close']:.2f} SL {s['sl']:.2f}")

    state = load_state()
    first_run = not state.get("initialized")
    sent = set(state.get("sent", []))

    if first_run:
        prev_days = [d for d in all_dates if d < today]
        if not prev_days:
            print("No previous trading day found in data — exiting.")
            send_telegram("⚠️ <b>ROMYO scanner</b>: pichle working day ka data nahi mila.")
            sys.exit(1)
        prev_day = max(prev_days)
        # every signal that exists right now is marked as seen; only prev day's are sent
        for u, s in all_signals:
            sent.add(sig_key(u, s))
        to_send = [(u, s) for u, s in all_signals if s["time"].date() == prev_day]
        header = f"📌 <b>ROMYO Confirmed MC Breakout</b>\nPichle working day ({prev_day:%d %b %Y}) ke signals: <b>{len(to_send)}</b>"
        send_telegram(header)
        mark_after_send = False
    else:
        to_send = [(u, s) for u, s in all_signals
                   if s["time"].date() == today and sig_key(u, s) not in sent]
        mark_after_send = True

    print(f"Signals to send: {len(to_send)} (first_run={first_run})")
    if not first_run and not to_send:
        print("Not first run -> only NEW signals of TODAY are sent; none found. (State reset ke bina pichle din ke signals nahi aate.)")

    ok_count = 0
    for u, s in to_send:
        if send_telegram(build_message(u, s)):
            ok_count += 1
            if mark_after_send and TOKEN and CHAT_ID:
                sent.add(sig_key(u, s))
                state["sent"] = list(sent)
                state["initialized"] = True
                save_state(state, today)

    if not (TOKEN and CHAT_ID):
        print("Dry run (Telegram secrets missing) — state NOT saved.")
        return
    state["initialized"] = True
    state["sent"] = list(sent)
    state["last_run"] = now.isoformat()
    save_state(state, today)
    print(f"Done. Sent {ok_count}/{len(to_send)}")


if __name__ == "__main__":
    main()
