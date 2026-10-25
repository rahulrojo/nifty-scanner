"""
15m Mother Candle Top & Bottom [1:3 RR]  ->  Telegram scanner
Port of the TradingView Pine script (pivot 5/2 + EMA50 filter, SL at mother candle, TP 1:3)

Flow:
  - First ever run (no state.json): sends last FRIDAY's signals, then goes live
  - Next runs: only new live signals of today
  - Manual run with mode=friday: Friday signals dobara bhejta hai

Secrets needed (GitHub -> Settings -> Secrets -> Actions):
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  (channel id like -100xxxxxxxxxx)
"""
import os
import sys
import json
import time
import html
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ----------------------------- CONFIG -----------------------------
IST = ZoneInfo("Asia/Kolkata")

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
RUN_MODE = os.environ.get("RUN_MODE", "auto").strip().lower()   # auto | friday
STATE_FILE = "state.json"

# Pine inputs
LEFT_BARS = 5
RIGHT_BARS = 2          # mother candle offset
USE_TREND = True
EMA_LEN = 50
RR = 3.0

INTERVAL = "15m"
BAR_MIN = 15
PERIOD = "30d"
MAX_SENT_KEYS = 3000

STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC", "SBIN",
    "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA",
    "TITAN", "ULTRACEMCO", "BAJFINANCE", "BAJAJFINSV", "NESTLEIND", "WIPRO", "HCLTECH",
    "TECHM", "POWERGRID", "NTPC", "ONGC", "COALINDIA", "TATASTEEL", "JSWSTEEL",
    "HINDALCO", "ADANIENT", "ADANIPORTS", "GRASIM", "CIPLA", "DRREDDY", "DIVISLAB",
    "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "M&M", "BRITANNIA", "APOLLOHOSP",
    "INDUSINDBK", "SBILIFE", "HDFCLIFE", "BPCL", "IOC", "TATACONSUM", "SHRIRAMFIN",
    "ETERNAL", "TRENT", "JIOFIN", "BEL", "HAL", "VEDL", "DLF", "GODREJCP", "PIDILITIND",
    "SIEMENS", "ABB", "AMBUJACEM", "BANKBARODA", "PNB", "CANBK", "IDFCFIRSTB",
    "FEDERALBNK", "AUBANK", "BANDHANBNK", "RECLTD", "PFC", "IRCTC", "TVSMOTOR",
    "ASHOKLEY", "BHARATFORG", "MOTHERSON", "TATAPOWER", "TATAELXSI", "PERSISTENT",
    "COFORGE", "LTIM", "DIXON", "POLYCAB", "HAVELLS", "VOLTAS", "JINDALSTEL", "SAIL",
    "NMDC", "GAIL", "PETRONET", "LUPIN", "AUROPHARMA", "BIOCON", "TORNTPHARM",
    "ZYDUSLIFE", "ALKEM", "MAXHEALTH", "INDIGO", "LODHA", "GODREJPROP", "TMPV",
]

# (display name, yahoo symbol, tradingview symbol)
INDEXES = [
    ("NIFTY", "^NSEI", "NSE:NIFTY"),
    ("BANKNIFTY", "^NSEBANK", "NSE:BANKNIFTY"),
    ("SENSEX", "^BSESN", "BSE:SENSEX"),
    ("FINNIFTY", "NIFTY_FIN_SERVICE.NS", "NSE:CNXFINANCE"),
    ("NIFTY IT", "^CNXIT", "NSE:CNXIT"),
    ("NIFTY AUTO", "^CNXAUTO", "NSE:CNXAUTO"),
    ("NIFTY PHARMA", "^CNXPHARMA", "NSE:CNXPHARMA"),
    ("NIFTY FMCG", "^CNXFMCG", "NSE:CNXFMCG"),
    ("NIFTY METAL", "^CNXMETAL", "NSE:CNXMETAL"),
    ("NIFTY ENERGY", "^CNXENERGY", "NSE:CNXENERGY"),
    ("NIFTY REALTY", "^CNXREALTY", "NSE:CNXREALTY"),
    ("NIFTY PSU BANK", "^CNXPSUBANK", "NSE:CNXPSUBANK"),
]


def build_universe():
    uni = []
    for s in STOCKS:
        uni.append({
            "name": s,
            "yf": s + ".NS",
            "tv": "NSE:" + s.replace("-", "_").replace("&", "_"),
        })
    for name, y, tv in INDEXES:
        uni.append({"name": name, "yf": y, "tv": tv})
    return uni


# ----------------------------- DATA -----------------------------
def clean(df):
    if df is None or df.empty:
        return None
    need = ["Open", "High", "Low", "Close"]
    if any(c not in df.columns for c in need):
        return None
    df = df.dropna(subset=need).copy()
    if df.empty:
        return None
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert(IST)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # only regular session candles (09:15 - 15:29 start)
    t = df.index.time
    df = df[(t >= dtime(9, 15)) & (t < dtime(15, 30))]
    if df.empty:
        return None
    return df[need]


def extract(raw, sym):
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if sym in raw.columns.get_level_values(0):
                sub = raw[sym]
            elif sym in raw.columns.get_level_values(1):
                sub = raw.xs(sym, axis=1, level=1)
            else:
                return None
        else:
            sub = raw
        return clean(sub)
    except Exception:
        return None


def fetch_all(universe):
    data = {}
    syms = [u["yf"] for u in universe]
    for k in range(0, len(syms), 25):
        batch = syms[k:k + 25]
        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(batch, period=PERIOD, interval=INTERVAL,
                                  group_by="ticker", auto_adjust=False,
                                  progress=False, threads=True)
                break
            except Exception as e:
                print(f"batch download error ({attempt + 1}/3): {e}")
                time.sleep(3)
        if raw is None or raw.empty:
            continue
        for s in batch:
            df = extract(raw, s)
            if df is not None and len(df) > EMA_LEN + 20:
                data[s] = df

    missing = [s for s in syms if s not in data]
    for s in missing:
        try:
            h = yf.Ticker(s).history(period=PERIOD, interval=INTERVAL, auto_adjust=False)
            df = clean(h)
            if df is not None and len(df) > EMA_LEN + 20:
                data[s] = df
        except Exception:
            pass

    still = [s for s in syms if s not in data]
    print(f"Data ok: {len(data)}/{len(syms)}")
    if still:
        print("Skipped (no data): " + ", ".join(still))
    return data


# ----------------------------- STRATEGY -----------------------------
def find_signals(u, df):
    """Exact port: pivot(5,2) confirmed on bar i, mother candle = bar i-2."""
    n = len(df)
    if n < EMA_LEN + LEFT_BARS + RIGHT_BARS + 5:
        return []
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    ema = df["Close"].ewm(span=EMA_LEN, adjust=False).mean().values
    times = df.index

    out = []
    start = max(LEFT_BARS + RIGHT_BARS, n - 400)
    for i in range(start, n):
        m = i - RIGHT_BARS

        # pivot low (bottom): strictly lower than left bars, lower/equal than right bars
        is_low = l[m] < l[m - LEFT_BARS:m].min() and l[m] <= l[m + 1:i + 1].min()
        # pivot high (top)
        is_high = h[m] > h[m - LEFT_BARS:m].max() and h[m] >= h[m + 1:i + 1].max()

        if is_low and (not USE_TREND or c[i] > ema[i]):
            entry = float(c[i])
            sl = float(l[m])
            risk = entry - sl
            if risk > 0:
                out.append({
                    "name": u["name"], "tv": u["tv"], "side": "CE",
                    "ts": times[i], "mother_ts": times[m],
                    "entry": entry, "sl": sl, "tp": entry + RR * risk, "risk": risk,
                })

        if is_high and (not USE_TREND or c[i] < ema[i]):
            entry = float(c[i])
            sl = float(h[m])
            risk = sl - entry
            if risk > 0:
                out.append({
                    "name": u["name"], "tv": u["tv"], "side": "PE",
                    "ts": times[i], "mother_ts": times[m],
                    "entry": entry, "sl": sl, "tp": entry - RR * risk, "risk": risk,
                })
    return out


# ----------------------------- TELEGRAM -----------------------------
def tg_send(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    for attempt in range(5):
        try:
            r = requests.post(url, data=payload, timeout=30)
            if r.status_code == 200:
                time.sleep(1.2)
                return True
            if r.status_code == 429:
                wait = r.json().get("parameters", {}).get("retry_after", 5)
                time.sleep(wait + 1)
                continue
            print("Telegram error:", r.status_code, r.text)
            return False
        except Exception as e:
            print("Telegram exception:", e)
            time.sleep(3)
    return False


def link(tv):
    return html.escape(f"https://www.tradingview.com/chart/?symbol={tv}&interval=15", quote=True)


def px(x):
    return f"{x:.2f}"


def single_msg(s):
    ce = s["side"] == "CE"
    icon = "🟢" if ce else "🔴"
    side = "BUY CE" if ce else "BUY PE"
    mother = "Bottom" if ce else "Top"
    return (
        f"{icon} <b>{side}</b> — <b>{html.escape(s['name'])}</b> (15m)\n"
        f"🕯 Signal candle: {s['ts']:%H:%M}, {s['ts']:%d %b %Y}\n"
        f"📍 Mother ({mother}) candle: {s['mother_ts']:%H:%M}, {s['mother_ts']:%d %b}\n"
        f"Entry: <b>{px(s['entry'])}</b>\n"
        f"SL: {px(s['sl'])}\n"
        f"TP (1:3): {px(s['tp'])}\n"
        f"Risk: {px(s['risk'])} pts\n"
        f"📈 <a href=\"{link(s['tv'])}\">TradingView</a>"
    )


def line_msg(s):
    icon = "🟢" if s["side"] == "CE" else "🔴"
    return (
        f"{icon} <b>{s['side']}</b> {html.escape(s['name'])} | {s['ts']:%H:%M} | "
        f"E {px(s['entry'])} SL {px(s['sl'])} TP {px(s['tp'])} | "
        f"<a href=\"{link(s['tv'])}\">Chart</a>"
    )


def send_batched(header, sigs):
    chunk = header + "\n\n"
    ok = True
    for s in sigs:
        ln = line_msg(s)
        if len(chunk) + len(ln) + 1 > 3800:
            ok = tg_send(chunk) and ok
            chunk = ""
        chunk += ln + "\n"
    if chunk.strip():
        ok = tg_send(chunk) and ok
    return ok


# ----------------------------- STATE -----------------------------
def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            st = json.load(f)
            st.setdefault("initialized", False)
            st.setdefault("sent", [])
            return st
    except Exception:
        return {"initialized": False, "sent": []}


def save_state(st):
    st["sent"] = st["sent"][-MAX_SENT_KEYS:]
    with open(STATE_FILE, "w") as f:
        json.dump(st, f)


def key(s):
    return f"{s['name']}|{s['ts'].isoformat()}|{s['side']}"


# ----------------------------- MAIN -----------------------------
def main():
    if not TOKEN or not CHAT_ID:
        sys.exit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID secrets missing")

    now = datetime.now(IST)
    today = now.date()
    state = load_state()
    sent = set(state["sent"])
    sent_order = list(state["sent"])

    backfill = RUN_MODE == "friday" or not state["initialized"]

    friday = today
    while friday.weekday() != 4:
        friday -= timedelta(days=1)

    universe = build_universe()
    data = fetch_all(universe)
    if not data:
        sys.exit("No data fetched - exiting without changing state")

    now_ts = pd.Timestamp(now)
    all_sigs = []
    for u in universe:
        df = data.get(u["yf"])
        if df is None:
            continue
        df = df[df.index + pd.Timedelta(minutes=BAR_MIN) <= now_ts]   # only closed candles
        if len(df) == 0:
            continue
        all_sigs.extend(find_signals(u, df))

    all_sigs.sort(key=lambda s: (s["ts"], s["name"]))

    # ---- Friday backfill ----
    back_ok = True
    back_keys = set()
    if backfill:
        back = [s for s in all_sigs if s["ts"].date() == friday and key(s) not in sent]
        back_keys = {key(s) for s in back}
        ce = sum(1 for s in back if s["side"] == "CE")
        pe = len(back) - ce
        if back:
            header = (f"📅 <b>{friday:%A, %d %b %Y}</b> — Mother Candle Top/Bottom (15m)\n"
                      f"Total {len(back)} signals | 🟢 CE {ce} | 🔴 PE {pe}")
            back_ok = send_batched(header, back)
        else:
            back_ok = tg_send(f"📅 <b>{friday:%A, %d %b %Y}</b> — koi signal nahi mila "
                              f"(ya market band tha).")
        if back_ok:
            for s in back:
                k = key(s)
                sent.add(k)
                sent_order.append(k)

    # ---- Live signals (today) ----
    live = [s for s in all_sigs
            if s["ts"].date() == today and key(s) not in sent and key(s) not in back_keys]
    if live:
        if len(live) > 8:
            header = f"🔔 <b>Live signals</b> — {today:%d %b %Y} (15m) | {len(live)} new"
            if send_batched(header, live):
                for s in live:
                    sent.add(key(s))
                    sent_order.append(key(s))
        else:
            for s in live:
                if tg_send(single_msg(s)):
                    sent.add(key(s))
                    sent_order.append(key(s))

    if back_ok:
        state["initialized"] = True
    state["sent"] = sent_order
    save_state(state)
    print(f"Done. Friday-backfill={backfill}, live new={len(live)}")


if __name__ == "__main__":
    main()
