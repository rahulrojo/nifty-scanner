"""
ROMYO Confirmed Mother Candle Breakout (30m)  ->  Telegram scanner

Candle grid fix: NSE 30m candles = 09:15, 09:45, 10:15 ... 15:15 (last one is 15 min only).
15m data is resampled with offset=15min so candles match TradingView exactly.

Flow:
  mother candle -> breakout candle (close beyond mother) -> confirmation candle
  (forms fully beyond the breakout candle, no touch) -> BUY CE / BUY PE on confirm close

  - First run (no state file): previous working day's signals, then live
  - Next runs: only new live signals of today
  - Manual run with mode=backfill: previous working day dobara bhejta hai
  - DEBUG_SYMBOL=PIDILITIND: logs me aligned 30m candles print karta hai (TradingView se match karne ke liye)

Secrets: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
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
RUN_MODE = os.environ.get("RUN_MODE", "auto").strip().lower()      # auto | backfill
DEBUG_SYMBOL = os.environ.get("DEBUG_SYMBOL", "").strip().upper()
STATE_FILE = "state_mother30.json"

# ---- strategy inputs ----
EMA_LEN = 21
ATR_LEN = 14
VOL_LEN = 20
MOTHER_MIN_ATR = 1.0      # mother range >= 1.0 ATR
MOTHER_MAX_ATR = 2.5      # mother range <= 2.5 ATR
MOTHER_BODY = 0.5         # mother body >= 50% of its range
WAIT_BARS = 8             # breakout must come within 8 bars of mother
USE_VOL = True
VOL_MIN = 1.0             # breakout volume >= 1.0 x avg(20)
USE_TREND = True          # CE: confirm close > EMA21, PE: confirm close < EMA21

TF_MIN = 30
PERIOD = "45d"
TG_DELAY = 3.1
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
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df["Volume"] = df["Volume"].fillna(0.0)
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert(IST)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    t = df.index.time
    df = df[(t >= dtime(9, 15)) & (t < dtime(15, 30))]      # regular session only
    if df.empty:
        return None
    return df[need + ["Volume"]]


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
                raw = yf.download(batch, period=PERIOD, interval="15m",
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
            if df is not None and len(df) > 150:
                data[s] = df

    for s in [x for x in syms if x not in data]:
        try:
            h = yf.Ticker(s).history(period=PERIOD, interval="15m", auto_adjust=False)
            df = clean(h)
            if df is not None and len(df) > 150:
                data[s] = df
        except Exception:
            pass

    missing = [s for s in syms if s not in data]
    print(f"Data ok: {len(data)}/{len(syms)}")
    if missing:
        print("Skipped (no data): " + ", ".join(missing))
    return data


def candle_minutes(idx):
    """30 min for every candle except 15:15 which is only 15 min."""
    return np.array([15 if t == dtime(15, 15) else TF_MIN for t in idx.time])


def cend(ts):
    return ts + pd.Timedelta(minutes=15 if ts.time() == dtime(15, 15) else TF_MIN)


def to_30m(d15):
    """TradingView NSE grid: 09:15, 09:45, ... 15:15."""
    g = d15.resample("30min", offset="15min", label="left", closed="left")
    o = g.agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
              Close=("Close", "last"), Volume=("Volume", "sum"), N=("Open", "count"))
    o = o[o["N"] > 0]
    expected = np.where(candle_minutes(o.index) == 15, 1, 2)    # 15m bars needed per candle
    o = o[o["N"].values == expected]                            # drop half-built candles
    return o[["Open", "High", "Low", "Close", "Volume"]]


def closed_only(d30, now_ts):
    if len(d30) == 0:
        return d30
    end = d30.index + pd.to_timedelta(candle_minutes(d30.index), unit="m")
    return d30[end <= now_ts]


def atr_rma(d):
    pc = d["Close"].shift(1)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - pc).abs(), (d["Low"] - pc).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / ATR_LEN, adjust=False).mean().values      # Wilder RMA = ta.atr


# ----------------------------- STRATEGY -----------------------------
def quality(vr, mother_atr, dist_ema_atr, bo_body, bo_beyond_atr, gap_atr):
    """Quality out of 10 (volume 3, mother size 2, EMA distance 2, breakout body 1,
    breakout strength 1, confirm gap 1)."""
    s = 0
    if np.isnan(vr):
        s += 1
    elif vr >= 2.0:
        s += 3
    elif vr >= 1.5:
        s += 2
    elif vr >= 1.2:
        s += 1
    s += 2 if 1.2 <= mother_atr <= 2.0 else 1
    s += 2 if dist_ema_atr >= 0.5 else (1 if dist_ema_atr >= 0.2 else 0)
    s += 1 if bo_body >= 0.6 else 0
    s += 1 if bo_beyond_atr >= 0.25 else 0
    s += 1 if gap_atr >= 0.1 else 0
    return int(min(s, 10))


def detect(u, d):
    n = len(d)
    if n < 40:
        return []
    o = d["Open"].values.astype(float)
    h = d["High"].values.astype(float)
    l = d["Low"].values.astype(float)
    c = d["Close"].values.astype(float)
    v = d["Volume"].values.astype(float)
    ema = d["Close"].ewm(span=EMA_LEN, adjust=False).mean().values
    atr = atr_rma(d)
    vsma = pd.Series(v).rolling(VOL_LEN).mean().values
    idx = d.index

    def vol_ratio(i):
        if np.isnan(vsma[i]) or vsma[i] <= 0:
            return np.nan
        return v[i] / vsma[i]

    out = []
    mi, mh, ml = -1, 0.0, 0.0          # active mother
    stage, bk, side = 0, -1, 0         # stage 1 = breakout seen, waiting for confirm candle
    b_m = (-1, 0.0, 0.0)               # mother saved at breakout time

    for i in range(max(ATR_LEN, VOL_LEN) + 1, n):
        # ---- 1) confirmation candle (the candle right after breakout) ----
        if stage == 1:
            ok = (l[i] > h[bk]) if side == 1 else (h[i] < l[bk])
            trend_ok = (not USE_TREND) or ((c[i] > ema[i]) if side == 1 else (c[i] < ema[i]))
            if ok and trend_ok and atr[i] > 0:
                mi_b, mh_b, ml_b = b_m
                entry = float(c[i])
                sl = ml_b if side == 1 else mh_b
                risk = abs(entry - sl)
                if risk > 0:
                    vr = vol_ratio(bk)
                    mother_atr = (mh_b - ml_b) / atr[mi_b]
                    rng_b = max(h[bk] - l[bk], 1e-9)
                    bo_body = abs(c[bk] - o[bk]) / rng_b
                    beyond = ((c[bk] - mh_b) if side == 1 else (ml_b - c[bk])) / atr[bk]
                    gap = ((l[i] - h[bk]) if side == 1 else (l[bk] - h[i])) / atr[i]
                    dist = abs(c[i] - ema[i]) / atr[i]
                    q = quality(vr, mother_atr, dist, bo_body, beyond, gap)
                    out.append({
                        "name": u["name"], "tv": u["tv"], "side": side,
                        "ts": idx[i], "bk_ts": idx[bk], "m_ts": idx[mi_b],
                        "mh": mh_b, "ml": ml_b,
                        "entry": entry, "sl": float(sl), "risk": float(risk),
                        "q": q, "vr": vr, "mother_atr": float(mother_atr),
                    })
            stage = 0
            mi = -1                      # one attempt only; confirm candle may become a new mother below

        # ---- 2) active mother -> look for breakout candle ----
        elif mi >= 0:
            if i - mi > WAIT_BARS:
                mi = -1
            else:
                vr_i = vol_ratio(i)
                vol_ok = (not USE_VOL) or np.isnan(vr_i) or vr_i >= VOL_MIN
                if vol_ok and c[i] > mh:
                    stage, side, bk, b_m = 1, 1, i, (mi, mh, ml)
                    continue
                if vol_ok and c[i] < ml:
                    stage, side, bk, b_m = 1, -1, i, (mi, mh, ml)
                    continue

        # ---- 3) look for a new mother candle ----
        if mi < 0 and stage == 0 and atr[i] > 0:
            rng = h[i] - l[i]
            if rng > 0 and MOTHER_MIN_ATR <= rng / atr[i] <= MOTHER_MAX_ATR \
                    and abs(c[i] - o[i]) >= MOTHER_BODY * rng:
                mi, mh, ml = i, h[i], l[i]

    return out


# ----------------------------- TELEGRAM -----------------------------
def tg_send(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML",
               "disable_web_page_preview": True}
    for attempt in range(5):
        try:
            r = requests.post(url, data=payload, timeout=30)
            if r.status_code == 200:
                time.sleep(TG_DELAY)
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


def m(x):
    return f"{x:,.2f}"


def alert_msg(s):
    ce = s["side"] == 1
    icon = "🟢" if ce else "🔴"
    sign = 1 if ce else -1
    e, r = s["entry"], s["risk"]
    url = html.escape(f"https://www.tradingview.com/chart/?symbol={s['tv']}&interval=30", quote=True)
    vol = "n/a" if np.isnan(s["vr"]) else f"x{s['vr']:.1f}"
    c_ts, b_ts, m_ts = s["ts"], s["bk_ts"], s["m_ts"]
    return (
        f"{icon} <b>BUY {'CE' if ce else 'PE'}</b> | <b>{html.escape(s['name'])}</b> | 30m\n\n"
        f"🕒 Confirm candle: {c_ts:%d %b %Y} | {c_ts:%H:%M}-{cend(c_ts):%H:%M}\n"
        f"⚡ Breakout candle: {b_ts:%H:%M} ({b_ts:%d %b})\n"
        f"👩 Mother candle: {m_ts:%H:%M} ({m_ts:%d %b}) | H {m(s['mh'])} | L {m(s['ml'])}\n\n"
        f"💰 Entry (confirm close): <b>{m(e)}</b>\n"
        f"🛑 Stoploss: {m(s['sl'])}  (risk {m(r)})\n"
        f"🎯 T1 (1:1): {m(e + sign * r)}\n"
        f"🎯 T2 (1:2): {m(e + sign * 2 * r)}\n"
        f"🎯 T3 (1:3): {m(e + sign * 3 * r)}\n\n"
        f"⭐ Quality: {s['q']}/10 | Vol {vol} | Mother {s['mother_atr']:.1f} ATR\n"
        f"📈 <a href=\"{url}\">Open TradingView chart</a>"
    )


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
    return f"{s['name']}|{s['ts'].isoformat()}|{'CE' if s['side'] == 1 else 'PE'}"


def prev_working_day(today):
    d = today - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


# ----------------------------- MAIN -----------------------------
def main():
    if not TOKEN or not CHAT_ID:
        sys.exit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID secrets missing")

    now = datetime.now(IST)
    now_ts = pd.Timestamp(now)
    today = now.date()
    state = load_state()
    sent = set(state["sent"])
    sent_order = list(state["sent"])

    backfill = RUN_MODE == "backfill" or not state["initialized"]
    prev_day = prev_working_day(today)

    universe = build_universe()
    data = fetch_all(universe)
    if not data:
        sys.exit("No data fetched - exiting without changing state")

    all_sigs = []
    for u in universe:
        d15 = data.get(u["yf"])
        if d15 is None:
            continue
        d30 = closed_only(to_30m(d15), now_ts)
        if len(d30) == 0:
            continue
        dbg = DEBUG_SYMBOL and DEBUG_SYMBOL == u["name"]
        if dbg:
            print(f"--- DEBUG {u['name']} : last 30m candles (TradingView se match karo) ---")
            print(d30.tail(30).to_string())
        sigs = detect(u, d30)
        if dbg:
            print(f"--- DEBUG {u['name']} : all signals found ---")
            for s in sigs:
                print(key(s), "entry", round(s["entry"], 2), "sl", round(s["sl"], 2), "q", s["q"])
        all_sigs.extend(sigs)

    all_sigs.sort(key=lambda s: (s["ts"], s["name"]))

    back_ok = True
    back_keys = set()
    if backfill:
        back = [s for s in all_sigs if s["ts"].date() == prev_day and key(s) not in sent]
        back_keys = {key(s) for s in back}
        ce = sum(1 for s in back if s["side"] == 1)
        pe = len(back) - ce
        if back:
            back_ok = tg_send(f"📅 <b>{prev_day:%A, %d %b %Y}</b> — pichle working day ke signals (30m)\n"
                              f"Total {len(back)} | 🟢 CE {ce} | 🔴 PE {pe}")
            for s in back:
                if back_ok and tg_send(alert_msg(s)):
                    sent.add(key(s))
                    sent_order.append(key(s))
                else:
                    back_ok = False
        else:
            back_ok = tg_send(f"📅 <b>{prev_day:%A, %d %b %Y}</b> — koi signal nahi mila (30m).")

    live = [s for s in all_sigs
            if s["ts"].date() == today and key(s) not in sent and key(s) not in back_keys]
    for s in live:
        if tg_send(alert_msg(s)):
            sent.add(key(s))
            sent_order.append(key(s))

    if back_ok:
        state["initialized"] = True
    state["sent"] = sent_order
    save_state(state)
    print(f"Done. backfill={backfill} live_new={len(live)}")


if __name__ == "__main__":
    main()
