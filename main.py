"""
CPR by VickyBobby  ->  Pivot (P / black line) TOUCH alerts, 30 min, Telegram

Rules
  * Daily CPR from PREVIOUS trading day H/L/C  (P = (H+L+C)/3, BC = (H+L)/2, TC = 2P-BC)  -- same as Pine
  * A 30m candle "touches" P when  low <= P <= high
  * Alert only on the FIRST touch: previous candle (same day) must NOT have touched P
  * Direction (decided on candle close):  close above P -> BUY CE ,  close below P -> BUY PE
  * Quality score /10 : close distance from P (ATR), candle body+colour, EMA21/EMA50 trend,
                        volume, CPR width (narrow = trending day)

Run behaviour
  * First run (no state file) -> TODAY's signals so far, once
  * Next runs                 -> only NEW live signals of today

ENV
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   (required)
  MIN_QUALITY=6      -> send only signals with quality >= 6   (default 0 = send all)
  DEBUG=1            -> print all touches of last 3 days (to compare with TradingView)
  ONLY=SBIN,TCS      -> scan only these symbols
"""

import json
import os
import sys
import time
from bisect import bisect_left
from datetime import datetime, timedelta
from html import escape
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ===================== SETTINGS =====================
EMA_FAST = 21
EMA_SLOW = 50
ATR_LEN = 14
VOL_LEN = 20
NARROW_PCT = 0.5          # CPR width % below this = NARROW (trending)  (same as Pine default)
WIDE_PCT = 1.0            # above this = WIDE (sideways)
TOUCH_TOL_PCT = 0.0       # extra tolerance around P in % (0 = exact candle-range touch)
MIN_QUALITY = float(os.getenv("MIN_QUALITY", "0"))

FETCH_INTERVAL = "5m"     # 30m bars are built from 5m, anchored at 09:15 like TradingView (NSE)
PERIOD = "45d"
DAILY_PERIOD = "3mo"
STATE_FILE = "cpr_state.json"
BATCH = 25
IST = ZoneInfo("Asia/Kolkata")
CANDLE_MIN = 30
SESSION_END = (15, 30)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DEBUG = os.getenv("DEBUG", "0") == "1"
ONLY = {x.strip().upper() for x in os.getenv("ONLY", "").split(",") if x.strip()}

# ===================== SYMBOLS =====================
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
def to_30m(df):
    """5m -> 30m bars at 09:15, 09:45 ... 15:15 (same grid as TradingView NSE)."""
    df = df.between_time("09:15", "15:29")
    if df.empty:
        return None
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = df.resample("30min", origin="start_day", offset="9h15min", label="left", closed="left").agg(agg)
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out if not out.empty else None


def clean_intraday(df):
    if df is None or df.empty:
        return None
    try:
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    except KeyError:
        return None
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        return None
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["Volume"] = df["Volume"].fillna(0)
    return to_30m(df)


def clean_daily(df):
    """-> (sorted list of dates, ndarray [H, L, C])"""
    if df is None or df.empty:
        return None
    try:
        df = df[["High", "Low", "Close"]].dropna()
    except KeyError:
        return None
    if df.empty:
        return None
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(IST)
    rows = {}
    for d, h, l, c in zip(idx.date, df["High"].to_numpy(float), df["Low"].to_numpy(float), df["Close"].to_numpy(float)):
        rows[d] = (h, l, c)
    dates = sorted(rows)
    return dates, np.array([rows[d] for d in dates])


def daily_from_intraday(df):
    g = df.groupby(df.index.date)
    dates = sorted(g.groups)
    arr = np.array([[g.get_group(d)["High"].max(), g.get_group(d)["Low"].min(), g.get_group(d)["Close"].iloc[-1]]
                    for d in dates])
    return dates, arr


def drop_open_candle(df, now):
    """Keep only fully closed 30m candles."""
    keep = []
    for ts in df.index:
        close_t = ts + timedelta(minutes=CANDLE_MIN)
        sess_end = ts.replace(hour=SESSION_END[0], minute=SESSION_END[1], second=0, microsecond=0)
        keep.append(min(close_t, sess_end) <= now)
    return df[np.array(keep, dtype=bool)]


def download(tickers, period, interval):
    for attempt in range(3):
        try:
            return yf.download(tickers, period=period, interval=interval, group_by="ticker",
                               auto_adjust=False, threads=False, progress=False)
        except Exception as e:  # noqa
            print(f"download error {interval} ({attempt + 1}/3): {e}")
            time.sleep(3)
    return None


def get_frame(raw, sym):
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        return raw[sym] if sym in raw.columns.get_level_values(0) else None
    return raw


def fetch_all(universe):
    intraday, daily = {}, {}
    for i in range(0, len(universe), BATCH):
        chunk = universe[i:i + BATCH]
        tickers = [u["y"] for u in chunk]
        raw5 = download(tickers, PERIOD, FETCH_INTERVAL)
        rawd = download(tickers, DAILY_PERIOD, "1d")
        for u in chunk:
            try:
                df = clean_intraday(get_frame(raw5, u["y"]))
            except Exception as e:  # noqa
                print(f"parse error {u['name']}: {e}")
                df = None
            if df is None:
                try:
                    df = clean_intraday(yf.Ticker(u["y"]).history(period=PERIOD, interval=FETCH_INTERVAL, auto_adjust=False))
                except Exception:
                    df = None
            if df is None or len(df) < EMA_SLOW + 5:
                print(f"SKIP (no data): {u['name']} [{u['y']}]")
                continue
            intraday[u["y"]] = df
            try:
                dd = clean_daily(get_frame(rawd, u["y"]))
            except Exception:
                dd = None
            if dd is None:
                print(f"daily missing for {u['name']} -> using intraday-derived H/L/C")
                dd = daily_from_intraday(df)
            daily[u["y"]] = dd
    return intraday, daily


# ===================== CPR / PIVOT (same formulas as Pine) =====================
def calc_levels(h, l, c):
    p = (h + l + c) / 3
    bc = (h + l) / 2
    tc = (p - bc) + p
    return {
        "p": p, "tc": max(tc, bc), "bc": min(tc, bc),
        "r1": 2 * p - l, "s1": 2 * p - h,
        "r2": p + (h - l), "s2": p - (h - l),
        "r3": h + 2 * (p - l), "s3": l - 2 * (h - p),
        "pdh": h, "pdl": l,
    }


def levels_for_day(day, dates, arr):
    k = bisect_left(dates, day) - 1          # last daily row strictly BEFORE this day
    if k < 0:
        return None
    h, l, c = arr[k]
    return calc_levels(h, l, c)


# ===================== SCAN =====================
def quality(side, o, h, l, c, atr, ema_f, ema_s, vr, lv):
    p = lv["p"]
    pts, mx, notes = 0, 0, []

    # 1) close distance beyond P (in ATR)
    mx += 2
    dist = abs(c - p) / atr if (not np.isnan(atr) and atr > 0) else 0.0
    pts += 2 if dist >= 0.5 else 1 if dist >= 0.25 else 0
    notes.append(f"close {dist:.1f} ATR {'upar' if c > p else 'neeche'}")

    # 2) candle body + colour
    mx += 2
    rng = h - l
    bpct = abs(c - o) / rng * 100 if rng > 0 else 0.0
    color_ok = (c > o) if side == "CE" else (c < o)
    if color_ok:
        pts += 2 if bpct >= 60 else 1 if bpct >= 40 else 0
    notes.append(f"body {bpct:.0f}%" + ("" if color_ok else " (ulta color)"))

    # 3) trend (EMA21 / EMA50)
    mx += 2
    if side == "CE":
        t = int(c > ema_f) + int(ema_f > ema_s)
    else:
        t = int(c < ema_f) + int(ema_f < ema_s)
    pts += t
    notes.append("trend ✓" if t == 2 else "trend ½" if t == 1 else "trend ✗")

    # 4) volume (skipped for indices / no-volume data -> score is scaled)
    if vr is not None:
        mx += 2
        pts += 2 if vr >= 1.5 else 1 if vr >= 1.0 else 0
        notes.append(f"vol {vr:.1f}x")

    # 5) CPR width
    mx += 2
    width = abs(lv["tc"] - lv["bc"]) / p * 100 if p else 0.0
    if width < NARROW_PCT:
        pts += 2
        wtype = "NARROW"
    elif width <= WIDE_PCT:
        pts += 1
        wtype = "NORMAL"
    else:
        wtype = "WIDE"
    notes.append(f"CPR {wtype} {width:.2f}%")

    return int(round(pts / mx * 10)), notes, width, wtype


def pick_targets(side, close, lv):
    if side == "CE":
        cands = [("TC", lv["tc"]), ("R1", lv["r1"]), ("R2", lv["r2"]), ("R3", lv["r3"]), ("PDH", lv["pdh"])]
        cands = sorted([x for x in cands if x[1] > close], key=lambda x: x[1])
    else:
        cands = [("BC", lv["bc"]), ("S1", lv["s1"]), ("S2", lv["s2"]), ("S3", lv["s3"]), ("PDL", lv["pdl"])]
        cands = sorted([x for x in cands if x[1] < close], key=lambda x: -x[1])
    return cands[:3]


def scan_symbol(df, dd, is_index):
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    v = df["Volume"].to_numpy(float)
    times = list(df.index)
    days = [t.date() for t in times]
    n = len(df)
    d_dates, d_arr = dd

    ema_f = pine_ema(c, EMA_FAST)
    ema_s = pine_ema(c, EMA_SLOW)
    atr = pine_atr(h, l, c, ATR_LEN)
    vol_ma = pd.Series(v).rolling(VOL_LEN).mean().to_numpy()
    has_vol = not (is_index or v.sum() == 0)

    lv_cache = {}
    touched = [False] * n
    signals = []
    tol = TOUCH_TOL_PCT / 100.0

    for i in range(n):
        day = days[i]
        if day not in lv_cache:
            lv_cache[day] = levels_for_day(day, d_dates, d_arr)
        lv = lv_cache[day]
        if lv is None:
            continue
        p = lv["p"]
        touched[i] = (l[i] <= p * (1 + tol)) and (h[i] >= p * (1 - tol))
        if not touched[i]:
            continue

        same_day_prev = i > 0 and days[i - 1] == day
        if same_day_prev and touched[i - 1]:
            continue                                   # previous candle already touched P -> no alert

        # direction on candle close
        if c[i] > p:
            side = "CE"
        elif c[i] < p:
            side = "PE"
        else:
            side = "CE" if c[i] >= o[i] else "PE"

        # where did price come from?
        ref = c[i - 1] if same_day_prev else o[i]
        came_from_above = ref > p
        if came_from_above:
            kind = "Support bounce" if side == "CE" else "Breakdown"
        else:
            kind = "Breakout" if side == "CE" else "Resistance rejection"

        vr = None
        if has_vol and not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vr = v[i] / vol_ma[i]

        score, notes, width, wtype = quality(side, o[i], h[i], l[i], c[i], atr[i], ema_f[i], ema_s[i], vr, lv)
        sl = float(l[i]) if side == "CE" else float(h[i])
        signals.append({
            "side": side, "time": times[i], "close": float(c[i]), "p": float(p),
            "tc": float(lv["tc"]), "bc": float(lv["bc"]),
            "sl": sl, "targets": pick_targets(side, float(c[i]), lv),
            "kind": kind, "from_above": came_from_above,
            "score": score, "notes": notes, "width": width, "wtype": wtype,
        })
    return signals


# ===================== TELEGRAM =====================
def fmt_price(x):
    return f"{x:,.2f}"


def grade(score):
    if score >= 8:
        return "STRONG ⭐"
    if score >= 6:
        return "GOOD"
    if score >= 4:
        return "AVERAGE"
    return "WEAK"


def build_message(u, s):
    is_ce = s["side"] == "CE"
    icon = "🟢" if is_ce else "🔴"
    risk = abs(s["close"] - s["sl"])
    risk_pct = risk / s["close"] * 100 if s["close"] else 0
    link = f"https://in.tradingview.com/chart/?symbol={quote(u['tv'], safe='')}&interval=30"
    bar = "█" * s["score"] + "░" * (10 - s["score"])

    if s["targets"]:
        tg = " | ".join(f"T{k + 1} {nm} {fmt_price(px)}" for k, (nm, px) in enumerate(s["targets"]))
    else:
        tg = "—"
    approach = "upar se aayi" if s["from_above"] else "neeche se aayi"

    lines = [
        f"{icon} <b>BUY {s['side']}</b> — <b>{escape(u['name'])}</b>  (P touch)",
        f"🕒 <b>Candle:</b> {s['time'].strftime('%d %b %Y, %H:%M')} IST (30m)",
        f"⭐ <b>Quality: {s['score']}/10</b> {grade(s['score'])}  {bar}",
        f"🎯 Setup: {s['kind']} (price {approach})",
        f"⚫ P (pivot): {fmt_price(s['p'])}  | TC {fmt_price(s['tc'])} | BC {fmt_price(s['bc'])}",
        f"💰 Entry (candle close): {fmt_price(s['close'])}",
        f"🛑 SL: {fmt_price(s['sl'])}  (risk {fmt_price(risk)} | {risk_pct:.2f}%)",
        f"🎯 {tg}",
        "🔎 " + " • ".join(s["notes"]),
        f"📈 <a href=\"{link}\">TradingView chart</a>",
        "<i>Levels underlying price ke hain, option premium ke nahi.</i>",
    ]
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
    print(f"Now (IST): {now:%Y-%m-%d %H:%M} | symbols: {len(universe)} | MIN_QUALITY={MIN_QUALITY:g}")

    intraday, daily = fetch_all(universe)
    print(f"Fetched data for {len(intraday)}/{len(universe)} symbols")
    if not intraday:
        print("No data fetched at all — exiting without touching state.")
        send_telegram("⚠️ <b>CPR scanner</b>: Yahoo se data nahi mila (rate-limit/blocked ho sakta hai). Agli run mein dobara try hoga.")
        sys.exit(1)

    all_signals = []
    all_dates = set()
    for u in universe:
        df = intraday.get(u["y"])
        if df is None:
            continue
        df = drop_open_candle(df, now)
        if len(df) < EMA_SLOW + 5:
            continue
        all_dates.update(df.index.date)
        for s in scan_symbol(df, daily[u["y"]], u["index"]):
            all_signals.append((u, s))
    all_signals.sort(key=lambda x: (x[1]["time"], x[0]["name"]))

    if all_dates:
        print(f"Latest closed candle date in data: {max(all_dates)}")
    by_day = {}
    for _, s_ in all_signals:
        by_day[s_["time"].date()] = by_day.get(s_["time"].date(), 0) + 1
    print("P-touch signals per day (last 5):", dict(sorted(by_day.items())[-5:]))

    if DEBUG:
        d3 = sorted(all_dates)[-3:]
        print("---- DEBUG: signals of last 3 trading days ----")
        for u, s in all_signals:
            if s["time"].date() in d3:
                print(f"{u['name']:<12} BUY {s['side']} {s['time']:%d-%b %H:%M} P {s['p']:.2f} close {s['close']:.2f} Q {s['score']}/10 {s['kind']}")

    state = load_state()
    first_run = not state.get("initialized")
    sent = set(state.get("sent", []))

    if first_run:
        for u, s in all_signals:                       # everything existing now = seen
            sent.add(sig_key(u, s))
        to_send = [(u, s) for u, s in all_signals if s["time"].date() == today and s["score"] >= MIN_QUALITY]
        send_telegram(f"📌 <b>CPR P-touch scanner</b>\nAaj ({today:%d %b %Y}) ke ab tak ke signals: <b>{len(to_send)}</b>\n"
                      f"Iske baad sirf live signals aayenge.")
        mark_after_send = False
    else:
        to_send = [(u, s) for u, s in all_signals
                   if s["time"].date() == today and sig_key(u, s) not in sent and s["score"] >= MIN_QUALITY]
        mark_after_send = True

    print(f"Signals to send: {len(to_send)} (first_run={first_run})")
    if not first_run and not to_send:
        print("Not first run -> only NEW signals of TODAY are sent; none found.")

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
