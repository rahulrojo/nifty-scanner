#!/usr/bin/env python3
"""
Squeeze Signal Level Mapper  ->  Telegram alerts (30m, NSE F&O stocks + indexes)

Env vars (GitHub Secrets):
  TELEGRAM_BOT_TOKEN   : BotFather wala token
  TELEGRAM_CHAT_ID     : channel id (e.g. -1001234567890) ya @channelusername
                         (bot ko channel ka admin banana zaroori hai)
"""
import os
import json
import time
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")
try:
    yf.set_tz_cache_location("/tmp/yfcache")   # "database is locked" error se bachne ke liye
except Exception:
    pass
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
STATE_FILE = "state.json"

INTERVAL = "5m"         # 5m data lete hain, phir 9:15 se align karke 30m bante hain (TradingView jaisa)
PERIOD = "20d"          # warm-up ke liye (EMA/ATR seed settle ho jaye)
BAR_MIN = 30

# ---- Pine inputs ----
LENGTH = 20
MULT_BB = 2.0
MULT_KC = 1.5

# ---- INDEXES: display name -> (yahoo symbol, tradingview symbol) ----
INDEXES = {
    "NIFTY 50":        ("^NSEI",                    "NSE:NIFTY"),
    "BANKNIFTY":       ("^NSEBANK",                 "NSE:BANKNIFTY"),
    "SENSEX":          ("^BSESN",                   "BSE:SENSEX"),
    "FINNIFTY":        ("NIFTY_FIN_SERVICE.NS",     "NSE:CNXFINANCE"),
    "MIDCPNIFTY":      ("NIFTY_MID_SELECT.NS",      "NSE:NIFTY_MID_SELECT"),
    "NIFTY NEXT 50":   ("^NSMIDCP",                 "NSE:NIFTYJR"),
    "NIFTY IT":        ("^CNXIT",                   "NSE:CNXIT"),
    "NIFTY AUTO":      ("^CNXAUTO",                 "NSE:CNXAUTO"),
    "NIFTY PHARMA":    ("^CNXPHARMA",               "NSE:CNXPHARMA"),
    "NIFTY METAL":     ("^CNXMETAL",                "NSE:CNXMETAL"),
    "NIFTY ENERGY":    ("^CNXENERGY",               "NSE:CNXENERGY"),
    "NIFTY FMCG":      ("^CNXFMCG",                 "NSE:CNXFMCG"),
    "NIFTY REALTY":    ("^CNXREALTY",               "NSE:CNXREALTY"),
    "NIFTY PSU BANK":  ("^CNXPSUBANK",              "NSE:CNXPSUBANK"),
    "NIFTY MEDIA":     ("^CNXMEDIA",                "NSE:CNXMEDIA"),
}

# ---- 100 high-volatility F&O stocks (apni marzi se edit kar sakte ho) ----
STOCKS = [
    "ADANIENT", "ADANIPORTS", "ADANIGREEN", "ADANIPOWER", "ANGELONE", "APLAPOLLO", "ASHOKLEY", "AUROPHARMA", "AXISBANK", "BANDHANBNK",
    "BANKBARODA", "BANKINDIA", "BEL", "BHEL", "BIOCON", "BSE", "CANBK", "CDSL", "CGPOWER", "COALINDIA",
    "COFORGE", "CROMPTON", "CUMMINSIND", "DELHIVERY", "DIXON", "DLF", "ETERNAL", "EXIDEIND", "FEDERALBNK", "GAIL",
    "GLENMARK", "GMRAIRPORT", "GRANULES", "HAL", "HINDALCO", "HINDCOPPER", "HUDCO", "IDFCFIRSTB", "IEX", "INDHOTEL",
    "INDIGO", "INDUSINDBK", "INOXWIND", "IRCTC", "IREDA", "IRFC", "JINDALSTEL", "JSWENERGY", "JSWSTEEL", "JUBLFOOD",
    "KALYANKJIL", "KEI", "KPITTECH", "LAURUSLABS", "LICHSGFIN", "LODHA", "LTF", "LUPIN", "MANAPPURAM", "MAZDOCK",
    "MCX", "MOTHERSON", "MPHASIS", "MUTHOOTFIN", "NATIONALUM", "NBCC", "NCC", "NHPC", "NMDC", "NYKAA",
    "OIL", "PAYTM", "PERSISTENT", "PETRONET", "PFC", "PIIND", "PNB", "POLICYBZR", "POLYCAB", "PRESTIGE",
    "RBLBANK", "RECLTD", "SAIL", "SBIN", "SHRIRAMFIN", "SIEMENS", "SOLARINDS", "SONACOMS", "SUPREMEIND", "SUZLON",
    "TATAELXSI", "TATAPOWER", "TATASTEEL", "TITAGARH", "TRENT", "TVSMOTOR", "UNIONBANK", "UNOMINDA", "VEDL", "YESBANK",
]


def build_universe():
    """name -> (yahoo, tradingview)"""
    uni = dict(INDEXES)
    for s in STOCKS:
        tv = "NSE:" + s.replace("-", "_").replace("&", "_")
        uni[s] = (s + ".NS", tv)
    return uni


# ------------------------------------------------------------------ data
def fetch_all(uni):
    out = {}
    names = list(uni.keys())
    for i in range(0, len(names), 20):
        chunk = names[i:i + 20]
        ysyms = [uni[n][0] for n in chunk]
        try:
            raw = yf.download(ysyms, period=PERIOD, interval=INTERVAL, group_by="ticker",
                              auto_adjust=False, progress=False, threads=True)
        except Exception as e:
            print(f"[warn] download fail {chunk}: {e}")
            continue
        for n in chunk:
            ys = uni[n][0]
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if ys not in raw.columns.get_level_values(0):
                        continue
                    df = raw[ys]
                else:
                    df = raw
                df = df[["Open", "High", "Low", "Close"]].dropna()
                if df.empty:
                    continue
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                df.index = df.index.tz_convert(IST)
                # NSE ke 30m candles: 9:15, 9:45 ... 15:15 (last candle sirf 15 min ka)
                df = df.resample("30min", origin="start_day", offset="9h15min",
                                 label="left", closed="left").agg(
                    {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
                if df.empty:
                    continue
                out[n] = df
            except Exception as e:
                print(f"[warn] {n}: {e}")
        time.sleep(1)
    return out


# ------------------------------------------------------------------ indicator
def linreg_last(series, n):
    """ta.linreg(src, n, 0)"""
    x = np.arange(n)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()

    def f(y):
        ym = y.mean()
        slope = ((x - xm) * (y - ym)).sum() / denom
        return ym + slope * ((n - 1) - xm)

    return series.rolling(n).apply(f, raw=True)


def analyze(name, df, today):
    """Pine logic ko bar-by-bar chalata hai. Sirf aaj ke events return karta hai."""
    c, h, l, o = df["Close"], df["High"], df["Low"], df["Open"]

    mid = c.rolling(LENGTH).mean()
    sd = c.rolling(LENGTH).std(ddof=0)           # Pine stdev = population
    bb_u = mid + MULT_BB * sd
    bb_l = mid - MULT_BB * sd

    ema = c.ewm(span=LENGTH, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / LENGTH, adjust=False).mean()   # RMA
    kc_u = ema + atr * MULT_KC
    kc_l = ema - atr * MULT_KC

    squeezed = ((bb_u < kc_u) & (bb_l > kc_l)).values

    hh = h.rolling(LENGTH).max()
    ll = l.rolling(LENGTH).min()
    mom = linreg_last(c - ((hh + ll) / 2 + ema) / 2, LENGTH).values

    C, O, H, Lw, E = c.values, o.values, h.values, l.values, ema.values
    idx = df.index

    events = []
    sq_no = 0
    lvl_hi = lvl_lo = None
    lvl_no = 0

    for i in range(1, len(df)):
        ts = idx[i]
        is_today = ts.date() == today
        if ts.date() < today:
            sq_no = 0  # numbering sirf aaj ki shuruaat se
        start = squeezed[i] and not squeezed[i - 1]
        release = squeezed[i - 1] and not squeezed[i]

        if start and is_today:
            sq_no += 1
        if start:
            lvl_hi, lvl_lo, lvl_no = H[i], Lw[i], sq_no

        def ev(kind, **kw):
            if is_today:
                events.append(dict(name=name, ts=ts, kind=kind, close=C[i], **kw))

        if start:
            ev("SQ_START", sq_no=sq_no, hi=H[i], lo=Lw[i])

        m = mom[i]
        if release and not np.isnan(m):
            if m > 0 and C[i] > E[i]:
                ev("BUY", sq_no=sq_no)
            elif m < 0 and C[i] < E[i]:
                ev("SELL", sq_no=sq_no)

        if lvl_hi is not None and C[i] > O[i] and C[i] > lvl_hi:
            ev("SQ_BUY", sq_no=lvl_no, hi=lvl_hi, lo=lvl_lo)
            lvl_hi = lvl_lo = None
        elif lvl_lo is not None and C[i] < O[i] and C[i] < lvl_lo:
            ev("SQ_SELL", sq_no=lvl_no, hi=lvl_hi, lo=lvl_lo)
            lvl_hi = lvl_lo = None

    return events


# ------------------------------------------------------------------ telegram
def sq_label(n):
    return f"#{n}" if n and n > 0 else "(pichle din se chalu)"


def bar_end(t0):
    """Candle ka close time (market 15:30 par band hota hai)"""
    return min(t0 + timedelta(minutes=BAR_MIN), t0.replace(hour=15, minute=30, second=0, microsecond=0))


def fmt(e, tv):
    t0 = e["ts"]
    t1 = bar_end(t0)
    when = f"{t0.strftime('%H:%M')}–{t1.strftime('%H:%M')} IST | {t0.strftime('%d %b %Y')}"
    link = f"https://www.tradingview.com/chart/?symbol={tv}&interval={BAR_MIN}"
    k = e["kind"]
    if k == "SQ_START":
        head = f"🟡 <b>SQUEEZE SHURU — Squeeze {sq_label(e['sq_no'])}</b>"
        body = f"High: {e['hi']:.2f} | Low: {e['lo']:.2f}\n(Dashed levels: isi candle ka High/Low)"
    elif k == "SQ_BUY":
        head = f"🟢 <b>BUY — Squeeze High Break</b> (Squeeze {sq_label(e['sq_no'])})"
        body = f"Close: {e['close']:.2f} > High level {e['hi']:.2f}"
    elif k == "SQ_SELL":
        head = f"🔴 <b>SELL — Squeeze Low Break</b> (Squeeze {sq_label(e['sq_no'])})"
        body = f"Close: {e['close']:.2f} < Low level {e['lo']:.2f}"
    elif k == "BUY":
        head = f"⚡🟢 <b>SQUEEZE RELEASE — BUY</b> (Squeeze {sq_label(e['sq_no'])})"
        body = f"Momentum +ve, Close > EMA | Close: {e['close']:.2f}"
    else:
        head = f"⚡🔴 <b>SQUEEZE RELEASE — SELL</b> (Squeeze {sq_label(e['sq_no'])})"
        body = f"Momentum -ve, Close < EMA | Close: {e['close']:.2f}"
    body = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f"{head}\n📌 <b>{e['name']}</b>\n🕒 Candle: {when}\n{body}\n"
            f"📈 <a href=\"{link}\">TradingView chart</a>")


def send(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML",
               "disable_web_page_preview": True}
    for _ in range(5):
        try:
            r = requests.post(url, json=payload, timeout=30)
        except Exception as ex:
            print(f"[warn] telegram error: {ex}")
            time.sleep(5)
            continue
        if r.status_code == 200:
            return True
        if r.status_code == 429:
            wait = r.json().get("parameters", {}).get("retry_after", 10)
            print(f"[info] rate limit, {wait}s wait")
            time.sleep(wait + 1)
            continue
        print(f"[err] telegram {r.status_code}: {r.text}")
        if r.status_code == 400 and payload.get("parse_mode"):
            # HTML parse fail: plain text mein bhej do (alert miss na ho)
            import re
            plain = re.sub(r"<[^>]+>", "", text).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
            payload = {"chat_id": CHAT_ID, "text": plain, "disable_web_page_preview": True}
            continue
        return False
    return False


# ------------------------------------------------------------------ state
def load_state(today_str):
    try:
        with open(STATE_FILE) as f:
            st = json.load(f)
        if st.get("date") == today_str:
            return st
    except Exception:
        pass
    return {"date": today_str, "sent": []}


def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f)


# ------------------------------------------------------------------ main
def main():
    if not TOKEN or not CHAT_ID:
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID set nahi hai")
        sys.exit(1)

    print("[info] VERSION v5: html-escape + plain fallback + 5m->30m resample + cache fix")
    now = datetime.now(IST)
    today = now.date()
    state = load_state(today.isoformat())
    sent = set(state["sent"])

    uni = build_universe()
    data = fetch_all(uni)
    print(f"[info] data mila: {len(data)}/{len(uni)} symbols")

    all_events = []
    for name, df in data.items():
        # sirf complete (close ho chuki) candles
        df = df[[bar_end(t) <= now + timedelta(minutes=1) for t in df.index]]
        if len(df) < LENGTH * 2:
            continue
        try:
            all_events += analyze(name, df, today)
        except Exception as e:
            print(f"[warn] analyze {name}: {e}")

    all_events.sort(key=lambda e: (e["ts"], e["name"]))
    new = [e for e in all_events if f"{e['name']}|{e['ts'].isoformat()}|{e['kind']}" not in sent]
    print(f"[info] aaj ke events: {len(all_events)}, naye: {len(new)}")

    for e in new:
        key = f"{e['name']}|{e['ts'].isoformat()}|{e['kind']}"
        if send(fmt(e, uni[e["name"]][1])):
            sent.add(key)
            state["sent"] = sorted(sent)
            save_state(state)
        time.sleep(3.2)   # Telegram limit (~20 msg/min channel)

    state["sent"] = sorted(sent)
    save_state(state)


if __name__ == "__main__":
    main()
