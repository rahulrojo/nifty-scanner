"""
ROMYO Confirmed Mother Candle Breakout  (30m)  ->  Telegram scanner
Line-by-line port of the Pine v6 script.

Pine logic (per closed candle, in this order):
  1. expire   : waitingBreakout and bar - motherBar > maxWait  -> reset setup
  2. mother   : candle range 1.0-2.5 ATR and body >= 50%, only if no setup active
  3. breakout : first CLOSE outside mother (prev close still inside), volume >= avg(20),
                EMA21 trend filter on the BREAKOUT candle close
  4. confirm  : very next candle: close AND low > mother high (CE)  /  close AND high < mother low (PE)
  Entry = confirm close, SL = opposite side of mother candle.

Candles: NSE 30m grid 09:15, 09:45 ... 15:15 (last candle 15 min), same as TradingView.

Modes (RUN_MODE): auto  = first run previous working day + today, then only new live signals
                  backfill = previous working day dobara
                  today = aaj ke saare signals dobara (TradingView se match karne ke liye)
                  test = sirf Telegram connection test (data download nahi hota)
DEBUG_SYMBOL=PIDILITIND -> logs me 30m candles + EMA/ATR/VolMA print hota hai.

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
RUN_MODE = os.environ.get("RUN_MODE", "auto").strip().lower()      # auto | backfill | today
DEBUG_SYMBOL = os.environ.get("DEBUG_SYMBOL", "").strip().upper()
STATE_FILE = "state_romyo_mc30.json"

# ---- Pine inputs (same defaults) ----
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

# Indices have no volume on Yahoo. In Pine, volume = na -> volumeOK = false (no signals).
# Set True only if TradingView shows signals on indices for you.
INDEX_NO_VOLUME_PASSES = False

TF_MIN = 30
PERIOD = "45d"
TG_DELAY = 3.1
MAX_SENT_KEYS = 3000
TG_ERRORS = []          # Telegram failures -> run red ho jayegi, reason logs me

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


# ----------------------------- PINE INDICATORS -----------------------------
def pine_rma(x, n):
    """ta.rma : seeded with SMA(n), then Wilder smoothing."""
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.mean(x[:n])
    a = 1.0 / n
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def pine_ema(x, n):
    """ta.ema : seeded with SMA(n)."""
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.mean(x[:n])
    a = 2.0 / (n + 1)
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def indicators(d):
    h = d["High"].values.astype(float)
    l = d["Low"].values.astype(float)
    c = d["Close"].values.astype(float)
    v = d["Volume"].values.astype(float)
    tr = h - l
    tr[1:] = np.maximum(tr[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = pine_rma(tr, ATR_LEN)
    ema = pine_ema(c, EMA_LEN)
    vsma = pd.Series(v).rolling(VOL_LEN).mean().values
    return ema, atr, vsma


# ----------------------------- STRATEGY (exact Pine port) -----------------------------
def quality(vr, mother_atr, dist_ema_atr, bo_body, bo_beyond_atr, gap_atr):
    """Extra score only for the alert (does NOT affect which signals fire).
    volume 3, mother size 2, EMA distance 2, breakout body 1, breakout strength 1, confirm gap 1."""
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
    ema, atr, vsma = indicators(d)
    idx = d.index
    has_vol = float(np.nansum(v)) > 0

    # state (Pine var)
    m_high = m_low = np.nan
    m_bar = -10 ** 6
    waiting_bo = False
    waiting_conf = False
    bo_dir = 0
    bo_bar = -10 ** 6
    bo_level = bo_sl = np.nan

    out = []
    for i in range(1, n):
        a = atr[i]
        rng = h[i] - l[i]
        body = abs(c[i] - o[i])
        body_pct = body / rng * 100 if rng > 0 else 0.0
        cand = (not np.isnan(a)) and rng >= a * MIN_MOTHER_ATR and rng <= a * MAX_MOTHER_ATR \
            and body_pct >= MIN_BODY_PCT

        # 1) expire old setup
        if waiting_bo and i - m_bar > MAX_WAIT:
            waiting_bo = False
            waiting_conf = False
            bo_dir = 0

        # 2) new mother candle (only when no setup is active)
        if cand and not waiting_bo and not waiting_conf:
            m_high, m_low, m_bar = h[i], l[i], i
            waiting_bo = True

        # filters
        if not USE_VOLUME:
            vol_ok = True
        elif not has_vol:
            vol_ok = INDEX_NO_VOLUME_PASSES
        else:
            vol_ok = (not np.isnan(vsma[i])) and v[i] >= vsma[i] * VOL_MULT
        bull_ok = (not USE_EMA) or (not np.isnan(ema[i]) and c[i] > ema[i])
        bear_ok = (not USE_EMA) or (not np.isnan(ema[i]) and c[i] < ema[i])

        # 3) first close outside the mother candle
        break_up = waiting_bo and i > m_bar and c[i] > m_high and c[i - 1] <= m_high
        break_dn = waiting_bo and i > m_bar and c[i] < m_low and c[i - 1] >= m_low

        if break_up and vol_ok and bull_ok:
            waiting_bo = False
            waiting_conf = True
            bo_dir, bo_bar, bo_level, bo_sl = 1, i, m_high, m_low
        if break_dn and vol_ok and bear_ok:
            waiting_bo = False
            waiting_conf = True
            bo_dir, bo_bar, bo_level, bo_sl = -1, i, m_low, m_high

        # 4) confirmation: very next candle must close AND stay beyond the mother level
        if waiting_conf and i == bo_bar + 1:
            buy_ce = bo_dir == 1 and c[i] > bo_level and l[i] > bo_level
            buy_pe = bo_dir == -1 and c[i] < bo_level and h[i] < bo_level
            if buy_ce or buy_pe:
                side = 1 if buy_ce else -1
                entry = float(c[i])
                risk = abs(entry - bo_sl)
                bk = bo_bar
                if risk > 0 and not np.isnan(a) and atr[bk] > 0 and atr[m_bar] > 0:
                    vr = (v[bk] / vsma[bk]) if (has_vol and not np.isnan(vsma[bk]) and vsma[bk] > 0) else np.nan
                    mother_atr = (m_high - m_low) / atr[m_bar]
                    rng_b = max(h[bk] - l[bk], 1e-9)
                    bo_body = abs(c[bk] - o[bk]) / rng_b
                    beyond = ((c[bk] - m_high) if side == 1 else (m_low - c[bk])) / atr[bk]
                    gap = ((l[i] - bo_level) if side == 1 else (bo_level - h[i])) / a
                    dist = abs(c[i] - ema[i]) / a
                    q = quality(vr, mother_atr, dist, bo_body, beyond, gap)
                    out.append({
                        "name": u["name"], "tv": u["tv"], "side": side,
                        "ts": idx[i], "bk_ts": idx[bk], "m_ts": idx[m_bar],
                        "mh": float(m_high), "ml": float(m_low),
                        "entry": entry, "sl": float(bo_sl), "risk": float(risk),
                        "q": q, "vr": vr, "mother_atr": float(mother_atr),
                    })
            waiting_conf = False

    return out


# ----------------------------- TELEGRAM -----------------------------
def tg_send(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML",
               "disable_web_page_preview": True}
    last_err = ""
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
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            print("Telegram error:", last_err)
            break
        except Exception as e:
            last_err = str(e)
            print("Telegram exception:", e)
            time.sleep(3)
    TG_ERRORS.append(last_err or "unknown Telegram error")
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
        sys.exit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID secrets missing "
                 "(Settings > Secrets and variables > Actions)")

    now = datetime.now(IST)
    now_ts = pd.Timestamp(now)
    today = now.date()
    print(f"pandas {pd.__version__} | mode={RUN_MODE} | now={now:%Y-%m-%d %H:%M} IST")

    # quick Telegram connection test (no data download)
    if RUN_MODE == "test":
        if tg_send("✅ Telegram test OK — ROMYO Mother Candle scanner (30m) connected"):
            print("Telegram OK")
            return
        sys.exit("Telegram send failed: " + " | ".join(TG_ERRORS))

    state = load_state()
    sent = set(state["sent"])
    sent_order = list(state["sent"])

    today_mode = RUN_MODE == "today"
    backfill = RUN_MODE == "backfill" or (RUN_MODE == "auto" and not state["initialized"])
    prev_day = prev_working_day(today)

    universe = build_universe()
    data = fetch_all(universe)
    if len(data) < max(5, len(universe) // 2):
        tg_send(f"⚠️ Yahoo data problem: sirf {len(data)}/{len(universe)} symbols ka data mila. "
                f"Is run me scan skip hua.")
        sys.exit(f"Not enough data: {len(data)}/{len(universe)}")

    all_sigs = []
    last_candle = None
    for u in universe:
        d15 = data.get(u["yf"])
        if d15 is None:
            continue
        d30 = closed_only(to_30m(d15), now_ts)
        if len(d30) == 0:
            continue
        if last_candle is None or d30.index[-1] > last_candle:
            last_candle = d30.index[-1]
        dbg = bool(DEBUG_SYMBOL) and DEBUG_SYMBOL == u["name"]
        if dbg:
            ema, atr, vsma = indicators(d30)
            show = d30.copy()
            show["EMA21"] = ema
            show["ATR14"] = atr
            show["VolMA20"] = vsma
            print(f"--- DEBUG {u['name']} : last 30m candles (TradingView se match karo) ---")
            print(show.tail(30).round(2).to_string())
        sigs = detect(u, d30)
        if dbg:
            print(f"--- DEBUG {u['name']} : all signals found ---")
            for s in sigs:
                print(key(s), "entry", round(s["entry"], 2), "sl", round(s["sl"], 2), "q", s["q"])
        all_sigs.extend(sigs)

    all_sigs.sort(key=lambda s: (s["ts"], s["name"]))
    status = (f"Data {len(data)}/{len(universe)} | last candle "
              f"{last_candle:%H:%M, %d %b}" if last_candle is not None else "koi candle nahi mili")
    print(status)

    # ---- previous working day (first run / manual backfill) ----
    back_ok = True
    back_keys = set()
    if backfill:
        back = [s for s in all_sigs if s["ts"].date() == prev_day and key(s) not in sent]
        back_keys = {key(s) for s in back}
        ce = sum(1 for s in back if s["side"] == 1)
        pe = len(back) - ce
        head = (f"📅 <b>{prev_day:%A, %d %b %Y}</b> — pichle working day ke signals (30m)\n"
                f"Total {len(back)} | 🟢 CE {ce} | 🔴 PE {pe}\n🛰 {status}")
        back_ok = tg_send(head)
        for s in back:
            if back_ok and tg_send(alert_msg(s)):
                sent.add(key(s))
                sent_order.append(key(s))
            else:
                back_ok = False

    # ---- today ----
    live = [s for s in all_sigs
            if s["ts"].date() == today and key(s) not in back_keys
            and (today_mode or key(s) not in sent)]
    if today_mode:
        ce = sum(1 for s in live if s["side"] == 1)
        tg_send(f"📅 <b>Aaj, {today:%d %b %Y}</b> — ab tak ke saare signals (30m)\n"
                f"Total {len(live)} | 🟢 CE {ce} | 🔴 PE {len(live) - ce}\n🛰 {status}")
    for s in live:
        if tg_send(alert_msg(s)):
            if key(s) not in sent:
                sent.add(key(s))
                sent_order.append(key(s))

    if back_ok:
        state["initialized"] = True
    state["sent"] = sent_order
    save_state(state)
    print(f"Done. mode={RUN_MODE} backfill={backfill} today_sent={len(live)}")

    if TG_ERRORS:
        sys.exit("Telegram errors: " + " | ".join(sorted(set(TG_ERRORS))))


if __name__ == "__main__":
    main()
