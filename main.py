"""
Mother Candle Breakout | BUY CE / BUY PE  -> 30 min Telegram scanner
Pine v5 indicator ka Python port (Auto mode).

Secrets (GitHub -> Settings -> Secrets -> Actions):
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  (channel id jaise -1001234567890)
"""
import os
import json
import time
import html
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

IST = "Asia/Kolkata"
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
STATE_FILE = "state.json"
FIRST_RUN_SESSIONS = int(os.getenv("FIRST_RUN_SESSIONS", "2"))  # pehli run: pichla working day + aaj

# ===== Pine inputs (defaults same as indicator) =====
MIN_ATR_MULT = 1.0
MAX_ATR_MULT = 2.5
MIN_BODY_PCT = 50
MIN_INSIDE = 1
MAX_BARS = 8
MOTHER_SESS = (9 * 60 + 15, 13 * 60 + 30)   # 0915-1330 (bar start time)
SIG_SESS = (9 * 60 + 15, 14 * 60 + 30)      # 0915-1430 (bar start time)
BRK_BODY_PCT = 40
MAX_BRK_ATR = 2.0
EMA_LEN = 21
MIN_SCORE = 2
RR1 = 1.0
RR2 = 2.0
MAX_RISK_ATR = 2.0
TIME_STOP = 6
# useClose=True, bufPct=0, fullyOutside=False, keepAlive=True, SL = Mother Midpoint, trend+VWAP filter ON

# ===== Symbols =====
STOCKS = """ADANIENT ADANIPORTS ADANIGREEN APOLLOHOSP ASHOKLEY AUBANK AUROPHARMA AXISBANK BAJAJ-AUTO
BAJFINANCE BAJAJFINSV BANDHANBNK BANKBARODA BEL BHEL BHARTIARTL BIOCON BPCL CANBK CHOLAFIN CIPLA
COALINDIA COFORGE DIVISLAB DIXON DLF DRREDDY EICHERMOT FEDERALBNK GAIL GLENMARK GODREJPROP GRASIM
HAL HAVELLS HCLTECH HDFCBANK HEROMOTOCO HINDALCO HINDPETRO ICICIBANK IDFCFIRSTB IEX INDHOTEL INDIGO
INDUSINDBK INFY IOC IRCTC IRFC ITC JINDALSTEL JIOFIN JSWSTEEL JUBLFOOD KOTAKBANK LAURUSLABS LODHA LT
LUPIN M&M MANAPPURAM MARUTI MAXHEALTH MCX MUTHOOTFIN NATIONALUM NAUKRI NMDC NTPC OBEROIRLTY ONGC
PERSISTENT PFC PIDILITIND PNB POLYCAB POWERGRID RECLTD RELIANCE SAIL SBILIFE SBIN SHRIRAMFIN SIEMENS
SUNPHARMA TATAPOWER TATASTEEL TCS TECHM TITAN TORNTPHARM TRENT TVSMOTOR ULTRACEMCO UNIONBANK VEDL
WIPRO YESBANK ETERNAL""".split()

# name: (yfinance ticker, tradingview symbol)
INDEXES = {
    "NIFTY": ("^NSEI", "NSE:NIFTY"),
    "BANKNIFTY": ("^NSEBANK", "NSE:BANKNIFTY"),
    "FINNIFTY": ("NIFTY_FIN_SERVICE.NS", "NSE:CNXFINANCE"),
    "MIDCPNIFTY": ("NIFTY_MID_SELECT.NS", "NSE:NIFTY_MID_SELECT"),
    "SENSEX": ("^BSESN", "BSE:SENSEX"),
}


def build_universe():
    u = {}
    for s in STOCKS:
        tv = "NSE:" + s.replace("&", "_").replace("-", "_")
        u[s] = (s + ".NS", tv)
    u.update(INDEXES)
    return u


# ===== Data =====
def to_30m(raw):
    """15m data -> NSE 30m candles (09:15, 09:45 ... 15:15), sirf poori bani candles."""
    d = raw.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if d.empty:
        return None
    idx = pd.to_datetime(d.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    d.index = idx.tz_convert(IST)
    d = d.between_time("09:15", "15:29")
    if d.empty:
        return None
    if "Volume" not in d:
        d["Volume"] = 0.0
    d["Volume"] = d["Volume"].fillna(0)
    g = d.resample("30min", offset="15min")
    out = g.agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    out["n"] = g["Close"].count()
    out = out[out["n"] > 0]
    last_slot = (out.index.hour == 15) & (out.index.minute == 15)
    need = np.where(last_slot, 1, 2)          # 15:15 candle me sirf ek 15m bar hota hai
    out = out[out["n"].values >= need]
    last_slot = (out.index.hour == 15) & (out.index.minute == 15)
    ends = out.index + pd.Timedelta(minutes=30)
    ends = pd.DatetimeIndex(np.where(last_slot, out.index + pd.Timedelta(minutes=15), ends))
    now = pd.Timestamp.now(tz=IST)
    out = out[ends <= now]                      # chalu (adhuri) candle hata do
    out["End"] = ends[ends <= now]
    return out if len(out) > 40 else None


def download_all(universe):
    tickers = [v[0] for v in universe.values()]
    rev = {v[0]: k for k, v in universe.items()}
    frames = {}
    for i in range(0, len(tickers), 25):
        chunk = tickers[i:i + 25]
        for attempt in range(3):
            try:
                data = yf.download(chunk, period="30d", interval="15m", group_by="ticker",
                                   auto_adjust=False, threads=True, progress=False)
                break
            except Exception as e:
                print("download retry", e)
                time.sleep(3)
        else:
            continue
        for t in chunk:
            try:
                df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                f = to_30m(df)
                if f is not None:
                    frames[rev[t]] = f
            except Exception:
                pass
        time.sleep(1)
    return frames


# ===== Indicators =====
def rma(arr, n):
    out = np.full(len(arr), np.nan)
    if len(arr) < n:
        return out
    out[n - 1] = np.nanmean(arr[:n])
    for i in range(n, len(arr)):
        out[i] = (out[i - 1] * (n - 1) + arr[i]) / n
    return out


def nz(x):
    return 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else x


def scan_symbol(df, dbg=False):
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    v = df["Volume"].values.astype(float)
    n = len(df)
    days = np.array([t.date() for t in df.index])
    mins = np.array([t.hour * 60 + t.minute for t in df.index])

    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = rma(tr, 14)
    ema = pd.Series(c).ewm(span=EMA_LEN, adjust=False).mean().values
    hlc3 = (h + l + c) / 3
    pv = pd.Series(hlc3 * v).groupby(days).cumsum().values
    cv = pd.Series(v).groupby(days).cumsum().values
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(cv > 0, pv / cv, np.nan)
    avgvol = pd.Series(v).rolling(20).mean().values

    in_mother = (mins >= MOTHER_SESS[0]) & (mins < MOTHER_SESS[1])
    in_sig = (mins >= SIG_SESS[0]) & (mins < SIG_SESS[1])

    m_active = False
    m_high = m_low = m_vol = last_child = np.nan
    m_bar = -1
    m_time = None
    inside_count = 0
    sigs = []

    for i in range(1, n):
        day_chg = days[i] != days[i - 1]
        if m_active and (i - m_bar > MAX_BARS or day_chg):
            m_active = False

        inside_bar = h[i] <= h[i - 1] and l[i] >= l[i - 1]
        c_range = h[i - 1] - l[i - 1]
        c_body = abs(c[i - 1] - o[i - 1])
        c_atr = nz(atr[i - 1])
        c_ok = (c_range >= MIN_ATR_MULT * c_atr and c_range <= MAX_ATR_MULT * c_atr
                and c_body >= MIN_BODY_PCT / 100.0 * c_range and in_mother[i - 1])

        if (not m_active) and inside_bar and c_ok:
            m_high, m_low, m_vol = h[i - 1], l[i - 1], v[i - 1]
            m_bar, m_active, inside_count = i - 1, True, 1
            last_child = h[i] - l[i]
            m_time = df.index[i - 1]
        elif m_active and i > m_bar and h[i] <= m_high and l[i] >= m_low:
            inside_count += 1
            last_child = h[i] - l[i]

        if not (m_active and i > m_bar):
            continue

        has_vol = v[i] > 0
        sc1 = (not has_vol) or (m_vol >= avgvol[i])
        sc2 = inside_count >= 2
        sc3 = (not np.isnan(last_child)) and last_child <= 0.6 * (m_high - m_low)
        sc4 = (not has_vol) or (v[i] >= avgvol[i])
        score4 = int(sc1) + int(sc2) + int(sc3) + int(sc4)

        up_lv, dn_lv = m_high, m_low
        armed = inside_count >= MIN_INSIDE
        brk_up = c[i] > up_lv
        brk_dn = c[i] < dn_lv
        rng = h[i] - l[i]
        body_ok = rng > 0 and abs(c[i] - o[i]) >= BRK_BODY_PCT / 100.0 * rng
        size_ok = rng <= MAX_BRK_ATR * atr[i]
        mid = (m_high + m_low) / 2
        vw_na = np.isnan(vwap[i])
        trend_ce, trend_pe = c[i] > ema[i], c[i] < ema[i]
        vwap_ce = vw_na or c[i] > vwap[i]
        vwap_pe = vw_na or c[i] < vwap[i]
        risk_ce, risk_pe = c[i] - mid, mid - c[i]
        risk_ok_ce = 0 < risk_ce <= MAX_RISK_ATR * atr[i]
        risk_ok_pe = 0 < risk_pe <= MAX_RISK_ATR * atr[i]

        common = armed and in_sig[i] and body_ok and size_ok and score4 >= MIN_SCORE
        ce = common and brk_up and not brk_dn and c[i] > o[i] and trend_ce and vwap_ce and risk_ok_ce
        pe = common and brk_dn and not brk_up and c[i] < o[i] and trend_pe and vwap_pe and risk_ok_pe
        if dbg and i >= n - 40:
            print(f"{df.index[i].strftime('%d%b %H:%M')} C={c[i]:.2f} mother={m_time.strftime('%d%b %H:%M')} "
                  f"H={m_high:.2f} L={m_low:.2f} mRange/ATR={(m_high - m_low) / atr[i]:.2f} ATR={atr[i]:.2f} "
                  f"EMA={ema[i]:.2f} VWAP={vwap[i]:.2f} vol={v[i]:.0f} avg={avgvol[i]:.0f} mVol={m_vol:.0f} "
                  f"Q4={score4}(s1={int(sc1)} s2={int(sc2)} s3={int(sc3)} s4={int(sc4)}) inside={inside_count} "
                  f"bodyOk={body_ok} sizeOk={size_ok} brkDn={brk_dn} brkUp={brk_up} "
                  f"trendPE={trend_pe} vwapPE={vwap_pe} riskPE/ATR={risk_pe / atr[i]:.2f} "
                  f"sig={'CE' if ce else ('PE' if pe else '-')}")
        if not (ce or pe):
            continue

        is_ce = ce
        entry = c[i]
        sl = mid
        risk = abs(entry - sl)
        t1 = entry + risk * RR1 if is_ce else entry - risk * RR1
        t2 = entry + risk * RR2 if is_ce else entry - risk * RR2

        # ===== Quality /10 =====
        q = score4                                                    # Pine score (max 4)
        q += 1 if abs(c[i] - o[i]) >= 0.7 * rng else 0                # strong body
        slope_ok = i >= 3 and ((ema[i] > ema[i - 3]) if is_ce else (ema[i] < ema[i - 3]))
        q += 1 if slope_ok else 0                                     # EMA slope saath me
        dist = (c[i] - ema[i]) if is_ce else (ema[i] - c[i])
        q += 1 if dist >= 0.5 * atr[i] else 0                         # EMA se dur (trend strength)
        q += 1 if risk <= 1.0 * atr[i] else 0                         # tight stoploss
        near_ext = (c[i] - l[i]) / rng >= 0.75 if is_ce else (h[i] - c[i]) / rng >= 0.75
        q += 1 if near_ext else 0                                     # close candle ke extreme par
        q += 1 if ((not has_vol) or v[i] >= 1.5 * avgvol[i]) else 0   # strong volume
        q = int(min(q, 10))

        sigs.append(dict(
            idx=i, side="BUY CE" if is_ce else "BUY PE",
            start=df.index[i], end=df["End"].iloc[i],
            entry=entry, sl=sl, t1=t1, t2=t2, score4=score4, q10=q,
            m_time=m_time, m_high=m_high, m_low=m_low,
        ))
        m_active = False   # signal aate hi mother ka kaam khatam
    return sigs


# ===== Telegram =====
def send(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    for _ in range(4):
        try:
            r = requests.post(url, json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML",
                                         "disable_web_page_preview": True}, timeout=30)
        except Exception as e:
            print("send error", e)
            time.sleep(3)
            continue
        if r.status_code == 200:
            return True
        if r.status_code == 429:
            time.sleep(r.json().get("parameters", {}).get("retry_after", 5) + 1)
            continue
        print("telegram error", r.text)
        return False
    return False


def fmt_msg(name, tv, s):
    ce = s["side"] == "BUY CE"
    q = s["q10"]
    label = "Strong" if q >= 8 else ("Good" if q >= 6 else "Average")
    mt = s["m_time"]
    link = f"https://in.tradingview.com/chart/?symbol={tv}&interval=30"
    return (
        f"{'🟢' if ce else '🔴'} <b>{s['side']}</b> | <b>{html.escape(name)}</b>\n"
        f"🕒 Candle: <b>{s['start'].strftime('%d %b %Y')}</b>, "
        f"{s['start'].strftime('%H:%M')} - {s['end'].strftime('%H:%M')} IST (30m)\n"
        f"⭐ Quality: <b>{q}/10</b> ({label}) | Pine Q {s['score4']}/4\n"
        f"━━━━━━━━━━━━\n"
        f"Entry (close): <b>{s['entry']:.2f}</b>\n"
        f"🛑 SL: <b>{s['sl']:.2f}</b>\n"
        f"🎯 T1: <b>{s['t1']:.2f}</b> | T2: <b>{s['t2']:.2f}</b>\n"
        f"━━━━━━━━━━━━\n"
        f"📦 Mother: {mt.strftime('%d %b, %H:%M')} | H {s['m_high']:.2f} / L {s['m_low']:.2f}\n"
        f"⏳ Time stop: {TIME_STOP} candles\n"
        f"📈 <a href=\"{link}\">TradingView chart</a>"
    )


# ===== State =====
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f), False
        except Exception:
            pass
    return {"sent": []}, True


def save_state(st):
    st["sent"] = st["sent"][-3000:]
    with open(STATE_FILE, "w") as f:
        json.dump(st, f)


def main():
    if not TOKEN or not CHAT_ID:
        raise SystemExit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID secrets set karo")

    universe = build_universe()
    frames = download_all(universe)
    print(f"{len(frames)}/{len(universe)} symbols ka data aaya")
    if not frames:
        return

    dbg = os.getenv("DEBUG_SYMBOL", "").strip().upper()
    if dbg:
        if dbg in frames:
            d = frames[dbg]
            print(f"--- DEBUG {dbg}: last 30 candles (IST, candle start time) ---")
            print(d.tail(30)[["Open", "High", "Low", "Close", "Volume"]].round(2).to_string())
            for s in scan_symbol(d, dbg=True):
                print("SIGNAL", s["side"], s["start"], "| mother", s["m_time"],
                      round(s["m_high"], 2), round(s["m_low"], 2), "| Q10", s["q10"], "Q4", s["score4"])
        else:
            print(f"DEBUG: {dbg} ka data nahi mila")

    state, first_run = load_state()
    sent = set(state["sent"])

    all_days = sorted({d for f in frames.values() for d in f.index.date})
    keep_n = FIRST_RUN_SESSIONS if first_run else 1
    allowed = set(all_days[-keep_n:])

    pending = []
    for name, df in frames.items():
        try:
            for s in scan_symbol(df):
                if s["start"].date() not in allowed:
                    continue
                key = f"{name}|{s['start'].isoformat()}|{s['side']}"
                if key in sent:
                    continue
                pending.append((key, name, s))
        except Exception as e:
            print("scan error", name, e)

    pending.sort(key=lambda x: (x[2]["start"], x[1]))
    print(f"{len(pending)} naye signal")
    for key, name, s in pending:
        if send(fmt_msg(name, universe[name][1], s)):
            state["sent"].append(key)
        time.sleep(3.2)   # Telegram rate limit

    save_state(state)


if __name__ == "__main__":
    main()
