"""
Breaks and Retests [HG]  ->  Python port + Telegram alerts (GitHub Actions)

Signals
  Resistance Breakout      -> BUY CE
  Resistance Retest (valid) -> BUY CE   (breakout ke baad zone par wapas aakar bounce)
  Support Breakout         -> BUY PE
  Support Retest (valid)   -> BUY PE

Env:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   (GitHub Secrets)
  TF         "30" (default) ya "15"
  SCAN_DATE  optional, YYYY-MM-DD (comma se kai) - khali = aaj (IST)
  TEST_MESSAGE=1  -> Telegram test message
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ===================== SETTINGS =====================
BAR_MIN = int(os.environ.get("TF", "30"))
INTERVAL = "15m"               # Yahoo ka "30m" 09:00/09:30 se banta hai; hum 15m lekar khud 09:15 se 30m banate hain
PERIOD = "58d"                 # zyada history = levels sahi ban-te hain (Yahoo 15m max 60d)

# --- Pine inputs (same defaults) ---
LOOKBACK = 20                  # Lookback Range
RET_SINCE = 2                  # Bars Since Breakout
RET_VALID = 2                  # Retest Detection Limiter

# --- Kaun se alerts chahiye ---
SEND_BREAKOUT = True
SEND_RETEST = True

# --- Quality / risk (capital kam ho to MIN_QUALITY badhao) ---
MIN_QUALITY = 6                # /10  (0,2,4,6,8,10)
MAX_RISK_ATR = 2.5             # SL entry se 2.5 ATR se door ho to signal skip
RR1 = 1.0
RR2 = 2.0
EMA_LEN = 21

# --- Session (IST): sirf is time ke candles par signal ---
USE_SESSION = True
SIGNAL_SESS = (9 * 60 + 15, 14 * 60 + 30)

STATE_FILE = f"sent_br_{BAR_MIN}m.json"
IST = "Asia/Kolkata"

# ===================== SYMBOLS =====================
import re

TOP_N = 120                    # F&O ke top 120 stocks (+ saare indexes)
RANK_BY = "volume"             # "volume" (shares) ya "turnover" (volume x price)

NSE_LOT_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"

# Yahoo ticker -> (naam, TradingView symbol)
INDEXES = {
    "^NSEI": ("NIFTY", "NSE:NIFTY"),
    "^NSEBANK": ("BANKNIFTY", "NSE:BANKNIFTY"),
    "NIFTY_FIN_SERVICE.NS": ("FINNIFTY", "NSE:CNXFINANCE"),
    "NIFTY_MID_SELECT.NS": ("MIDCPNIFTY", "NSE:MIDCPNIFTY"),
    "^NSMIDCP": ("NIFTYNXT50", "NSE:NIFTYJR"),
    "^BSESN": ("SENSEX", "BSE:SENSEX"),
}
_INDEX_NAMES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "SENSEX", "BANKEX"}

# NSE se list na mile to ye built-in list (F&O stocks, ~230). Jo Yahoo par na mile wo skip ho jaata hai.
FALLBACK = """
ABB ABCAPITAL ADANIENSOL ADANIENT ADANIGREEN ADANIPORTS ADANIPOWER ALKEM AMBER AMBUJACEM ANGELONE
APLAPOLLO APOLLOHOSP ASHOKLEY ASIANPAINT ASTRAL AUBANK AUROPHARMA AXISBANK BAJAJ-AUTO BAJAJFINSV
BAJAJHLDNG BAJFINANCE BANDHANBNK BANKBARODA BANKINDIA BDL BEL BHARATFORG BHARTIARTL BHEL BIOCON
BLUESTARCO BOSCHLTD BPCL BRITANNIA BSE CAMS CANBK CDSL CGPOWER CHOLAFIN CIPLA COALINDIA COFORGE
COLPAL CONCOR CROMPTON CUMMINS DABUR DALBHARAT DELHIVERY DIVISLAB DIXON DLF DMART DRREDDY EICHERMOT
ETERNAL EXIDEIND FEDERALBNK FORTIS GAIL GLENMARK GMRAIRPORT GODREJCP GODREJPROP GRASIM HAL HAVELLS
HCLTECH HDFCAMC HDFCBANK HDFCLIFE HEROMOTOCO HFCL HINDALCO HINDPETRO HINDUNILVR HINDZINC HUDCO
ICICIBANK ICICIGI ICICIPRULI IDEA IDFCFIRSTB IEX IGL IIFL INDHOTEL INDIGO INDUSINDBK INDUSTOWER INFY
INOXWIND IOC IRCTC IREDA IRFC ITC JINDALSTEL JIOFIN JSWENERGY JSWSTEEL JUBLFOOD KALYANKJIL KAYNES
KEI KFINTECH KOTAKBANK KPITTECH LAURUSLABS LICHSGFIN LICI LODHA LT LTF LTIM LUPIN M&M MANAPPURAM
MANKIND MARICO MARUTI MAXHEALTH MAZDOCK MCX MFSL MOTHERSON MPHASIS MUTHOOTFIN NATIONALUM NAUKRI NBCC
NCC NESTLEIND NHPC NMDC NTPC NUVAMA NYKAA OBEROIRLTY OFSS OIL ONGC PAGEIND PATANJALI PAYTM PERSISTENT
PETRONET PFC PGEL PHOENIXLTD PIDILITIND PIIND PNB PNBHOUSING POLICYBZR POLYCAB POONAWALLA POWERGRID
POWERINDIA PPLPHARMA PRESTIGE RBLBANK RECLTD RELIANCE RVNL SAIL SAMMAANCAP SBICARD SBILIFE SBIN
SHREECEM SHRIRAMFIN SIEMENS SOLARINDS SONACOMS SRF SUNPHARMA SUPREMEIND SUZLON SYNGENE TATACHEM
TATACONSUMER TATAELXSI TATAPOWER TATASTEEL TATATECH TCS TECHM TIINDIA TITAGARH TITAN TMPV TMCV
TORNTPHARM TORNTPOWER TRENT TVSMOTOR ULTRACEMCO UNIONBANK UNITDSPR UNOMINDA UPL VBL VEDL VOLTAS WIPRO
YESBANK ZYDUSLIFE COCHINSHIP FORCEMOT GODFRYPHLP HYUNDAI MOTILALOFS NAM-INDIA VMM
""".split()


def tv_symbol(sym):
    return "NSE:" + re.sub(r"[&\-]", "_", sym)


def nse_fno_symbols():
    """NSE ki official F&O lot-size file se stock symbols. Fail ho to []."""
    try:
        r = requests.get(
            NSE_LOT_URL, timeout=20,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*", "Referer": "https://www.nseindia.com/"},
        )
        if r.status_code != 200:
            return []
        syms = []
        for line in r.text.splitlines()[1:]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) > 1:
                s = parts[1].upper()
                if s and s != "SYMBOL" and s not in _INDEX_NAMES and re.fullmatch(r"[A-Z0-9&\-]+", s):
                    syms.append(s)
        return list(dict.fromkeys(syms))
    except Exception as e:
        print("NSE list error:", e)
        return []


def build_universe():
    """Saare candidate stocks + indexes (ranking baad me hoti hai)."""
    syms = nse_fno_symbols()
    if len(syms) >= 100:
        print(f"NSE F&O list mili: {len(syms)} stocks")
    else:
        print("NSE list nahi mili, built-in list use ho rahi hai")
        syms = list(dict.fromkeys(FALLBACK))
    u = dict(INDEXES)
    for s in syms:
        u[f"{s}.NS"] = (s, tv_symbol(s))
    return u


def activity(df, mode="volume"):
    """Pichle (poore) 10 dino ka average daily volume / turnover."""
    day = df.index.date
    val = df["Volume"].fillna(0)
    if mode == "turnover":
        val = val * df["Close"]
    daily = val.groupby(day).sum()
    if len(daily) > 1:
        daily = daily.iloc[:-1]          # aaj ka adhura din hatao
    return float(daily.tail(10).mean()) if len(daily) else 0.0


def pick_top(data, act, n):
    """data me se indexes + volume ke hisaab se top n stocks rakho."""
    idx = [t for t in data if t in INDEXES]
    stocks = sorted((t for t in data if t not in INDEXES), key=lambda t: act.get(t, 0.0), reverse=True)[:n]
    keep = set(idx) | set(stocks)
    return {t: df for t, df in data.items() if t in keep}, len(stocks), len(idx)


# ===================== INDICATORS =====================
def rma_atr(h, l, c, length=14):
    n = len(c)
    tr = np.full(n, np.nan)
    for i in range(n):
        if i == 0:
            tr[i] = h[i] - l[i]
        else:
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.full(n, np.nan)
    if n >= length:
        atr[length - 1] = tr[:length].mean()
        for i in range(length, n):
            atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length
    return atr


def daily_vwap(df):
    hlc3 = (df["High"] + df["Low"] + df["Close"]) / 3.0
    day = df.index.date
    pv = (hlc3 * df["Volume"]).groupby(day).cumsum()
    vv = df["Volume"].groupby(day).cumsum()
    return (pv / vv.replace(0, np.nan)).to_numpy()


def in_sess(ts):
    if not USE_SESSION:
        return True
    m = ts.hour * 60 + ts.minute
    return SIGNAL_SESS[0] <= m < SIGNAL_SESS[1]


# ===================== RETEST ENGINE (Pine retestEvent ka port) =====================
class Retest:
    """
    pt = 'ph' -> resistance retest (y1 = high, y2 = low)   => bullish (BUY CE)
    pt = 'pl' -> support retest    (y1 = low,  y2 = high)  => bearish (BUY PE)
    """

    def __init__(self, pt):
        self.pt = pt
        self.prev_active = False
        self.orig_y2 = np.nan
        self.e_bar = None
        self.e_y1 = np.nan
        self.e_y2 = np.nan
        self.occurred = False

    def step(self, i, active, y1, y2, close, chg_pivot):
        orig = active and not self.prev_active          # retEvent (pehli active candle)
        if orig:
            self.orig_y2 = y2
        cond = (y2 < self.orig_y2) if self.pt == "ph" else (y2 > self.orig_y2)
        event = orig
        if cond:
            event = active                              # Pine: retEvent := retActive
        if event:
            self.e_bar, self.e_y1, self.e_y2 = i, y1, y2
            self.occurred = False
        valid = False
        if self.e_bar is not None:
            since = i - self.e_bar
            ok = (close >= self.e_y1) if self.pt == "ph" else (close <= self.e_y1)
            if 0 < since <= RET_VALID and ok and not self.occurred:
                valid = True
                self.occurred = True
        if chg_pivot and self.occurred:
            self.occurred = False
        self.prev_active = active
        return valid


EVENTS = []   # debug: (kind, name, time)


# ===================== STRATEGY =====================
def scan_symbol(df, name=""):
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    v = df["Volume"].fillna(0).to_numpy(float)
    n = len(df)
    bb = LOOKBACK
    if n < 2 * bb + 10:
        return []

    atr = rma_atr(h, l, c, 14)
    ema = df["Close"].ewm(span=EMA_LEN, adjust=False).mean().to_numpy()
    vwap = daily_vwap(df)
    avgv = df["Volume"].fillna(0).rolling(20).mean().to_numpy()
    times = df.index

    # --- pivots (confirm hote hain bb bars baad) + fixnan ---
    pl_raw = np.full(n, np.nan)
    ph_raw = np.full(n, np.nan)
    for i in range(2 * bb, n):
        p = i - bb
        if l[p] <= l[i - 2 * bb:i + 1].min():
            pl_raw[i] = l[p]
        if h[p] >= h[i - 2 * bb:i + 1].max():
            ph_raw[i] = h[p]
    pl = pd.Series(pl_raw).ffill().to_numpy()
    ph = pd.Series(ph_raw).ffill().to_numpy()

    sTop = sBot = rTop = rBot = np.nan
    sBar = rBar = -1
    prev_sBot = prev_rTop = np.nan
    sBreak = rBreak = False
    lastNaS = lastNaR = None

    rRet = Retest("ph")
    sRet = Retest("pl")
    out = []

    def emit(kind, side, i, sl, zone, zone_bar, label):
        entry = c[i]
        risk = abs(entry - sl)
        if not (risk > 0) or not (risk <= MAX_RISK_ATR * atr[i]):
            EVENTS.append(("skip", name, times[i]))
            return
        hasVol = v[i] > 0
        pts = 0
        pts += 1 if ((not hasVol) or v[i] >= avgv[i]) else 0
        if side == "CE":
            trend = c[i] > ema[i] and (np.isnan(vwap[i]) or c[i] > vwap[i])
            dir_ok = c[i] > o[i]
        else:
            trend = c[i] < ema[i] and (np.isnan(vwap[i]) or c[i] < vwap[i])
            dir_ok = c[i] < o[i]
        rng = h[i] - l[i]
        body_ok = rng > 0 and abs(c[i] - o[i]) >= 0.5 * rng
        pts += 1 if trend else 0
        pts += 1 if (body_ok and dir_ok) else 0
        pts += 1 if risk <= 1.5 * atr[i] else 0
        pts += 1 if (i - zone_bar) >= 10 else 0
        q = pts * 2
        if q < MIN_QUALITY:
            EVENTS.append(("skip", name, times[i]))
            return
        t1 = entry + risk * RR1 if side == "CE" else entry - risk * RR1
        t2 = entry + risk * RR2 if side == "CE" else entry - risk * RR2
        out.append({
            "side": f"BUY {side}", "kind": kind, "label": label, "time": times[i],
            "entry": float(entry), "sl": float(sl), "t1": float(t1), "t2": float(t2),
            "q": int(q), "zone": (float(min(zone)), float(max(zone))),
        })

    for i in range(n):
        chg_pl = i > 0 and not np.isnan(pl[i]) and not np.isnan(pl[i - 1]) and pl[i] != pl[i - 1]
        chg_ph = i > 0 and not np.isnan(ph[i]) and not np.isnan(ph[i - 1]) and ph[i] != ph[i - 1]

        # --- naya box ---
        if chg_pl:
            sTop = min(l[i - bb - 1], l[i - bb + 1])
            sBot = pl[i]
            sBar = i - bb
        if chg_ph:
            rTop = ph[i]
            rBot = max(h[i - bb - 1], h[i - bb + 1])
            rBar = i - bb

        # --- breakout (Pine switch: pehle support, phir resistance) ---
        cu = i > 0 and c[i] < sBot and c[i - 1] >= prev_sBot
        co = i > 0 and c[i] > rTop and c[i - 1] <= prev_rTop
        if cu and not sBreak:
            sBreak = True
        elif co and not rBreak:
            rBreak = True
        if chg_pl:
            sBreak = False
        if chg_ph:
            rBreak = False

        # --- barssince(na(break)) ---
        if not sBreak:
            lastNaS = i
        if not rBreak:
            lastNaR = i
        bsS = (i - lastNaS) if lastNaS is not None else np.nan
        bsR = (i - lastNaR) if lastNaR is not None else np.nan

        sBreakEvt = bsS == 1
        rBreakEvt = bsR == 1

        # --- retest conditions ---
        rc_s = (not np.isnan(bsS)) and bsS > RET_SINCE
        rc_r = (not np.isnan(bsR)) and bsR > RET_SINCE
        s_active = rc_s and (
            (h[i] >= sTop and c[i] <= sBot) or
            (h[i] >= sTop and c[i] >= sBot and c[i] <= sTop) or
            (h[i] >= sBot and h[i] <= sTop) or
            (h[i] >= sBot and h[i] <= sTop and c[i] < sBot))
        r_active = rc_r and (
            (l[i] <= rBot and c[i] >= rTop) or
            (l[i] <= rBot and c[i] <= rTop and c[i] >= rBot) or
            (l[i] <= rTop and l[i] >= rBot) or
            (l[i] <= rTop and l[i] >= rBot and c[i] > rTop))

        rRetValid = rRet.step(i, bool(r_active), h[i], l[i], c[i], chg_ph)
        sRetValid = sRet.step(i, bool(s_active), l[i], h[i], c[i], chg_pl)

        # --- signals (sirf session ke andar) ---
        if in_sess(times[i]):
            if rBreakEvt:
                EVENTS.append(("break", name, times[i]))
                if SEND_BREAKOUT:
                    emit("Breakout", "CE", i, rBot, (rBot, rTop), rBar, "Resistance Breakout")
            if sBreakEvt:
                EVENTS.append(("break", name, times[i]))
                if SEND_BREAKOUT:
                    emit("Breakout", "PE", i, sTop, (sBot, sTop), sBar, "Support Breakout")
            if rRetValid:
                EVENTS.append(("retest", name, times[i]))
                if SEND_RETEST:
                    sl = min(rRet.e_y2, rBot)
                    emit("Retest", "CE", i, sl, (rBot, rTop), rBar, "Resistance Retest (bounce confirm)")
            if sRetValid:
                EVENTS.append(("retest", name, times[i]))
                if SEND_RETEST:
                    sl = max(sRet.e_y2, sTop)
                    emit("Retest", "PE", i, sl, (sBot, sTop), sBar, "Support Retest (rejection confirm)")

        prev_sBot, prev_rTop = sBot, rTop

    return out


# ===================== DATA =====================
def to_30m(df):
    """15m -> 30m, candles 09:15-09:45 ... 15:15-15:30 (TradingView jaisa)."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = df.resample("30min", offset="15min", closed="left", label="left").agg(agg)
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def fetch_all(tickers):
    data = {}
    act = {}
    now = pd.Timestamp.now(tz=IST)
    for k in range(0, len(tickers), 25):
        batch = tickers[k:k + 25]
        try:
            raw = yf.download(batch, period=PERIOD, interval=INTERVAL, group_by="ticker",
                              auto_adjust=False, threads=True, progress=False)
        except Exception as e:
            print("download error:", e)
            continue
        for t in batch:
            try:
                df = raw[t] if len(batch) > 1 else raw
                df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
                if df.empty:
                    continue
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                df.index = df.index.tz_convert(IST)
                act[t] = activity(df, RANK_BY)
                if BAR_MIN == 30:
                    df = to_30m(df)
                mins = df.index.hour * 60 + df.index.minute
                span = np.where(mins >= 15 * 60 + 15, 15, BAR_MIN)
                ends = df.index + pd.to_timedelta(span, unit="m")
                df = df[ends <= now]            # chalti candle hatao
                if len(df):
                    data[t] = df
            except Exception:
                pass
        time.sleep(1)
    return data, act


# ===================== TELEGRAM =====================
def fmt(x):
    return f"{x:,.2f}"


def build_msg(name, tv_sym, s):
    t = s["time"]
    end = t + timedelta(minutes=BAR_MIN)
    link = f"https://in.tradingview.com/chart/?symbol={quote(tv_sym, safe='')}&interval={BAR_MIN}"
    emoji = "🟢🚀" if s["side"] == "BUY CE" else "🔴💥"
    risk = abs(s["entry"] - s["sl"])
    return (
        f"{emoji} <b>{s['side']}</b> | <b>{name}</b>\n"
        f"📌 {s['label']}\n"
        f"📅 {t.strftime('%d %b %Y')} | 🕒 {t.strftime('%H:%M')} - {end.strftime('%H:%M')} ({BAR_MIN}m candle)\n"
        f"⭐ Quality: <b>{s['q']}/10</b>\n"
        f"➡️ Entry: {fmt(s['entry'])}\n"
        f"🛑 Stoploss: <b>{fmt(s['sl'])}</b>\n"
        f"🎯 Target 1: <b>{fmt(s['t1'])}</b>\n"
        f"🎯 Target 2: <b>{fmt(s['t2'])}</b>\n"
        f"📏 Risk: {fmt(risk)} pts ({risk / s['entry'] * 100:.2f}%)\n"
        f"🧱 Zone: {fmt(s['zone'][0])} - {fmt(s['zone'][1])}\n"
        f"📈 <a href=\"{link}\">TradingView chart</a>"
    )


def send_telegram(text):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat = os.environ["TELEGRAM_CHAT_ID"]
    payload = {"chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code == 429:
        time.sleep(r.json().get("parameters", {}).get("retry_after", 5) + 1)
        r = requests.post(url, json=payload, timeout=30)
    if r.status_code != 200:
        print("Telegram error:", r.status_code, r.text[:200])
    return r.status_code == 200


# ===================== STATE =====================
def load_state():
    try:
        with open(STATE_FILE) as f:
            return set(json.load(f))
    except Exception:
        return set()


def save_state(keys):
    cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    keep = sorted(k for k in keys if k.split("|")[3][:10] >= cutoff)
    with open(STATE_FILE, "w") as f:
        json.dump(keep, f, indent=0)


# ===================== MAIN =====================
def main():
    raw_dates = os.environ.get("SCAN_DATE", "").strip()
    if raw_dates:
        scan_dates = {d.strip() for d in raw_dates.split(",") if d.strip()}
    else:
        scan_dates = {pd.Timestamp.now(tz=IST).strftime("%Y-%m-%d")}
    print("Scan dates:", sorted(scan_dates))

    if os.environ.get("TEST_MESSAGE") == "1" or raw_dates:
        print("Test message sent:", send_telegram("✅ Test - Breaks & Retests scanner Telegram se connected hai."))

    universe = build_universe()
    data, act = fetch_all(list(universe.keys()))
    print(f"Data mila: {len(data)}/{len(universe)} symbols")
    data, n_st, n_ix = pick_top(data, act, TOP_N)
    print(f"Scan universe: top {n_st} stocks (volume ke hisaab se) + {n_ix} indexes")
    last_bar = max((df.index[-1] for df in data.values()), default=None)

    sent = load_state()
    new_alerts = []
    for tkr, df in data.items():
        name, tv_sym = universe[tkr]
        try:
            sigs = scan_symbol(df, name)
        except Exception as e:
            print("scan error", tkr, e)
            continue
        for s in sigs:
            if s["time"].strftime("%Y-%m-%d") not in scan_dates:
                continue
            key = f"{name}|{s['side']}|{s['kind']}|{s['time'].strftime('%Y-%m-%d %H:%M')}"
            if key in sent:
                continue
            new_alerts.append((s["time"], key, build_msg(name, tv_sym, s)))

    new_alerts.sort(key=lambda x: x[0])
    for _, key, msg in new_alerts:
        if send_telegram(msg):
            sent.add(key)
        time.sleep(1.1)

    def cnt(kind):
        return sum(1 for k, _, t in EVENTS if k == kind and t.strftime("%Y-%m-%d") in scan_dates)

    summary = (f"Scan {', '.join(sorted(scan_dates))} ({BAR_MIN}m): universe {n_st} stocks + {n_ix} indexes, "
               f"last candle {last_bar.strftime('%d %b %H:%M') if last_bar is not None else 'NA'}, "
               f"breakout {cnt('break')}, retest {cnt('retest')}, risk/quality se reject {cnt('skip')}, "
               f"BUY signals {len(new_alerts)}")
    print(summary)
    if os.environ.get("DEBUG_SUMMARY") == "1" or raw_dates:
        send_telegram("🔎 " + summary)

    save_state(sent)


if __name__ == "__main__":
    if not os.environ.get("TELEGRAM_BOT_TOKEN") or not os.environ.get("TELEGRAM_CHAT_ID"):
        sys.exit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID secrets set karo")
    main()
