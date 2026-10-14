"""
Mother Candle Confirmed Breakout | BUY CE / BUY PE  (30 min)
Pine v5 indicator ka Python port -> Telegram alerts (GitHub Actions)

Env:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   (GitHub Secrets)
  SCAN_DATE   (optional, YYYY-MM-DD) - us din ke saare signals bhejo; khali = aaj (IST)
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

# ===================== SETTINGS (Pine inputs ke same defaults) =====================
INTERVAL = "30m"
PERIOD = "30d"                 # yfinance 30m max ~60 din; warm-up ke liye 30d kaafi
BAR_MIN = 30

MIN_ATR_MULT = 1.0
MAX_ATR_MULT = 2.5
MIN_BODY_PCT = 50
MIN_INSIDE = 1
MAX_BARS = 8                   # mother valid for (bars)

MOTHER_SESS = (9 * 60 + 15, 13 * 60 + 30)   # 09:15 - 13:30 IST
SIGNAL_SESS = (9 * 60 + 15, 14 * 60 + 30)   # 09:15 - 14:30 IST
USE_SESSION = True

USE_CLOSE = True
BUF_PCT = 0.0
BRK_BODY_PCT = 40
MAX_BRK_ATR = 2.0
USE_TREND = True
EMA_LEN = 21
USE_VWAP = True
MIN_SCORE = 2                  # 0-4
FULLY_OUTSIDE = False
KEEP_ALIVE = True

SL_MIDPOINT = True             # True = Mother Midpoint (50%), False = opposite end
RR1 = 1.0
RR2 = 2.0
MAX_RISK_ATR = 2.0
TIME_STOP = 6

# Pine me pending confirmation kabhi expire nahi hota (0 = bilkul Pine jaisa).
# Agar chahiye ki breakout ke sirf agle N candles me hi confirm ho, to N likho (e.g. 1).
PENDING_MAX_BARS = 0

STATE_FILE = "sent_signals.json"
IST = "Asia/Kolkata"

# ===================== SYMBOLS =====================
INDEXES = {
    "^NSEI": ("NIFTY", "NSE:NIFTY"),
    "^NSEBANK": ("BANKNIFTY", "NSE:BANKNIFTY"),
    "NIFTY_FIN_SERVICE.NS": ("FINNIFTY", "NSE:CNXFINANCE"),
}

STOCKS = """
ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJFINANCE BAJAJFINSV BHARTIARTL BPCL BRITANNIA
CIPLA COALINDIA DIVISLAB DRREDDY EICHERMOT GRASIM HCLTECH HDFCBANK HDFCLIFE HEROMOTOCO HINDALCO
HINDUNILVR ICICIBANK INDUSINDBK INFY ITC JSWSTEEL KOTAKBANK LT M&M MARUTI NESTLEIND NTPC ONGC
POWERGRID RELIANCE SBILIFE SBIN SUNPHARMA TATACONSUMER TATASTEEL TCS TECHM TITAN ULTRACEMCO WIPRO
ABB ADANIGREEN ADANIPOWER AMBUJACEM BANDHANBNK BANKBARODA BEL BHEL BIOCON CANBK CHOLAFIN CUMMINS
DLF DABUR DIXON GAIL GODREJCP GODREJPROP HAL HAVELLS ICICIPRULI IDFCFIRSTB IEX IGL INDIGO IOC IRCTC
IRFC JINDALSTEL JIOFIN JUBLFOOD LICHSGFIN LUPIN MANAPPURAM MARICO MAXHEALTH MFSL MPHASIS MUTHOOTFIN
NMDC OBEROIRLTY OFSS PAGEIND PERSISTENT PETRONET PFC PIDILITIND PNB POLYCAB RECLTD SAIL SHREECEM
SHRIRAMFIN SIEMENS SRF TATACHEM TATAPOWER TORNTPHARM TRENT TVSMOTOR UNIONBANK UPL VEDL VOLTAS
YESBANK ZYDUSLIFE PIIND
""".split()


def build_universe():
    u = dict(INDEXES)
    for s in STOCKS:
        u[f"{s}.NS"] = (s, f"NSE:{s}")
    return u


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


def in_sess(ts, sess):
    if not USE_SESSION:
        return True
    m = ts.hour * 60 + ts.minute
    return sess[0] <= m < sess[1]


# ===================== STRATEGY (bar-by-bar, Pine jaisa) =====================
EVENTS = []   # debug: (kind, name, time)


def scan_symbol(df, name=""):
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    v = df["Volume"].fillna(0).to_numpy(float)
    n = len(df)
    if n < 40:
        return []

    atr = rma_atr(h, l, c, 14)
    ema = df["Close"].ewm(span=EMA_LEN, adjust=False).mean().to_numpy()
    vwap = daily_vwap(df)
    avgv = df["Volume"].fillna(0).rolling(20).mean().to_numpy()
    times = df.index
    days = np.array([t.date() for t in times])

    mActive = False
    mHigh = mLow = mVol = np.nan
    mBar = -1
    insideCount = 0
    lastChild = np.nan
    pendCE = pendPE = False
    pendCEHigh = pendPELow = np.nan
    pendCEBar = pendPEBar = -1
    out = []

    for i in range(n):
        dayChg = i > 0 and days[i] != days[i - 1]

        # --- expire ---
        if mActive and (i - mBar > MAX_BARS or dayChg):
            mActive = False

        # --- mother select (Auto) ---
        if i >= 1:
            insideBar = h[i] <= h[i - 1] and l[i] >= l[i - 1]
            cRange = h[i - 1] - l[i - 1]
            cBody = abs(c[i - 1] - o[i - 1])
            cAtr = 0.0 if np.isnan(atr[i - 1]) else atr[i - 1]
            cOk = (cRange >= MIN_ATR_MULT * cAtr and cRange <= MAX_ATR_MULT * cAtr
                   and cBody >= MIN_BODY_PCT / 100.0 * cRange
                   and in_sess(times[i - 1], MOTHER_SESS))
            newAuto = (not mActive) and insideBar and cOk
        else:
            newAuto = False

        if newAuto:
            mHigh, mLow, mVol, mBar = h[i - 1], l[i - 1], v[i - 1], i - 1
            mActive = True
            insideCount = 1
            lastChild = h[i] - l[i]
            EVENTS.append(("mother", name, times[i - 1]))
        elif mActive and i > mBar and h[i] <= mHigh and l[i] >= mLow:
            insideCount += 1
            lastChild = h[i] - l[i]

        # --- quality score (0-4) ---
        hasVol = v[i] > 0
        mv = 0.0 if np.isnan(mVol) else mVol
        sc1 = (not hasVol) or (mv >= avgv[i])
        sc2 = insideCount >= 2
        sc3 = (not np.isnan(lastChild)) and lastChild <= 0.6 * (mHigh - mLow)
        sc4 = (not hasVol) or (v[i] >= avgv[i])
        score = int(sc1) + int(sc2) + int(sc3) + int(sc4)

        # --- breakout candle ---
        buf = (mHigh - mLow) * BUF_PCT / 100.0
        upLv, dnLv = mHigh + buf, mLow - buf
        base = mActive and i > mBar
        armed = insideCount >= MIN_INSIDE

        closeUp, closeDn = c[i] > upLv, c[i] < dnLv
        if FULLY_OUTSIDE:
            brkUp, brkDn = l[i] > upLv, h[i] < dnLv
        elif USE_CLOSE:
            brkUp, brkDn = closeUp, closeDn
        else:
            brkUp, brkDn = h[i] > upLv, l[i] < dnLv

        rng = h[i] - l[i]
        bodyOk = rng > 0 and abs(c[i] - o[i]) >= BRK_BODY_PCT / 100.0 * rng
        sizeOk = rng <= MAX_BRK_ATR * atr[i]            # nan -> False (Pine jaisa)
        trendCE = (not USE_TREND) or c[i] > ema[i]
        trendPE = (not USE_TREND) or c[i] < ema[i]
        vwapCE = (not USE_VWAP) or np.isnan(vwap[i]) or c[i] > vwap[i]
        vwapPE = (not USE_VWAP) or np.isnan(vwap[i]) or c[i] < vwap[i]

        mid = (mHigh + mLow) / 2
        slCE = mid if SL_MIDPOINT else mLow
        slPE = mid if SL_MIDPOINT else mHigh

        common = base and armed and in_sess(times[i], SIGNAL_SESS) and bodyOk and sizeOk and score >= MIN_SCORE
        breakoutUp = bool(common and brkUp and not brkDn and c[i] > o[i] and trendCE and vwapCE)
        breakoutDn = bool(common and brkDn and not brkUp and c[i] < o[i] and trendPE and vwapPE)

        if breakoutUp:
            pendCE, pendCEHigh, pendCEBar = True, h[i], i
            EVENTS.append(("breakout", name, times[i]))
        if breakoutDn:
            pendPE, pendPELow, pendPEBar = True, l[i], i
            EVENTS.append(("breakout", name, times[i]))

        if PENDING_MAX_BARS and pendCE and i - pendCEBar > PENDING_MAX_BARS:
            pendCE = False
        if PENDING_MAX_BARS and pendPE and i - pendPEBar > PENDING_MAX_BARS:
            pendPE = False

        # --- confirmation candle ---
        confirmCE = pendCE and c[i] > o[i] and l[i] > pendCEHigh
        confirmPE = pendPE and c[i] < o[i] and h[i] < pendPELow

        riskCE, riskPE = c[i] - slCE, slPE - c[i]
        riskOkCE = riskCE > 0 and riskCE <= MAX_RISK_ATR * atr[i]
        riskOkPE = riskPE > 0 and riskPE <= MAX_RISK_ATR * atr[i]
        ceSig = bool(confirmCE and riskOkCE)
        peSig = bool(confirmPE and riskOkPE)

        if ceSig:
            pendCE = False
        if peSig:
            pendPE = False

        if ceSig or peSig:
            isCE = ceSig
            entry = c[i]
            sl = slCE if isCE else slPE
            risk = abs(entry - sl)
            t1 = entry + risk * RR1 if isCE else entry - risk * RR1
            t2 = entry + risk * RR2 if isCE else entry - risk * RR2
            out.append({
                "side": "BUY CE" if isCE else "BUY PE",
                "time": times[i],
                "entry": float(entry), "sl": float(sl),
                "t1": float(t1), "t2": float(t2),
                "score": score,
            })

        # --- breakout ke baad mother ka kaam khatam ---
        if base and (breakoutUp or breakoutDn or (not KEEP_ALIVE and (closeUp or closeDn))):
            mActive = False

    return out


# ===================== DATA =====================
def fetch_all(tickers):
    data = {}
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
                df = df[df.index + pd.Timedelta(minutes=BAR_MIN) <= now]   # chalti candle hatao
                if len(df):
                    data[t] = df
            except Exception:
                pass
        time.sleep(1)
    return data


# ===================== TELEGRAM =====================
def fmt(x):
    return f"{x:,.2f}"


def build_msg(name, tv_sym, s):
    t = s["time"]
    end = t + timedelta(minutes=BAR_MIN)
    q10 = s["score"] / 4 * 10
    q_txt = f"{q10:.1f}".rstrip("0").rstrip(".")
    link = f"https://in.tradingview.com/chart/?symbol={quote(tv_sym, safe='')}&interval=30"
    emoji = "🟢🚀" if s["side"] == "BUY CE" else "🔴💥"
    return (
        f"{emoji} <b>{s['side']}</b> | <b>{name}</b>\n"
        f"📅 {t.strftime('%d %b %Y')} | 🕒 {t.strftime('%H:%M')} - {end.strftime('%H:%M')} (30m candle)\n"
        f"⭐ Quality: <b>{q_txt}/10</b>\n"
        f"➡️ Entry: {fmt(s['entry'])}\n"
        f"🛑 Stoploss: <b>{fmt(s['sl'])}</b>\n"
        f"🎯 Target 1: <b>{fmt(s['t1'])}</b>\n"
        f"🎯 Target 2: <b>{fmt(s['t2'])}</b>\n"
        f"⏱ Time stop: {TIME_STOP} bars\n"
        f"📈 <a href=\"{link}\">TradingView chart</a>"
    )


def send_telegram(text):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat = os.environ["TELEGRAM_CHAT_ID"]
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat, "text": text, "parse_mode": "HTML",
              "disable_web_page_preview": True},
        timeout=30,
    )
    if r.status_code == 429:
        time.sleep(r.json().get("parameters", {}).get("retry_after", 5) + 1)
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=30,
        )
    ok = r.status_code == 200
    if not ok:
        print("Telegram error:", r.status_code, r.text[:200])
    return ok


# ===================== STATE (duplicate alert se bachne ke liye) =====================
def load_state():
    try:
        with open(STATE_FILE) as f:
            return set(json.load(f))
    except Exception:
        return set()


def save_state(keys):
    cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    keep = sorted(k for k in keys if k.split("|")[2][:10] >= cutoff)
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

    if os.environ.get("TEST_MESSAGE") == "1":
        ok = send_telegram("✅ Test message - Mother Candle scanner Telegram se connected hai.")
        print("Test message sent:", ok)

    universe = build_universe()
    data = fetch_all(list(universe.keys()))
    print(f"Data mila: {len(data)}/{len(universe)} symbols")
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
            key = f"{name}|{s['side']}|{s['time'].strftime('%Y-%m-%d %H:%M')}"
            if key in sent:
                continue
            new_alerts.append((s["time"], key, build_msg(name, tv_sym, s)))

    new_alerts.sort(key=lambda x: x[0])
    print("Naye signals:", len(new_alerts))
    for _, key, msg in new_alerts:
        if send_telegram(msg):
            sent.add(key)
        time.sleep(1.1)

    mothers = sum(1 for k, _, t in EVENTS if k == "mother" and t.strftime("%Y-%m-%d") in scan_dates)
    brks = sum(1 for k, _, t in EVENTS if k == "breakout" and t.strftime("%Y-%m-%d") in scan_dates)
    summary = (f"Scan {', '.join(sorted(scan_dates))}: data {len(data)}/{len(universe)} symbols, "
               f"last candle {last_bar.strftime('%d %b %H:%M') if last_bar is not None else 'NA'}, "
               f"mother {mothers}, breakout {brks}, BUY signals {len(new_alerts)}")
    print(summary)
    if os.environ.get("DEBUG_SUMMARY") == "1":
        send_telegram("🔎 " + summary)

    save_state(sent)


if __name__ == "__main__":
    if not os.environ.get("TELEGRAM_BOT_TOKEN") or not os.environ.get("TELEGRAM_CHAT_ID"):
        sys.exit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID secrets set karo")
    main()
