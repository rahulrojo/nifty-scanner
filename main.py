"""
Mother Candle Breakout (30M) -> Telegram alerts
Rule (TradingView "Mother Candle Breakout" script jaisa): mother candle ban jaye, uske baad
green candle mother ke high ke upar band ho -> BUY CE, red candle mother ke low ke neeche band ho -> BUY PE.
FULLY_OUTSIDE=true karne par candle mother ke high/low ko touch kiye bina bahar band honi chahiye.
Scan: sabse volatile 100 F&O stocks + main indices.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ------------------------------------------------------------------
# SETTINGS (TradingView indicator wali values, 30m ke hisaab se)
# ------------------------------------------------------------------
MIN_ATR, MAX_ATR = 1.0, 2.5          # mother size (x ATR)
MIN_BODY_PCT = 50                    # mother body %
MIN_INSIDE = 1                       # mother ke andar kam se kam itni candles
MAX_BARS = int(os.getenv("MAX_BARS", "8"))            # mother valid (script jaisa: 8 candles)
MOTHER_START, MOTHER_END = 9 * 60 + 15, 13 * 60 + 30  # mother candle session (open time)
SIGNAL_START, SIGNAL_END = 9 * 60 + 15, 14 * 60 + 30  # signal session (open time)
BRK_BODY_PCT = 40                    # breakout candle min body %
MAX_BRK_ATR = 2.0                    # breakout candle max size (x ATR)
USE_TREND, EMA_LEN = True, 21
USE_VWAP = True
MIN_SCORE = 2
FULLY_OUTSIDE = os.getenv("FULLY_OUTSIDE", "false").lower() == "true"   # touch kiye bina bahar band
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "true").lower() == "true"        # pehli candle reject ho to bhi mother zinda
SL_MID = os.getenv("SL_MODE", "mid") == "mid"        # mid = mother ka 50%, warna opposite end
RR1, RR2 = 1.0, 2.0
MAX_RISK_ATR = 2.0
TIME_STOP_BARS = int(os.getenv("TIME_STOP_BARS", "6"))  # script jaisa: 6 candles

TOP_N = int(os.getenv("TOP_N", "100"))
VOL_DAYS = 20
SEND_TEST = os.getenv("SEND_TEST", "true").lower() == "true"
ALERT_DAYS = int(os.getenv("ALERT_DAYS", "1"))        # 1 = sirf aaj, 5 = pichle 5 din
ALERT_DATE = os.getenv("ALERT_DATE", "").strip().lower()    # optional: "yesterday" ya YYYY-MM-DD, sirf us din ke signals dobara bhejo
DIAGNOSE_DATE = os.getenv("DIAGNOSE_DATE", "").strip()   # YYYY-MM-DD: us din har mother candle ka natija (sirf ek summary msg)
REPLAY_ONCE_DATE = os.getenv("REPLAY_ONCE_DATE", "2026-10-09").strip()  # is din ke mother + buy signals EK BAAR bhejo, phir band. Khali = band

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
IST = "Asia/Kolkata"
SENT_FILE = "sent_mother.json"

INDICES = {
    "^NSEI": "NIFTY 50",
    "^NSEBANK": "BANK NIFTY",
    "NIFTY_FIN_SERVICE.NS": "FINNIFTY",
    "^BSESN": "SENSEX",
}
TV_INDEX = {
    "^NSEI": "NSE:NIFTY",
    "^NSEBANK": "NSE:BANKNIFTY",
    "NIFTY_FIN_SERVICE.NS": "NSE:CNXFINANCE",
    "^BSESN": "BSE:SENSEX",
}

# F&O stocks (Yahoo .NS naam). List time-time par badalti hai, naye add / band hata sakte ho.
FNO_STOCKS = """
AARTIIND ABB ABCAPITAL ABFRL ACC ADANIENSOL ADANIENT ADANIGREEN ADANIPORTS ALKEM AMBUJACEM
ANGELONE APLAPOLLO APOLLOHOSP ASHOKLEY ASIANPAINT ASTRAL AUBANK AUROPHARMA AXISBANK BAJAJ-AUTO
BAJAJFINSV BAJFINANCE BALKRISIND BANDHANBNK BANKBARODA BANKINDIA BDL BEL BHARATFORG BHARTIARTL
BHEL BIOCON BLUESTARCO BOSCHLTD BPCL BRITANNIA BSE BSOFT CAMS CANBK CDSL CESC CGPOWER
CHAMBLFERT CHOLAFIN CIPLA COALINDIA COFORGE COLPAL CONCOR CROMPTON CUMMINSIND CYIENT DABUR
DALBHARAT DELHIVERY DIVISLAB DIXON DLF DMART DRREDDY EICHERMOT ESCORTS EXIDEIND FEDERALBNK
FORTIS GAIL GLENMARK GMRAIRPORT GODREJCP GODREJPROP GRANULES GRASIM HAL HAVELLS HCLTECH
HDFCAMC HDFCBANK HDFCLIFE HEROMOTOCO HFCL HINDALCO HINDCOPPER HINDPETRO HINDUNILVR HINDZINC
HUDCO ICICIBANK ICICIGI ICICIPRULI IDEA IDFCFIRSTB IEX IGL IIFL INDHOTEL INDIANB INDIGO
INDUSINDBK INDUSTOWER INFY INOXWIND IOC IRB IRCTC IREDA IRFC ITC JINDALSTEL JIOFIN JSWENERGY
JSWSTEEL JUBLFOOD KALYANKJIL KEI KFINTECH KOTAKBANK KPITTECH LAURUSLABS LICHSGFIN LICI LODHA
LT LTF LTIM LUPIN M&M MANAPPURAM MANKIND MARICO MARUTI MAXHEALTH MAZDOCK MCX MFSL MGL
MOTHERSON MPHASIS MUTHOOTFIN NATIONALUM NAUKRI NBCC NCC NESTLEIND NHPC NMDC NTPC NYKAA
OBEROIRLTY OFSS OIL ONGC PAGEIND PATANJALI PAYTM PERSISTENT PETRONET PFC PGEL PHOENIXLTD
PIDILITIND PIIND PNB PNBHOUSING POLICYBZR POLYCAB POONAWALLA POWERGRID PPLPHARMA PRESTIGE
RBLBANK RECLTD RELIANCE RVNL SAIL SBICARD SBILIFE SBIN SHREECEM SHRIRAMFIN SIEMENS SOLARINDS
SONACOMS SRF SUNPHARMA SUPREMEIND SUZLON SYNGENE TATACHEM TATACONSUM TATAELXSI TATAMOTORS
TATAPOWER TATASTEEL TATATECH TCS TECHM TIINDIA TITAGUARH TITAN TORNTPHARM TRENT TVSMOTOR
ULTRACEMCO UNIONBANK UNITDSPR UNOMINDA UPL VBL VEDL VOLTAS WIPRO YESBANK ZYDUSLIFE
""".split()


def tv_link(ticker: str) -> str:
    sym = TV_INDEX.get(ticker) or "NSE:" + ticker.replace(".NS", "").replace("&", "_").replace("-", "_")
    return f"https://www.tradingview.com/chart/?symbol={sym}&interval=30"


# ------------------------------------------------------------------
# INDICATOR MATHS (TradingView jaisa)
# ------------------------------------------------------------------
def pine_ema(s: pd.Series, length: int) -> pd.Series:
    v = s.values.astype(float)
    out = np.full(len(v), np.nan)
    if len(v) >= length:
        a = 2.0 / (length + 1)
        out[length - 1] = v[:length].mean()
        for i in range(length, len(v)):
            out[i] = a * v[i] + (1 - a) * out[i - 1]
    return pd.Series(out, index=s.index)


def pine_atr(df: pd.DataFrame, length: int = 14) -> np.ndarray:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    v = tr.values.astype(float)
    out = np.full(len(v), np.nan)
    if len(v) >= length:
        out[length - 1] = v[:length].mean()
        for i in range(length, len(v)):
            out[i] = (out[i - 1] * (length - 1) + v[i]) / length
    return out


def day_vwap(df: pd.DataFrame) -> np.ndarray:
    typ = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.normalize()
    cum_pv = (typ * df["volume"]).groupby(day).cumsum()
    cum_v = df["volume"].groupby(day).cumsum()
    return (cum_pv / cum_v).where(cum_v > 0).values


# ------------------------------------------------------------------
# VOLATILITY RANKING + DATA
# ------------------------------------------------------------------
def top_volatile_stocks(n: int) -> dict:
    tickers = [s + ".NS" for s in FNO_STOCKS]
    raw = yf.download(tickers, period="2mo", interval="1d", group_by="ticker",
                      progress=False, auto_adjust=False, threads=True)
    scores = {}
    for t in tickers:
        try:
            d = raw[t].dropna().tail(VOL_DAYS)
        except KeyError:
            continue
        if len(d) < 10:
            continue
        scores[t] = float(((d["High"] - d["Low"]) / d["Close"]).mean() * 100)
    return dict(sorted(scores.items(), key=lambda x: -x[1])[:n])


def prepare(df: pd.DataFrame):
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df["volume"] = df["volume"].fillna(0)
    df = df.dropna()
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)

    # 15m -> 30m, candle 9:15 se shuru (TradingView jaisa): 9:15, 9:45 ... 15:15
    df = df.resample("30min", offset="15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        return df

    ends = pd.Series(df.index + pd.Timedelta(minutes=30), index=df.index)
    mkt_close = pd.Series(df.index.normalize() + pd.Timedelta(hours=15, minutes=30), index=df.index)
    ends = ends.where(ends <= mkt_close, mkt_close)
    now = pd.Timestamp.now(tz=IST)
    df = df[ends <= now].copy()
    df["end"] = ends[ends <= now]
    return df


def fetch_all(tickers: list) -> dict:
    raw = yf.download(tickers, period="59d", interval="15m", group_by="ticker",
                      progress=False, auto_adjust=False, threads=True)
    out = {}
    for t in tickers:
        try:
            sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = prepare(sub)
        except KeyError:
            continue
        if len(df) > 30:
            out[t] = df
    return out


# ------------------------------------------------------------------
# STRATEGY: Mother Candle Breakout (Pine indicator ka port)
# ------------------------------------------------------------------
def run_strategy(df: pd.DataFrame) -> list:
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close"))
    v = df["volume"].values.astype(float)
    t = df.index
    ends = df["end"]
    atr = pine_atr(df)
    ema = pine_ema(df["close"], EMA_LEN).values
    vwap = day_vwap(df)
    avgv = df["volume"].rolling(20).mean().values
    mins = np.asarray(t.hour * 60 + t.minute)
    n = len(df)

    events = []
    m_active = False
    m_high = m_low = m_vol = last_child = np.nan
    m_bar = -1
    inside = 0
    announced = False
    rej = None   # is mother par pehla reject hua breakout ka karan

    for i in range(1, n):
        newday = t[i].date() != t[i - 1].date()
        if m_active and (i - m_bar > MAX_BARS or newday):
            m_active = False
            events.append(dict(kind="MOTHER_END", time=t[i], m_time=t[m_bar],
                               result=rej or "breakout aaya hi nahi"))

        inside_bar = h[i] <= h[i - 1] and l[i] >= l[i - 1]
        new_auto = False
        if not m_active and inside_bar:
            cr = h[i - 1] - l[i - 1]
            cb = abs(c[i - 1] - o[i - 1])
            ca = atr[i - 1]
            in_m = MOTHER_START <= mins[i - 1] < MOTHER_END
            if (not np.isnan(ca)) and MIN_ATR * ca <= cr <= MAX_ATR * ca \
                    and cb >= MIN_BODY_PCT / 100.0 * cr and in_m:
                new_auto = True

        if new_auto:
            m_active = True
            rej = None
            m_high, m_low, m_vol, m_bar = h[i - 1], l[i - 1], v[i - 1], i - 1
            inside = 1
            last_child = h[i] - l[i]
        elif m_active and i > m_bar and h[i] <= m_high and l[i] >= m_low:
            inside += 1
            last_child = h[i] - l[i]

        # mother candle ready (inside candles poori hui) -> MOTHER alert event
        if new_auto:
            announced = False
        if m_active and not announced and inside >= MIN_INSIDE:
            announced = True
            events.append(dict(
                kind="MOTHER", time=t[i], end=ends.iloc[i],
                m_time=t[m_bar], m_end=ends.iloc[m_bar], m_high=m_high, m_low=m_low,
            ))

        if not (m_active and i > m_bar):
            continue

        # quality score (0-4)
        has_vol = v[i] > 0
        sc1 = (not has_vol) or bool(m_vol >= avgv[i])
        sc2 = inside >= 2
        sc3 = (not np.isnan(last_child)) and last_child <= 0.6 * (m_high - m_low)
        sc4 = (not has_vol) or bool(v[i] >= avgv[i])
        score = int(sc1) + int(sc2) + int(sc3) + int(sc4)

        armed = inside >= MIN_INSIDE
        close_up = c[i] > m_high
        close_dn = c[i] < m_low
        brk_up = (l[i] > m_high) if FULLY_OUTSIDE else close_up      # touch kiye bina upar
        brk_dn = (h[i] < m_low) if FULLY_OUTSIDE else close_dn       # touch kiye bina neeche

        rng = h[i] - l[i]
        body_ok = rng > 0 and abs(c[i] - o[i]) >= BRK_BODY_PCT / 100.0 * rng
        size_ok = bool(rng <= MAX_BRK_ATR * atr[i])
        in_sig = SIGNAL_START <= mins[i] < SIGNAL_END

        trend_ce = (not USE_TREND) or bool(c[i] > ema[i])
        trend_pe = (not USE_TREND) or bool(c[i] < ema[i])
        vw_ok = np.isnan(vwap[i])
        vwap_ce = (not USE_VWAP) or vw_ok or bool(c[i] > vwap[i])
        vwap_pe = (not USE_VWAP) or vw_ok or bool(c[i] < vwap[i])

        mid = (m_high + m_low) / 2
        sl_ce = mid if SL_MID else m_low
        sl_pe = mid if SL_MID else m_high
        risk_ce = c[i] - sl_ce
        risk_pe = sl_pe - c[i]
        risk_ok_ce = risk_ce > 0 and bool(risk_ce <= MAX_RISK_ATR * atr[i])
        risk_ok_pe = risk_pe > 0 and bool(risk_pe <= MAX_RISK_ATR * atr[i])

        common = armed and in_sig and body_ok and size_ok and score >= MIN_SCORE
        ce = common and brk_up and not brk_dn and c[i] > o[i] and trend_ce and vwap_ce and risk_ok_ce
        pe = common and brk_dn and not brk_up and c[i] < o[i] and trend_pe and vwap_pe and risk_ok_pe

        if (close_up or close_dn) and not (ce or pe) and rej is None:
            up_side = close_up
            if not armed:
                rej = "compression kam"
            elif not in_sig:
                rej = "signal time ke baad"
            elif not body_ok:
                rej = "breakout candle ki body chhoti"
            elif not size_ok:
                rej = "breakout candle bahut badi"
            elif score < MIN_SCORE:
                rej = "quality score kam"
            elif up_side and not c[i] > o[i]:
                rej = "close upar par candle green nahi"
            elif (not up_side) and not c[i] < o[i]:
                rej = "close neeche par candle red nahi"
            elif up_side and not trend_ce:
                rej = "CE: EMA ke neeche"
            elif (not up_side) and not trend_pe:
                rej = "PE: EMA ke upar"
            elif up_side and not vwap_ce:
                rej = "CE: VWAP ke neeche"
            elif (not up_side) and not vwap_pe:
                rej = "PE: VWAP ke upar"
            elif (up_side and not risk_ok_ce) or ((not up_side) and not risk_ok_pe):
                rej = "SL bahut door"
            else:
                rej = "clean breakout nahi (mother ko touch kiya)"

        if ce or pe:
            side = "CE" if ce else "PE"
            entry = c[i]
            sl = sl_ce if ce else sl_pe
            risk = abs(entry - sl)
            sign = 1 if ce else -1
            events.append(dict(
                kind="BUY", side=side, time=t[i], end=ends.iloc[i], entry=entry, sl=sl,
                t1=entry + sign * risk * RR1, t2=entry + sign * risk * RR2,
                risk=risk, score=score, m_time=t[m_bar], m_high=m_high, m_low=m_low,
            ))

        if ce or pe or ((not KEEP_ALIVE) and (close_up or close_dn)):
            m_active = False
            events.append(dict(kind="MOTHER_END", time=t[i], m_time=t[m_bar],
                               result="BUY" if (ce or pe) else (rej or "reject")))

    # din khatam (3:30 PM candle) par bhi mother active ho to uska natija likho
    if m_active and ends.iloc[-1].hour * 60 + ends.iloc[-1].minute >= 15 * 60 + 30:
        events.append(dict(kind="MOTHER_END", time=ends.iloc[-1], m_time=t[m_bar],
                           result=rej or "din khatam, breakout nahi aaya"))

    return events


# ------------------------------------------------------------------
# TELEGRAM
# ------------------------------------------------------------------
def fmt_time(ts) -> str:
    return ts.strftime("%I:%M %p")


def build_buy_message(ev: dict) -> str:
    icon = "🚀" if ev["side"] == "CE" else "💥"
    day = ev["time"].strftime("%d-%b-%Y")
    stop_t = ev["end"] + pd.Timedelta(minutes=30 * TIME_STOP_BARS)
    close_t = ev["end"].normalize() + pd.Timedelta(hours=15, minutes=30)
    stop_t = min(stop_t, close_t)   # intraday: market close (3:30 PM) ke baad nahi
    vol = f"\n📊 Volatility (20d range): {ev['vol']:.2f}%" if ev.get("vol") else ""
    return (
        f"{icon} BUY {ev['side']} NOW - {ev['name']}  (Mother Candle 30m)\n"
        f"🕒 Breakout Candle: {fmt_time(ev['time'])} - {fmt_time(ev['end'])} IST ({day})\n"
        f"🟧 Mother Candle: {fmt_time(ev['m_time'])} | High {ev['m_high']:.2f} | Low {ev['m_low']:.2f}\n"
        f"📍 Entry (candle close): {ev['entry']:.2f}\n"
        f"🛑 Stoploss: {ev['sl']:.2f}\n"
        f"🎯 Target 1: {ev['t1']:.2f} | Target 2: {ev['t2']:.2f}\n"
        f"⏱ Time stop: {TIME_STOP_BARS} candles ya market close, jo pehle (~{fmt_time(stop_t)}). T1 na aaye to nikal jao\n"
        f"⭐ Quality: Q{ev['score']}/4{vol}\n"
        f"(SL/Target underlying price par hain)\n"
        f"📈 Chart: {ev['tv']}"
    )


def build_mother_message(ev: dict) -> str:
    day = ev["m_time"].strftime("%d-%b-%Y")
    close_t = ev["m_end"].normalize() + pd.Timedelta(hours=15, minutes=30)
    valid_t = min(ev["m_end"] + pd.Timedelta(minutes=30 * MAX_BARS), close_t)
    vol = f"\n📊 Volatility (20d range): {ev['vol']:.2f}%" if ev.get("vol") else ""
    return (
        f"🟧 MOTHER CANDLE BANI - {ev['name']}  (30m)\n"
        f"🕒 Mother Candle: {fmt_time(ev['m_time'])} - {fmt_time(ev['m_end'])} IST ({day})\n"
        f"🔺 High: {ev['m_high']:.2f} | 🔻 Low: {ev['m_low']:.2f}\n"
        f"✅ Confirm: {fmt_time(ev['time'])} - {fmt_time(ev['end'])} wali candle uske andar band hui\n"
        f"🚀 BUY CE: green candle {ev['m_high']:.2f} ke upar band ho\n"
        f"💥 BUY PE: red candle {ev['m_low']:.2f} ke neeche band ho\n"
        f"⏳ Valid: {MAX_BARS} candles ya market close, jo pehle (~{fmt_time(valid_t)}){vol}\n"
        f"📈 Chart: {ev['tv']}"
    )


def build_message(ev: dict) -> str:
    return build_mother_message(ev) if ev.get("kind") == "MOTHER" else build_buy_message(ev)


def send_telegram(text: str) -> None:
    if not TOKEN or not CHAT_ID:
        print("Telegram secrets missing!")
        sys.exit(1)
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "text": text},
        timeout=20,
    )
    r.raise_for_status()
    time.sleep(0.5)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def load_sent() -> list:
    if os.path.exists(SENT_FILE):
        with open(SENT_FILE) as f:
            return json.load(f)
    return []


def check_mode(sym: str, date_str: str) -> None:
    """Ek stock ka history check: us din ke saare signals Telegram par bhejo."""
    t = sym if (sym.startswith("^") or sym.endswith(".NS")) else sym + ".NS"
    data = fetch_all([t])
    if t not in data:
        send_telegram(f"🔎 {sym}: data nahi mila.")
        return
    events = run_strategy(data[t])
    if date_str:
        events = [e for e in events if str(e["time"].date()) == date_str]
    if not events:
        send_telegram(f"🔎 {sym}: {date_str or 'is period'} me koi Mother Candle signal nahi mila.")
        return
    for ev in events:
        ev["name"], ev["vol"], ev["tv"] = sym, None, tv_link(t)
        send_telegram("🔎 CHECK\n" + build_message(ev))


def main() -> None:
    check_sym = os.getenv("CHECK_SYMBOL", "").strip().upper()
    if check_sym:
        check_mode(check_sym, os.getenv("CHECK_DATE", "").strip())
        return

    if SEND_TEST:
        now_s = pd.Timestamp.now(tz=IST).strftime("%d-%b-%Y %I:%M %p")
        mode_s = f"REPLAY ({ALERT_DATE})" if ALERT_DATE else "NORMAL (aaj ke signals)"
        send_telegram(f"✅ Mother Candle 30m TEST - bot connected.\nRun time: {now_s} IST\nMode: {mode_s}")

    vol_map = top_volatile_stocks(TOP_N)
    names = dict(INDICES)
    for t in vol_map:
        names[t] = t.replace(".NS", "")
    tickers = list(names.keys())

    data = fetch_all(tickers)
    print(f"Intraday data mila: {len(data)} / {len(tickers)} symbols")

    now = pd.Timestamp.now(tz=IST)
    sent = load_sent()
    all_events = {tk: run_strategy(df) for tk, df in data.items()}

    def ev_key(tk, ev):
        if ev["kind"] == "MOTHER":
            return f"{tk}|MOTHER|{ev['m_time'].isoformat()}"
        return f"{tk}|{ev['side']}|{ev['time'].isoformat()}"

    def gather(date_ok, dedup):
        out = []
        for tk, evs in all_events.items():
            for ev in evs:
                if ev["kind"] not in ("MOTHER", "BUY"):
                    continue
                if not date_ok(ev["time"].date()):
                    continue
                key = ev_key(tk, ev)
                if dedup and key in sent:
                    continue
                e = dict(ev)
                e.update(name=names[tk], vol=vol_map.get(tk), key=key, tv=tv_link(tk))
                out.append(e)
        out.sort(key=lambda e: (e["time"], 0 if e["kind"] == "MOTHER" else 1))
        return out

    def outcome_text(day):
        cnt = {}
        for evs in all_events.values():
            for e in evs:
                if e["kind"] == "MOTHER_END" and e["m_time"].date() == day:
                    cnt[e["result"]] = cnt.get(e["result"], 0) + 1
        if not cnt:
            return ""
        items = sorted(cnt.items(), key=lambda x: -x[1])
        return "\nMother ka natija: " + " | ".join(f"{k}: {v}" for k, v in items)

    def replay(day):
        """Kisi ek din ke saare MOTHER + BUY signals bhejo (dedup ignore)."""
        evs = gather(lambda d: d == day, dedup=False)
        tag = f"🔁 {day.strftime('%d-%b-%Y')} KA SIGNAL\n"
        for ev in evs:
            send_telegram(tag + build_message(ev))
        n_m = sum(1 for e in evs if e["kind"] == "MOTHER")
        n_b = sum(1 for e in evs if e["kind"] == "BUY")
        send_telegram(
            f"📋 {day.strftime('%d-%b-%Y')} ka scan complete\n"
            f"Symbols scanned: {len(data)} / {len(tickers)}\n"
            f"Mother candle: {n_m} | Buy signal: {n_b}"
            + outcome_text(day)
            + ("" if evs else "\nIs din koi Mother Candle signal nahi bana.")
        )

    # 0) Diagnose: us din har mother ka natija ek hi message me (alert dobara nahi)
    if DIAGNOSE_DATE:
        day = pd.Timestamp(DIAGNOSE_DATE).date()
        rows = sorted(
            (e["m_time"], names[tk], e["result"])
            for tk, evs in all_events.items() for e in evs
            if e["kind"] == "MOTHER_END" and e["m_time"].date() == day
        )
        lines = [f"{nm} ({mt.strftime('%I:%M %p')}) -> {res}" for mt, nm, res in rows[:40]]
        send_telegram(
            f"🔎 {day.strftime('%d-%b-%Y')} mother candle natija ({len(rows)} mother)"
            + outcome_text(day) + "\n\n"
            + ("\n".join(lines) if lines else "Us din koi mother candle poori nahi hui.")
        )
        return

    # 1) Explicit date: sirf us din ke signals (state nahi badalti)
    target = None
    if ALERT_DATE:
        if ALERT_DATE in ("yesterday", "kal"):
            dates = sorted({ts.date() for d in data.values() for ts in d.index})
            prior = [d for d in dates if d < now.date()]
            target = prior[-1] if prior else None
        else:
            target = pd.Timestamp(ALERT_DATE).date()
    if target is not None:
        replay(target)
        return

    # 2) Ek baar: REPLAY_ONCE_DATE ke dono signals (mother + buy). Phir dobara nahi.
    if REPLAY_ONCE_DATE:
        rkey = f"REPLAY|{REPLAY_ONCE_DATE}"
        rday = pd.Timestamp(REPLAY_ONCE_DATE).date()
        if rkey not in sent and rday < now.date():
            replay(rday)
            sent.append(rkey)

    # 3) Normal: jis din scan karo, usi din ke signals (mother + buy)
    lo = (now - pd.Timedelta(days=ALERT_DAYS - 1)).date()
    pending = gather(lambda d: d >= lo, dedup=True)
    for ev in pending:
        send_telegram(build_message(ev))
        sent.append(ev["key"])
        print("Sent:", ev["key"])

    with open(SENT_FILE, "w") as f:
        json.dump(sent[-1500:], f)

    if SEND_TEST:
        today_evs = [e for evs in all_events.values() for e in evs if e["time"].date() == now.date()]
        n_m = sum(1 for e in today_evs if e["kind"] == "MOTHER")
        n_b = sum(1 for e in today_evs if e["kind"] == "BUY")
        last = max((d.index[-1] for d in data.values()), default=None)
        last_s = last.strftime("%d-%b %I:%M %p") if last is not None else "koi data nahi"
        buys = [(e["time"], names[tk], e["side"]) for tk, evs in all_events.items() for e in evs if e["kind"] == "BUY"]
        lb = max(buys) if buys else None
        lb_s = f"{lb[1]} {lb[2]} @ {lb[0].strftime('%d-%b %I:%M %p')}" if lb else "koi nahi"
        send_telegram(
            f"📋 Mother Candle scan complete\n"
            f"Symbols scanned: {len(data)} / {len(tickers)}\n"
            f"Last candle start: {last_s} IST\n"
            f"Aaj mile: {n_m} mother, {n_b} buy signal{outcome_text(now.date())}\n"
            f"Pichla BUY signal (59 din me): {lb_s}\n"
            f"Naye alerts bheje: {len(pending)}"
        )

if __name__ == "__main__":
    main()
