"""
Romy 9.5 Master Breakout Engine (30M) -> Telegram alerts
Scan: sabse volatile 100 F&O (option trading) stocks + main indices.
GitHub Actions har 30 min candle close ke baad ye script chalata hai.
"""
import json
import os
import sys
import time

import pandas as pd
import requests
import yfinance as yf

# ------------------------------------------------------------------
# SETTINGS (Pine Script wali same values)
# ------------------------------------------------------------------
INTERVAL = "30m"
EMA_FAST, EMA_SLOW, EMA_TREND = 9, 21, 50
USE_TREND = True
MAX_WAIT_BARS = 4
RR_RATIO = 1.5

TOP_N = int(os.getenv("TOP_N", "100"))                     # kitne volatile stocks
VOL_DAYS = 20                                               # volatility kitne din ki dekhni hai
SEND_WAIT_ALERTS = os.getenv("SEND_WAIT", "false").lower() == "true"

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
IST = "Asia/Kolkata"
SENT_FILE = "sent_signals.json"

# Indices (hamesha scan honge)
INDICES = {
    "^NSEI": "NIFTY 50",
    "^NSEBANK": "BANK NIFTY",
    "NIFTY_FIN_SERVICE.NS": "FINNIFTY",
    "^BSESN": "SENSEX",
}

# F&O stocks ki list (Yahoo ke .NS naam). F&O list time-time par badalti hai,
# isliye isme naye stock add / band stock hata sakte ho.
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


# ------------------------------------------------------------------
# VOLATILITY RANKING
# ------------------------------------------------------------------
def top_volatile_stocks(n: int) -> dict:
    """Pichle 20 din ki average daily range % ke hisaab se top-n stocks."""
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
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:n]
    return dict(ranked)


# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
def prepare(df: pd.DataFrame):
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)

    # sirf poori band hui candles (9:15 se shuru, last candle 15:15-15:30)
    ends = pd.Series(df.index + pd.Timedelta(minutes=30), index=df.index)
    mkt_close = pd.Series(df.index.normalize() + pd.Timedelta(hours=15, minutes=30), index=df.index)
    ends = ends.where(ends <= mkt_close, mkt_close)
    now = pd.Timestamp.now(tz=IST)
    df = df[ends <= now].copy()
    df["end"] = ends[ends <= now]
    return df


def fetch_all(tickers: list) -> dict:
    raw = yf.download(tickers, period="15d", interval=INTERVAL, group_by="ticker",
                      progress=False, auto_adjust=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = prepare(raw[t])
        except KeyError:
            continue
        if len(df) > EMA_TREND:
            out[t] = df
    return out


# ------------------------------------------------------------------
# STRATEGY (Pine logic ka bar-by-bar port)
# ------------------------------------------------------------------
def run_strategy(df: pd.DataFrame) -> list:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    ef = c.ewm(span=EMA_FAST, adjust=False).mean()
    es = c.ewm(span=EMA_SLOW, adjust=False).mean()
    et = c.ewm(span=EMA_TREND, adjust=False).mean()

    strong = (c - o).abs() > (h - l) * 0.4
    cross_up = (ef > es) & (ef.shift(1) <= es.shift(1))
    cross_dn = (ef < es) & (ef.shift(1) >= es.shift(1))
    trend_up = (c > et) if USE_TREND else pd.Series(True, index=df.index)
    trend_dn = (c < et) if USE_TREND else pd.Series(True, index=df.index)
    ce_cond = (cross_up & trend_up & (c > o) & strong).tolist()
    pe_cond = (cross_dn & trend_dn & (c < o) & strong).tolist()

    events = []
    pos = None
    waiting = None
    rh = rl = None
    wc = 0
    sig_time = None
    last_day = None

    for i in range(len(df)):
        t = df.index[i]
        end = df["end"].iloc[i]
        hi, lo, cl = h.iloc[i], l.iloc[i], c.iloc[i]

        if last_day is not None and t.date() != last_day:
            pos = None
            waiting = None
        last_day = t.date()

        if pos is not None and i > pos["i"]:
            if pos["side"] == "CE":
                if lo <= pos["sl"] or hi >= pos["tp"]:
                    pos = None
            else:
                if hi >= pos["sl"] or lo <= pos["tp"]:
                    pos = None

        flat = pos is None

        if ce_cond[i] and waiting != "CE" and flat:
            rh, rl, waiting, wc, sig_time = hi, lo, "CE", 0, t
            events.append(dict(kind="WAIT", side="CE", time=t, end=end, high=rh, low=rl))
        elif pe_cond[i] and waiting != "PE" and flat:
            rh, rl, waiting, wc, sig_time = hi, lo, "PE", 0, t
            events.append(dict(kind="WAIT", side="PE", time=t, end=end, high=rh, low=rl))

        if waiting:
            wc += 1
        if waiting and wc > MAX_WAIT_BARS:
            waiting = None

        confirmed = None
        if waiting == "CE" and cl > rh and flat:
            confirmed = "CE"
        elif waiting == "PE" and cl < rl and flat:
            confirmed = "PE"

        if waiting == "CE" and cl < rl:
            waiting = None
        if waiting == "PE" and cl > rh:
            waiting = None

        if confirmed == "CE":
            entry, sl = cl, rl
            risk = entry - sl
            if risk > 0:
                tp = entry + risk * RR_RATIO
                pos = dict(side="CE", sl=sl, tp=tp, i=i)
                events.append(dict(kind="BUY", side="CE", time=t, end=end, entry=entry,
                                   sl=sl, tp=tp, risk=risk, sig_time=sig_time))
                waiting = None
        elif confirmed == "PE":
            entry, sl = cl, rh
            risk = sl - entry
            if risk > 0:
                tp = entry - risk * RR_RATIO
                pos = dict(side="PE", sl=sl, tp=tp, i=i)
                events.append(dict(kind="BUY", side="PE", time=t, end=end, entry=entry,
                                   sl=sl, tp=tp, risk=risk, sig_time=sig_time))
                waiting = None

    return events


# ------------------------------------------------------------------
# TELEGRAM
# ------------------------------------------------------------------
def fmt_time(ts) -> str:
    return ts.strftime("%I:%M %p")


def build_message(ev: dict) -> str:
    name = ev["name"]
    vol = f"\n📊 Volatility (20d range): {ev['vol']:.2f}%" if ev.get("vol") else ""
    day = ev["time"].strftime("%d-%b-%Y")
    candle = f"{fmt_time(ev['time'])} - {fmt_time(ev['end'])} IST ({day})"

    if ev["kind"] == "WAIT":
        return (
            f"⏳ WAIT {ev['side']} - {name}\n"
            f"🕒 Signal Candle: {candle}\n"
            f"🔺 Breakout High: {ev['high']:.2f}\n"
            f"🔻 Breakout Low: {ev['low']:.2f}{vol}\n"
            f"Confirmation ka wait karo (max {MAX_WAIT_BARS} candles)."
        )

    icon = "🚀" if ev["side"] == "CE" else "💥"
    sig = f"\n⏳ Signal Candle: {fmt_time(ev['sig_time'])}" if ev.get("sig_time") is not None else ""
    return (
        f"{icon} {ev['side']} BUY NOW - {name}\n"
        f"🕒 Candle Time: {candle}{sig}\n"
        f"📍 Entry (candle close): {ev['entry']:.2f}\n"
        f"🛑 Stoploss: {ev['sl']:.2f}\n"
        f"🎯 Target: {ev['tp']:.2f}\n"
        f"📏 Risk: {ev['risk']:.2f} pts | RR 1:{RR_RATIO}{vol}\n"
        f"(SL/Target underlying price par hain)"
    )


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
    time.sleep(0.5)  # Telegram rate limit se bachne ke liye


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def load_sent() -> list:
    if os.path.exists(SENT_FILE):
        with open(SENT_FILE) as f:
            return json.load(f)
    return []


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        send_telegram("✅ Romy Alerts bot connected. Telegram test successful.")
        return

    vol_map = top_volatile_stocks(TOP_N)
    print(f"Top volatile stocks mile: {len(vol_map)}")

    names = dict(INDICES)
    for t in vol_map:
        names[t] = t.replace(".NS", "")
    tickers = list(names.keys())

    data = fetch_all(tickers)
    print(f"Intraday data mila: {len(data)} / {len(tickers)} symbols")

    now = pd.Timestamp.now(tz=IST)
    sent = load_sent()

    for t, df in data.items():
        for ev in run_strategy(df):
            if ev["end"] < now - pd.Timedelta(minutes=70):
                continue
            if ev["kind"] == "WAIT" and not SEND_WAIT_ALERTS:
                continue
            key = f"{t}|{ev['kind']}|{ev['side']}|{ev['time'].isoformat()}"
            if key in sent:
                continue
            ev["name"] = names[t]
            ev["vol"] = vol_map.get(t)
            send_telegram(build_message(ev))
            sent.append(key)
            print("Sent:", key)

    with open(SENT_FILE, "w") as f:
        json.dump(sent[-500:], f)


if __name__ == "__main__":
    main()
