"""
ROMYO Confirmed Mother Candle Breakout  ->  Telegram scanner (30 min)

Pine v6 logic ported bar-by-bar:
  Mother candle -> Breakout candle (close beyond mother) -> Confirmation candle
  (close AND full candle beyond breakout level). Only the confirmation candle
  gives BUY CE / BUY PE.

Run by GitHub Actions (scanner.yml). Needs secrets:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""
import os
import sys
import json
import time
import html

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ======================= SETTINGS (same as Pine inputs) =======================
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

# ======================= BOT SETTINGS =======================
TIMEFRAME_MIN = 30
HISTORY_PERIOD = "60d"          # yfinance limit for 30m data
STATE_FILE = "mc_state.json"
IST = "Asia/Kolkata"
BATCH_SIZE = 20
TARGET_RR = (1, 2, 3)           # T1 / T2 / T3 as multiples of risk (extra, not in Pine)

# ======================= SYMBOLS =======================
# Top 100 liquid F&O stocks. Edit freely. Symbols that fail to load are skipped.
STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "BHARTIARTL", "ITC",
    "LT", "KOTAKBANK", "AXISBANK", "HINDUNILVR", "BAJFINANCE", "MARUTI", "SUNPHARMA",
    "NTPC", "POWERGRID", "ONGC", "ADANIENT", "ADANIPORTS", "COALINDIA", "ULTRACEMCO",
    "ASIANPAINT", "TITAN", "WIPRO", "HCLTECH", "TECHM", "JSWSTEEL", "TATASTEEL", "M&M",
    "BAJAJFINSV", "BAJAJ-AUTO", "HINDALCO", "GRASIM", "CIPLA", "DRREDDY", "DIVISLAB",
    "EICHERMOT", "HEROMOTOCO", "BPCL", "IOC", "GAIL", "VEDL", "SBILIFE", "HDFCLIFE",
    "INDUSINDBK", "BRITANNIA", "NESTLEIND", "TATACONSUM", "APOLLOHOSP", "PIDILITIND",
    "SIEMENS", "ABB", "HAL", "BEL", "BHEL", "SAIL", "NMDC", "PNB", "BANKBARODA",
    "CANBK", "IDFCFIRSTB", "FEDERALBNK", "AUBANK", "RECLTD", "PFC", "IRCTC", "TRENT",
    "DLF", "GODREJPROP", "JINDALSTEL", "TVSMOTOR", "ASHOKLEY", "BOSCHLTD", "MOTHERSON",
    "CHOLAFIN", "SHRIRAMFIN", "MUTHOOTFIN", "JIOFIN", "ETERNAL", "DMART", "HAVELLS",
    "POLYCAB", "DIXON", "PERSISTENT", "COFORGE", "MPHASIS", "TATAPOWER", "ADANIGREEN",
    "TORNTPHARM", "LUPIN", "AUROPHARMA", "BIOCON", "ALKEM", "ZYDUSLIFE", "INDIGO",
    "VOLTAS", "CUMMINSIND", "PAGEIND", "UNIONBANK",
]

# name: (yahoo ticker, tradingview symbol)
INDEXES = {
    "NIFTY": ("^NSEI", "NSE:NIFTY"),
    "BANKNIFTY": ("^NSEBANK", "NSE:BANKNIFTY"),
    "FINNIFTY": ("NIFTY_FIN_SERVICE.NS", "NSE:CNXFINANCE"),
    "MIDCPNIFTY": ("NIFTY_MID_SELECT.NS", "NSE:NIFTY_MID_SELECT"),
    "SENSEX": ("^BSESN", "BSE:SENSEX"),
}


def build_symbols():
    out = {}
    for s in STOCKS:
        tv = "NSE:" + s.replace("&", "_").replace("-", "_")
        out[s] = {"yf": f"{s}.NS", "tv": tv}
    for name, (yfs, tv) in INDEXES.items():
        out[name] = {"yf": yfs, "tv": tv}
    return out


SYMBOLS = build_symbols()


# ======================= PINE-STYLE INDICATORS =======================
def pine_ema(src, length):
    out = np.full(len(src), np.nan)
    if len(src) < length:
        return out
    alpha = 2.0 / (length + 1)
    out[length - 1] = np.mean(src[:length])
    for i in range(length, len(src)):
        out[i] = alpha * src[i] + (1 - alpha) * out[i - 1]
    return out


def pine_rma(src, length):
    out = np.full(len(src), np.nan)
    if len(src) < length:
        return out
    out[length - 1] = np.mean(src[:length])
    for i in range(length, len(src)):
        out[i] = (out[i - 1] * (length - 1) + src[i]) / length
    return out


def true_range(h, l, c):
    tr = np.empty(len(h))
    tr[0] = h[0] - l[0]
    for i in range(1, len(h)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    return tr


# ======================= STRATEGY (1:1 port of Pine) =======================
def scan_symbol(df):
    """Returns list of confirmed signals found in the dataframe (closed bars only)."""
    n = len(df)
    if n < max(EMA_LEN, ATR_LEN, VOL_LEN) + 5:
        return []

    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    v = df["Volume"].fillna(0).to_numpy(float)
    idx = df.index

    ema = pine_ema(c, EMA_LEN)
    atr = pine_rma(true_range(h, l, c), ATR_LEN)
    vol_ma = pd.Series(v).rolling(VOL_LEN).mean().to_numpy()

    mother_high = mother_low = np.nan
    mother_bar = None
    mother_atr_x = mother_body = 0.0
    waiting_breakout = False
    waiting_conf = False
    breakout_dir = 0
    breakout_bar = None
    breakout_level = breakout_sl = np.nan
    breakout_vol_ratio = 1.0

    signals = []

    for i in range(n):
        # ---- expire old setup ----
        if waiting_breakout and mother_bar is not None and i - mother_bar > MAX_WAIT:
            waiting_breakout = False
            waiting_conf = False
            breakout_dir = 0

        # ---- mother candle candidate ----
        rng = h[i] - l[i]
        body = abs(c[i] - o[i])
        body_pct = body / rng * 100 if rng > 0 else 0.0
        cand = (
            not np.isnan(atr[i])
            and rng >= atr[i] * MIN_MOTHER_ATR
            and rng <= atr[i] * MAX_MOTHER_ATR
            and body_pct >= MIN_BODY_PCT
        )
        if cand and not waiting_breakout and not waiting_conf:
            mother_high = h[i]
            mother_low = l[i]
            mother_bar = i
            mother_atr_x = rng / atr[i]
            mother_body = body_pct
            waiting_breakout = True

        # ---- filters ----
        vol_ok = (not USE_VOLUME) or (not np.isnan(vol_ma[i]) and v[i] >= vol_ma[i] * VOL_MULT)
        bull_ok = (not USE_EMA) or (c[i] > ema[i])
        bear_ok = (not USE_EMA) or (c[i] < ema[i])

        prev_c = c[i - 1] if i > 0 else np.nan
        break_up = (
            waiting_breakout and i > mother_bar and c[i] > mother_high and prev_c <= mother_high
        )
        break_down = (
            waiting_breakout and i > mother_bar and c[i] < mother_low and prev_c >= mother_low
        )

        # ---- register first breakout (no signal yet) ----
        if break_up and vol_ok and bull_ok:
            waiting_breakout = False
            waiting_conf = True
            breakout_dir = 1
            breakout_bar = i
            breakout_level = mother_high
            breakout_sl = mother_low
            breakout_vol_ratio = v[i] / vol_ma[i] if vol_ma[i] and vol_ma[i] > 0 else 1.0

        if break_down and vol_ok and bear_ok:
            waiting_breakout = False
            waiting_conf = True
            breakout_dir = -1
            breakout_bar = i
            breakout_level = mother_low
            breakout_sl = mother_high
            breakout_vol_ratio = v[i] / vol_ma[i] if vol_ma[i] and vol_ma[i] > 0 else 1.0

        # ---- confirmation candle ----
        if waiting_conf and i == breakout_bar + 1:
            fired = False
            if breakout_dir == 1 and c[i] > breakout_level and l[i] > breakout_level:
                fired = True
            elif breakout_dir == -1 and c[i] < breakout_level and h[i] < breakout_level:
                fired = True

            if fired:
                conf_rng = h[i] - l[i]
                conf_body = abs(c[i] - o[i]) / conf_rng * 100 if conf_rng > 0 else 0.0
                signals.append({
                    "dir": breakout_dir,
                    "time": idx[i],
                    "breakout_time": idx[breakout_bar],
                    "mother_time": idx[mother_bar],
                    "mother_high": float(mother_high),
                    "mother_low": float(mother_low),
                    "entry": float(c[i]),
                    "sl": float(breakout_sl),
                    "breakout_level": float(breakout_level),
                    "vol_ratio": float(breakout_vol_ratio),
                    "mother_atr": float(mother_atr_x),
                    "mother_body": float(mother_body),
                    "ema_dist": float(abs(c[i] - ema[i]) / atr[i]) if atr[i] > 0 else 0.0,
                    "conf_body": float(conf_body),
                })
            waiting_conf = False

    return signals


def quality_score(s):
    """Extra quality score /10 (my addition, does NOT change the Pine signal)."""
    q = 0
    q += 2 if s["vol_ratio"] >= 1.5 else (1 if s["vol_ratio"] >= 1.0 else 0)
    q += 2 if 1.2 <= s["mother_atr"] <= 2.0 else (1 if 1.0 <= s["mother_atr"] <= 2.5 else 0)
    q += 2 if s["mother_body"] >= 70 else (1 if s["mother_body"] >= 60 else 0)
    q += 2 if s["ema_dist"] >= 0.75 else (1 if s["ema_dist"] >= 0.25 else 0)
    q += 2 if s["conf_body"] >= 60 else (1 if s["conf_body"] >= 40 else 0)
    return q


# ======================= DATA =======================
def clean_df(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        return None
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(IST)
    else:
        df.index = df.index.tz_convert(IST)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.between_time("09:15", "15:29")
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df if not df.empty else None


def drop_open_bar(df, now):
    """Remove the candle that is still forming (NSE last 30m candle is 15:15-15:30)."""
    start = pd.Series(df.index, index=df.index)
    end = start + pd.Timedelta(minutes=TIMEFRAME_MIN)
    mkt_close = pd.Series(df.index.normalize() + pd.Timedelta(hours=15, minutes=30), index=df.index)
    end = end.where(end <= mkt_close, mkt_close)
    return df[end <= now]


def fetch_all():
    out = {}
    names = list(SYMBOLS.keys())
    for b in range(0, len(names), BATCH_SIZE):
        batch = names[b:b + BATCH_SIZE]
        tickers = [SYMBOLS[s]["yf"] for s in batch]
        data = None
        for attempt in range(2):
            try:
                data = yf.download(
                    tickers, period=HISTORY_PERIOD, interval=f"{TIMEFRAME_MIN}m",
                    group_by="ticker", auto_adjust=False, progress=False, threads=True,
                )
                break
            except Exception as e:
                print(f"batch download error ({attempt + 1}): {e}")
                time.sleep(3)
        for s in batch:
            t = SYMBOLS[s]["yf"]
            df = None
            try:
                if data is not None and not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        if t in data.columns.get_level_values(0):
                            df = data[t]
                    else:
                        df = data
                df = clean_df(df)
                if df is None:  # single retry
                    df = clean_df(yf.Ticker(t).history(
                        period=HISTORY_PERIOD, interval=f"{TIMEFRAME_MIN}m", auto_adjust=False))
            except Exception as e:
                print(f"{s}: fetch failed ({e})")
                df = None
            if df is not None:
                out[s] = df
            else:
                print(f"{s}: no data, skipped")
    return out


# ======================= TELEGRAM =======================
def send_telegram(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    for _ in range(5):
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 200:
                return True
            if r.status_code == 429:
                wait = r.json().get("parameters", {}).get("retry_after", 5)
                time.sleep(wait + 1)
                continue
            print("telegram error:", r.status_code, r.text[:200])
        except Exception as e:
            print("telegram exception:", e)
        time.sleep(2)
    return False


def fmt_price(x):
    return f"{x:,.2f}"


def fmt_candle(ts):
    end = ts + pd.Timedelta(minutes=TIMEFRAME_MIN)
    return f"{ts.strftime('%d %b %Y')} | {ts.strftime('%H:%M')}-{end.strftime('%H:%M')}"


def build_message(name, sig):
    is_ce = sig["dir"] == 1
    head = "🟢 <b>BUY CE</b>" if is_ce else "🔴 <b>BUY PE</b>"
    entry, sl = sig["entry"], sig["sl"]
    risk = abs(entry - sl)
    sign = 1 if is_ce else -1
    targets = [entry + sign * risk * m for m in TARGET_RR]
    q = quality_score(sig)
    tv = SYMBOLS[name]["tv"]
    link = html.escape(f"https://in.tradingview.com/chart/?symbol={tv}&interval={TIMEFRAME_MIN}")

    lines = [
        f"{head} | <b>{html.escape(name)}</b> | {TIMEFRAME_MIN}m",
        "",
        f"🕒 Confirm candle: {fmt_candle(sig['time'])}",
        f"⚡ Breakout candle: {sig['breakout_time'].strftime('%H:%M')} ({sig['breakout_time'].strftime('%d %b')})",
        f"👩 Mother candle: {sig['mother_time'].strftime('%H:%M')} ({sig['mother_time'].strftime('%d %b')}) "
        f"| H {fmt_price(sig['mother_high'])} | L {fmt_price(sig['mother_low'])}",
        "",
        f"💰 Entry (confirm close): {fmt_price(entry)}",
        f"🛑 Stoploss: {fmt_price(sl)}  (risk {fmt_price(risk)})",
        f"🎯 T1 (1:1): {fmt_price(targets[0])}",
        f"🎯 T2 (1:2): {fmt_price(targets[1])}",
        f"🎯 T3 (1:3): {fmt_price(targets[2])}",
        "",
        f"⭐ Quality: {q}/10 | Vol x{sig['vol_ratio']:.1f} | Mother {sig['mother_atr']:.1f} ATR",
        f'📈 <a href="{link}">Open TradingView chart</a>',
    ]
    return "\n".join(lines)


# ======================= STATE =======================
def load_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None


def save_state(sent):
    keep = sorted(sent)
    cutoff = (pd.Timestamp.now(tz=IST) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    keep = [k for k in keep if k.split("|")[2][:10] >= cutoff]
    with open(STATE_FILE, "w") as f:
        json.dump({"sent": keep}, f, indent=0)


# ======================= MAIN =======================
def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID secrets missing")
        sys.exit(1)

    state = load_state()
    first_run = state is None
    sent = set(state["sent"]) if state else set()

    now = pd.Timestamp.now(tz=IST)
    today = now.date()

    raw = fetch_all()
    data = {}
    for name, df in raw.items():
        df = drop_open_bar(df, now)
        if df is not None and not df.empty:
            data[name] = df

    print(f"Loaded {len(data)}/{len(SYMBOLS)} symbols")
    if len(data) < len(SYMBOLS) * 0.5:
        send_telegram(token, chat_id,
                      f"⚠️ Data problem: only {len(data)}/{len(SYMBOLS)} symbols loaded. "
                      f"Yahoo Finance might be slow/down.")
        if not data:
            sys.exit(0)

    # allowed signal dates: previous trading day + today
    days = set()
    for df in data.values():
        days.update(df.index.date)
    prev_days = [d for d in days if d < today]
    prev_day = max(prev_days) if prev_days else None
    allowed = {today}
    if prev_day:
        allowed.add(prev_day)

    new_signals = []
    for name, df in data.items():
        try:
            sigs = scan_symbol(df)
        except Exception as e:
            print(f"{name}: scan error {e}")
            continue
        for s in sigs:
            if s["time"].date() not in allowed:
                continue
            key = f"{name}|{'CE' if s['dir'] == 1 else 'PE'}|{s['time'].strftime('%Y-%m-%d %H:%M')}"
            if key in sent:
                continue
            new_signals.append((s["time"], name, key, s))

    new_signals.sort(key=lambda x: (x[0], x[1]))
    print(f"New signals: {len(new_signals)}")

    if first_run:
        pd_txt = prev_day.strftime("%d %b %Y") if prev_day else "-"
        send_telegram(
            token, chat_id,
            f"✅ <b>ROMYO Mother Candle Scanner started</b>\n"
            f"Timeframe: {TIMEFRAME_MIN}m | Symbols: {len(data)}\n"
            f"First run: signals of previous trading day ({pd_txt}) + today. "
            f"After this only live signals.\nSignals found: {len(new_signals)}",
        )

    for _, name, key, s in new_signals:
        ok = send_telegram(token, chat_id, build_message(name, s))
        if ok:
            sent.add(key)
        time.sleep(0.6)

    save_state(sent)
    print("Done.")


if __name__ == "__main__":
    main()
