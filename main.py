from datetime import datetime, time, timedelta, timezone
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
import yfinance as yf

# ==============================================================================
# 1. TELEGRAM & SCRIPT CONFIGURATION
# ==============================================================================
# Yahan apna Token aur Chat ID dalein:
TELEGRAM_BOT_TOKEN = "8828219474:AAFlDta3ZZd6BpSNLg3lu7OTghoL9BQBgOQ"
TELEGRAM_CHAT_ID = "1046208187"

# TEST_MODE:
# - True  : Run karte hi 1 Test Telegram alert bhejega connection check karne ke liye.
# - False : Live market scanning ke liye setup set ho jayega.
TEST_MODE = True

MARKET_START = time(9, 0)
MARKET_END = time(15, 40)
IST = timezone(timedelta(hours=5, minutes=30))

WATCHLIST = [
    "^NSEI",  # Nifty 50 Index
    "^NSEBANK",  # Bank Nifty Index
    "RELIANCE.NS",  # Reliance
    "HDFCBANK.NS",  # HDFC Bank
    "ICICIBANK.NS",  # ICICI Bank
    "INFY.NS",  # Infosys
    "TCS.NS",  # TCS
    "SBIN.NS",  # SBI
    "BHARTIARTL.NS",  # Airtel
]


# ==============================================================================
# 2. TELEGRAM SENDER FUNCTION
# ==============================================================================
def send_telegram_alert(message: str):
    """Telegram Bot API se message bhejta hai."""
    if (
        not TELEGRAM_BOT_TOKEN
        or TELEGRAM_BOT_TOKEN == "8828219474:AAFlDta3ZZd6BpSNLg3lu7OTghoL9BQBgOQ"
        or not TELEGRAM_CHAT_ID
        or TELEGRAM_CHAT_ID == "-1003921675472"
    ):
        print(
            "⚠️ Telegram Token ya Chat ID missing hai! Top par add karein."
        )
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = urllib.parse.urlencode(
        {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    ).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload)
        with urllib.request.urlopen(req) as response:
            if response.getcode() == 200:
                print("📲 Telegram Alert Sent Successfully!")
    except Exception as e:
        print(f"❌ Telegram Sending Error: {e}")


# ==============================================================================
# 3. PINE SCRIPT INDICATOR ENGINE
# ==============================================================================
def pine_rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def pine_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def pine_atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return pine_rma(tr, length)


def pine_linreg_0(series: pd.Series, length: int) -> pd.Series:
    x = np.arange(length)
    x_mean = (length - 1) / 2.0
    x_var = np.sum((x - x_mean) ** 2)

    def calc_endpoint(window):
        y_mean = np.mean(window)
        slope = np.sum((x - x_mean) * (window - y_mean)) / x_var
        return y_mean + (slope * x_mean)

    return series.rolling(window=length).apply(calc_endpoint, raw=True)


def run_squeeze_trail_engine(
    df: pd.DataFrame,
    length=20,
    mult_bb=2.0,
    mult_kc=1.5,
    atr_period=14,
    atr_mult=2.0,
):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    bb_mid = close.rolling(window=length).mean()
    bb_std = close.rolling(window=length).std(ddof=1)
    bb_upper = bb_mid + (mult_bb * bb_std)
    bb_lower = bb_mid - (mult_bb * bb_std)

    kc_ema = pine_ema(close, length)
    kc_range = pine_atr(df, length)
    kc_upper = kc_ema + (kc_range * mult_kc)
    kc_lower = kc_ema - (kc_range * mult_kc)

    is_squeezed = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    squeeze_release = is_squeezed.shift(1) & (~is_squeezed)

    highest_high = high.rolling(length).max()
    lowest_low = low.rolling(length).min()
    donchian_avg = (highest_high + lowest_low) / 2.0
    mom_src = close - ((donchian_avg + kc_ema) / 2.0)
    mom = pine_linreg_0(mom_src, length)

    buy_signal = squeeze_release & (mom > 0) & (close > kc_ema)
    sell_signal = squeeze_release & (mom < 0) & (close < kc_ema)

    atr_val = pine_atr(df, atr_period)

    n = len(df)
    pos_state = ["NONE"] * n
    trail_sl = [np.nan] * n
    valid_buy = [False] * n
    valid_sell = [False] * n

    current_pos = "NONE"
    current_sl = 0.0

    for i in range(n):
        prev_pos = current_pos

        b_sig = buy_signal.iloc[i]
        s_sig = sell_signal.iloc[i]
        c_price = close.iloc[i]
        h_price = high.iloc[i]
        l_price = low.iloc[i]
        a_val = atr_val.iloc[i]

        if pd.isna(a_val):
            pos_state[i] = current_pos
            continue

        if b_sig and current_pos == "NONE":
            current_pos = "LONG"
            current_sl = c_price - (a_val * atr_mult)

        if s_sig and current_pos == "NONE":
            current_pos = "SHORT"
            current_sl = c_price + (a_val * atr_mult)

        if current_pos == "LONG":
            current_sl = max(current_sl, c_price - (a_val * atr_mult))
            if l_price <= current_sl:
                current_pos = "NONE"

        if current_pos == "SHORT":
            current_sl = min(current_sl, c_price + (a_val * atr_mult))
            if h_price >= current_sl:
                current_pos = "NONE"

        v_buy = b_sig and (current_pos == "LONG") and (prev_pos == "NONE")
        v_sell = s_sig and (current_pos == "SHORT") and (prev_pos == "NONE")

        pos_state[i] = current_pos
        trail_sl[i] = current_sl if current_pos != "NONE" else np.nan
        valid_buy[i] = v_buy
        valid_sell[i] = v_sell

    df["Pos_State"] = pos_state
    df["Trail_SL"] = trail_sl
    df["Valid_Buy"] = valid_buy
    df["Valid_Sell"] = valid_sell

    return df


def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return MARKET_START <= now.time() <= MARKET_END


# ==============================================================================
# 4. MAIN EXECUTION ENGINE
# ==============================================================================
def main():
    now = datetime.now(IST)
    print(
        f"\n=================================================================="
    )
    print(f"🚀 SQUEEZE ENGINE RUNNING | {now.strftime('%Y-%m-%d %H:%M:%S IST')}")
    print(
        f"=================================================================="
    )

    # 1. TEST MODE TELEGRAM CHECK
    if TEST_MODE:
        test_msg = (
            f"🧪 *TEST ALERT VERIFICATION*\n\n"
            f"Asset: `^NSEI` (Nifty 50)\n"
            f"Time: `{now.strftime('%Y-%m-%d %H:%M IST')}`\n"
            f"Signal: *BIG MOVE BUY (Test)*\n\n"
            f"Aapka Telegram Bot successfully connect ho gaya hai!"
        )
        print("\nSending Test Telegram Alert...")
        send_telegram_alert(test_msg)

    # 2. MARKET HOURS CHECK
    if not is_market_open() and not TEST_MODE:
        print("⏸ Market is CLOSED. Skipping scanner.")
        return

    # 3. LIVE SCANNER FOR LATEST 5M CANDLE
    signals_found = 0

    for ticker in WATCHLIST:
        try:
            df = yf.download(
                ticker, period="3d", interval="5m", progress=False
            )
            if df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.dropna()
            df = run_squeeze_trail_engine(df)

            # Latest 2 completed candles lookback (to avoid missing freshly closed 5m candle)
            latest_candles = df.iloc[-2:]

            for idx, row in latest_candles.iterrows():
                candle_time = idx.strftime("%Y-%m-%d %H:%M")

                if row["Valid_Buy"]:
                    signals_found += 1
                    msg = (
                        f"🚀 *BIG MOVE BUY ALERT*\n\n"
                        f"📈 *Asset:* `{ticker}`\n"
                        f"⏰ *Time:* `{candle_time}`\n"
                        f"💰 *Entry Close:* `{row['Close']:.2f}`\n"
                        f"🛡 *Trail SL:* `{row['Trail_SL']:.2f}`"
                    )
                    print(f"BUY Signal found for {ticker} at {candle_time}")
                    send_telegram_alert(msg)

                elif row["Valid_Sell"]:
                    signals_found += 1
                    msg = (
                        f"💥 *BIG MOVE SELL ALERT*\n\n"
                        f"📉 *Asset:* `{ticker}`\n"
                        f"⏰ *Time:* `{candle_time}`\n"
                        f"💰 *Entry Close:* `{row['Close']:.2f}`\n"
                        f"🛡 *Trail SL:* `{row['Trail_SL']:.2f}`"
                    )
                    print(f"SELL Signal found for {ticker} at {candle_time}")
                    send_telegram_alert(msg)

        except Exception as e:
            print(f"Error scanning {ticker}: {e}")

    print(f"\n✅ Scan Complete. Sent {signals_found} live alerts to Telegram.")


if __name__ == "__main__":
    main()
