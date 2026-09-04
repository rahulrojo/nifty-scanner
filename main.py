from datetime import datetime, time, timedelta, timezone
import numpy as np
import pandas as pd
import yfinance as yf

# ==============================================================================
# CONFIGURATION & WATCHLIST
# ==============================================================================
# 1. TEST MODE: Set to True for testing. Set to False once verified.
TEST_MODE = True

# 2. MARKET HOURS (IST)
MARKET_START = time(9, 0)
MARKET_END = time(15, 40)
IST = timezone(timedelta(hours=5, minutes=30))

# 3. MAJOR OPTION STOCKS & INDICES WATCHLIST
WATCHLIST = [
    "^NSEI",  # Nifty 50 Index
    "^NSEBANK",  # Bank Nifty Index
    "RELIANCE.NS",  # Reliance Industries
    "HDFCBANK.NS",  # HDFC Bank
    "ICICIBANK.NS",  # ICICI Bank
    "INFY.NS",  # Infosys
    "TCS.NS",  # TCS
    "SBIN.NS",  # State Bank of India
    "BHARTIARTL.NS",  # Bharti Airtel
]


# ==============================================================================
# PINE SCRIPT COMPATIBLE INDICATOR FUNCTIONS
# ==============================================================================
def pine_rma(series: pd.Series, length: int) -> pd.Series:
    """Pine Script's ta.rma / Wilder's Smoothing (alpha = 1/length)"""
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def pine_ema(series: pd.Series, length: int) -> pd.Series:
    """Pine Script's ta.ema"""
    return series.ewm(span=length, adjust=False).mean()


def pine_atr(df: pd.DataFrame, length: int) -> pd.Series:
    """Pine Script's ta.atr(length) using RMA on True Range"""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return pine_rma(tr, length)


def pine_linreg_0(series: pd.Series, length: int) -> pd.Series:
    """Pine Script's ta.linreg(source, length, 0) - Linear Regression Endpoint"""
    x = np.arange(length)
    x_mean = (length - 1) / 2.0
    x_var = np.sum((x - x_mean) ** 2)

    def calc_endpoint(window):
        y_mean = np.mean(window)
        slope = np.sum((x - x_mean) * (window - y_mean)) / x_var
        return y_mean + slope * x_mean

    return series.rolling(window=length).apply(calc_endpoint, raw=True)


# ==============================================================================
# SQUEEZE & TRAILING ENGINE
# ==============================================================================
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

    # 1. BOLLINGER BANDS
    bb_mid = close.rolling(window=length).mean()
    bb_std = close.rolling(window=length).std(ddof=1)
    bb_upper = bb_mid + (mult_bb * bb_std)
    bb_lower = bb_mid - (mult_bb * bb_std)

    # 2. KELTNER CHANNELS
    kc_ema = pine_ema(close, length)
    kc_range = pine_atr(df, length)
    kc_upper = kc_ema + (kc_range * mult_kc)
    kc_lower = kc_ema - (kc_range * mult_kc)

    # 3. SQUEEZE CONDITION
    is_squeezed = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    squeeze_release = is_squeezed.shift(1) & (~is_squeezed)

    # 4. TTM MOMENTUM OSCILLATOR
    highest_high = high.rolling(length).max()
    lowest_low = low.rolling(length).min()
    donchian_avg = (highest_high + lowest_low) / 2.0
    mom_src = close - ((donchian_avg + kc_ema) / 2.0)
    mom = pine_linreg_0(mom_src, length)

    # 5. RAW SIGNALS
    buy_signal = squeeze_release & (mom > 0) & (close > kc_ema)
    sell_signal = squeeze_release & (mom < 0) & (close < kc_ema)

    atr_val = pine_atr(df, atr_period)

    # 6. EXACT PINE SCRIPT STATE MACHINE
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


# ==============================================================================
# MAIN EXECUTION & TIME FILTER
# ==============================================================================
def is_market_open() -> bool:
    """Checks if current time is within Indian Stock Market hours (Mon-Fri 09:00 - 15:40 IST)"""
    now = datetime.now(IST)

    # Weekend Check (5 = Saturday, 6 = Sunday)
    if now.weekday() >= 5:
        return False

    current_time = now.time()
    return MARKET_START <= current_time <= MARKET_END


def main():
    now = datetime.now(IST)
    print(
        f"[{now.strftime('%Y-%m-%d %H:%M:%S IST')}] Squeeze Engine Running..."
    )

    # --- 1. TEST MODE MESSAGE ---
    if TEST_MODE:
        print(
            "------------------------------------------------------------------"
        )
        print("🧪 [TEST MODE ACTIVE]: Your main.py executed successfully!")
        print(
            "   (Is message ke aane ke baad main.py me TEST_MODE = False kar dena)"
        )
        print(
            "------------------------------------------------------------------"
        )

    # --- 2. MARKET HOURS CHECK ---
    if not is_market_open() and not TEST_MODE:
        print(
            "⏸ Market is currently CLOSED (Active Hours: Mon-Fri 09:00 AM to 03:40 PM IST). Skipping scanner."
        )
        return

    # --- 3. SCAN WATCHLIST FOR NEW 5M SIGNALS ---
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

            # Check the last 2 closed 5m candles to avoid missing latest signal
            latest_candles = df.iloc[-2:]

            for idx, row in latest_candles.iterrows():
                if row["Valid_Buy"]:
                    signals_found += 1
                    print(
                        f"🚀 [BIG MOVE BUY ALERT] | Asset: {ticker} | Time: {idx} | Entry: {row['Close']:.2f} | Trail SL: {row['Trail_SL']:.2f}"
                    )
                elif row["Valid_Sell"]:
                    signals_found += 1
                    print(
                        f"💥 [BIG MOVE SELL ALERT] | Asset: {ticker} | Time: {idx} | Entry: {row['Close']:.2f} | Trail SL: {row['Trail_SL']:.2f}"
                    )

        except Exception as e:
            print(f"Error scanning {ticker}: {e}")

    if signals_found == 0:
        print(
            f"✅ Scan completed. No new 5-minute signals found across Watchlist."
        )


if __name__ == "__main__":
    main()
