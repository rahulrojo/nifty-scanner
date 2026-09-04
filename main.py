from datetime import datetime, time, timedelta, timezone
import numpy as np
import pandas as pd
import yfinance as yf

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TEST_MODE = True

MARKET_START = time(9, 0)
MARKET_END = time(15, 40)
IST = timezone(timedelta(hours=5, minutes=30))

WATCHLIST = [
    "^NSEI",        # Nifty 50 Index
    "^NSEBANK",     # Bank Nifty Index
    "RELIANCE.NS",  # Reliance
    "HDFCBANK.NS",  # HDFC Bank
    "ICICIBANK.NS",  # ICICI Bank
    "INFY.NS",       # Infosys
    "TCS.NS",        # TCS
    "SBIN.NS",       # SBI
    "BHARTIARTL.NS"  # Airtel
]

# ==============================================================================
# INDICATOR FUNCTIONS
# ==============================================================================
def pine_rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1.0 / length, adjust=False).mean()

def pine_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def pine_atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return pine_rma(tr, length)

def pine_linreg_0(series: pd.Series, length: int) -> pd.Series:
    x = np.arange(length)
    x_mean = (length - 1) / 2.0
    x_var = np.sum((x - x_mean)**2)
    
    def calc_endpoint(window):
        y_mean = np.mean(window)
        slope = np.sum((x - x_mean) * (window - y_mean)) / x_var
        return y_mean + (slope * x_mean)

    return series.rolling(window=length).apply(calc_endpoint, raw=True)

# ==============================================================================
# ENGINE LOGIC
# ==============================================================================
def run_squeeze_trail_engine(df: pd.DataFrame, length=20, mult_bb=2.0, mult_kc=1.5, atr_period=14, atr_mult=2.0):
    close = df['Close']
    high = df['High']
    low = df['Low']

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

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    now = datetime.now(IST)
    print(f"\n==================================================================")
    print(f"🚀 SQUEEZE ENGINE RUNNING | {now.strftime('%Y-%m-%d %H:%M:%S IST')}")
    print(f"==================================================================")

    # 1. FORCED TEST MESSAGE PRINT (ALWAYS EXECUTED IF TEST_MODE IS TRUE)
    if TEST_MODE:
        print("\n🧪 [TEST ALERT VERIFICATION]:")
        print("------------------------------------------------------------------")
        print(f"🚀 [TEST BUY ALERT]  | Asset: ^NSEI | Time: {now.strftime('%Y-%m-%d %H:%M')} | Entry: 24500.50 | Trail SL: 24420.00")
        print(f"💥 [TEST SELL ALERT] | Asset: ^NSEBANK | Time: {now.strftime('%Y-%m-%d %H:%M')} | Entry: 51200.10 | Trail SL: 51350.00")
        print("------------------------------------------------------------------\n")

    print("🔍 Scanning Watchlist Assets...\n")
    total_signals = 0

    for ticker in WATCHLIST:
        try:
            # Download 5-day data
            df = yf.download(ticker, period="5d", interval="5m", progress=False)
            
            if df.empty:
                print(f"⚠️ {ticker}: No data received from Yahoo Finance.")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.dropna()
            df = run_squeeze_trail_engine(df)

            # Filter signals across entire 5-day dataset to guarantee historical signals
            signals_df = df[df["Valid_Buy"] | df["Valid_Sell"]]

            if not signals_df.empty:
                print(f"📊 --- Signals found for {ticker} ---")
                for idx, row in signals_df.iterrows():
                    total_signals += 1
                    candle_time = idx.strftime('%Y-%m-%d %H:%M')
                    if row["Valid_Buy"]:
                        print(f"🚀 [BUY]  | Time: {candle_time} | Candle Close: {row['Close']:.2f} | Trail SL: {row['Trail_SL']:.2f}")
                    elif row["Valid_Sell"]:
                        print(f"💥 [SELL] | Time: {candle_time} | Candle Close: {row['Close']:.2f} | Trail SL: {row['Trail_SL']:.2f}")
                print("")
            else:
                print(f"ℹ️ {ticker}: Scanned {len(df)} candles. No Squeeze Signals found.")

        except Exception as e:
            print(f"❌ Error scanning {ticker}: {e}")

    print("==================================================================")
    print(f"✅ Scan Complete. Total Signals Identified: {total_signals}")
    print("==================================================================\n")

if __name__ == "__main__":
    main()
