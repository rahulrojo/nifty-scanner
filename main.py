"""
Aurora Signals v11.6.10.3 -> Python port (Aurora mode, 30m = "Intra" profile)
BUY / SELL entry hote hi Telegram alert: candle time + quality + TradingView link.

Env:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  (GitHub Secrets)
  SEND_TODAY=true   -> aaj ke saare signals bhejo
  SEND_EXITS=1      -> EXIT alerts bhi bhejo (default: sirf BUY/SELL)
  MAX_AGE_MIN=90    -> itne minute tak purane candle ka signal bhi bhejo (missed run cover)
"""
import json
import os
import time
from datetime import datetime
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import yfinance as yf

IST = "Asia/Kolkata"
TF_MIN = 30
MINTICK = 0.05
WARMUP = 80
STATE_FILE = "sent_signals.json"

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SEND_TODAY = os.getenv("SEND_TODAY", "false").lower() == "true"
SEND_EXITS = os.getenv("SEND_EXITS", "0") == "1"
MAX_AGE_MIN = int(os.getenv("MAX_AGE_MIN", "90"))

# ---- Yahoo symbol list (apni purani 100 F&O volatile list yahan paste kar sakte ho) ----
STOCKS = """RELIANCE TCS HDFCBANK ICICIBANK INFY SBIN BHARTIARTL ITC LT AXISBANK KOTAKBANK
BAJFINANCE MARUTI SUNPHARMA TATAMOTORS TATASTEEL NTPC POWERGRID ONGC COALINDIA ADANIENT
ADANIPORTS JSWSTEEL HINDALCO VEDL M&M TITAN WIPRO HCLTECH TECHM BAJAJFINSV INDUSINDBK
BANKBARODA PNB CANBK FEDERALBNK IDFCFIRSTB AUBANK SHRIRAMFIN CHOLAFIN MUTHOOTFIN
MANAPPURAM PFC RECLTD TATAPOWER ADANIPOWER BHEL SAIL NMDC GAIL IOC BPCL HINDPETRO
DLF GODREJPROP INDIGO TVSMOTOR BAJAJ-AUTO EICHERMOT ASHOKLEY BHARATFORG MOTHERSON
APOLLOHOSP DRREDDY CIPLA DIVISLAB LUPIN AUROPHARMA BIOCON TRENT ETERNAL HAL BEL
SIEMENS ABB CGPOWER POLYCAB HAVELLS DIXON PERSISTENT COFORGE NAUKRI PIDILITIND SRF
UPL TATACONSUM VBL PAYTM IEX CDSL ANGELONE HDFCLIFE SBILIFE ICICIGI SBICARD
INDHOTEL IRCTC CONCOR JINDALSTEL NATIONALUM LICHSGFIN""".split()

INDEXES = {  # yahoo -> TradingView symbol
    "^NSEI": "NSE:NIFTY",
    "^NSEBANK": "NSE:BANKNIFTY",
    "^CNXIT": "NSE:CNXIT",
}


def tv_symbol(name):
    return "NSE:" + name.replace("&", "_").replace("-", "_")


# ============================ indicator helpers ============================
def clamp(x, lo, hi):
    return np.clip(x, lo, hi)


def sma(s, n):
    return s.rolling(n).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rma(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def stdev(s, n):
    return s.rolling(n).std(ddof=0)


def crossover(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


def rsi_f(c, n=14):
    ch = c.diff()
    up = rma(ch.clip(lower=0), n)
    dn = rma((-ch).clip(lower=0), n)
    rs = up / dn.replace(0, np.nan)
    r = 100 - 100 / (1 + rs)
    return r.fillna(100.0).where(dn.notna())


def kama(src, length, fast=2, slow=30):
    x = src.values.astype(float)
    n = len(x)
    out = np.full(n, np.nan)
    d = np.abs(np.diff(x, prepend=np.nan))
    vol = pd.Series(d).rolling(length).sum().values
    ch = np.abs(x - pd.Series(x).shift(length).values)
    er = np.where(vol != 0, ch / np.where(vol == 0, 1, vol), 0.0)
    fsc, ssc = 2.0 / (fast + 1), 2.0 / (slow + 1)
    sc = (er * (fsc - ssc) + ssc) ** 2
    for t in range(n):
        if np.isnan(sc[t]):
            continue
        if t == 0 or np.isnan(out[t - 1]):
            out[t] = x[t]
        else:
            out[t] = out[t - 1] + sc[t] * (x[t] - out[t - 1])
    return pd.Series(out, index=src.index)


def barssince(cond):
    arr = cond.fillna(False).values
    out = np.full(len(arr), np.inf)
    last = None
    for i, v in enumerate(arr):
        if v:
            last = i
        if last is not None:
            out[i] = i - last
    return pd.Series(out, index=cond.index)


def supertrend(df, adx, atr, atr_slow):
    c, h, l = df["close"].values, df["high"].values, df["low"].values
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift(1)).abs(),
                    (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
    st_atr = rma(tr, 10).values
    src = (h + l) / 2.0
    adxN = np.clip((adx / 40.0).values, 0, 1)
    atrN = np.clip((atr / atr_slow).values, 0, 2)
    mult = 3.0 * (1 + 0.6 * adxN + 0.3 * atrN)
    n = len(df)
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    trend = np.ones(n)
    for i in range(n):
        if np.isnan(mult[i]) or np.isnan(st_atr[i]):
            if i > 0:
                trend[i] = trend[i - 1]
            continue
        u = src[i] - mult[i] * st_atr[i]
        pu = up[i - 1] if i > 0 and not np.isnan(up[i - 1]) else u
        if i > 0 and c[i - 1] > pu:
            u = max(u, pu)
        d = src[i] + mult[i] * st_atr[i]
        pd_ = dn[i - 1] if i > 0 and not np.isnan(dn[i - 1]) else d
        if i > 0 and c[i - 1] < pd_:
            d = min(d, pd_)
        up[i], dn[i] = u, d
        t = trend[i - 1] if i > 0 else 1
        if t == -1 and c[i] > pd_:
            t = 1
        elif t == 1 and c[i] < pu:
            t = -1
        trend[i] = t
    idx = df.index
    return pd.Series(up, idx), pd.Series(dn, idx), pd.Series(trend, idx)


def build_htf(df):
    """4H (9:15 anchored) confirmed HTF values, 1 HTF candle offset (no lookahead)."""
    h = df.resample("240min", offset="9h15min", closed="left", label="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])
    c = h["close"]
    b, bias = ema(c, 34), ema(c, 55)
    e13, e21 = ema(c, 13), ema(c, 21)
    rs = rsi_f(c, 14)
    out = pd.DataFrame({
        "base": b.shift(1), "bias": bias.shift(1), "e13": e13.shift(1), "e21": e21.shift(1),
        "rsi": rs.shift(1), "slope": b.shift(1) - b.shift(2), "close": c.shift(1),
    })
    return out.reindex(df.index, method="ffill")


# ============================ Aurora engine ============================
def analyze(df, use_vol=True):
    n = len(df)
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    idx = df.index
    c1 = c.shift(1)
    tr = pd.concat([h - l, (h - c1).abs(), (l - c1).abs()], axis=1).max(axis=1)
    atr = rma(tr, 14)
    atr_slow = sma(atr, 50)
    vr = atr / atr_slow
    ema13, ema21 = ema(c, 13), ema(c, 21)
    emaFast = ema(c, 21)  # Intra lenFast = 21
    baseline = kama(c, 55)  # Intra + Balanced style -> KAMA(55)
    macd = ema(c, 12) - ema(c, 26)
    macdBuy = (macd > 0) & (macd.shift(1) <= 0)
    macdSell = (macd < 0) & (macd.shift(1) >= 0)

    slope = baseline.diff()
    slopeTh = np.maximum(0.15 * MINTICK, atr.fillna(MINTICK) * 0.010)
    baseRising, baseFalling = slope > slopeTh, slope < -slopeTh
    slopeUp, slopeDn = baseRising, baseFalling

    # ADX
    upm, dnm = h.diff(), -l.diff()
    pdm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), index=idx)
    mdm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), index=idx)
    smTR = rma(tr, 14)
    diP = 100 * rma(pdm, 14) / smTR
    diM = 100 * rma(mdm, 14) / smTR
    dx = (100 * (diP - diM).abs() / (diP + diM).replace(0, np.nan)).fillna(0.0)
    adx = rma(dx, 14)
    adxTh = 18.0  # max(20, 18.4) - densityAdxAdj(2.0)

    # Volatility / chop / no-trade
    sma20 = sma(c, 20)
    sd20 = stdev(c, 20)
    bbU, bbL = sma20 + 2 * sd20, sma20 - 2 * sd20
    atr20 = rma(tr, 20)
    kcU, kcL = ema(c, 20) + 1.5 * atr20, ema(c, 20) - 1.5 * atr20
    volMA = sma(v, 20)
    volZ = (v - volMA) / (stdev(v, 20) + 1e-10)
    if use_vol:
        volOK = (volZ > 0.7) & (v > 1.2 * volMA)
    else:
        volOK = pd.Series(True, index=idx)
    nearBase = (c - baseline).abs() < 0.14 * atr
    chop = (c < np.minimum(kcU, bbU)) & (c > np.maximum(kcL, bbL))
    weak = (adx < adxTh) & (atr < atr_slow)
    inNoTrade = (weak & (nearBase | chop)).fillna(False)
    isChop = slope.abs() < slopeTh

    # RSI / Stoch
    rsi = rsi_f(c, 14)
    rLL, rHH = rsi.rolling(14).min(), rsi.rolling(14).max()
    stoch = ((rsi - rLL) / (rHH - rLL).replace(0, np.nan) * 100).fillna(0.0)
    k = sma(stoch, 3)
    d = sma(k, 3)
    stBuy = (k < 20) & crossover(k, d)
    stSell = (k > 80) & crossunder(k, d)

    # Regime
    bbw = (bbU - bbL) / c
    reg_score = (clamp(adx / 50.0, 0, 1) * 0.35
                 + (clamp(atr / atr_slow, 0, 2) / 2.0) * 0.25
                 + clamp(slope.abs() / (atr * 0.5), 0, 1) * 0.25
                 + (1.0 - clamp(bbw / sma(bbw, 20), 0, 2) / 2.0) * 0.15)
    regime = pd.Series(np.where(reg_score > 0.65, "trending",
                                np.where(reg_score < 0.35, "ranging", "transitional")), index=idx)
    th_mul = np.where(regime == "trending", 0.92, np.where(regime == "ranging", 1.08, 0.98))
    thConfEff = float(np.clip(0.59 * 1.05, 0.35, 0.97))
    thConfReg = np.clip(thConfEff * th_mul - 0.005 - 0.015, 0.56, 0.97)
    isBreakout = (v > volMA * 1.5) & (bbw > sma(bbw, 20) * 1.3)
    regime_ok = (regime != "ranging") | isBreakout

    # Supertrend
    st_up, st_dn, st_trend = supertrend(df, adx, atr, atr_slow)
    stTrendUp, stTrendDn = st_trend == 1, st_trend == -1
    chg = st_trend.diff()
    stFlipUp, stFlipDn = chg == 2, chg == -2
    pdn = st_dn.shift(1).fillna(st_dn)
    pup = st_up.shift(1).fillna(st_up)
    stEarlyUp = (((c > st_dn) & (c1 > pdn)) | ((c - st_dn) >= 0.15 * atr)) & ~stTrendUp & (barssince(c > st_dn) <= 2)
    stEarlyDn = (((c < st_up) & (c1 < pup)) | ((st_up - c) >= 0.15 * atr)) & ~stTrendDn & (barssince(c < st_up) <= 2)
    stRetestUp = (barssince(stFlipUp) <= 4) & (l <= st_dn) & (c > st_dn)
    stRetestDn = (barssince(stFlipDn) <= 4) & (h >= st_up) & (c < st_up)
    stStrongBuy, stStrongSell = stFlipUp | stRetestUp, stFlipDn | stRetestDn
    stExitLong = stFlipDn | ((c < baseline) & stTrendDn)
    stExitShort = stFlipUp | ((c > baseline) & stTrendUp)

    # HTF (4H confirmed)
    htf = build_htf(df)
    sB = ((htf["close"] > htf["base"]).astype(int) + (htf["close"] > htf["bias"]).astype(int)
          + (htf["e13"] > htf["e21"]).astype(int) + (htf["slope"] > 0).astype(int)
          + (htf["rsi"] > 50).astype(int))
    sS = ((htf["close"] < htf["base"]).astype(int) + (htf["close"] < htf["bias"]).astype(int)
          + (htf["e13"] < htf["e21"]).astype(int) + (htf["slope"] < 0).astype(int)
          + (htf["rsi"] < 50).astype(int))

    # MTF boost
    def relax_of(diff):
        return pd.Series(np.select([diff >= 4, diff >= 2, diff <= -4, diff <= -2],
                                   [1.0, 0.5, -1.0, -0.5], 0.0), index=idx)
    relB, relS = relax_of(sB - sS), relax_of(sS - sB)
    thConfB = np.clip(thConfReg - relB * 0.10, 0.50, 0.97)
    thConfS = np.clip(thConfReg - relS * 0.10, 0.50, 0.97)
    volZB = np.maximum(0.0, 0.7 - relB * 0.40)
    volZS = np.maximum(0.0, 0.7 - relS * 0.40)
    cdBaseB = np.maximum(2, np.floor(4 * (1 - relB * 0.5) + 0.5))
    cdBaseS = np.maximum(2, np.floor(4 * (1 - relS * 0.5) + 0.5))
    adxThB = np.maximum(5.0, adxTh - relB * 3.0)
    adxThS = np.maximum(5.0, adxTh - relS * 3.0)

    # Scoring
    hh = c > h.rolling(5).max().shift(1)
    ll = c < l.rolling(5).min().shift(1)
    sUp = clamp(slope / np.maximum(atr * 0.08, MINTICK), 0, 1)
    sDn = clamp(-slope / np.maximum(atr * 0.08, MINTICK), 0, 1)
    adxP = clamp((adx - 12.0) / 28.0, 0, 1)
    diBP = clamp((diP - diM) / 25.0, 0, 1)
    diSP = clamp((diM - diP) / 25.0, 0, 1)
    volP = clamp((volZ + 0.5) / 2.5, 0, 1) if use_vol else pd.Series(0.5, index=idx)
    stBP = pd.Series(np.where(hh, 1.0, np.where(c > ema13, 0.5, 0.0)), index=idx)
    stSP = pd.Series(np.where(ll, 1.0, np.where(c < ema13, 0.5, 0.0)), index=idx)
    score_buy = clamp(52 + 14 * sUp + 12 * adxP * diBP + 8 * volP + 8 * stBP + 6 * (sB / 5.0), 0, 100)
    score_sell = clamp(52 + 14 * sDn + 12 * adxP * diSP + 8 * volP + 8 * stSP + 6 * (sS / 5.0), 0, 100)
    confB, confS = score_buy / 100.0, score_sell / 100.0

    dirBuy = (c > emaFast) & (emaFast > baseline) & (emaFast > emaFast.shift(1)) & slopeUp
    dirSell = (c < emaFast) & (emaFast < baseline) & (emaFast < emaFast.shift(1)) & slopeDn

    # Gates
    localStrongBuy = (score_buy >= 80) & dirBuy & (adx >= adxTh)
    localStrongSell = (score_sell >= 80) & dirSell & (adx >= adxTh)
    trending = regime == "trending"
    assistB = trending & (sB == 2) & (sS < 3) & localStrongBuy
    assistS = trending & (sS == 2) & (sB < 3) & localStrongSell
    htf_ok_buy = (sB >= 3) | assistB | ((sB >= 2) & (sS < 4))
    htf_ok_sell = (sS >= 3) | assistS | ((sS >= 2) & (sB < 4))

    relief = np.where(trending, 2.5, 0.0)
    adxVolB, adxVolS = np.maximum(5.0, adxThB - relief), np.maximum(5.0, adxThS - relief)
    volAssistB = trending & (adx >= adxTh + 3) & (c > baseline) & baseRising
    volAssistS = trending & (adx >= adxTh + 3) & (c < baseline) & baseFalling
    if use_vol:
        volOK_buy = (volZ > volZB) & (v > 1.2 * volMA)
        volOK_sell = (volZ > volZS) & (v > 1.2 * volMA)
    else:
        volOK_buy = volOK_sell = pd.Series(True, index=idx)
    volExp = (v > volMA * 1.05) & (atr >= atr_slow * 0.80)
    earlyBull = (c > ema13) & (ema13 > ema21) & (diP > diM) & (adx > 13) & (c > baseline)
    earlyBear = (c < ema13) & (ema13 < ema21) & (diM > diP) & (adx > 13) & (c < baseline)
    vol_ok_buy = ((~isChop & (volOK_buy | volAssistB) & (adx >= adxVolB)) | (earlyBull & volExp))
    vol_ok_sell = ((~isChop & (volOK_sell | volAssistS) & (adx >= adxVolS)) | (earlyBear & volExp))

    # Triggers
    distBase = (c - baseline).abs()
    dynMaxB = 3.8 * (1.0 + relB * 0.20)
    dynMaxS = 3.8 * (1.0 + relS * 0.20)
    dist_ok_b = (c > baseline) & ((c - baseline) > 0.14 * atr) & (distBase < dynMaxB * atr)
    dist_ok_s = (c < baseline) & ((baseline - c) > 0.14 * atr) & (distBase < dynMaxS * atr)
    hi6, lo6 = h.rolling(6).max().shift(1), l.rolling(6).min().shift(1)
    breakB = (c > hi6) & (c > o) & (ema13 >= ema21)
    breakS = (c < lo6) & (c < o) & (ema13 <= ema21)
    pullB = (l <= ema13) & (c > ema13) & (c > o) & (ema13 > ema21)
    pullS = (h >= ema13) & (c < ema13) & (c < o) & (ema13 < ema21)
    contB = crossover(c, ema13) & (c > baseline) & (ema13 > ema21)
    contS = crossunder(c, ema13) & (c < baseline) & (ema13 < ema21)
    revB = crossover(ema13, ema21) & (c > baseline) & (diP > diM)
    revS = crossunder(ema13, ema21) & (c < baseline) & (diM > diP)
    dirOkB = (c > baseline) & (ema13 > ema21) & (diP >= diM) & (rsi < 74)
    dirOkS = (c < baseline) & (ema13 < ema21) & (diM >= diP) & (rsi > 26)
    legB = macdBuy | stBuy | stStrongBuy | stEarlyUp
    legS = macdSell | stSell | stStrongSell | stEarlyDn
    candB = dirOkB & (breakB | pullB | contB | revB | legB)
    candS = dirOkS & (breakS | pullS | contS | revS | legS)
    trigB, trigS = candB & dist_ok_b, candS & dist_ok_s
    etB = np.select([revB, breakB, pullB, contB], ["REVERSAL", "BREAKOUT", "PULLBACK", "CONTINUATION"], "LEGACY")
    etS = np.select([revS, breakS, pullS, contS], ["REVERSAL", "BREAKOUT", "PULLBACK", "CONTINUATION"], "LEGACY")

    blockedB = (~regime_ok | ~htf_ok_buy | ~vol_ok_buy | inNoTrade).fillna(True)
    blockedS = (~regime_ok | ~htf_ok_sell | ~vol_ok_sell | inNoTrade).fillna(True)
    finalBuy = trigB & ~blockedB & (confB >= np.minimum(thConfB, 0.60))
    finalSell = trigS & ~blockedS & (confS >= np.minimum(thConfS, 0.60))

    # Quality tiers (Standard / High / Exceptional)
    maxBandB, maxBandS = dynMaxB * atr, dynMaxS * atr
    highB = (confB >= 0.76) & (sB >= 4) & (adx >= 20) & (diP > diM + 5) & (distBase <= 0.70 * maxBandB)
    highS = (confS >= 0.76) & (sS >= 4) & (adx >= 20) & (diM > diP + 5) & (distBase <= 0.70 * maxBandS)
    excB = highB & (confB >= 0.86) & (sB == 5) & (volOK_buy | volOK) & trending
    excS = highS & (confS >= 0.86) & (sS == 5) & (volOK_sell | volOK) & trending

    # Exit helpers
    pivlow = l.shift(5).where(l.shift(5) <= l.rolling(11).min())
    pivhigh = h.shift(5).where(h.shift(5) >= h.rolling(11).max())
    lastLow, lastHigh = pivlow.ffill(), pivhigh.ffill()
    sl_long = (c < lastLow).fillna(False)
    sl_short = (c > lastHigh).fillna(False)
    isPara = distBase > 3.5 * atr
    force_long = (isPara & (c < o) & ((regime != "trending") | stFlipDn | macdSell)).fillna(False)
    force_short = (isPara & (c > o) & ((regime != "trending") | stFlipUp | macdBuy)).fillna(False)
    peakDecL = (crossunder(c, ema13) | crossunder(c, ema21) | stEarlyDn | stFlipDn
                | ((rsi < rsi.shift(1)) & (rsi < 58) & (c < o)))
    peakDecS = (crossover(c, ema13) | crossover(c, ema21) | stEarlyUp | stFlipUp
                | ((rsi > rsi.shift(1)) & (rsi > 42) & (c > o)))
    extraCD = np.where(vr > 1.7, 3, np.where(vr > 1.3, 2, 0))
    cdDynB = np.maximum(cdBaseB + extraCD, 2)
    cdDynS = np.maximum(cdBaseS + extraCD, 2)

    A = lambda s: np.asarray(s.fillna(False).values if hasattr(s, "fillna") else s)
    F = lambda s: np.asarray(s.values if hasattr(s, "values") else s, dtype=float)
    finalBuy_a, finalSell_a = A(finalBuy), A(finalSell)
    blockedB_a, blockedS_a = A(blockedB), A(blockedS)
    inNoTrade_a = A(inNoTrade)
    dirBuy_a, dirSell_a = A(dirBuy), A(dirSell)
    sl_l, sl_s, fl_l, fl_s = A(sl_long), A(sl_short), A(force_long), A(force_short)
    pdl, pds = A(peakDecL), A(peakDecS)
    trl, trs = A(stExitLong), A(stExitShort)
    cl, op, lo_, hi_ = F(c), F(o), F(l), F(h)
    atr_a, base_a = F(atr), F(baseline)
    e13a, e21a = F(ema13), F(ema21)
    sb, ss = F(score_buy), F(score_sell)
    cdB, cdS = F(pd.Series(cdDynB)), F(pd.Series(cdDynS))

    RISK_MULT, MIN_HOLD = 2.5, 3
    state = 0
    lastFlip = lastEntB = lastEntS = lastExit = None
    pDir, pBar = 0, None
    entry = eatr = stop = None
    peak = 0.0
    bars = 0
    events = []

    for i in range(WARMUP, n):
        if np.isnan(atr_a[i]) or np.isnan(base_a[i]):
            continue
        swingOk = lastFlip is None or i - lastFlip >= 3
        recentExit = lastExit is not None and i - lastExit < 3
        canB = ((lastFlip is None or i - lastFlip >= cdB[i]) and swingOk
                and (lastEntB is None or i - lastEntB >= 3) and not blockedB_a[i] and not recentExit)
        canS = ((lastFlip is None or i - lastFlip >= cdS[i]) and swingOk
                and (lastEntS is None or i - lastEntS >= 3) and not blockedS_a[i] and not recentExit)
        flipToShort = finalSell_a[i] and canS and (ss[i] - sb[i] >= 12)
        flipToLong = finalBuy_a[i] and canB and (sb[i] - ss[i] >= 12)

        if state == 0:
            window = pBar is not None and i == pBar + 1
            fB = (window and pDir == 1 and not inNoTrade_a[i] and dirBuy_a[i] and cl[i] > base_a[i]
                  and e13a[i] > e21a[i] and sb[i] - ss[i] >= 12)
            fS = (window and pDir == -1 and not inNoTrade_a[i] and dirSell_a[i] and cl[i] < base_a[i]
                  and e13a[i] < e21a[i] and ss[i] - sb[i] >= 12)
            pDir, pBar = 0, None
            side = None
            if (finalBuy_a[i] and canB and not inNoTrade_a[i]) or fB:
                side, state, lastEntB = "BUY", 1, i
                etype = "FLIP_RECONFIRMED" if fB else str(etB[i])
                conf = float(confB.iloc[i]); qual = "Exceptional" if excB.iloc[i] else "High" if highB.iloc[i] else "Standard"
            elif (finalSell_a[i] and canS and not inNoTrade_a[i]) or fS:
                side, state, lastEntS = "SELL", -1, i
                etype = "FLIP_RECONFIRMED" if fS else str(etS[i])
                conf = float(confS.iloc[i]); qual = "Exceptional" if excS.iloc[i] else "High" if highS.iloc[i] else "Standard"
            if side:
                lastFlip = i
                entry, eatr = cl[i], max(atr_a[i], MINTICK)
                peak, bars = 0.0, 0
                stop = entry - RISK_MULT * eatr if side == "BUY" else entry + RISK_MULT * eatr
                m = 1 if side == "BUY" else -1
                events.append(dict(
                    i=i, kind="ENTRY", side=side, price=entry, conf=int(round(conf * 100)),
                    quality=qual, etype=etype, stop=stop,
                    t1=entry + m * 0.90 * eatr, t2=entry + m * 1.65 * eatr, sn=entry + m * 2.90 * eatr,
                    regime=str(regime.iloc[i]), htf=int(sB.iloc[i] if side == "BUY" else sS.iloc[i]),
                    adx=float(adx.iloc[i]), rsi=float(rsi.iloc[i])))
        elif state == 1:
            bars += 1
            p = (cl[i] - entry) / eatr
            peak = max(peak, p)
            stopped = stop is not None and lo_[i] <= stop
            protect = peak >= 2.2 and (peak - p) >= 0.9 and pdl[i]
            hard = stopped or sl_l[i]
            close_ = hard or flipToShort or (bars >= MIN_HOLD and (protect or trl[i] or fl_l[i]))
            if close_:
                reason = ("STOP" if stopped else "STRUCTURE" if sl_l[i] else "FLIP_TO_SHORT" if flipToShort
                          else "PROFIT_PROTECT" if protect else "PARABOLIC" if fl_l[i] else "TRAIL")
                if flipToShort and not hard:
                    pDir, pBar = -1, i
                state, lastExit, stop = 0, i, None
                events.append(dict(i=i, kind="EXIT", side="BUY", price=cl[i], reason=reason))
            elif peak >= RISK_MULT:
                stop = max(stop, entry)
        elif state == -1:
            bars += 1
            p = (entry - cl[i]) / eatr
            peak = max(peak, p)
            stopped = stop is not None and hi_[i] >= stop
            protect = peak >= 2.2 and (peak - p) >= 0.9 and pds[i]
            hard = stopped or sl_s[i]
            close_ = hard or flipToLong or (bars >= MIN_HOLD and (protect or trs[i] or fl_s[i]))
            if close_:
                reason = ("STOP" if stopped else "STRUCTURE" if sl_s[i] else "FLIP_TO_LONG" if flipToLong
                          else "PROFIT_PROTECT" if protect else "PARABOLIC" if fl_s[i] else "TRAIL")
                if flipToLong and not hard:
                    pDir, pBar = 1, i
                state, lastExit, stop = 0, i, None
                events.append(dict(i=i, kind="EXIT", side="SELL", price=cl[i], reason=reason))
            elif peak >= RISK_MULT:
                stop = min(stop, entry)

    ends = bar_end_of(idx)
    for e in events:
        e["bar_start"] = idx[e["i"]]
        e["bar_end"] = ends[e["i"]]
    return events


# ============================ data / telegram ============================
def bar_end_of(idx):
    end = idx + pd.Timedelta(minutes=TF_MIN)
    cap = idx.normalize() + pd.Timedelta(hours=15, minutes=30)
    return end.where(end <= cap, cap)  # last candle 15:15-15:30


def clean(df):
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)
    # 15m -> 30m on NSE grid (9:15, 9:45, 10:15 ...), same as TradingView
    g = df.resample(f"{TF_MIN}min", offset="9h15min", closed="left", label="left")
    out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    out["n"] = g["close"].count()
    out = out.dropna(subset=["close"])
    # adhuri candle (Yahoo ne 2nd 15m bar nahi diya) hata do; 15:15 candle me sirf 1 bar hota hai
    out = out[(out["n"] >= 2) | (out.index.strftime("%H:%M") == "15:15")].drop(columns="n")
    now = pd.Timestamp.now(tz=IST)
    return out[bar_end_of(out.index) <= now]  # sirf closed candles


def download(tickers, batch=25):
    out = {}
    for s in range(0, len(tickers), batch):
        chunk = tickers[s:s + batch]
        try:
            raw = yf.download(chunk, period="60d", interval="15m", group_by="ticker",
                              auto_adjust=False, threads=True, progress=False)
        except Exception as ex:
            print("download error", chunk[:2], ex)
            continue
        for t in chunk:
            try:
                d = raw[t] if len(chunk) > 1 else raw
                d = clean(d)
                if len(d) > 150:
                    out[t] = d
            except Exception:
                print("skip", t)
        time.sleep(1)
    return out


def tg_send(text):
    if not TOKEN or not CHAT_ID:
        print("[NO TELEGRAM CREDS]\n" + text + "\n")
        return True
    for _ in range(3):
        r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data={
            "chat_id": CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=20)
        if r.status_code == 200:
            return True
        if r.status_code == 429:
            time.sleep(r.json().get("parameters", {}).get("retry_after", 5) + 1)
            continue
        print("telegram error", r.status_code, r.text[:200])
        return False
    return False


def fmt(x):
    return f"{x:,.2f}"


def build_msg(name, tv, e):
    link = f"https://in.tradingview.com/chart/?symbol={quote(tv, safe='')}&interval={TF_MIN}"
    ctime = f"{e['bar_start']:%d-%b-%Y} | {e['bar_start']:%H:%M}-{e['bar_end']:%H:%M} IST ({TF_MIN}m)"
    if e["kind"] == "EXIT":
        side = "LONG (CE)" if e["side"] == "BUY" else "SHORT (PE)"
        return (f"🔔 <b>EXIT {side}</b> — <b>{name}</b>\n🕒 Candle: {ctime}\n"
                f"💰 Price: {fmt(e['price'])}\nReason: {e['reason']}\n📈 <a href=\"{link}\">TradingView Chart</a>")
    buy = e["side"] == "BUY"
    star = {"Exceptional": "🌟🌟🌟", "High": "🌟🌟", "Standard": "🌟"}[e["quality"]]
    return (f"{'🟢' if buy else '🔴'} <b>{e['side']} ({'CE' if buy else 'PE'})</b> — <b>{name}</b>\n"
            f"🕒 Candle: {ctime}\n"
            f"{star} Quality: <b>{e['quality']}</b> | Confidence: <b>{e['conf']}%</b>\n"
            f"💰 Entry (close): {fmt(e['price'])}\n"
            f"🛑 Stop: {fmt(e['stop'])}\n"
            f"🎯 T1: {fmt(e['t1'])} | T2: {fmt(e['t2'])} | SN: {fmt(e['sn'])}\n"
            f"📊 {e['etype']} | Regime: {e['regime']} | HTF: {e['htf']}/5 | ADX {e['adx']:.0f} | RSI {e['rsi']:.0f}\n"
            f"📈 <a href=\"{link}\">TradingView Chart</a>")


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    now = pd.Timestamp.now(tz=IST)
    sent = load_state()
    universe = {f"{s}.NS": (s, tv_symbol(s), True) for s in STOCKS}
    for y, tv in INDEXES.items():
        universe[y] = (tv.split(":")[1], tv, False)  # index: volume nahi hota

    data = download(list(universe.keys()))
    print(f"Data mila: {len(data)}/{len(universe)} symbols")
    todo = []
    for y, df in data.items():
        name, tv, has_vol = universe[y]
        use_vol = has_vol and df["volume"].tail(200).sum() > 0
        try:
            events = analyze(df, use_vol=use_vol)
        except Exception as ex:
            print("analyze error", y, ex)
            continue
        for e in events:
            if e["kind"] == "EXIT" and not SEND_EXITS:
                continue
            fresh = e["bar_end"] >= now - pd.Timedelta(minutes=MAX_AGE_MIN)
            today = e["bar_start"].date() == now.date()
            if not (fresh or (SEND_TODAY and today)):
                continue
            key = f"{name}|{e['bar_start'].isoformat()}|{e['kind']}|{e['side']}"
            if key in sent:
                continue
            todo.append((e["bar_start"], key, name, tv, e))

    todo.sort(key=lambda x: x[0])
    print(f"Naye signals: {len(todo)}")
    for _, key, name, tv, e in todo:
        if tg_send(build_msg(name, tv, e)):
            sent[key] = now.isoformat()
        time.sleep(0.6)

    cutoff = now - pd.Timedelta(days=4)
    sent = {k: v for k, v in sent.items() if pd.Timestamp(v) > cutoff}
    with open(STATE_FILE, "w") as f:
        json.dump(sent, f, indent=0)


if __name__ == "__main__":
    main()
