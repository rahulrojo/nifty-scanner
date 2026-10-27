"""
ROMYO Confirmed Mother Candle Breakout  (30m)  ->  Telegram scanner
Line-by-line port of the Pine v6 script.

Pine logic (per closed candle, in this order):
  1. expire   : waitingBreakout and bar - motherBar > maxWait  -> reset setup
  2. mother   : candle range 1.0-2.5 ATR and body >= 50%, only if no setup active
  3. breakout : first CLOSE outside mother (prev close still inside), volume >= avg(20),
                EMA21 trend filter on the BREAKOUT candle close
  4. confirm  : very next candle: close AND low > mother high (CE)  /  close AND high < mother low (PE)
  Entry = confirm close, SL = opposite side of mother candle.

Candles: NSE 30m grid 09:15, 09:45 ... 15:15 (last candle 15 min), same as TradingView.

Modes (RUN_MODE): auto  = first run previous working day + today, then only new live signals
                  backfill = previous working day dobara
                  today = aaj ke saare signals dobara (TradingView se match karne ke liye)
DEBUG_SYMBOL=PIDILITIND -> logs me 30m candles + EMA/ATR/VolMA print hota hai.

Secrets: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""
import os
import sys
import json
import time
import html
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
