"""
Trendslinger Options Radar - FRIENDS EDITION (hosted on Streamlit Community Cloud)
Everyone's watchlist, journal, notes and events live in their own browser session (Save/Load backup),
no Webull connection, and each person uses their own Claude key and (optional) Tradier token.

Trendslinger Options Radar  (no signals - it does the checklist legwork, you make the call)

Tabs
  Market         VIX vs 50-day MA, CNN Fear & Greed, sector strength, discount windows,
                 quad witching countdown, weekend gap warning
  Scanner        Nasdaq / S&P / Dow names graded on the course rules: weekly trend first,
                 daily confirm, RSI, 50-day MA, Fib zone, Fib + SMA confluence, ranges, wicks
  Contracts      LEAPS finder: strike target from 10-yr return (halved), open interest,
                 spread, computed delta, suggested limit, loss-limit levels, fundamentals, news
  Gamma          GEX estimate by strike (call wall, put wall, rough flip) + open interest map
  History        What the market did after past events (elections, Fed, crises, your own)
  Journal        Your trades checked against the rules (20-30% loss, roll once, 6-12 a year)
  Ask            "Ask the Radar" chat powered by the Claude API

Run (Command Prompt, from the folder holding this file):
    pip install -U streamlit yfinance plotly pandas numpy requests anthropic tzdata
    streamlit run trendslinger_options_radar.py --theme.base dark
Stop it with Ctrl+C.

Data: free Yahoo Finance. Options quotes are delayed about 15 minutes and Yahoo gives no Greeks,
so delta and gamma are calculated here with Black-Scholes from Yahoo's implied volatility.
GEX is an ESTIMATE using the usual convention (dealers long calls, short puts).
Journal and custom events save next to this file as JSON.
"""

import datetime as dt
import functools
import json
import math
import os
import pickle
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Trendslinger Options Radar", page_icon="🎯", layout="wide")
FRIENDS_EDITION = True

# ----------------------------------------------------------------- look
BG, PANEL, BORDER = "#0E161A", "#131E23", "#1E2B31"
TEXT, MUTED, ACCENT = "#E6EEF0", "#7F9399", "#2EE6D0"
UP, DOWN, WARN = "#2EE6A0", "#FF5C7A", "#F5C04A"

HERE = Path(__file__).parent
JOURNAL_FILE = HERE / "options_radar_journal.json"
EVENTS_FILE = HERE / "options_radar_events.json"
WATCHLIST_FILE = HERE / "options_radar_watchlist.json"
NOTES_FILE = HERE / "options_radar_event_notes.json"
WB_CONFIG = HERE / "webull_config.json"      # written by webull_connect.py; keep private
WB_TOKEN_DIR = HERE / "webull_token"
TRADIER_FILE = HERE / "tradier_config.json"   # your Tradier token; keep private
CACHE_DIR = HERE / "cache"                     # price-history cache; safe to delete
TV_SYMBOLS_FILE = HERE / "options_radar_tv_symbols.json"      # exact TradingView names, for export
SECTORS_FILE = HERE / "options_radar_sectors.json"            # sectors looked up for your names
WATCHLIST_BACKUP = HERE / "options_radar_watchlist_backup.json"
# Watchlist is saved as groups, like a TradingView watchlist. ^SPX = S&P 500 index, ^TNX = US 10-year yield.
DEFAULT_WATCHLIST = {
    "Technology": ["AAPL", "ADSK", "ANET", "CDNS", "COHR", "CRM", "CRWD", "MSFT", "MSTR", "ORCL", "PLTR", "SNOW"],
    "Semiconductors": ["ADI", "AMD", "AVGO", "MU", "NVDA", "QCOM", "TSM"],
    "Communication": ["GOOGL", "META"],
    "Consumer Discretionary": ["AMZN", "BYD", "DECK", "TSLA"],
    "Consumer Staples": ["COST"],
    "Health Care": ["BSX", "ISRG", "LLY", "NVO"],
    "Financials": ["GS", "KKR", "MARA", "PYPL"],
    "Industrials": ["UBER"],
    "Utilities": ["VST"],
    "ETFs & Indexes": ["QQQ", "^SPX", "VOO", "SCHD", "SPY", "^TNX"],
    "Lookout": ["BKNG", "SHOP", "JNJ", "AMGN", "JPM", "MRK", "KO", "PANW"],
}
TICKER_LABELS = {"^SPX": "SPX", "^TNX": "US10Y", "^VIX": "VIX", "^DJI": "DJI", "^NDX": "NDX", "^RUT": "RUT"}


def load_watch_groups():
    data = load_json(WATCHLIST_FILE, None)
    if isinstance(data, list):                 # older flat format
        data = {"Watchlist": data}
    if not isinstance(data, dict) or not any(data.values()):
        data = {k: list(v) for k, v in DEFAULT_WATCHLIST.items()}
    return {g: [t for t in ts if t] for g, ts in data.items()}


def flatten_groups(groups):
    seen, out = set(), []
    for ts in groups.values():
        for t in ts:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out

# ----------------------------------------------------------------- universe
UNIVERSES = {
    "Nasdaq (NQ) top 20": ["NVDA", "MSFT", "AAPL", "AMZN", "AVGO", "META", "GOOGL", "TSLA", "NFLX", "COST",
                           "PLTR", "AMD", "CSCO", "TMUS", "ASML", "MU", "PEP", "INTU", "LIN", "GOOG"],
    "S&P 500 (ES) top 20": ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK-B", "JPM",
                            "LLY", "V", "WMT", "ORCL", "XOM", "MA", "NFLX", "COST", "JNJ", "GOOG"],
    "Dow 30 (YM)": ["MMM", "AXP", "AMGN", "AMZN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO",
                    "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
                    "NKE", "NVDA", "PG", "CRM", "SHW", "TRV", "UNH", "V", "WMT", "GOOGL"],
}
UNIVERSES["All three"] = sorted({t for v in UNIVERSES.values() for t in v})

SECTOR_ETFS = {"XLK": "Tech", "SMH": "Semis", "XLC": "Communications", "XLY": "Discretionary",
               "XLF": "Financials", "XLV": "Health care", "XLI": "Industrials", "XLE": "Energy",
               "XLP": "Staples", "XLB": "Materials", "XLU": "Utilities", "XLRE": "Real estate"}

SECTOR_MAP = {
    "NVDA": "SMH", "AVGO": "SMH", "AMD": "SMH", "MU": "SMH", "ASML": "SMH",
    "MSFT": "XLK", "AAPL": "XLK", "PLTR": "XLK", "CSCO": "XLK", "INTU": "XLK", "ORCL": "XLK",
    "IBM": "XLK", "CRM": "XLK",
    "META": "XLC", "GOOGL": "XLC", "GOOG": "XLC", "NFLX": "XLC", "TMUS": "XLC", "DIS": "XLC",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY", "NKE": "XLY",
    "COST": "XLP", "PEP": "XLP", "WMT": "XLP", "KO": "XLP", "PG": "XLP",
    "BRK-B": "XLF", "JPM": "XLF", "V": "XLF", "MA": "XLF", "AXP": "XLF", "GS": "XLF", "TRV": "XLF",
    "LLY": "XLV", "JNJ": "XLV", "AMGN": "XLV", "MRK": "XLV", "UNH": "XLV",
    "MMM": "XLI", "BA": "XLI", "CAT": "XLI", "HON": "XLI",
    "XOM": "XLE", "CVX": "XLE", "LIN": "XLB", "SHW": "XLB",
    "ADI": "SMH", "QCOM": "SMH", "TSM": "SMH", "ADSK": "XLK", "ANET": "XLK", "CDNS": "XLK", "COHR": "XLK",
    "CRWD": "XLK", "SNOW": "XLK", "SHOP": "XLK", "PANW": "XLK", "DECK": "XLY", "BYD": "XLY", "BKNG": "XLY",
    "NVO": "XLV", "ISRG": "XLV", "BSX": "XLV", "KKR": "XLF", "PYPL": "XLF", "UBER": "XLI", "VST": "XLU",
}
INDEX_ETFS = ["SPY", "QQQ", "DIA"]

# Built-in history events. timing: "after_close" = news hit after the bell (measure from that day's
# close); "before_open" = news hit before or during the session (measure from the prior close).
DEFAULT_EVENTS = [
    {"date": "2008-09-15", "name": "Lehman Brothers bankruptcy", "category": "Crisis", "timing": "before_open"},
    {"date": "2008-11-04", "name": "US presidential election 2008", "category": "Election", "timing": "after_close"},
    {"date": "2011-08-05", "name": "S&P downgrades US credit rating", "category": "Macro", "timing": "after_close"},
    {"date": "2012-11-06", "name": "US presidential election 2012", "category": "Election", "timing": "after_close"},
    {"date": "2016-06-23", "name": "Brexit vote", "category": "Macro", "timing": "after_close"},
    {"date": "2016-11-08", "name": "US presidential election 2016", "category": "Election", "timing": "after_close"},
    {"date": "2019-07-31", "name": "Fed first rate cut of the cycle (2019)", "category": "Fed", "timing": "before_open"},
    {"date": "2020-02-24", "name": "COVID selloff begins", "category": "Crisis", "timing": "before_open"},
    {"date": "2020-11-03", "name": "US presidential election 2020", "category": "Election", "timing": "after_close"},
    {"date": "2022-03-16", "name": "Fed first rate hike of the cycle (2022)", "category": "Fed", "timing": "before_open"},
    {"date": "2023-03-10", "name": "Silicon Valley Bank collapse", "category": "Crisis", "timing": "before_open"},
    {"date": "2024-08-05", "name": "Yen carry trade unwind", "category": "Macro", "timing": "before_open"},
    {"date": "2024-09-18", "name": "Fed first rate cut of the cycle (2024)", "category": "Fed", "timing": "before_open"},
    {"date": "2024-11-05", "name": "US presidential election 2024", "category": "Election", "timing": "after_close"},
    {"date": "2025-01-27", "name": "DeepSeek AI selloff", "category": "Tech shock", "timing": "before_open"},
    {"date": "2025-04-02", "name": "Liberation Day tariff announcement", "category": "Tariffs/Trade", "timing": "after_close"},
    {"date": "2010-11-02", "name": "US midterm elections 2010", "category": "Midterm election", "timing": "after_close"},
    {"date": "2014-11-04", "name": "US midterm elections 2014", "category": "Midterm election", "timing": "after_close"},
    {"date": "2018-11-06", "name": "US midterm elections 2018", "category": "Midterm election", "timing": "after_close"},
    {"date": "2022-11-08", "name": "US midterm elections 2022", "category": "Midterm election", "timing": "after_close"},
    {"date": "2026-11-03", "name": "US midterm elections 2026", "category": "Midterm election", "timing": "after_close"},
]
BREAKDOWN_SYMS = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow 30", "IWM": "Russell 2000 (small caps)"}

def event_key(e):
    return f"{e['date']}|{e['name']}"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@500;700&display=swap');
.stApp {{ background: {BG}; }}
html, body, [class*="css"], .stMarkdown {{ font-family: 'JetBrains Mono', monospace; }}
.block-container {{ padding-top: 1.1rem; }}
.ts-header {{ display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:10px;
  border:1px solid {BORDER}; border-radius:12px; padding:12px 16px; background:{PANEL}; margin-bottom:10px; }}
.ts-title {{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:21px; color:{TEXT}; }}
.ts-title b {{ color:{ACCENT}; }}
.ts-chip {{ border:1px solid {BORDER}; border-radius:999px; padding:3px 10px; color:{MUTED}; font-size:12px; margin-left:6px; }}
.ts-card {{ border:1px solid {BORDER}; border-radius:12px; background:{PANEL}; padding:12px 14px; margin-bottom:10px; }}
.ts-card h4 {{ margin:0 0 6px 0; font-family:'Space Grotesk',sans-serif; font-size:14px; color:{TEXT}; }}
.ts-big {{ font-family:'Space Grotesk',sans-serif; font-size:24px; font-weight:700; }}
.ts-muted {{ color:{MUTED}; font-size:12px; }}
.ts-flag {{ border-left:3px solid {WARN}; padding:6px 10px; margin:4px 0; background:{BG}; border-radius:6px; font-size:13px; }}
.ts-good {{ border-left-color:{UP}; }}
.ts-bad {{ border-left-color:{DOWN}; }}
@media (max-width: 700px) {{
  .block-container {{ padding: 0.6rem 0.65rem 3rem 0.65rem !important; }}
  .ts-header {{ padding: 10px 12px; gap: 6px; }}
  .ts-title {{ font-size: 18px; }}
  .ts-chip {{ font-size: 11px; padding: 2px 8px; margin: 2px 4px 0 0; display: inline-block; }}
  .ts-chip.ts-wide {{ display: none; }}
  .ts-card {{ padding: 10px 12px; margin-bottom: 8px; }}
  .ts-big {{ font-size: 21px; }}
  .ts-flag {{ font-size: 12.5px; }}
  [data-baseweb="tab-list"] {{ gap: 2px; overflow-x: auto; scrollbar-width: none; }}
  [data-baseweb="tab-list"]::-webkit-scrollbar {{ display: none; }}
  button[data-baseweb="tab"] {{ padding: 6px 10px !important; white-space: nowrap; }}
  button[data-baseweb="tab"] p {{ font-size: 14px !important; }}
  [data-testid="stHorizontalBlock"] {{ gap: 0.5rem; }}
  h4 {{ font-size: 1.05rem !important; }}
  h5 {{ font-size: 0.98rem !important; }}
}}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------- small helpers
def ny_now():
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return dt.datetime.utcnow() - dt.timedelta(hours=5)


def keep(key, default):
    """Give a widget a remembered value. Lazy tabs only draw the open tab, and Streamlit forgets
    widgets that aren't drawn, so every tab widget stores its value under a w_ key that the
    keep-alive loop at the top of the page carries from run to run."""
    if key not in st.session_state:
        st.session_state[key] = default
    return key


def keep_choice(key, options, default=None):
    """Same as keep(), for pick-lists whose options can change (resets if the old pick is gone)."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = default if default in options else (options[0] if options else None)
    return key


def keep_multi(key, options, default=()):
    """Same as keep(), for multi-pick lists: drops any old pick that's no longer an option."""
    if key not in st.session_state:
        st.session_state[key] = [d for d in default if d in options]
    else:
        st.session_state[key] = [v for v in st.session_state[key] if v in options]
    return key


def show_df(df, phone=None, **kw):
    """phone = the few columns worth showing on a phone screen."""
    if PHONE and phone:
        cols = df.data.columns if hasattr(df, "data") else df.columns
        keep_cols = [c for c in phone if c in cols]
        if keep_cols:
            kw["column_order"] = keep_cols
    if "height" not in kw:                      # size small tables to their rows: no inner scrollbar
        try:
            n = len(df.data) if hasattr(df, "data") else len(df)
            if n <= 40:
                kw["height"] = 35 * (n + 1) + 3
        except Exception:
            pass
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, **kw)


def show_fig(fig):
    cfg = {"displaylogo": False}
    if PHONE:
        h = fig.layout.height or 400
        fig.update_layout(height=max(240, int(h * 0.72)), font=dict(size=11),
                          margin=dict(l=4, r=36, t=fig.layout.margin.t or 36, b=fig.layout.margin.b or 10),
                          title=dict(font=dict(size=12)))
        cfg.update({"displayModeBar": False, "scrollZoom": False})
    try:
        st.plotly_chart(fig, width="stretch", config=cfg)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config=cfg)


def style_map(styler, func, subset):
    subset = [c for c in subset if c in styler.data.columns]
    if not subset:
        return styler
    try:
        return styler.map(func, subset=subset)
    except AttributeError:
        return styler.applymap(func, subset=subset)


def color_signed(v):
    try:
        if pd.isna(v):
            return ""
        return f"color: {UP}" if float(v) > 0 else (f"color: {DOWN}" if float(v) < 0 else "")
    except Exception:
        return ""


def color_trend(v):
    return {"Up": f"color: {UP}", "Down": f"color: {DOWN}"}.get(v, f"color: {WARN}" if v else "")


def money(v):
    a = abs(v)
    txt = f"${a / 1e9:,.2f}B" if a >= 1e9 else (f"${a / 1e6:,.0f}M" if a >= 1e6 else f"${a / 1e3:,.0f}K")
    return ("-" if v < 0 else "") + txt


def yes_no(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(lambda v: "Yes" if v is True or v == 1 else ("" if v is False or v == 0 else v))
    return df


def card(title, body_html):
    st.markdown(f'<div class="ts-card"><h4>{title}</h4>{body_html}</div>', unsafe_allow_html=True)


def flag(text, kind=""):
    st.markdown(f'<div class="ts-flag {kind}">{text}</div>', unsafe_allow_html=True)


def _store():
    # Each visitor gets a private store in their own session. Nothing is written to the server.
    return st.session_state.setdefault("my_data", {})


def load_json(path, default):
    v = _store().get(Path(path).name)
    return json.loads(json.dumps(v)) if v is not None else default


def save_json(path, data):
    _store()[Path(path).name] = json.loads(json.dumps(data, default=str))


# ----------------------------------------------------------------- data
@st.cache_resource(show_spinner=False)
def _mem():
    """Process-wide stores. Thread-safe, so background threads can use them without Streamlit."""
    return {"soft": {}, "hist": {}, "miss": {}, "disk_loaded": set(), "analyze": {}, "lock": threading.RLock()}


def soft_cache(ttl, neg_ttl=60, default=None, reraise=False, empty=lambda v: False):
    """TTL cache that never lets a failure get stuck.
    Good results are kept for `ttl` seconds. A failure or an empty result is only remembered for
    `neg_ttl` seconds (so one bad Yahoo reply can't blank a panel for an hour, and a blocked site
    isn't retried on every single click either)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args):
            m = _mem()
            key = (fn.__name__, args)
            hit = m["soft"].get(key)
            now_ts = time.time()
            if hit and now_ts < hit[0]:
                if hit[2] is not None and reraise:
                    raise hit[2]
                return hit[1]
            try:
                val = fn(*args)
                bad = empty(val)
                m["soft"][key] = (now_ts + (neg_ttl if bad else ttl), val, None)
                return val
            except Exception as e:
                m["soft"][key] = (now_ts + neg_ttl, default, e)
                if reraise:
                    raise
                return default
        return wrapper
    return deco


def clear_all_caches():
    m = _mem()
    with m["lock"]:
        for k in ("soft", "hist", "miss", "analyze"):
            m[k].clear()
        m["disk_loaded"].clear()
    try:
        for f in CACHE_DIR.glob("hist_*.pkl"):
            f.unlink()
    except Exception:
        pass
    st.cache_data.clear()


def market_hours():
    n = ny_now()
    return n.weekday() < 5 and dt.time(4, 0) <= n.time() <= dt.time(20, 0)


def _hist_fresh_secs():
    n = ny_now()
    regular = n.weekday() < 5 and dt.time(9, 30) <= n.time() <= dt.time(16, 15)
    return 900 if regular else 6 * 3600


YF_ALIAS = {"^SPX": "^GSPC"}     # tried if Yahoo has nothing under the first symbol


def _download(tickers, period):
    out = {}
    if not tickers:
        return out
    try:
        data = yf.download(list(tickers), period=period, interval="1d", auto_adjust=True,
                           group_by="ticker", progress=False, threads=True)
    except Exception:
        return out
    for t in tickers:
        try:
            df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            df = df.dropna(how="all")
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if len(df) > 60:
                out[t] = df
        except KeyError:
            pass
    return out


def _hist_disk(period):
    return CACHE_DIR / f"hist_{period}.pkl"


def get_history(tickers, period: str = "10y", spinner: str = None):
    """Daily history per ticker. Only downloads what is missing or stale (so adding one stock
    fetches one stock), and keeps a copy on disk so a restart doesn't start from zero."""
    m = _mem()
    tickers = tuple(dict.fromkeys(tickers))
    now_ts, fresh = time.time(), _hist_fresh_secs()
    with m["lock"]:
        if period not in m["disk_loaded"]:
            m["disk_loaded"].add(period)
            try:
                for t, item in pickle.loads(_hist_disk(period).read_bytes()).items():
                    m["hist"].setdefault((t, period), item)
            except Exception:
                pass
        need = [t for t in tickers
                if ((t, period) not in m["hist"] or now_ts - m["hist"][(t, period)][0] > fresh)
                and now_ts - m["miss"].get((t, period), 0) > 300]
    if need:
        def fetch():
            got = _download(need, period)
            for t in need:
                if t not in got and t in YF_ALIAS:
                    alt = _download([YF_ALIAS[t]], period)
                    if YF_ALIAS[t] in alt:
                        got[t] = alt[YF_ALIAS[t]]
            return got
        if spinner:
            with st.spinner(spinner):
                got = fetch()
        else:
            got = fetch()
        with m["lock"]:
            for t in need:
                if t in got:
                    m["hist"][(t, period)] = (now_ts, got[t])
                else:
                    m["miss"][(t, period)] = now_ts          # don't hammer Yahoo for a bad symbol
            if got:
                try:
                    CACHE_DIR.mkdir(exist_ok=True)
                    snap = {t: v for (t, p), v in m["hist"].items() if p == period}
                    tmp = _hist_disk(period).with_suffix(".tmp")
                    tmp.write_bytes(pickle.dumps(snap, protocol=pickle.HIGHEST_PROTOCOL))
                    tmp.replace(_hist_disk(period))
                except Exception:
                    pass
    with m["lock"]:
        return {t: m["hist"][(t, period)][1] for t in tickers if (t, period) in m["hist"]}


CNN_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.cnn.com/",
    "Origin": "https://www.cnn.com",
}


@soft_cache(ttl=1800, neg_ttl=300, default=(None, None))
def fear_greed():
    # A failure is remembered for 5 minutes, then retried (never stuck, never retried on every click).
    base = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    last_err = None
    for url in (base, f"{base}/{(ny_now().date() - dt.timedelta(days=7)).isoformat()}"):
        try:
            r = requests.get(url, headers=CNN_HEADERS, timeout=4)
            r.raise_for_status()
            j = r.json()["fear_and_greed"]
            return round(float(j["score"])), str(j["rating"]).title()
        except Exception as e:
            last_err = e
    raise RuntimeError(f"CNN Fear & Greed unavailable: {last_err}")


def fg_rating_from(score):
    return ("Extreme fear" if score < 25 else "Fear" if score < 45 else "Neutral" if score <= 55
            else "Greed" if score < 75 else "Extreme greed")


def fear_greed_estimate(hist):
    """Backup reading built from the same ideas CNN uses: momentum, volatility,
    strength, safe-haven demand and junk-bond demand. Each part is scaled 0-100."""
    parts = {}
    spy, vix = hist.get("SPY"), hist.get("^VIX")
    if spy is not None and len(spy) > 130:
        c = spy["Close"]
        mom = c.iloc[-1] / c.rolling(125).mean().iloc[-1] - 1
        parts["Momentum (SPY vs 125-day MA)"] = float(np.clip(50 + mom * 500, 0, 100))
        parts["Strength (SPY RSI)"] = float(np.clip((rsi(c).iloc[-1] - 30) * 2.5, 0, 100))
    if vix is not None and len(vix) > 55:
        v = vix["Close"]
        vr = v.iloc[-1] / v.rolling(50).mean().iloc[-1] - 1
        parts["Volatility (VIX vs 50-day MA)"] = float(np.clip(50 - vr * 166, 0, 100))
    tlt, hyg, ief = hist.get("TLT"), hist.get("HYG"), hist.get("IEF")
    if spy is not None and tlt is not None:
        diff = (pct_change(spy["Close"], 20) or 0) - (pct_change(tlt["Close"], 20) or 0)
        parts["Safe havens (stocks vs bonds, 20d)"] = float(np.clip(50 + diff * 5, 0, 100))
    if hyg is not None and ief is not None:
        diff = (pct_change(hyg["Close"], 20) or 0) - (pct_change(ief["Close"], 20) or 0)
        parts["Junk bond demand (HYG vs IEF, 20d)"] = float(np.clip(50 + diff * 15, 0, 100))
    if not parts:
        return None, None, {}
    score = round(sum(parts.values()) / len(parts))
    return score, fg_rating_from(score), parts


@soft_cache(ttl=15, neg_ttl=45, default=None, empty=lambda v: v is None or v[1] is None)
def _one_quote_cached(t):
    return _one_quote(t)


def _one_quote(t):
    try:
        fi = yf.Ticker(t).fast_info
        last = float(fi["last_price"])
        prev = float(fi["previous_close"])
        return t, {"Last": last, "Prev close": prev, "Chg": last - prev, "Chg %": (last / prev - 1) * 100,
                   "Day low": float(fi["day_low"]), "Day high": float(fi["day_high"]),
                   "Volume": float(fi["last_volume"] or 0)}
    except Exception:
        return t, None


# ----------------------------------------------------------------- Webull (live data when connected)
ETF_SYMBOLS = set(SECTOR_ETFS) | set(INDEX_ETFS) | {"VOO", "SCHD", "IWM", "TLT", "HYG", "IEF", "VTI", "ARKK"}
WB_STATUS = {"stocks": None, "options": None, "fear_greed": None}   # None = untried, True = ok, str = error


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def get_webull():
    """Returns (DataClient or None, message). Never blocks long: if the token needs verifying,
    it says so and the Radar keeps using Yahoo. Run webull_connect.py to verify."""
    if FRIENDS_EDITION:
        return None, "Prices from Yahoo Finance unless you add a Tradier token above."
    if not WB_CONFIG.exists():
        return None, "Not connected. Run webull_connect.py once to connect Webull."
    stamp = (WB_CONFIG.stat().st_mtime, max([f.stat().st_mtime for f in WB_TOKEN_DIR.glob("*")] or [0])
             if WB_TOKEN_DIR.exists() else 0)
    return _connect_webull(stamp)


@st.cache_resource(show_spinner=False)
def _connect_webull(_stamp):
    # re-runs automatically whenever webull_config.json or the saved token changes
    try:
        cfg = json.loads(WB_CONFIG.read_text())
        from webull.core.client import ApiClient
        from webull.data.data_client import DataClient
    except ImportError:
        return None, "Webull SDK not installed. Run: pip install webull-openapi-python-sdk"
    except Exception as e:
        return None, f"Couldn't read webull_config.json: {e}"
    try:
        api = ApiClient(cfg["app_key"], cfg["app_secret"], "us",
                        token_check_duration_seconds=6, token_check_interval_seconds=2)
        api.set_token_dir(str(WB_TOKEN_DIR))
        api.add_endpoint("us", cfg.get("endpoint", "api.webull.com"))
        return DataClient(api), "Connected"
    except Exception as e:
        return None, ("Webull token needs verifying. Run webull_connect.py, approve it in the Webull app, "
                      f"then click Reconnect Webull. ({str(e)[:120]})")


def _wb_err(res):
    try:
        j = res.json()
        msg = j.get("message") or j.get("error_code") or ""
    except Exception:
        msg = ""
    if res.status_code in (401, 403):
        return f"HTTP {res.status_code}: needs the OpenAPI market-data subscription. {msg}".strip()
    return f"HTTP {res.status_code} {msg}".strip()


@st.cache_data(ttl=5, show_spinner=False)
def wb_stock_quotes(tickers: tuple):
    """Live stock/ETF snapshots from Webull. Indexes (^SPX, ^TNX, ^VIX) are left to Yahoo."""
    dc, _ = get_webull()
    if dc is None:
        return {}, None
    syms = [t for t in tickers if not t.startswith("^")]
    out, err = {}, None

    def pull(group, category):
        nonlocal err
        for i in range(0, len(group), 100):
            chunk = [t.replace("-", ".") if t == "BRK-B" else t for t in group[i:i + 100]]
            try:
                res = dc.market_data.get_snapshot(chunk, category)
            except Exception as e:
                err = str(e)[:150]
                continue
            if res.status_code != 200:
                err = _wb_err(res)
                continue
            for r in res.json() or []:
                sym = str(r.get("symbol", "")).upper().replace(".", "-")
                last, prev = _f(r.get("price")), _f(r.get("pre_close"))
                if last is None:
                    continue
                chg = _f(r.get("change"))
                ratio = _f(r.get("change_ratio"))
                out[sym] = {"Last": last, "Prev close": prev or last,
                            "Chg": chg if chg is not None else (last - prev if prev else 0.0),
                            "Chg %": ratio * 100 if ratio is not None else ((last / prev - 1) * 100 if prev else 0.0),
                            "Day low": _f(r.get("low")) or last, "Day high": _f(r.get("high")) or last,
                            "Volume": _f(r.get("volume")) or 0.0, "src": "Webull"}

    etfs = [t for t in syms if t in ETF_SYMBOLS]
    stocks = [t for t in syms if t not in ETF_SYMBOLS]
    if stocks:
        pull(stocks, "US_STOCK")
    if etfs:
        pull(etfs, "US_ETF")
    retry_etf = [t for t in stocks if t not in out]          # unknown ETFs land here
    retry_stock = [t for t in etfs if t not in out]
    if retry_etf and len(retry_etf) < len(stocks):
        pull(retry_etf, "US_ETF")
    if retry_stock and len(retry_stock) < len(etfs):
        pull(retry_stock, "US_STOCK")
    return out, (None if out else err)


@st.cache_data(ttl=15, show_spinner=False)
def wb_option_quotes(symbols: tuple):
    """Live option snapshots (bid/ask, last, volume, OI, IV and real Greeks) by OCC symbol."""
    dc, _ = get_webull()
    if dc is None or not symbols:
        return {}, None
    out, err = {}, None
    for i in range(0, len(symbols), 20):
        chunk = list(symbols[i:i + 20])
        try:
            res = dc.option_market_data.get_option_snapshot(chunk, "US_OPTION")
        except Exception as e:
            err = str(e)[:150]
            continue
        if res.status_code != 200:
            err = _wb_err(res)
            if res.status_code in (401, 403):
                break
            continue
        for r in res.json() or []:
            out[str(r.get("symbol", ""))] = r
    return out, (None if out else err)


def wb_fear_greed():
    dc, _ = get_webull()
    if dc is None:
        raise LookupError("not connected")
    return _wb_fear_greed(id(dc))


@soft_cache(ttl=600, neg_ttl=300, reraise=True)
def _wb_fear_greed(_client_id):
    dc, _ = get_webull()
    res = dc.screener.get_fear_and_greed_index()
    if res.status_code != 200:
        raise RuntimeError(_wb_err(res))
    j = res.json() or {}
    cur = j.get("current") or {}
    if cur.get("fear_greed_index") is None:
        raise RuntimeError("empty")
    hist_pts = {k: (j.get(k) or {}).get("fear_greed_index") for k in ("previous_day", "one_week_ago", "one_month_ago")}
    return int(cur["fear_greed_index"]), (cur.get("describe") or str(cur.get("grade", "")).replace("_", " ").title()), hist_pts


# ----------------------------------------------------------------- Tradier (live data with a Tradier account)
TRADIER_URL = "https://api.tradier.com/v1"


def tradier_token():
    tok = (st.session_state.get("tradier_token") or "").strip()
    if tok:
        return tok
    if not FRIENDS_EDITION:
        try:
            return (json.loads(TRADIER_FILE.read_text()).get("token") or "").strip()
        except Exception:
            return ""
    return ""


def _tradier_get(path, params, tok):
    r = requests.get(TRADIER_URL + path, params=params, timeout=10,
                     headers={"Authorization": f"Bearer {tok}", "Accept": "application/json"})
    if r.status_code != 200:
        hint = " (token wrong, or it's a Sandbox token)" if r.status_code == 401 else ""
        raise RuntimeError(f"Tradier HTTP {r.status_code}{hint}")
    return r.json()


def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


@st.cache_data(ttl=300, show_spinner=False)
def tradier_check(tok):
    try:
        _tradier_get("/markets/quotes", {"symbols": "SPY"}, tok)
        return True, "Tradier connected"
    except Exception as e:
        return False, str(e)


@st.cache_data(ttl=5, show_spinner=False)
def tradier_quotes(tickers: tuple, tok):
    syms = [t for t in tickers if not t.startswith("^")]
    out = {}
    for i in range(0, len(syms), 150):
        chunk = [t.replace("-", ".") for t in syms[i:i + 150]]
        j = _tradier_get("/markets/quotes", {"symbols": ",".join(chunk)}, tok)
        for q in _as_list((j.get("quotes") or {}).get("quote")):
            last = _f(q.get("last"))
            if last is None:
                continue
            prev = _f(q.get("prevclose")) or last
            sym = str(q.get("symbol", "")).upper()
            out[sym if sym in tickers else sym.replace(".", "-")] = {
                "Last": last, "Prev close": prev, "Chg": _f(q.get("change")) or (last - prev),
                "Chg %": _f(q.get("change_percentage")) or ((last / prev - 1) * 100 if prev else 0.0),
                "Day low": _f(q.get("low")) or last, "Day high": _f(q.get("high")) or last,
                "Volume": _f(q.get("volume")) or 0.0, "src": "Tradier",
                "bid": _f(q.get("bid")), "ask": _f(q.get("ask"))}
    return out


@st.cache_data(ttl=15, show_spinner=False)
def tradier_chain(ticker, exp, tok):
    j = _tradier_get("/markets/options/chains",
                     {"symbol": ticker.replace("-", "."), "expiration": exp, "greeks": "true"}, tok)
    out = {}
    for o in _as_list((j.get("options") or {}).get("option")):
        g = o.get("greeks") or {}
        out[str(o.get("symbol"))] = {
            "bid": o.get("bid"), "ask": o.get("ask"), "price": o.get("last"), "volume": o.get("volume"),
            "open_interest": o.get("open_interest"), "imp_vol": g.get("mid_iv") or g.get("smv_vol"),
            "delta": g.get("delta"), "gamma": g.get("gamma"), "theta": g.get("theta"), "vega": g.get("vega")}
    return out


def apply_live_quotes(df, quotes):
    df = df.copy()
    for col in ("wb_delta", "wb_gamma", "wb_theta", "wb_vega"):
        df[col] = np.nan
    for i, sym in df["contractSymbol"].astype(str).items():
        q = quotes.get(sym)
        if not q:
            continue
        for src, dst in (("bid", "bid"), ("ask", "ask"), ("price", "lastPrice"), ("volume", "volume"),
                         ("open_interest", "openInterest"), ("imp_vol", "impliedVolatility"),
                         ("delta", "wb_delta"), ("gamma", "wb_gamma"), ("theta", "wb_theta"), ("vega", "wb_vega")):
            v = _f(q.get(src))
            if v is not None:
                df.at[i, dst] = v
    return df


def enrich_chain_live(df, S, ticker, exp):
    """Webull first, then Tradier, else Yahoo as-is. Returns (df, source name or None)."""
    if df is None or df.empty or "contractSymbol" not in df:
        return df, None
    df2, ok = enrich_chain_webull(df, S)
    if ok:
        return df2, "Webull"
    tok = tradier_token()
    if tok:
        try:
            q = tradier_chain(ticker, exp, tok)
            if q:
                WB_STATUS["tradier_options"] = True
                return apply_live_quotes(df, q), "Tradier"
        except Exception as e:
            WB_STATUS["tradier_options"] = str(e)[:120]
    return df, None


def enrich_chain_webull(df, S):
    """Overwrite Yahoo's delayed chain with Webull live quotes for strikes near the money."""
    if df is None or df.empty or "contractSymbol" not in df:
        return df, False
    near = df[(df["strike"] >= S * 0.5) & (df["strike"] <= S * 1.5)]
    quotes, err = wb_option_quotes(tuple(near["contractSymbol"].astype(str)))
    WB_STATUS["options"] = True if quotes else err
    if not quotes:
        return df, False
    return apply_live_quotes(df, quotes), True


def occ_symbol(ticker, expiry, right, strike):
    d = pd.to_datetime(expiry)
    return f"{ticker.replace('-', '').upper()}{d:%y%m%d}{'C' if right == 'Call' else 'P'}{int(round(strike * 1000)):08d}"


def live_quotes(tickers: tuple):
    return _live_quotes(tickers, tradier_token())


@st.cache_data(ttl=10, show_spinner=False)
def _live_quotes(tickers: tuple, tok_key):  # no leading underscore: the token must be part of the cache key
    out, err = wb_stock_quotes(tickers)
    WB_STATUS["stocks"] = True if out else err
    tok = tradier_token()
    rest = tuple(t for t in tickers if t not in out)
    if tok and rest:
        try:
            out.update(tradier_quotes(rest, tok))
        except Exception as e:
            WB_STATUS["tradier"] = str(e)[:120]
    rest = tuple(t for t in tickers if t not in out)
    if rest:
        for res in run_parallel(_one_quote_cached, rest, workers=12):
            if res and res[1]:
                out[res[0]] = res[1]
    return out


def live_price(t, fallback):
    q = live_quotes((t,)).get(t)
    return q["Last"] if q else fallback


@soft_cache(ttl=86400, neg_ttl=120, default={}, empty=lambda v: not v)
def etf_holdings(etf):
    """Top holdings and weights from Yahoo (usually the top 10)."""
    th = yf.Ticker(etf).funds_data.top_holdings
    if th is None or th.empty:
        return {}
    col = "Holding Percent" if "Holding Percent" in th.columns else th.columns[-1]
    w = {str(sym).upper().replace(".", "-"): float(v) for sym, v in th[col].items() if pd.notna(v)}
    if sum(w.values()) > 1.5:     # came back as percents
        w = {k: v / 100 for k, v in w.items()}
    return w


@soft_cache(ttl=1800, neg_ttl=90, default=(), empty=lambda v: not v)
def _expirations(t):
    return tuple(yf.Ticker(t).options)


def expirations(t):
    return list(_expirations(t))


@soft_cache(ttl=600, neg_ttl=20, reraise=True)
def _option_chain(t, exp):
    oc = yf.Ticker(t).option_chain(exp)
    return oc.calls, oc.puts


def option_chain(t, exp):
    calls, puts = _option_chain(t, exp)
    return calls.copy(), puts.copy()


@soft_cache(ttl=21600, neg_ttl=120, default={}, empty=lambda v: not v)
def ticker_info(t):
    return yf.Ticker(t).info or {}


@soft_cache(ttl=21600, neg_ttl=120, default=None)
def earnings_date(t):
    cal = yf.Ticker(t).calendar
    if isinstance(cal, dict):
        v = cal.get("Earnings Date")
        if v:
            return pd.to_datetime(v[0]).date()
    elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
        return pd.to_datetime(cal.loc["Earnings Date"].iloc[0]).date()
    return None


@soft_cache(ttl=900, neg_ttl=120, default=[], empty=lambda v: not v)
def get_news(t):
    out = []
    items = yf.Ticker(t).news or []
    for n in items[:8]:
        c = n.get("content", n) if isinstance(n, dict) else {}
        title = c.get("title") or n.get("title")
        url = (c.get("canonicalUrl") or {}).get("url") or c.get("link") or n.get("link")
        pub = c.get("pubDate") or n.get("providerPublishTime")
        if isinstance(pub, (int, float)):
            pub = dt.datetime.fromtimestamp(pub).strftime("%Y-%m-%d")
        elif isinstance(pub, str):
            pub = pub[:10]
        src = (c.get("provider") or {}).get("displayName") or n.get("publisher", "")
        if title:
            out.append({"title": title, "url": url, "date": pub or "", "source": src})
    return out


def run_parallel(fn, items, workers=8):
    """Run fn over items on background threads; errors become None. Order is kept."""
    items = list(items)
    if not items:
        return []

    def safe(x):
        try:
            return fn(*x) if isinstance(x, tuple) else fn(x)
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
        return list(ex.map(safe, items))


# ----------------------------------------------------------------- TradingView import / export + sectors
SECTOR_GROUP = {"XLK": "Technology", "SMH": "Semiconductors", "XLC": "Communication",
                "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLV": "Health Care",
                "XLF": "Financials", "XLI": "Industrials", "XLE": "Energy", "XLU": "Utilities",
                "XLB": "Materials", "XLRE": "Real Estate"}
GROUP_TO_ETF = {v: k for k, v in SECTOR_GROUP.items()}
SECTOR_ORDER = list(SECTOR_GROUP.values()) + ["Crypto", "Futures", "Forex", "ETFs & Indexes", "Other"]
YAHOO_SECTOR = {"Technology": "Technology", "Communication Services": "Communication",
                "Consumer Cyclical": "Consumer Discretionary", "Consumer Defensive": "Consumer Staples",
                "Healthcare": "Health Care", "Financial Services": "Financials", "Industrials": "Industrials",
                "Energy": "Energy", "Utilities": "Utilities", "Basic Materials": "Materials",
                "Real Estate": "Real Estate"}
_LEARNED_SECTORS = {}          # filled from SECTORS_FILE at startup


def sector_etf(t):
    return SECTOR_MAP.get(t) or GROUP_TO_ETF.get(_LEARNED_SECTORS.get(t))


def sector_group(t):
    """Which sector a ticker belongs in. Our own map first, then what Yahoo says."""
    if t.startswith("^") or t in ETF_SYMBOLS:
        return "ETFs & Indexes"
    if t.endswith("=F"):
        return "Futures"
    if t.endswith("-USD"):
        return "Crypto"
    if t.endswith("=X"):
        return "Forex"
    if t in SECTOR_MAP:
        return SECTOR_GROUP[SECTOR_MAP[t]]
    if t in _LEARNED_SECTORS:
        return _LEARNED_SECTORS[t]
    info = ticker_info(t)
    if info.get("quoteType") in ("ETF", "MUTUALFUND", "INDEX"):
        return "ETFs & Indexes"
    if "semiconductor" in str(info.get("industry", "")).lower():
        return "Semiconductors"
    return YAHOO_SECTOR.get(info.get("sector"), "Other")


def sort_into_sectors(groups, keep=()):
    """Regroup a watchlist by sector. Groups named in `keep` (like Lookout) stay as they are."""
    tickers = [t for g, ts in groups.items() if g not in keep for t in ts]
    kept_set = {t for g in keep for t in groups.get(g, [])}
    tickers = [t for t in dict.fromkeys(tickers) if t not in kept_set]
    unknown = []
    run_parallel(ticker_info, [t for t in tickers if t not in SECTOR_MAP and not t.startswith("^")
                               and t not in ETF_SYMBOLS and t not in _LEARNED_SECTORS and "=" not in t
                               and not t.endswith("-USD")])
    by = {}
    learned = {}
    for t in tickers:
        g = sector_group(t)
        by.setdefault(g, []).append(t)
        if t not in SECTOR_MAP and g in GROUP_TO_ETF:
            learned[t] = g
        elif g == "Other":
            unknown.append(t)
    out = {g: by[g] for g in SECTOR_ORDER if g in by}
    out.update({g: by[g] for g in by if g not in out})
    for g in keep:
        if groups.get(g):
            out[g] = list(groups[g])
    return out, learned, unknown


TV_TO_YAHOO = {"SP:SPX": "^SPX", "TVC:SPX": "^SPX", "CBOE:SPX": "^SPX", "TVC:US10Y": "^TNX", "TVC:VIX": "^VIX",
               "CBOE:VIX": "^VIX", "DJ:DJI": "^DJI", "TVC:DJI": "^DJI", "NASDAQ:NDX": "^NDX", "TVC:NDX": "^NDX",
               "TVC:RUT": "^RUT", "RUSSELL:RUT": "^RUT", "TVC:DXY": "DX-Y.NYB", "TVC:GOLD": "GC=F",
               "TVC:SILVER": "SI=F", "TVC:USOIL": "CL=F"}
YAHOO_TO_TV = {"^SPX": "SP:SPX", "^GSPC": "SP:SPX", "^TNX": "TVC:US10Y", "^VIX": "TVC:VIX", "^DJI": "DJ:DJI",
               "^NDX": "NASDAQ:NDX", "^RUT": "TVC:RUT", "DX-Y.NYB": "TVC:DXY"}
TV_FUT_EXCH = {"NQ": "CME_MINI", "ES": "CME_MINI", "MNQ": "CME_MINI", "MES": "CME_MINI", "YM": "CBOT_MINI",
               "MYM": "CBOT_MINI", "RTY": "CME_MINI", "M2K": "CME_MINI", "GC": "COMEX", "MGC": "COMEX",
               "SI": "COMEX", "SIL": "COMEX", "HG": "COMEX", "CL": "NYMEX", "MCL": "NYMEX", "NG": "NYMEX"}
YF_EXCH_TO_TV = {"NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ", "NAS": "NASDAQ", "NYQ": "NYSE",
                 "NYS": "NYSE", "ASE": "AMEX", "AMX": "AMEX", "PCX": "AMEX", "BTS": "CBOE", "CXE": "CBOE"}
CRYPTO_EXCH = {"BINANCE", "COINBASE", "BITSTAMP", "KRAKEN", "BYBIT", "CRYPTO", "BITFINEX", "OKX", "GEMINI",
               "CRYPTOCAP", "BINANCEUS", "COINBASEPRO"}


def tv_to_yahoo(sym):
    """TradingView name (NASDAQ:AAPL, SP:SPX, CME_MINI:NQ1!) -> Yahoo name the Radar uses, or None."""
    sym = sym.strip().upper()
    if not sym:
        return None
    if sym in TV_TO_YAHOO:
        return TV_TO_YAHOO[sym]
    exch, _, tick = sym.rpartition(":")
    if not tick:
        return None
    if tick.endswith("1!") or tick.endswith("2!"):
        return tick[:-2] + "=F"
    if exch in CRYPTO_EXCH:
        for q in ("USDT", "USDC", "USD"):
            if tick.endswith(q) and len(tick) > len(q):
                return tick[: -len(q)] + "-USD"
        return None
    if exch in ("FX", "OANDA", "FX_IDC", "FOREXCOM", "SAXO") and len(tick) == 6:
        return tick + "=X"
    if exch in ("ECONOMICS", "FRED", "QUANDL"):
        return None
    return tick.replace(".", "-").replace("/", "-")


def parse_tradingview_file(text):
    """TradingView's exported .txt: comma (or line) separated, sections start with ###."""
    groups, tv_names, skipped = {}, {}, []
    current = "Watchlist"
    seen = set()
    for raw in text.replace("\r", "\n").replace("\n", ",").split(","):
        tok = raw.strip().strip("\ufeff")
        if not tok:
            continue
        if tok.startswith("###"):
            current = tok.lstrip("#").strip() or "Watchlist"
            continue
        y = tv_to_yahoo(tok)
        if not y:
            skipped.append(tok)
            continue
        if y in seen:
            continue
        seen.add(y)
        groups.setdefault(current, []).append(y)
        tv_names[y] = tok.upper()
    return groups, tv_names, skipped


@soft_cache(ttl=30 * 86400, neg_ttl=600, default=None)
def yahoo_exchange(t):
    return yf.Ticker(t).fast_info["exchange"]


def tv_symbol(t, known):
    if t in known:
        return known[t]
    if t in YAHOO_TO_TV:
        return YAHOO_TO_TV[t]
    if t.endswith("=F"):
        root = t[:-2]
        return f"{TV_FUT_EXCH.get(root, 'CME')}:{root}1!"
    if t.endswith("-USD"):
        return f"COINBASE:{t[:-4]}USD"
    if t.endswith("=X"):
        return f"FX:{t[:-2]}"
    ex = YF_EXCH_TO_TV.get(str(yahoo_exchange(t) or "").upper())
    return f"{ex}:{t.replace('-', '.')}" if ex else t.replace("-", ".")


def build_tradingview_file(groups, known, sections=True):
    flat = [t for ts in groups.values() for t in ts]
    run_parallel(yahoo_exchange, [t for t in dict.fromkeys(flat)
                                  if t not in known and t not in YAHOO_TO_TV and "=" not in t
                                  and not t.endswith("-USD")])
    parts, names = [], {}
    for g, ts in groups.items():
        if sections:
            parts.append("###" + g)
        for t in ts:
            names[t] = tv_symbol(t, known)
            parts.append(names[t])
    return ",".join(parts), names


# ----------------------------------------------------------------- indicators
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def to_weekly(df):
    return df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min",
                                     "Close": "last", "Volume": "sum"}).dropna()


def weekly_trend(w):
    if len(w) < 25:
        return "n/a"
    c = w["Close"]
    s10, s20 = c.rolling(10).mean().iloc[-1], c.rolling(20).mean().iloc[-1]
    hh = w["High"].iloc[-8:].max() > w["High"].iloc[-16:-8].max()
    hl = w["Low"].iloc[-8:].min() > w["Low"].iloc[-16:-8].min()
    lh = w["High"].iloc[-8:].max() < w["High"].iloc[-16:-8].max()
    ll = w["Low"].iloc[-8:].min() < w["Low"].iloc[-16:-8].min()
    if c.iloc[-1] > s20 and s10 > s20 and (hh or hl):
        return "Up"
    if c.iloc[-1] < s20 and s10 < s20 and (lh or ll):
        return "Down"
    return "Sideways"


def daily_trend(df):
    c = df["Close"]
    s50 = c.rolling(50).mean()
    if len(c) < 61:
        return "n/a"
    slope = s50.iloc[-1] - s50.iloc[-11]
    if c.iloc[-1] > s50.iloc[-1] and slope > 0:
        return "Up"
    if c.iloc[-1] < s50.iloc[-1] and slope < 0:
        return "Down"
    return "Mixed"


def fib_read(w, lookback=52):
    seg = w.iloc[-lookback:]
    if len(seg) < 10:
        return None
    hi, lo = float(seg["High"].max()), float(seg["Low"].min())
    rng = hi - lo
    if rng <= 0:
        return None
    up_leg = seg["Low"].idxmin() < seg["High"].idxmax()   # low came first = up leg
    c = float(seg["Close"].iloc[-1])
    retr = (hi - c) / rng if up_leg else (c - lo) / rng
    levels = {lv: (hi - lv * rng) if up_leg else (lo + lv * rng) for lv in (0.382, 0.5, 0.618, 0.786)}
    return {"up": up_leg, "hi": hi, "lo": lo, "retr": retr, "levels": levels}


def fib_zone(f):
    if not f:
        return "n/a"
    r = f["retr"]
    if r < 0.236:
        return "Near swing extreme"
    if r < 0.382:
        return "Shallow (<0.382)"
    if r < 0.5:
        return "0.382-0.5"
    if r < 0.618:
        return "Golden 0.5-0.618"
    if r < 0.786:
        return "Deep 0.618-0.786"
    return "Past 0.786"


def fib_confluence(f, price, mas):
    if not f:
        return False
    for lv in f["levels"].values():
        if abs(price - lv) / price <= 0.02:
            for m in mas:
                if m is not None and not np.isnan(m) and abs(m - lv) / lv <= 0.015:
                    return True
    return False


def in_range(df):
    c = df["Close"]
    if len(c) < 61:
        return False
    width = (df["High"].iloc[-20:].max() - df["Low"].iloc[-20:].min()) / c.iloc[-1]
    s50 = c.rolling(50).mean()
    flat = abs(s50.iloc[-1] / s50.iloc[-11] - 1) < 0.01
    return bool(width < 0.06 and flat)


def weekly_wick(w, df):
    comp = w[w.index <= df.index[-1]]
    if comp.empty:
        return ""
    b = comp.iloc[-1]
    body = abs(b.Close - b.Open)
    upper = b.High - max(b.Open, b.Close)
    lower = min(b.Open, b.Close) - b.Low
    body = max(body, 1e-9)
    if upper > 2 * body and upper > lower:
        return "Upper wick (rejection)"
    if lower > 2 * body and lower > upper:
        return "Lower wick (buyers defended)"
    return ""


def ret_years(c, n):
    past = c[c.index <= c.index[-1] - pd.DateOffset(years=n)]
    if past.empty:
        return None
    return (c.iloc[-1] / past.iloc[-1] - 1) * 100


def cagr10(c):
    past = c[c.index <= c.index[-1] - pd.DateOffset(years=10)]
    base = past.iloc[-1] if not past.empty else c.iloc[0]
    base_date = past.index[-1] if not past.empty else c.index[0]
    years = (c.index[-1] - base_date).days / 365.25
    if years < 1 or base <= 0:
        return None, years
    return (c.iloc[-1] / base) ** (1 / years) - 1, years


def pct_change(c, n):
    if len(c) <= n:
        return None
    return (c.iloc[-1] / c.iloc[-1 - n] - 1) * 100


# ----------------------------------------------------------------- options math
_erf = np.vectorize(math.erf)


def ncdf(x):
    return 0.5 * (1 + _erf(np.asarray(x, float) / math.sqrt(2)))


def npdf(x):
    x = np.asarray(x, float)
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def years_to(exp):
    d = (pd.to_datetime(exp).date() - ny_now().date()).days
    return max(d, 1) / 365.0


def days_to(exp):
    return (pd.to_datetime(exp).date() - ny_now().date()).days


def greeks(S, K, T, iv, r, call=True):
    K = np.asarray(K, float)
    iv = np.clip(np.asarray(iv, float), 0.01, 5.0)
    T = max(T, 1 / 365)
    d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
    delta = ncdf(d1) if call else ncdf(d1) - 1
    gamma = npdf(d1) / (S * iv * np.sqrt(T))
    return delta, gamma


def contract_table(df, S, exp, r, call):
    df = df.copy()
    for col in ("bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility"):
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    both = (df["bid"] > 0) & (df["ask"] > 0)
    df["mid"] = np.where(both, (df["bid"] + df["ask"]) / 2, df["lastPrice"])
    df["spread%"] = np.where(both & (df["mid"] > 0), (df["ask"] - df["bid"]) / df["mid"] * 100, np.nan)
    d, g = greeks(S, df["strike"].values, years_to(exp), df["impliedVolatility"].values, r, call)
    df["delta"], df["gamma"] = d, g
    if "wb_delta" in df:                       # live Greeks from Webull beat our estimate
        df["delta"] = np.where(df["wb_delta"].notna(), df["wb_delta"], df["delta"])
        df["gamma"] = np.where(df["wb_gamma"].notna(), df["wb_gamma"], df["gamma"])
        df["theta"] = df["wb_theta"]
    if call:
        df["breakeven"] = df["strike"] + df["mid"]
        df["type"] = np.where(df["delta"] >= 0.8, "Deep ITM", np.where(df["strike"] < S, "ITM", "OTM"))
    else:
        df["breakeven"] = df["strike"] - df["mid"]
        df["type"] = np.where(df["delta"] <= -0.8, "Deep ITM", np.where(df["strike"] > S, "ITM", "OTM"))
    df["to BE %"] = (df["breakeven"] / S - 1) * 100
    # suggested limit: last trade if it sits inside the spread, otherwise the mid
    inside = both & (df["lastPrice"] >= df["bid"]) & (df["lastPrice"] <= df["ask"])
    df["limit"] = np.where(inside, df["lastPrice"], df["mid"]).round(2)
    return df


@st.cache_data(ttl=900, show_spinner=False)
def gex_profile(t, S, max_days, max_exps, r):
    exps = [e for e in expirations(t) if 0 <= days_to(e) <= max_days][:max_exps]
    frames = []
    run_parallel(_option_chain, [(t, e) for e in exps])       # warm the cache in parallel
    for e in exps:
        try:
            calls, puts = option_chain(t, e)
        except Exception:
            continue
        T = years_to(e)
        for df, is_call in ((calls, True), (puts, False)):
            df = df.copy()
            df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)
            df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors="coerce").fillna(0)
            df = df[(df["openInterest"] > 0) & (df["impliedVolatility"] > 0.01)]
            if df.empty:
                continue
            _, g = greeks(S, df["strike"].values, T, df["impliedVolatility"].values, r, is_call)
            sign = 1 if is_call else -1
            frames.append(pd.DataFrame({
                "strike": df["strike"].values,
                "gex": sign * g * df["openInterest"].values * 100 * S * S * 0.01,
                "oi_call": df["openInterest"].values if is_call else 0,
                "oi_put": 0 if is_call else df["openInterest"].values,
                "call_gex": sign * g * df["openInterest"].values * 100 * S * S * 0.01 if is_call else 0,
                "put_gex": 0 if is_call else sign * g * df["openInterest"].values * 100 * S * S * 0.01,
            }))
    if not frames:
        return None, exps
    prof = pd.concat(frames).groupby("strike", as_index=False).sum().sort_values("strike")
    return prof, exps


# ----------------------------------------------------------------- fundamentals (proxy scores)
def strength_score(i):
    s = 0
    cr = i.get("currentRatio")
    if cr is not None:
        s += 2 if cr >= 1.5 else (1 if cr >= 1 else 0)
    de = i.get("debtToEquity")      # Yahoo gives this as a percent (150 = 1.5x)
    if de is not None:
        s += 3 if de < 50 else (2 if de < 100 else (1 if de < 200 else 0))
    cash, debt = i.get("totalCash"), i.get("totalDebt")
    if cash is not None and debt:
        s += 2 if cash >= debt else (1 if cash >= 0.5 * debt else 0)
    elif cash and not debt:
        s += 2
    fcf = i.get("freeCashflow")
    if fcf is not None and fcf > 0:
        s += 2
    qr = i.get("quickRatio")
    if qr is not None and qr >= 1:
        s += 1
    return s


def profit_score(i):
    s = 0
    roe = i.get("returnOnEquity")
    if roe is not None:
        s += 3 if roe >= 0.20 else (2 if roe >= 0.10 else (1 if roe > 0 else 0))
    roa = i.get("returnOnAssets")
    if roa is not None:
        s += 2 if roa >= 0.10 else (1 if roa >= 0.05 else 0)
    pm = i.get("profitMargins")
    if pm is not None:
        s += 2 if pm >= 0.20 else (1 if pm >= 0.10 else 0)
    om = i.get("operatingMargins")
    if om is not None:
        s += 2 if om >= 0.20 else (1 if om >= 0.10 else 0)
    eg = i.get("earningsGrowth")
    if eg is not None and eg > 0:
        s += 1
    return s


# ----------------------------------------------------------------- analysis per stock
def analyze(t, df, sector_perf):
    c = df["Close"]
    w = to_weekly(df)
    price = float(c.iloc[-1])
    s50 = c.rolling(50).mean()
    s20w = w["Close"].rolling(20).mean()
    f = fib_read(w)
    cg, _ = cagr10(c)
    etf = sector_etf(t)
    sp = sector_perf.get(etf, {}) if etf else {}
    row = {
        "Ticker": t,
        "Price": round(price, 2),
        "Day %": round(pct_change(c, 1) or 0, 2),
        "Weekly": weekly_trend(w),
        "Daily": daily_trend(df),
        "RSI": round(float(rsi(c).iloc[-1]), 1),
        "50D rising": bool(len(s50) > 11 and s50.iloc[-1] > s50.iloc[-11]),
        "Fib zone": fib_zone(f),
        "Fib+SMA": fib_confluence(f, price, [float(s50.iloc[-1]), float(s20w.iloc[-1]) if len(s20w) else None]),
        "Range": in_range(df),
        "Weekly wick": weekly_wick(w, df),
        "5Y %": round(ret_years(c, 5), 1) if ret_years(c, 5) is not None else None,
        "10Y CAGR %": round(cg * 100, 1) if cg is not None else None,
        "Sector": SECTOR_ETFS.get(etf, "?") if etf else "?",
        "Sector 1M %": round(sp.get("1M"), 1) if sp.get("1M") is not None else None,
        "_fib_up": f["up"] if f else None,
        "_retr": f["retr"] if f else None,
    }
    s = 0
    s += 25 if row["Weekly"] == "Up" else 0
    s += 15 if row["Daily"] == "Up" else 0
    s += 15 if 50 <= row["RSI"] < 70 else (5 if row["RSI"] < 50 else 0)
    s += 10 if row["50D rising"] else 0
    if f and f["up"]:
        s += 15 if 0.382 <= f["retr"] <= 0.618 else (5 if f["retr"] < 0.382 else 0)
    s += 10 if (row["Sector 1M %"] or 0) > 0 else 0
    s += 10 if not row["Range"] else 0
    row["Score"] = s
    return row


def analyze_cached(t, df, sector_perf):
    """analyze() is the slow part of every refresh, so remember the answer until the data changes."""
    sp = (sector_perf.get(sector_etf(t), {}) or {}).get("1M")
    key = (t, len(df), df.index[-1], float(df["Close"].iloc[-1]), None if sp is None else round(sp, 2))
    memo = _mem()["analyze"]
    hit = memo.get(t)
    if hit and hit[0] == key:
        return dict(hit[1])
    row = analyze(t, df, sector_perf)
    memo[t] = (key, row)
    return dict(row)


def sector_table(hist):
    rows, perf = [], {}
    for etf, name in SECTOR_ETFS.items():
        df = hist.get(etf)
        if df is None:
            continue
        c = df["Close"]
        p = {"1W": pct_change(c, 5), "1M": pct_change(c, 21), "3M": pct_change(c, 63)}
        perf[etf] = p
        rows.append({"ETF": etf, "Sector": name,
                     "1W %": round(p["1W"], 2) if p["1W"] is not None else None,
                     "1M %": round(p["1M"], 2) if p["1M"] is not None else None,
                     "3M %": round(p["3M"], 2) if p["3M"] is not None else None,
                     "Weekly": weekly_trend(to_weekly(df))})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("1M %", ascending=False)
    return out, perf


def period_change(t, period, hist_map, quotes):
    """Returns (change in $, change in %) for Today / 1 week / 1 month."""
    if period == "Today":
        q = quotes.get(t)
        if q:
            return q["Chg"], q["Chg %"]
    df = hist_map.get(t)
    if df is None or len(df) < 25:
        return None, None
    c = df["Close"]
    n = {"Today": 1, "1 week": 5, "1 month": 21}[period]
    return float(c.iloc[-1] - c.iloc[-1 - n]), float((c.iloc[-1] / c.iloc[-1 - n] - 1) * 100)


def movers_table(choice, period, hist_map):
    """Who is pushing the index or sector. Dow uses real Dow points (price-weighted);
    everything else uses fund weight x % change = percentage points of the fund's move."""
    if choice == "DIA":
        names = UNIVERSES["Dow 30 (YM)"]
        extra = get_history(tuple(n for n in names if n not in hist_map), "1y") if any(
            n not in hist_map for n in names) else {}
        hm = {**hist_map, **extra}
        quotes = live_quotes(tuple(names)) if period == "Today" else {}
        dji = hm.get("^DJI")
        closes = {n: float(hm[n]["Close"].iloc[-1]) for n in names if n in hm}
        if dji is None or len(closes) < 25:
            return None, "Dow data is incomplete right now.", None
        divisor = sum(closes.values()) / float(dji["Close"].iloc[-1])
        rows = []
        for n in names:
            d, pc = period_change(n, period, hm, quotes)
            if d is None:
                continue
            rows.append({"Ticker": n, "Weight %": closes.get(n, 0) / sum(closes.values()) * 100,
                         "Change %": pc, "Impact": d / divisor})
        out = pd.DataFrame(rows)
        unit = "Dow points"
        note = f"All 30 Dow stocks. Dow divisor {divisor:.4f}: a $1 move in any Dow stock moves the Dow about {1 / divisor:.1f} points."
    else:
        w = etf_holdings(choice)
        if not w:
            fallback = {"QQQ": UNIVERSES["Nasdaq (NQ) top 20"], "SPY": UNIVERSES["S&P 500 (ES) top 20"]}.get(choice)
            if not fallback:
                return None, f"Yahoo didn't return holdings for {choice}. Try again later.", None
            w = {t: 1 / len(fallback) for t in fallback}
            note = "Yahoo didn't return weights, so these are equal-weighted. Treat as a rough read."
        else:
            note = f"Top {len(w)} holdings, about {sum(w.values()) * 100:.0f}% of the fund."
        names = list(w)
        missing = [n for n in names if n not in hist_map]
        hm = {**hist_map, **(get_history(tuple(missing), "1y") if missing else {})}
        quotes = live_quotes(tuple(names)) if period == "Today" else {}
        rows = []
        for n in names:
            d, pc = period_change(n, period, hm, quotes)
            if pc is None:
                continue
            rows.append({"Ticker": n, "Weight %": w[n] * 100, "Change %": pc, "Impact": w[n] * pc})
        out = pd.DataFrame(rows)
        unit = "% pts of the fund's move"
    if out.empty:
        return None, "No price data for these holdings.", None
    total = out["Impact"].sum()
    out["Share of move %"] = np.where(abs(total) > 1e-9, out["Impact"] / total * 100, np.nan)
    out = out.reindex(out["Impact"].abs().sort_values(ascending=False).index)
    return out, note, unit


def third_friday(y, m):
    d = dt.date(y, m, 15)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)


def next_quad_witching(today):
    for y in (today.year, today.year + 1):
        for m in (3, 6, 9, 12):
            d = third_friday(y, m)
            if d >= today:
                return d
    return None


def discount_flags(hist, vix_on):
    flags = []
    spy = hist.get("SPY")
    if spy is not None:
        c = spy["Close"]
        downs = 0
        for i in range(len(c) - 1, 0, -1):
            if c.iloc[i] < c.iloc[i - 1]:
                downs += 1
            else:
                break
        if downs >= 3:
            flags.append(f"SPY has closed down {downs} days in a row")
        off20 = (c.iloc[-1] / c.iloc[-20:].max() - 1) * 100
        if off20 <= -5:
            flags.append(f"SPY is {off20:.1f}% below its 20-day high")
        off52 = (c.iloc[-1] / c.iloc[-252:].max() - 1) * 100
        if off52 <= -10:
            flags.append(f"SPY is {off52:.1f}% off its 52-week high (correction territory)")
    if vix_on is False:
        flags.append("VIX is above its 50-day MA, so fear is elevated and premiums can be on sale for LEAPS buyers")
    return flags


# ----------------------------------------------------------------- history analogs
def forward_returns(df, events):
    c = df["Close"]
    rows = []
    for e in events:
        d = pd.Timestamp(e["date"])
        if e.get("timing", "before_open") == "after_close":
            base_idx = c.index[c.index <= d]
        else:
            base_idx = c.index[c.index < d]
        if len(base_idx) == 0:
            continue
        pos = c.index.get_loc(base_idx[-1])
        base = c.iloc[pos]
        r = {"Date": e["date"], "Event": e["name"], "Category": e["category"]}
        for label, n in (("+1D %", 1), ("+1W %", 5), ("+1M %", 21), ("+3M %", 63)):
            r[label] = round((c.iloc[pos + n] / base - 1) * 100, 2) if pos + n < len(c) else None
        window = c.iloc[pos + 1: pos + 22]
        r["Worst in 1M %"] = round((window.min() / base - 1) * 100, 2) if len(window) else None
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- Ask the Radar
SYSTEM_PROMPT = """You are the Trendslinger Options Radar assistant for Louis, a trader who buys options,
mostly LEAPS on strong large-cap stocks. Answer from the radar data provided. Be direct and practical.
You do not tell him what to buy; you point out what the data says against his rules and what to check.

His rules (from the course he follows):
- Weekly trend is the head coach. Don't fight it. Confirm on the daily, execute with hourly timing.
- Supply/demand zones matter most on higher timeframes; S&R on lower timeframes.
- Avoid ranges; wait for a breakout with confirmation. Strong candle closes confirm; wicks show rejection.
- Fib pullbacks: 0.382 shallow, 0.5 halfway (stop area), 0.618 golden zone, past 0.786 = trend at risk.
  Best when a Fib level lines up with the 20 or 50 SMA. Confirm with structure, volume and price action.
- Favor RSI under 70 (ideally 50+) with a rising 50-day MA. Favor sectors that are green.
- VIX vs its 50-day MA: above = fear (entries/discounts), below = complacency (take profits).
- LEAPS 9-18 months out. Strike target: take the 10-year average annual return, halve it, add to price.
- Pick contracts with high open interest and volume and tight bid/ask; use limit orders near last trade.
- Fundamentals matter: financial strength >= 5 and profitability >= 7 (GuruFocus style; radar shows proxies).
- Loss limit 20-30%. Roll only if down about 25-30% with fundamentals intact, once per trade.
  Rolled positions often need about 60% to recover.
- Discount windows: quad witching, corrections, multi-day drops, panics. Avoid holding new entries
  into weekends when gap risk is high. 6-12 trades a year, start with 1 contract, journal everything.
GEX numbers are estimates from free data. Say so when they matter to the answer."""


def ask_radar(question, context, history, api_key, model):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]
    msgs.append({"role": "user", "content": question})
    resp = client.messages.create(model=model, max_tokens=1500,
                                  system=SYSTEM_PROMPT + "\n\nCurrent radar data:\n" + context,
                                  messages=msgs)
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


# ----------------------------------------------------------------- Contract Search engine
MONEYNESS = {   # choice -> (contract types allowed, starting delta range)
    "Any": (None, (0.55, 0.90)),
    "ITM": (("ITM", "Deep ITM"), (0.55, 0.90)),
    "Deep ITM": (("Deep ITM",), (0.80, 1.00)),
    "OTM": (("OTM",), (0.20, 0.50)),
}


MONEY_LABELS = {"Any": "Any", "ITM": "In the money (ITM)", "Deep ITM": "Deep ITM (delta 0.80+)",
                "OTM": "Out of the money (OTM)"}
MONEY_NOTES = {"Any": "ITM or OTM, whichever fits best.",
               "ITM": "Strike below the stock price for calls (above for puts). Costs more, holds value better.",
               "Deep ITM": "Mostly intrinsic value. Moves almost like the stock. The course's safer LEAPS pick.",
               "OTM": "Strike above the stock price for calls (below for puts). Cheaper, bigger upside, "
                      "higher risk of expiring worthless."}


def best_contract_for(t, S, cg, side_call, exps, budget, dlo, dhi, min_oi, max_spread, rate, moneyness="Any"):
    """Best contract for one stock across the given expirations, or (None, reason)."""
    allowed = MONEYNESS.get(moneyness, MONEYNESS["Any"])[0]
    label = "" if allowed is None else f"{moneyness} "
    best, best_src, reason = None, None, f"no {label}contract fit the budget and rules"
    tgt_delta = (dlo + dhi) / 2
    width = max(dhi - dlo, 0.1)
    for e in exps:
        try:
            calls, puts = option_chain(t, e)
        except Exception:
            continue
        raw, src = enrich_chain_live(calls if side_call else puts, S, t, e)
        if raw is None or raw.empty:
            continue
        tab = contract_table(raw, S, e, rate, side_call)
        tab["cost"] = tab["limit"] * 100
        ok = tab[(tab["limit"] > 0) & (tab["cost"] <= budget) &
                 (tab["delta"].abs() >= dlo) & (tab["delta"].abs() <= dhi) &
                 (tab["openInterest"] >= min_oi) &
                 ((tab["spread%"] <= max_spread) | tab["spread%"].isna())].copy()
        if allowed is not None:
            ok = ok[ok["type"].isin(allowed)]
        if side_call and cg is not None and not ok.empty:
            target = S * (1 + cg / 2) ** years_to(e)
            ok = ok[ok["strike"] <= target * 1.02]
        if ok.empty:
            pool_t = tab if allowed is None else tab[tab["type"].isin(allowed)]
            if len(pool_t) and (pool_t["cost"] > budget).all():
                reason = (f"cheapest {label}contract is over budget "
                          f"(from ${pool_t['cost'][pool_t['cost'] > 0].min():,.0f})")
            continue
        spread_pts = np.where(ok["spread%"].isna(), 0.25, (1 - ok["spread%"] / max_spread).clip(0, 1))
        ok["Fit"] = (40 * (1 - (ok["delta"].abs() - tgt_delta).abs() / width).clip(0, 1)
                     + 20 * spread_pts
                     + 20 * (np.log10(ok["openInterest"].clip(lower=1)) / 4).clip(0, 1)
                     + 20 * (1 - ok["to BE %"].abs() / 30).clip(0, 1)).round(0)
        top = ok.loc[ok["Fit"].idxmax()].copy()
        top["expiry"] = e
        if best is None or top["Fit"] > best["Fit"]:
            best, best_src = top, src
    return (best, best_src) if best is not None else (None, reason)


# ================================================================= UI
now = ny_now()
for _k in list(st.session_state.keys()):                 # keep-alive for tab widgets (see keep())
    if isinstance(_k, str) and _k.startswith("w_"):
        st.session_state[_k] = st.session_state[_k]
for _k, _v in st.session_state.pop("apply_pending", {}).items():   # values queued by buttons last run
    st.session_state[_k] = _v


def looks_like_phone():
    try:
        ua = st.context.headers.get("User-Agent", "") or ""
    except Exception:
        return False
    return any(x in ua for x in ("iPhone", "iPod", "Windows Phone")) or ("Android" in ua and "Mobile" in ua)


_layout = st.session_state.get("layout_mode", "Auto")
PHONE = (_layout == "Phone") or (_layout == "Auto" and looks_like_phone())
# ----------------------------------------------------------------- this device: auto-save + home-screen icon
LS_DATA, LS_KEYS = "ts_options_radar_data_v1", "ts_options_radar_keys_v1"
try:
    from streamlit_js_eval import streamlit_js_eval as _js
except Exception:
    _js = None
if _js is None or os.environ.get("RADAR_NO_LOCALSTORE"):
    st.session_state.setdefault("ls_ready", True)

HOME_SCREEN_JS = r"""
setFrameHeight(0);
(function () {
  var href = window.location.href, i = href.indexOf('/component/');
  var base = (i > 0 ? href.slice(0, i) : window.parent.location.origin) + '/app/static/';
  [window.parent, window.top].forEach(function (w) {
    try {
      var d = w.document;
      if (d.getElementById('ts-home')) return;
      function add(tag, attrs) { var e = d.createElement(tag); for (var k in attrs) e.setAttribute(k, attrs[k]); d.head.appendChild(e); }
      add('link', {id: 'ts-home', rel: 'apple-touch-icon', href: base + 'apple-touch-icon.png'});
      add('meta', {name: 'apple-mobile-web-app-title', content: 'Options Radar'});
      add('meta', {name: 'application-name', content: 'Options Radar'});
      add('meta', {name: 'theme-color', content: '#0E161A'});
      var m = {name: 'Trendslinger Options Radar', short_name: 'Options Radar', display: 'browser',
               background_color: '#0E161A', theme_color: '#0E161A', start_url: w.location.origin + '/',
               icons: [{src: base + 'icon-192.png', sizes: '192x192', type: 'image/png'},
                       {src: base + 'icon-512.png', sizes: '512x512', type: 'image/png'},
                       {src: base + 'icon-maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable'}]};
      var old = d.querySelector('link[rel="manifest"]'); if (old) old.remove();
      add('link', {rel: 'manifest', href: URL.createObjectURL(new Blob([JSON.stringify(m)], {type: 'application/manifest+json'}))});
    } catch (e) {}
  });
})();
"""

if not st.session_state.get("ls_ready"):
    _raw = _js(js_expressions=HOME_SCREEN_JS + "JSON.stringify({d: localStorage.getItem('" + LS_DATA + "'), "
               "k: localStorage.getItem('" + LS_KEYS + "')})", key="ls_load")
    if _raw is None:
        st.markdown('<div class="ts-header"><div class="ts-title">Trendslinger <b>Options Radar</b></div></div>',
                    unsafe_allow_html=True)
        st.info("Loading your saved watchlist and journal...")
        if st.button("Continue without my saved data", key="ls_skip"):
            st.session_state["ls_ready"] = True
            st.rerun()
        st.stop()
    try:
        _got = json.loads(_raw) or {}
        if _got.get("d"):
            st.session_state["my_data"] = json.loads(_got["d"])
        if _got.get("k"):
            _k = json.loads(_got["k"])
            st.session_state["tradier_token"] = _k.get("tradier", "")
            st.session_state["claude_key"] = _k.get("claude", "")
            st.session_state["remember_keys"] = True
    except Exception:
        pass
    st.session_state["ls_ready"] = True


def device_save():
    """Write this visitor's data (and keys, if they asked) to their own browser. Runs at the end of each page."""
    if _js is None or os.environ.get("RADAR_NO_LOCALSTORE") or not st.session_state.get("ls_ready"):
        return
    data = json.dumps(_store(), default=str, sort_keys=True)
    keys = (json.dumps({"tradier": st.session_state.get("tradier_token", ""),
                        "claude": st.session_state.get("claude_key", "")})
            if st.session_state.get("remember_keys") else "")
    sig = str(hash((data, keys)))
    if sig == st.session_state.get("ls_saved"):
        return
    st.session_state["ls_saved"] = sig
    js = ("setFrameHeight(0);localStorage.setItem(" + json.dumps(LS_DATA) + "," + json.dumps(data) + ");"
          + ("localStorage.setItem(" + json.dumps(LS_KEYS) + "," + json.dumps(keys) + ");" if keys
             else "localStorage.removeItem(" + json.dumps(LS_KEYS) + ");") + "1")
    _js(js_expressions=js, want_output=False, key="ls_save_" + sig.replace("-", "m"))


hdr_slot = st.container()
focus_slot = st.container()
with st.sidebar:
    _LEARNED_SECTORS.clear()
    _LEARNED_SECTORS.update(load_json(SECTORS_FILE, {}) or {})
    groups = load_watch_groups()
    watchlist = flatten_groups(groups)
    st.markdown("### Watchlist")
    if st.session_state.get("focus_pending") in watchlist:
        st.session_state["focus"] = st.session_state.pop("focus_pending")
    if st.session_state.get("focus") not in watchlist:
        st.session_state["focus"] = watchlist[0]
    focus = (focus_slot if PHONE else st).selectbox(
        "Stock" if PHONE else "Focus ticker (Contracts, Gamma, news)", watchlist, key="focus",
        format_func=lambda t: TICKER_LABELS.get(t, t))
    with st.form("add_watch", clear_on_submit=True):
        new_t = st.text_input("Add tickers", placeholder="TSLA, AMD")
        grp = st.selectbox("Into group", list(groups) + ["+ New group"])
        new_grp = st.text_input("New group name (if + New group)")
        if st.form_submit_button("Add to watchlist"):
            adds = [x.strip().upper() for x in new_t.replace(" ", ",").split(",") if x.strip()]
            adds = [a for a in adds if a not in watchlist]
            target = new_grp.strip().upper() if grp == "+ New group" else grp
            if adds and target:
                groups.setdefault(target, []).extend(adds)
                save_json(WATCHLIST_FILE, groups)
                st.session_state["focus_pending"] = adds[0]
                st.rerun()
    drop = st.multiselect("Remove from watchlist", watchlist, key="wl_drop",
                          format_func=lambda t: TICKER_LABELS.get(t, t))
    if drop and st.button("Remove selected", key="wl_remove"):
        kept = {g: [t for t in ts if t not in drop] for g, ts in groups.items()}
        kept = {g: ts for g, ts in kept.items() if ts}
        save_json(WATCHLIST_FILE, kept or DEFAULT_WATCHLIST)
        st.rerun()

    st.markdown("### Radar settings")
    uni_name = st.selectbox("Stocks to scan", ["My watchlist"] + list(UNIVERSES.keys()) + ["Custom"], index=0)
    custom = ""
    if uni_name == "Custom":
        custom = st.text_area("Tickers (comma separated)", "AAPL, MSFT, NVDA")
    if uni_name == "Custom":
        universe = [x.strip().upper() for x in custom.split(",") if x.strip()]
    elif uni_name == "My watchlist":
        universe = watchlist
    else:
        universe = UNIVERSES[uni_name]
    live_on = st.toggle("Live updates on the Watchlist tab", True)
    live_secs = st.select_slider("Update every", [30, 60, 120, 300], 60, format_func=lambda s: f"{s} sec")
    rate = st.number_input("Risk-free rate (for delta)", 0.0, 0.10, 0.04, 0.0025, format="%.4f")
    st.markdown("### Ask the Radar")
    api_key = st.text_input("Your Claude API key", type="password", key="claude_key",
                            help="Get one at console.anthropic.com.")
    st.checkbox("Remember my keys on this device", key="remember_keys",
                help="Saves your Claude key and Tradier token in this phone's or computer's browser only, "
                     "never on the server. Leave it off on a shared device.")
    model = st.text_input("Model", "claude-sonnet-5-5")
    st.markdown("### Live data (optional)")
    saved_tok = tradier_token()
    tok_in = st.text_input("Tradier token", saved_tok, type="password",
                           help="From your Tradier account's API settings. Use the live token, not Sandbox.")
    if tok_in.strip() != saved_tok:
        st.session_state["tradier_token"] = tok_in.strip()
        if not FRIENDS_EDITION:
            if tok_in.strip():
                TRADIER_FILE.write_text(json.dumps({"token": tok_in.strip()}))
            elif TRADIER_FILE.exists():
                TRADIER_FILE.unlink()
        st.rerun()
    if saved_tok:
        ok_t, msg_t = tradier_check(saved_tok)
        (st.success if ok_t else st.error)(msg_t)
    st.markdown("### Data source")
    st.caption(get_webull()[1])

    st.markdown("### Backup")
    st.caption("Your watchlist, journal and notes save on this device by themselves. Use these to move them "
               "to another phone or computer. (Keys and tokens are never included.)")
    st.download_button("Download a backup", json.dumps(_store(), indent=2, default=str),
                       file_name="my_options_radar_data.json", mime="application/json")
    up = st.file_uploader("Load a backup", type=["json"], key="my_data_upload")
    if up is not None and st.session_state.get("my_data_loaded") != up.name + str(up.size):
        try:
            st.session_state["my_data"] = json.loads(up.getvalue().decode("utf-8"))
            st.session_state["my_data_loaded"] = up.name + str(up.size)
            st.rerun()
        except Exception:
            st.error("That file isn't a Radar backup.")
    if st.button("Refresh data now"):
        clear_all_caches()
        st.rerun()
    st.markdown("### Layout")
    st.radio("Screen", ["Auto", "Phone", "Desktop"], horizontal=True, key="layout_mode",
             help="Auto picks the phone layout on phones. Change it if it guessed wrong.")

hdr_slot.markdown(f"""<div class="ts-header"><div class="ts-title">Trendslinger <b>Options Radar</b></div>
<div><span class="ts-chip">{now:%a %b %d, %I:%M %p} ET</span><span class="ts-chip">{"Webull live + Yahoo history" if get_webull()[0] is not None else ("Tradier live + Yahoo history" if tradier_token() else "Yahoo data, options ~15 min delayed")}</span>
<span class="ts-chip ts-wide">No signals: your call</span></div></div>""", unsafe_allow_html=True)
st.caption("Trendslingers educational tool. Not financial advice. Data may be delayed or incomplete.")

needed = tuple(sorted(set(universe) | set(watchlist) | set(SECTOR_ETFS) | set(INDEX_ETFS)
                     | {"^VIX", "^DJI", "TLT", "HYG", "IEF", "IWM", focus}))
hist = get_history(needed, spinner="Pulling price history (first load takes the longest)...")

if not hist:
    st.error("No price data came back from Yahoo. Check your internet connection, then click Refresh data now.")
    st.stop()

sec_df, sector_perf = sector_table(hist)


def scan_rows(tickers):
    rows = []
    for t in tickers:
        df = hist.get(t)
        if df is None or len(df) < 120:
            continue
        try:
            rows.append(analyze_cached(t, df, sector_perf))
        except Exception:
            continue
    return rows


_wl = pd.DataFrame(scan_rows(watchlist))
st.session_state["watch_scan"] = _wl.set_index("Ticker") if not _wl.empty else None

# regime
vix_df = hist.get("^VIX")
vix_val = vix_ma = None
vix_on = None
if vix_df is not None:
    vix_val = float(vix_df["Close"].iloc[-1])
    vix_ma = float(vix_df["Close"].rolling(50).mean().iloc[-1])
    vix_on = vix_val < vix_ma
fg_parts = {}
fg_hist = {}
try:
    _wfg = wb_fear_greed()
except LookupError:
    _wfg = None
except Exception as _e:
    _wfg = None
    WB_STATUS["fear_greed"] = str(_e)[:120]
if _wfg:
    fg_score, fg_rating, fg_hist = _wfg
    fg_source = "Webull"
    WB_STATUS["fear_greed"] = True
else:
    fg_score, fg_rating = fear_greed()
    fg_source = "CNN"
if fg_score is None:
    fg_score, fg_rating, fg_parts = fear_greed_estimate(hist)
    fg_source = "Radar estimate"
quad = next_quad_witching(now.date())
dflags = discount_flags(hist, vix_on)

TAB_NAMES = ["Watchlist", "Market", "Scanner", "Contracts", "Contract Search", "Gamma", "History", "Journal", "Ask"]
LAZY = not os.environ.get("RADAR_EAGER")
if LAZY:
    try:      # only the tab you're looking at does any work
        tabs = st.tabs(TAB_NAMES, key="nav", on_change="rerun")
    except TypeError:                       # older Streamlit: draw every tab like before
        tabs, LAZY = st.tabs(TAB_NAMES), False
else:
    tabs = st.tabs(TAB_NAMES)


def tab_open(i):
    return (not LAZY) or getattr(tabs[i], "open", True) is not False


# ----------------------------------------------------------------- Watchlist (live)
def watchlist_panel():
    quotes = live_quotes(tuple(watchlist))
    scan_df = st.session_state.get("watch_scan")
    rows = []
    for g, ts in groups.items():
      for t in ts:
        q = quotes.get(t)
        df = hist.get(t)
        r = {"Group": g, "Ticker": TICKER_LABELS.get(t, t)}
        if q:
            r.update({"Last": q["Last"], "Chg": q["Chg"], "Chg %": q["Chg %"],
                      "Day range": f'{q["Day low"]:,.2f} - {q["Day high"]:,.2f}', "Volume": q["Volume"]})
        if df is not None and len(df) > 60:
            s50 = float(df["Close"].rolling(50).mean().iloc[-1])
            last = q["Last"] if q else float(df["Close"].iloc[-1])
            r["vs 50D %"] = (last / s50 - 1) * 100
            r["Weekly"] = weekly_trend(to_weekly(df))
            r["Daily"] = daily_trend(df)
        if scan_df is not None and t in scan_df.index:
            r["Score"] = scan_df.loc[t, "Score"]
            r["Fib zone"] = scan_df.loc[t, "Fib zone"]
        rows.append(r)
    wl = pd.DataFrame(rows)
    stamp = ny_now().strftime("%I:%M:%S %p ET")
    src = sorted({q.get("src", "Yahoo") for q in quotes.values()})
    if WB_STATUS.get("stocks") not in (None, True) and get_webull()[0] is not None:
        st.caption(f"Webull stock quotes: {WB_STATUS['stocks']}. Falling back to Yahoo.")
    st.caption(f"Updated {stamp}. Quotes from {' + '.join(src) or 'Yahoo'}"
               + ("." if ("Webull" in src or "Tradier" in src) else "; option quotes stay about 15 minutes delayed.")
               + (f" Refreshing every {live_secs} sec." if (live_on and market_hours()) else ""))
    missing = [t for t in watchlist if t not in quotes]
    if wl.empty:
        st.info("Add tickers in the sidebar to build your watchlist.")
        return
    fmt = {"Last": "{:,.2f}", "Chg": "{:+,.2f}", "Chg %": "{:+.2f}", "Volume": "{:,.0f}", "vs 50D %": "{:+.1f}"}
    for g in groups:
        part = wl[wl["Group"] == g].drop(columns="Group")
        if part.empty:
            continue
        up = int((part.get("Chg %", pd.Series(dtype=float)) > 0).sum())
        st.markdown(f'<div class="ts-muted" style="margin:10px 0 4px 2px;letter-spacing:.08em">'
                    f'{g.upper()} &middot; {up}/{len(part)} green</div>', unsafe_allow_html=True)
        sty = part.style.format({k: v for k, v in fmt.items() if k in part.columns}, na_rep="-")
        sty = style_map(sty, color_signed, ["Chg", "Chg %", "vs 50D %"])
        sty = style_map(sty, color_trend, ["Weekly", "Daily"])
        show_df(sty, phone=["Ticker", "Last", "Chg %", "Weekly", "Score"], hide_index=True)
    if missing:
        st.caption("No live quote for: " + ", ".join(missing) + ". Check the symbol (Yahoo uses BRK-B, ^VIX).")


def set_watchlist(new_groups, note):
    """Save a new watchlist, keeping the old one for Undo."""
    old = load_watch_groups()
    save_json(WATCHLIST_BACKUP, old)
    st.session_state["wl_undo"] = old
    save_json(WATCHLIST_FILE, new_groups)
    st.session_state["wl_msg"] = note
    st.rerun()


def render_manage_watchlist():
    with st.expander("Manage watchlist: TradingView import/export, sort into sectors",
                     expanded=bool(st.session_state.get("wl_msg"))):
        if st.session_state.get("wl_msg"):
            st.success(st.session_state.pop("wl_msg"))
        t1, t2, t3 = st.tabs(["Import from TradingView", "Export to TradingView", "Sort into sectors"])

        with t1:
            st.caption("In TradingView: open your watchlist, click **Advanced view**, then **Export list**. "
                       "Upload that .txt file here.")
            up = st.file_uploader("TradingView watchlist (.txt)", type=["txt"], key="tv_upload")
            if up is not None:
                g_tv, names_tv, skipped = parse_tradingview_file(up.getvalue().decode("utf-8", "ignore"))
                n = sum(len(v) for v in g_tv.values())
                if not n:
                    st.error("No symbols found in that file. Make sure it's the file from TradingView's Export list.")
                else:
                    st.write(f"Found **{n} symbols** in **{len(g_tv)} sections**: " + ", ".join(g_tv))
                    if skipped:
                        st.caption("Skipped (the Radar can't chart these): " + ", ".join(skipped[:20]))
                    how = st.radio("Groups", ["Sort into sectors", "Keep my TradingView sections"],
                                   horizontal=True, key=keep("w_tv_how", "Sort into sectors"))
                    keep_secs = []
                    if how == "Sort into sectors":
                        dflt = [g for g in g_tv if g.strip().lower() in ("lookout", "watch", "watching", "on deck")]
                        keep_secs = st.multiselect("Keep these sections as they are (not sorted)", list(g_tv),
                                                   key=keep_multi("w_tv_keep", list(g_tv), dflt))
                    mode = st.radio("Your current Radar watchlist", ["Replace it", "Add these to it"],
                                    horizontal=True, key=keep("w_tv_mode", "Replace it"))
                    if st.button("Import", type="primary", key="btn_tv_import"):
                        with st.spinner("Sorting into sectors..." if how == "Sort into sectors" else "Importing..."):
                            known = load_json(TV_SYMBOLS_FILE, {}) or {}
                            known.update(names_tv)
                            save_json(TV_SYMBOLS_FILE, known)
                            if mode == "Add these to it":
                                base = load_watch_groups()
                                have = set(flatten_groups(base))
                                merged = {g: list(v) for g, v in base.items()}
                                for g, ts in g_tv.items():
                                    merged.setdefault(g, []).extend([t for t in ts if t not in have])
                                g_tv = {g: v for g, v in merged.items() if v}
                            if how == "Sort into sectors":
                                g_tv, learned, unknown = sort_into_sectors(g_tv, keep=tuple(keep_secs))
                                ls = load_json(SECTORS_FILE, {}) or {}
                                ls.update(learned)
                                save_json(SECTORS_FILE, ls)
                        set_watchlist(g_tv, f"Imported {n} symbols from TradingView into {len(g_tv)} groups."
                                      + (f" No sector found for: {', '.join(unknown)} (put in Other)."
                                         if how == "Sort into sectors" and unknown else ""))

        with t2:
            st.caption("Makes a .txt file you can load into TradingView: click your watchlist's name, "
                       "then **Import list...**")
            secs = st.checkbox("Include my groups as TradingView sections", key=keep("w_tv_secs", True))
            if st.button("Make the TradingView file", key="btn_tv_export"):
                with st.spinner("Looking up each stock's exchange..."):
                    known = load_json(TV_SYMBOLS_FILE, {}) or {}
                    text, names = build_tradingview_file(groups, known, secs)
                    known.update({t: v for t, v in names.items() if ":" in v})
                    save_json(TV_SYMBOLS_FILE, known)
                st.session_state["tv_export"] = (text, [t for t, v in names.items() if ":" not in v])
            if st.session_state.get("tv_export"):
                text, no_ex = st.session_state["tv_export"]
                st.download_button("Download for TradingView", text, file_name="Trendslinger_watchlist.txt",
                                   mime="text/plain", type="primary", key="btn_tv_dl")
                if no_ex:
                    st.caption("Couldn't find the exchange for: " + ", ".join(no_ex) +
                               ". TradingView may ask you to pick these by hand.")

        with t3:
            st.caption("Regroups your watchlist into real sectors (Technology, Semiconductors, Health Care, and so on).")
            keep_now = st.multiselect("Keep these groups as they are", list(groups),
                                      key=keep_multi("w_sort_keep", list(groups),
                                                     [g for g in groups if g.strip().lower() == "lookout"]))
            if st.button("Sort my watchlist into sectors", type="primary", key="btn_sort"):
                with st.spinner("Looking up sectors..."):
                    new_g, learned, unknown = sort_into_sectors(groups, keep=tuple(keep_now))
                    ls = load_json(SECTORS_FILE, {}) or {}
                    ls.update(learned)
                    save_json(SECTORS_FILE, ls)
                set_watchlist(new_g, f"Sorted {len(flatten_groups(new_g))} names into {len(new_g)} groups."
                              + (f" No sector found for: {', '.join(unknown)} (put in Other)." if unknown else ""))

        undo = st.session_state.get("wl_undo") or (load_json(WATCHLIST_BACKUP, None) if WATCHLIST_BACKUP.exists()
                                                   or FRIENDS_EDITION else None)
        if undo and st.button("Undo last watchlist change", key="btn_wl_undo"):
            save_json(WATCHLIST_FILE, undo)
            save_json(WATCHLIST_BACKUP, None)
            st.session_state.pop("wl_undo", None)
            st.session_state["wl_msg"] = "Watchlist put back the way it was."
            st.rerun()


def render_watchlist():
    render_manage_watchlist()
    ticking = live_on and market_hours()
    if hasattr(st, "fragment"):
        st.fragment(run_every=live_secs if ticking else None)(watchlist_panel)()
    else:
        watchlist_panel()
    if live_on and not ticking:
        st.caption("Market is closed, so live updates are paused. They start again at 4:00 AM ET on trading days.")
    st.caption("Pick any watchlist name as the focus ticker in the sidebar to open it in Contracts and Gamma.")

# ----------------------------------------------------------------- Market
def render_market():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if vix_val is not None:
            col = UP if vix_on else DOWN
            card("VIX vs 50-day MA", f'<div class="ts-big" style="color:{col}">{vix_val:.2f}</div>'
                 f'<div class="ts-muted">50-day MA {vix_ma:.2f}: '
                 f'{"risk-on, buy/hold" if vix_on else "risk-off, trim or hunt discounts"}</div>')
        else:
            card("VIX vs 50-day MA", '<div class="ts-muted">VIX data unavailable</div>')
    with c2:
        if fg_score is not None:
            col = DOWN if fg_score < 45 else (UP if fg_score > 55 else WARN)
            note = ("" if fg_source == "CNN" else
                    '<div class="ts-muted">CNN did not respond, so this is the Radar\'s own estimate '
                    'from the same ingredients. It retries CNN on every refresh.</div>')
            past = ""
            if fg_hist:
                past = '<div class="ts-muted">' + " &middot; ".join(
                    f"{lbl} {v}" for lbl, v in (("Yesterday", fg_hist.get("previous_day")),
                                                ("1W ago", fg_hist.get("one_week_ago")),
                                                ("1M ago", fg_hist.get("one_month_ago"))) if v is not None) + "</div>"
            card(f"Fear & Greed ({fg_source})", f'<div class="ts-big" style="color:{col}">{fg_score}</div>'
                 f'<div class="ts-muted">{fg_rating}</div>{past}{note}')
            if fg_parts:
                with st.expander("What's in the estimate"):
                    for k, v in fg_parts.items():
                        st.write(f"{k}: {v:.0f}")
        else:
            card("Fear & Greed", '<div class="ts-muted">CNN did not respond and there wasn\'t enough market '
                 'data for an estimate. Click Refresh data now.</div>')
    with c3:
        dq = (quad - now.date()).days if quad else None
        card("Next quad witching", f'<div class="ts-big">{quad:%b %d}</div>'
             f'<div class="ts-muted">{dq} days away. Discounts often show up around it.</div>' if quad else "")
    with c4:
        rows = ""
        for e in INDEX_ETFS:
            df = hist.get(e)
            if df is None:
                continue
            ch = pct_change(df["Close"], 1) or 0
            wt = weekly_trend(to_weekly(df))
            rows += (f'<div>{e} <span style="color:{UP if ch >= 0 else DOWN}">{ch:+.2f}%</span> '
                     f'<span class="ts-muted">weekly {wt.lower()}</span></div>')
        card("Indexes", rows)

    if now.weekday() == 4:
        flag("It's Friday. New entries held over the weekend carry gap risk at Monday's open.", "ts-bad")
    if quad and 0 <= (quad - now.date()).days <= 5:
        flag(f"Quad witching week ({quad:%b %d}). Watch for discounted premiums.", "ts-good")
    for fl in dflags:
        flag(fl, "ts-good" if "fear" in fl or "SPY" in fl else "")
    if not dflags:
        flag("No discount window right now: no multi-day SPY drop, no correction, VIX below its MA.")

    st.markdown("#### Sector strength")
    st.caption("Favor the green sectors. Avoid sectors with negative or fading momentum.")
    if not sec_df.empty:
        sty = sec_df.style.format({"1W %": "{:+.2f}", "1M %": "{:+.2f}", "3M %": "{:+.2f}"}, na_rep="-")
        sty = style_map(sty, color_signed, ["1W %", "1M %", "3M %"])
        sty = style_map(sty, color_trend, ["Weekly"])
        show_df(sty, phone=["Sector", "1W %", "1M %", "Weekly"], hide_index=True, column_config={
            "ETF": st.column_config.TextColumn(width="small"),
            "Sector": st.column_config.TextColumn(width="medium"),
            "1W %": st.column_config.TextColumn(width="small"),
            "1M %": st.column_config.TextColumn(width="small"),
            "3M %": st.column_config.TextColumn(width="small"),
            "Weekly": st.column_config.TextColumn(width="small")})

    st.markdown("#### What's moving it")
    st.caption("Pick an index or sector to see which stocks are pushing it the most.")
    choices = {"QQQ": "Nasdaq 100 (QQQ / NQ)", "SPY": "S&P 500 (SPY / ES)", "DIA": "Dow 30 (DIA / YM), in Dow points"}
    choices.update({k: f"{k} {v}" for k, v in SECTOR_ETFS.items()})
    m1, m2 = st.columns([2, 1])
    mv_pick = m1.selectbox("Index or sector", list(choices), format_func=lambda k: choices[k],
                           key=keep("w_mv_pick", "QQQ"))
    mv_period = m2.radio("Over", ["Today", "1 week", "1 month"], horizontal=True, key=keep("w_mv_period", "Today"))
    with st.spinner("Working out who's pushing it..."):
        mv, mv_note, mv_unit = movers_table(mv_pick, mv_period, hist)
    if mv is None:
        st.info(mv_note)
    else:
        top = mv.head(12).iloc[::-1]
        figm = go.Figure(go.Bar(x=top["Impact"], y=top["Ticker"], orientation="h",
                                marker_color=[UP if v > 0 else DOWN for v in top["Impact"]],
                                text=[f"{v:+.2f}" for v in top["Impact"]], textposition="outside"))
        figm.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
                           height=max(260, 26 * len(top) + 80), margin=dict(l=10, r=30, t=40, b=10),
                           xaxis_title=mv_unit, title=dict(text=f"Biggest impact on {mv_pick}, {mv_period.lower()}",
                                                           font=dict(size=13)))
        show_fig(figm)
        st.caption(mv_note)
        with st.expander("Full list"):
            sty = mv.style.format({"Weight %": "{:.2f}", "Change %": "{:+.2f}", "Impact": "{:+.2f}",
                                   "Share of move %": "{:+.0f}"}, na_rep="-")
            show_df(style_map(sty, color_signed, ["Change %", "Impact"]), phone=["Ticker", "Change %", "Impact"],
                    hide_index=True)
        st.session_state["movers"] = f"{mv_pick} {mv_period} ({mv_unit}):\n" + mv.head(10).to_csv(index=False)

    notes = load_json(NOTES_FILE, {})
    if notes:
        st.markdown("#### Recent notes")
        recent = sorted(notes.items(), key=lambda kv: kv[0], reverse=True)[:3]
        for k, v in recent:
            d, name = k.split("|", 1)
            card(f"{name} ({d})", f'<div style="white-space:pre-wrap">{v}</div>')

# ----------------------------------------------------------------- Scanner
def render_scanner():
    rows = []
    for t in universe:
        df = hist.get(t)
        if df is None or len(df) < 120:
            continue
        try:
            rows.append(analyze_cached(t, df, sector_perf))
        except Exception:
            continue
    scan = pd.DataFrame(rows)
    st.session_state["scan"] = scan
    if scan.empty:
        st.warning("None of these tickers returned enough data. Check the symbols (Yahoo uses BRK-B, not BRK.B).")
    else:
        f1, f2, f3, f4 = st.columns(4)
        min_score = f1.slider("Minimum score", 0, 100, step=5, key=keep("w_sc_min", 50))
        only_up = f2.checkbox("Weekly uptrend only", key=keep("w_sc_up", True))
        hide_range = f3.checkbox("Hide ranges", key=keep("w_sc_range", True))
        pull_earn = f4.checkbox("Show earnings dates", key=keep("w_sc_earn", False))
        view = scan[scan["Score"] >= min_score]
        if only_up:
            view = view[view["Weekly"] == "Up"]
        if hide_range:
            view = view[~view["Range"]]
        view = view.sort_values("Score", ascending=False).copy()
        if pull_earn and not view.empty:
            with st.spinner("Looking up earnings dates..."):
                view["Earnings"] = run_parallel(earnings_date, list(view["Ticker"]))
            view["Earnings soon"] = [bool(d and 0 <= (d - now.date()).days <= 14) for d in view["Earnings"]]
        cols = [c for c in view.columns if not c.startswith("_")]
        cols = ["Ticker", "Score"] + [c for c in cols if c not in ("Ticker", "Score")]
        st.caption(f"{len(view)} of {len(scan)} names pass. Score weights: weekly up 25, daily up 15, "
                   "RSI 50-70 15, 50D rising 10, Fib pullback 15, green sector 10, not ranging 10.")
        disp = yes_no(view[cols], ["50D rising", "Fib+SMA", "Range", "Earnings soon"])
        sty = disp.style.format({"Price": "{:.2f}", "Day %": "{:+.2f}", "RSI": "{:.1f}",
                                       "5Y %": "{:+.1f}", "10Y CAGR %": "{:.1f}", "Sector 1M %": "{:+.1f}"},
                                      na_rep="-")
        sty = style_map(sty, color_signed, ["Day %", "5Y %", "Sector 1M %"])
        sty = style_map(sty, color_trend, ["Weekly", "Daily"])
        show_df(sty, phone=["Ticker", "Score", "Weekly", "Daily", "RSI", "Fib zone"], hide_index=True)
        downs = scan[(scan["Weekly"] == "Down") & (scan["Daily"] == "Down")]["Ticker"].tolist()
        if downs:
            st.caption("Weekly and daily both down (put watch, or simply avoid calls): " + ", ".join(downs))
        with st.expander("What the columns mean"):
            st.markdown(
                "- **Weekly / Daily**: trend on that timeframe (the weekly is the head coach).\n"
                "- **50D rising**: the 50-day moving average is higher than 2 weeks ago.\n"
                "- **Fib zone**: where price sits in its last 52-week swing. 0.382-0.618 is the pullback buy area.\n"
                "- **Fib+SMA**: Yes = price is at a Fib level that lines up with the 50-day or 20-week SMA. "
                "That's the course's high-probability confluence. It's a flag, not a button.\n"
                "- **Range**: Yes = price is stuck in a tight sideways range. The course says wait for a breakout.\n"
                "- **Weekly wick**: last weekly candle showed a rejection wick.\n"
                "- **Sector 1M %**: how the stock's sector ETF did over the past month.")
        st.caption("Trendlines and supply/demand zones still need your eyes on the chart; the scanner "
                   "covers trend, RSI, MAs, Fib, ranges and wicks.")

# ----------------------------------------------------------------- Contracts
def render_contracts():
    df = hist.get(focus)
    if df is None:
        st.warning(f"No price data for {focus}. Check the symbol in the sidebar.")
    else:
        c = df["Close"]
        S = live_price(focus, float(c.iloc[-1]))
        w = to_weekly(df)
        f = fib_read(w)
        cg, yrs = cagr10(c)
        run_parallel(lambda fn: fn(focus), [ticker_info, earnings_date, get_news, _expirations])
        info = ticker_info(focus)
        stg, prf = strength_score(info), profit_score(info)
        ed = earnings_date(focus)

        a, b, cc, d = st.columns(4)
        with a:
            card(focus, f'<div class="ts-big">{S:,.2f}</div><div class="ts-muted">'
                 f'Weekly {weekly_trend(w).lower()}, daily {daily_trend(df).lower()}, '
                 f'RSI {float(rsi(c).iloc[-1]):.0f}</div>')
        with b:
            col_s = UP if stg >= 5 else DOWN
            col_p = UP if prf >= 7 else DOWN
            card("Fundamentals (proxy)", f'<div><span style="color:{col_s}">Strength {stg}/10</span> '
                 f'(want 5+)</div><div><span style="color:{col_p}">Profitability {prf}/10</span> (want 7+)</div>'
                 f'<div class="ts-muted">Built from Yahoo data, not GuruFocus. Check GuruFocus before entry.</div>')
        with cc:
            if cg is not None:
                card("10-yr return", f'<div class="ts-big">{cg * 100:.1f}%/yr</div>'
                     f'<div class="ts-muted">Over {yrs:.1f} yrs. Halved for strike math: {cg * 50:.1f}%/yr</div>')
            else:
                card("10-yr return", '<div class="ts-muted">Not enough history</div>')
        with d:
            if ed:
                days = (ed - now.date()).days
                col = WARN if 0 <= days <= 14 else TEXT
                card("Next earnings", f'<div class="ts-big" style="color:{col}">{ed:%b %d}</div>'
                     f'<div class="ts-muted">{days} days away</div>')
            else:
                card("Next earnings", '<div class="ts-muted">No date from Yahoo</div>')

        # chart
        last = df.iloc[-260:]
        fig = go.Figure(go.Candlestick(x=last.index, open=last["Open"], high=last["High"], low=last["Low"],
                                       close=last["Close"], increasing_line_color=UP,
                                       decreasing_line_color=DOWN, name=focus))
        fig.add_trace(go.Scatter(x=last.index, y=c.rolling(20).mean().iloc[-260:], name="20 SMA",
                                 line=dict(color=WARN, width=1.2)))
        fig.add_trace(go.Scatter(x=last.index, y=c.rolling(50).mean().iloc[-260:], name="50 SMA",
                                 line=dict(color=ACCENT, width=1.2)))
        if f:
            for lv, px in f["levels"].items():
                fig.add_hline(y=px, line=dict(color=MUTED, dash="dot", width=1),
                              annotation_text=f"{lv}", annotation_position="right")
        fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=420,
                          margin=dict(l=10, r=40, t=50, b=60), xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h", yanchor="top", y=-0.13, x=0),
                          title=dict(text=f"{focus} daily with Fib levels of the last 52-week swing",
                                     font=dict(size=13)))
        show_fig(fig)
        if f:
            st.caption(f"Price sits in the {fib_zone(f)} zone of the {'up' if f['up'] else 'down'} leg "
                       f"({f['lo']:,.2f} to {f['hi']:,.2f}). The 0.5 level is the course's stop area.")

        st.markdown("#### LEAPS contract finder")
        exps = expirations(focus)
        if not exps:
            st.warning(f"Yahoo returned no option expirations for {focus}.")
        else:
            g1, g2, g3 = st.columns(3)
            min_d = g1.number_input("Min days to expiry", 0, 1000, step=15, key=keep("w_c_min_d", 270))
            max_d = g2.number_input("Max days to expiry", 0, 1200, step=15, key=keep("w_c_max_d", 550))
            side = g3.radio("Side", ["Calls", "Puts"], horizontal=True, key=keep("w_c_side", "Calls"))
            in_window = [e for e in exps if min_d <= days_to(e) <= max_d]
            pool = in_window or exps
            if not in_window:
                st.caption("No expirations in that window, so all of them are listed.")
            exp = st.selectbox("Expiration", pool, key=keep_choice("w_c_exp", pool),
                               format_func=lambda e: f"{e}  ({days_to(e)} days)")
            c_money = st.radio("Contracts", list(MONEYNESS), horizontal=True, key=keep("w_c_money", "Any"),
                               format_func=MONEY_LABELS.get)
            st.caption(MONEY_NOTES[c_money])
            h1, h2, h3 = st.columns(3)
            dlo, dhi = h1.slider(f"Delta range, absolute ({c_money})", 0.05, 1.0, step=0.05,
                                 key=keep(f"w_c_delta_{c_money.replace(' ', '_')}", MONEYNESS[c_money][1]))
            min_oi = h2.number_input("Min open interest", 0, 100000, step=50, key=keep("w_c_oi", 100))
            max_spread = h3.number_input("Max bid/ask spread %", 1.0, 100.0, step=1.0, key=keep("w_c_spread", 12.0))
            try:
                calls, puts = option_chain(focus, exp)
                is_call = side == "Calls"
                raw, live_src = enrich_chain_live(calls if is_call else puts, S, focus, exp)
                tab = contract_table(raw, S, exp, rate, is_call)
                if live_src:
                    st.caption(f"Live from {live_src}: bid/ask, last, volume, open interest, IV, delta, gamma and theta.")
                else:
                    why = [f"{k}: {v}" for k, v in (("Webull options", WB_STATUS.get("options")),
                                                    ("Tradier options", WB_STATUS.get("tradier_options")))
                           if v not in (None, True)]
                    st.caption(("; ".join(why) + ". " if why else "") +
                               "Option quotes from Yahoo (about 15 min delayed) with calculated delta.")
                target = S * (1 + cg / 2) ** years_to(exp) if (cg is not None and is_call) else None
                if target:
                    st.caption(f"Strike target for {exp}: {target:,.2f} (price grown at half the 10-yr rate). "
                               "Strikes at or below it give the stock room to get there.")
                picks = tab[(tab["delta"].abs() >= dlo) & (tab["delta"].abs() <= dhi) &
                            (tab["openInterest"] >= min_oi) &
                            ((tab["spread%"] <= max_spread) | tab["spread%"].isna())].copy()
                allowed = MONEYNESS[c_money][0]
                if allowed is not None:
                    picks = picks[picks["type"].isin(allowed)]
                if target is not None:
                    picks = picks[picks["strike"] <= target * 1.02]
                picks = picks.sort_values("openInterest", ascending=False)

                if picks.empty:
                    st.info(f"No {'' if allowed is None else c_money + ' '}contracts pass these filters. "
                            "Widen the delta range, lower min open interest, "
                            "or allow a wider spread. Outside market hours Yahoo often shows 0 bids, so "
                            "spreads can look blank.")
                else:
                    top = picks.head(3)
                    pc = st.columns(len(top))
                    for col, (_, r) in zip(pc, top.iterrows()):
                        cost = r["limit"] * 100
                        spr = "-" if pd.isna(r["spread%"]) else f"{r['spread%']:.1f}%"
                        with col:
                            card(f"{focus} {exp} {r['strike']:g} {'C' if is_call else 'P'}",
                                 f'<div class="ts-big">${r["limit"]:.2f}</div>'
                                 f'<div class="ts-muted">Suggested limit. Cost ${cost:,.0f} per contract</div>'
                                 f'<div>Delta {r["delta"]:+.2f}, {r["type"]}</div>'
                                 f'<div>OI {int(r["openInterest"]):,}, vol {int(r["volume"]):,}, '
                                 f'spread {spr}</div>'
                                 f'<div>Breakeven {r["breakeven"]:,.2f} ({r["to BE %"]:+.1f}%)</div>'
                                 f'<div class="ts-muted">Loss limit: 20% = ${cost * 0.8:,.0f}, '
                                 f'30% = ${cost * 0.7:,.0f}</div>')
                show_cols = ["contractSymbol", "strike", "type", "bid", "ask", "lastPrice", "limit", "spread%",
                             "delta", "theta", "impliedVolatility", "volume", "openInterest", "breakeven", "to BE %"]
                show_cols = [x for x in show_cols if x in tab.columns]
                st.markdown("##### Full chain")
                only_pass = st.checkbox("Show only contracts that pass the filters", key=keep("w_c_pass", True))
                base = picks if only_pass else tab.sort_values("strike")
                sty = base[show_cols].style.format({
                    "strike": "{:g}", "bid": "{:.2f}", "ask": "{:.2f}", "lastPrice": "{:.2f}", "limit": "{:.2f}",
                    "spread%": "{:.1f}", "delta": "{:+.2f}", "theta": "{:+.3f}", "impliedVolatility": "{:.0%}",
                    "volume": "{:,.0f}", "openInterest": "{:,.0f}", "breakeven": "{:,.2f}", "to BE %": "{:+.1f}"},
                    na_rep="-")
                show_df(sty, phone=["strike", "type", "limit", "delta", "openInterest", "spread%", "to BE %"],
                        hide_index=True)
                st.session_state["picks"] = picks.head(5)[show_cols].to_csv(index=False)
            except Exception as e:
                st.error(f"Couldn't load the {exp} chain: {e}")

        st.markdown("#### News")
        nws = get_news(focus)
        if not nws:
            st.caption("No headlines from Yahoo right now.")
        for n in nws:
            st.markdown(f"- [{n['title']}]({n['url']})  \n  <span class='ts-muted'>{n['source']} {n['date']}</span>",
                        unsafe_allow_html=True)

# ----------------------------------------------------------------- Contract Search
def render_search():
    st.caption("Give it a budget. It checks the option chains for your names and finds the best contract per stock "
               "that fits the budget and your rules. It's a shortlist, not a signal: check the chart before you buy.")
    s_side = st.radio("Calls or puts", ["Calls", "Puts", "Follow the trend"], horizontal=True,
                      key=keep("w_s_side", "Calls"))
    st.caption({"Calls": "Betting the stock goes up.",
                "Puts": "Betting the stock goes down.",
                "Follow the trend": "Calls on names in a weekly uptrend, puts on names in a weekly downtrend "
                                    "(the course's 'don't fight the weekly' rule)."}[s_side])
    pool_names = {"Best setups on my watchlist": {"Puts": "Best put setups on my watchlist"},
                  "My whole watchlist": {},
                  "Top performers (1 month)": {"Puts": "Weakest performers (1 month)",
                                               "Follow the trend": "Biggest movers, up or down (1 month)"}}
    a1, a2, a3 = st.columns(3)
    budget = a1.number_input("Budget per trade ($)", 50, 1_000_000, step=50, key=keep("w_s_budget", 1000))
    pool_pick = a2.selectbox("Search in", list(pool_names),
                             format_func=lambda k: pool_names[k].get(s_side, k),
                             key=keep("w_s_pool", "Best setups on my watchlist"))
    max_names = a3.slider("How many stocks to check", 3, 30, key=keep("w_s_names", 12))
    b1, b2, b3 = st.columns(3)
    s_min_d = b1.number_input("Min days to expiry", 0, 1000, step=15, key=keep("w_s_min_d", 270))
    s_max_d = b2.number_input("Max days to expiry", 0, 1200, step=15, key=keep("w_s_max_d", 550))
    n_exps = b3.slider("Expirations to check per stock", 1, 4, key=keep("w_s_nexp", 2))
    s_money = st.radio("Contracts", list(MONEYNESS), horizontal=True, key=keep("w_s_money", "Any"),
                       format_func=MONEY_LABELS.get)
    st.caption(MONEY_NOTES[s_money])
    with st.expander("Rules (defaults follow the course)"):
        r1, r2, r3 = st.columns(3)
        s_dlo, s_dhi = r1.slider(f"Delta range ({s_money})", 0.05, 1.0, step=0.05,
                                 key=keep(f"w_s_delta_{s_money.replace(' ', '_')}", MONEYNESS[s_money][1]))
        s_oi = r2.number_input("Min open interest", 0, 100000, step=50, key=keep("w_s_oi", 100))
        s_spread = r3.number_input("Max spread %", 1.0, 100.0, step=1.0, key=keep("w_s_spread", 12.0))
        s_min_score = r1.slider("Min setup score", 0, 100, step=5, key=keep("w_s_score", 60))
        s_skip_earn = r3.checkbox("Skip names with earnings in the next 14 days", key=keep("w_s_earn", False))
        st.caption("Min setup score works both ways: for puts it means how bearish the setup is. Calls also have "
                   "to sit at or under the strike target (10-yr return halved), like the Contracts tab.")

    if st.button("Find contracts", type="primary", key="btn_s_go"):
        pool = [t for t in dict.fromkeys(watchlist + list(universe))
                if not t.startswith("^") and t in hist and len(hist[t]) >= 120]
        rows = []
        for t in pool:
            try:
                r = analyze_cached(t, hist[t], sector_perf)
                r["1M %"] = pct_change(hist[t]["Close"], 21)
                rows.append(r)
            except Exception:
                pass
        cand = pd.DataFrame(rows)
        if cand.empty:
            st.warning("No stocks with enough price history to search.")
        else:
            if s_side == "Calls":
                cand["_call"] = True
            elif s_side == "Puts":
                cand["_call"] = False
            else:
                cand["_call"] = cand["Weekly"] != "Down"
            cand["_setup"] = np.where(cand["_call"], cand["Score"], 100 - cand["Score"])
            in_wl = cand["Ticker"].isin(watchlist)
            if pool_pick == "Best setups on my watchlist":
                with_trend = np.where(cand["_call"], cand["Weekly"] == "Up", cand["Weekly"] == "Down")
                cand = cand[in_wl & with_trend & (cand["_setup"] >= s_min_score)].sort_values("_setup", ascending=False)
            elif pool_pick == "My whole watchlist":
                cand = cand[in_wl].sort_values("_setup", ascending=False)
            else:
                m1 = cand["1M %"].fillna(0)
                order = {"Calls": -m1, "Puts": m1, "Follow the trend": -m1.abs()}[s_side]
                cand = cand.assign(_o=order).sort_values("_o")
            cand = cand.head(max_names)
            found, misses = [], []
            bar = st.progress(0.0, text="Loading expirations for all stocks at once...")
            names = list(cand["Ticker"])
            mid = (s_min_d + s_max_d) / 2
            exp_lists = run_parallel(_expirations, names, workers=10)
            wanted = []
            for t, el in zip(names, exp_lists):
                ok_e = [e for e in (el or ()) if s_min_d <= days_to(e) <= s_max_d]
                wanted += [(t, e) for e in sorted(ok_e, key=lambda e: abs(days_to(e) - mid))[:n_exps]]
            bar.progress(0.15, text=f"Loading {len(wanted)} option chains at once...")
            run_parallel(_option_chain, wanted, workers=10)
            if s_skip_earn:
                run_parallel(earnings_date, names, workers=10)
            live_quotes(tuple(names))
            for k, (_, row) in enumerate(cand.iterrows()):
                t = row["Ticker"]
                bar.progress(0.3 + 0.7 * k / max(len(cand), 1), text=f"Ranking {t} ({k + 1} of {len(cand)})")
                side_call = bool(row["_call"])
                if s_skip_earn:
                    ed = earnings_date(t)
                    if ed and 0 <= (ed - now.date()).days <= 14:
                        misses.append({"Ticker": t, "Why": f"earnings {ed:%b %d}"})
                        continue
                exps_all = [e for e in expirations(t) if s_min_d <= days_to(e) <= s_max_d]
                if not exps_all:
                    misses.append({"Ticker": t, "Why": "no expirations in that day range"})
                    continue
                exps_pick = sorted(exps_all, key=lambda e: abs(days_to(e) - mid))[:n_exps]
                S = (live_quotes(tuple(names)).get(t) or {}).get("Last") or float(hist[t]["Close"].iloc[-1])
                cg, _ = cagr10(hist[t]["Close"])
                best, src = best_contract_for(t, S, cg, side_call, exps_pick, budget, s_dlo, s_dhi,
                                              s_oi, s_spread, rate, s_money)
                if best is None:
                    misses.append({"Ticker": t, "Why": src})
                    continue
                setup = row["_setup"]
                against = (side_call and row["Weekly"] == "Down") or (not side_call and row["Weekly"] == "Up")
                found.append({
                    "Rank": int(round(0.6 * best["Fit"] + 0.4 * setup)), "Ticker": t,
                    "Side": "Call" if side_call else "Put", "Expiry": best["expiry"], "Days": days_to(best["expiry"]),
                    "Strike": float(best["strike"]), "Type": best["type"], "Delta": float(best["delta"]),
                    "Limit $": float(best["limit"]), "Cost $": float(best["cost"]),
                    "Fits": int(budget // best["cost"]) if best["cost"] > 0 else 0,
                    "Spread %": float(best["spread%"]) if pd.notna(best["spread%"]) else np.nan,
                    "OI": int(best["openInterest"]), "Breakeven": float(best["breakeven"]),
                    "To BE %": float(best["to BE %"]), "Setup": int(setup), "Fit": int(best["Fit"]),
                    "Trend": "Against weekly trend" if against else ("Sideways" if row["Weekly"] == "Sideways"
                                                                     else "With trend"),
                    "Weekly": row["Weekly"], "Stock": S, "Data": src or "Yahoo (delayed)",
                    "Contract": best.get("contractSymbol", "")})
            bar.progress(1.0, text="Done")
            res = pd.DataFrame(found)
            if not res.empty:
                res = res.sort_values("Rank", ascending=False).reset_index(drop=True)
            st.session_state["search_results"] = res
            st.session_state["search_misses"] = pd.DataFrame(misses)
            st.session_state["search_meta"] = (f"${budget:,.0f} budget, {s_side.lower()}, "
                                               f"{s_money if s_money != 'Any' else 'ITM or OTM'}, "
                                               f"{pool_names[pool_pick].get(s_side, pool_pick).lower()}, "
                                               f"{s_min_d}-{s_max_d} days, delta {s_dlo:.2f}-{s_dhi:.2f}, "
                                               f"{len(cand)} stocks checked")

    res = st.session_state.get("search_results")
    if res is not None:
        st.markdown(f"##### Results: {st.session_state.get('search_meta', '')}")
        if res.empty:
            st.info("Nothing fit. Try a bigger budget, a wider delta range, fewer days to expiry, "
                    "or a lower minimum open interest.")
        else:
            top = res.head(3)
            cols = st.columns(len(top))
            for col, (_, r) in zip(cols, top.iterrows()):
                spr = "-" if pd.isna(r["Spread %"]) else f"{r['Spread %']:.1f}%"
                with col:
                    card(f"#{_ + 1}  {r['Ticker']} {r['Expiry']} {r['Strike']:g} {r['Side'][0]}",
                         f'<div class="ts-big">${r["Limit $"]:.2f}</div>'
                         f'<div class="ts-muted">Suggested limit. ${r["Cost $"]:,.0f} per contract, '
                         f'budget fits {r["Fits"]}</div>'
                         f'<div>Delta {r["Delta"]:+.2f}, {r["Type"]}, {r["Days"]} days</div>'
                         f'<div>OI {r["OI"]:,}, spread {spr}</div>'
                         f'<div>Breakeven {r["Breakeven"]:,.2f} ({r["To BE %"]:+.1f}%)</div>'
                         f'<div class="ts-muted">Setup {r["Setup"]}, contract fit {r["Fit"]}, rank {r["Rank"]}. '
                         f'Loss limit: 20% = ${r["Cost $"] * 0.8:,.0f}, 30% = ${r["Cost $"] * 0.7:,.0f}. '
                         f'{r["Data"]}</div>'
                         + (f'<div style="color:{WARN}">Against the weekly trend. The course says don\'t fight '
                            f'the weekly.</div>' if r["Trend"] == "Against weekly trend" else ""))
                    if st.button("Open in Contracts", key=f"open_{r['Ticker']}_{r['Expiry']}"):
                        if r["Ticker"] not in watchlist:
                            groups.setdefault("Search finds", []).append(r["Ticker"])
                            save_json(WATCHLIST_FILE, groups)
                        st.session_state["focus_pending"] = r["Ticker"]
                        st.session_state["apply_pending"] = {
                            "w_c_exp": r["Expiry"], "w_c_side": "Puts" if r["Side"] == "Put" else "Calls",
                            "w_c_money": s_money,
                            f"w_c_delta_{s_money.replace(' ', '_')}": (s_dlo, s_dhi),
                            "w_c_min_d": int(min(s_min_d, r["Days"])), "w_c_max_d": int(max(s_max_d, r["Days"])),
                            **({"nav": "Contracts"} if LAZY else {})}
                        st.rerun()
            if not LAZY:
                st.caption("After clicking Open in Contracts, go to the Contracts tab: the stock and expiry are set.")
            show = res.drop(columns=["Contract", "Weekly", "Stock"])
            sty_trend = lambda v: f"color: {WARN}" if v == "Against weekly trend" else ""
            sty = show.style.format({"Strike": "{:g}", "Delta": "{:+.2f}", "Limit $": "{:.2f}", "Cost $": "{:,.0f}",
                                     "Spread %": "{:.1f}", "OI": "{:,}", "Breakeven": "{:,.2f}", "To BE %": "{:+.1f}"},
                                    na_rep="-")
            show_df(style_map(sty, sty_trend, ["Trend"]),
                    phone=["Rank", "Ticker", "Side", "Strike", "Expiry", "Limit $", "Delta", "Trend"], hide_index=True)
            st.caption("Rank = 60% contract fit (delta near the middle of your range, tight spread, open interest, "
                       "short distance to breakeven) + 40% setup score. Earnings dates, news and the chart are "
                       "still yours to check.")
            st.session_state["search_ctx"] = res.head(10).to_csv(index=False)
        misses = st.session_state.get("search_misses")
        if misses is not None and not misses.empty:
            with st.expander(f"No match for {len(misses)} stocks (why)"):
                show_df(misses, hide_index=True)

# ----------------------------------------------------------------- Gamma
def render_gamma():
    df = hist.get(focus)
    if df is None:
        st.warning(f"No price data for {focus}.")
    else:
        S = live_price(focus, float(df["Close"].iloc[-1]))
        st.caption("Estimated gamma exposure from Yahoo open interest and implied volatility. Convention: "
                   "dealers long calls, short puts. Treat levels as zones, not exact prices. "
                   "For NQ/ES, use QQQ or SPY as the focus ticker.")
        k1, k2, k3 = st.columns(3)
        gdays = k1.number_input("Expirations within (days)", 1, 365, step=5, key=keep("w_g_days", 45))
        gmax = k2.number_input("Max expirations", 1, 20, step=1, key=keep("w_g_max", 6))
        band = k3.slider("Strike window around price (%)", 5, 40, step=1, key=keep("w_g_band", 15))
        with st.spinner("Building the gamma profile..."):
            prof, used = gex_profile(focus, S, int(gdays), int(gmax), rate)
        if prof is None:
            st.info("No usable option data for that window. Try more days or a more liquid ticker.")
        else:
            win = prof[(prof["strike"] >= S * (1 - band / 100)) & (prof["strike"] <= S * (1 + band / 100))].copy()
            net = prof["gex"].sum()
            call_wall = prof.loc[prof["call_gex"].idxmax(), "strike"] if (prof["call_gex"] > 0).any() else None
            put_wall = prof.loc[prof["put_gex"].idxmin(), "strike"] if (prof["put_gex"] < 0).any() else None
            cum = prof["gex"].cumsum().values
            flip = None
            for i in range(1, len(cum)):
                if np.sign(cum[i]) != np.sign(cum[i - 1]):
                    cand = prof["strike"].iloc[i]
                    if flip is None or abs(cand - S) < abs(flip - S):
                        flip = cand
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                card("Net GEX", f'<div class="ts-big" style="color:{UP if net > 0 else DOWN}">'
                     f'{money(net)}</div><div class="ts-muted">per 1% move. '
                     f'{"Positive: dealers tend to dampen moves" if net > 0 else "Negative: dealers can amplify moves"}</div>')
            with m2:
                card("Call wall", f'<div class="ts-big">{call_wall:g}</div><div class="ts-muted">Biggest call gamma, often resistance</div>'
                     if call_wall else "-")
            with m3:
                card("Put wall", f'<div class="ts-big">{put_wall:g}</div><div class="ts-muted">Biggest put gamma, often support</div>'
                     if put_wall else "-")
            with m4:
                card("Rough flip", f'<div class="ts-big">{flip:g}</div><div class="ts-muted">Where cumulative gamma changes sign</div>'
                     if flip else '<div class="ts-muted">No sign change found</div>')

            fig = go.Figure(go.Bar(x=win["strike"], y=win["gex"] / 1e6,
                                   marker_color=[UP if v > 0 else DOWN for v in win["gex"]], name="GEX"))
            fig.add_vline(x=S, line=dict(color=ACCENT, width=2), annotation_text=f"Price {S:,.2f}")
            fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=360,
                              margin=dict(l=10, r=10, t=30, b=10), yaxis_title="$M per 1% move",
                              title=dict(text=f"{focus} gamma by strike ({len(used)} expirations)", font=dict(size=13)))
            show_fig(fig)

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=win["strike"], y=win["oi_call"], name="Call OI", marker_color=UP))
            fig2.add_trace(go.Bar(x=win["strike"], y=-win["oi_put"], name="Put OI", marker_color=DOWN))
            fig2.add_vline(x=S, line=dict(color=ACCENT, width=2))
            fig2.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=300,
                               barmode="relative", margin=dict(l=10, r=10, t=40, b=10),
                               legend=dict(orientation="h", yanchor="top", y=-0.12, x=0),
                               title=dict(text="Open interest by strike (puts shown below zero)", font=dict(size=13)))
            show_fig(fig2)
            st.caption("Expirations used: " + ", ".join(used))
            st.session_state["gex"] = (f"{focus} price {S:.2f}, net GEX ${net / 1e6:,.0f}M per 1%, call wall {call_wall}, "
                                       f"put wall {put_wall}, rough flip {flip}, expirations {', '.join(used)}")

# ----------------------------------------------------------------- History
def render_history():
    user_events = load_json(EVENTS_FILE, [])
    all_events = DEFAULT_EVENTS + user_events
    st.caption("How the market moved after past events. Pattern-spotting, not prediction: two or three "
               "examples is a small sample.")
    y1, y2 = st.columns(2)
    sym_opts = list(dict.fromkeys(["SPY", "QQQ", "DIA", focus]))
    sym = y1.selectbox("Measure on", sym_opts, key=keep_choice("w_h_sym", sym_opts))
    cats = sorted({e["category"] for e in all_events})
    if "w_h_cats" not in st.session_state or st.session_state.get("w_h_cats_all") != cats:
        st.session_state["w_h_cats"], st.session_state["w_h_cats_all"] = list(cats), list(cats)
    pick_cats = y2.multiselect("Categories", cats, key="w_h_cats")
    with st.spinner("Loading full history..."):
        long_hist = get_history((sym,), "max").get(sym)
    if long_hist is None:
        st.warning(f"No long history for {sym}.")
    else:
        evs = [e for e in all_events if e["category"] in pick_cats]
        fr = forward_returns(long_hist, evs)
        if fr.empty:
            st.info(f"None of these events fall inside {sym}'s price history.")
        else:
            fr = fr.sort_values("Date", ascending=False)
            notes = load_json(NOTES_FILE, {})
            fr["Your notes"] = [notes.get(f"{d}|{n}", "") for d, n in zip(fr["Date"], fr["Event"])]
            num = ["+1D %", "+1W %", "+1M %", "+3M %", "Worst in 1M %"]
            sty = style_map(fr.style.format({c: "{:+.2f}" for c in num}, na_rep="-"), color_signed, num)
            show_df(sty, phone=["Date", "Event", "+1D %", "+1M %"], hide_index=True)
            avg = fr.groupby("Category")[num].mean().round(2).reset_index()
            avg.insert(1, "Events", fr.groupby("Category").size().values)
            st.markdown("##### Average by category")
            show_df(style_map(avg.style.format({c: "{:+.2f}" for c in num}, na_rep="-"), color_signed, num),
                    phone=["Category", "Events", "+1D %", "+1M %"],
                    hide_index=True)
            st.session_state["history"] = fr.head(20).to_csv(index=False)

    # ---- one event, every index and sector side by side
    st.markdown("#### Event breakdown")
    st.caption("Pick one event to see how each index and sector reacted, and keep your own read on why.")
    ev_sorted = sorted(all_events, key=lambda e: e["date"], reverse=True)
    ev_by_key = {event_key(e): e for e in ev_sorted}
    ev_keys = list(ev_by_key)
    ev = ev_by_key[st.selectbox("Event", ev_keys, key=keep_choice("w_h_event", ev_keys),
                                format_func=lambda k: k.replace("|", "  "))]
    syms = {**BREAKDOWN_SYMS, **SECTOR_ETFS}
    with st.spinner("Comparing indexes and sectors..."):
        bh = get_history(tuple(syms), "max")
    rows = []
    for sy, label in syms.items():
        df = bh.get(sy)
        if df is None:
            continue
        fr1 = forward_returns(df, [ev])
        if fr1.empty:
            continue
        r = fr1.iloc[0]
        rows.append({"Symbol": sy, "Name": label, "Group": "Index" if sy in BREAKDOWN_SYMS else "Sector",
                     "+1D %": r["+1D %"], "+1W %": r["+1W %"], "+1M %": r["+1M %"]})
    bd = pd.DataFrame(rows)
    if bd.empty or bd["+1D %"].isna().all():
        st.info("Not enough trading days after this event yet. The 1-day number fills in after the next close.")
    else:
        bd = bd.sort_values("+1D %", ascending=False)
        figb = go.Figure(go.Bar(x=bd["Symbol"], y=bd["+1D %"],
                                marker_color=[UP if (v or 0) > 0 else DOWN for v in bd["+1D %"]],
                                text=[("-" if pd.isna(v) else f"{v:+.2f}%") for v in bd["+1D %"]],
                                textposition="outside"))
        figb.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=320,
                           margin=dict(l=10, r=10, t=40, b=10), yaxis_title="% next session",
                           title=dict(text=f"Next-session reaction: {ev['name']}", font=dict(size=13)))
        show_fig(figb)
        bnum = ["+1D %", "+1W %", "+1M %"]
        show_df(style_map(bd.style.format({c: "{:+.2f}" for c in bnum}, na_rep="-"), color_signed, bnum),
                phone=["Symbol", "+1D %", "+1W %", "+1M %"],
                hide_index=True)
        st.session_state["breakdown"] = f"{ev['date']} {ev['name']}:\n" + bd.to_csv(index=False)

    notes = load_json(NOTES_FILE, {})
    k = event_key(ev)
    with st.form(f"note_{k}"):
        txt = st.text_area("Your notes on this event (what happened, what moved, why you think it moved)",
                           notes.get(k, ""), height=110)
        if st.form_submit_button("Save note"):
            if txt.strip():
                notes[k] = txt.strip()
            else:
                notes.pop(k, None)
            save_json(NOTES_FILE, notes)
            st.rerun()

    with st.expander("Add your own event"):
        with st.form("add_event", clear_on_submit=True):
            e1, e2 = st.columns(2)
            ev_date = e1.date_input("Date", now.date())
            ev_cat = e2.text_input("Category", "Election")
            ev_name = st.text_input("What happened")
            ev_timing = st.radio("When did the news hit?",
                                 ["Before the open or during the session", "After the close"], horizontal=True)
            if st.form_submit_button("Add event") and ev_name:
                user_events.append({"date": str(ev_date), "name": ev_name, "category": ev_cat.strip() or "Other",
                                    "timing": "after_close" if ev_timing.startswith("After") else "before_open"})
                save_json(EVENTS_FILE, user_events)
                st.rerun()
        if user_events:
            names = [f"{e['date']}  {e['name']}" for e in user_events]
            drop = st.multiselect("Remove your events", names, key="ev_drop")
            if drop and st.button("Remove selected", key="ev_remove"):
                save_json(EVENTS_FILE, [e for e, n in zip(user_events, names) if n not in drop])
                st.rerun()

# ----------------------------------------------------------------- Journal
def render_journal():
    trades = load_json(JOURNAL_FILE, [])
    max_trades = st.number_input("Trades per year budget", 1, 50, key=keep("w_j_max", 12))
    this_year = [t for t in trades if str(t.get("entry_date", ""))[:4] == str(now.year)]
    col = DOWN if len(this_year) > max_trades else (WARN if len(this_year) >= max_trades - 1 else UP)
    st.markdown(f'<div class="ts-card"><span class="ts-big" style="color:{col}">{len(this_year)}</span>'
                f' <span class="ts-muted">of {max_trades} trades used in {now.year} (course range: 6-12)</span></div>',
                unsafe_allow_html=True)

    with st.expander("Log a new trade", expanded=not trades):
        with st.form("new_trade", clear_on_submit=True):
            j1, j2, j3, j4 = st.columns(4)
            jt = j1.text_input("Ticker", focus).upper()
            jr = j2.selectbox("Side", ["Call", "Put"])
            jk = j3.number_input("Strike", 0.0, 100000.0, 0.0, 1.0)
            jx = j4.date_input("Expiration", now.date() + dt.timedelta(days=365))
            j5, j6, j7 = st.columns(3)
            jn = j5.number_input("Contracts", 1, 1000, 1)
            jp = j6.number_input("Entry premium (per share)", 0.0, 10000.0, 0.0, 0.05)
            jd = j7.date_input("Entry date", now.date())
            why = st.text_area("Why this trade (trend, level, Fib, fundamentals, catalyst)")
            if st.form_submit_button("Save trade") and jk > 0 and jp > 0:
                trades.append({"id": int(dt.datetime.now().timestamp()), "ticker": jt, "right": jr, "strike": jk,
                               "expiry": str(jx), "contracts": int(jn), "entry": jp, "entry_date": str(jd),
                               "why": why, "rolls": 0, "status": "open", "current": None, "exit": None})
                save_json(JOURNAL_FILE, trades)
                st.rerun()

    open_trades = [t for t in trades if t["status"] == "open"]
    if open_trades and st.button("Update prices for open trades"):
        occs = {t["id"]: occ_symbol(t["ticker"], t["expiry"], t["right"], t["strike"]) for t in open_trades}
        wbq, _ = wb_option_quotes(tuple(occs.values()))
        if not wbq and tradier_token():
            try:
                tq = tradier_quotes(tuple(occs.values()), tradier_token())
                wbq = {k: {"bid": v.get("bid"), "ask": v.get("ask"), "price": v.get("Last")} for k, v in tq.items()}
            except Exception:
                wbq = {}
        for t in open_trades:
            q = wbq.get(occs[t["id"]])
            if q:
                bid, ask, last = _f(q.get("bid")) or 0, _f(q.get("ask")) or 0, _f(q.get("price")) or 0
                if bid > 0 or ask > 0 or last > 0:
                    t["current"] = round((bid + ask) / 2 if bid > 0 and ask > 0 else last, 2)
                    continue
            try:
                calls, puts = option_chain(t["ticker"], t["expiry"])
                ch = calls if t["right"] == "Call" else puts
                row = ch.iloc[(ch["strike"] - t["strike"]).abs().argsort()[:1]]
                bid, ask, last = (float(row["bid"].iloc[0] or 0), float(row["ask"].iloc[0] or 0),
                                  float(row["lastPrice"].iloc[0] or 0))
                t["current"] = round((bid + ask) / 2 if bid > 0 and ask > 0 else last, 2)
            except Exception:
                pass
        save_json(JOURNAL_FILE, trades)
        st.rerun()

    for t in trades:
        cur = t.get("exit") if t["status"] == "closed" else t.get("current")
        pnl = (cur / t["entry"] - 1) * 100 if cur else None
        dte = (pd.to_datetime(t["expiry"]).date() - now.date()).days
        head = (f"{t['ticker']} {t['expiry']} {t['strike']:g} {t['right']} x{t['contracts']} "
                f"({t['status']})" + (f"  {pnl:+.1f}%" if pnl is not None else ""))
        with st.expander(head):
            st.write(f"Entry ${t['entry']:.2f} on {t['entry_date']}. "
                     + (f"Now ${cur:.2f}. " if cur else "No current price yet. ")
                     + f"{dte} days to expiry. Rolled {t['rolls']} time(s).")
            if t.get("why"):
                st.caption(t["why"])
            if t["status"] == "open" and pnl is not None:
                if pnl <= -30:
                    flag("Past the 30% loss limit. The rules say take the loss.", "ts-bad")
                elif pnl <= -20:
                    flag("Inside the 20-30% loss zone. Decide now, don't drift.", "ts-bad")
                if -30 <= pnl <= -25 and t["rolls"] == 0:
                    flag("Roll candidate, but only if the company's story and fundamentals are still intact. "
                         "A rolled position often needs about 60% to recover.")
                if t["rolls"] >= 1 and pnl < 0:
                    flag("Already rolled once. No more rolls on this trade.", "ts-bad")
                if pnl > 0:
                    flag(f"Up {pnl:.1f}%. Check VIX vs its MA: below it is take-profit weather.", "ts-good")
            if dte < 90 and t["status"] == "open":
                flag(f"Only {dte} days left. Time decay speeds up from here.")
            if t["status"] == "open":
                b1, b2, b3 = st.columns(3)
                ex = b1.number_input("Exit premium", 0.0, 10000.0, float(cur or 0), 0.05, key=f"ex{t['id']}")
                if b1.button("Close trade", key=f"cl{t['id']}"):
                    t["status"], t["exit"], t["exit_date"] = "closed", ex, str(now.date())
                    save_json(JOURNAL_FILE, trades)
                    st.rerun()
                if b2.button("Mark as rolled", key=f"ro{t['id']}"):
                    t["rolls"] += 1
                    save_json(JOURNAL_FILE, trades)
                    st.rerun()
            if st.button("Delete", key=f"del{t['id']}"):
                save_json(JOURNAL_FILE, [x for x in trades if x["id"] != t["id"]])
                st.rerun()
    st.session_state["journal"] = json.dumps(open_trades, default=str)

# ----------------------------------------------------------------- Ask
def render_ask():
    st.caption("Ask about anything on the radar: \"Which names are in the golden zone with a green sector?\", "
               "\"Does the AAPL pick fit my rules?\", \"What did SPY do after past elections?\"")
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q = st.chat_input("Ask the Radar")
    if q:
        if not api_key:
            st.error("Add your Claude API key in the sidebar (or set ANTHROPIC_API_KEY) to use this tab.")
        else:
            scan_df = st.session_state.get("scan", pd.DataFrame())
            scan_csv = "" if scan_df.empty else scan_df.drop(
                columns=[c for c in scan_df.columns if c.startswith("_")]).sort_values(
                "Score", ascending=False).to_csv(index=False)
            ctx = (f"Time: {now:%Y-%m-%d %H:%M} ET\n"
                   f"VIX {vix_val} vs 50MA {vix_ma} (risk-on={vix_on}). Fear & Greed {fg_score} {fg_rating}. "
                   f"Next quad witching {quad}. Discount flags: {dflags}\n"
                   f"Sectors:\n{sec_df.to_csv(index=False) if not sec_df.empty else 'n/a'}\n"
                   f"Scanner:\n{scan_csv}\n"
                   f"Focus ticker {focus}. Top contract picks:\n{st.session_state.get('picks', 'n/a')}\n"
                   f"Contract Search results ({st.session_state.get('search_meta', 'not run')}):\n"
                   f"{st.session_state.get('search_ctx', 'n/a')}\n"
                   f"Gamma: {st.session_state.get('gex', 'n/a')}\n"
                   f"History analogs:\n{st.session_state.get('history', 'n/a')}\n"
                   f"Event breakdown:\n{st.session_state.get('breakdown', 'n/a')}\n"
                   f"Who is moving the index/sector:\n{st.session_state.get('movers', 'n/a')}\n"
                   f"Louis's event notes (his own read, not verified): {json.dumps(load_json(NOTES_FILE, {}))}\n"
                   f"Open journal trades: {st.session_state.get('journal', '[]')}")
            st.session_state["chat"].append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Reading the radar..."):
                    try:
                        ans = ask_radar(q, ctx, st.session_state["chat"][:-1], api_key, model)
                    except Exception as e:
                        ans = f"The Claude API call failed: {e}"
                st.markdown(ans)
            st.session_state["chat"].append({"role": "assistant", "content": ans})


# ----------------------------------------------------------------- draw only the open tab
RENDERERS = [render_watchlist, render_market, render_scanner, render_contracts, render_search,
             render_gamma, render_history, render_journal, render_ask]
for _i, _fn in enumerate(RENDERERS):
    if tab_open(_i):
        with tabs[_i]:
            try:
                _fn()
            except Exception as _e:          # one tab's problem never takes down the whole page
                st.error(f"This tab hit a problem: {_e}. Click **Refresh data now** in the left menu. "
                         "If it keeps happening, send a screenshot.")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())
device_save()
