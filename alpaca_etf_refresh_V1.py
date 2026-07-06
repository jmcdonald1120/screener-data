#!/usr/bin/env python3
"""
alpaca_etf_refresh_V1.py
========================
Read-only daily-data refresh for the ETF Conviction Screen.

Pulls ~3y of daily bars for the ETF universe from the Alpaca **Market Data
API** (read-only), computes the price-derived rating factors, and writes
`etf-data.json` for the React app to load.

This script NEVER places trades and never touches the trading/orders API.
It reads only the market-data endpoints. Credentials are read from the
environment (or a local .env) and are never written to the output file.

What this fills (updates daily):
    px, ret3, m3, m6, m12, trend, sd, sortino, dollarVolM
What it deliberately does NOT fill (slow-moving fundamentals -> seed/CSV):
    expense ratio, AUM, holdings quality, tracking difference,
    top-10 concentration, valuation z-score, fund flows, tax-efficiency

Version: 1.0.0
Changelog:
    1.0.0 - Initial release. Multi-symbol batched bars, pagination,
            429 backoff, split/dividend-adjusted returns, self-test.

Setup (macOS zsh):
    cd ~/Desktop/TradingBot/Scripts
    python3 -m venv .venv && source .venv/bin/activate
    pip install requests
    export APCA_API_KEY_ID="your_key_id"
    export APCA_API_SECRET_KEY="your_secret_key"
    python3 alpaca_etf_refresh_V1.py --tickers tickers.txt --out etf-data.json

Then in the app: Data source -> "Load daily data" -> pick etf-data.json.

Daily automation (macOS, weekdays ~6pm ET) via crontab -e:
    0 18 * * 1-5 cd ~/Desktop/TradingBot/Scripts && ./.venv/bin/python \
        alpaca_etf_refresh_V1.py --tickers tickers.txt --out etf-data.json >> refresh.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install requests")

__version__ = "1.0.0"

DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"
TRADING_DAYS = 252
BATCH = 100          # symbols per request (multi-bars endpoint)
PAGE_LIMIT = 10000   # bars per page

log = logging.getLogger("alpaca_etf_refresh")


# --------------------------------------------------------------------------- #
# Credentials (environment only; never hardcoded, never logged)
# --------------------------------------------------------------------------- #
def load_env_file(path: Path) -> None:
    """Best-effort .env loader so users can keep keys out of their shell history."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_credentials() -> tuple[str, str]:
    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        sys.exit(
            "Missing credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY\n"
            "as environment variables (or in a local .env file). These are your\n"
            "Alpaca Market Data keys and must never be committed or shared."
        )
    return key, secret


# --------------------------------------------------------------------------- #
# Data fetch (read-only market data)
# --------------------------------------------------------------------------- #
def fetch_bars(symbols: list[str], years: int, feed: str,
               headers: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    """Fetch daily bars for a batch of symbols, handling pagination + rate limits."""
    start = (datetime.now(timezone.utc) - timedelta(days=int(years * 365.5) + 30)).date().isoformat()
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in symbols}
    page_token: str | None = None

    while True:
        params: dict[str, Any] = {
            "symbols": ",".join(symbols),
            "timeframe": "1Day",
            "start": start,
            "adjustment": "all",   # split + dividend adjusted (total-return proxy)
            "feed": feed,
            "limit": PAGE_LIMIT,
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token

        for attempt in range(6):
            resp = requests.get(DATA_URL, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:  # rate limited -> exponential backoff
                wait = 2 ** attempt
                log.warning("Rate limited (429). Backing off %ss...", wait)
                time.sleep(wait)
                continue
            if resp.status_code in (401, 403):
                sys.exit("Auth failed (401/403). Check your Alpaca Market Data keys and plan/feed access.")
            resp.raise_for_status()
            break
        else:
            log.error("Giving up on batch after repeated 429s: %s", symbols[:3])
            return out

        payload = resp.json()
        for sym, bars in (payload.get("bars") or {}).items():
            out.setdefault(sym, []).extend(bars)

        page_token = payload.get("next_page_token")
        if not page_token:
            break
        time.sleep(0.05)  # gentle pacing between pages

    return out


# --------------------------------------------------------------------------- #
# Factor math (from adjusted daily closes; no look-ahead)
# --------------------------------------------------------------------------- #
def annualized_return(closes: list[float], lookback: int) -> float | None:
    if len(closes) <= lookback or closes[-lookback - 1] <= 0:
        return None
    total = closes[-1] / closes[-lookback - 1] - 1.0
    yrs = lookback / TRADING_DAYS
    if yrs <= 0:
        return None
    return ((1.0 + total) ** (1.0 / yrs) - 1.0) * 100.0


def trailing_return(closes: list[float], lookback: int) -> float | None:
    if len(closes) <= lookback or closes[-lookback - 1] <= 0:
        return None
    return (closes[-1] / closes[-lookback - 1] - 1.0) * 100.0


def compute_factors(bars: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Compute price-derived factors from a symbol's ascending daily bars."""
    bars = [b for b in bars if b.get("c")]
    if len(bars) < 60:  # need a meaningful history
        return None

    closes = [float(b["c"]) for b in bars]
    vols = [float(b.get("v", 0)) for b in bars]
    n = len(closes)

    # daily simple returns
    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, n) if closes[i - 1] > 0]
    if not rets:
        return None

    mean_d = sum(rets) / len(rets)
    var_d = sum((r - mean_d) ** 2 for r in rets) / max(len(rets) - 1, 1)
    sd_ann = math.sqrt(var_d) * math.sqrt(TRADING_DAYS) * 100.0

    downside = [min(r, 0.0) for r in rets]
    dvar = sum(d * d for d in downside) / max(len(downside) - 1, 1)
    dd_ann = math.sqrt(dvar) * math.sqrt(TRADING_DAYS)  # fraction, not %
    ann_mean = mean_d * TRADING_DAYS                     # fraction
    sortino = round(ann_mean / dd_ann, 2) if dd_ann > 1e-9 else None

    # 200-day trend (% above simple moving average)
    sma_win = min(200, n)
    sma200 = sum(closes[-sma_win:]) / sma_win
    trend = round((closes[-1] / sma200 - 1.0) * 100.0) if sma200 > 0 else None

    # ~30-session average dollar volume, in $M/day
    win = min(30, n)
    dvol = [closes[i] * vols[i] for i in range(n - win, n)]
    dollar_vol_m = round((sum(dvol) / win) / 1e6, 1) if win else None

    def r1(x: float | None) -> float | None:
        return None if x is None else round(x, 1)

    ret3 = annualized_return(closes, min(3 * TRADING_DAYS, n - 1))

    return {
        "px": round(closes[-1], 2),
        "ret3": r1(ret3),
        "m12": r1(trailing_return(closes, min(TRADING_DAYS, n - 1))),
        "m6": r1(trailing_return(closes, min(TRADING_DAYS // 2, n - 1))),
        "m3": r1(trailing_return(closes, min(TRADING_DAYS // 4, n - 1))),
        "trend": trend,
        "sd": round(sd_ann, 1),
        "sortino": sortino,
        "dollarVolM": dollar_vol_m,
        "bars": n,
        "asOf": bars[-1].get("t", "")[:10],
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def read_tickers(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"Ticker file not found: {path}")
    syms = [ln.strip().upper() for ln in path.read_text().splitlines() if ln.strip()]
    if not syms:
        sys.exit("Ticker file is empty.")
    # de-dup, preserve order
    seen: set[str] = set()
    return [s for s in syms if not (s in seen or seen.add(s))]


def run(args: argparse.Namespace) -> int:
    load_env_file(Path(args.env))
    key, secret = get_credentials()
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    tickers = read_tickers(Path(args.tickers))
    log.info("Universe: %d tickers | feed=%s | history=%dy", len(tickers), args.feed, args.years)

    data: dict[str, dict[str, Any]] = {}
    missing: list[str] = []

    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        log.info("Fetching %d-%d of %d ...", i + 1, i + len(batch), len(tickers))
        bars_by_sym = fetch_bars(batch, args.years, args.feed, headers)
        for sym in batch:
            factors = compute_factors(bars_by_sym.get(sym, []))
            if factors is None:
                missing.append(sym)
            else:
                data[sym] = factors
        time.sleep(0.35)  # stay well under 200 req/min on the free tier

    if not data:
        log.error("No data computed for any symbol. Check keys, feed access, and connectivity.")
        return 2

    out = {
        "asOf": max((v["asOf"] for v in data.values() if v.get("asOf")), default=""),
        "feed": args.feed,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(data),
        "data": data,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    log.info("Wrote %s | %d ETFs | as of %s", args.out, len(data), out["asOf"])
    if missing:
        log.warning("No/short data for %d tickers: %s", len(missing), ", ".join(missing))
    return 0


# --------------------------------------------------------------------------- #
# Self-test (no network / no keys): validates the factor math
# --------------------------------------------------------------------------- #
def selftest() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    base = datetime(2023, 1, 2, tzinfo=timezone.utc)
    bars = []
    # exponential trend line (~13%/yr) with a bounded oscillation overlay:
    # the trend dominates (endpoint above its 200d MA) while the overlay
    # produces real down days so downside deviation / Sortino are defined.
    # (a monotonic series has zero downside -> Sortino is correctly undefined)
    for i in range(600):
        trend = 100.0 * (1.0005 ** i)
        close = trend * (1.0 + 0.02 * math.sin(i / 5.0))
        bars.append({"c": round(close, 4), "v": 1_000_000, "t": (base + timedelta(days=i)).isoformat()})
    f = compute_factors(bars)
    assert f is not None, "compute_factors returned None on valid series"
    assert f["px"] > 100, "last price should be above start"
    assert f["ret3"] and 10 < f["ret3"] < 16, f"unexpected ret3: {f['ret3']}"
    assert f["trend"] is not None and f["trend"] > 0, "uptrend should be above its 200d MA"
    assert f["sortino"] is not None and f["sortino"] > 0, "positive-drift series should have positive Sortino"
    assert compute_factors(bars[:10]) is None, "should reject too-short history"
    print("Self-test passed. Sample factors:")
    print(json.dumps(f, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only Alpaca daily refresh for the ETF screener.")
    p.add_argument("--tickers", default="tickers.txt", help="Path to newline-delimited ticker file.")
    p.add_argument("--out", default="etf-data.json", help="Output JSON path for the app to load.")
    p.add_argument("--years", type=int, default=3, help="Years of daily history to pull (default 3).")
    p.add_argument("--feed", default="iex", choices=["iex", "sip"],
                   help="Data feed. 'iex' is free; 'sip' needs a paid Alpaca plan.")
    p.add_argument("--env", default=".env", help="Optional .env file with API keys.")
    p.add_argument("--selftest", action="store_true", help="Run offline math self-test and exit.")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.selftest:
        return selftest()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return run(args)
    except requests.RequestException as e:
        log.error("Network error talking to Alpaca: %s", e)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
