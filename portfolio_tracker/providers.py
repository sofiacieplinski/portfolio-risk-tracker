"""
Multi-provider market data with automatic fallback.

Priority order (highest quality first):
  1. Polygon.io   — best free tier, 2 years history, unlimited tickers
  2. Finnhub      — 60 calls/min free, good coverage
  3. yfinance     — no key required, always available

Get free API keys at:
  Polygon:  https://polygon.io           (free tier: 5 calls/min, unlimited history)
  Finnhub:  https://finnhub.io           (free tier: 60 calls/min)

Configure with:
  portfolio-tracker set-key polygon  <your-key>
  portfolio-tracker set-key finnhub  <your-key>
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

from .config import get_key

PERIOD_DAYS = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}


def _period_to_dates(period: str) -> tuple[str, str]:
    days = PERIOD_DAYS.get(period, 365)
    end   = datetime.today()
    start = end - timedelta(days=days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# ─── Abstract base ────────────────────────────────────────────────────────────

class PriceProvider(ABC):
    name: str

    @abstractmethod
    def fetch(self, tickers: list[str], period: str) -> pd.DataFrame:
        """Return DataFrame of adjusted daily close prices, columns = tickers."""

    def _validate(self, df: pd.DataFrame, tickers: list[str]) -> bool:
        if df.empty:
            return False
        return all(t in df.columns and not df[t].dropna().empty for t in tickers)


# ─── Polygon.io ───────────────────────────────────────────────────────────────

class PolygonProvider(PriceProvider):
    name = "polygon"
    _BASE = "https://api.polygon.io/v2/aggs/ticker"

    def fetch(self, tickers: list[str], period: str) -> pd.DataFrame:
        key = get_key("polygon")
        if not key:
            raise RuntimeError("No Polygon API key configured.")

        start, end = _period_to_dates(period)
        frames: dict[str, pd.Series] = {}

        for ticker in tickers:
            url = f"{self._BASE}/{ticker}/range/1/day/{start}/{end}"
            params = {
                "adjusted": "true",
                "sort":     "asc",
                "limit":    50000,
                "apiKey":   key,
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "ERROR" or not data.get("results"):
                raise ValueError(f"Polygon: no data for {ticker} — {data.get('error', '')}")

            rows = data["results"]
            idx  = pd.to_datetime([r["t"] for r in rows], unit="ms", utc=True).tz_localize(None)
            closes = [r["c"] for r in rows]
            frames[ticker] = pd.Series(closes, index=idx, name=ticker)

            # Polygon free tier: 5 requests/minute — be conservative
            time.sleep(0.15)

        return pd.DataFrame(frames)


# ─── Finnhub ──────────────────────────────────────────────────────────────────

class FinnhubProvider(PriceProvider):
    name = "finnhub"
    _BASE = "https://finnhub.io/api/v1/stock/candle"

    def fetch(self, tickers: list[str], period: str) -> pd.DataFrame:
        key = get_key("finnhub")
        if not key:
            raise RuntimeError("No Finnhub API key configured.")

        start_str, end_str = _period_to_dates(period)
        from_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp())
        to_ts   = int(datetime.strptime(end_str,   "%Y-%m-%d").timestamp())
        frames: dict[str, pd.Series] = {}

        for ticker in tickers:
            resp = requests.get(
                self._BASE,
                params={"symbol": ticker, "resolution": "D",
                        "from": from_ts, "to": to_ts, "token": key},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("s") != "ok" or not data.get("c"):
                raise ValueError(f"Finnhub: no data for {ticker}")

            idx = pd.to_datetime(data["t"], unit="s", utc=True).tz_localize(None)
            frames[ticker] = pd.Series(data["c"], index=idx, name=ticker)

            time.sleep(0.1)   # 60 calls/min free tier

        return pd.DataFrame(frames)


# ─── yfinance (fallback) ──────────────────────────────────────────────────────

class YFinanceProvider(PriceProvider):
    name = "yfinance"

    def fetch(self, tickers: list[str], period: str) -> pd.DataFrame:
        unique = list(dict.fromkeys(tickers))
        for attempt in range(2):
            try:
                raw = yf.download(unique, period=period, auto_adjust=True, progress=False)
                break
            except Exception as exc:
                if attempt == 0 and "database is locked" in str(exc).lower():
                    time.sleep(2)
                    continue
                raise
        else:
            raw = yf.download(unique, period=period, auto_adjust=True, progress=False)

        if len(unique) == 1:
            prices = raw[["Close"]].rename(columns={"Close": unique[0]})
        else:
            close = raw["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(name=unique[0])
            prices = close

        return prices.dropna(how="all")


# ─── Public interface ─────────────────────────────────────────────────────────

_PROVIDERS: list[PriceProvider] = [
    PolygonProvider(),
    FinnhubProvider(),
    YFinanceProvider(),
]


def get_prices(tickers: list[str], period: str = "1y") -> tuple[pd.DataFrame, str]:
    """
    Fetch adjusted close prices using the best available provider.
    Returns (DataFrame, provider_name_used).
    Raises RuntimeError if all providers fail.
    """
    errors: list[str] = []

    for provider in _PROVIDERS:
        # Skip paid providers if no key is set
        if provider.name in ("polygon", "finnhub") and not get_key(provider.name):
            continue
        try:
            df = provider.fetch(list(tickers), period)
            if provider._validate(df, list(tickers)):
                return df, provider.name
            errors.append(f"{provider.name}: incomplete data")
        except Exception as exc:
            errors.append(f"{provider.name}: {exc}")

    raise RuntimeError(
        "All market data providers failed:\n" + "\n".join(f"  • {e}" for e in errors)
    )
