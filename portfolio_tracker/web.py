"""
Web dashboard for Portfolio Risk Metrics Tracker.
Start with: portfolio-tracker serve
"""
from __future__ import annotations

import asyncio
import json
import time
import webbrowser
import xml.etree.ElementTree as ET
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests as _requests
import uvicorn
import yfinance as yf
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .providers import get_prices

import os as _os
DATA_DIR       = Path(_os.environ.get("PT_DATA_DIR", Path.home() / ".portfolio_tracker"))
HOLDINGS_FILE  = DATA_DIR / "holdings.json"
WATCHLIST_FILE = DATA_DIR / "watchlist.json"
HF_CACHE_FILE  = DATA_DIR / "hf_cache.json"

app = FastAPI(title="Portfolio Risk Tracker", docs_url=None, redoc_url=None)

ANN    = 252
PERIOD = "1y"

BENCHMARKS = {
    "SPY":  "S&P 500",
    "QQQ":  "Nasdaq 100",
    "VONE": "Russell 1000",
    "IWM":  "Russell 2000",
    "DIA":  "Dow Jones 30",
    "AGG":  "US Bond Agg",
}

SECTOR_COLORS = {
    # yfinance canonical names (vary by version)
    "Technology":              "#3b82f6",
    "Health Care":             "#22c55e",
    "Healthcare":              "#22c55e",   # alt spelling
    "Financials":              "#f59e0b",
    "Financial Services":      "#f59e0b",   # alt
    "Consumer Discretionary":  "#8b5cf6",
    "Consumer Cyclical":       "#8b5cf6",   # alt
    "Industrials":             "#06b6d4",
    "Communication Services":  "#ec4899",
    "Consumer Staples":        "#84cc16",
    "Consumer Defensive":      "#84cc16",   # alt
    "Energy":                  "#f97316",
    "Real Estate":             "#14b8a6",
    "Materials":               "#a78bfa",
    "Basic Materials":         "#a78bfa",   # alt
    "Utilities":               "#fb923c",
    "ETF":                     "#6b7280",
    "Other":                   "#4b5563",
}

DEFAULT_WATCHLIST = ["NVDA", "META", "TSLA", "AMD", "PLTR", "GOOG", "NFLX"]

# S&P 500 approximate sector weights (2025)
SPY_SECTOR_WEIGHTS = {
    "Technology":             31.0,
    "Financials":             13.5, "Financial Services":    13.5,
    "Health Care":            12.0, "Healthcare":            12.0,
    "Consumer Discretionary": 10.5, "Consumer Cyclical":     10.5,
    "Industrials":             8.5,
    "Communication Services":  8.5,
    "Consumer Staples":        6.0, "Consumer Defensive":     6.0,
    "Energy":                  4.0,
    "Real Estate":             2.5,
    "Materials":               2.5, "Basic Materials":        2.5,
    "Utilities":               2.5,
}

HEDGE_FUNDS = {
    "Lone Pine Capital":  "1040273",
    "Pershing Square":    "1336528",
    "Tiger Global":       "1167483",
    "Coatue Management":  "1135730",
    "Viking Global":      "1048268",
}

SCENARIOS = {
    "2008 GFC":    {"equity": -0.565, "credit": -0.20,  "label": "2008 Global Financial Crisis"},
    "2000 Dotcom": {"equity": -0.491, "credit": -0.05,  "label": "2000 Dot-com Crash"},
    "2020 COVID":  {"equity": -0.338, "credit": -0.115, "label": "2020 COVID Crash"},
    "1994 Bonds":  {"equity": -0.015, "credit": -0.02,  "label": "1994 Bond Massacre"},
    "2013 Taper":  {"equity":  0.295, "credit":  0.025, "label": "2013 Taper Tantrum"},
}

# ── Persistence ────────────────────────────────────────────────────────────────

def _load() -> dict:
    if HOLDINGS_FILE.exists():
        with open(HOLDINGS_FILE) as f:
            return json.load(f)
    return {"holdings": [], "benchmark": "SPY", "risk_free_rate": 0.05}

def _load_watchlist() -> list:
    if WATCHLIST_FILE.exists():
        with open(WATCHLIST_FILE) as f:
            return json.load(f)
    return list(DEFAULT_WATCHLIST)

# ── Core computation ───────────────────────────────────────────────────────────

def _compute(data: dict) -> Optional[dict]:
    holdings  = data["holdings"]
    if not holdings:
        return None

    benchmark = data.get("benchmark", "SPY")
    rf_rate   = data.get("risk_free_rate", 0.05)
    rf_daily  = (1 + rf_rate) ** (1 / ANN) - 1
    tickers   = [h["ticker"] for h in holdings]
    all_tix   = list(dict.fromkeys(tickers + [benchmark]))

    prices, provider = get_prices(all_tix, PERIOD)
    prices = prices.ffill()

    missing = [t for t in all_tix if t not in prices.columns or prices[t].dropna().empty]
    if missing:
        return None

    returns = prices.pct_change(fill_method=None).dropna()
    last    = prices.iloc[-1]
    prev    = prices.iloc[-2] if len(prices) > 1 else prices.iloc[-1]

    # Portfolio weights
    port_vals: dict = {}
    total_val = 0.0
    for h in holdings:
        mv = float(last[h["ticker"]]) * h["shares"]
        port_vals[h["ticker"]] = mv
        total_val += mv
    weights = {t: v / total_val for t, v in port_vals.items()}

    # Daily returns
    w_arr = np.array([weights.get(t, 0.0) for t in tickers])
    pr    = pd.Series(returns[tickers].values @ w_arr, index=returns.index)
    br    = returns[benchmark]
    idx   = pr.index.intersection(br.index)
    pr, br = pr[idx], br[idx]

    # ── Standard metrics ───────────────────────────────────────────────────────
    cov_mat   = np.cov(pr.values, br.values)
    beta      = float(cov_mat[0, 1] / cov_mat[1, 1])
    port_ann  = float((1 + pr.mean()) ** ANN - 1)
    bench_ann = float((1 + br.mean()) ** ANN - 1)
    sharpe    = float(((pr - rf_daily).mean() / pr.std()) * np.sqrt(ANN))
    down      = pr[pr < rf_daily] - rf_daily
    down_std  = float(np.sqrt((down**2).mean()) * np.sqrt(ANN)) if len(down) > 1 else float(pr.std() * np.sqrt(ANN))
    sortino   = (port_ann - rf_rate) / down_std if down_std > 1e-10 else 0.0
    active    = pr - br
    te        = float(active.std() * np.sqrt(ANN))
    var_95    = float(np.percentile(pr.values, 5))
    cum       = (1 + pr).cumprod()
    roll_max  = cum.cummax()
    dd_series = (cum - roll_max) / roll_max
    max_dd    = float(dd_series.min())
    alpha     = port_ann - (rf_rate + beta * (bench_ann - rf_rate))
    ir        = float(active.mean() / active.std() * np.sqrt(ANN)) if active.std() > 0 else 0.0
    r_sq      = float(np.corrcoef(pr.values, br.values)[0, 1] ** 2)

    # ── Buy-side metrics ───────────────────────────────────────────────────────
    calmar  = (port_ann / abs(max_dd)) if max_dd != 0 else 0.0
    treynor = ((port_ann - rf_rate) / beta) if beta != 0 else 0.0

    up_mask = br > 0
    dn_mask = br < 0
    up_cap  = float((pr[up_mask].mean() / br[up_mask].mean()) * 100) if up_mask.sum() > 0 and br[up_mask].mean() != 0 else 0.0
    dn_cap  = float((pr[dn_mask].mean() / br[dn_mask].mean()) * 100) if dn_mask.sum() > 0 and br[dn_mask].mean() != 0 else 0.0

    pr_m    = (1 + pr).resample("ME").prod() - 1
    br_m    = (1 + br).resample("ME").prod() - 1
    common  = pr_m.index.intersection(br_m.index)
    bat_avg = float((pr_m[common] > br_m[common]).mean() * 100) if len(common) > 0 else 0.0

    excess  = pr - rf_daily
    gains   = excess[excess > 0].sum()
    losses  = abs(excess[excess < 0].sum())
    omega   = float(gains / losses) if losses > 0 else float("inf")

    day_ret = float(pr.iloc[-1]) if len(pr) else 0.0
    day_chg = total_val * day_ret

    # ── Equity curve ──────────────────────────────────────────────────────────
    port_cum  = (1 + pr).cumprod() * 100
    bench_cum = (1 + br).cumprod() * 100
    dates     = [d.strftime("%Y-%m-%d") for d in pr.index]
    dd_vals   = [round(v * 100, 2) for v in dd_series.tolist()]

    # ── Rolling 63-day ────────────────────────────────────────────────────────
    win = 63
    roll_sharpe, roll_beta = [], []
    for i in range(len(pr)):
        if i < win:
            roll_sharpe.append(None)
            roll_beta.append(None)
        else:
            w_pr = pr.iloc[i-win:i]
            w_br = br.iloc[i-win:i]
            rs   = float(((w_pr - rf_daily).mean() / w_pr.std()) * np.sqrt(ANN)) if w_pr.std() > 0 else 0
            cm   = np.cov(w_pr.values, w_br.values)
            rb   = float(cm[0,1] / cm[1,1]) if cm[1,1] > 0 else 0
            roll_sharpe.append(round(rs, 3))
            roll_beta.append(round(rb, 3))

    # ── Monthly heatmap ───────────────────────────────────────────────────────
    monthly_port  = ((1 + pr).resample("ME").prod() - 1) * 100
    monthly_bench = ((1 + br).resample("ME").prod() - 1) * 100
    monthly = []
    for dt in monthly_port.index:
        monthly.append({
            "year":  dt.year,
            "month": dt.month,
            "port":  round(float(monthly_port[dt]), 2),
            "bench": round(float(monthly_bench.get(dt, 0)), 2),
        })

    # ── Per-stock risk heatmap ────────────────────────────────────────────────
    stock_risks = []
    for h in holdings:
        t = h["ticker"]
        if t not in returns.columns:
            continue
        sr   = returns[t]
        cm_s = np.cov(sr.values, br.values)
        s_beta = float(cm_s[0,1] / cm_s[1,1]) if cm_s[1,1] > 0 else 0.0
        s_vol  = float(sr.std() * np.sqrt(ANN) * 100)
        s_var  = float(np.percentile(sr.values, 5) * 100)
        s_cum  = (1 + sr).cumprod()
        s_rm   = s_cum.cummax()
        s_dd   = float(((s_cum - s_rm) / s_rm).min() * 100)
        s_corr = float(np.corrcoef(sr.values, pr.values)[0, 1]) if len(pr) == len(sr) else 0.0
        # Composite risk score 0–100 (higher = more risk)
        beta_r = min(abs(s_beta) / 2.0, 1.0)
        vol_r  = min(s_vol / 50.0, 1.0)
        var_r  = min(abs(s_var) / 5.0, 1.0)
        dd_r   = min(abs(s_dd) / 50.0, 1.0)
        score  = round((beta_r * 0.25 + vol_r * 0.35 + var_r * 0.2 + dd_r * 0.2) * 100, 1)

        stock_risks.append({
            "ticker":     t,
            "beta":       round(s_beta, 2),
            "vol":        round(s_vol, 1),
            "var_95":     round(s_var, 2),
            "max_dd":     round(s_dd, 1),
            "corr_port":  round(s_corr, 2),
            "weight":     round(weights.get(t, 0) * 100, 1),
            "risk_score": score,
        })
    stock_risks.sort(key=lambda x: -x["weight"])

    # ── MCTR (marginal contribution to risk) ─────────────────────────────────
    port_cov = returns[tickers].cov().values * ANN
    port_var = float(w_arr @ port_cov @ w_arr)
    port_vol = float(np.sqrt(port_var))
    if port_vol > 0:
        mctr_arr = (port_cov @ w_arr) * w_arr / port_vol
    else:
        mctr_arr = np.zeros(len(tickers))
    mctr_map = {t: float(mctr_arr[i]) for i, t in enumerate(tickers)}

    # ── Risk Summary Score 0-100 (higher = more risk) ─────────────────────
    def _norm(val, low, high):
        return max(0.0, min(1.0, (val - low) / (high - low)))

    sharpe_risk = _norm(2.5 - sharpe, 0, 2.5)
    beta_risk   = _norm(abs(beta), 0.5, 2.0)
    dd_risk     = _norm(abs(max_dd) * 100, 0, 40)
    var_risk    = _norm(abs(var_95) * 100, 0, 5)
    hhi_val     = sum(w**2 for w in weights.values()) * 10000
    hhi_risk    = _norm(hhi_val, 400, 3000)
    risk_summary_score = round(
        (sharpe_risk * 0.20 + beta_risk * 0.25 + dd_risk * 0.20 + var_risk * 0.20 + hhi_risk * 0.15) * 100, 1
    )

    # ── Drawdown analytics ────────────────────────────────────────────────
    dd_arr    = dd_series.values
    neg_dd    = dd_arr[dd_arr < 0]
    avg_dd    = float(neg_dd.mean() * 100) if len(neg_dd) else 0.0
    ulcer_idx = float(np.sqrt(np.mean(dd_arr**2)) * 100)
    # Recovery days: longest run below water
    in_dd, longest, cur_run = False, 0, 0
    for v in dd_arr:
        if v < 0:
            in_dd = True; cur_run += 1; longest = max(longest, cur_run)
        else:
            in_dd = False; cur_run = 0
    drawdown_analytics = {
        "max_drawdown":   round(max_dd * 100, 2),
        "avg_drawdown":   round(avg_dd, 2),
        "ulcer_index":    round(ulcer_idx, 3),
        "recovery_days":  longest,
        "skewness":       round(float(pd.Series(pr.values).skew()), 3),
        "kurtosis":       round(float(pd.Series(pr.values).kurt()), 3),
    }

    # ── Active risk summary ───────────────────────────────────────────────────
    sorted_vals = sorted(port_vals.values(), reverse=True)
    top3_w = sum(sorted_vals[:3]) / total_val * 100 if total_val else 0
    top5_w = sum(sorted_vals[:5]) / total_val * 100 if total_val else 0
    largest = max(weights.items(), key=lambda x: x[1])
    hhi = sum(w**2 for w in weights.values()) * 10000  # Herfindahl index

    active_risks = {
        "concentration_top3": round(top3_w, 1),
        "concentration_top5": round(top5_w, 1),
        "n_positions":        len(holdings),
        "largest_ticker":     largest[0],
        "largest_weight":     round(largest[1] * 100, 1),
        "hhi":                round(hhi, 0),
        "tracking_error":     round(te * 100, 2),
        "info_ratio":         round(ir, 3),
        "var_95_port":        round(var_95 * 100, 2),
        "active_vol":         round(te * 100, 2),
    }

    # ── Attribution ───────────────────────────────────────────────────────────
    attribution = []
    for h in holdings:
        t = h["ticker"]
        if t not in returns.columns:
            continue
        cur  = float(last[t])
        pv   = float(prev[t])
        dchg = (cur - pv) / pv if pv else 0.0
        pret = float((1 + returns[t]).prod() - 1)
        w    = weights.get(t, 0.0)
        gl   = (cur - h["avg_cost"]) * h["shares"]
        glp  = (cur - h["avg_cost"]) / h["avg_cost"]
        attribution.append({
            "ticker":         t,
            "shares":         h["shares"],
            "avg_cost":       h["avg_cost"],
            "price":          round(cur, 4),
            "market_value":   round(port_vals[t], 2),
            "gain_loss":      round(gl, 2),
            "gain_loss_pct":  round(glp * 100, 2),
            "weight":         round(w * 100, 2),
            "day_change_pct": round(dchg * 100, 2),
            "period_return":  round(pret * 100, 2),
            "contribution":   round(w * pret * 100, 2),
        })
    attribution.sort(key=lambda x: -x["market_value"])

    # Attach MCTR + 52-week range to attribution list
    for item in attribution:
        t = item["ticker"]
        item["mctr"] = round(mctr_map.get(t, 0.0) * 100, 3)
        if t in prices.columns:
            s = prices[t].dropna()
            item["high_52w"] = round(float(s.max()), 2) if len(s) else None
            item["low_52w"]  = round(float(s.min()), 2) if len(s) else None
        else:
            item["high_52w"] = None
            item["low_52w"]  = None

    # Biggest returners / losers (by 1-year period return)
    by_period = sorted(attribution, key=lambda x: -x["period_return"])
    returners = by_period[:5]
    losers    = list(reversed(by_period[-5:]))

    return {
        # Summary
        "total_value":      round(total_val, 2),
        "day_change":       round(day_chg, 2),
        "day_change_pct":   round(day_ret * 100, 2),
        "port_ann_return":  round(port_ann * 100, 2),
        "bench_ann_return": round(bench_ann * 100, 2),
        "active_return":    round((port_ann - bench_ann) * 100, 2),
        "benchmark":        benchmark,
        "rf_rate":          rf_rate,
        "n_days":           len(pr),
        "provider":         provider,
        "as_of":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # Core risk
        "beta":             round(beta, 3),
        "sharpe":           round(sharpe, 3),
        "sortino":          round(sortino, 3),
        "alpha":            round(alpha * 100, 2),
        "info_ratio":       round(ir, 3),
        "tracking_error":   round(te * 100, 2),
        "var_95":           round(var_95 * 100, 2),
        "max_drawdown":     round(max_dd * 100, 2),
        "r_squared":        round(r_sq * 100, 2),
        # Buy-side
        "calmar":           round(calmar, 3),
        "treynor":          round(treynor * 100, 2),
        "upside_capture":   round(up_cap, 1),
        "downside_capture": round(dn_cap, 1),
        "batting_avg":      round(bat_avg, 1),
        "omega":            round(min(omega, 99.9), 2),
        # Charts
        "chart": {
            "dates":     dates,
            "portfolio": [round(v, 2) for v in port_cum.tolist()],
            "benchmark": [round(v, 2) for v in bench_cum.tolist()],
            "drawdown":  dd_vals,
        },
        "rolling":       {"dates": dates, "sharpe": roll_sharpe, "beta": roll_beta},
        "monthly":       monthly,
        "attribution":   attribution,
        "stock_risks":   stock_risks,
        "active_risks":  active_risks,
        "returners":           returners,
        "losers":              losers,
        "risk_summary_score":  risk_summary_score,
        "drawdown_analytics":  drawdown_analytics,
        "port_vol":            round(port_vol * 100, 2),
        "_pr_mean":            float(pr.mean()),
        "_pr_std":             float(pr.std()),
    }


def _compute_benchmarks(data: dict) -> list:
    holdings  = data["holdings"]
    benchmark = data.get("benchmark", "SPY")
    rf_rate   = data.get("risk_free_rate", 0.05)
    rf_daily  = (1 + rf_rate) ** (1 / ANN) - 1
    tickers   = [h["ticker"] for h in holdings]

    bm_tickers = list(BENCHMARKS.keys())
    all_tix    = list(dict.fromkeys(tickers + bm_tickers))
    prices, _  = get_prices(all_tix, "1y")
    prices     = prices.ffill()
    returns    = prices.pct_change(fill_method=None).dropna()

    last    = prices.iloc[-1]
    total_v = sum(float(last[h["ticker"]]) * h["shares"] for h in holdings if h["ticker"] in last)
    weights = {h["ticker"]: float(last[h["ticker"]]) * h["shares"] / total_v for h in holdings if h["ticker"] in last}
    w_arr   = np.array([weights.get(t, 0.0) for t in tickers if t in returns.columns])
    valid_t = [t for t in tickers if t in returns.columns]
    pr      = pd.Series(returns[valid_t].values @ w_arr, index=returns.index)

    port_ann    = float((1 + pr.mean()) ** ANN - 1)
    port_vol    = float(pr.std() * np.sqrt(ANN))
    port_sharpe = float(((pr - rf_daily).mean() / pr.std()) * np.sqrt(ANN))

    def sub_ret(series, days):
        s = series.iloc[-days:] if len(series) >= days else series
        return float((1 + s).prod() - 1)

    today    = date.today()
    ytd_days = (today - date(today.year, 1, 1)).days

    rows = []
    for sym, name in BENCHMARKS.items():
        if sym not in returns.columns:
            continue
        br    = returns[sym]
        idx   = pr.index.intersection(br.index)
        p, b  = pr[idx], br[idx]
        cm    = np.cov(p.values, b.values)
        beta  = float(cm[0,1] / cm[1,1]) if cm[1,1] > 0 else 0.0
        b_ann = float((1 + b.mean()) ** ANN - 1)
        b_vol = float(b.std() * np.sqrt(ANN))
        alpha = port_ann - (rf_rate + beta * (b_ann - rf_rate))
        rows.append({
            "symbol":   sym,
            "name":     name,
            "active":   sym == benchmark,
            "bm_1m":    round(sub_ret(b, 21) * 100, 2),
            "bm_3m":    round(sub_ret(b, 63) * 100, 2),
            "bm_ytd":   round(sub_ret(b, ytd_days) * 100, 2),
            "bm_1y":    round(b_ann * 100, 2),
            "bm_vol":   round(b_vol * 100, 2),
            "port_1m":  round(sub_ret(p, 21) * 100, 2),
            "port_3m":  round(sub_ret(p, 63) * 100, 2),
            "port_ytd": round(sub_ret(p, ytd_days) * 100, 2),
            "port_1y":  round(port_ann * 100, 2),
            "beta_vs":  round(beta, 3),
            "alpha_vs": round(alpha * 100, 2),
        })
    return rows


def _compute_sectors(data: dict) -> dict:
    holdings = data["holdings"]
    last_prices, _ = get_prices([h["ticker"] for h in holdings], "5d")
    last = last_prices.ffill().iloc[-1]

    sector_vals: dict = {}
    for h in holdings:
        t  = h["ticker"]
        mv = float(last.get(t, h["avg_cost"])) * h["shares"]
        try:
            info   = yf.Ticker(t).info
            sector = info.get("sector") or ("ETF" if info.get("quoteType") == "ETF" else "Other")
        except Exception:
            sector = "Other"
        sector_vals[sector] = sector_vals.get(sector, 0.0) + mv

    total = sum(sector_vals.values()) or 1
    sectors = sorted([
        {
            "sector": k,
            "value":  round(v, 2),
            "pct":    round(v / total * 100, 1),
            "color":  SECTOR_COLORS.get(k, "#6b7280"),
        }
        for k, v in sector_vals.items()
    ], key=lambda x: -x["value"])

    # ── Rebalancing vs S&P 500 ────────────────────────────────────────────
    # Deduplicated SPY target (use canonical names only)
    CANONICAL_SPY = {
        "Technology": 31.0, "Financials": 13.5, "Health Care": 12.0,
        "Consumer Discretionary": 10.5, "Industrials": 8.5,
        "Communication Services": 8.5, "Consumer Staples": 6.0,
        "Energy": 4.0, "Real Estate": 2.5, "Materials": 2.5, "Utilities": 2.5,
    }
    # Map alternate spellings to canonical
    SECTOR_CANONICAL = {
        "Healthcare": "Health Care", "Financial Services": "Financials",
        "Consumer Cyclical": "Consumer Discretionary",
        "Consumer Defensive": "Consumer Staples",
        "Basic Materials": "Materials",
    }
    rebal: dict = {}
    for s in sectors:
        canon = SECTOR_CANONICAL.get(s["sector"], s["sector"])
        target = CANONICAL_SPY.get(canon, 0.0)
        drift  = round(s["pct"] - target, 1)
        rebal[canon] = {
            "sector":      canon,
            "color":       s["color"],
            "current_pct": s["pct"],
            "target_pct":  target,
            "drift":       drift,
            "action":      "TRIM" if drift > 1.5 else "ADD" if drift < -1.5 else "HOLD",
            "dollar_adj":  round(-drift / 100 * total),
        }
    for canon, target in CANONICAL_SPY.items():
        if canon not in rebal:
            rebal[canon] = {
                "sector": canon, "color": SECTOR_COLORS.get(canon, "#6b7280"),
                "current_pct": 0.0, "target_pct": target,
                "drift": round(-target, 1), "action": "ADD",
                "dollar_adj": round(target / 100 * total),
            }
    rebalancing = sorted(rebal.values(), key=lambda x: abs(x["drift"]), reverse=True)

    return {"sectors": sectors, "rebalancing": rebalancing}


def _compute_watchlist() -> list:
    tickers = _load_watchlist()
    if not tickers:
        return []
    try:
        prices_raw, _ = get_prices(tickers, "3mo")
        prices_raw    = prices_raw.ffill()
    except Exception:
        return []
    last     = prices_raw.iloc[-1]
    prev     = prices_raw.iloc[-2] if len(prices_raw) > 1 else last
    mo_ago   = prices_raw.iloc[-21] if len(prices_raw) >= 21 else prices_raw.iloc[0]
    yr_ago   = prices_raw.iloc[0]
    result   = []
    for t in tickers:
        if t not in last.index or pd.isna(last[t]):
            continue
        p    = float(last[t])
        pp   = float(prev[t])   if not pd.isna(prev[t])   else p
        pmo  = float(mo_ago[t]) if not pd.isna(mo_ago[t]) else p
        pyr  = float(yr_ago[t]) if not pd.isna(yr_ago[t]) else p
        result.append({
            "ticker":  t,
            "price":   round(p, 2),
            "day_chg": round((p - pp)  / pp  * 100, 2) if pp  else 0.0,
            "mo_chg":  round((p - pmo) / pmo * 100, 2) if pmo else 0.0,
            "yr_chg":  round((p - pyr) / pyr * 100, 2) if pyr else 0.0,
        })
    return result


def _compute_correlation(data: dict) -> dict:
    holdings = data["holdings"]
    tickers  = [h["ticker"] for h in holdings]
    prices, _ = get_prices(tickers, PERIOD)
    prices = prices.ffill()
    valid = [t for t in tickers if t in prices.columns and not prices[t].dropna().empty]
    rets  = prices[valid].pct_change(fill_method=None).dropna()
    corr  = rets.corr().round(3)
    return {"tickers": valid, "matrix": corr.values.tolist()}


def _compute_monte_carlo(data: dict) -> dict:
    holdings  = data["holdings"]
    benchmark = data.get("benchmark", "SPY")
    tickers   = [h["ticker"] for h in holdings]
    all_tix   = list(dict.fromkeys(tickers + [benchmark]))
    prices, _ = get_prices(all_tix, PERIOD)
    prices    = prices.ffill()
    returns   = prices.pct_change(fill_method=None).dropna()
    last      = prices.iloc[-1]
    total_val = sum(float(last[h["ticker"]]) * h["shares"] for h in holdings if h["ticker"] in last)
    weights   = {h["ticker"]: float(last[h["ticker"]]) * h["shares"] / total_val for h in holdings if h["ticker"] in last}
    tickers   = [t for t in tickers if t in returns.columns]
    w_arr     = np.array([weights.get(t, 0.0) for t in tickers])
    pr        = pd.Series(returns[tickers].values @ w_arr, index=returns.index)
    mu, sigma = float(pr.mean()), float(pr.std())

    rng     = np.random.default_rng(42)
    n_paths, n_days = 500, 252
    sims    = rng.normal(mu, sigma, (n_paths, n_days))
    paths   = np.cumprod(1 + sims, axis=1) * 100

    pcts = {
        "p5":  np.percentile(paths, 5,  axis=0).round(2).tolist(),
        "p25": np.percentile(paths, 25, axis=0).round(2).tolist(),
        "p50": np.percentile(paths, 50, axis=0).round(2).tolist(),
        "p75": np.percentile(paths, 75, axis=0).round(2).tolist(),
        "p95": np.percentile(paths, 95, axis=0).round(2).tolist(),
    }
    return {"days": list(range(1, n_days + 1)), **pcts, "mu_ann": round((mu * ANN) * 100, 2), "sigma_ann": round(sigma * np.sqrt(ANN) * 100, 2)}


def _compute_stress(data: dict) -> list:
    holdings  = data["holdings"]
    benchmark = data.get("benchmark", "SPY")
    tickers   = [h["ticker"] for h in holdings]
    all_tix   = list(dict.fromkeys(tickers + [benchmark]))
    prices, _ = get_prices(all_tix, PERIOD)
    prices    = prices.ffill()
    returns   = prices.pct_change(fill_method=None).dropna()
    last      = prices.iloc[-1]
    total_val = sum(float(last[h["ticker"]]) * h["shares"] for h in holdings if h["ticker"] in last)
    weights   = {h["ticker"]: float(last[h["ticker"]]) * h["shares"] / total_val for h in holdings if h["ticker"] in last}
    tickers   = [t for t in tickers if t in returns.columns]
    w_arr     = np.array([weights.get(t, 0.0) for t in tickers])
    pr        = pd.Series(returns[tickers].values @ w_arr, index=returns.index)
    br        = returns[benchmark]
    idx       = pr.index.intersection(br.index)
    pr, br    = pr[idx], br[idx]
    cov_mat   = np.cov(pr.values, br.values)
    beta      = float(cov_mat[0, 1] / cov_mat[1, 1]) if cov_mat[1, 1] > 0 else 1.0

    rows = []
    for key, sc in SCENARIOS.items():
        mkt_shock   = sc["equity"]
        port_shock  = beta * mkt_shock          # CAPM estimate
        dollar_loss = total_val * port_shock
        rows.append({
            "scenario":    key,
            "label":       sc["label"],
            "mkt_return":  round(mkt_shock * 100, 1),
            "port_return": round(port_shock * 100, 1),
            "dollar_pnl":  round(dollar_loss, 2),
        })
    return rows


def _fetch_13f(fund_name: str, cik: str) -> dict:
    """Fetch latest two 13F-HR filings from SEC EDGAR and return holdings + changes."""
    hdrs = {"User-Agent": "portfolio-tracker-app research@portfoliotracker.app", "Accept-Encoding": "gzip, deflate"}
    cik10 = cik.zfill(10)
    try:
        sub = _requests.get(f"https://data.sec.gov/submissions/CIK{cik10}.json", headers=hdrs, timeout=15)
        sub.raise_for_status()
        subs = sub.json()
    except Exception as e:
        return {"fund": fund_name, "error": str(e), "holdings": [], "changes": [], "filing_date": ""}

    recent     = subs.get("filings", {}).get("recent", {})
    forms      = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates      = recent.get("filingDate", [])
    idxs       = [i for i, f in enumerate(forms) if f == "13F-HR"][:2]
    if not idxs:
        return {"fund": fund_name, "error": "No 13F-HR found", "holdings": [], "changes": [], "filing_date": ""}

    def _parse(idx: int) -> dict:
        import re as _re
        acc_d  = accessions[idx]
        acc_nd = acc_d.replace("-", "")
        base   = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nd}"

        # Find XML filename via HTML directory listing (most reliable)
        xml_name = "infotable.xml"
        try:
            dr = _requests.get(f"{base}/", headers=hdrs, timeout=10)
            if dr.ok:
                # Match infotable.xml OR information_table.xml (both used by SEC filers)
                hits = _re.findall(r'href="([^"]*(?:infotable|information_table)[^"]*\.xml)"', dr.text, _re.IGNORECASE)
                if hits:
                    xml_name = hits[0].split("/")[-1]
        except Exception:
            pass

        xr = _requests.get(f"{base}/{xml_name}", headers=hdrs, timeout=20)
        xr.raise_for_status()
        root = ET.fromstring(xr.content)
        out: dict = {}
        for node in root.iter():
            if not node.tag.endswith("infoTable"):
                continue
            name, cusip, value, shares = "", "", 0, 0
            for child in node:
                tag = child.tag.split("}")[-1]
                if tag == "nameOfIssuer":
                    name = (child.text or "").strip().title()
                elif tag == "cusip":
                    cusip = (child.text or "").strip()
                elif tag == "value":
                    # SEC 13F reports value in thousands of USD;
                    # however many modern filers report in full dollars.
                    # We detect this: if raw sum >> plausible AUM we skip ×1000.
                    try: value = int(child.text or 0)
                    except: pass
                elif tag == "shrsOrPrnAmt":
                    # shares are nested one level deeper
                    for gc in child:
                        if gc.tag.split("}")[-1] == "sshPrnamt":
                            try: shares = int(gc.text or 0)
                            except: pass
            if not name:
                continue
            key = cusip or name
            if key in out:
                out[key]["value"]  += value
                out[key]["shares"] += shares
            else:
                out[key] = {"name": name, "cusip": cusip, "value": value, "shares": shares}
        return out

    try:
        curr = _parse(idxs[0])
    except Exception as e:
        return {"fund": fund_name, "error": str(e), "holdings": [], "changes": [], "filing_date": dates[idxs[0]] if idxs else ""}

    prev: dict = {}
    prev_date  = ""
    if len(idxs) > 1:
        try:
            prev      = _parse(idxs[1])
            prev_date = dates[idxs[1]]
        except Exception:
            pass

    # Modern EDGAR 13F XML filers (including all funds here) report value in full USD,
    # not in $thousands as the original SEC spec states. Use values as-is.
    total = sum(h["value"] for h in curr.values()) or 1
    holdings_list = sorted(curr.values(), key=lambda x: -x["value"])
    for h in holdings_list:
        h["pct"] = round(h["value"] / total * 100, 2)

    changes = []
    for key, h in curr.items():
        p = prev.get(key)
        if p is None:
            changes.append({"name": h["name"], "type": "NEW",       "pct": h["pct"], "chg": None})
        elif h["shares"] > p["shares"] * 1.05:
            chg = round((h["shares"] - p["shares"]) / p["shares"] * 100, 1)
            changes.append({"name": h["name"], "type": "INCREASED", "pct": h["pct"], "chg": f"+{chg}%"})
        elif h["shares"] < p["shares"] * 0.95:
            chg = round((h["shares"] - p["shares"]) / p["shares"] * 100, 1)
            changes.append({"name": h["name"], "type": "DECREASED", "pct": h["pct"], "chg": f"{chg}%"})
    for key, h in prev.items():
        if key not in curr:
            changes.append({"name": h["name"], "type": "CLOSED",    "pct": 0,        "chg": None})

    order = {"NEW": 0, "CLOSED": 1, "INCREASED": 2, "DECREASED": 3}
    changes.sort(key=lambda x: order.get(x["type"], 9))

    return {
        "fund":        fund_name,
        "filing_date": dates[idxs[0]],
        "prev_date":   prev_date,
        "n_holdings":  len(holdings_list),
        "total_aum":   total,
        "holdings":    holdings_list[:15],
        "changes":     changes,
    }


def _compute_hf_positions() -> list:
    """Load all hedge fund 13F positions with 6-hour file cache."""
    now = time.time()
    if HF_CACHE_FILE.exists():
        try:
            with open(HF_CACHE_FILE) as f:
                cache = json.load(f)
            if now - cache.get("_at", 0) < 3600 * 6:
                return cache["data"]
        except Exception:
            pass

    results = []
    for name, cik in HEDGE_FUNDS.items():
        try:
            results.append(_fetch_13f(name, cik))
        except Exception as e:
            results.append({"fund": name, "error": str(e), "holdings": [], "changes": [], "filing_date": ""})
        time.sleep(0.5)   # be polite to SEC servers

    DATA_DIR.mkdir(exist_ok=True)
    with open(HF_CACHE_FILE, "w") as f:
        json.dump({"_at": now, "data": results}, f)
    return results


# ── API routes ─────────────────────────────────────────────────────────────────

@app.get("/api/data")
async def api_data():
    loop   = asyncio.get_event_loop()
    data   = _load()
    result = await loop.run_in_executor(None, _compute, data)
    if result is None:
        return JSONResponse({"error": "No holdings or data unavailable."}, status_code=503)
    return result


@app.get("/api/benchmarks")
async def api_benchmarks():
    loop = asyncio.get_event_loop()
    data = _load()
    rows = await loop.run_in_executor(None, _compute_benchmarks, data)
    return rows


@app.get("/api/sectors")
async def api_sectors():
    loop   = asyncio.get_event_loop()
    data   = _load()
    result = await loop.run_in_executor(None, _compute_sectors, data)
    return result   # {"sectors": [...], "rebalancing": [...]}


@app.get("/api/hedge-funds")
async def api_hedge_funds():
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _compute_hf_positions)
    return result


@app.get("/api/watchlist")
async def api_watchlist():
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _compute_watchlist)
    return result


@app.get("/api/correlation")
async def api_correlation():
    loop = asyncio.get_event_loop()
    data = _load()
    return await loop.run_in_executor(None, _compute_correlation, data)


@app.get("/api/monte-carlo")
async def api_monte_carlo():
    loop = asyncio.get_event_loop()
    data = _load()
    return await loop.run_in_executor(None, _compute_monte_carlo, data)


@app.get("/api/stress")
async def api_stress():
    loop = asyncio.get_event_loop()
    data = _load()
    return await loop.run_in_executor(None, _compute_stress, data)


@app.get("/api/holdings")
async def api_get_holdings():
    data = _load()
    return data["holdings"]


@app.post("/api/holdings")
async def api_add_holding(request: Request):
    body   = await request.json()
    ticker = body.get("ticker", "").upper().strip()
    shares = float(body.get("shares", 0))
    avg_cost = float(body.get("avg_cost", 0))
    if not ticker or shares <= 0:
        return JSONResponse({"error": "ticker and shares > 0 required"}, status_code=400)
    data = _load()
    existing = next((h for h in data["holdings"] if h["ticker"] == ticker), None)
    if existing:
        # Update: if avg_cost provided replace it, else keep; update shares
        if avg_cost > 0:
            total_cost   = existing["avg_cost"] * existing["shares"] + avg_cost * shares
            total_shares = existing["shares"] + shares
            existing["avg_cost"] = round(total_cost / total_shares, 4)
            existing["shares"]   = round(total_shares, 6)
        else:
            existing["shares"] = round(existing["shares"] + shares, 6)
    else:
        data["holdings"].append({"ticker": ticker, "shares": round(shares, 6), "avg_cost": round(avg_cost, 4)})
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HOLDINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return {"ok": True, "holdings": data["holdings"]}


@app.put("/api/holdings/{ticker}")
async def api_update_holding(ticker: str, request: Request):
    ticker = ticker.upper().strip()
    body   = await request.json()
    data   = _load()
    existing = next((h for h in data["holdings"] if h["ticker"] == ticker), None)
    if not existing:
        return JSONResponse({"error": f"{ticker} not in portfolio"}, status_code=404)
    if "shares" in body:
        existing["shares"] = round(float(body["shares"]), 6)
    if "avg_cost" in body:
        existing["avg_cost"] = round(float(body["avg_cost"]), 4)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HOLDINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return {"ok": True, "holdings": data["holdings"]}


@app.delete("/api/holdings/{ticker}")
async def api_delete_holding(ticker: str):
    ticker = ticker.upper().strip()
    data   = _load()
    before = len(data["holdings"])
    data["holdings"] = [h for h in data["holdings"] if h["ticker"] != ticker]
    if len(data["holdings"]) == before:
        return JSONResponse({"error": f"{ticker} not found"}, status_code=404)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HOLDINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return {"ok": True, "holdings": data["holdings"]}


@app.post("/api/simulate")
async def api_simulate(request: Request):
    body     = await request.json()
    data     = _load()
    holdings = [dict(h) for h in data["holdings"]]  # deep copy

    for change in body.get("changes", []):
        ticker = change["ticker"].upper().strip()
        shares = float(change.get("shares", 0))
        action = change.get("action", "buy")
        price  = float(change.get("price", 0))

        existing = next((h for h in holdings if h["ticker"] == ticker), None)

        if action == "buy":
            if price <= 0:
                # Try to fetch current price
                try:
                    info = yf.Ticker(ticker).fast_info
                    price = float(info.last_price or info.previous_close or 0)
                except Exception:
                    price = 0
            if existing:
                total_cost   = existing["avg_cost"] * existing["shares"] + price * shares
                total_shares = existing["shares"] + shares
                existing["shares"]   = total_shares
                existing["avg_cost"] = total_cost / total_shares if total_shares else price
            else:
                holdings.append({"ticker": ticker, "shares": shares, "avg_cost": price})
        elif action == "sell":
            if existing:
                existing["shares"] -= shares
                if existing["shares"] <= 0:
                    holdings = [h for h in holdings if h["ticker"] != ticker]

    sim_data   = {**data, "holdings": [h for h in holdings if h["shares"] > 0]}
    loop       = asyncio.get_event_loop()
    result     = await loop.run_in_executor(None, _compute, sim_data)
    if result is None:
        return JSONResponse({"error": "Simulation failed — check tickers."}, status_code=400)

    return {
        "sharpe":            result["sharpe"],
        "sortino":           result["sortino"],
        "beta":              result["beta"],
        "alpha":             result["alpha"],
        "var_95":            result["var_95"],
        "max_drawdown":      result["max_drawdown"],
        "tracking_error":    result["tracking_error"],
        "port_ann_return":   result["port_ann_return"],
        "risk_summary_score": result["risk_summary_score"],
        "active_risks":      result["active_risks"],
        "attribution":       result["attribution"],
    }


@app.get("/api/news")
async def api_news():
    data    = _load()
    tickers = [h["ticker"] for h in data["holdings"]][:8]
    articles, seen = [], set()
    for t in tickers:
        try:
            for n in (yf.Ticker(t).news or [])[:4]:
                # yfinance ≥0.2.50 wraps everything under 'content'
                content  = n.get("content") or n
                title    = content.get("title", "") or n.get("title", "")
                if not title or title in seen:
                    continue
                seen.add(title)
                link = (
                    (content.get("clickThroughUrl") or {}).get("url")
                    or (content.get("canonicalUrl") or {}).get("url")
                    or n.get("link", "#")
                )
                publisher = (
                    (content.get("provider") or {}).get("displayName")
                    or n.get("publisher", "")
                )
                pub_date = content.get("pubDate") or ""
                articles.append({
                    "title":     title,
                    "publisher": publisher,
                    "link":      link or "#",
                    "ticker":    t,
                    "age":       _age_iso(pub_date) or _age(n.get("providerPublishTime", 0)),
                })
        except Exception:
            pass
    return articles[:20]


_earnings_cache: dict = {}
_EARNINGS_TTL = 4 * 3600  # 4 hours

def _fetch_earnings_one(ticker: str) -> dict:
    result: dict = {"ticker": ticker}
    try:
        t = yf.Ticker(ticker)

        # ── Next date + analyst estimates (calendar is a dict in current yfinance) ─
        try:
            cal = t.calendar
            if isinstance(cal, dict) and cal:
                # Next earnings date
                ed_list = cal.get("Earnings Date") or []
                if ed_list:
                    result["next_date"] = str(ed_list[0])[:10]
                # EPS estimate — yfinance key is "Earnings Average"
                for k in ("Earnings Average", "EPS Estimate", "epsAverage"):
                    v = cal.get(k)
                    if v is not None:
                        result["eps_estimate"] = round(float(v), 4); break
                # Revenue estimate
                for k in ("Revenue Average", "Revenue Estimate", "revenueAverage"):
                    v = cal.get(k)
                    if v is not None:
                        result["revenue_estimate"] = float(v); break
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                # Legacy DataFrame format
                def _cv(key):
                    return cal.loc[key].dropna().values[0] if key in cal.index and len(cal.loc[key].dropna()) else None
                ed = _cv("Earnings Date")
                if ed is not None: result["next_date"] = str(ed)[:10]
                for k in ("Earnings Average", "EPS Estimate"):
                    v = _cv(k)
                    if v is not None: result["eps_estimate"] = round(float(v), 4); break
                for k in ("Revenue Average", "Revenue Estimate"):
                    v = _cv(k)
                    if v is not None: result["revenue_estimate"] = float(v); break
        except Exception:
            pass

        # ── Historical earnings → last EPS + beat/miss streak ─────────────────
        # earnings_history is a DataFrame with epsActual, epsEstimate, surprisePercent
        try:
            eh = t.earnings_history
            if eh is not None and not eh.empty:
                # Sort descending by quarter index so most recent is first
                eh = eh.sort_index(ascending=False)
                past = eh[eh["epsActual"].notna()].head(8)
                if not past.empty:
                    result["last_eps"] = round(float(past.iloc[0]["epsActual"]), 2)
                    streak, sdir = 0, None
                    for _, row in past.iterrows():
                        surp = row.get("surprisePercent")
                        if surp is None or (hasattr(surp, "__float__") and pd.isna(float(surp))):
                            break
                        beat = float(surp) > 0
                        if sdir is None:
                            sdir = beat; streak = 1
                        elif beat == sdir:
                            streak += 1
                        else:
                            break
                    result["streak_dir"]   = "beat" if sdir is True else ("miss" if sdir is False else None)
                    result["streak_count"] = streak
        except Exception:
            pass

        # ── Last quarter revenue ───────────────────────────────────────────────
        try:
            qf = None
            for attr in ("quarterly_income_stmt", "quarterly_financials"):
                _qf = getattr(t, attr, None)
                if _qf is not None and isinstance(_qf, pd.DataFrame) and not _qf.empty:
                    qf = _qf; break
            if qf is not None:
                for rev_key in ("Total Revenue", "Revenue"):
                    if rev_key in qf.index:
                        result["last_revenue"] = float(qf.loc[rev_key].iloc[0])
                        break
        except Exception:
            pass

        # ── Days until ────────────────────────────────────────────────────────
        if "next_date" in result:
            try:
                nd = datetime.strptime(result["next_date"][:10], "%Y-%m-%d").date()
                result["days_until"] = (nd - date.today()).days
            except Exception:
                pass

    except Exception as exc:
        result["error"] = str(exc)
    return result


@app.get("/api/earnings")
async def api_earnings():
    data    = _load()
    tickers = [h["ticker"] for h in data.get("holdings", [])]
    if not tickers:
        return []

    cache_key = tuple(sorted(tickers))
    now = time.time()
    if cache_key in _earnings_cache:
        rows, fetched_at = _earnings_cache[cache_key]
        if now - fetched_at < _EARNINGS_TTL:
            return rows

    from concurrent.futures import ThreadPoolExecutor
    import math
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(_fetch_earnings_one, tickers))

    # Sanitize: replace NaN/inf floats (non-JSON-serializable) with None
    def _clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    rows = [
        {k: _clean(v) for k, v in r.items()}
        for r in rows
        if r.get("next_date") and r.get("days_until") is not None
    ]
    rows.sort(key=lambda x: x["days_until"])
    _earnings_cache[cache_key] = (rows, now)
    return rows


def _age_iso(iso: str) -> str:
    """Parse ISO-8601 pubDate string → '3h ago' style label."""
    if not iso:
        return ""
    try:
        from datetime import timezone
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        d  = datetime.now(timezone.utc) - dt
        h  = int(d.total_seconds() // 3600)
        return f"{int(d.total_seconds()//60)}m ago" if h < 1 else (f"{h}h ago" if h < 24 else f"{h//24}d ago")
    except Exception:
        return ""


def _age(ts: int) -> str:
    if not ts:
        return ""
    d = datetime.now() - datetime.fromtimestamp(ts)
    h = int(d.total_seconds() // 3600)
    return f"{int(d.total_seconds()//60)}m ago" if h < 1 else (f"{h}h ago" if h < 24 else f"{h//24}d ago")


# ── Dashboard HTML ─────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Portfolio Risk Tracker</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg:#080b14; --surface:#0d1120; --card:#111827; --card2:#0f1621;
  --border:#1f2a3d; --border2:#263044;
  --gold:#e8a020; --gold-dim:rgba(232,160,32,.15);
  --green:#22c55e; --green-dim:rgba(34,197,94,.12);
  --red:#ef4444;   --red-dim:rgba(239,68,68,.12);
  --blue:#3b82f6;  --blue-dim:rgba(59,130,246,.12);
  --purple:#8b5cf6; --yellow:#eab308;
  --muted:#6b7280; --text:#e2e8f0; --text2:#94a3b8;
  --radius:10px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',sans-serif;font-size:13px;min-height:100vh}

/* Nav */
nav{display:flex;align-items:center;justify-content:space-between;padding:0 24px;height:52px;background:var(--surface);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:100}
.nav-brand{font-size:15px;font-weight:700;color:var(--gold);letter-spacing:.5px}
.nav-tabs{display:flex;gap:4px}
.nav-tab{padding:6px 16px;border-radius:6px;cursor:pointer;color:var(--text2);font-weight:500;font-size:12px;border:none;background:none;transition:all .15s;text-decoration:none;display:inline-flex;align-items:center}
.nav-tab.active,.nav-tab:hover{background:var(--border);color:var(--text)}
.nav-right{display:flex;align-items:center;gap:16px}
#clock{color:var(--text2);font-size:12px}
.refresh-row{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--text2)}
#refresh-dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* Layout */
.page{display:grid;grid-template-columns:1fr 340px;gap:14px;padding:14px 18px;max-width:1700px;margin:0 auto}

/* Metric bar */
.metric-bar{grid-column:1/-1;display:grid;grid-template-columns:2fr repeat(5,1fr);gap:10px}
.metric-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:14px 18px}
.metric-card.primary{border-color:var(--gold)}
.metric-label{color:var(--text2);font-size:10px;text-transform:uppercase;letter-spacing:.8px;margin-bottom:5px}
.metric-value{font-size:26px;font-weight:700;color:var(--text);line-height:1}
.metric-card.primary .metric-value{color:var(--gold)}
.metric-sub{margin-top:5px;font-size:12px}
.metric-card:not(.primary) .metric-value{font-size:18px}

/* Cards */
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:14px 18px}
.card-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:var(--text2);margin-bottom:12px;display:flex;align-items:center;justify-content:space-between;gap:8px}
.card-title .right{display:flex;align-items:center;gap:6px}

/* Chart */
.chart-wrap{position:relative;height:210px}
.chart-sm{position:relative;height:140px}
.chart-xs{position:relative;height:120px}

/* Chart tab buttons */
.chart-tabs{display:flex;gap:4px}
.chart-tab{padding:3px 10px;border-radius:4px;border:1px solid var(--border2);background:none;color:var(--text2);font-size:11px;cursor:pointer;transition:all .12s}
.chart-tab.active{background:var(--gold-dim);border-color:var(--gold);color:var(--gold)}

/* Tables */
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:12px}
thead th{color:var(--text2);font-weight:600;text-transform:uppercase;font-size:10px;letter-spacing:.5px;padding:7px 10px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}
thead th:first-child{text-align:left}
tbody tr{border-bottom:1px solid rgba(31,42,61,.4);transition:background .1s}
tbody tr:hover{background:rgba(255,255,255,.03)}
tbody td{padding:8px 10px;text-align:right;white-space:nowrap}
tbody td:first-child{text-align:left;font-weight:600}
tfoot td{padding:9px 10px;text-align:right;font-weight:700;border-top:1px solid var(--border);color:var(--text2)}
tfoot td:first-child{text-align:left}

/* Ticker cell */
.ticker-cell{display:flex;align-items:center;gap:7px}
.ticker-dot{width:6px;height:6px;border-radius:50%;background:var(--gold);flex-shrink:0}

/* Inline bar */
.bar-wrap{display:flex;align-items:center;gap:5px;justify-content:flex-end}
.bar-bg{width:52px;height:3px;background:var(--border);border-radius:2px;overflow:hidden}
.bar-fill{height:100%;border-radius:2px;background:var(--gold)}
.bar-fill.neg{background:var(--red)}

/* Risk metric rows */
.metric-row{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(31,42,61,.4)}
.metric-row:last-child{border-bottom:none}
.metric-row-label{color:var(--text2);font-size:12px}
.metric-section{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);padding:10px 0 4px;border-bottom:1px solid var(--border2);margin-bottom:2px}

/* Pills */
.pill{padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}
.pill.green{background:var(--green-dim);color:var(--green)}
.pill.yellow{background:rgba(234,179,8,.12);color:var(--yellow)}
.pill.red{background:var(--red-dim);color:var(--red)}
.pill.blue{background:var(--blue-dim);color:var(--blue)}
.pill.purple{background:rgba(139,92,246,.12);color:var(--purple)}
.pill.gold{background:var(--gold-dim);color:var(--gold)}

/* Capture ratio display */
.capture-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:6px}
.capture-card{background:var(--card2);border:1px solid var(--border2);border-radius:8px;padding:10px 12px;text-align:center}
.capture-val{font-size:22px;font-weight:700;margin:4px 0}
.capture-label{font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:.6px}

/* Sector legend */
.sector-legend{display:flex;flex-direction:column;gap:5px}
.sector-row{display:flex;align-items:center;justify-content:space-between;font-size:12px}
.sector-name{display:flex;align-items:center;gap:7px}
.sector-dot{width:8px;height:8px;border-radius:2px;flex-shrink:0}

/* Benchmark table */
.bm-active-row td{background:rgba(232,160,32,.05)!important}

/* Monthly heatmap */
.heatmap{display:grid;grid-template-columns:repeat(12,1fr);gap:2px;font-size:10px}
.hm-cell{padding:4px 2px;border-radius:3px;text-align:center;font-weight:600}

/* Risk bubble map tooltip */
.rbm-tooltip{position:absolute;background:#1e293b;border:1px solid #334155;border-radius:6px;padding:8px 10px;font-size:11px;pointer-events:none;display:none;z-index:10;min-width:140px}

/* Active risk cards */
.active-risk-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:4px}
.ar-card{background:var(--card2);border:1px solid var(--border2);border-radius:8px;padding:10px;text-align:center}
.ar-val{font-size:18px;font-weight:700;margin:3px 0}
.ar-label{font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:.5px}

/* Returners / Losers */
.rl-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.rl-row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid rgba(31,42,61,.3)}
.rl-row:last-child{border-bottom:none}
.rl-ticker{font-weight:700;font-size:13px}
.rl-sub{font-size:10px;color:var(--text2)}
.rl-ret{font-weight:700;font-size:13px}

/* Watchlist */
.wl-row{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(31,42,61,.3)}
.wl-row:last-child{border-bottom:none}
.wl-ticker{font-weight:700;font-size:13px;color:var(--gold)}
.wl-price{font-size:13px;font-weight:600}
.wl-chg{font-size:11px}
.wl-cols{display:flex;gap:10px;align-items:center}

/* Sidebar */
.sidebar{display:flex;flex-direction:column;gap:12px}

/* News */
.news-item{padding:9px 0;border-bottom:1px solid rgba(31,42,61,.4);display:flex;flex-direction:column;gap:3px}
.news-item:last-child{border-bottom:none}
.news-title a{color:var(--text);text-decoration:none;font-size:12px;line-height:1.4;display:block}
.news-title a:hover{color:var(--gold)}
.news-meta{display:flex;gap:8px;font-size:11px;color:var(--text2)}
.news-ticker{color:var(--gold);font-weight:600}

/* Colors */
.green{color:var(--green)} .red{color:var(--red)} .gold{color:var(--gold)} .muted{color:var(--text2)} .blue{color:var(--blue)}

/* Spinner */
.spinner{display:flex;align-items:center;justify-content:center;height:80px;color:var(--text2);gap:10px}
.spin{width:16px;height:16px;border:2px solid var(--border);border-top-color:var(--gold);border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* Provider badge */
.provider-badge{font-size:10px;padding:2px 7px;border-radius:4px;background:var(--blue-dim);color:var(--blue);border:1px solid rgba(59,130,246,.2)}

@media(max-width:1100px){.page{grid-template-columns:1fr}.metric-bar{grid-template-columns:repeat(3,1fr)}.rl-grid{grid-template-columns:1fr}}

/* Tab panels */
.tab-panel{display:none}
#tab-overview.active{display:contents}
#tab-risk.active,#tab-scenarios.active{display:block;grid-column:1/-1}

/* Risk score ring */
.risk-score-wrap{display:flex;align-items:center;gap:12px;margin-top:6px}
.risk-ring{position:relative;width:56px;height:56px;flex-shrink:0}
.risk-ring svg{transform:rotate(-90deg)}
.risk-ring-label{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;font-weight:700;font-size:14px;line-height:1}
.risk-ring-sub{font-size:8px;color:var(--text2);margin-top:1px;font-weight:400}

/* Collapsible */
.collapsible-toggle{cursor:pointer;user-select:none}
.collapsible-toggle::after{content:' ▾';font-size:10px;color:var(--text2)}
.collapsible-toggle.collapsed::after{content:' ▸'}
.collapsible-body{overflow-y:hidden;overflow-x:visible;transition:max-height .2s}

/* Correlation heatmap */
.corr-wrap{overflow:auto}

/* Stress table */
.stress-tbl td.neg{color:var(--red)} .stress-tbl td.pos{color:var(--green)}

/* Form inputs */
.sim-input{background:var(--card2);border:1px solid var(--border2);border-radius:6px;color:var(--text);font-size:12px;padding:7px 10px;width:100%;outline:none;font-family:inherit}
.sim-input:focus{border-color:var(--gold)}
.sim-select{background:var(--card2);border:1px solid var(--border2);border-radius:6px;color:var(--text);font-size:12px;padding:7px 10px;width:100%;outline:none;font-family:inherit;cursor:pointer}
.btn{padding:7px 16px;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;border:none;transition:all .15s;font-family:inherit}
.btn-gold{background:var(--gold);color:#000}
.btn-gold:hover{opacity:.85}
.btn-outline{background:none;border:1px solid var(--border2);color:var(--text2)}
.btn-outline:hover{border-color:var(--gold);color:var(--gold)}
.btn-danger{background:var(--red-dim);border:1px solid var(--red);color:var(--red)}

/* Sim trade tag */
.sim-trade-tag{display:inline-flex;align-items:center;gap:6px;background:var(--card2);border:1px solid var(--border2);border-radius:6px;padding:5px 10px;font-size:12px;margin:3px}
.sim-trade-tag.buy{border-color:rgba(34,197,94,.4)}
.sim-trade-tag.sell{border-color:rgba(239,68,68,.4)}

/* Before/after comparison */
.ba-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px}
.ba-card{background:var(--card2);border:1px solid var(--border2);border-radius:8px;padding:10px;text-align:center}
.ba-label{font-size:9px;color:var(--text2);text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px}
.ba-before{font-size:11px;color:var(--text2)}
.ba-after{font-size:16px;font-weight:700;margin:2px 0}
.ba-delta{font-size:11px;font-weight:600}

/* Metric explanation tooltip */
.metric-row{position:relative}
.metric-expl{font-size:10px;color:var(--text2);margin-top:2px;padding-bottom:4px;line-height:1.4;display:none}
.metric-row.expanded .metric-expl{display:block}
.metric-row-label{cursor:pointer;display:flex;align-items:center;gap:4px}
.metric-row-label::after{content:'ⓘ';font-size:9px;color:var(--muted);opacity:.6}
.metric-row-label:hover::after{opacity:1;color:var(--gold)}

/* Sector concentration bar */
.sector-conc-bar{height:6px;border-radius:3px;background:var(--border);overflow:hidden;margin-top:4px;margin-bottom:8px}
.sector-conc-fill{height:100%;border-radius:3px;transition:width .3s}

/* Rebalancing */
.rebal-row{display:grid;grid-template-columns:1fr 38px 38px 54px 70px;gap:4px;align-items:center;padding:7px 0;border-bottom:1px solid rgba(31,42,61,.3);font-size:11px}
.rebal-row:last-child{border-bottom:none}
.rebal-bar-wrap{position:relative;height:4px;background:var(--border);border-radius:2px;overflow:visible;grid-column:1}
.rebal-bar-cur{position:absolute;top:0;left:0;height:100%;border-radius:2px;transition:width .3s}
.rebal-target-line{position:absolute;top:-3px;width:2px;height:10px;background:var(--text2);border-radius:1px}
.action-trim{color:var(--red);font-weight:700;font-size:10px}
.action-add{color:var(--green);font-weight:700;font-size:10px}
.action-hold{color:var(--muted);font-size:10px}

/* Hedge fund tracker */
.hf-fund-header{display:flex;justify-content:space-between;align-items:baseline;padding:10px 0 6px;border-bottom:1px solid var(--border2);margin-bottom:6px}
.hf-fund-name{font-weight:700;font-size:13px;color:var(--gold)}
.hf-fund-meta{font-size:10px;color:var(--text2)}
.hf-change-badge{display:inline-flex;align-items:center;gap:4px;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:700;margin:2px}
.hf-new{background:rgba(34,197,94,.15);color:var(--green);border:1px solid rgba(34,197,94,.3)}
.hf-closed{background:rgba(239,68,68,.12);color:var(--red);border:1px solid rgba(239,68,68,.25)}
.hf-inc{background:rgba(59,130,246,.12);color:var(--blue);border:1px solid rgba(59,130,246,.25)}
.hf-dec{background:rgba(234,179,8,.12);color:var(--yellow);border:1px solid rgba(234,179,8,.25)}
.hf-holding-row{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid rgba(31,42,61,.25);font-size:11px}
.hf-holding-row:last-child{border-bottom:none}
.hf-tabs{display:flex;gap:4px;margin-bottom:10px}
.hf-tab{padding:3px 10px;border-radius:4px;border:1px solid var(--border2);background:none;color:var(--text2);font-size:11px;cursor:pointer;font-family:inherit}
.hf-tab.active{background:var(--gold-dim);border-color:var(--gold);color:var(--gold)}

/* ── Earnings ────────────────────────────────────────────────────────────── */
.earn-table{width:100%;border-collapse:collapse;font-size:11px}
.earn-table th{font-size:9px;text-transform:uppercase;letter-spacing:.5px;color:var(--text2);font-weight:600;padding:5px 8px;border-bottom:1px solid var(--border);text-align:center;white-space:nowrap}
.earn-table th:first-child{text-align:left}
.earn-table td{padding:7px 8px;border-bottom:1px solid rgba(31,42,61,.25);vertical-align:middle;text-align:center}
.earn-table td:first-child{text-align:left;font-weight:700}
.earn-table tr:last-child td{border-bottom:none}
.earn-table tr.earn-expanded td{border-bottom:none}
.earn-row{cursor:pointer}
.earn-row:hover td{background:rgba(255,255,255,.02)}
.earn-countdown{display:inline-flex;align-items:center;justify-content:center;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:700;white-space:nowrap}
.earn-urgent{background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.3)}
.earn-soon{background:rgba(234,179,8,.15);color:var(--gold);border:1px solid rgba(234,179,8,.3)}
.earn-future{background:rgba(100,116,139,.12);color:var(--text2);border:1px solid var(--border2)}
.earn-beat{color:var(--green);font-weight:700}
.earn-miss{color:var(--red);font-weight:700}
.earn-detail-row{display:none}
.earn-detail-row.open{display:table-row}
.earn-detail-cell{background:var(--card2);padding:12px 16px!important;border-bottom:1px solid var(--border)!important}
.earn-detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:700px){.earn-detail-grid{grid-template-columns:1fr}}
.earn-notes-label{font-size:9px;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;font-weight:600;display:block;margin-bottom:5px}
.earn-notes-text{font-size:12px;color:var(--text);line-height:1.5;white-space:pre-wrap}
.earn-notes-empty{font-size:11px;color:var(--muted);font-style:italic}

/* ── Alert panels (review queue + 52W) ──────────────────────────────────── */
.alert-panel{grid-column:1/-1;background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:12px 16px;margin-bottom:0}
.alert-panel-hdr{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.alert-panel-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.8px}
.alert-count{display:inline-flex;align-items:center;justify-content:center;min-width:20px;height:20px;border-radius:10px;font-size:10px;font-weight:700;padding:0 5px}
.alert-row{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid rgba(31,42,61,.3);font-size:12px;flex-wrap:wrap}
.alert-row:last-child{border-bottom:none}
.alert-tag{padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.3px;flex-shrink:0}

/* ── Thesis ──────────────────────────────────────────────────────────────── */
.thesis-dot{width:10px;height:10px;border-radius:50%;display:inline-block;cursor:pointer;flex-shrink:0;transition:transform .12s;vertical-align:middle;border:1px solid rgba(255,255,255,.15)}
.thesis-dot:hover{transform:scale(1.4)}
.thesis-dot-none{background:#374151}
.thesis-card-row td{padding:0!important;background:var(--card2)!important}
.thesis-card{padding:16px 20px;border-top:2px solid var(--border2)}
.thesis-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:700px){.thesis-grid{grid-template-columns:1fr}}
.thesis-field{margin-bottom:12px}
.thesis-field-label{font-size:9px;color:var(--text2);text-transform:uppercase;letter-spacing:.6px;display:block;margin-bottom:4px;font-weight:600}
.thesis-field p{font-size:12px;color:var(--text);margin:0;line-height:1.55}
.thesis-meta-strip{display:flex;gap:14px;flex-wrap:wrap;background:rgba(255,255,255,.03);border:1px solid var(--border2);border-radius:8px;padding:10px 12px;margin-bottom:12px}
.thesis-stat-label{font-size:9px;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;display:block;margin-bottom:2px}
.thesis-stat-val{font-size:13px;font-weight:600;color:var(--text)}
.thesis-notes-log{max-height:140px;overflow-y:auto;background:rgba(0,0,0,.2);border-radius:6px;padding:8px;margin-top:4px}
.thesis-note-item{padding:5px 0;border-bottom:1px solid rgba(31,42,61,.3);font-size:11px;color:var(--text)}
.thesis-note-item:last-child{border-bottom:none}
.thesis-note-ts{font-size:9px;color:var(--text2);margin-bottom:2px;font-weight:600}
.thesis-actions{display:flex;gap:8px;margin-top:14px;padding-top:12px;border-top:1px solid var(--border2)}
.thesis-broken-row td{background:rgba(239,68,68,.05)!important}
.thesis-broken-row td:first-child{border-left:3px solid var(--red)}
.catalyst-passed{color:var(--gold);font-size:10px;margin-top:3px}

/* ── Thesis modal ────────────────────────────────────────────────────────── */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:200;display:flex;align-items:center;justify-content:center}
.modal-box{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:24px;width:580px;max-width:96vw;max-height:90vh;overflow-y:auto;z-index:201;position:relative}
.modal-title{font-size:15px;font-weight:700;margin-bottom:16px;color:var(--text);display:flex;align-items:center;gap:8px}
.modal-ticker{color:var(--gold)}
.form-group{margin-bottom:11px}
.form-group label{font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;display:block;margin-bottom:4px;font-weight:600}
.form-group input,.form-group textarea,.form-group select{width:100%;background:var(--card2);border:1px solid var(--border2);border-radius:6px;color:var(--text);padding:7px 10px;font-size:12px;font-family:inherit;outline:none;box-sizing:border-box}
.form-group textarea{resize:vertical;min-height:56px}
.form-group input:focus,.form-group textarea:focus,.form-group select:focus{border-color:var(--gold)}
.form-group select option{background:var(--card)}
.form-row-2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.form-row-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.modal-actions{display:flex;gap:8px;justify-content:flex-end;margin-top:16px;padding-top:12px;border-top:1px solid var(--border)}

/* ── 52-week range bar ───────────────────────────────────────────────────── */
.range-bar-wrap{min-width:72px}
.range-bar-track{height:3px;background:rgba(255,255,255,.08);border-radius:2px;position:relative;margin-bottom:3px;margin-top:6px}
.range-bar-marker{width:8px;height:8px;border-radius:50%;position:absolute;top:50%;transform:translate(-50%,-50%)}
.range-bar-labels{display:flex;justify-content:space-between;font-size:8px;color:var(--text2);line-height:1}
.range-pct-lbl{color:var(--text);font-weight:700}
</style>
</head>
<body>

<nav>
  <div class="nav-brand">&#9650; Portfolio Risk Tracker</div>
  <div class="nav-tabs">
    <button class="nav-tab active" onclick="switchTab('overview',this)">Overview</button>
    <button class="nav-tab" onclick="switchTab('risk',this)">Risk</button>
    <button class="nav-tab" onclick="switchTab('scenarios',this)">Scenarios</button>
    <a class="nav-tab" href="https://github.com/sofiacieplinski/portfolio-risk-tracker" target="_blank" rel="noopener" style="margin-left:8px">GitHub ↗</a>
  </div>
  <div class="nav-right">
    <div class="refresh-row"><div id="refresh-dot"></div><span id="last-updated">Loading…</span></div>
    <div id="clock"></div>
  </div>
</nav>

<div class="page">

  <!-- Metric bar -->
  <div class="metric-bar" style="grid-template-columns:2fr repeat(6,1fr)">
    <div class="metric-card primary">
      <div class="metric-label">Total Portfolio Value</div>
      <div class="metric-value" id="total-value">—</div>
      <div class="metric-sub" id="day-change-row">—</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Annual Return</div>
      <div class="metric-value" id="ann-return">—</div>
      <div class="metric-sub muted" id="bench-return">—</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Active Return</div>
      <div class="metric-value" id="active-return">—</div>
      <div class="metric-sub muted">vs <span id="bench-name">SPY</span></div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Sharpe / Sortino</div>
      <div class="metric-value" id="sharpe-val">—</div>
      <div class="metric-sub muted" id="sortino-val">—</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Alpha / Beta</div>
      <div class="metric-value" id="alpha-val">—</div>
      <div class="metric-sub muted" id="beta-val">—</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Max Drawdown</div>
      <div class="metric-value" id="maxdd-val">—</div>
      <div class="metric-sub muted" id="calmar-val">—</div>
    </div>
    <div class="metric-card" style="text-align:center">
      <div class="metric-label">Risk Score</div>
      <div style="display:flex;align-items:center;justify-content:center;margin-top:4px">
        <div class="risk-ring"><svg width="56" height="56" viewBox="0 0 56 56">
          <circle cx="28" cy="28" r="22" fill="none" stroke="#1f2a3d" stroke-width="5"/>
          <circle cx="28" cy="28" r="22" fill="none" id="risk-ring-arc" stroke="#ef4444" stroke-width="5" stroke-linecap="round" stroke-dasharray="138.2" stroke-dashoffset="138.2"/>
        </svg>
        <div class="risk-ring-label"><span id="risk-score-val">—</span><span class="risk-ring-sub" id="risk-score-lbl">—</span></div></div>
      </div>
    </div>
  </div>

  <!-- ── Review Queue (only visible when broken/flagged theses exist) ──── -->
  <div id="review-queue" class="alert-panel" style="display:none;border-left:3px solid var(--red)">
    <div class="alert-panel-hdr">
      <span class="alert-panel-title" style="color:var(--red)">Thesis Review Queue</span>
      <span class="alert-count" id="rq-count" style="background:rgba(239,68,68,.15);color:var(--red)">0</span>
      <span style="font-size:10px;color:var(--text2);margin-left:4px">Broken theses or overdue catalysts requiring attention</span>
    </div>
    <div id="review-queue-body"></div>
  </div>

  <!-- ── Price Alerts (52-week range) ─────────────────────────────────── -->
  <div id="price-alerts" class="alert-panel" style="display:none;border-left:3px solid var(--gold)">
    <div class="alert-panel-hdr">
      <span class="alert-panel-title" style="color:var(--gold)">Price Range Alerts</span>
      <span class="alert-count" id="pa-count" style="background:rgba(234,179,8,.15);color:var(--gold)">0</span>
      <span style="font-size:10px;color:var(--text2);margin-left:4px">Holdings near 52-week extremes</span>
      <span style="margin-left:auto;font-size:10px;color:var(--text2)">
        Near 52W High: <span id="stat-near-high" style="color:var(--red);font-weight:700">0</span> &nbsp;·&nbsp;
        Near 52W Low: <span id="stat-near-low" style="color:var(--green);font-weight:700">0</span>
      </span>
    </div>
    <div id="price-alerts-body"></div>
  </div>

  <!-- ── Earnings Alerts (≤7 days) ───────────────────────────────────────── -->
  <div id="earnings-alerts" class="alert-panel" style="display:none;border-left:3px solid var(--blue)">
    <div class="alert-panel-hdr">
      <span class="alert-panel-title" style="color:var(--blue)">Upcoming Earnings</span>
      <span class="alert-count" id="ea-count" style="background:rgba(59,130,246,.15);color:var(--blue)">0</span>
      <span style="font-size:10px;color:var(--text2);margin-left:4px">Holdings reporting within 7 days — scroll down to Earnings Calendar for full details</span>
    </div>
    <div id="earnings-alerts-body"></div>
  </div>

  <!-- ── OVERVIEW TAB ────────────────────────────────────────────────────── -->
  <div id="tab-overview" class="tab-panel active">

  <!-- LEFT COLUMN -->
  <div style="display:flex;flex-direction:column;gap:12px">

    <!-- Equity / Drawdown chart -->
    <div class="card">
      <div class="card-title">
        <div style="display:flex;align-items:center;gap:10px">
          <div class="chart-tabs">
            <button class="chart-tab active" onclick="setChartMode('equity',this)">Equity Curve</button>
            <button class="chart-tab" onclick="setChartMode('drawdown',this)">Drawdown</button>
          </div>
          <div style="display:flex;align-items:center;gap:6px;font-size:11px;font-weight:400;text-transform:none;letter-spacing:0;color:var(--text2)">
            <span style="display:inline-block;width:18px;height:2px;background:var(--gold)"></span>Portfolio
            <span style="display:inline-block;width:18px;height:2px;background:#374151;margin-left:5px"></span>Benchmark
          </div>
        </div>
        <div class="right"><span id="provider-badge" class="provider-badge">yfinance</span></div>
      </div>
      <div class="chart-wrap"><canvas id="equity-chart"></canvas></div>
    </div>

    <!-- Risk Metrics + Active Risks side by side -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
      <div class="card" style="overflow-y:auto;max-height:600px">
        <div class="card-title">Risk Metrics <span style="color:var(--text2);font-size:10px;font-weight:400;text-transform:none">1Y Annualised</span></div>
        <div id="risk-metrics"><div class="spinner"><div class="spin"></div></div></div>
      </div>
      <div class="card" style="overflow-y:auto;max-height:600px">
        <div class="card-title">Active Risks &amp; Concentration</div>
        <div id="active-risks"><div class="spinner"><div class="spin"></div></div></div>
      </div>
    </div>

    <!-- Biggest Returners & Losers -->
    <div class="rl-grid">
      <div class="card">
        <div class="card-title" style="color:var(--green)">&#9650; Biggest Returners <span style="color:var(--text2);font-size:10px;font-weight:400;text-transform:none;letter-spacing:0">1-year period</span></div>
        <div id="returners-list"><div class="spinner"><div class="spin"></div></div></div>
      </div>
      <div class="card">
        <div class="card-title" style="color:var(--red)">&#9660; Biggest Losers <span style="color:var(--text2);font-size:10px;font-weight:400;text-transform:none;letter-spacing:0">1-year period</span></div>
        <div id="losers-list"><div class="spinner"><div class="spin"></div></div></div>
      </div>
    </div>

    <!-- Holdings table -->
    <div class="card">
      <div class="card-title" style="display:flex;justify-content:space-between;align-items:center">
        <span>Holdings</span>
        <button class="btn btn-outline" style="font-size:11px;padding:4px 12px;text-transform:none;letter-spacing:0" onclick="openPositionsModal()">+ Manage Positions</button>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead><tr>
            <th>Ticker</th><th>Shares</th><th>Avg Cost</th><th>Price</th>
            <th>Mkt Value</th><th>Gain/Loss</th><th>Day Chg</th>
            <th>Weight</th><th>Period Ret</th><th>Contribution</th><th>MCTR</th>
            <th title="Thesis status — click dot to open">Thesis</th>
            <th title="52-week range position">52W Range</th>
          </tr></thead>
          <tbody id="holdings-body"><tr><td colspan="13"><div class="spinner"><div class="spin"></div>Fetching market data…</div></td></tr></tbody>
          <tfoot id="holdings-foot"></tfoot>
        </table>
      </div>
    </div>

    <!-- Earnings Calendar -->
    <div class="card">
      <div class="card-title" style="display:flex;justify-content:space-between;align-items:center">
        <span>Earnings Calendar</span>
        <span style="font-size:10px;font-weight:400;color:var(--text2);text-transform:none;letter-spacing:0">Click a row to view/edit notes · data refreshes every 4h</span>
      </div>
      <div id="earnings-wrap"><div class="spinner"><div class="spin"></div>Loading earnings data…</div></div>
    </div>

    <!-- Benchmark comparison -->
    <div class="card">
      <div class="card-title collapsible-toggle" onclick="toggleCollapse(this)">Benchmark Comparison
        <span style="color:var(--text2);font-size:10px;font-weight:400;text-transform:none;letter-spacing:0">Portfolio vs major indexes</span>
      </div>
      <div class="collapsible-body">
        <div class="tbl-wrap">
          <table id="bm-table">
            <thead><tr>
              <th>Index</th><th>1M</th><th>3M</th><th>YTD</th><th>1Y</th><th>Vol</th>
              <th>Port 1M</th><th>Port 1Y</th><th>Beta vs</th><th>Alpha vs</th>
            </tr></thead>
            <tbody id="bm-body"><tr><td colspan="10"><div class="spinner"><div class="spin"></div>Loading…</div></td></tr></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Hedge Fund Tracker -->
    <div class="card">
      <div class="card-title collapsible-toggle" onclick="toggleCollapse(this)">Hedge Fund Position Tracker
        <span style="font-size:10px;font-weight:400;text-transform:none;letter-spacing:0;color:var(--text2)">13F filings · quarterly · SEC EDGAR</span>
      </div>
      <div class="collapsible-body">
        <div class="hf-tabs" id="hf-tabs"></div>
        <div id="hf-panel"><div class="spinner"><div class="spin"></div>Loading SEC filings…</div></div>
      </div>
    </div>

    <!-- Portfolio Simulator (starts collapsed) -->
    <div class="card">
      <div class="card-title collapsible-toggle collapsed" onclick="toggleCollapse(this)">Portfolio Simulator
        <span style="font-size:10px;font-weight:400;text-transform:none;letter-spacing:0;color:var(--text2)">test hypothetical buys &amp; sells</span>
      </div>
      <div class="collapsible-body" style="max-height:0">
        <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr auto;gap:8px;align-items:end;margin-bottom:10px">
          <div>
            <div style="font-size:10px;color:var(--text2);margin-bottom:3px;text-transform:uppercase;letter-spacing:.5px">Ticker</div>
            <input id="sim-ticker" class="sim-input" type="text" placeholder="e.g. NVDA" style="text-transform:uppercase">
          </div>
          <div>
            <div style="font-size:10px;color:var(--text2);margin-bottom:3px;text-transform:uppercase;letter-spacing:.5px">Shares</div>
            <input id="sim-shares" class="sim-input" type="number" placeholder="100" min="0">
          </div>
          <div>
            <div style="font-size:10px;color:var(--text2);margin-bottom:3px;text-transform:uppercase;letter-spacing:.5px">Price ($)</div>
            <input id="sim-price" class="sim-input" type="number" placeholder="auto" min="0" step="0.01">
          </div>
          <div>
            <div style="font-size:10px;color:var(--text2);margin-bottom:3px;text-transform:uppercase;letter-spacing:.5px">Action</div>
            <select id="sim-action" class="sim-select">
              <option value="buy">Buy</option>
              <option value="sell">Sell</option>
            </select>
          </div>
          <button class="btn btn-outline" onclick="addSimTrade()">+ Add</button>
        </div>
        <div id="sim-trades" style="min-height:28px;margin-bottom:10px">
          <span style="color:var(--text2);font-size:11px">No trades added yet.</span>
        </div>
        <div style="display:flex;gap:8px;align-items:center">
          <button class="btn btn-gold" onclick="runSimulation()" id="sim-run-btn">Run Simulation</button>
          <button class="btn btn-outline" onclick="clearSimulation()">Clear</button>
          <span id="sim-status" style="font-size:11px;color:var(--text2)"></span>
        </div>
        <div id="sim-results" style="display:none;margin-top:14px">
          <div style="font-size:11px;color:var(--text2);margin-bottom:6px;text-transform:uppercase;letter-spacing:.6px">Before → After</div>
          <div class="ba-grid" id="ba-grid"></div>
          <div style="margin-top:12px">
            <div style="font-size:11px;color:var(--text2);margin-bottom:6px;text-transform:uppercase;letter-spacing:.6px">Simulated Holdings</div>
            <div class="tbl-wrap">
              <table style="font-size:11px">
                <thead><tr><th>Ticker</th><th>Shares</th><th>Weight</th><th>Period Ret</th><th>MCTR</th></tr></thead>
                <tbody id="sim-holdings-body"></tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>

  </div>

  <!-- RIGHT SIDEBAR -->
  <div class="sidebar">

    <!-- Capture ratios + batting avg -->
    <div class="card">
      <div class="card-title">Capture Ratios</div>
      <div class="capture-grid">
        <div class="capture-card">
          <div class="capture-label">Upside Capture</div>
          <div class="capture-val green" id="up-cap">—</div>
          <div style="font-size:10px;color:var(--text2)">% of benchmark gains</div>
        </div>
        <div class="capture-card">
          <div class="capture-label">Downside Capture</div>
          <div class="capture-val" id="dn-cap">—</div>
          <div style="font-size:10px;color:var(--text2)">% of benchmark losses</div>
        </div>
      </div>
      <div style="margin-top:10px;display:flex;justify-content:space-between">
        <div style="text-align:center;flex:1">
          <div style="font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:.6px">Batting Avg</div>
          <div style="font-size:18px;font-weight:700;margin-top:3px" id="bat-avg">—</div>
          <div style="font-size:10px;color:var(--text2)">months beating bmk</div>
        </div>
        <div style="text-align:center;flex:1">
          <div style="font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:.6px">R-Squared</div>
          <div style="font-size:18px;font-weight:700;margin-top:3px" id="r-sq">—</div>
          <div style="font-size:10px;color:var(--text2)">explained by bmk</div>
        </div>
      </div>
    </div>

    <!-- Rolling Sharpe / Beta -->
    <div class="card">
      <div class="card-title">Rolling 63-Day
        <div class="chart-tabs">
          <button class="chart-tab active" onclick="setRollingMode('sharpe',this)">Sharpe</button>
          <button class="chart-tab" onclick="setRollingMode('beta',this)">Beta</button>
        </div>
      </div>
      <div class="chart-sm"><canvas id="rolling-chart"></canvas></div>
    </div>


    <!-- Sector Allocation -->
    <div class="card">
      <div class="card-title">Sector Allocation</div>
      <div style="position:relative;height:190px;margin-bottom:10px"><canvas id="sector-chart"></canvas></div>
      <div class="sector-legend" id="sector-legend"><div class="spinner" style="height:40px"><div class="spin"></div></div></div>
    </div>

    <!-- Rebalancing Suggestions -->
    <div class="card">
      <div class="card-title collapsible-toggle" onclick="toggleCollapse(this)">Rebalancing Suggestions
        <span style="font-size:10px;font-weight:400;text-transform:none;letter-spacing:0;color:var(--text2)">vs S&amp;P 500</span>
      </div>
      <div class="collapsible-body">
        <div style="display:grid;grid-template-columns:1fr 38px 38px 54px 70px;gap:4px;padding:4px 0 6px;font-size:9px;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid var(--border2)">
          <span>Sector</span><span style="text-align:right">Cur%</span><span style="text-align:right">Tgt%</span><span style="text-align:right">Drift</span><span style="text-align:right">Action</span>
        </div>
        <div id="rebal-list"><div class="spinner" style="height:50px"><div class="spin"></div></div></div>
      </div>
    </div>

    <!-- Watchlist -->
    <div class="card">
      <div class="card-title">Watchlist
        <span style="color:var(--text2);font-size:10px;font-weight:400;text-transform:none;letter-spacing:0">~/.portfolio_tracker/watchlist.json</span>
      </div>
      <div id="watchlist-feed"><div class="spinner"><div class="spin"></div></div></div>
    </div>

    <!-- Market News -->
    <div class="card">
      <div class="card-title">Market News</div>
      <div id="news-feed"><div class="spinner"><div class="spin"></div></div></div>
    </div>

  </div>
  </div><!-- /tab-overview -->

  <!-- ── RISK TAB ──────────────────────────────────────────────────────────── -->
  <div id="tab-risk" class="tab-panel" style="grid-column:1/-1">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">

    <!-- Correlation Matrix -->
    <div class="card" style="grid-column:1/-1">
      <div class="card-title collapsible-toggle" onclick="toggleCollapse(this)">Correlation Matrix
        <span style="font-size:10px;font-weight:400;text-transform:none;letter-spacing:0;color:var(--text2)">pairs &gt;0.75 highlighted</span>
      </div>
      <div class="collapsible-body">
        <div id="corr-wrap" style="overflow:auto"><div class="spinner"><div class="spin"></div></div></div>
      </div>
    </div>

    <!-- Drawdown Analytics -->
    <div class="card">
      <div class="card-title collapsible-toggle" onclick="toggleCollapse(this)">Drawdown Analytics</div>
      <div class="collapsible-body" id="dd-analytics"><div class="spinner"><div class="spin"></div></div></div>
    </div>

    <!-- Risk Bubble Map (repeated in risk tab) -->
    <div class="card">
      <div class="card-title collapsible-toggle" onclick="toggleCollapse(this)">Risk Map by Position
        <span style="font-size:10px;font-weight:400;text-transform:none;letter-spacing:0;color:var(--text2)">size = composite risk score</span>
      </div>
      <div class="collapsible-body">
        <div id="risk-bubble-map-2" style="position:relative;width:100%;height:300px"></div>
      </div>
    </div>

    <!-- MCTR Chart -->
    <div class="card" style="grid-column:1/-1">
      <div class="card-title collapsible-toggle" onclick="toggleCollapse(this)">Marginal Contribution to Risk (MCTR)
        <span style="font-size:10px;font-weight:400;text-transform:none;letter-spacing:0;color:var(--text2)">% of portfolio volatility</span>
      </div>
      <div class="collapsible-body">
        <div style="position:relative;height:160px"><canvas id="mctr-chart"></canvas></div>
      </div>
    </div>

    <!-- Monthly Returns Heatmap -->
    <div class="card" style="grid-column:1/-1">
      <div class="card-title" style="display:flex;justify-content:space-between;align-items:center">
        <span>Monthly Returns</span>
        <span style="display:flex;align-items:center;gap:10px;font-size:11px;font-weight:400;text-transform:none;letter-spacing:0">
          <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#15803d;margin-right:3px;vertical-align:middle"></span>Port outperforms benchmark</span>
          <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#b91c1c;margin-right:3px;vertical-align:middle"></span>Underperforms</span>
        </span>
      </div>
      <div id="heatmap-wrap"></div>
    </div>

  </div>
  </div><!-- /tab-risk -->

  <!-- ── SCENARIOS TAB ──────────────────────────────────────────────────────── -->
  <div id="tab-scenarios" class="tab-panel" style="grid-column:1/-1">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">

    <!-- Monte Carlo -->
    <div class="card" style="grid-column:1/-1">
      <div class="card-title collapsible-toggle" onclick="toggleCollapse(this)">Monte Carlo Simulation — 500 Paths × 252 Days
        <span style="font-size:10px;font-weight:400;text-transform:none;letter-spacing:0;color:var(--text2)" id="mc-params"></span>
      </div>
      <div class="collapsible-body">
        <div style="position:relative;height:280px"><canvas id="mc-chart"></canvas></div>
      </div>
    </div>

    <!-- Stress Testing -->
    <div class="card" style="grid-column:1/-1">
      <div class="card-title collapsible-toggle" onclick="toggleCollapse(this)">Historical Stress Tests
        <span style="font-size:10px;font-weight:400;text-transform:none;letter-spacing:0;color:var(--text2)">CAPM-estimated portfolio impact</span>
      </div>
      <div class="collapsible-body">
        <div class="tbl-wrap">
          <table class="stress-tbl">
            <thead><tr><th>Scenario</th><th>Market Return</th><th>Est. Portfolio Return</th><th>Est. P&L</th></tr></thead>
            <tbody id="stress-body"><tr><td colspan="4"><div class="spinner"><div class="spin"></div></div></td></tr></tbody>
          </table>
        </div>
      </div>
    </div>

  </div>
  </div><!-- /tab-scenarios -->

</div>

<script>
// ── Tab switching ──────────────────────────────────────────────────────────────
function switchTab(name, btn){
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
  if(name==='risk'    && !_riskLoaded)  { _riskLoaded=true;  loadRisk(); }
  if(name==='scenarios' && !_scLoaded) { _scLoaded=true;    loadScenarios(); }
}
let _riskLoaded=false, _scLoaded=false;

// ── Collapsible panels ─────────────────────────────────────────────────────────
function toggleCollapse(hdr){
  hdr.classList.toggle('collapsed');
  const body=hdr.nextElementSibling;
  body.style.maxHeight=hdr.classList.contains('collapsed')?'0':'none';
}

// ── Helpers ────────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
function fmt(n,d=2){return n==null?'—':n.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d})}
function fmtUSD(n){if(n==null)return'—';const neg=n<0;return(neg?'-':'')+'$'+Math.abs(n).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}
function fmtB(n){if(n==null||n===0)return'—';if(Math.abs(n)>=1e9)return'$'+(n/1e9).toFixed(2)+'B';if(Math.abs(n)>=1e6)return'$'+(n/1e6).toFixed(1)+'M';return'$'+Math.round(n).toLocaleString()}
function fmtPct(n,plus=true){if(n==null)return'—';const s=n>0&&plus?'+':'';return`${s}${fmt(n)}%`}
function cc(n){return n>=0?'green':'red'}

// Clock
function clock(){$('clock').textContent=new Date().toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit',second:'2-digit'})}
setInterval(clock,1000); clock();

// ── Chart state ────────────────────────────────────────────────────────────────
let equityChart=null, rollingChart=null, sectorChart=null, attrChart=null;
let chartData=null, rollingData=null;
let chartMode='equity', rollingMode='sharpe';

const CHART_OPTS = {
  responsive:true, maintainAspectRatio:false,
  interaction:{mode:'index',intersect:false},
  plugins:{legend:{display:false},tooltip:{backgroundColor:'#1e293b',borderColor:'#334155',borderWidth:1}},
  scales:{
    x:{grid:{color:'rgba(31,42,61,0.5)'},ticks:{color:'#6b7280',maxTicksLimit:8,font:{size:10},callback:(v,i,a)=>chartData?.dates[i]?.slice(5)||''}},
    y:{position:'right',grid:{color:'rgba(31,42,61,0.5)'},ticks:{color:'#6b7280',font:{size:10}}}
  }
};

function setChartMode(mode, btn){
  chartMode=mode;
  document.querySelectorAll('.chart-tab').forEach(b=>{if(b.parentElement===btn.parentElement)b.classList.remove('active')});
  btn.classList.add('active');
  if(chartData) buildEquityChart(chartData);
}

function buildEquityChart(d){
  chartData=d;
  const ctx=$('equity-chart').getContext('2d');
  if(equityChart) equityChart.destroy();
  const grad=ctx.createLinearGradient(0,0,0,210);
  if(chartMode==='equity'){
    grad.addColorStop(0,'rgba(232,160,32,0.22)'); grad.addColorStop(1,'rgba(232,160,32,0)');
    equityChart=new Chart(ctx,{type:'line',data:{labels:d.dates,datasets:[
      {label:'Portfolio',data:d.portfolio,borderColor:'#e8a020',borderWidth:2,backgroundColor:grad,fill:true,pointRadius:0,tension:.3},
      {label:'Benchmark',data:d.benchmark,borderColor:'#374151',borderWidth:1.5,backgroundColor:'transparent',fill:false,pointRadius:0,tension:.3}
    ]},options:{...CHART_OPTS,scales:{...CHART_OPTS.scales,y:{...CHART_OPTS.scales.y,ticks:{...CHART_OPTS.scales.y.ticks,callback:v=>fmt(v)}}}}});
  } else {
    grad.addColorStop(0,'rgba(239,68,68,0.2)'); grad.addColorStop(1,'rgba(239,68,68,0)');
    equityChart=new Chart(ctx,{type:'line',data:{labels:d.dates,datasets:[
      {label:'Drawdown',data:d.drawdown,borderColor:'#ef4444',borderWidth:1.5,backgroundColor:grad,fill:true,pointRadius:0,tension:.2}
    ]},options:{...CHART_OPTS,scales:{...CHART_OPTS.scales,y:{...CHART_OPTS.scales.y,ticks:{...CHART_OPTS.scales.y.ticks,callback:v=>fmtPct(v)}}}}});
  }
}

function setRollingMode(mode, btn){
  rollingMode=mode;
  document.querySelectorAll('.chart-tab').forEach(b=>{if(b.parentElement===btn.parentElement)b.classList.remove('active')});
  btn.classList.add('active');
  if(rollingData) buildRollingChart(rollingData);
}

function buildRollingChart(d){
  rollingData=d;
  const ctx=$('rolling-chart').getContext('2d');
  if(rollingChart) rollingChart.destroy();
  const vals=rollingMode==='sharpe'?d.sharpe:d.beta;
  const col=rollingMode==='sharpe'?'#3b82f6':'#e8a020';
  rollingChart=new Chart(ctx,{type:'line',data:{labels:d.dates,datasets:[
    {label:rollingMode==='sharpe'?'Rolling Sharpe':'Rolling Beta',data:vals,borderColor:col,borderWidth:1.5,backgroundColor:'transparent',fill:false,pointRadius:0,tension:.3,spanGaps:true}
  ]},options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},
    plugins:{legend:{display:false},tooltip:{backgroundColor:'#1e293b',borderColor:'#334155',borderWidth:1,callbacks:{label:c=>` ${c.dataset.label}: ${fmt(c.raw,3)}`}}},
    scales:{
      x:{grid:{color:'rgba(31,42,61,0.5)'},ticks:{color:'#6b7280',maxTicksLimit:6,font:{size:10},callback:(_,i)=>d.dates[i]?.slice(5)||''}},
      y:{position:'right',grid:{color:'rgba(31,42,61,0.5)'},ticks:{color:'#6b7280',font:{size:10},callback:v=>fmt(v,2)}}
    }
  }});
}

function buildSectorChart(sectors){
  const ctx=$('sector-chart').getContext('2d');
  if(sectorChart) sectorChart.destroy();
  sectorChart=new Chart(ctx,{type:'pie',data:{
    labels:sectors.map(s=>s.sector),
    datasets:[{data:sectors.map(s=>s.pct),backgroundColor:sectors.map(s=>s.color),borderColor:'#111827',borderWidth:2,hoverOffset:6}]
  },options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{display:false},tooltip:{backgroundColor:'#1e293b',borderColor:'#334155',borderWidth:1,callbacks:{label:c=>` ${c.label}: ${fmt(c.raw,1)}%`}}}
  }});
  $('sector-legend').innerHTML=sectors.map(s=>
    `<div class="sector-row"><div class="sector-name"><div class="sector-dot" style="background:${s.color}"></div>${s.sector}</div><span class="muted">${fmt(s.pct,1)}%</span></div>`
  ).join('');
}

function buildAttrChart(attribution){
  const ctx=$('attr-chart').getContext('2d');
  if(attrChart) attrChart.destroy();
  const top=attribution.slice(0,10);
  attrChart=new Chart(ctx,{type:'bar',data:{
    labels:top.map(h=>h.ticker),
    datasets:[{data:top.map(h=>h.contribution),backgroundColor:top.map(h=>h.contribution>=0?'rgba(34,197,94,.7)':'rgba(239,68,68,.7)'),borderRadius:3}]
  },options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{display:false},tooltip:{backgroundColor:'#1e293b',callbacks:{label:c=>` ${fmtPct(c.raw)} contribution`}}},
    scales:{
      x:{grid:{display:false},ticks:{color:'#6b7280',font:{size:10}}},
      y:{position:'right',grid:{color:'rgba(31,42,61,0.5)'},ticks:{color:'#6b7280',font:{size:10},callback:v=>fmtPct(v)}}
    }
  }});
}

// ── Risk bubble map ────────────────────────────────────────────────────────────
let _rbmStocks = [];

function renderRiskBubbleMap(stockRisks) {
  _rbmStocks = stockRisks;
  const wrap = $('risk-bubble-map');
  if (!wrap || !stockRisks.length) return;

  const W = wrap.clientWidth || 340;
  const H = wrap.clientHeight || 320;
  const DPR = window.devicePixelRatio || 1;

  const maxScore = Math.max(...stockRisks.map(s => s.risk_score), 1);
  const MIN_R = 16, MAX_R = Math.min(56, W / 7);

  // Build circle list sorted largest-first
  const circles = stockRisks.map(s => {
    const norm = s.risk_score / maxScore;
    const r = MIN_R + Math.pow(norm, 0.65) * (MAX_R - MIN_R);
    const col = norm > 0.62 ? '#ef4444' : norm > 0.36 ? '#eab308' : '#22c55e';
    const fillA = norm > 0.62 ? 0.22 : norm > 0.36 ? 0.18 : 0.14;
    return { ...s, r, col, fillA, x: 0, y: 0, norm };
  }).sort((a, b) => b.r - a.r);

  // Greedy circle packing — spiral outward from center
  const placed = [];
  const cx = W / 2, cy = H / 2;

  for (let i = 0; i < circles.length; i++) {
    const c = circles[i];
    if (i === 0) { c.x = cx; c.y = cy; placed.push(c); continue; }
    let ok = false;
    outer: for (let dist = 0; dist < Math.max(W, H); dist += 3) {
      const steps = Math.max(8, Math.round(dist * 0.7));
      for (let step = 0; step < steps; step++) {
        const ang = (step / steps) * Math.PI * 2;
        const tx = cx + dist * Math.cos(ang);
        const ty = cy + dist * Math.sin(ang);
        if (tx - c.r < 6 || tx + c.r > W - 6 || ty - c.r < 6 || ty + c.r > H - 6) continue;
        let clash = false;
        for (const p of placed) {
          const dx = tx - p.x, dy = ty - p.y;
          if (dx*dx + dy*dy < (c.r + p.r + 5) ** 2) { clash = true; break; }
        }
        if (!clash) { c.x = tx; c.y = ty; placed.push(c); ok = true; break outer; }
      }
    }
    if (!ok) { c.x = cx + (i % 6 - 3) * 55; c.y = cy + Math.floor(i / 6) * 55; placed.push(c); }
  }

  // Build canvas
  const canvas = document.createElement('canvas');
  canvas.width = W * DPR; canvas.height = H * DPR;
  canvas.style.cssText = `width:${W}px;height:${H}px;cursor:crosshair`;
  const ctx = canvas.getContext('2d');
  ctx.scale(DPR, DPR);

  function draw() {
    ctx.clearRect(0, 0, W, H);
    for (const c of placed) {
      // Glow
      const grd = ctx.createRadialGradient(c.x, c.y, c.r * 0.3, c.x, c.y, c.r * 1.4);
      grd.addColorStop(0, c.col + '28');
      grd.addColorStop(1, 'transparent');
      ctx.beginPath(); ctx.arc(c.x, c.y, c.r * 1.4, 0, Math.PI*2);
      ctx.fillStyle = grd; ctx.fill();

      // Fill
      ctx.beginPath(); ctx.arc(c.x, c.y, c.r, 0, Math.PI*2);
      const hex = c.col.slice(1);
      const rr=parseInt(hex.slice(0,2),16), gg=parseInt(hex.slice(2,4),16), bb=parseInt(hex.slice(4,6),16);
      ctx.fillStyle = `rgba(${rr},${gg},${bb},${c.fillA})`; ctx.fill();
      ctx.strokeStyle = c.col; ctx.lineWidth = 1.8; ctx.stroke();

      // Ticker label
      const fsize = Math.max(9, Math.min(14, c.r * 0.38));
      ctx.fillStyle = c.col;
      ctx.font = `700 ${fsize}px Inter,sans-serif`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(c.ticker, c.x, c.y - (c.r > 26 ? 5 : 0));

      // Risk score subtitle (only if bubble is big enough)
      if (c.r > 28) {
        ctx.font = `${Math.max(8, fsize - 3)}px Inter,sans-serif`;
        ctx.fillStyle = 'rgba(148,163,184,0.85)';
        ctx.fillText(c.risk_score.toFixed(0), c.x, c.y + fsize - 2);
      }
    }
  }
  draw();

  // Tooltip
  let tip = wrap.querySelector('.rbm-tooltip');
  if (!tip) {
    tip = document.createElement('div');
    tip.className = 'rbm-tooltip';
    wrap.style.position = 'relative';
    wrap.appendChild(tip);
  }

  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let hit = null;
    for (const c of placed) {
      const dx = mx - c.x, dy = my - c.y;
      if (dx*dx + dy*dy < c.r*c.r) { hit = c; break; }
    }
    if (hit) {
      tip.style.display = 'block';
      tip.style.left = (hit.x + hit.r + 6) + 'px';
      tip.style.top  = Math.max(0, hit.y - 40) + 'px';
      tip.innerHTML = `<div style="font-weight:700;color:${hit.col};margin-bottom:4px">${hit.ticker}</div>
        <div style="color:#94a3b8;font-size:10px">Risk Score: <span style="color:#e2e8f0">${hit.risk_score.toFixed(0)}/100</span></div>
        <div style="color:#94a3b8;font-size:10px">Vol: <span style="color:#e2e8f0">${fmt(hit.vol,1)}%</span></div>
        <div style="color:#94a3b8;font-size:10px">Beta: <span style="color:#e2e8f0">${fmt(hit.beta,2)}</span></div>
        <div style="color:#94a3b8;font-size:10px">Max DD: <span style="color:#e2e8f0">${fmt(hit.max_dd,1)}%</span></div>
        <div style="color:#94a3b8;font-size:10px">Weight: <span style="color:#e2e8f0">${fmt(hit.weight,1)}%</span></div>`;
    } else {
      tip.style.display = 'none';
    }
  });
  canvas.addEventListener('mouseleave', () => { tip.style.display = 'none'; });

  wrap.innerHTML = '';
  wrap.style.position = 'relative';
  wrap.appendChild(canvas);
  wrap.appendChild(tip);
}

// ── Render functions ───────────────────────────────────────────────────────────

function renderMetricBar(d){
  $('total-value').textContent=fmtUSD(d.total_value);
  const s=d.day_change>=0?'+':'';
  $('day-change-row').innerHTML=`<span class="${cc(d.day_change)}">${s}${fmtUSD(d.day_change)} (${fmtPct(d.day_change_pct)}) today</span>`;
  $('ann-return').innerHTML=`<span class="${cc(d.port_ann_return)}">${fmtPct(d.port_ann_return)}</span>`;
  $('bench-return').textContent=`vs ${d.benchmark} ${fmtPct(d.bench_ann_return)}`;
  $('bench-name').textContent=d.benchmark;
  $('active-return').innerHTML=`<span class="${cc(d.active_return)}">${fmtPct(d.active_return)}</span>`;
  $('sharpe-val').innerHTML=`<span class="${d.sharpe>=1?'green':'red'}">${fmt(d.sharpe)}</span>`;
  $('sortino-val').textContent=`Sortino ${fmt(d.sortino)}`;
  $('alpha-val').innerHTML=`<span class="${cc(d.alpha)}">${fmtPct(d.alpha)}</span>`;
  $('beta-val').textContent=`Beta ${fmt(d.beta,3)}`;
  $('maxdd-val').innerHTML=`<span class="red">${fmtPct(d.max_drawdown)}</span>`;
  $('calmar-val').textContent=`Calmar ${fmt(d.calmar,2)}`;
  $('provider-badge').textContent=d.provider;
  // Risk score ring
  const sc=d.risk_summary_score||0;
  const circ=138.2;
  const offset=circ*(1-sc/100);
  const col=sc>66?'#ef4444':sc>33?'#eab308':'#22c55e';
  const lbl=sc>66?'High':sc>33?'Moderate':'Low';
  $('risk-ring-arc').style.strokeDashoffset=offset;
  $('risk-ring-arc').style.stroke=col;
  $('risk-score-val').textContent=sc.toFixed(0);
  $('risk-score-val').style.color=col;
  $('risk-score-lbl').textContent=lbl;
}

const RISK_RATIO_INFO = {
  'Sharpe Ratio':   {expl:'Excess return above the risk-free rate divided by total volatility. The most widely used risk-adjusted return metric. Higher is better.',avg:'S&P 500 ≈ 0.5 · Good active fund: >1.0 · Excellent: >2.0 · Elite: >3.0'},
  'Sortino Ratio':  {expl:'Like Sharpe but only penalizes downside volatility — it ignores upside swings. Better suited for asymmetric return profiles.',avg:'Good: >1.0 · Excellent: >2.0 · Typically runs ~1.4× your Sharpe if returns are close to normal'},
  'Calmar Ratio':   {expl:'Annualized return divided by maximum drawdown. Popular with CTAs and hedge funds because it ties reward directly to worst-case loss.',avg:'Good: >0.5 · Excellent: >1.0 · Top-tier funds target: >2.0'},
  'Treynor Ratio':  {expl:'Excess return per unit of systematic (market) risk only. Ignores idiosyncratic risk — most useful for comparing diversified portfolios against the same benchmark.',avg:'S&P 500 ≈ 5% · Good active fund: >8% · Only meaningful vs same-benchmark peers'},
  'Info Ratio':     {expl:'Active return (your return minus benchmark) divided by tracking error. Answers: "How much alpha do I generate per unit of active bet?" The gold standard for active managers.',avg:'Rare to sustain >0.5 · Top-quartile PM: 0.5–1.0 · Top-decile: >1.0 · Elite: >2.0'},
  'Omega Ratio':    {expl:'Probability-weighted gains divided by probability-weighted losses above/below a threshold. Captures the full return distribution — not just variance. Anything above 1 means more wins than losses in expected value.',avg:'Above 1.0 = more gains than losses · Good: >1.5 · Strong: >2.0'},
  'Beta':           {expl:'Sensitivity to market moves. A beta of 1.2 means for every 1% the benchmark moves, your portfolio moves 1.2%. Negative beta means you move opposite to the market.',avg:'Market = 1.0 · Low-vol / defensive: 0.6–0.8 · Aggressive growth: 1.2–1.6'},
  'Alpha (CAPM)':   {expl:'The return not explained by market exposure. Positive alpha means you outperformed what your beta-adjusted risk level would predict. True alpha is rare and hard to sustain net of fees.',avg:'Median active manager ≈ 0% after fees · Top quartile: +1–3%/yr · Fees typically consume 1–2%'},
  'R-Squared':      {expl:'How much of your portfolio\'s movement is explained by the benchmark. High R² = you essentially own the index. Low R² = you\'re running a concentrated or uncorrelated book.',avg:'Index ETF: ~99% · Diversified active fund: 85–95% · Concentrated / alt: 40–80%'},
  'Tracking Error': {expl:'Annualized standard deviation of your daily returns minus the benchmark\'s. Quantifies how big your active bets are. High TE = big deviations from the index, for better or worse.',avg:'Index ETF: <0.5% · Active mutual fund: 3–8% · Hedge fund: 6–15%'},
  'VaR 95% (day)':  {expl:'On the worst 5% of trading days historically, you lost at least this much. Based on your actual past returns (historical simulation). Not a worst-case — losses can exceed this on tail days.',avg:'S&P 500 daily VaR 95% ≈ −1.5% · Conservative fund: −0.8% · Aggressive: −2.5%+'},
  'Max Drawdown':   {expl:'Largest peak-to-trough decline in the measurement period. Key measure of tail risk and the psychological difficulty of staying invested. Recovering from large drawdowns can take years.',avg:'S&P 500 avg annual max DD ≈ −15% · Target for balanced portfolio: <−10% · 2008 GFC: −57%'},
};

function riskMetricRow(label, val, pillCls){
  const info = RISK_RATIO_INFO[label] || {};
  return `<div style="padding:10px 0;border-bottom:1px solid rgba(31,42,61,.4)">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
      <span style="font-size:12px;font-weight:600;color:var(--text)">${label}</span>
      <span class="pill ${pillCls}">${val}</span>
    </div>
    ${info.expl?`<div style="font-size:11px;color:var(--text2);line-height:1.5;margin-bottom:4px">${info.expl}</div>`:''}
    ${info.avg?`<div style="font-size:10px;color:var(--gold);font-weight:600">⌀ ${info.avg}</div>`:''}
  </div>`;
}

function renderRiskMetrics(d){
  $('risk-metrics').innerHTML=`
    <div class="metric-section">Return Ratios</div>
    ${riskMetricRow('Sharpe Ratio',   fmt(d.sharpe,3),   d.sharpe>=2?'green':d.sharpe>=1?'yellow':'red')}
    ${riskMetricRow('Sortino Ratio',  fmt(d.sortino,3),  d.sortino>=2?'green':d.sortino>=1?'yellow':'red')}
    ${riskMetricRow('Calmar Ratio',   fmt(d.calmar,3),   d.calmar>=2?'green':d.calmar>=1?'yellow':'red')}
    ${riskMetricRow('Treynor Ratio',  fmtPct(d.treynor), d.treynor>=5?'green':d.treynor>=2?'yellow':'red')}
    ${riskMetricRow('Info Ratio',     fmt(d.info_ratio,3),d.info_ratio>=2?'green':d.info_ratio>=1?'yellow':'red')}
    ${riskMetricRow('Omega Ratio',    fmt(d.omega,2),    d.omega>=2?'green':d.omega>=1?'yellow':'red')}
    <div class="metric-section" style="margin-top:6px">Market Exposure</div>
    ${riskMetricRow('Beta',           fmt(d.beta,3),     d.beta<1?'green':d.beta<1.3?'yellow':'red')}
    ${riskMetricRow('Alpha (CAPM)',   fmtPct(d.alpha),   d.alpha>=0?'green':'red')}
    ${riskMetricRow('R-Squared',      fmtPct(d.r_squared,1),d.r_squared<40?'green':d.r_squared<70?'yellow':'red')}
    ${riskMetricRow('Tracking Error', fmtPct(d.tracking_error),d.tracking_error<5?'green':d.tracking_error<10?'yellow':'red')}
    <div class="metric-section" style="margin-top:6px">Downside</div>
    ${riskMetricRow('VaR 95% (day)',  fmtPct(d.var_95),  d.var_95>-1.5?'green':d.var_95>-2.5?'yellow':'red')}
    ${riskMetricRow('Max Drawdown',   fmtPct(d.max_drawdown),d.max_drawdown>-10?'green':d.max_drawdown>-20?'yellow':'red')}`;
}

const METRIC_INFO = {
  hhi:   {label:'HHI Concentration',avg:'S&P 500 ≈ 80 | Typical active: 500–1500',expl:'Herfindahl-Hirschman Index — sum of squared weights × 10,000. Below 1,000 = diversified; 1,000–1,800 = moderate; above 1,800 = concentrated. A perfectly equal portfolio of 25 stocks scores 400.'},
  top3:  {label:'Top 3 Concentration',avg:'S&P 500 ≈ 12% | Typical active: 25–45%',expl:'What percent of your portfolio is in the three largest positions. Above 50% is high single-name risk — a bad quarter in one stock significantly hurts performance.'},
  top5:  {label:'Top 5 Concentration',avg:'S&P 500 ≈ 22% | Well-diversified: <50%',expl:'Same idea extended to five positions. Institutional guidelines often cap top-5 at 60–70% to avoid over-reliance on a handful of names.'},
  te:    {label:'Tracking Error (Ann.)',avg:'Index fund <1% | Active fund: 4–8% | Hedge fund: 6–15%',expl:'Annualized standard deviation of your daily returns minus the benchmark\'s. Measures how far you deviate from the index. High TE means you are taking big active bets — which can mean big alpha or big pain.'},
  ir:    {label:'Information Ratio',avg:'Good: >0.5 | Excellent: >1.0 | Elite: >2.0',expl:'Active return divided by tracking error. Answers: "How much extra return do I earn per unit of active risk?" Above 0.5 is competitive with professional fund managers.'},
  var:   {label:'Portfolio VaR 95% (daily)',avg:'S&P 500 ≈ −1.5% | Typical portfolio: −1.0% to −2.5%',expl:'On the worst 5% of trading days historically, you lost at least this much. Not a worst-case — it is the threshold. Losses can exceed this on tail days (see Sortino and Ulcer Index for those).'},
  large: {label:'Largest Single Position',avg:'Guideline: <10% per holding',expl:'Single-name concentration is the most common source of avoidable risk. Many institutional managers hard-cap any position at 5–10% of AUM. Above 15% is considered high-conviction / high-risk.'},
  active_vol: {label:'Active Volatility',avg:'Same as Tracking Error — see above',expl:'The standard deviation of daily outperformance vs the benchmark, annualized. Identical to tracking error in this calculation.'},
};

function metricRow(key, val, pillCls){
  const info = METRIC_INFO[key] || {};
  return `<div style="padding:10px 0;border-bottom:1px solid rgba(31,42,61,.4)">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
      <span style="font-size:12px;font-weight:600;color:var(--text)">${info.label||key}</span>
      <span class="pill ${pillCls}">${val}</span>
    </div>
    <div style="font-size:11px;color:var(--text2);line-height:1.5;margin-bottom:4px">${info.expl||''}</div>
    <div style="font-size:10px;color:var(--gold);font-weight:600">⌀ ${info.avg||'—'}</div>
  </div>`;
}

function renderActiveRisks(ar, sectors){
  const hhi=ar.hhi;
  const hhiCls=hhi<1000?'green':hhi<1800?'yellow':'red';
  const top3Cls=ar.concentration_top3>50?'red':ar.concentration_top3>35?'yellow':'green';
  const top5Cls=ar.concentration_top5>65?'red':ar.concentration_top5>50?'yellow':'green';
  const teCls=ar.tracking_error<5?'green':ar.tracking_error<10?'yellow':'red';
  const irCls=ar.info_ratio>=1?'green':ar.info_ratio>=0?'yellow':'red';
  const varCls=ar.var_95_port>-1.5?'green':ar.var_95_port>-2.5?'yellow':'red';
  const lgCls=ar.largest_weight<10?'green':ar.largest_weight<15?'yellow':'red';

  let sectorHtml='';
  if(sectors&&sectors.length){
    sectorHtml=`<div class="metric-section" style="margin-top:10px">Sector Concentration</div>`;
    const top=sectors.slice(0,6);
    sectorHtml+=top.map(s=>{
      const w=s.pct;
      const barCls=w>30?'#ef4444':w>20?'#eab308':'#22c55e';
      return `<div style="margin-bottom:8px">
        <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px">
          <span style="display:flex;align-items:center;gap:6px">
            <span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${s.color}"></span>${s.sector}
          </span>
          <span class="${w>30?'red':w>20?'gold':'muted'}" style="font-weight:600">${fmt(w,1)}%</span>
        </div>
        <div class="sector-conc-bar"><div class="sector-conc-fill" style="width:${Math.min(w,100)}%;background:${barCls}"></div></div>
      </div>`;
    }).join('');
    if(sectors.length>1){
      const vals=sectors.map(s=>s.pct/100);
      const sectorHHI=Math.round(vals.reduce((a,v)=>a+v*v,0)*10000);
      const sHhiCls=sectorHHI<1500?'green':sectorHHI<3000?'yellow':'red';
      sectorHtml+=`<div style="padding:10px 0;border-bottom:1px solid rgba(31,42,61,.4)">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
          <span style="font-size:12px;font-weight:600;color:var(--text)">Sector HHI</span>
          <span class="pill ${sHhiCls}">${sectorHHI}</span>
        </div>
        <div style="font-size:11px;color:var(--text2);line-height:1.5;margin-bottom:4px">Sector-level concentration index. Perfectly equal across 11 GICS sectors = 909. A heavy tilt to one sector pushes this above 2,500.</div>
        <div style="font-size:10px;color:var(--gold);font-weight:600">⌀ S&P 500 ≈ 1,800 (Tech-heavy)</div>
      </div>`;
    }
  }

  $('active-risks').innerHTML=`
    <div class="active-risk-grid">
      <div class="ar-card">
        <div class="ar-label">Positions</div>
        <div class="ar-val">${ar.n_positions}</div>
        <div style="font-size:10px;color:var(--text2)">holdings</div>
      </div>
      <div class="ar-card">
        <div class="ar-label">Largest</div>
        <div class="ar-val ${lgCls}">${ar.largest_ticker}</div>
        <div style="font-size:10px;color:var(--text2)">${fmt(ar.largest_weight,1)}% of AUM</div>
      </div>
      <div class="ar-card">
        <div class="ar-label">Top 3</div>
        <div class="ar-val ${top3Cls}">${fmt(ar.concentration_top3,1)}%</div>
        <div style="font-size:10px;color:var(--text2)">of portfolio</div>
      </div>
      <div class="ar-card">
        <div class="ar-label">Top 5</div>
        <div class="ar-val ${top5Cls}">${fmt(ar.concentration_top5,1)}%</div>
        <div style="font-size:10px;color:var(--text2)">of portfolio</div>
      </div>
    </div>
    <div style="margin-top:10px">
      <div class="metric-section">Concentration</div>
      ${metricRow('hhi',  `${Math.round(ar.hhi)}`, hhiCls)}
      ${metricRow('top3', `${fmt(ar.concentration_top3,1)}%`, top3Cls)}
      ${metricRow('top5', `${fmt(ar.concentration_top5,1)}%`, top5Cls)}
      ${metricRow('large',`${ar.largest_ticker} · ${fmt(ar.largest_weight,1)}%`, lgCls)}
      <div class="metric-section" style="margin-top:10px">Active Risk</div>
      ${metricRow('te',  `${fmt(ar.tracking_error,1)}%`, teCls)}
      ${metricRow('ir',  `${fmt(ar.info_ratio,2)}`, irCls)}
      ${metricRow('var', `${fmt(ar.var_95_port,2)}%`, varCls)}
    </div>
    ${sectorHtml}`;
}

function renderCapture(d){
  const uc=$('up-cap'), dc=$('dn-cap');
  uc.textContent=fmt(d.upside_capture,1)+'%';
  uc.className='capture-val '+(d.upside_capture>100?'green':d.upside_capture>85?'gold':'red');
  dc.textContent=fmt(d.downside_capture,1)+'%';
  dc.className='capture-val '+(d.downside_capture<100?'green':d.downside_capture<115?'yellow':'red');
  $('bat-avg').innerHTML=`<span class="${d.batting_avg>=50?'green':'red'}">${fmt(d.batting_avg,1)}%</span>`;
  $('r-sq').innerHTML=`<span>${fmt(d.r_squared,1)}%</span>`;
}

function renderHoldings(attribution){
  const maxC = Math.max(...attribution.map(h=>Math.abs(h.contribution)), 1);
  const allThesis = loadThesis();
  let rows = '', tv = 0, tg = 0;

  for(const h of attribution){
    tv += h.market_value; tg += h.gain_loss;
    const bw = Math.round(Math.abs(h.contribution)/maxC*100);
    const bc = h.contribution >= 0 ? '' : 'neg';

    // Thesis dot
    const thesis   = allThesis[h.ticker] || {};
    const tStatus  = thesis.status || '';
    const dotColor = tStatus==='Intact'?'var(--green)':tStatus==='Weakening'?'var(--gold)':tStatus==='Broken'?'var(--red)':'#374151';
    const dotTitle = tStatus ? `${tStatus} — click to ${tStatus===''?'add':'view'} thesis` : 'Click to add thesis';
    const rowClass = tStatus === 'Broken' ? 'thesis-broken-row' : '';

    // 52W range bar
    const hi = h.high_52w, lo = h.low_52w, cur = h.price;
    let rangeCell = `<td class="muted" style="text-align:center;font-size:11px">—</td>`;
    if(hi != null && lo != null && hi > lo){
      const pct  = Math.max(0, Math.min(100, (cur - lo)/(hi - lo)*100));
      const dotC = pct >= 90 ? 'var(--red)' : pct <= 10 ? 'var(--green)' : '#6b7280';
      rangeCell  = `<td title="52W Low: $${fmt(lo,2)} · Current: $${fmt(cur,2)} · 52W High: $${fmt(hi,2)}">
        <div class="range-bar-wrap">
          <div class="range-bar-track"><div class="range-bar-marker" style="left:${pct.toFixed(1)}%;background:${dotC}"></div></div>
          <div class="range-bar-labels"><span>$${fmt(lo,0)}</span><span class="range-pct-lbl">${Math.round(pct)}%</span><span>$${fmt(hi,0)}</span></div>
        </div></td>`;
    }

    rows += `<tr id="holding-row-${h.ticker}" class="${rowClass}">
      <td><div class="ticker-cell"><div class="ticker-dot" style="background:${h.gain_loss>=0?'var(--green)':'var(--red)'}"></div>${h.ticker}</div></td>
      <td>${fmt(h.shares,0)}</td><td>${fmtUSD(h.avg_cost)}</td><td>${fmtUSD(h.price)}</td>
      <td>${fmtUSD(h.market_value)}</td>
      <td class="${cc(h.gain_loss)}">${h.gain_loss>=0?'+':''}${fmtUSD(h.gain_loss)} <span class="muted">(${fmtPct(h.gain_loss_pct)})</span></td>
      <td class="${cc(h.day_change_pct)}">${fmtPct(h.day_change_pct)}</td>
      <td>${fmt(h.weight,1)}%</td>
      <td class="${cc(h.period_return)}">${fmtPct(h.period_return)}</td>
      <td><div class="bar-wrap"><span class="${cc(h.contribution)}">${fmtPct(h.contribution)}</span><div class="bar-bg"><div class="bar-fill ${bc}" style="width:${bw}%"></div></div></div></td>
      <td class="muted">${h.mctr!=null?fmt(h.mctr,2)+'%':'—'}</td>
      <td style="text-align:center"><span class="thesis-dot" style="background:${dotColor}" onclick="toggleThesisCard('${h.ticker}')" title="${dotTitle}"></span></td>
      ${rangeCell}
    </tr>
    ${thesisCardHtml(h.ticker, thesis)}`;
  }

  $('holdings-body').innerHTML = rows;
  $('holdings-foot').innerHTML = `<tr><td colspan="4">TOTAL</td><td>${fmtUSD(tv)}</td><td class="${cc(tg)}">${tg>=0?'+':''}${fmtUSD(tg)}</td><td colspan="7"></td></tr>`;
}


function renderReturnersLosers(returners, losers){
  function rlHtml(list){
    return list.map(h=>`<div class="rl-row">
      <div><div class="rl-ticker">${h.ticker}</div><div class="rl-sub">${fmtUSD(h.price)} · ${fmt(h.weight,1)}% wt</div></div>
      <div style="text-align:right">
        <div class="rl-ret ${cc(h.period_return)}">${fmtPct(h.period_return)}</div>
        <div class="rl-sub ${cc(h.day_change_pct)}">${fmtPct(h.day_change_pct)} today</div>
      </div>
    </div>`).join('');
  }
  $('returners-list').innerHTML=rlHtml(returners);
  $('losers-list').innerHTML=rlHtml(losers);
}

function renderBenchmarks(rows){
  $('bm-body').innerHTML=rows.map(r=>`<tr class="${r.active?'bm-active-row':''}">
    <td><div class="ticker-cell"><div class="ticker-dot" style="background:${r.active?'var(--gold)':'var(--muted)'}"></div>
      <div><div style="font-weight:700">${r.symbol}</div><div class="muted" style="font-size:10px">${r.name}</div></div></div></td>
    <td class="${cc(r.bm_1m)}">${fmtPct(r.bm_1m)}</td>
    <td class="${cc(r.bm_3m)}">${fmtPct(r.bm_3m)}</td>
    <td class="${cc(r.bm_ytd)}">${fmtPct(r.bm_ytd)}</td>
    <td class="${cc(r.bm_1y)}">${fmtPct(r.bm_1y)}</td>
    <td class="muted">${fmt(r.bm_vol,1)}%</td>
    <td class="${cc(r.port_1m)}">${fmtPct(r.port_1m)}</td>
    <td class="${cc(r.port_1y)}">${fmtPct(r.port_1y)}</td>
    <td class="${r.beta_vs<1?'green':r.beta_vs<1.3?'gold':'red'}">${fmt(r.beta_vs,3)}</td>
    <td class="${cc(r.alpha_vs)}">${fmtPct(r.alpha_vs)}</td>
  </tr>`).join('');
}

const MONTH_ABBR=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
function renderHeatmap(monthly){
  if(!monthly||!monthly.length){$('heatmap-wrap').innerHTML='<div class="muted" style="padding:8px;font-size:12px">No monthly data available.</div>';return;}
  const byYear={};
  for(const m of monthly){
    if(!byYear[m.year]) byYear[m.year]=Array(13).fill(null);
    byYear[m.year][m.month]=m;
  }
  const years=Object.keys(byYear).sort();
  const COL=`36px repeat(12,1fr) 54px`;
  let html=`<div style="display:grid;grid-template-columns:${COL};gap:2px;margin-bottom:6px;font-size:10px;color:var(--text2);font-weight:600">
    <div></div>`;
  for(let i=1;i<=12;i++) html+=`<div style="text-align:center">${MONTH_ABBR[i-1]}</div>`;
  html+=`<div style="text-align:right">YTD</div></div>`;
  for(const y of years){
    let ytd=0, ytdCount=0;
    html+=`<div style="display:grid;grid-template-columns:${COL};gap:2px;margin-bottom:2px;align-items:center">`;
    html+=`<div style="font-size:11px;color:var(--text2);font-weight:700">${y}</div>`;
    for(let i=1;i<=12;i++){
      const m=byYear[y][i];
      if(!m){html+=`<div style="height:32px;border-radius:4px;background:rgba(255,255,255,.04)"></div>`;continue;}
      const v=m.port;
      ytd+=v; ytdCount++;
      const intensity=Math.min(Math.abs(v)/10,1);
      const bg=v>=0?`rgba(34,197,94,${0.15+intensity*0.7})`:`rgba(239,68,68,${0.15+intensity*0.7})`;
      const border=m.port>m.bench?'1px solid rgba(34,197,94,0.5)':'1px solid rgba(255,255,255,.04)';
      const tc=v>=0?'#86efac':'#fca5a5';
      html+=`<div style="height:32px;border-radius:4px;background:${bg};border:${border};display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;color:${tc};cursor:default;overflow:hidden" title="${MONTH_ABBR[i-1]} ${y} — Port: ${fmtPct(v)} | Bmk: ${fmtPct(m.bench)}">${v>=0?'+':''}${fmt(v,1)}</div>`;
    }
    if(ytdCount>0){
      const ytdColor=ytd>=0?'#86efac':'#fca5a5';
      html+=`<div style="text-align:right;font-size:10px;font-weight:700;color:${ytdColor}">${ytd>=0?'+':''}${fmt(ytd,1)}%</div>`;
    } else {
      html+=`<div></div>`;
    }
    html+='</div>';
  }
  $('heatmap-wrap').innerHTML=html;
}

function renderWatchlist(items){
  if(!items||!items.length){
    $('watchlist-feed').innerHTML='<div class="muted" style="padding:8px 0;font-size:12px">No watchlist items. Add tickers to ~/.portfolio_tracker/watchlist.json</div>';
    return;
  }
  $('watchlist-feed').innerHTML=items.map(w=>`<div class="wl-row">
    <div class="wl-ticker">${w.ticker}</div>
    <div class="wl-cols">
      <div style="text-align:right">
        <div class="wl-price">${fmtUSD(w.price)}</div>
        <div class="wl-chg ${cc(w.day_chg)}">${fmtPct(w.day_chg)} today</div>
      </div>
      <div style="text-align:right;min-width:54px">
        <div class="wl-chg ${cc(w.mo_chg)}" style="font-size:10px;color:var(--text2)">1M</div>
        <div class="wl-chg ${cc(w.mo_chg)}">${fmtPct(w.mo_chg)}</div>
      </div>
    </div>
  </div>`).join('');
}

function renderNews(articles){
  if(!articles.length){$('news-feed').innerHTML='<div class="muted" style="padding:8px 0;font-size:12px">No recent news.</div>';return}
  $('news-feed').innerHTML=articles.map(a=>`<div class="news-item">
    <div class="news-title"><a href="${a.link}" target="_blank">${a.title}</a></div>
    <div class="news-meta"><span class="news-ticker">${a.ticker}</span><span>${a.publisher}</span><span>${a.age}</span></div>
  </div>`).join('');
}

// ── Earnings Calendar ──────────────────────────────────────────────────────────

const EARN_NOTES_KEY = 'pt_earn_notes';
let _earnModalTicker = null;
let _cachedEarnings  = [];

function loadEarnNotes(){ try{ return JSON.parse(localStorage.getItem(EARN_NOTES_KEY)||'{}'); }catch(e){ return {}; } }
function saveEarnNotes(d){ localStorage.setItem(EARN_NOTES_KEY, JSON.stringify(d)); }

function openEarnModal(ticker){
  _earnModalTicker = ticker;
  const n = loadEarnNotes()[ticker] || {};
  $('em-ticker').textContent = ticker;
  $('em-pre').value  = n.pre  || '';
  $('em-post').value = n.post || '';
  $('earn-modal').style.display = 'flex';
}
function closeEarnModal(){ $('earn-modal').style.display='none'; _earnModalTicker=null; }
function saveEarnModal(){
  if(!_earnModalTicker) return;
  const all = loadEarnNotes();
  all[_earnModalTicker] = { pre: $('em-pre').value.trim(), post: $('em-post').value.trim() };
  saveEarnNotes(all);
  closeEarnModal();
  renderEarningsTable(_cachedEarnings);
}

function toggleEarnRow(ticker){
  const det = document.getElementById('earn-det-'+ticker);
  if(!det) return;
  // Close others
  document.querySelectorAll('[id^="earn-det-"]').forEach(r=>r.classList.remove('open'));
  if(!det.classList.contains('open')) det.classList.add('open');
}

function fmtRev(n){
  if(n==null) return '—';
  if(Math.abs(n)>=1e9) return '$'+(n/1e9).toFixed(2)+'B';
  if(Math.abs(n)>=1e6) return '$'+(n/1e6).toFixed(1)+'M';
  return '$'+Math.round(n).toLocaleString();
}

function renderEarningsTable(rows){
  _cachedEarnings = rows;
  const wrap = $('earnings-wrap');
  if(!wrap) return;
  if(!rows.length){ wrap.innerHTML='<div class="muted" style="padding:10px;font-size:12px">No upcoming earnings data available for current holdings.</div>'; return; }
  const notes = loadEarnNotes();
  let html = `<div class="tbl-wrap"><table class="earn-table">
    <thead><tr>
      <th>Ticker</th><th>Date</th><th>Countdown</th>
      <th>EPS Est</th><th>Rev Est</th>
      <th>Last EPS</th><th>Last Rev</th>
      <th>Streak</th><th>Notes</th>
    </tr></thead><tbody>`;
  for(const r of rows){
    const du = r.days_until;
    const cdClass = du<=3?'earn-urgent':du<=7?'earn-soon':'earn-future';
    const cdLabel = du===0?'Today':du===1?'Tomorrow':`${du}d`;
    const streakHtml = r.streak_dir
      ? `<span class="${r.streak_dir==='beat'?'earn-beat':'earn-miss'}">${r.streak_dir==='beat'?'▲':'▼'} ${r.streak_count}Q ${r.streak_dir}</span>`
      : '<span class="muted">—</span>';
    const n = notes[r.ticker] || {};
    const hasNotes = n.pre || n.post;
    const dateStr = r.next_date ? r.next_date.slice(0,10) : '—';
    html += `<tr class="earn-row" onclick="toggleEarnRow('${r.ticker}')" id="earn-row-${r.ticker}">
      <td>${r.ticker}</td>
      <td>${dateStr}</td>
      <td><span class="earn-countdown ${cdClass}">${cdLabel}</span></td>
      <td>${r.eps_estimate!=null?'$'+fmt(r.eps_estimate,2):'—'}</td>
      <td>${fmtRev(r.revenue_estimate)}</td>
      <td class="${r.last_eps!=null?(r.last_eps>=0?'green':'red'):''}">${r.last_eps!=null?'$'+fmt(r.last_eps,2):'—'}</td>
      <td>${fmtRev(r.last_revenue)}</td>
      <td>${streakHtml}</td>
      <td style="color:${hasNotes?'var(--gold)':'var(--muted)'};font-size:10px">${hasNotes?'📝 Edit':'+ Add'}</td>
    </tr>
    <tr class="earn-detail-row" id="earn-det-${r.ticker}">
      <td colspan="9" class="earn-detail-cell">
        <div class="earn-detail-grid">
          <div>
            <span class="earn-notes-label">Pre-Earnings Notes</span>
            ${n.pre?`<div class="earn-notes-text">${n.pre}</div>`:'<div class="earn-notes-empty">No pre-earnings notes yet.</div>'}
          </div>
          <div>
            <span class="earn-notes-label">Post-Earnings Action Plan</span>
            ${n.post?`<div class="earn-notes-text">${n.post}</div>`:'<div class="earn-notes-empty">No action plan set yet.</div>'}
          </div>
        </div>
        <div style="margin-top:10px">
          <button class="btn btn-gold" style="padding:4px 14px;font-size:11px" onclick="event.stopPropagation();openEarnModal('${r.ticker}')">Edit Notes</button>
          <button class="btn btn-outline" style="padding:4px 14px;font-size:11px;margin-left:6px" onclick="event.stopPropagation();toggleEarnRow('${r.ticker}')">Close</button>
        </div>
      </td>
    </tr>`;
  }
  html += '</tbody></table></div>';
  wrap.innerHTML = html;
}

function renderEarningsAlerts(rows){
  const panel = $('earnings-alerts');
  if(!panel) return;
  const urgent = rows.filter(r=>r.days_until!=null && r.days_until<=7);
  if(!urgent.length){ panel.style.display='none'; return; }
  panel.style.display = 'block';
  $('ea-count').textContent = urgent.length;
  $('earnings-alerts-body').innerHTML = urgent.map(r=>{
    const du = r.days_until;
    const label = du===0?'TODAY':du===1?'TOMORROW':`${du} DAYS`;
    const col   = du<=3?'var(--red)':'var(--gold)';
    return `<div class="alert-row">
      <span class="alert-tag" style="background:rgba(59,130,246,.12);color:var(--blue)">${r.ticker}</span>
      <span style="font-size:12px">Reports <strong style="color:${col}">${label}</strong> · ${r.next_date?.slice(0,10)||''}</span>
      ${r.eps_estimate!=null?`<span style="font-size:11px;color:var(--text2)">EPS est: <span style="color:var(--text)">${'$'+fmt(r.eps_estimate,2)}</span></span>`:''}
      ${r.revenue_estimate!=null?`<span style="font-size:11px;color:var(--text2)">Rev est: <span style="color:var(--text)">${fmtRev(r.revenue_estimate)}</span></span>`:''}
      <button class="btn btn-outline" style="margin-left:auto;padding:3px 10px;font-size:11px" onclick="openEarnModal('${r.ticker}')">Add notes</button>
    </div>`;
  }).join('');
}

async function loadEarnings(){
  try{
    const rows = await (await fetch('/api/earnings')).json();
    renderEarningsTable(rows);
    renderEarningsAlerts(rows);
  }catch(e){
    const w=$('earnings-wrap');
    if(w) w.innerHTML=`<div class="muted" style="padding:10px;font-size:12px">Could not load earnings data.</div>`;
  }
}

// ── Thesis Tracker ─────────────────────────────────────────────────────────────

const THESIS_KEY = 'pt_thesis';
const W52D_KEY   = 'pt_52w_dismissed';

function loadThesis(){
  try{ return JSON.parse(localStorage.getItem(THESIS_KEY)||'{}'); }catch(e){ return {}; }
}
function saveThesis(data){ localStorage.setItem(THESIS_KEY, JSON.stringify(data)); }

let _thesisModalTicker = null;

function toggleThesisCard(ticker){
  const row = document.getElementById('thesis-row-'+ticker);
  if(!row) return;
  const visible = row.style.display !== 'none';
  // Close any open thesis cards
  document.querySelectorAll('[id^="thesis-row-"]').forEach(r=>{ r.style.display='none'; });
  if(!visible){
    row.style.display = 'table-row';
    setTimeout(()=>row.scrollIntoView({behavior:'smooth',block:'nearest'}), 50);
  }
}

function thesisCardHtml(ticker, thesis){
  const t = thesis || {};
  const status = t.status || '';
  const statusColor = status==='Intact'?'var(--green)':status==='Weakening'?'var(--gold)':status==='Broken'?'var(--red)':'var(--text2)';
  const conviction = t.conviction || '';
  const convColor  = conviction==='High'?'var(--green)':conviction==='Medium'?'var(--gold)':conviction==='Low'?'var(--red)':'var(--text2)';
  const catalystDate = t.catalyst_date || '';
  const catalystPassed = catalystDate && new Date(catalystDate) < new Date();
  const notes = t.notes || [];
  const notesHtml = notes.length
    ? notes.slice().reverse().map(n=>`<div class="thesis-note-item"><div class="thesis-note-ts">${n.ts}</div><div>${n.text}</div></div>`).join('')
    : '<div class="muted" style="font-size:11px;padding:4px 0">No updates logged yet.</div>';
  return `<tr id="thesis-row-${ticker}" class="thesis-card-row" style="display:none">
    <td colspan="13">
      <div class="thesis-card">
        <div class="thesis-grid">
          <div>
            <div class="thesis-field">
              <span class="thesis-field-label">Buy Thesis</span>
              <p>${t.thesis||'<span class="muted">Not set — click Update Thesis to add.</span>'}</p>
            </div>
            <div class="thesis-field">
              <span class="thesis-field-label">Expected Catalyst</span>
              <p>${t.catalyst||'<span class="muted">Not set</span>'}</p>
              ${catalystDate?`<div class="catalyst-passed">${catalystPassed?'⚠ ':''}${catalystDate}${catalystPassed?' — date has passed':''}</div>`:''}
            </div>
            <div class="thesis-field">
              <span class="thesis-field-label">Key Monitoring Metrics</span>
              <p>${t.metrics||'<span class="muted">Not set</span>'}</p>
            </div>
          </div>
          <div>
            <div class="thesis-meta-strip">
              <div><span class="thesis-stat-label">Entry Date</span><span class="thesis-stat-val">${t.entry_date||'—'}</span></div>
              <div><span class="thesis-stat-label">Price Target</span><span class="thesis-stat-val">${t.price_target?'$'+fmt(t.price_target,2):'—'}</span></div>
              <div><span class="thesis-stat-label">Bear Case</span><span class="thesis-stat-val">${t.bear_price?'$'+fmt(t.bear_price,2):'—'}</span></div>
              <div><span class="thesis-stat-label">Status</span><span class="thesis-stat-val" style="color:${statusColor};font-weight:700">${status||'—'}</span></div>
              <div><span class="thesis-stat-label">Conviction</span><span class="thesis-stat-val" style="color:${convColor}">${conviction||'—'}</span></div>
            </div>
            <div class="thesis-field">
              <span class="thesis-field-label">Update Log</span>
              <div class="thesis-notes-log">${notesHtml}</div>
            </div>
          </div>
        </div>
        <div class="thesis-actions">
          <button class="btn btn-gold" onclick="openThesisModal('${ticker}')">Update Thesis</button>
          <button class="btn btn-outline" onclick="toggleThesisCard('${ticker}')">Close</button>
        </div>
      </div>
    </td>
  </tr>`;
}

function openThesisModal(ticker){
  _thesisModalTicker = ticker;
  const t = loadThesis()[ticker] || {};
  $('tm-ticker').textContent = ticker;
  $('tf-thesis').value        = t.thesis       || '';
  $('tf-entry-date').value    = t.entry_date   || '';
  $('tf-target').value        = t.price_target || '';
  $('tf-bear').value          = t.bear_price   || '';
  $('tf-catalyst').value      = t.catalyst     || '';
  $('tf-catalyst-date').value = t.catalyst_date|| '';
  $('tf-status').value        = t.status       || 'Intact';
  $('tf-conviction').value    = t.conviction   || 'High';
  $('tf-metrics').value       = t.metrics      || '';
  $('tf-note').value          = '';
  $('thesis-modal').style.display = 'flex';
}

function closeThesisModal(){
  $('thesis-modal').style.display = 'none';
  _thesisModalTicker = null;
}

function saveThesisModal(){
  if(!_thesisModalTicker) return;
  const all      = loadThesis();
  const existing = all[_thesisModalTicker] || {};
  const noteText = $('tf-note').value.trim();
  const notes    = existing.notes || [];
  if(noteText) notes.push({ts: new Date().toLocaleString(), text: noteText});
  all[_thesisModalTicker] = {
    ...existing,
    thesis:        $('tf-thesis').value.trim(),
    entry_date:    $('tf-entry-date').value,
    price_target:  parseFloat($('tf-target').value)||null,
    bear_price:    parseFloat($('tf-bear').value)||null,
    catalyst:      $('tf-catalyst').value.trim(),
    catalyst_date: $('tf-catalyst-date').value,
    status:        $('tf-status').value,
    conviction:    $('tf-conviction').value,
    metrics:       $('tf-metrics').value.trim(),
    notes,
    updated_at:    new Date().toISOString(),
  };
  saveThesis(all);
  closeThesisModal();
  if(_cachedAttribution){ renderHoldings(_cachedAttribution); renderReviewQueue(_cachedAttribution); }
}

function renderReviewQueue(attribution){
  if(!attribution) return;
  const allThesis = loadThesis();
  const now = new Date();
  const flagged = [];
  for(const h of attribution){
    const t = allThesis[h.ticker] || {};
    if(!t.status && !t.thesis) continue;
    const isBroken      = t.status === 'Broken';
    const catalystPassed = t.catalyst_date && new Date(t.catalyst_date) < now;
    if(!isBroken && !catalystPassed) continue;
    const lastUpdated   = t.updated_at ? new Date(t.updated_at) : null;
    const daysSince     = lastUpdated ? Math.floor((now - lastUpdated)/86400000) : null;
    flagged.push({ticker:h.ticker, status:t.status, isBroken, catalystPassed, daysSince});
  }
  const panel = $('review-queue');
  if(!flagged.length){ panel.style.display='none'; return; }
  panel.style.display = 'block';
  $('rq-count').textContent = flagged.length;
  $('review-queue-body').innerHTML = flagged.map(f=>`
    <div class="alert-row">
      <span class="alert-tag" style="background:${f.isBroken?'rgba(239,68,68,.15)':'rgba(234,179,8,.15)'};color:${f.isBroken?'var(--red)':'var(--gold)'}">${f.isBroken?'Broken':'Catalyst Passed'}</span>
      <span style="font-weight:700">${f.ticker}</span>
      <span style="color:var(--text2);font-size:11px">${f.status||''}${f.catalystPassed?' · catalyst date passed':''}</span>
      <span style="color:var(--muted);font-size:11px">${f.daysSince!=null?f.daysSince+'d since last update':'Never updated'}</span>
      <button class="btn btn-outline" style="margin-left:auto;padding:4px 12px;font-size:11px" onclick="scrollToHolding('${f.ticker}')">Review now →</button>
    </div>`).join('');
}

function scrollToHolding(ticker){
  // Ensure we're on the overview tab
  const overviewBtn = document.querySelector('.nav-tab');
  if(overviewBtn && !overviewBtn.classList.contains('active')) switchTab('overview', overviewBtn);
  setTimeout(()=>{
    const row = document.getElementById('holding-row-'+ticker);
    if(row){ row.scrollIntoView({behavior:'smooth',block:'center'}); setTimeout(()=>toggleThesisCard(ticker),400); }
  }, 150);
}

// ── 52-Week Range Alerts ────────────────────────────────────────────────────────

function renderPriceAlerts(attribution){
  if(!attribution) return;
  let dismissed = {};
  try{ dismissed = JSON.parse(localStorage.getItem(W52D_KEY)||'{}'); }catch(e){}
  const now = Date.now();
  const alerts = [];
  for(const h of attribution){
    if(h.high_52w==null || h.low_52w==null || h.high_52w <= h.low_52w) continue;
    const pct = (h.price - h.low_52w) / (h.high_52w - h.low_52w) * 100;
    const dis = dismissed[h.ticker];
    if(dis && now - dis < 86400000) continue;
    if(pct >= 90) alerts.push({ticker:h.ticker, pct, type:'high',
      msg:`⚠ ${h.ticker} is near its 52-week high (${Math.round(pct)}% of range) — review thesis and valuation`});
    else if(pct <= 10) alerts.push({ticker:h.ticker, pct, type:'low',
      msg:`📉 ${h.ticker} is near its 52-week low (${Math.round(pct)}% of range) — confirm thesis is intact`});
  }
  // Update summary stats regardless of dismissals
  const allPcts = attribution.filter(h=>h.high_52w!=null&&h.low_52w!=null&&h.high_52w>h.low_52w)
    .map(h=>({ticker:h.ticker, pct:(h.price-h.low_52w)/(h.high_52w-h.low_52w)*100}));
  if($('stat-near-high')) $('stat-near-high').textContent = allPcts.filter(x=>x.pct>=90).length;
  if($('stat-near-low'))  $('stat-near-low').textContent  = allPcts.filter(x=>x.pct<=10).length;
  const panel = $('price-alerts');
  if(!alerts.length){ panel.style.display='none'; return; }
  panel.style.display = 'block';
  $('pa-count').textContent = alerts.length;
  $('price-alerts-body').innerHTML = alerts.map(a=>`
    <div class="alert-row" id="palert-${a.ticker}">
      <span style="flex:1;font-size:12px">${a.msg}</span>
      <button class="btn btn-outline" style="padding:3px 10px;font-size:11px;flex-shrink:0" onclick="scrollToHolding('${a.ticker}')">View thesis</button>
      <button class="btn btn-outline" style="padding:3px 10px;font-size:11px;color:var(--muted);flex-shrink:0" onclick="dismissPriceAlert('${a.ticker}')">Dismiss 24h</button>
    </div>`).join('');
}

function dismissPriceAlert(ticker){
  let dismissed = {};
  try{ dismissed = JSON.parse(localStorage.getItem(W52D_KEY)||'{}'); }catch(e){}
  dismissed[ticker] = Date.now();
  localStorage.setItem(W52D_KEY, JSON.stringify(dismissed));
  const el = document.getElementById('palert-'+ticker);
  if(el) el.remove();
  if($('price-alerts-body') && !$('price-alerts-body').children.length)
    $('price-alerts').style.display='none';
}

// ── Data loading ───────────────────────────────────────────────────────────────

async function loadMain(){
  $('last-updated').textContent='Refreshing…';
  try{
    const d=await (await fetch('/api/data')).json();
    if(d.error){$('last-updated').textContent='Error: '+d.error;return}
    _cachedMainData=d;
    _cachedAttribution=d.attribution;
    renderMetricBar(d);
    renderRiskMetrics(d);
    renderActiveRisks(d.active_risks, _cachedSectors);
    renderCapture(d);
    renderHoldings(d.attribution);
    renderReviewQueue(d.attribution);
    renderPriceAlerts(d.attribution);
    renderReturnersLosers(d.returners, d.losers);
    buildEquityChart(d.chart);
    buildRollingChart(d.rolling);
    $('last-updated').textContent='Updated '+new Date().toLocaleTimeString();
    // Refresh risk tab if already open
    if(_riskLoaded){ renderDDAnalytics(d); buildMCTRChart(d.attribution); renderRiskBubbleMapEl('risk-bubble-map-2',d.stock_risks); }
  }catch(e){$('last-updated').textContent='Error: '+e.message}
}

async function loadBenchmarks(){
  try{
    const rows=await (await fetch('/api/benchmarks')).json();
    renderBenchmarks(rows);
  }catch(e){$('bm-body').innerHTML=`<tr><td colspan="10" class="muted" style="padding:10px">Failed to load.</td></tr>`}
}

let _cachedSectors=null;
async function loadSectors(){
  try{
    const resp=await (await fetch('/api/sectors')).json();
    // resp is now {sectors:[...], rebalancing:[...]}
    const s=resp.sectors||resp;
    _cachedSectors=s;
    buildSectorChart(s);
    renderRebalancing(resp.rebalancing||[]);
    if(_cachedMainData) renderActiveRisks(_cachedMainData.active_risks, s);
  }catch(e){console.error('sectors',e)}
}

// ── Rebalancing ────────────────────────────────────────────────────────────────
function renderRebalancing(rows){
  if(!rows.length){$('rebal-list').innerHTML='<div class="muted" style="padding:8px 0;font-size:11px">No sector data available yet.</div>';return}
  $('rebal-list').innerHTML=rows.map(r=>{
    const w=Math.min(r.current_pct,50);
    const tw=Math.min(r.target_pct,50);
    const barCol=r.action==='TRIM'?'#ef4444':r.action==='ADD'?'#22c55e':'#6b7280';
    const driftStr=(r.drift>0?'+':'')+fmt(r.drift,1)+'%';
    const adjStr=r.dollar_adj>0?'+$'+Math.abs(r.dollar_adj).toLocaleString():(r.dollar_adj<0?'-$'+Math.abs(r.dollar_adj).toLocaleString():'—');
    const actionCls=r.action==='TRIM'?'action-trim':r.action==='ADD'?'action-add':'action-hold';
    return `<div class="rebal-row">
      <div>
        <div style="display:flex;align-items:center;gap:5px;margin-bottom:4px;font-size:11px;font-weight:600">
          <span style="display:inline-block;width:7px;height:7px;border-radius:2px;background:${r.color}"></span>${r.sector}
        </div>
        <div style="position:relative;height:4px;background:var(--border);border-radius:2px">
          <div style="position:absolute;top:0;left:0;height:100%;width:${w*2}%;background:${barCol};border-radius:2px;opacity:.8"></div>
          <div style="position:absolute;top:-3px;left:${tw*2}%;width:2px;height:10px;background:var(--text2);border-radius:1px"></div>
        </div>
      </div>
      <span style="text-align:right;color:var(--text)">${fmt(r.current_pct,1)}%</span>
      <span style="text-align:right;color:var(--text2)">${fmt(r.target_pct,1)}%</span>
      <span style="text-align:right;color:${r.drift>0?'var(--red)':r.drift<0?'var(--green)':'var(--muted)'};font-weight:600">${driftStr}</span>
      <span style="text-align:right"><span class="${actionCls}">${r.action}</span></span>
    </div>`;
  }).join('')+`<div style="font-size:10px;color:var(--text2);margin-top:8px;line-height:1.5">Bar = current weight · line = S&amp;P 500 target. TRIM/ADD suggests direction to reduce drift.</div>`;
}

async function loadWatchlist(){
  try{renderWatchlist(await (await fetch('/api/watchlist')).json())}catch{}
}

async function loadNews(){
  try{renderNews(await (await fetch('/api/news')).json())}catch{}
}

// ── Drawdown Analytics render ──────────────────────────────────────────────────
function renderDDAnalytics(d){
  if(!d) return;
  const da=d.drawdown_analytics;
  if(!da) return;
  $('dd-analytics').innerHTML=`
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:4px">
      <div class="ar-card"><div class="ar-label">Max Drawdown</div><div class="ar-val red">${fmtPct(da.max_drawdown)}</div></div>
      <div class="ar-card"><div class="ar-label">Avg Drawdown</div><div class="ar-val">${fmtPct(da.avg_drawdown)}</div></div>
      <div class="ar-card"><div class="ar-label">Ulcer Index</div><div class="ar-val">${fmt(da.ulcer_index,2)}</div></div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px">
      <div class="ar-card"><div class="ar-label">Recovery Days</div><div class="ar-val">${da.recovery_days}</div></div>
      <div class="ar-card"><div class="ar-label">Skewness</div><div class="ar-val ${da.skewness<0?'red':'green'}">${fmt(da.skewness,3)}</div></div>
      <div class="ar-card"><div class="ar-label">Excess Kurtosis</div><div class="ar-val">${fmt(da.kurtosis,3)}</div></div>
    </div>`;
}

// ── MCTR Chart ─────────────────────────────────────────────────────────────────
let mctrChart=null;
function buildMCTRChart(attribution){
  const ctx=$('mctr-chart').getContext('2d');
  if(mctrChart) mctrChart.destroy();
  const sorted=[...attribution].sort((a,b)=>b.mctr-a.mctr).slice(0,12);
  mctrChart=new Chart(ctx,{type:'bar',data:{
    labels:sorted.map(h=>h.ticker),
    datasets:[{data:sorted.map(h=>h.mctr),backgroundColor:sorted.map(h=>h.mctr>0?'rgba(239,68,68,.65)':'rgba(34,197,94,.65)'),borderRadius:3}]
  },options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{display:false},tooltip:{backgroundColor:'#1e293b',callbacks:{label:c=>` MCTR: ${fmt(c.raw,2)}%`}}},
    scales:{
      x:{grid:{display:false},ticks:{color:'#6b7280',font:{size:10}}},
      y:{position:'right',grid:{color:'rgba(31,42,61,0.5)'},ticks:{color:'#6b7280',font:{size:10},callback:v=>fmt(v,2)+'%'}}
    }
  }});
}

// ── Correlation Matrix ─────────────────────────────────────────────────────────
function renderCorrelation(data){
  const wrap=$('corr-wrap');
  if(!data||!data.tickers) return;
  const {tickers,matrix}=data;
  const N=tickers.length;
  const cellSz=Math.min(44, Math.floor((wrap.offsetWidth||600)/N));
  let html=`<div style="display:inline-block;font-size:${Math.max(8,cellSz*0.28)}px">`;
  // Header row
  html+=`<div style="display:flex;margin-left:${cellSz*2}px">`;
  for(const t of tickers) html+=`<div style="width:${cellSz}px;text-align:center;color:var(--text2);font-weight:600;overflow:hidden;white-space:nowrap">${t}</div>`;
  html+='</div>';
  for(let i=0;i<N;i++){
    html+=`<div style="display:flex;align-items:center">`;
    html+=`<div style="width:${cellSz*2}px;color:var(--text2);font-weight:600;overflow:hidden;white-space:nowrap;text-align:right;padding-right:6px">${tickers[i]}</div>`;
    for(let j=0;j<N;j++){
      const v=matrix[i][j];
      const abs=Math.abs(v);
      const r=v>0?Math.round(abs*59):0, g=v<0?Math.round(abs*68):0, b=v>0?0:Math.round(abs*68);
      const alpha=i===j?0.08:Math.pow(abs,0.6)*0.75;
      const highlight=(i!==j&&abs>0.75)?'outline:1px solid #f59e0b;':'';
      const fclr=i===j?'var(--text2)':abs>0.5?'#e2e8f0':'#94a3b8';
      html+=`<div style="width:${cellSz}px;height:${cellSz}px;display:flex;align-items:center;justify-content:center;background:rgba(${r},${g},${b===0?59:0},${alpha});${highlight}border-radius:2px;font-weight:600;color:${fclr}">${i===j?'—':fmt(v,2)}</div>`;
    }
    html+='</div>';
  }
  html+='</div>';
  wrap.innerHTML=html;
}

// ── Monte Carlo Chart ──────────────────────────────────────────────────────────
let mcChart=null;
function buildMCChart(mc){
  const ctx=$('mc-chart').getContext('2d');
  if(mcChart) mcChart.destroy();
  const days=mc.days;
  $('mc-params').textContent=`μ=${fmtPct(mc.mu_ann)} ann · σ=${fmtPct(mc.sigma_ann)} ann`;
  mcChart=new Chart(ctx,{type:'line',data:{labels:days,datasets:[
    {label:'P95',data:mc.p95,borderColor:'rgba(34,197,94,0.25)',backgroundColor:'rgba(34,197,94,0.06)',fill:'+1',pointRadius:0,borderWidth:1,tension:.3},
    {label:'P75',data:mc.p75,borderColor:'rgba(34,197,94,0.5)',backgroundColor:'rgba(34,197,94,0.12)',fill:'+1',pointRadius:0,borderWidth:1,tension:.3},
    {label:'P50',data:mc.p50,borderColor:'#e8a020',backgroundColor:'transparent',fill:false,pointRadius:0,borderWidth:2,tension:.3},
    {label:'P25',data:mc.p25,borderColor:'rgba(239,68,68,0.5)',backgroundColor:'rgba(239,68,68,0.12)',fill:'+1',pointRadius:0,borderWidth:1,tension:.3},
    {label:'P5', data:mc.p5, borderColor:'rgba(239,68,68,0.25)',backgroundColor:'rgba(239,68,68,0.06)',fill:false,pointRadius:0,borderWidth:1,tension:.3},
  ]},options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{display:true,position:'top',labels:{color:'#6b7280',font:{size:10},boxWidth:20,padding:12}},
      tooltip:{backgroundColor:'#1e293b',callbacks:{label:c=>` ${c.dataset.label}: ${fmt(c.raw,1)}`}}},
    scales:{
      x:{grid:{color:'rgba(31,42,61,0.5)'},ticks:{color:'#6b7280',maxTicksLimit:8,font:{size:10},callback:(_,i)=>i%30===0?`Day ${mc.days[i]}`:''}},
      y:{position:'right',grid:{color:'rgba(31,42,61,0.5)'},ticks:{color:'#6b7280',font:{size:10},callback:v=>fmt(v)}}
    }
  }});
}

// ── Stress Tests ───────────────────────────────────────────────────────────────
function renderStress(rows){
  $('stress-body').innerHTML=rows.map(r=>`<tr>
    <td><div style="font-weight:700">${r.scenario}</div><div style="font-size:10px;color:var(--text2)">${r.label}</div></td>
    <td class="${r.mkt_return<0?'neg':'pos'}">${fmtPct(r.mkt_return)}</td>
    <td class="${r.port_return<0?'neg':'pos'}">${fmtPct(r.port_return)}</td>
    <td class="${r.dollar_pnl<0?'neg':'pos'}">${fmtUSD(r.dollar_pnl)}</td>
  </tr>`).join('');
}

// ── Risk tab loader ────────────────────────────────────────────────────────────
let _cachedAttribution=null;
async function loadRisk(){
  if(_cachedAttribution) buildMCTRChart(_cachedAttribution);
  renderDDAnalytics(_cachedMainData);
  if(_cachedMainData?.stock_risks) renderRiskBubbleMapEl('risk-bubble-map-2',_cachedMainData.stock_risks);
  if(_cachedMainData?.monthly) renderHeatmap(_cachedMainData.monthly);
  try{
    const corr=await (await fetch('/api/correlation')).json();
    renderCorrelation(corr);
  }catch{}
}

function renderRiskBubbleMapEl(elId, stockRisks){
  // Reuse existing bubble map renderer logic on a different element
  const wrap=$(elId);
  if(!wrap||!stockRisks.length) return;
  _rbmStocks=stockRisks;
  const W=wrap.clientWidth||500,H=wrap.clientHeight||300;
  const DPR=window.devicePixelRatio||1;
  const maxScore=Math.max(...stockRisks.map(s=>s.risk_score),1);
  const MIN_R=16,MAX_R=Math.min(52,W/7);
  const circles=stockRisks.map(s=>{
    const norm=s.risk_score/maxScore;
    const r=MIN_R+Math.pow(norm,0.65)*(MAX_R-MIN_R);
    const col=norm>0.62?'#ef4444':norm>0.36?'#eab308':'#22c55e';
    return{...s,r,col,fillA:norm>0.62?0.22:norm>0.36?0.18:0.14,x:0,y:0,norm};
  }).sort((a,b)=>b.r-a.r);
  const placed=[],cx=W/2,cy=H/2;
  for(let i=0;i<circles.length;i++){
    const c=circles[i];
    if(i===0){c.x=cx;c.y=cy;placed.push(c);continue;}
    let ok=false;
    outer:for(let dist=0;dist<Math.max(W,H);dist+=3){
      const steps=Math.max(8,Math.round(dist*0.7));
      for(let step=0;step<steps;step++){
        const ang=(step/steps)*Math.PI*2,tx=cx+dist*Math.cos(ang),ty=cy+dist*Math.sin(ang);
        if(tx-c.r<6||tx+c.r>W-6||ty-c.r<6||ty+c.r>H-6)continue;
        let clash=false;
        for(const p of placed){const dx=tx-p.x,dy=ty-p.y;if(dx*dx+dy*dy<(c.r+p.r+5)**2){clash=true;break;}}
        if(!clash){c.x=tx;c.y=ty;placed.push(c);ok=true;break outer;}
      }
    }
    if(!ok){c.x=cx+(i%6-3)*50;c.y=cy+Math.floor(i/6)*50;placed.push(c);}
  }
  const canvas=document.createElement('canvas');
  canvas.width=W*DPR;canvas.height=H*DPR;canvas.style.cssText=`width:${W}px;height:${H}px`;
  const ctx2=canvas.getContext('2d');ctx2.scale(DPR,DPR);
  for(const c of placed){
    const grd=ctx2.createRadialGradient(c.x,c.y,c.r*0.3,c.x,c.y,c.r*1.4);
    grd.addColorStop(0,c.col+'28');grd.addColorStop(1,'transparent');
    ctx2.beginPath();ctx2.arc(c.x,c.y,c.r*1.4,0,Math.PI*2);ctx2.fillStyle=grd;ctx2.fill();
    ctx2.beginPath();ctx2.arc(c.x,c.y,c.r,0,Math.PI*2);
    const hx=c.col.slice(1);
    ctx2.fillStyle=`rgba(${parseInt(hx.slice(0,2),16)},${parseInt(hx.slice(2,4),16)},${parseInt(hx.slice(4,6),16)},${c.fillA})`;
    ctx2.fill();ctx2.strokeStyle=c.col;ctx2.lineWidth=1.8;ctx2.stroke();
    const fsize=Math.max(9,Math.min(13,c.r*0.38));
    ctx2.fillStyle=c.col;ctx2.font=`700 ${fsize}px Inter,sans-serif`;
    ctx2.textAlign='center';ctx2.textBaseline='middle';
    ctx2.fillText(c.ticker,c.x,c.y-(c.r>24?4:0));
    if(c.r>26){ctx2.font=`${Math.max(8,fsize-3)}px Inter,sans-serif`;ctx2.fillStyle='rgba(148,163,184,0.85)';ctx2.fillText(c.risk_score.toFixed(0),c.x,c.y+fsize-2);}
  }
  // ── Tooltip ──────────────────────────────────────────────────────────────────
  const tip=document.createElement('div');
  tip.style.cssText='position:absolute;display:none;pointer-events:none;background:rgba(15,23,42,0.95);border:1px solid rgba(148,163,184,0.25);border-radius:8px;padding:10px 13px;font-size:11px;line-height:1.7;color:#e2e8f0;white-space:nowrap;z-index:999;box-shadow:0 4px 20px rgba(0,0,0,0.5)';
  wrap.style.position='relative';
  wrap.innerHTML='';
  wrap.appendChild(canvas);
  wrap.appendChild(tip);

  canvas.addEventListener('mousemove',function(e){
    const rect=canvas.getBoundingClientRect();
    const mx=(e.clientX-rect.left);
    const my=(e.clientY-rect.top);
    let hit=null;
    for(const c of placed){
      if(Math.hypot(mx-c.x,my-c.y)<=c.r){hit=c;break;}
    }
    if(hit){
      const fmtPct=v=>(v==null?'—':(v*100).toFixed(1)+'%');
      const fmtN=v=>(v==null?'—':v.toFixed(2));
      const riskCol=hit.norm>0.62?'#ef4444':hit.norm>0.36?'#eab308':'#22c55e';
      tip.innerHTML=`
        <div style="font-weight:700;font-size:13px;color:${riskCol};margin-bottom:5px">${hit.ticker}</div>
        <div><span style="color:#94a3b8">Risk Score</span>  <strong style="color:${riskCol}">${hit.risk_score.toFixed(0)}/100</strong></div>
        <div><span style="color:#94a3b8">Weight</span>      <strong>${fmtPct(hit.weight)}</strong></div>
        <div><span style="color:#94a3b8">Volatility</span>  <strong>${fmtPct(hit.vol)}</strong></div>
        <div><span style="color:#94a3b8">Beta</span>        <strong>${fmtN(hit.beta)}</strong></div>
        <div><span style="color:#94a3b8">Max DD</span>      <strong>${fmtPct(hit.max_dd)}</strong></div>
        <div><span style="color:#94a3b8">VaR 95%</span>    <strong>${fmtPct(hit.var_95)}</strong></div>
      `.replace(/  +/g,' ');
      // Position tooltip — prefer right of cursor, flip left if near edge
      let tx=mx+14, ty=my-10;
      const TW=180, TH=130;
      if(tx+TW>W-8) tx=mx-TW-10;
      if(ty+TH>H-8) ty=H-TH-8;
      tip.style.left=tx+'px';
      tip.style.top=ty+'px';
      tip.style.display='block';
      canvas.style.cursor='pointer';
    } else {
      tip.style.display='none';
      canvas.style.cursor='default';
    }
  });
  canvas.addEventListener('mouseleave',function(){
    tip.style.display='none';
    canvas.style.cursor='default';
  });
}

// ── Scenarios tab loader ────────────────────────────────────────────────────────
let _cachedMainData=null;
async function loadScenarios(){
  try{
    const mc=await (await fetch('/api/monte-carlo')).json();
    buildMCChart(mc);
  }catch(e){console.error(e)}
  try{
    const stress=await (await fetch('/api/stress')).json();
    renderStress(stress);
  }catch{}
}

// ── Hedge Fund Tracker ─────────────────────────────────────────────────────────
let _hfData=[], _hfActive=0;

async function loadHedgeFunds(){
  try{
    $('hf-panel').innerHTML='<div class="spinner"><div class="spin"></div>Fetching SEC 13F filings…</div>';
    const data=await (await fetch('/api/hedge-funds')).json();
    _hfData=data;
    renderHFTabs();
    renderHFPanel(0);
  }catch(e){
    $('hf-panel').innerHTML=`<div class="muted" style="padding:10px;font-size:12px">Failed to load: ${e.message}</div>`;
  }
}

function renderHFTabs(){
  $('hf-tabs').innerHTML=_hfData.map((f,i)=>`
    <button class="hf-tab${i===_hfActive?' active':''}" onclick="renderHFPanel(${i})">${f.fund}</button>
  `).join('');
}

function renderHFPanel(idx){
  _hfActive=idx;
  renderHFTabs();
  const f=_hfData[idx];
  if(!f){return}
  if(f.error){
    $('hf-panel').innerHTML=`<div class="muted" style="font-size:12px;padding:8px 0">⚠ ${f.error}</div>`;
    return;
  }
  const totalAUM=fmtB(f.total_aum);
  const changeCount=f.changes?.length||0;
  const changeBadge=changeCount?`<span class="pill ${f.changes.some(c=>c.type==='NEW')?'green':'blue'}" style="font-size:10px">${changeCount} change${changeCount>1?'s':''} vs prior filing</span>`:'';

  let changesHtml='';
  if(f.changes&&f.changes.length){
    changesHtml=`<div style="margin-bottom:10px">
      <div style="font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px">Changes since ${f.prev_date||'prior filing'}</div>
      <div style="display:flex;flex-wrap:wrap;gap:3px">
        ${f.changes.map(c=>{
          const cls=c.type==='NEW'?'hf-new':c.type==='CLOSED'?'hf-closed':c.type==='INCREASED'?'hf-inc':'hf-dec';
          const icon=c.type==='NEW'?'★':c.type==='CLOSED'?'✕':c.type==='INCREASED'?'↑':'↓';
          const extra=c.chg?` ${c.chg}`:'';
          return `<span class="hf-change-badge ${cls}">${icon} ${c.name}${extra}</span>`;
        }).join('')}
      </div>
    </div>`;
  }

  let holdingsHtml=f.holdings.map((h,i)=>`
    <div class="hf-holding-row">
      <div style="display:flex;align-items:center;gap:8px">
        <span style="color:var(--text2);width:16px;text-align:right;font-size:10px">${i+1}</span>
        <span style="font-weight:600">${h.name}</span>
      </div>
      <div style="text-align:right">
        <div style="font-weight:600">${fmt(h.pct,2)}%</div>
        <div style="color:var(--text2);font-size:10px">${fmtB(h.value)}</div>
      </div>
    </div>`).join('');

  $('hf-panel').innerHTML=`
    <div class="hf-fund-header">
      <div>
        <div class="hf-fund-name">${f.fund}</div>
        <div class="hf-fund-meta">Filed ${f.filing_date} · ${f.n_holdings} positions · AUM ${totalAUM}</div>
      </div>
      ${changeBadge}
    </div>
    ${changesHtml}
    <div style="font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px">Top Holdings</div>
    ${holdingsHtml}
    <div style="font-size:10px;color:var(--muted);margin-top:8px;line-height:1.4">Source: SEC EDGAR 13F-HR · Filed quarterly · 45-day lag after period end. Positions shown are as of the filing date, not current.</div>`;
}

// ── Portfolio Simulator ────────────────────────────────────────────────────────
let _simTrades = [];

function addSimTrade(){
  const ticker = $('sim-ticker').value.trim().toUpperCase();
  const shares = parseFloat($('sim-shares').value);
  const price  = parseFloat($('sim-price').value) || 0;
  const action = $('sim-action').value;
  if(!ticker || !shares || shares <= 0){ $('sim-status').textContent='Enter a ticker and shares.'; return; }
  _simTrades.push({ticker, shares, price, action});
  $('sim-ticker').value=''; $('sim-shares').value=''; $('sim-price').value='';
  $('sim-status').textContent='';
  renderSimTrades();
}

function removeSimTrade(i){
  _simTrades.splice(i,1);
  renderSimTrades();
}

function renderSimTrades(){
  if(!_simTrades.length){
    $('sim-trades').innerHTML='<span style="color:var(--text2);font-size:11px">No trades added yet.</span>';
    return;
  }
  $('sim-trades').innerHTML=_simTrades.map((t,i)=>`
    <span class="sim-trade-tag ${t.action}">
      <span class="${t.action==='buy'?'green':'red'}" style="font-weight:700">${t.action.toUpperCase()}</span>
      <span style="font-weight:600">${t.shares} × ${t.ticker}</span>
      ${t.price?`<span class="muted">@ $${t.price}</span>`:'<span class="muted">@ market</span>'}
      <span onclick="removeSimTrade(${i})" style="cursor:pointer;color:var(--red);margin-left:2px">✕</span>
    </span>`).join('');
}

function clearSimulation(){
  _simTrades=[];
  renderSimTrades();
  $('sim-results').style.display='none';
  $('sim-status').textContent='';
}

async function runSimulation(){
  if(!_simTrades.length){ $('sim-status').textContent='Add at least one trade.'; return; }
  if(!_cachedMainData){ $('sim-status').textContent='Waiting for portfolio data…'; return; }
  const btn=$('sim-run-btn');
  btn.disabled=true; btn.textContent='Running…'; $('sim-status').textContent='';
  try{
    const resp = await fetch('/api/simulate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({changes:_simTrades})});
    const sim  = await resp.json();
    if(sim.error){ $('sim-status').textContent='Error: '+sim.error; return; }
    renderSimResults(sim);
  }catch(e){
    $('sim-status').textContent='Failed: '+e.message;
  }finally{
    btn.disabled=false; btn.textContent='Run Simulation';
  }
}

function renderSimResults(sim){
  const b=_cachedMainData;
  function baCard(label, before, after, fmt_fn, higherIsBetter=true){
    const delta=after-before;
    const pct=before!==0?delta/Math.abs(before)*100:0;
    const better=(higherIsBetter&&delta>=0)||(!higherIsBetter&&delta<=0);
    const deltaCls=delta===0?'muted':better?'green':'red';
    return `<div class="ba-card">
      <div class="ba-label">${label}</div>
      <div class="ba-before">${fmt_fn(before)}</div>
      <div class="ba-after" style="color:${delta===0?'var(--text)':better?'var(--green)':'var(--red)'}">${fmt_fn(after)}</div>
      <div class="ba-delta ${deltaCls}">${delta>=0?'+':''}${fmt_fn(delta)} ${Math.abs(pct)>0.1?'('+fmtPct(pct)+')':''}</div>
    </div>`;
  }
  $('ba-grid').innerHTML=
    baCard('Sharpe',    b.sharpe,            sim.sharpe,            v=>fmt(v,3), true)+
    baCard('Beta',      b.beta,              sim.beta,              v=>fmt(v,3), false)+
    baCard('Ann Return',b.port_ann_return,   sim.port_ann_return,   v=>fmtPct(v), true)+
    baCard('VaR 95%',   b.var_95,            sim.var_95,            v=>fmtPct(v), true)+
    baCard('Max DD',    b.max_drawdown,      sim.max_drawdown,      v=>fmtPct(v), true)+
    baCard('Alpha',     b.alpha,             sim.alpha,             v=>fmtPct(v), true)+
    baCard('Track.Err', b.tracking_error,    sim.tracking_error,    v=>fmt(v,2)+'%', false)+
    baCard('Risk Score',b.risk_summary_score,sim.risk_summary_score,v=>v.toFixed(0), false);

  // Simulated holdings table
  $('sim-holdings-body').innerHTML=(sim.attribution||[]).map(h=>`<tr>
    <td style="font-weight:700">${h.ticker}</td>
    <td>${fmt(h.shares,0)}</td>
    <td>${fmt(h.weight,1)}%</td>
    <td class="${cc(h.period_return)}">${fmtPct(h.period_return)}</td>
    <td class="muted">${h.mctr!=null?fmt(h.mctr,2)+'%':'—'}</td>
  </tr>`).join('');

  $('sim-results').style.display='block';
}

// Also allow Enter key to add trade
document.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&document.activeElement&&['sim-ticker','sim-shares','sim-price'].includes(document.activeElement.id)) addSimTrade();
});

// Boot
loadMain();
loadBenchmarks();
loadSectors();
loadWatchlist();
loadNews();
loadHedgeFunds();
loadEarnings();
setInterval(loadMain,      60_000);
setInterval(loadBenchmarks,300_000);
setInterval(loadSectors,   600_000);
setInterval(loadWatchlist, 120_000);
setInterval(loadNews,      300_000);
setInterval(loadEarnings, 4*3600*1000);
</script>

<!-- ── Earnings Notes Modal ──────────────────────────────────────────────── -->
<div id="earn-modal" style="display:none" class="modal-overlay" onclick="if(event.target===this)closeEarnModal()">
  <div class="modal-box" onclick="event.stopPropagation()">
    <div class="modal-title">Earnings Notes &mdash; <span class="modal-ticker" id="em-ticker"></span></div>
    <div class="form-group">
      <label>Pre-Earnings Notes</label>
      <textarea id="em-pre" rows="4" placeholder="Key metrics to watch, price levels, positioning notes…"></textarea>
    </div>
    <div class="form-group">
      <label>Post-Earnings Action Plan</label>
      <textarea id="em-post" rows="4" placeholder="If beats: … If misses: … Price targets / stop levels…"></textarea>
    </div>
    <div class="modal-actions">
      <button class="btn btn-outline" onclick="closeEarnModal()">Cancel</button>
      <button class="btn btn-gold" onclick="saveEarnModal()">Save Notes</button>
    </div>
  </div>
</div>

<!-- ── Thesis Modal ───────────────────────────────────────────────────────── -->
<div id="thesis-modal" style="display:none" class="modal-overlay" onclick="if(event.target===this)closeThesisModal()">
  <div class="modal-box" onclick="event.stopPropagation()">
    <div class="modal-title">Update Thesis &mdash; <span class="modal-ticker" id="tm-ticker"></span></div>
    <div class="form-group">
      <label>Buy Thesis</label>
      <textarea id="tf-thesis" rows="3" placeholder="Why do you own this position? 2-3 sentences."></textarea>
    </div>
    <div class="form-row-3">
      <div class="form-group"><label>Entry Date</label><input id="tf-entry-date" type="date"></div>
      <div class="form-group"><label>Price Target ($)</label><input id="tf-target" type="number" step="0.01" placeholder="Upside case"></div>
      <div class="form-group"><label>Bear Case ($)</label><input id="tf-bear" type="number" step="0.01" placeholder="Downside scenario"></div>
    </div>
    <div class="form-row-2">
      <div class="form-group"><label>Expected Catalyst</label><input id="tf-catalyst" placeholder="e.g. Q3 earnings beat"></div>
      <div class="form-group"><label>Catalyst Date</label><input id="tf-catalyst-date" type="date"></div>
    </div>
    <div class="form-row-2">
      <div class="form-group"><label>Thesis Status</label>
        <select id="tf-status">
          <option value="Intact">Intact</option>
          <option value="Weakening">Weakening</option>
          <option value="Broken">Broken</option>
        </select>
      </div>
      <div class="form-group"><label>Conviction Level</label>
        <select id="tf-conviction">
          <option value="High">High</option>
          <option value="Medium">Medium</option>
          <option value="Low">Low</option>
        </select>
      </div>
    </div>
    <div class="form-group">
      <label>Key Monitoring Metrics</label>
      <input id="tf-metrics" placeholder="e.g. gross margin > 40%, revenue growth > 15%">
    </div>
    <div class="form-group">
      <label>Add Note to Update Log</label>
      <textarea id="tf-note" rows="2" placeholder="Optional: timestamped note added to the update log…"></textarea>
    </div>
    <div class="modal-actions">
      <button class="btn btn-outline" onclick="closeThesisModal()">Cancel</button>
      <button class="btn btn-gold" onclick="saveThesisModal()">Save Thesis</button>
    </div>
  </div>
</div>

<!-- ── Manage Positions Modal ─────────────────────────────────────────────── -->
<div id="positions-modal" style="display:none" class="modal-overlay" onclick="if(event.target===this)closePositionsModal()">
  <div class="modal-box" style="width:680px;max-width:98vw">
    <div class="modal-title">&#9776; Manage Positions</div>

    <!-- Current holdings list -->
    <div style="margin-bottom:18px">
      <div style="font-size:11px;color:var(--text2);text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px">Current Holdings</div>
      <div id="pm-holdings-list" style="display:flex;flex-direction:column;gap:4px;max-height:280px;overflow-y:auto"></div>
    </div>

    <hr style="border:none;border-top:1px solid var(--border);margin:16px 0">

    <!-- Add / update position form -->
    <div style="margin-bottom:4px">
      <div style="font-size:11px;color:var(--text2);text-transform:uppercase;letter-spacing:.6px;margin-bottom:10px">Add or Update Position</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr auto;gap:8px;align-items:end">
        <div>
          <label style="font-size:10px;color:var(--text2);display:block;margin-bottom:4px">Ticker *</label>
          <input id="pm-ticker" type="text" placeholder="e.g. AAPL" style="width:100%;box-sizing:border-box;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:7px 10px;color:var(--text);font-size:13px;text-transform:uppercase" oninput="this.value=this.value.toUpperCase()">
        </div>
        <div>
          <label style="font-size:10px;color:var(--text2);display:block;margin-bottom:4px">Shares *</label>
          <input id="pm-shares" type="number" min="0.000001" step="any" placeholder="0.00" style="width:100%;box-sizing:border-box;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:7px 10px;color:var(--text);font-size:13px">
        </div>
        <div>
          <label style="font-size:10px;color:var(--text2);display:block;margin-bottom:4px">Avg Cost (optional)</label>
          <input id="pm-cost" type="number" min="0" step="any" placeholder="0.00" style="width:100%;box-sizing:border-box;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:7px 10px;color:var(--text);font-size:13px">
        </div>
        <button class="btn btn-gold" style="height:34px;white-space:nowrap" onclick="pmAddPosition()">Add / Update</button>
      </div>
      <div id="pm-add-msg" style="font-size:11px;margin-top:6px;min-height:16px"></div>
    </div>

    <div class="modal-actions">
      <button class="btn btn-outline" onclick="closePositionsModal()">Close</button>
      <button class="btn" style="background:var(--blue)" onclick="pmRefreshPage()">&#8635; Refresh Dashboard</button>
    </div>
  </div>
</div>

<script>
// ── Manage Positions ──────────────────────────────────────────────────────────
let _pmHoldings = [];

async function openPositionsModal(){
  const modal = document.getElementById('positions-modal');
  modal.style.display = 'flex';
  await pmLoadHoldings();
}

function closePositionsModal(){
  document.getElementById('positions-modal').style.display = 'none';
}

async function pmLoadHoldings(){
  try{
    const res = await fetch('/api/holdings');
    _pmHoldings = await res.json();
    pmRenderList();
  }catch(e){
    document.getElementById('pm-holdings-list').innerHTML = '<div style="color:var(--red);font-size:12px">Failed to load holdings</div>';
  }
}

function pmRenderList(){
  const el = document.getElementById('pm-holdings-list');
  if(!_pmHoldings.length){
    el.innerHTML = '<div style="color:var(--text2);font-size:12px;padding:8px 0">No positions yet.</div>';
    return;
  }
  el.innerHTML = _pmHoldings.map(h=>`
    <div style="display:grid;grid-template-columns:80px 1fr 1fr 1fr auto auto;gap:6px;align-items:center;padding:7px 10px;background:var(--bg);border-radius:6px;border:1px solid var(--border)">
      <span style="font-weight:700;color:var(--gold)">${h.ticker}</span>
      <span style="font-size:12px;color:var(--text2)">
        <input type="number" value="${h.shares}" min="0.000001" step="any" id="pm-s-${h.ticker}"
          style="width:90px;background:transparent;border:1px solid var(--border);border-radius:4px;padding:3px 6px;color:var(--text);font-size:12px">
        <span style="color:var(--text2);font-size:10px">shares</span>
      </span>
      <span style="font-size:12px;color:var(--text2)">
        <input type="number" value="${h.avg_cost}" min="0" step="any" id="pm-c-${h.ticker}"
          style="width:90px;background:transparent;border:1px solid var(--border);border-radius:4px;padding:3px 6px;color:var(--text);font-size:12px">
        <span style="color:var(--text2);font-size:10px">avg cost</span>
      </span>
      <span style="font-size:11px;color:var(--text2)">Mkt Val: <strong>—</strong></span>
      <button onclick="pmSaveEdit('${h.ticker}')" style="background:none;border:1px solid var(--border);border-radius:5px;color:var(--text2);font-size:11px;padding:3px 8px;cursor:pointer">Save</button>
      <button onclick="pmDelete('${h.ticker}')" style="background:none;border:1px solid rgba(239,68,68,.4);border-radius:5px;color:#ef4444;font-size:11px;padding:3px 8px;cursor:pointer">Remove</button>
    </div>
  `).join('');
}

async function pmSaveEdit(ticker){
  const shares  = parseFloat(document.getElementById('pm-s-'+ticker)?.value||0);
  const avgCost = parseFloat(document.getElementById('pm-c-'+ticker)?.value||0);
  if(!shares || shares<=0){ alert('Shares must be > 0'); return; }
  try{
    const r = await fetch('/api/holdings/'+ticker, {
      method:'PUT', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({shares, avg_cost: avgCost})
    });
    const data = await r.json();
    if(!r.ok){ alert(data.error||'Update failed'); return; }
    _pmHoldings = data.holdings;
    pmRenderList();
    pmSetMsg('Saved '+ticker, 'var(--green)');
  }catch(e){ alert('Error: '+e.message); }
}

async function pmDelete(ticker){
  if(!confirm(`Remove ${ticker} from portfolio?`)) return;
  try{
    const r = await fetch('/api/holdings/'+ticker, {method:'DELETE'});
    const data = await r.json();
    if(!r.ok){ alert(data.error||'Delete failed'); return; }
    _pmHoldings = data.holdings;
    pmRenderList();
    pmSetMsg('Removed '+ticker, 'var(--red)');
  }catch(e){ alert('Error: '+e.message); }
}

async function pmAddPosition(){
  const ticker  = document.getElementById('pm-ticker').value.trim().toUpperCase();
  const shares  = parseFloat(document.getElementById('pm-shares').value||0);
  const avgCost = parseFloat(document.getElementById('pm-cost').value||0);
  if(!ticker){ pmSetMsg('Enter a ticker symbol', 'var(--red)'); return; }
  if(!shares||shares<=0){ pmSetMsg('Shares must be > 0', 'var(--red)'); return; }
  try{
    const r = await fetch('/api/holdings', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ticker, shares, avg_cost: avgCost})
    });
    const data = await r.json();
    if(!r.ok){ pmSetMsg(data.error||'Add failed', 'var(--red)'); return; }
    _pmHoldings = data.holdings;
    pmRenderList();
    document.getElementById('pm-ticker').value  = '';
    document.getElementById('pm-shares').value  = '';
    document.getElementById('pm-cost').value    = '';
    pmSetMsg('Added '+ticker, 'var(--green)');
  }catch(e){ pmSetMsg('Error: '+e.message, 'var(--red)'); }
}

function pmSetMsg(msg, color){
  const el = document.getElementById('pm-add-msg');
  el.style.color = color;
  el.textContent = msg;
  setTimeout(()=>{ el.textContent=''; }, 4000);
}

function pmRefreshPage(){
  closePositionsModal();
  // Clear provider cache then reload all dashboard data
  if(typeof loadMain === 'function') loadMain();
  if(typeof loadBenchmarks === 'function') loadBenchmarks();
  if(typeof loadSectors === 'function') loadSectors();
  if(typeof loadEarnings === 'function') loadEarnings();
}
</script>

</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


# ── CLI entry ──────────────────────────────────────────────────────────────────

def serve(host: str = "127.0.0.1", port: int = 5000, open_browser: bool = True) -> None:
    if open_browser:
        import threading
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
