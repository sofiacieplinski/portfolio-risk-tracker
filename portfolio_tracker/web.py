"""
Web dashboard for Portfolio Risk Metrics Tracker.
Start with: portfolio-tracker serve
"""
from __future__ import annotations

import asyncio
import json
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from pathlib import Path
DATA_DIR      = Path.home() / ".portfolio_tracker"
HOLDINGS_FILE = DATA_DIR / "holdings.json"
from .providers import get_prices

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Portfolio Risk Tracker", docs_url=None, redoc_url=None)

ANN   = 252
PERIOD = "1y"

# ── Data helpers ──────────────────────────────────────────────────────────────

def _load() -> dict:
    if HOLDINGS_FILE.exists():
        with open(HOLDINGS_FILE) as f:
            return json.load(f)
    return {"holdings": [], "benchmark": "SPY", "risk_free_rate": 0.05}


def _compute(data: dict) -> dict | None:
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

    last = prices.iloc[-1]
    port_vals: dict[str, float] = {}
    total_val = 0.0
    for h in holdings:
        mv = float(last[h["ticker"]]) * h["shares"]
        port_vals[h["ticker"]] = mv
        total_val += mv

    weights = {t: v / total_val for t, v in port_vals.items()}
    w_arr   = np.array([weights.get(t, 0.0) for t in tickers])
    pr      = pd.Series(returns[tickers].values @ w_arr, index=returns.index)
    br      = returns[benchmark]
    idx     = pr.index.intersection(br.index)
    pr, br  = pr[idx], br[idx]

    cov_mat  = np.cov(pr.values, br.values)
    beta     = float(cov_mat[0, 1] / cov_mat[1, 1])
    port_ann = float((1 + pr.mean()) ** ANN - 1)
    bench_ann= float((1 + br.mean()) ** ANN - 1)
    sharpe   = float(((pr - rf_daily).mean() / pr.std()) * np.sqrt(ANN))
    down     = pr[pr < rf_daily] - rf_daily
    down_std = float(np.sqrt((down**2).mean()) * np.sqrt(ANN)) if len(down) > 1 else float(pr.std() * np.sqrt(ANN))
    sortino  = (port_ann - rf_rate) / down_std if down_std > 1e-10 else 0.0
    active   = pr - br
    te       = float(active.std() * np.sqrt(ANN))
    var_95   = float(np.percentile(pr.values, 5))
    cum      = (1 + pr).cumprod()
    max_dd   = float(((cum - cum.cummax()) / cum.cummax()).min())
    alpha    = port_ann - (rf_rate + beta * (bench_ann - rf_rate))
    ir       = float(active.mean() / active.std() * np.sqrt(ANN)) if active.std() > 0 else 0.0

    # Day change (last day's portfolio return)
    day_ret  = float(pr.iloc[-1]) if len(pr) else 0.0
    day_chg  = total_val * day_ret

    # Previous close for holdings (for day change per ticker)
    prev = prices.iloc[-2] if len(prices) > 1 else prices.iloc[-1]

    # Equity curve — cumulative portfolio & benchmark (indexed to 100)
    port_cum  = (1 + pr).cumprod() * 100
    bench_cum = (1 + br).cumprod() * 100
    dates     = [d.strftime("%Y-%m-%d") for d in pr.index]

    # Attribution
    attribution = []
    for h in holdings:
        t = h["ticker"]
        if t not in returns.columns:
            continue
        cur_price  = float(last[t])
        prev_price = float(prev[t])
        day_chg_t  = (cur_price - prev_price) / prev_price if prev_price else 0.0
        period_ret = float((1 + returns[t]).prod() - 1)
        w          = weights.get(t, 0.0)
        gl         = (cur_price - h["avg_cost"]) * h["shares"]
        gl_pct     = (cur_price - h["avg_cost"]) / h["avg_cost"]
        attribution.append({
            "ticker":       t,
            "shares":       h["shares"],
            "avg_cost":     h["avg_cost"],
            "price":        round(cur_price, 4),
            "market_value": round(port_vals[t], 2),
            "gain_loss":    round(gl, 2),
            "gain_loss_pct":round(gl_pct * 100, 2),
            "weight":       round(w * 100, 2),
            "day_change_pct": round(day_chg_t * 100, 2),
            "period_return":round(period_ret * 100, 2),
            "contribution": round(w * period_ret * 100, 2),
        })

    attribution.sort(key=lambda x: -x["market_value"])

    return {
        "total_value":    round(total_val, 2),
        "day_change":     round(day_chg, 2),
        "day_change_pct": round(day_ret * 100, 2),
        "port_ann_return":round(port_ann * 100, 2),
        "bench_ann_return":round(bench_ann * 100, 2),
        "active_return":  round((port_ann - bench_ann) * 100, 2),
        "beta":           round(beta, 3),
        "sharpe":         round(sharpe, 3),
        "sortino":        round(sortino, 3),
        "alpha":          round(alpha * 100, 2),
        "tracking_error": round(te * 100, 2),
        "var_95":         round(var_95 * 100, 2),
        "max_drawdown":   round(max_dd * 100, 2),
        "info_ratio":     round(ir, 3),
        "benchmark":      benchmark,
        "rf_rate":        rf_rate,
        "n_days":         len(pr),
        "provider":       provider,
        "attribution":    attribution,
        "chart": {
            "dates":     dates,
            "portfolio": [round(v, 2) for v in port_cum.tolist()],
            "benchmark": [round(v, 2) for v in bench_cum.tolist()],
        },
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/data")
async def api_data():
    loop = asyncio.get_event_loop()
    data = _load()
    result = await loop.run_in_executor(None, _compute, data)
    if result is None:
        return JSONResponse({"error": "No holdings or data unavailable."}, status_code=503)
    return result


@app.get("/api/news")
async def api_news():
    import yfinance as yf
    data     = _load()
    tickers  = [h["ticker"] for h in data["holdings"]][:6]
    articles = []
    seen     = set()
    for t in tickers:
        try:
            news = yf.Ticker(t).news or []
            for n in news[:3]:
                title = n.get("title", "")
                if title and title not in seen:
                    seen.add(title)
                    articles.append({
                        "title":     title,
                        "publisher": n.get("publisher", ""),
                        "link":      n.get("link", "#"),
                        "ticker":    t,
                        "age":       _age(n.get("providerPublishTime", 0)),
                    })
        except Exception:
            pass
    return articles[:12]


def _age(ts: int) -> str:
    if not ts:
        return ""
    delta = datetime.now() - datetime.fromtimestamp(ts)
    h = int(delta.total_seconds() // 3600)
    if h < 1:   return f"{int(delta.total_seconds()//60)}m ago"
    if h < 24:  return f"{h}h ago"
    return f"{h//24}d ago"


# ── Dashboard HTML ────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Portfolio Risk Tracker</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-filler@0.6.0/src/index.js"></script>
<style>
  :root {
    --bg:       #080b14;
    --surface:  #0d1120;
    --card:     #111827;
    --border:   #1f2a3d;
    --gold:     #e8a020;
    --gold-dim: rgba(232,160,32,0.15);
    --green:    #22c55e;
    --red:      #ef4444;
    --blue:     #3b82f6;
    --muted:    #6b7280;
    --text:     #e2e8f0;
    --text2:    #94a3b8;
    --radius:   10px;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
    font-size: 13px;
    min-height: 100vh;
  }

  /* ── Nav ── */
  nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 24px; height: 52px;
    background: var(--surface); border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 100;
  }
  .nav-brand { font-size: 15px; font-weight: 700; color: var(--gold); letter-spacing: .5px; }
  .nav-tabs { display: flex; gap: 4px; }
  .nav-tab {
    padding: 6px 16px; border-radius: 6px; cursor: pointer;
    color: var(--text2); font-weight: 500; font-size: 12px;
    border: none; background: none; transition: all .15s;
  }
  .nav-tab.active, .nav-tab:hover { background: var(--border); color: var(--text); }
  .nav-right { display: flex; align-items: center; gap: 16px; }
  #clock { color: var(--text2); font-size: 12px; }
  .badge {
    padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: 600;
    background: rgba(34,197,94,.12); color: var(--green);
  }
  .badge.red { background: rgba(239,68,68,.12); color: var(--red); }

  /* ── Layout ── */
  .page { display: grid; grid-template-columns: 1fr 320px; gap: 16px; padding: 16px 20px; max-width: 1600px; margin: 0 auto; }

  /* ── Metric bar ── */
  .metric-bar {
    grid-column: 1 / -1;
    display: grid; grid-template-columns: 2fr repeat(4, 1fr);
    gap: 12px;
  }
  .metric-card {
    background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 16px 20px;
  }
  .metric-card.primary { border-color: var(--gold); }
  .metric-label { color: var(--text2); font-size: 11px; text-transform: uppercase; letter-spacing: .8px; margin-bottom: 6px; }
  .metric-value { font-size: 28px; font-weight: 700; color: var(--text); line-height: 1; }
  .metric-card.primary .metric-value { color: var(--gold); }
  .metric-sub { margin-top: 6px; font-size: 12px; }
  .metric-card:not(.primary) .metric-value { font-size: 20px; }

  /* ── Cards ── */
  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 16px 20px;
  }
  .card-title {
    font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .8px; color: var(--text2); margin-bottom: 14px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .card-title span { color: var(--gold); font-size: 10px; }

  /* ── Chart ── */
  .chart-wrap { position: relative; height: 220px; }

  /* ── Holdings table ── */
  .tbl-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  thead th {
    color: var(--text2); font-weight: 600; text-transform: uppercase;
    font-size: 10px; letter-spacing: .6px;
    padding: 8px 12px; border-bottom: 1px solid var(--border);
    text-align: right; white-space: nowrap;
  }
  thead th:first-child { text-align: left; }
  tbody tr { border-bottom: 1px solid rgba(31,42,61,.5); transition: background .1s; }
  tbody tr:hover { background: rgba(255,255,255,.03); }
  tbody td { padding: 9px 12px; text-align: right; white-space: nowrap; }
  tbody td:first-child { text-align: left; font-weight: 600; }
  .ticker-cell { display: flex; align-items: center; gap: 8px; }
  .ticker-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--gold); flex-shrink: 0; }
  tfoot td { padding: 10px 12px; text-align: right; font-weight: 700; border-top: 1px solid var(--border); color: var(--text2); }
  tfoot td:first-child { text-align: left; }

  /* ── Mini bar chart in table ── */
  .bar-wrap { display: flex; align-items: center; gap: 6px; justify-content: flex-end; }
  .bar-bg { width: 60px; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 2px; background: var(--gold); }
  .bar-fill.neg { background: var(--red); }

  /* ── Risk metrics ── */
  .metric-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 0; border-bottom: 1px solid rgba(31,42,61,.5);
  }
  .metric-row:last-child { border-bottom: none; }
  .metric-row-label { color: var(--text2); }
  .metric-row-val { font-weight: 600; font-size: 13px; }
  .pill {
    padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;
  }
  .pill.green { background: rgba(34,197,94,.12); color: var(--green); }
  .pill.yellow { background: rgba(234,179,8,.12); color: #eab308; }
  .pill.red { background: rgba(239,68,68,.12); color: var(--red); }
  .pill.blue { background: rgba(59,130,246,.12); color: var(--blue); }

  /* ── Right sidebar ── */
  .sidebar { display: flex; flex-direction: column; gap: 12px; }

  /* ── News ── */
  .news-item {
    padding: 10px 0; border-bottom: 1px solid rgba(31,42,61,.5);
    display: flex; flex-direction: column; gap: 4px;
  }
  .news-item:last-child { border-bottom: none; }
  .news-title { color: var(--text); font-size: 12px; line-height: 1.4; }
  .news-title a { color: inherit; text-decoration: none; }
  .news-title a:hover { color: var(--gold); }
  .news-meta { display: flex; gap: 8px; font-size: 11px; color: var(--text2); }
  .news-ticker { color: var(--gold); font-weight: 600; }

  /* ── Colours ── */
  .green { color: var(--green); }
  .red   { color: var(--red); }
  .gold  { color: var(--gold); }
  .muted { color: var(--text2); }

  /* ── Spinner ── */
  .spinner {
    display: flex; align-items: center; justify-content: center;
    height: 200px; color: var(--text2); gap: 10px;
  }
  .spin {
    width: 18px; height: 18px; border: 2px solid var(--border);
    border-top-color: var(--gold); border-radius: 50%;
    animation: spin .7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Refresh indicator ── */
  #refresh-dot {
    width: 6px; height: 6px; border-radius: 50%; background: var(--green);
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .3; } }
  .refresh-row { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--text2); }

  /* ── Provider badge ── */
  .provider-badge {
    font-size: 10px; padding: 2px 7px; border-radius: 4px;
    background: rgba(59,130,246,.1); color: var(--blue); border: 1px solid rgba(59,130,246,.2);
  }

  /* ── Responsive ── */
  @media (max-width: 1000px) {
    .page { grid-template-columns: 1fr; }
    .metric-bar { grid-template-columns: 1fr 1fr; }
  }
</style>
</head>
<body>

<!-- Nav -->
<nav>
  <div class="nav-brand">&#9650; Portfolio Risk Tracker</div>
  <div class="nav-tabs">
    <button class="nav-tab active">Dashboard</button>
    <button class="nav-tab" onclick="window.open('https://pypi.org/project/portfolio-risk-tracker/','_blank')">PyPI</button>
    <button class="nav-tab" onclick="window.open('https://github.com/sofiacieplinski/portfolio-risk-tracker','_blank')">GitHub</button>
  </div>
  <div class="nav-right">
    <div class="refresh-row"><div id="refresh-dot"></div><span id="last-updated">Loading…</span></div>
    <div id="clock"></div>
  </div>
</nav>

<!-- Main -->
<div class="page">

  <!-- Metric bar -->
  <div class="metric-bar" id="metric-bar">
    <div class="metric-card primary">
      <div class="metric-label">Total Portfolio Value</div>
      <div class="metric-value" id="total-value">—</div>
      <div class="metric-sub" id="day-change-row">—</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Annual Return</div>
      <div class="metric-value" id="ann-return">—</div>
      <div class="metric-sub muted" id="bench-return">vs benchmark —</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Active Return</div>
      <div class="metric-value" id="active-return">—</div>
      <div class="metric-sub muted">vs <span id="bench-name">SPY</span></div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Sharpe / Sortino</div>
      <div class="metric-value" id="sharpe-val">—</div>
      <div class="metric-sub muted" id="sortino-val">Sortino —</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Holdings</div>
      <div class="metric-value" id="n-holdings">—</div>
      <div class="metric-sub muted" id="data-window">— trading days</div>
    </div>
  </div>

  <!-- Left column -->
  <div style="display:flex;flex-direction:column;gap:12px;">

    <!-- Chart -->
    <div class="card">
      <div class="card-title">
        <div style="display:flex;align-items:center;gap:12px;">
          Equity Curve
          <span style="display:flex;align-items:center;gap:6px;font-size:11px;font-weight:400;text-transform:none;letter-spacing:0;">
            <span style="display:inline-block;width:20px;height:2px;background:var(--gold);border-radius:1px;"></span>Portfolio
            <span style="display:inline-block;width:20px;height:2px;background:#4b5563;border-radius:1px;margin-left:6px;"></span>Benchmark
          </span>
        </div>
        <span id="provider-badge" class="provider-badge">yfinance</span>
      </div>
      <div class="chart-wrap">
        <canvas id="equity-chart"></canvas>
      </div>
    </div>

    <!-- Holdings table -->
    <div class="card">
      <div class="card-title">Holdings</div>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Shares</th>
              <th>Avg Cost</th>
              <th>Price</th>
              <th>Mkt Value</th>
              <th>Gain / Loss</th>
              <th>Day Chg</th>
              <th>Weight</th>
              <th>Period Ret</th>
              <th>Contribution</th>
            </tr>
          </thead>
          <tbody id="holdings-body">
            <tr><td colspan="10"><div class="spinner"><div class="spin"></div>Fetching market data…</div></td></tr>
          </tbody>
          <tfoot id="holdings-foot"></tfoot>
        </table>
      </div>
    </div>

  </div>

  <!-- Right sidebar -->
  <div class="sidebar">

    <!-- Risk metrics -->
    <div class="card">
      <div class="card-title">Risk Metrics <span>1Y Annualised</span></div>
      <div id="risk-metrics">
        <div class="spinner" style="height:120px;"><div class="spin"></div></div>
      </div>
    </div>

    <!-- Return attribution mini chart -->
    <div class="card">
      <div class="card-title">Return Attribution</div>
      <div style="position:relative;height:180px;">
        <canvas id="attr-chart"></canvas>
      </div>
    </div>

    <!-- News -->
    <div class="card">
      <div class="card-title">Market News</div>
      <div id="news-feed">
        <div class="spinner" style="height:80px;"><div class="spin"></div></div>
      </div>
    </div>

  </div>

</div>

<script>
// ── Helpers ──────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

function fmt(n, decimals=2) {
  return n == null ? '—' : n.toLocaleString('en-US', {minimumFractionDigits: decimals, maximumFractionDigits: decimals});
}
function fmtUSD(n) {
  if (n == null) return '—';
  const neg = n < 0;
  return (neg ? '-' : '') + '$' + Math.abs(n).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
}
function fmtPct(n, plusSign=true) {
  if (n == null) return '—';
  const sign = n > 0 && plusSign ? '+' : '';
  return `${sign}${fmt(n)}%`;
}
function colorClass(n) { return n >= 0 ? 'green' : 'red'; }
function colored(n, fmt_fn) {
  const cls = colorClass(n);
  return `<span class="${cls}">${fmt_fn(n)}</span>`;
}

// ── Clock ────────────────────────────────────────────────────────────────────

function updateClock() {
  const now = new Date();
  $('clock').textContent = now.toLocaleTimeString('en-US', {hour:'2-digit', minute:'2-digit', second:'2-digit'});
}
setInterval(updateClock, 1000);
updateClock();

// ── Chart instances ──────────────────────────────────────────────────────────

let equityChart = null;
let attrChart   = null;

function buildEquityChart(dates, portfolio, benchmark) {
  const ctx = $('equity-chart').getContext('2d');
  if (equityChart) equityChart.destroy();

  const grad = ctx.createLinearGradient(0, 0, 0, 220);
  grad.addColorStop(0,   'rgba(232,160,32,0.25)');
  grad.addColorStop(1,   'rgba(232,160,32,0)');

  equityChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [
        {
          label: 'Portfolio',
          data: portfolio,
          borderColor: '#e8a020',
          borderWidth: 2,
          backgroundColor: grad,
          fill: true,
          pointRadius: 0,
          tension: 0.3,
        },
        {
          label: 'Benchmark',
          data: benchmark,
          borderColor: '#374151',
          borderWidth: 1.5,
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          tension: 0.3,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1e293b',
          borderColor: '#334155',
          borderWidth: 1,
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${fmt(ctx.raw)} (base 100)`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(31,42,61,0.6)' },
          ticks: {
            color: '#6b7280', maxTicksLimit: 8, font: { size: 10 },
            callback: (val, idx) => {
              const d = dates[idx];
              return d ? d.slice(5) : '';
            }
          }
        },
        y: {
          position: 'right',
          grid: { color: 'rgba(31,42,61,0.6)' },
          ticks: { color: '#6b7280', font: { size: 10 }, callback: v => fmt(v) }
        }
      }
    }
  });
}

function buildAttrChart(attribution) {
  const ctx = $('attr-chart').getContext('2d');
  if (attrChart) attrChart.destroy();

  const top = attribution.slice(0, 8);
  const labels = top.map(h => h.ticker);
  const values = top.map(h => h.contribution);
  const colors = values.map(v => v >= 0 ? 'rgba(34,197,94,0.75)' : 'rgba(239,68,68,0.75)');

  attrChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderRadius: 3,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1e293b',
          callbacks: { label: ctx => ` ${fmtPct(ctx.raw)} contribution` }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#6b7280', font: { size: 10 } } },
        y: {
          grid: { color: 'rgba(31,42,61,0.6)' },
          ticks: { color: '#6b7280', font: { size: 10 }, callback: v => fmtPct(v) }
        }
      }
    }
  });
}

// ── Render helpers ───────────────────────────────────────────────────────────

function renderMetricBar(d) {
  $('total-value').textContent = fmtUSD(d.total_value);
  const daySign = d.day_change >= 0 ? '+' : '';
  $('day-change-row').innerHTML =
    `<span class="${colorClass(d.day_change)}">${daySign}${fmtUSD(d.day_change)} (${fmtPct(d.day_change_pct)}) today</span>`;
  $('ann-return').innerHTML = `<span class="${colorClass(d.port_ann_return)}">${fmtPct(d.port_ann_return)}</span>`;
  $('bench-return').textContent = `vs ${d.benchmark} ${fmtPct(d.bench_ann_return)}`;
  $('active-return').innerHTML = `<span class="${colorClass(d.active_return)}">${fmtPct(d.active_return)}</span>`;
  $('bench-name').textContent = d.benchmark;
  $('sharpe-val').innerHTML = `<span class="${d.sharpe >= 1 ? 'green' : 'red'}">${fmt(d.sharpe, 2)}</span>`;
  $('sortino-val').textContent = `Sortino ${fmt(d.sortino, 2)}`;
  $('n-holdings').textContent = d.attribution.length;
  $('data-window').textContent = `${d.n_days} trading days`;
  $('provider-badge').textContent = d.provider;
}

function renderHoldings(attribution) {
  const maxContrib = Math.max(...attribution.map(h => Math.abs(h.contribution)), 1);
  let rows = '';
  let totalVal = 0, totalGL = 0;
  for (const h of attribution) {
    totalVal += h.market_value;
    totalGL  += h.gain_loss;
    const barW  = Math.round(Math.abs(h.contribution) / maxContrib * 100);
    const barCls= h.contribution >= 0 ? '' : ' neg';
    rows += `<tr>
      <td><div class="ticker-cell"><div class="ticker-dot"></div>${h.ticker}</div></td>
      <td>${fmt(h.shares, 0)}</td>
      <td>${fmtUSD(h.avg_cost)}</td>
      <td>${fmtUSD(h.price)}</td>
      <td>${fmtUSD(h.market_value)}</td>
      <td class="${colorClass(h.gain_loss)}">${h.gain_loss >= 0 ? '+' : ''}${fmtUSD(h.gain_loss)} <span class="muted">(${fmtPct(h.gain_loss_pct)})</span></td>
      <td class="${colorClass(h.day_change_pct)}">${fmtPct(h.day_change_pct)}</td>
      <td>${fmt(h.weight, 1)}%</td>
      <td class="${colorClass(h.period_return)}">${fmtPct(h.period_return)}</td>
      <td>
        <div class="bar-wrap">
          <span class="${colorClass(h.contribution)}">${fmtPct(h.contribution)}</span>
          <div class="bar-bg"><div class="bar-fill${barCls}" style="width:${barW}%"></div></div>
        </div>
      </td>
    </tr>`;
  }
  $('holdings-body').innerHTML = rows;
  $('holdings-foot').innerHTML = `<tr>
    <td colspan="4">TOTAL</td>
    <td>${fmtUSD(totalVal)}</td>
    <td class="${colorClass(totalGL)}">${totalGL >= 0 ? '+' : ''}${fmtUSD(totalGL)}</td>
    <td colspan="4"></td>
  </tr>`;
}

function pillClass(val, thresholds) {
  if (val >= thresholds[0]) return 'green';
  if (val >= thresholds[1]) return 'yellow';
  return 'red';
}

function renderRiskMetrics(d) {
  const rows = [
    ['Beta',           fmt(d.beta, 3),          d.beta < 1 ? 'green' : d.beta < 1.3 ? 'yellow' : 'red'],
    ['Sharpe Ratio',   fmt(d.sharpe, 3),         pillClass(d.sharpe, [2, 1])],
    ['Sortino Ratio',  fmt(d.sortino, 3),        pillClass(d.sortino, [2, 1])],
    ['Info Ratio',     fmt(d.info_ratio, 3),     pillClass(d.info_ratio, [2, 1])],
    ['Alpha (CAPM)',   fmtPct(d.alpha),          d.alpha >= 0 ? 'green' : 'red'],
    ['Tracking Error', fmtPct(d.tracking_error), d.tracking_error < 5 ? 'green' : d.tracking_error < 10 ? 'yellow' : 'red'],
    ['VaR 95% (day)',  fmtPct(d.var_95),         d.var_95 > -1.5 ? 'green' : d.var_95 > -2.5 ? 'yellow' : 'red'],
    ['Max Drawdown',   fmtPct(d.max_drawdown),   d.max_drawdown > -10 ? 'green' : d.max_drawdown > -20 ? 'yellow' : 'red'],
  ];
  $('risk-metrics').innerHTML = rows.map(([label, val, cls]) =>
    `<div class="metric-row">
       <span class="metric-row-label">${label}</span>
       <span class="pill ${cls}">${val}</span>
     </div>`
  ).join('');
}

function renderNews(articles) {
  if (!articles.length) {
    $('news-feed').innerHTML = '<div class="muted" style="font-size:12px;padding:8px 0;">No recent news.</div>';
    return;
  }
  $('news-feed').innerHTML = articles.map(a =>
    `<div class="news-item">
       <div class="news-title"><a href="${a.link}" target="_blank">${a.title}</a></div>
       <div class="news-meta">
         <span class="news-ticker">${a.ticker}</span>
         <span>${a.publisher}</span>
         <span>${a.age}</span>
       </div>
     </div>`
  ).join('');
}

// ── Main data fetch ──────────────────────────────────────────────────────────

async function loadData() {
  $('last-updated').textContent = 'Refreshing…';
  try {
    const res  = await fetch('/api/data');
    if (!res.ok) throw new Error(await res.text());
    const d    = await res.json();

    renderMetricBar(d);
    renderHoldings(d.attribution);
    renderRiskMetrics(d);
    buildEquityChart(d.chart.dates, d.chart.portfolio, d.chart.benchmark);
    buildAttrChart(d.attribution);

    $('last-updated').textContent = 'Updated ' + new Date().toLocaleTimeString();
  } catch(e) {
    $('last-updated').textContent = 'Error — ' + e.message;
  }
}

async function loadNews() {
  try {
    const articles = await (await fetch('/api/news')).json();
    renderNews(articles);
  } catch { /* silent */ }
}

// ── Boot ─────────────────────────────────────────────────────────────────────

loadData();
loadNews();
setInterval(loadData, 60_000);   // refresh every 60s
setInterval(loadNews, 300_000);  // refresh news every 5 min
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


# ── CLI entry ─────────────────────────────────────────────────────────────────

def serve(host: str = "127.0.0.1", port: int = 5000, open_browser: bool = True) -> None:
    if open_browser:
        import threading
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
