#!/usr/bin/env python3
from __future__ import annotations
"""
Portfolio Risk Metrics Tracker
Terminal-based dashboard for portfolio risk analysis using Rich TUI.

Commands:
  add <TICKER> <SHARES> <AVG_COST>   Add or average into a position
  remove <TICKER>                     Remove a position
  list                                List current holdings
  dashboard                           Full risk metrics dashboard
  set-benchmark <TICKER>              Set benchmark (default: SPY)
  set-risk-free <RATE>                Set annual risk-free rate (e.g. 0.05)
  set-key <PROVIDER> <KEY>            Store an API key (polygon, finnhub, plaid-*)
  connect                             Connect a brokerage via Plaid
  sync                                Sync holdings from connected brokerage
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box
from rich.align import Align
from rich.text import Text
from rich.rule import Rule

from .providers import get_prices
from .config import get_key, set_key as _set_key, configured_providers

# ─── Config ───────────────────────────────────────────────────────────────────

DATA_DIR      = Path.home() / ".portfolio_tracker"
HOLDINGS_FILE = DATA_DIR / "holdings.json"
PERIOD = "1y"       # lookback window for all metrics
ANN = 252           # trading days per year

console = Console()

# ─── Persistence ──────────────────────────────────────────────────────────────

def load_data() -> dict:
    # Migrate from legacy local-directory location on first run
    legacy = Path.cwd() / "holdings.json"
    if not HOLDINGS_FILE.exists() and legacy.exists():
        import shutil
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(legacy, HOLDINGS_FILE)
        console.print(f"[dim]Migrated holdings → {HOLDINGS_FILE}[/dim]")

    if HOLDINGS_FILE.exists():
        with open(HOLDINGS_FILE) as f:
            return json.load(f)
    return {"holdings": [], "benchmark": "SPY", "risk_free_rate": 0.05}


def save_data(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HOLDINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ─── Risk Metrics ─────────────────────────────────────────────────────────────

def compute_metrics(data: dict) -> Optional[dict]:
    holdings = data["holdings"]
    if not holdings:
        return None

    benchmark: str = data.get("benchmark", "SPY")
    rf_rate: float = data.get("risk_free_rate", 0.05)
    rf_daily: float = (1 + rf_rate) ** (1 / ANN) - 1

    tickers = [h["ticker"] for h in holdings]
    all_tickers = list(dict.fromkeys(tickers + [benchmark]))

    provider_name = "unknown"
    with console.status("[bold cyan]Fetching market data…[/bold cyan]", spinner="dots2"):
        try:
            prices, provider_name = get_prices(all_tickers, PERIOD)
        except Exception as exc:
            console.print(f"[red]Failed to fetch data:[/red] {exc}")
            sys.exit(1)

    # Validate all requested tickers are present and have data
    missing = [t for t in all_tickers if t not in prices.columns or prices[t].dropna().empty]
    if missing:
        console.print(f"[red]Could not retrieve data for:[/red] {', '.join(missing)}")
        sys.exit(1)

    returns: pd.DataFrame = prices.pct_change(fill_method=None).dropna()

    # ── Portfolio weights (by current market value) ──────────────────────────
    last = prices.iloc[-1]
    port_vals: dict[str, float] = {}
    total_val = 0.0
    for h in holdings:
        t = h["ticker"]
        mv = float(last[t]) * h["shares"]
        port_vals[t] = mv
        total_val += mv

    weights: dict[str, float] = {t: v / total_val for t, v in port_vals.items()}

    # ── Portfolio daily return series ────────────────────────────────────────
    w_arr = np.array([weights.get(t, 0.0) for t in tickers])
    port_ret_arr = returns[tickers].values @ w_arr
    port_ret = pd.Series(port_ret_arr, index=returns.index)
    bench_ret = returns[benchmark]

    idx = port_ret.index.intersection(bench_ret.index)
    pr = port_ret[idx]
    br = bench_ret[idx]

    # ── Core metrics ─────────────────────────────────────────────────────────

    # Beta
    cov_mat = np.cov(pr.values, br.values)
    beta = float(cov_mat[0, 1] / cov_mat[1, 1])

    # Annualised returns
    port_ann = float((1 + pr.mean()) ** ANN - 1)
    bench_ann = float((1 + br.mean()) ** ANN - 1)

    # Sharpe ratio
    sharpe = float(((pr - rf_daily).mean() / pr.std()) * np.sqrt(ANN))

    # Sortino ratio  (downside deviation vs rf)
    downside = pr[pr < rf_daily] - rf_daily
    if len(downside) > 1:
        down_std = float(np.sqrt((downside ** 2).mean()) * np.sqrt(ANN))
    else:
        down_std = float(pr.std() * np.sqrt(ANN))
    sortino = (port_ann - rf_rate) / down_std if down_std > 1e-10 else 0.0

    # Tracking error (annualised std of active returns)
    active = pr - br
    tracking_error = float(active.std() * np.sqrt(ANN))

    # VaR 95% (historical, daily)
    var_95 = float(np.percentile(pr.values, 5))

    # Max drawdown
    cum = (1 + pr).cumprod()
    roll_max = cum.cummax()
    drawdowns = (cum - roll_max) / roll_max
    max_dd = float(drawdowns.min())

    # Jensen's alpha
    alpha = port_ann - (rf_rate + beta * (bench_ann - rf_rate))

    # Information ratio
    info_ratio = float(active.mean() / active.std() * np.sqrt(ANN)) if active.std() > 0 else 0.0

    # ── Return attribution per holding ────────────────────────────────────────
    attribution: dict[str, dict] = {}
    for h in holdings:
        t = h["ticker"]
        if t not in returns.columns:
            continue
        period_ret = float((1 + returns[t]).prod() - 1)
        w = weights.get(t, 0.0)
        cur_price = float(last[t])
        gl = (cur_price - h["avg_cost"]) * h["shares"]
        gl_pct = (cur_price - h["avg_cost"]) / h["avg_cost"]
        attribution[t] = {
            "weight": w,
            "period_return": period_ret,
            "contribution": w * period_ret,
            "current_price": cur_price,
            "avg_cost": h["avg_cost"],
            "shares": h["shares"],
            "market_value": port_vals[t],
            "gain_loss": gl,
            "gain_loss_pct": gl_pct,
        }

    return {
        "beta": beta,
        "sharpe": sharpe,
        "sortino": sortino,
        "tracking_error": tracking_error,
        "var_95": var_95,
        "max_drawdown": max_dd,
        "alpha": alpha,
        "info_ratio": info_ratio,
        "port_ann_return": port_ann,
        "bench_ann_return": bench_ann,
        "total_value": total_val,
        "attribution": attribution,
        "benchmark": benchmark,
        "rf_rate": rf_rate,
        "n_days": len(pr),
        "provider": provider_name,
    }

# ─── Rich Display Helpers ──────────────────────────────────────────────────────

def _pct(val: float, invert: bool = False) -> str:
    """Colour-code a percentage value. invert=True makes negatives green (e.g. VaR)."""
    pos_col, neg_col = ("green", "red") if not invert else ("red", "green")
    sign = "+" if val > 0 else ""
    col = pos_col if val > 0 else (neg_col if val < 0 else "white")
    return f"[{col}]{sign}{val*100:.2f}%[/{col}]"


def _ratio(val: float) -> str:
    if val >= 2.0:
        col = "green"
    elif val >= 1.0:
        col = "yellow"
    else:
        col = "red"
    sign = "+" if val > 0 else ""
    return f"[{col}]{sign}{val:.3f}[/{col}]"


def _beta(b: float) -> str:
    if b > 1.3:
        col = "red"
    elif b > 1.0:
        col = "yellow"
    elif b < 0:
        col = "magenta"
    else:
        col = "green"
    return f"[{col}]{b:.3f}[/{col}]"


def _money(val: float) -> str:
    col = "green" if val >= 0 else "red"
    sign = "+" if val > 0 else ""
    return f"[{col}]{sign}${val:,.2f}[/{col}]"


# ─── Dashboard Panels ─────────────────────────────────────────────────────────

def _summary_panel(m: dict) -> Panel:
    active = m["port_ann_return"] - m["bench_ann_return"]
    lines = [
        f"  [dim]Total Value[/dim]          [bold white]${m['total_value']:>14,.2f}[/bold white]",
        f"  [dim]Holdings[/dim]             [white]{len(m['attribution']):>14}[/white]",
        f"  [dim]Benchmark[/dim]            [cyan]{m['benchmark']:>14}[/cyan]",
        f"  [dim]Risk-Free Rate[/dim]       [white]{m['rf_rate']*100:>13.2f}%[/white]",
        f"  [dim]Data Window[/dim]          [white]{m['n_days']:>11} days[/white]",
        "",
        f"  [dim]Portfolio Return[/dim]     {_pct(m['port_ann_return']):>24}",
        f"  [dim]Benchmark Return[/dim]     {_pct(m['bench_ann_return']):>24}",
        f"  [dim]Active Return[/dim]        {_pct(active):>24}",
        "",
        f"  [dim]As of[/dim]               [white]{datetime.now().strftime('%Y-%m-%d  %H:%M'):>14}[/white]",
    ]
    return Panel(
        "\n".join(lines),
        title="[bold cyan] Portfolio Summary [/bold cyan]",
        border_style="cyan",
        padding=(1, 1),
    )


def _metrics_panel(m: dict) -> Panel:
    rows = [
        ("Beta",                _beta(m["beta"])),
        ("Sharpe Ratio",        _ratio(m["sharpe"])),
        ("Sortino Ratio",       _ratio(m["sortino"])),
        ("Information Ratio",   _ratio(m["info_ratio"])),
        ("Alpha  (CAPM)",       _pct(m["alpha"])),
        ("Tracking Error",      _pct(m["tracking_error"], invert=True)),
        ("VaR 95%  (daily)",    _pct(m["var_95"],  invert=True)),
        ("Max Drawdown",        _pct(m["max_drawdown"], invert=True)),
    ]
    lines = [f"  [dim]{name:<22}[/dim]  {val}" for name, val in rows]
    # Interpretation key
    lines += [
        "",
        "  [dim]────────────────────────────────[/dim]",
        "  [dim]Sharpe / Sortino / IR: ≥2 [green]●[/green] ≥1 [yellow]●[/yellow] <1 [red]●[/red][/dim]",
        "  [dim]Beta: <1 [green]●[/green]  1–1.3 [yellow]●[/yellow]  >1.3 [red]●[/red][/dim]",
    ]
    return Panel(
        "\n".join(lines),
        title="[bold magenta] Risk Metrics (1Y) [/bold magenta]",
        border_style="magenta",
        padding=(1, 1),
    )


def _holdings_table(m: dict) -> Table:
    t = Table(
        title="[bold blue] Holdings [/bold blue]",
        box=box.SIMPLE_HEAVY,
        border_style="blue",
        header_style="bold blue",
        show_lines=True,
        title_justify="left",
        min_width=88,
    )
    t.add_column("Ticker",       style="bold white",  justify="left",  min_width=8)
    t.add_column("Shares",                            justify="right", min_width=10)
    t.add_column("Avg Cost",                          justify="right", min_width=10)
    t.add_column("Price",                             justify="right", min_width=10)
    t.add_column("Market Value",                      justify="right", min_width=13)
    t.add_column("Gain / Loss $",                     justify="right", min_width=14)
    t.add_column("Gain / Loss %",                     justify="right", min_width=13)
    t.add_column("Weight",                            justify="right", min_width=8)

    total_gl = 0.0
    for ticker, a in sorted(m["attribution"].items(), key=lambda x: -x[1]["market_value"]):
        gl = a["gain_loss"]
        gl_pct = a["gain_loss_pct"]
        total_gl += gl
        gl_str  = _money(gl)
        glp_str = _pct(gl_pct)

        t.add_row(
            ticker,
            f"{a['shares']:,.2f}",
            f"${a['avg_cost']:,.2f}",
            f"${a['current_price']:,.2f}",
            f"${a['market_value']:,.2f}",
            gl_str,
            glp_str,
            f"{a['weight']*100:.1f}%",
        )

    t.add_section()
    t.add_row(
        "[bold]TOTAL[/bold]",
        "", "", "",
        f"[bold]${m['total_value']:,.2f}[/bold]",
        _money(total_gl),
        "", "",
    )
    return t


def _attribution_table(m: dict) -> Table:
    t = Table(
        title="[bold green] Return Attribution (1Y) [/bold green]",
        box=box.SIMPLE_HEAVY,
        border_style="green",
        header_style="bold green",
        show_lines=True,
        title_justify="left",
        min_width=72,
    )
    t.add_column("Ticker",         style="bold white", justify="left",  min_width=8)
    t.add_column("Weight",                             justify="right", min_width=8)
    t.add_column("Ticker Return",                      justify="right", min_width=14)
    t.add_column("Contribution",                       justify="right", min_width=13)
    t.add_column("Bar",                                justify="left",  min_width=24)

    items = sorted(m["attribution"].items(), key=lambda x: -abs(x[1]["contribution"]))
    max_c = max((abs(a["contribution"]) for _, a in items), default=1.0) or 1.0

    total_contribution = 0.0
    for ticker, a in items:
        pr  = a["period_return"]
        ct  = a["contribution"]
        total_contribution += ct
        bar_len = max(1, int(abs(ct) / max_c * 22))
        bar_col = "green" if ct >= 0 else "red"
        bar = f"[{bar_col}]{'█' * bar_len}[/{bar_col}]"

        t.add_row(
            ticker,
            f"{a['weight']*100:.1f}%",
            _pct(pr),
            _pct(ct),
            bar,
        )

    t.add_section()
    t.add_row(
        "[bold]TOTAL[/bold]",
        "100.0%",
        _pct(m["port_ann_return"]),
        _pct(total_contribution),
        "",
    )
    return t


def show_dashboard(m: dict) -> None:
    console.print()
    console.print(
        Align.center(
            Text(
                "  PORTFOLIO RISK METRICS DASHBOARD  ",
                style="bold white on dark_blue",
            )
        )
    )
    console.print()
    console.print(Columns([_summary_panel(m), _metrics_panel(m)], equal=True, expand=True))
    console.print()
    console.print(_holdings_table(m))
    console.print()
    console.print(_attribution_table(m))
    console.print()
    console.print(Rule(style="dim"))
    console.print(
        f"[dim]  All return metrics annualised. Benchmark: [cyan]{m['benchmark']}[/cyan]. "
        f"Risk-free rate: {m['rf_rate']*100:.2f}%. "
        f"Data: {PERIOD} ({m['n_days']} trading days). "
        f"Provider: [cyan]{m['provider']}[/cyan][/dim]"
    )
    console.print()

# ─── CLI Handlers ─────────────────────────────────────────────────────────────

def cmd_add(args: argparse.Namespace) -> None:
    data = load_data()
    ticker = args.ticker.upper()
    new_shares = float(args.shares)
    new_cost   = float(args.avg_cost)

    if new_shares <= 0:
        console.print("[red]Shares must be positive.[/red]")
        sys.exit(1)
    if new_cost <= 0:
        console.print("[red]Average cost must be positive.[/red]")
        sys.exit(1)

    for h in data["holdings"]:
        if h["ticker"] == ticker:
            # Average into existing position
            old_value = h["shares"] * h["avg_cost"]
            new_value = new_shares * new_cost
            h["shares"]   += new_shares
            h["avg_cost"]  = (old_value + new_value) / h["shares"]
            save_data(data)
            console.print(
                f"[green]Updated[/green] [bold]{ticker}[/bold] — "
                f"{h['shares']:,.2f} shares @ [bold]${h['avg_cost']:,.2f}[/bold] avg cost"
            )
            return

    data["holdings"].append({"ticker": ticker, "shares": new_shares, "avg_cost": new_cost})
    save_data(data)
    console.print(
        f"[green]Added[/green] [bold]{ticker}[/bold] — "
        f"{new_shares:,.2f} shares @ [bold]${new_cost:,.2f}[/bold]"
    )


def cmd_remove(args: argparse.Namespace) -> None:
    data = load_data()
    ticker = args.ticker.upper()
    before = len(data["holdings"])
    data["holdings"] = [h for h in data["holdings"] if h["ticker"] != ticker]
    if len(data["holdings"]) < before:
        save_data(data)
        console.print(f"[yellow]Removed[/yellow] [bold]{ticker}[/bold]")
    else:
        console.print(f"[red]Not found:[/red] {ticker}")
        sys.exit(1)


def cmd_list(args: argparse.Namespace) -> None:
    data = load_data()
    if not data["holdings"]:
        console.print("[yellow]No holdings. Use [bold]add[/bold] to add positions.[/yellow]")
        return

    t = Table(
        title="Current Holdings",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
    )
    t.add_column("Ticker",    style="bold white", justify="left")
    t.add_column("Shares",                        justify="right")
    t.add_column("Avg Cost",                      justify="right")
    t.add_column("Book Value",                    justify="right")

    total = 0.0
    for h in sorted(data["holdings"], key=lambda x: x["ticker"]):
        bv = h["shares"] * h["avg_cost"]
        total += bv
        t.add_row(h["ticker"], f"{h['shares']:,.2f}", f"${h['avg_cost']:,.2f}", f"${bv:,.2f}")

    t.add_section()
    t.add_row("[bold]TOTAL[/bold]", "", "", f"[bold]${total:,.2f}[/bold]")

    console.print()
    console.print(t)
    console.print(
        f"\n[dim]Benchmark: [cyan]{data['benchmark']}[/cyan]  |  "
        f"Risk-free rate: {data['risk_free_rate']*100:.2f}%[/dim]"
    )
    console.print()


def cmd_dashboard(args: argparse.Namespace) -> None:
    data = load_data()
    if not data["holdings"]:
        console.print(
            "[yellow]No holdings. Add positions first:[/yellow]\n"
            "  [bold]python tracker.py add AAPL 10 175.00[/bold]"
        )
        return

    metrics = compute_metrics(data)
    if metrics is None:
        console.print("[red]Could not compute metrics.[/red]")
        sys.exit(1)

    show_dashboard(metrics)


def cmd_set_benchmark(args: argparse.Namespace) -> None:
    data = load_data()
    data["benchmark"] = args.ticker.upper()
    save_data(data)
    console.print(f"[green]Benchmark set to[/green] [bold cyan]{data['benchmark']}[/bold cyan]")


def cmd_set_rf(args: argparse.Namespace) -> None:
    rate = float(args.rate)
    if not 0 <= rate <= 1:
        console.print("[red]Rate should be a decimal between 0 and 1 (e.g. 0.05 for 5%).[/red]")
        sys.exit(1)
    data = load_data()
    data["risk_free_rate"] = rate
    save_data(data)
    console.print(f"[green]Risk-free rate set to[/green] [bold]{rate*100:.2f}%[/bold]")


def cmd_set_key(args: argparse.Namespace) -> None:
    valid = ("polygon", "finnhub", "plaid-client-id", "plaid-secret", "plaid-env")
    name = args.name.lower()
    if name not in valid:
        console.print(f"[red]Unknown key:[/red] {name}")
        console.print(f"[dim]Valid options: {', '.join(valid)}[/dim]")
        sys.exit(1)
    _set_key(name, args.value)
    masked = args.value[:6] + "…" if len(args.value) > 6 else args.value
    console.print(f"[green]Saved[/green] [bold]{name}[/bold] = {masked}")

    providers = configured_providers()
    console.print(f"[dim]Active data providers: {' → '.join(providers)}[/dim]")


def cmd_connect(args: argparse.Namespace) -> None:
    from .brokerage import connect
    try:
        _, institution = connect()
        console.print(f"\n[green]Connected to[/green] [bold]{institution}[/bold]")
        console.print("[dim]Run [bold]portfolio-tracker sync[/bold] to import your holdings.[/dim]")
    except Exception as exc:
        console.print(f"[red]Connection failed:[/red] {exc}")
        sys.exit(1)


def cmd_sync(args: argparse.Namespace) -> None:
    from .brokerage import fetch_holdings
    try:
        with console.status("[bold cyan]Syncing holdings from brokerage…[/bold cyan]", spinner="dots2"):
            new_holdings = fetch_holdings()
    except Exception as exc:
        console.print(f"[red]Sync failed:[/red] {exc}")
        sys.exit(1)

    if not new_holdings:
        console.print("[yellow]No equity holdings found in the connected account.[/yellow]")
        return

    data = load_data()
    if args.merge:
        existing = {h["ticker"]: h for h in data["holdings"]}
        for h in new_holdings:
            existing[h["ticker"]] = h
        data["holdings"] = list(existing.values())
    else:
        data["holdings"] = new_holdings
    save_data(data)

    institution = get_key("plaid-institution") or "brokerage"
    t = Table(title=f"Synced from {institution}", box=box.ROUNDED,
              border_style="green", header_style="bold green")
    t.add_column("Ticker",     style="bold white", justify="left")
    t.add_column("Shares",                         justify="right")
    t.add_column("Avg Cost",                       justify="right")
    for h in sorted(new_holdings, key=lambda x: x["ticker"]):
        t.add_row(h["ticker"], f"{h['shares']:,.4f}", f"${h['avg_cost']:,.2f}")
    console.print()
    console.print(t)
    console.print(f"\n[green]Synced {len(new_holdings)} position(s).[/green]")
    console.print("[dim]Run [bold]portfolio-tracker dashboard[/bold] to view metrics.[/dim]\n")


def cmd_export(args: argparse.Namespace) -> None:
    """Render the dashboard to HTML (and optionally PDF via Chrome headless)."""
    import subprocess
    import tempfile

    data = load_data()
    if not data["holdings"]:
        console.print("[yellow]No holdings to export.[/yellow]")
        return

    metrics = compute_metrics(data)
    if metrics is None:
        console.print("[red]Could not compute metrics.[/red]")
        sys.exit(1)

    # ── Render to HTML via a recording console ────────────────────────────────
    rec = Console(record=True, width=120)

    rec.print()
    rec.print(
        Align.center(
            Text("  PORTFOLIO RISK METRICS DASHBOARD  ", style="bold white on dark_blue")
        )
    )
    rec.print(f"[dim]  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
              f"Benchmark: {metrics['benchmark']}  |  "
              f"Risk-free rate: {metrics['rf_rate']*100:.2f}%  |  "
              f"Data window: {metrics['n_days']} trading days[/dim]")
    rec.print()
    rec.print(Columns([_summary_panel(metrics), _metrics_panel(metrics)], equal=True, expand=True))
    rec.print()
    rec.print(_holdings_table(metrics))
    rec.print()
    rec.print(_attribution_table(metrics))
    rec.print()

    html = rec.export_html(inline_styles=True)

    # Wrap in a full HTML page with dark background and print-friendly styles
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Portfolio Risk Metrics — {datetime.now().strftime('%Y-%m-%d')}</title>
<style>
  body {{
    background: #0d1117;
    margin: 0;
    padding: 24px 32px;
    font-family: 'Menlo', 'Courier New', monospace;
    font-size: 13px;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}
  @page {{ margin: 0.5in; size: A4 landscape; }}
  @media print {{
    body {{ background: #0d1117; padding: 0; }}
  }}
</style>
</head>
<body>
{html}
</body>
</html>"""

    # Determine output paths
    stem = f"portfolio-risk-{datetime.now().strftime('%Y%m%d')}"
    out_dir = Path(args.output).parent if args.output else Path.cwd()
    out_stem = Path(args.output).stem if args.output else stem

    html_path = out_dir / f"{out_stem}.html"
    html_path.write_text(html, encoding="utf-8")
    console.print(f"[green]HTML saved:[/green] {html_path}")

    if args.format == "pdf":
        pdf_path = out_dir / f"{out_stem}.pdf"
        chrome_bins = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "google-chrome", "chromium",
        ]
        chrome = next((b for b in chrome_bins if Path(b).exists() or b in ("google-chrome", "chromium")), None)
        if chrome and Path(chrome).exists():
            result = subprocess.run(
                [
                    chrome,
                    "--headless=new",
                    "--disable-gpu",
                    "--no-sandbox",
                    f"--print-to-pdf={pdf_path}",
                    "--print-to-pdf-no-header",
                    str(html_path),
                ],
                capture_output=True,
                text=True,
            )
            if pdf_path.exists():
                console.print(f"[green]PDF saved:[/green]  {pdf_path}")
                html_path.unlink()   # remove intermediate HTML
            else:
                console.print(f"[yellow]PDF generation failed (Chrome error).[/yellow]")
                console.print(f"[dim]{result.stderr.strip()}[/dim]")
                console.print(f"[dim]HTML kept at {html_path} — open in browser and File → Print → Save as PDF.[/dim]")
        else:
            console.print(
                f"[yellow]Chrome not found for PDF conversion.[/yellow]\n"
                f"[dim]Open {html_path} in your browser and use File → Print → Save as PDF.[/dim]"
            )


def _parse_dollar(s: str) -> Optional[float]:
    """Parse a dollar string like '+$2,913.00' or '-$87.06' into a float. Returns None on '--'."""
    s = s.strip()
    if not s or s.startswith("--"):
        return None
    sign = -1.0 if s.startswith("-") else 1.0
    s = s.lstrip("+-").lstrip("$").replace(",", "")
    try:
        return sign * float(s)
    except ValueError:
        return None


def cmd_import_csv(args: argparse.Namespace) -> None:
    """
    Import holdings from a Merrill Lynch brokerage CSV export.
    Derives avg_cost from: (market_value - unrealized_gain_loss) / shares
    """
    import csv

    path = Path(args.file)
    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        sys.exit(1)

    holdings: list[dict] = []
    skipped: list[str] = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            # Need at least 8 columns; col[1] is "Symbol Description"
            if len(row) < 8:
                continue
            desc = row[1].strip().strip('"')
            # Skip header rows, totals, blank descriptions
            if not desc or desc.lower() in ("symbol description", "total", "product class"):
                continue
            # Skip rows that are obviously not equities (start with digit = CUSIP/cash acct)
            first_word = desc.split()[0] if desc.split() else ""
            if not first_word or first_word[0].isdigit():
                continue

            ticker = first_word.upper()

            qty_str = row[2].strip().strip('"').replace(",", "")
            price_str = row[3].strip().strip('"')
            value_str = row[5].strip().strip('"')
            gl_str = row[7].strip().strip('"')  # "Unrealized Gain/Loss"

            # Parse quantity
            try:
                qty = float(qty_str)
            except ValueError:
                continue
            if qty <= 0:
                continue

            # Parse market value
            market_value = _parse_dollar(value_str.lstrip("+"))
            if market_value is None or market_value <= 0:
                continue

            # Parse unrealized gain/loss — field may be "$X.XX Y.YY%" so take first token
            gl_token = gl_str.split()[0] if gl_str.split() else "--"
            gl_dollar = _parse_dollar(gl_token)

            if gl_dollar is None:
                # Cost basis unavailable (e.g. VONE shows "-- --"); use current price
                avg_cost = market_value / qty
            else:
                cost_basis = market_value - gl_dollar
                avg_cost = cost_basis / qty

            if avg_cost <= 0:
                skipped.append(f"{ticker} (non-positive avg cost)")
                continue

            holdings.append({"ticker": ticker, "shares": qty, "avg_cost": round(avg_cost, 4)})

    if not holdings:
        console.print("[red]No valid equity positions found in the file.[/red]")
        sys.exit(1)

    data = load_data()
    if not args.merge:
        data["holdings"] = []

    merged, added = 0, 0
    for h in holdings:
        existing = next((x for x in data["holdings"] if x["ticker"] == h["ticker"]), None)
        if existing:
            old_val = existing["shares"] * existing["avg_cost"]
            new_val = h["shares"] * h["avg_cost"]
            existing["shares"] += h["shares"]
            existing["avg_cost"] = (old_val + new_val) / existing["shares"]
            merged += 1
        else:
            data["holdings"].append(h)
            added += 1

    save_data(data)

    # Summary table
    t = Table(title="Imported Holdings", box=box.ROUNDED, border_style="green", header_style="bold green")
    t.add_column("Ticker",    style="bold white", justify="left")
    t.add_column("Shares",                        justify="right")
    t.add_column("Avg Cost",                      justify="right")
    t.add_column("Book Value",                    justify="right")
    for h in sorted(holdings, key=lambda x: x["ticker"]):
        bv = h["shares"] * h["avg_cost"]
        t.add_row(h["ticker"], f"{h['shares']:,.2f}", f"${h['avg_cost']:,.2f}", f"${bv:,.2f}")

    console.print()
    console.print(t)
    console.print(f"\n[green]Imported {added} new position(s)[/green]", end="")
    if merged:
        console.print(f", [yellow]averaged into {merged} existing[/yellow]", end="")
    if skipped:
        console.print(f"\n[dim]Skipped: {', '.join(skipped)}[/dim]", end="")
    console.print()


# ─── Entry Point ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracker",
        description="Portfolio Risk Metrics Tracker — terminal dashboard for risk analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python tracker.py add AAPL 50 172.50
  python tracker.py add MSFT 20 415.00
  python tracker.py add SPY  30 520.00
  python tracker.py dashboard
  python tracker.py remove MSFT
  python tracker.py set-benchmark QQQ
  python tracker.py set-risk-free 0.045
        """,
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # add
    p = sub.add_parser("add", help="Add or average into a position")
    p.add_argument("ticker",   help="Stock ticker symbol (e.g. AAPL)")
    p.add_argument("shares",   type=float, help="Number of shares")
    p.add_argument("avg_cost", type=float, help="Average cost per share in USD")
    p.set_defaults(func=cmd_add)

    # remove
    p = sub.add_parser("remove", help="Remove a position entirely")
    p.add_argument("ticker", help="Ticker symbol to remove")
    p.set_defaults(func=cmd_remove)

    # list
    p = sub.add_parser("list", help="List all holdings from the JSON file")
    p.set_defaults(func=cmd_list)

    # dashboard
    p = sub.add_parser("dashboard", help="Show the full risk metrics dashboard")
    p.set_defaults(func=cmd_dashboard)

    # set-benchmark
    p = sub.add_parser("set-benchmark", help="Change the benchmark index (default: SPY)")
    p.add_argument("ticker", help="Benchmark ticker (e.g. QQQ, IWM, ^GSPC)")
    p.set_defaults(func=cmd_set_benchmark)

    # set-risk-free
    p = sub.add_parser("set-risk-free", help="Set the annual risk-free rate as a decimal")
    p.add_argument("rate", type=float, help="e.g. 0.05 for 5%%")
    p.set_defaults(func=cmd_set_rf)

    # import-csv
    p = sub.add_parser("import-csv", help="Import holdings from a Merrill Lynch CSV export")
    p.add_argument("file", help="Path to the CSV file")
    p.add_argument("--merge", action="store_true",
                   help="Merge into existing holdings instead of replacing them")
    p.set_defaults(func=cmd_import_csv)

    # export
    p = sub.add_parser("export", help="Export the dashboard to HTML or PDF")
    p.add_argument("--format", choices=["html", "pdf"], default="pdf",
                   help="Output format (default: pdf)")
    p.add_argument("--output", metavar="PATH",
                   help="Output file path (default: portfolio-risk-YYYYMMDD in current dir)")
    p.set_defaults(func=cmd_export)

    # set-key
    p = sub.add_parser("set-key", help="Store a provider API key")
    p.add_argument("name",  help="Key name: polygon | finnhub | plaid-client-id | plaid-secret | plaid-env")
    p.add_argument("value", help="API key value")
    p.set_defaults(func=cmd_set_key)

    # connect
    p = sub.add_parser("connect", help="Connect a brokerage account via Plaid (opens browser)")
    p.set_defaults(func=cmd_connect)

    # sync
    p = sub.add_parser("sync", help="Sync holdings from your connected brokerage")
    p.add_argument("--merge", action="store_true",
                   help="Merge with existing holdings instead of replacing")
    p.set_defaults(func=cmd_sync)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
