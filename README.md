# Portfolio Risk Metrics Tracker

A terminal-based portfolio risk dashboard built with [Rich](https://github.com/Textualize/rich).

## Install

```bash
pip install portfolio-risk-tracker
```

## Usage

```bash
# Add positions
portfolio-tracker add AAPL 50 172.50
portfolio-tracker add MSFT 20 415.00

# Import from a Merrill Lynch CSV export
portfolio-tracker import-csv ~/Downloads/holdings.csv

# Full risk dashboard
portfolio-tracker dashboard

# Export to PDF
portfolio-tracker export --format pdf

# Other commands
portfolio-tracker list
portfolio-tracker remove AAPL
portfolio-tracker set-benchmark QQQ
portfolio-tracker set-risk-free 0.045
```

## Metrics

| Metric | Description |
|---|---|
| Beta | Portfolio sensitivity to the benchmark |
| Sharpe Ratio | Risk-adjusted return (annualised) |
| Sortino Ratio | Sharpe using only downside deviation |
| Alpha | Excess return above CAPM prediction |
| Information Ratio | Active return per unit of tracking error |
| Tracking Error | Annualised std dev of active returns |
| VaR 95% | Daily loss not exceeded 95% of the time |
| Max Drawdown | Largest peak-to-trough decline |

## Data

Holdings are stored in `~/.portfolio_tracker/holdings.json`. Market data is fetched live from Yahoo Finance.

## Requirements

Python 3.9+
