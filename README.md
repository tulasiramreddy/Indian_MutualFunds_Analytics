# MF Analytics

A professional-grade analytics platform for Indian equity mutual funds, built with Streamlit. Analyse fund performance, compare against benchmarks, and explore sector/style breakdowns — all in an interactive web UI.

## Features

- **Fund Analysis** — NAV charts, trailing returns (1M–10Y + since inception), risk metrics (Sharpe, Sortino, Calmar), drawdown analysis, SIP XIRR, rolling metrics, and calendar year returns
- **Benchmark Comparison** — Alpha, beta, upside/downside capture ratio, batting average vs 28+ NSE/BSE indices
- **Fund Comparison** — Side-by-side metrics table with correlation matrix for up to N funds
- **Sector & Style** — Funds grouped by SEBI category and theme (Banking, IT, Healthcare, Infrastructure, PSU, etc.) with auto-selected relevant benchmarks
- **Automated Data Updates** — Daily fetcher pulls NAV data from AMFI/mfapi.in and index closes from NSE archives + Yahoo Finance

## Data Sources

| Source | Data |
|--------|------|
| [AMFI India](https://www.amfiindia.com) | Scheme list & metadata |
| [mfapi.in](https://www.mfapi.in) | Historical NAV data |
| [NSE Archives](https://archives.nseindia.com) | Index daily closes (Jan 2013+) |
| Yahoo Finance (`yfinance`) | Broad market & sector indices |

All sources are free and public — no API keys required.

## Setup

### Prerequisites

- Python 3.10+

### Install

```bash
git clone https://github.com/your-username/mf_analytics.git
cd mf_analytics

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Populate the database

The database is not included in the repo (it's ~90 MB). Run the updater to fetch all data:

```bash
python update.py
```

This will:
1. Download the full scheme list from AMFI (~2,400 equity funds)
2. Fetch historical NAV data for each scheme from mfapi.in
3. Fetch benchmark index data from NSE archives and Yahoo Finance

**Note:** The initial population takes ~30–60 minutes due to API rate limits.

### Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Keeping data up to date

Run `update.py` daily to pull the latest NAVs and index closes.

**Manual:**
```bash
python update.py
```

**Scheduled (runs at 22:00 IST on weekdays):**
```bash
python update.py --schedule
```

**Via cron:**
```
0 22 * * 1-5 cd /path/to/mf_analytics && source venv/bin/activate && python update.py
```

## Project Structure

```
mf_analytics/
├── app.py            # Streamlit UI (5 pages)
├── analytics.py      # Financial metrics engine (30+ metrics)
├── fetcher.py        # Data fetching from AMFI, mfapi.in, NSE, yfinance
├── tagger.py         # Fund categorization (SEBI category + theme tags)
├── db.py             # SQLite database layer
├── config.py         # Configuration: API URLs, benchmarks, thresholds
├── update.py         # Daily update script
├── requirements.txt
└── Whitepaper/       # Methodology documentation (PDF + LaTeX source)
```

## Metrics Reference

| Category | Metrics |
|----------|---------|
| Returns | CAGR, absolute return, trailing returns, rolling returns, SIP XIRR, calendar year returns |
| Risk | Volatility, downside volatility, Sharpe ratio, Sortino ratio, Calmar ratio, Martin ratio |
| Drawdown | Max drawdown, average drawdown, ulcer index, drawdown duration, % time underwater |
| Relative | Alpha, beta, correlation, upside/downside capture, batting average |
| Statistical | VaR (95% & 99%), monthly/daily return distributions |

Risk-free rate is set to **6.5%** (India 91-day T-bill) and can be changed in `config.py`.

## Configuration

Key settings in `config.py`:

```python
RISK_FREE_RATE = 0.065        # Used in Sharpe/Sortino/Calmar
TRADING_DAYS_PER_YEAR = 252
MFAPI_DELAY_SECONDS = 0.5     # Rate limit for mfapi.in
```

Benchmark indices and tag-to-benchmark mappings are also defined there.

## Whitepaper

`Whitepaper/whitepaper.pdf` documents the methodology, metric definitions, and data pipeline in detail.
