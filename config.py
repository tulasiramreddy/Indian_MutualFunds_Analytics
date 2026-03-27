"""Configuration constants for MF Analytics."""

import os

# --- API Endpoints ---
AMFI_NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
MFAPI_BASE_URL = "https://api.mfapi.in/mf"

# --- Rate Limiting ---
MFAPI_DELAY_SECONDS = 0.5  # delay between mfapi requests
MFAPI_BATCH_SIZE = 50      # commit to DB every N schemes

# --- Risk-Free Rate (annualized) ---
RISK_FREE_RATE = 0.065  # 6.5% — approximate India 91-day T-bill rate

# --- Trading Days ---
TRADING_DAYS_PER_YEAR = 252

# --- Benchmark Indices (yfinance tickers) ---
# Only tickers confirmed working via yfinance are listed here.
BENCHMARKS = {
    # Broad market
    "Nifty 50":               "^NSEI",
    "Nifty 100":              "^CNX100",
    "BSE Sensex":             "^BSESN",
    "Nifty Next 50":          "^NSEMDCP50",
    # Cap segments
    "Nifty Midcap 50":        "^NSMIDCP",
    "Nifty Midcap 150":       "NIFTYMIDCAP150.NS",
    # Smallcap / Microcap: real history via NSE archives (see NSE_DIRECT_INDICES below).
    # Sector indices
    "Nifty Bank":             "^NSEBANK",
    "Nifty PSU Bank":         "^CNXPSUBANK",
    "Nifty Financial Svc":    "NIFTY_FIN_SERVICE.NS",
    "Nifty IT":               "^CNXIT",
    "Nifty FMCG":             "^CNXFMCG",
    "Nifty Consumption":      "^CNXCONSUM",
    "Nifty Pharma":           "^CNXPHARMA",
    "Nifty Infrastructure":   "^CNXINFRA",
    "Nifty Energy":           "^CNXENERGY",
    "Nifty Metal":            "^CNXMETAL",
    "Nifty Media":            "^CNXMEDIA",
    "Nifty Commodities":      "^CNXCMDT",
    "Nifty Services":         "^CNXSERVICE",
    "Nifty PSE":              "^CNXPSE",
    "Nifty Auto":             "^CNXAUTO",
    "Nifty Realty":           "^CNXREALTY",
    "Nifty MNC":              "^CNXMNC",
}

# --- NSE Archive Indices ---
# These are fetched from https://archives.nseindia.com/content/indices/ind_close_all_DDMMYYYY.csv
# which provides all NSE index closes for each trading day (available from Jan 2013).
# Keys = display name used in the app; values = name as it appears in the CSV (case-insensitive match).
NSE_DIRECT_INDICES = {
    "Nifty Smallcap 50":      "Nifty Smallcap 50",
    "Nifty Smallcap 100":     "Nifty Smallcap 100",
    "Nifty Smallcap 250":     "Nifty Smallcap 250",
    "Nifty Microcap 250":     "Nifty Microcap 250",
    "Nifty Midcap 100":       "Nifty Midcap 100",
    "Nifty 500":              "Nifty 500",
    "Nifty MidSmallcap 400":  "Nifty MidSmallcap 400",
}

NSE_ARCHIVE_URL = (
    "https://archives.nseindia.com/content/indices/ind_close_all_{date}.csv"
)

# --- Recommended benchmark per fund tag ---
# Used by the Sector & Style page to pre-select a relevant index.
# Falls back to "Nifty 50" for any tag not listed here.
TAG_BENCHMARK = {
    # Style / Cap
    "Large Cap":              "Nifty 50",
    "Large & Mid Cap":        "Nifty 100",          # top-100 = large + next-50
    "Mid Cap":                "Nifty Midcap 100",
    "Small Cap":              "Nifty Smallcap 250",
    "Multi Cap":              "Nifty 100",
    "Flexi Cap":              "Nifty 50",
    "Focused":                "Nifty 50",
    "Value":                  "Nifty 50",
    "Contra":                 "Nifty 50",
    "ELSS":                   "Nifty 50",
    # Sector / Theme
    "Banking & Financial Services": "Nifty Financial Svc",
    "Technology":             "Nifty IT",
    "Healthcare":             "Nifty Pharma",
    "Consumption & FMCG":    "Nifty Consumption",  # broader than just FMCG
    "Infrastructure":         "Nifty Infrastructure",
    "Energy & Resources":     "Nifty Commodities",  # covers energy + metals + agri
    "Manufacturing":          "Nifty Metal",         # closest available
    "PSU":                    "Nifty PSE",
    "Auto & Logistics":       "Nifty Auto",
    "Housing":                "Nifty Realty",
    "MNC":                    "Nifty MNC",
    "Services":               "Nifty Services",
    "ESG & Ethical":          "Nifty 50",
    "Quant & Factor":         "Nifty 50",
    "Momentum":               "Nifty 50",
    "Business Cycle":         "Nifty 50",
    "Defence":                "Nifty 50",
    "Conglomerate":           "Nifty 50",
    "Innovation":             "Nifty IT",           # tech-heavy innovation funds
    "Special Situations":     "Nifty 50",
    "Rural & Agri":           "Nifty Consumption",
    "Quality":                "Nifty 50",
    "Dividend Yield":         "Nifty PSU Bank",     # PSU banks dominate dividend payers
    "IPO":                    "Nifty 50",
    "International":          "Nifty 50",
}

# --- Equity Category Prefixes (for filtering AMFI data) ---
EQUITY_CATEGORY_KEYWORDS = [
    "Equity Scheme",
    "ELSS",
]

# --- Analytics Defaults ---
ROLLING_WINDOWS = {
    "6M": 126,
    "1Y": 252,
    "2Y": 504,
    "3Y": 756,
    "5Y": 1260,
}

VAR_CONFIDENCE_LEVELS = [0.95, 0.99]
