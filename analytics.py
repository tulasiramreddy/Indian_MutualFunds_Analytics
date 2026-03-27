"""Analytics engine for mutual fund data."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import brentq

from config import RISK_FREE_RATE, ROLLING_WINDOWS, TRADING_DAYS_PER_YEAR, VAR_CONFIDENCE_LEVELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _daily_returns(prices: pd.Series) -> pd.Series:
    """Compute daily simple returns from a price/NAV series."""
    return prices.pct_change().dropna()


def _monthly_returns(prices: pd.Series) -> pd.Series:
    """Compute monthly returns from a daily price series.

    Resamples to month-end and computes return for each month.
    """
    monthly = prices.resample("ME").last().dropna()
    return monthly.pct_change().dropna()


def _annualized_return(total_return: float, days: int) -> float:
    """Convert a total return over N calendar days to CAGR."""
    if days <= 0 or total_return <= -1:
        return np.nan
    years = days / 365.25
    if years == 0:
        return np.nan
    return (1 + total_return) ** (1 / years) - 1


# ---------------------------------------------------------------------------
# Return Metrics
# ---------------------------------------------------------------------------

def cagr(prices: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    if len(prices) < 2:
        return np.nan
    total_ret = prices.iloc[-1] / prices.iloc[0] - 1
    days = (prices.index[-1] - prices.index[0]).days
    return _annualized_return(total_ret, days)


def absolute_return(prices: pd.Series) -> float:
    """Total absolute return."""
    if len(prices) < 2:
        return np.nan
    return prices.iloc[-1] / prices.iloc[0] - 1


def trailing_returns(prices: pd.Series) -> dict:
    """Compute trailing returns for standard periods.

    Returns dict with keys like '1M', '3M', '6M', '1Y', '3Y', '5Y', '10Y', 'SI'.
    """
    if len(prices) < 2:
        return {}

    end_date = prices.index[-1]
    result = {}
    periods = {
        "1M": 30, "3M": 91, "6M": 182,
        "1Y": 365, "2Y": 730, "3Y": 1095,
        "5Y": 1825, "7Y": 2555, "10Y": 3650,
    }

    for label, days in periods.items():
        target = end_date - pd.Timedelta(days=days)
        subset = prices[prices.index >= target]
        if len(subset) >= 2:
            ret = subset.iloc[-1] / subset.iloc[0] - 1
            ann = _annualized_return(ret, days) if days >= 365 else ret
            result[label] = ann
        else:
            result[label] = np.nan

    result["SI"] = cagr(prices)
    return result


def rolling_returns(prices: pd.Series, window_days: int = 252) -> pd.Series:
    """Compute rolling annualized returns.

    For each date, computes the CAGR over the trailing window.
    """
    if len(prices) < window_days:
        return pd.Series(dtype=float)

    ret = prices / prices.shift(window_days) - 1
    years = window_days / TRADING_DAYS_PER_YEAR
    rolling = (1 + ret) ** (1 / years) - 1
    return rolling.dropna()


# ---------------------------------------------------------------------------
# Risk Metrics
# ---------------------------------------------------------------------------

def annualized_volatility(prices: pd.Series) -> float:
    """Annualized standard deviation of daily returns."""
    dr = _daily_returns(prices)
    if len(dr) < 2:
        return np.nan
    return dr.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def downside_deviation(prices: pd.Series, mar: float = None) -> float:
    """Annualized downside deviation.

    mar: Minimum Acceptable Return (daily). Defaults to daily risk-free rate.
    """
    if mar is None:
        mar = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS_PER_YEAR) - 1

    dr = _daily_returns(prices)
    if len(dr) < 2:
        return np.nan

    downside = np.minimum(dr - mar, 0)
    return np.sqrt(np.mean(downside ** 2)) * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(prices: pd.Series) -> float:
    """Maximum drawdown (as a negative fraction, e.g. -0.25 for 25%)."""
    if len(prices) < 2:
        return np.nan
    peak = prices.cummax()
    dd = (prices - peak) / peak
    return dd.min()


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Compute the drawdown time series."""
    peak = prices.cummax()
    return (prices - peak) / peak


def drawdown_analysis(prices: pd.Series) -> list[dict]:
    """Analyze all drawdown periods.

    Returns a list of dicts, each with:
        start, trough_date, end (recovery date or None if ongoing),
        depth (max drawdown in period), duration_days, recovery_days
    Sorted by depth (worst first).
    """
    if len(prices) < 2:
        return []

    dd = drawdown_series(prices)
    periods = []
    in_dd = False
    start = None
    trough_date = None
    trough_val = 0

    for i, (date, val) in enumerate(dd.items()):
        if val < 0:
            if not in_dd:
                in_dd = True
                start = date
                trough_date = date
                trough_val = val
            elif val < trough_val:
                trough_date = date
                trough_val = val
        else:
            if in_dd:
                periods.append({
                    "start": start,
                    "trough_date": trough_date,
                    "end": date,
                    "depth": trough_val,
                    "duration_days": (date - start).days,
                    "recovery_days": (date - trough_date).days,
                })
                in_dd = False

    # Handle ongoing drawdown
    if in_dd:
        periods.append({
            "start": start,
            "trough_date": trough_date,
            "end": None,
            "depth": trough_val,
            "duration_days": (dd.index[-1] - start).days,
            "recovery_days": None,
        })

    periods.sort(key=lambda x: x["depth"])
    return periods


def ulcer_index(prices: pd.Series) -> float:
    """Ulcer Index: RMS of all drawdown values.

    Penalises deep *and* prolonged drawdowns more than max-drawdown alone.
    Returns a positive decimal (e.g. 0.08 for 8 % UI).
    """
    if len(prices) < 2:
        return np.nan
    dd = drawdown_series(prices)
    return float(np.sqrt(np.mean(dd ** 2)))


def average_drawdown(prices: pd.Series) -> float:
    """Mean drawdown over the entire period (negative fraction)."""
    if len(prices) < 2:
        return np.nan
    dd = drawdown_series(prices)
    return float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0


def pct_time_underwater(prices: pd.Series) -> float:
    """Fraction of days the fund was below its previous all-time high."""
    if len(prices) < 2:
        return np.nan
    dd = drawdown_series(prices)
    return float((dd < 0).sum() / len(dd))


def value_at_risk(prices: pd.Series, confidence: float = 0.95,
                  method: str = "historical") -> float:
    """Value at Risk (daily).

    Returns the loss threshold at the given confidence level (negative number).
    """
    dr = _daily_returns(prices)
    if len(dr) < 30:
        return np.nan

    if method == "historical":
        return np.percentile(dr, (1 - confidence) * 100)
    elif method == "parametric":
        mu = dr.mean()
        sigma = dr.std()
        z = sp_stats.norm.ppf(1 - confidence)
        return mu + z * sigma
    return np.nan


def conditional_var(prices: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall) — mean of losses beyond VaR."""
    dr = _daily_returns(prices)
    if len(dr) < 30:
        return np.nan
    var = value_at_risk(prices, confidence, "historical")
    tail = dr[dr <= var]
    return tail.mean() if len(tail) > 0 else var


def return_skewness(prices: pd.Series) -> float:
    """Skewness of daily returns."""
    dr = _daily_returns(prices)
    if len(dr) < 30:
        return np.nan
    return float(sp_stats.skew(dr))


def return_kurtosis(prices: pd.Series) -> float:
    """Excess kurtosis of daily returns."""
    dr = _daily_returns(prices)
    if len(dr) < 30:
        return np.nan
    return float(sp_stats.kurtosis(dr))


# ---------------------------------------------------------------------------
# Risk-Adjusted Return Metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(prices: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Annualized Sharpe Ratio."""
    ann_ret = cagr(prices)
    vol = annualized_volatility(prices)
    if np.isnan(ann_ret) or np.isnan(vol) or vol == 0:
        return np.nan
    return (ann_ret - rf) / vol


def sortino_ratio(prices: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Annualized Sortino Ratio."""
    ann_ret = cagr(prices)
    dd = downside_deviation(prices)
    if np.isnan(ann_ret) or np.isnan(dd) or dd == 0:
        return np.nan
    return (ann_ret - rf) / dd


def calmar_ratio(prices: pd.Series) -> float:
    """Calmar Ratio: CAGR / |max drawdown|."""
    ann_ret = cagr(prices)
    mdd = max_drawdown(prices)
    if np.isnan(ann_ret) or np.isnan(mdd) or mdd == 0:
        return np.nan
    return ann_ret / abs(mdd)


# ---------------------------------------------------------------------------
# Benchmark-Relative Metrics
# ---------------------------------------------------------------------------

def martin_ratio(prices: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Martin Ratio: (CAGR - rf) / Ulcer Index.

    A Sharpe-like ratio that uses Ulcer Index instead of volatility, making it
    more sensitive to prolonged drawdowns than to routine up-and-down noise.
    """
    ui = ulcer_index(prices)
    if np.isnan(ui) or ui == 0:
        return np.nan
    ann_ret = cagr(prices)
    if np.isnan(ann_ret):
        return np.nan
    return (ann_ret - rf) / ui


def pct_positive_periods(prices: pd.Series, window_days: int = 252) -> float:
    """Fraction of rolling windows (default 1Y) that delivered a positive return."""
    rr = rolling_returns(prices, window_days)
    if rr.empty:
        return np.nan
    return float((rr > 0).sum() / len(rr))


def _align_series(fund: pd.Series, benchmark: pd.Series):
    """Align fund and benchmark on common dates."""
    combined = pd.DataFrame({"fund": fund, "bench": benchmark}).dropna()
    return combined["fund"], combined["bench"]


def beta(fund_prices: pd.Series, bench_prices: pd.Series) -> float:
    """Beta of fund relative to benchmark."""
    fp, bp = _align_series(fund_prices, bench_prices)
    if len(fp) < 30:
        return np.nan
    fr = _daily_returns(fp)
    br = _daily_returns(bp)
    combined = pd.DataFrame({"f": fr, "b": br}).dropna()
    if len(combined) < 30:
        return np.nan
    cov = combined["f"].cov(combined["b"])
    var = combined["b"].var()
    return cov / var if var != 0 else np.nan


def alpha_jensen(fund_prices: pd.Series, bench_prices: pd.Series,
                 rf: float = RISK_FREE_RATE) -> float:
    """Jensen's Alpha (annualized)."""
    b = beta(fund_prices, bench_prices)
    if np.isnan(b):
        return np.nan

    fp, bp = _align_series(fund_prices, bench_prices)
    fund_ret = cagr(fp)
    bench_ret = cagr(bp)
    if np.isnan(fund_ret) or np.isnan(bench_ret):
        return np.nan
    return fund_ret - (rf + b * (bench_ret - rf))


def tracking_error(fund_prices: pd.Series, bench_prices: pd.Series) -> float:
    """Annualized Tracking Error."""
    fp, bp = _align_series(fund_prices, bench_prices)
    if len(fp) < 30:
        return np.nan
    fr = _daily_returns(fp)
    br = _daily_returns(bp)
    combined = pd.DataFrame({"f": fr, "b": br}).dropna()
    excess = combined["f"] - combined["b"]
    return excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def information_ratio(fund_prices: pd.Series, bench_prices: pd.Series) -> float:
    """Information Ratio: excess return / tracking error."""
    fp, bp = _align_series(fund_prices, bench_prices)
    fund_ret = cagr(fp)
    bench_ret = cagr(bp)
    te = tracking_error(fund_prices, bench_prices)
    if np.isnan(fund_ret) or np.isnan(bench_ret) or np.isnan(te) or te == 0:
        return np.nan
    return (fund_ret - bench_ret) / te


def treynor_ratio(fund_prices: pd.Series, bench_prices: pd.Series,
                  rf: float = RISK_FREE_RATE) -> float:
    """Treynor Ratio: (fund return - rf) / beta."""
    b = beta(fund_prices, bench_prices)
    if np.isnan(b) or b == 0:
        return np.nan
    fund_ret = cagr(fund_prices)
    if np.isnan(fund_ret):
        return np.nan
    return (fund_ret - rf) / b


def upside_capture(fund_prices: pd.Series, bench_prices: pd.Series) -> float:
    """Upside Capture Ratio (monthly)."""
    fp, bp = _align_series(fund_prices, bench_prices)
    if len(fp) < 60:  # need at least ~2 months of daily data
        return np.nan

    fm = _monthly_returns(fp)
    bm = _monthly_returns(bp)
    combined = pd.DataFrame({"f": fm, "b": bm}).dropna()

    up = combined[combined["b"] > 0]
    if len(up) < 3:
        return np.nan

    fund_geo = (1 + up["f"]).prod() ** (1 / len(up)) - 1
    bench_geo = (1 + up["b"]).prod() ** (1 / len(up)) - 1
    if bench_geo == 0:
        return np.nan
    return (fund_geo / bench_geo) * 100


def downside_capture(fund_prices: pd.Series, bench_prices: pd.Series) -> float:
    """Downside Capture Ratio (monthly).

    Lower is better. A ratio < 100 means the fund loses less than the benchmark
    in down months.
    """
    fp, bp = _align_series(fund_prices, bench_prices)
    if len(fp) < 60:
        return np.nan

    fm = _monthly_returns(fp)
    bm = _monthly_returns(bp)
    combined = pd.DataFrame({"f": fm, "b": bm}).dropna()

    down = combined[combined["b"] < 0]
    if len(down) < 3:
        return np.nan

    fund_geo = (1 + down["f"]).prod() ** (1 / len(down)) - 1
    bench_geo = (1 + down["b"]).prod() ** (1 / len(down)) - 1
    if bench_geo == 0:
        return np.nan
    return (fund_geo / bench_geo) * 100


def batting_average(fund_prices: pd.Series, bench_prices: pd.Series) -> float:
    """Batting Average: fraction of months the fund outperformed the benchmark."""
    fp, bp = _align_series(fund_prices, bench_prices)
    if len(fp) < 60:
        return np.nan
    fm = _monthly_returns(fp)
    bm = _monthly_returns(bp)
    combined = pd.DataFrame({"f": fm, "b": bm}).dropna()
    if len(combined) < 6:
        return np.nan
    return float((combined["f"] > combined["b"]).sum() / len(combined))


# ---------------------------------------------------------------------------
# SIP / XIRR Metrics
# ---------------------------------------------------------------------------

def _xirr_solve(cashflows: list[float], dates: list) -> float:
    """Solve for XIRR given cashflows and dates.

    Cashflow sign convention (same as Excel XIRR):
        Investments  → negative  (money leaving your pocket)
        Redemptions  → positive  (money received)
    """
    if len(cashflows) < 2:
        return np.nan
    t0 = dates[0]
    years = [(d - t0).days / 365.25 for d in dates]

    def npv(r):
        return sum(cf / (1 + r) ** t for cf, t in zip(cashflows, years))

    try:
        # Wide bracket: funds can have multi-hundred-percent short-term XIRRs
        return brentq(npv, -0.9999, 200.0, xtol=1e-6, maxiter=500)
    except (ValueError, RuntimeError):
        return np.nan


def sip_xirr(prices: pd.Series, years: int) -> float:
    """XIRR for a monthly SIP of ₹1 invested over `years` years.

    The SIP ends at the last NAV date in `prices`.  Each month's investment
    buys units at the NAV on (or just after) the 1st of that month.
    Returns annualised XIRR as a decimal (e.g. 0.15 for 15 %).
    """
    if len(prices) < 2:
        return np.nan

    end_date = prices.index[-1]
    n_months = years * 12
    month_starts = pd.date_range(end=end_date, periods=n_months, freq="MS")

    # Check we have NAV data going back far enough
    if prices.index[0] > month_starts[0]:
        return np.nan

    cashflows: list[float] = []
    dates: list = []
    total_units = 0.0

    for ms in month_starts:
        # Find the closest available NAV on or after the month-start date
        loc = prices.index.searchsorted(ms)
        if loc >= len(prices):
            loc = len(prices) - 1
        nav_date = prices.index[loc]
        nav_val = prices.iloc[loc]
        if nav_val <= 0:
            continue
        units = 1.0 / nav_val
        total_units += units
        cashflows.append(-1.0)   # investment (cash out)
        dates.append(nav_date)

    if total_units <= 0 or not cashflows:
        return np.nan

    portfolio_value = total_units * prices.iloc[-1]
    cashflows.append(portfolio_value)   # redemption (cash in)
    dates.append(end_date)

    return _xirr_solve(cashflows, dates)


def rolling_sip_xirr(prices: pd.Series, years: int) -> pd.Series:
    """XIRR for every possible `years`-year SIP window in the price history.

    Returns a Series indexed by SIP *start* date, values are XIRRs.
    Useful for histograms showing the distribution of SIP outcomes.
    """
    if len(prices) < 2:
        return pd.Series(dtype=float)

    results: dict = {}
    n_months = years * 12
    # All possible month-start dates within the price series
    all_starts = pd.date_range(
        start=prices.index[0], end=prices.index[-1], freq="MS"
    )

    for start in all_starts:
        end_approx = start + pd.DateOffset(months=n_months)
        if end_approx > prices.index[-1]:
            break
        # Slice the price series for this window
        slice_prices = prices[start:]
        if len(slice_prices) < n_months * 15:   # rough guard (~15 trading days/month)
            break
        xirr_val = sip_xirr(slice_prices, years)
        if not np.isnan(xirr_val):
            results[start] = xirr_val

    if not results:
        return pd.Series(dtype=float)
    return pd.Series(results)


# ---------------------------------------------------------------------------
# Comprehensive Analytics
# ---------------------------------------------------------------------------

def compute_fund_analytics(
    fund_prices: pd.Series,
    bench_prices: Optional[pd.Series] = None,
    rf: float = RISK_FREE_RATE,
) -> dict:
    """Compute all analytics for a fund.

    Args:
        fund_prices: DatetimeIndex-ed Series of NAV values.
        bench_prices: Optional benchmark price series (aligned by date).
        rf: Annual risk-free rate.

    Returns:
        Dict of metric_name -> value.
    """
    metrics = {}

    # --- Returns ---
    metrics["cagr"] = cagr(fund_prices)
    metrics["absolute_return"] = absolute_return(fund_prices)
    tr = trailing_returns(fund_prices)
    for k, v in tr.items():
        metrics[f"return_{k}"] = v

    # --- Risk ---
    metrics["volatility"] = annualized_volatility(fund_prices)
    metrics["downside_deviation"] = downside_deviation(fund_prices)
    metrics["max_drawdown"] = max_drawdown(fund_prices)
    metrics["ulcer_index"] = ulcer_index(fund_prices)
    metrics["average_drawdown"] = average_drawdown(fund_prices)
    metrics["pct_time_underwater"] = pct_time_underwater(fund_prices)
    metrics["skewness"] = return_skewness(fund_prices)
    metrics["kurtosis"] = return_kurtosis(fund_prices)

    for cl in VAR_CONFIDENCE_LEVELS:
        pct = int(cl * 100)
        metrics[f"var_{pct}"] = value_at_risk(fund_prices, cl)
        metrics[f"cvar_{pct}"] = conditional_var(fund_prices, cl)

    # --- Risk-Adjusted ---
    metrics["sharpe_ratio"] = sharpe_ratio(fund_prices, rf)
    metrics["sortino_ratio"] = sortino_ratio(fund_prices, rf)
    metrics["calmar_ratio"] = calmar_ratio(fund_prices)
    metrics["martin_ratio"] = martin_ratio(fund_prices, rf)
    metrics["pct_positive_1y"] = pct_positive_periods(fund_prices, 252)

    # --- Drawdown analysis ---
    dd_periods = drawdown_analysis(fund_prices)
    if dd_periods:
        metrics["worst_drawdown_depth"] = dd_periods[0]["depth"]
        metrics["worst_drawdown_duration"] = dd_periods[0]["duration_days"]
        metrics["worst_drawdown_recovery"] = dd_periods[0].get("recovery_days")
        metrics["num_drawdown_periods"] = len(dd_periods)
    metrics["_drawdown_periods"] = dd_periods  # store full list for charts

    # --- SIP XIRR ---
    for yrs in [1, 3, 5, 10]:
        metrics[f"sip_xirr_{yrs}y"] = sip_xirr(fund_prices, yrs)

    # --- Benchmark-relative ---
    if bench_prices is not None and len(bench_prices) > 30:
        metrics["beta"] = beta(fund_prices, bench_prices)
        metrics["alpha"] = alpha_jensen(fund_prices, bench_prices, rf)
        metrics["tracking_error"] = tracking_error(fund_prices, bench_prices)
        metrics["information_ratio"] = information_ratio(fund_prices, bench_prices)
        metrics["treynor_ratio"] = treynor_ratio(fund_prices, bench_prices, rf)
        metrics["upside_capture"] = upside_capture(fund_prices, bench_prices)
        metrics["downside_capture"] = downside_capture(fund_prices, bench_prices)
        metrics["batting_average"] = batting_average(fund_prices, bench_prices)

    return metrics


def calendar_year_returns(prices: pd.Series) -> pd.Series:
    """Calendar year returns. Index = int year, values = decimal return.

    Partial years at start/end are included as-is.
    """
    if len(prices) < 2:
        return pd.Series(dtype=float)
    year_ends = prices.resample("YE").last().dropna()
    if year_ends.empty:
        return pd.Series(dtype=float)
    # Prepend the first actual observation as the base for the first partial year
    first = prices.iloc[[0]]
    combined = pd.concat([first, year_ends])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    rets = combined.pct_change().dropna()
    rets.index = rets.index.year
    return rets


def rolling_volatility(prices: pd.Series, window_days: int = 252) -> pd.Series:
    """Rolling annualized volatility over a trailing window."""
    dr = _daily_returns(prices)
    if len(dr) < window_days:
        return pd.Series(dtype=float)
    return (dr.rolling(window_days).std() * np.sqrt(TRADING_DAYS_PER_YEAR)).dropna()


def rolling_sharpe(prices: pd.Series, window_days: int = 252,
                   rf: float = RISK_FREE_RATE) -> pd.Series:
    """Rolling annualized Sharpe ratio over a trailing window."""
    dr = _daily_returns(prices)
    if len(prices) < window_days + 1:
        return pd.Series(dtype=float)
    ann_ret = (prices / prices.shift(window_days)) ** (TRADING_DAYS_PER_YEAR / window_days) - 1
    ann_vol = (dr.rolling(window_days).std() * np.sqrt(TRADING_DAYS_PER_YEAR)).reindex(prices.index)
    sharpe = (ann_ret - rf) / ann_vol.replace(0, np.nan)
    return sharpe.dropna()


def rolling_sortino(prices: pd.Series, window_days: int = 252,
                    rf: float = RISK_FREE_RATE) -> pd.Series:
    """Rolling annualized Sortino ratio over a trailing window."""
    dr = _daily_returns(prices)
    if len(prices) < window_days + 1:
        return pd.Series(dtype=float)
    daily_mar = (1 + rf) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    downside = np.minimum(dr - daily_mar, 0)
    rolling_dd = (
        (downside ** 2).rolling(window_days).mean().apply(np.sqrt)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    ).reindex(prices.index)
    ann_ret = (prices / prices.shift(window_days)) ** (TRADING_DAYS_PER_YEAR / window_days) - 1
    sortino = (ann_ret - rf) / rolling_dd.replace(0, np.nan)
    return sortino.dropna()


def rolling_alpha(fund_prices: pd.Series, bench_prices: pd.Series,
                  window_days: int = 252,
                  rf: float = RISK_FREE_RATE) -> pd.Series:
    """Rolling Jensen's Alpha over a trailing window."""
    fp, bp = _align_series(fund_prices, bench_prices)
    fr = _daily_returns(fp)
    br = _daily_returns(bp)
    combined = pd.DataFrame({"f": fr, "b": br}).dropna()
    if len(combined) < window_days:
        return pd.Series(dtype=float)

    roll_cov = combined["f"].rolling(window_days).cov(combined["b"])
    roll_var = combined["b"].rolling(window_days).var()
    roll_beta = roll_cov / roll_var.replace(0, np.nan)

    fp_r = fp.reindex(combined.index)
    bp_r = bp.reindex(combined.index)
    ann_fund  = (fp_r / fp_r.shift(window_days)) ** (TRADING_DAYS_PER_YEAR / window_days) - 1
    ann_bench = (bp_r / bp_r.shift(window_days)) ** (TRADING_DAYS_PER_YEAR / window_days) - 1

    roll_alpha_s = ann_fund - (rf + roll_beta * (ann_bench - rf))
    return roll_alpha_s.dropna()


def rolling_beta(fund_prices: pd.Series, bench_prices: pd.Series,
                 window_days: int = 252) -> pd.Series:
    """Rolling Beta over a trailing window."""
    fp, bp = _align_series(fund_prices, bench_prices)
    fr = _daily_returns(fp)
    br = _daily_returns(bp)
    combined = pd.DataFrame({"f": fr, "b": br}).dropna()
    if len(combined) < window_days:
        return pd.Series(dtype=float)

    roll_cov = combined["f"].rolling(window_days).cov(combined["b"])
    roll_var = combined["b"].rolling(window_days).var()
    return (roll_cov / roll_var.replace(0, np.nan)).dropna()


def compute_comparison_table(
    fund_dict: dict[str, pd.Series],
    bench_prices: Optional[pd.Series] = None,
    rf: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    """Compute analytics for multiple funds and return as a comparison DataFrame.

    Args:
        fund_dict: {fund_name: NAV Series}
        bench_prices: Optional benchmark series.
        rf: Risk-free rate.

    Returns:
        DataFrame where rows are metrics and columns are fund names.
    """
    results = {}
    for name, prices in fund_dict.items():
        m = compute_fund_analytics(prices, bench_prices, rf)
        # Remove internal fields
        results[name] = {k: v for k, v in m.items() if not k.startswith("_")}

    df = pd.DataFrame(results)
    return df
