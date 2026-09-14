"""Cointegration tests and hedge-ratio estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


@dataclass
class CointegrationResult:
    statistic: float
    pvalue: float
    critical_values: dict[str, float]
    is_cointegrated: bool
    alpha_level: float


def engle_granger(
    y: pd.Series,
    x: pd.Series,
    alpha: float = 0.05,
    trend: str = "c",
    maxlag=None,
    autolag: str = "aic",
) -> CointegrationResult:
    """Engle-Granger two-step cointegration test of ``y`` on ``x``."""
    stat, pvalue, crit = coint(y, x, trend=trend, maxlag=maxlag, autolag=autolag)
    crit_values = {"1%": crit[0], "5%": crit[1], "10%": crit[2]}
    return CointegrationResult(
        statistic=float(stat),
        pvalue=float(pvalue),
        critical_values=crit_values,
        is_cointegrated=bool(pvalue < alpha),
        alpha_level=alpha,
    )


def adf_test(series: pd.Series, alpha: float = 0.05) -> dict:
    """Augmented Dickey-Fuller stationarity test."""
    stat, pvalue, _nlag, _nobs, crit, _ic = adfuller(series.dropna(), result_object=False)
    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "critical_values": {k: float(v) for k, v in crit.items()},
        "is_stationary": bool(pvalue < alpha),
    }


@dataclass
class HedgeRatio:
    alpha: float
    beta: float
    residuals: pd.Series
    r_squared: float


def estimate_hedge_ratio(y: pd.Series, x: pd.Series, method: str = "ols") -> HedgeRatio:
    """Estimate ``y ~ alpha + beta * x``.

    ``method`` is ``"ols"`` (statsmodels OLS with constant) or ``"tls"``
    (total least squares via SVD, symmetric in the two series).
    """
    if method == "ols":
        model = sm.OLS(y, sm.add_constant(x)).fit()
        alpha = float(model.params.iloc[0])
        beta = float(model.params.iloc[1])
        residuals = pd.Series(model.resid, index=y.index)
        r2 = float(model.rsquared)
    elif method == "tls":
        data = np.column_stack([np.asarray(x), np.asarray(y)])
        centred = data - data.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        v = vt[-1]
        beta = float(-v[0] / v[1])
        alpha = float(data[:, 1].mean() - beta * data[:, 0].mean())
        residuals = y - alpha - beta * x
        ss_res = float((residuals**2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot
    else:
        raise ValueError(f"Unknown hedge-ratio method: {method!r}")
    return HedgeRatio(alpha=alpha, beta=beta, residuals=residuals, r_squared=r2)


def rolling_hedge_ratio(y: pd.Series, x: pd.Series, window: int) -> pd.DataFrame:
    """Rolling OLS slope/intercept of ``y`` on ``x`` over ``window`` points."""
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    beta = cov / var
    alpha = y.rolling(window).mean() - beta * x.rolling(window).mean()
    return pd.DataFrame({"alpha": alpha, "beta": beta})
