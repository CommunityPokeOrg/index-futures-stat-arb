import numpy as np
import pytest

from index_futures_stat_arb import cointegration as cg
from index_futures_stat_arb import data


def _log_pair(n=3000, seed=0):
    df = data.generate_synthetic_pair(n=n, seed=seed, sigma_spread=0.05)
    return np.log(df["ES"]), np.log(df["NQ"])


def test_engle_granger_detects_cointegration():
    y, x = _log_pair()
    res = cg.engle_granger(y, x)
    assert res.pvalue < 0.05
    assert res.is_cointegrated
    assert set(res.critical_values) == {"1%", "5%", "10%"}


def test_adf_test_on_residuals():
    y, x = _log_pair()
    hedge = cg.estimate_hedge_ratio(y, x)
    out = cg.adf_test(hedge.residuals)
    assert out["pvalue"] < 0.05
    assert out["is_stationary"]


def test_ols_hedge_ratio_recovers_beta():
    y, x = _log_pair()
    hedge = cg.estimate_hedge_ratio(y, x, method="ols")
    assert hedge.beta == pytest.approx(1.3, abs=0.1)
    assert hedge.r_squared > 0.9
    assert len(hedge.residuals) == len(y)


def test_tls_close_to_ols():
    y, x = _log_pair()
    ols = cg.estimate_hedge_ratio(y, x, method="ols")
    tls = cg.estimate_hedge_ratio(y, x, method="tls")
    assert tls.beta == pytest.approx(ols.beta, abs=0.1)


def test_rolling_hedge_ratio_shape_and_value():
    y, x = _log_pair(n=1000)
    roll = cg.rolling_hedge_ratio(y, x, window=250)
    assert list(roll.columns) == ["alpha", "beta"]
    assert len(roll) == len(y)
    assert roll["beta"].iloc[:249].isna().all()
    assert roll["beta"].iloc[-1] == pytest.approx(1.3, abs=0.2)


def test_unknown_method_raises():
    y, x = _log_pair(n=100)
    with pytest.raises(ValueError):
        cg.estimate_hedge_ratio(y, x, method="wls")
