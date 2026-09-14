import math

import numpy as np
import pytest

from index_futures_stat_arb import ou


def test_simulate_and_recover_params():
    params = ou.OUParams(theta=0.2, mu=0.0, sigma=1.0, dt=1.0)
    x = ou.simulate_ou(params, n=20000, seed=0)
    fit = ou.fit_ou(x)
    assert fit.theta == pytest.approx(0.2, rel=0.25)
    assert fit.mu == pytest.approx(0.0, abs=0.2)
    assert fit.sigma == pytest.approx(1.0, rel=0.25)


def test_half_life_helpers():
    params = ou.OUParams(theta=0.2, mu=0.0, sigma=1.0, dt=1.0)
    assert params.half_life == pytest.approx(math.log(2) / 0.2)
    x = ou.simulate_ou(params, n=2000, seed=0)
    assert ou.half_life(x) == pytest.approx(params.half_life, rel=0.5)


def test_nonpositive_theta_half_life_inf():
    assert ou.OUParams(theta=0.0, mu=0, sigma=1, dt=1).half_life == math.inf
    assert ou.OUParams(theta=-1.0, mu=0, sigma=1, dt=1).half_life == math.inf
    assert math.isnan(ou.OUParams(theta=0.0, mu=0, sigma=1, dt=1).stationary_std)


def test_fit_ou_random_walk_no_raise():
    rw = np.cumsum(np.random.default_rng(0).normal(size=5000))
    fit = ou.fit_ou(rw)  # b >= 1 expected -> theta 0 or nan, no exception
    assert fit.theta == 0.0 or math.isnan(fit.theta) or fit.theta >= 0


def test_short_series_raises():
    with pytest.raises(ValueError):
        ou.fit_ou(np.arange(5, dtype=float))
