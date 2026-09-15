"""Dynamic hedge-ratio estimators used by the execution engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ..cointegration import engle_granger

HedgeMethod = Literal["ols", "kalman", "rolling_eg", "unit"]


@dataclass
class KalmanLevel:
    """Random-walk Kalman filter for a fixed unit hedge's log-basis level."""

    delta: float = 1e-5
    obs_var: float = 1e-4
    alpha: float = 0.0
    cov: float = 1.0
    _predicted: float | None = field(default=None, init=False, repr=False)
    _innovation_var: float | None = field(default=None, init=False, repr=False)

    def predict(self) -> tuple[float, float]:
        if not 0 < self.delta < 1:
            raise ValueError("delta must be between zero and one")
        process_var = self.delta / (1.0 - self.delta)
        self.cov += process_var
        self._predicted = self.alpha
        self._innovation_var = self.cov + self.obs_var
        return self.alpha, self._innovation_var

    def update(self, observation: float) -> float:
        if self._predicted is None or self._innovation_var is None:
            self.predict()
        assert self._predicted is not None
        assert self._innovation_var is not None
        innovation = float(observation - self._predicted)
        gain = self.cov / self._innovation_var
        self.alpha += gain * innovation
        self.cov -= gain * self.cov
        self._predicted = None
        self._innovation_var = None
        return innovation


@dataclass
class KalmanHedge:
    """Random-walk (beta, alpha) state with a scalar observation variance."""

    delta: float = 1e-5
    obs_var: float = 1e-4
    beta: float = 1.0
    alpha: float = 0.0
    cov: np.ndarray = field(default_factory=lambda: np.eye(2))
    _design: np.ndarray | None = field(default=None, init=False, repr=False)
    _predicted: float | None = field(default=None, init=False, repr=False)
    _innovation_var: float | None = field(default=None, init=False, repr=False)

    def predict(self, log_nq: float) -> tuple[float, float]:
        """Return the prior prediction and innovation variance."""
        if not 0 < self.delta < 1:
            raise ValueError("delta must be between zero and one")
        process_var = self.delta / (1.0 - self.delta)
        self.cov = np.asarray(self.cov, dtype=float) + process_var * np.eye(2)
        design = np.array([log_nq, 1.0], dtype=float)
        state = np.array([self.beta, self.alpha], dtype=float)
        predicted = float(design @ state)
        innovation_var = float(design @ self.cov @ design + self.obs_var)
        self._design = design
        self._predicted = predicted
        self._innovation_var = innovation_var
        return predicted, innovation_var

    def update(self, log_es: float, log_nq: float) -> float:
        """Update the state and return the prediction innovation."""
        if self._design is None or self._predicted is None or self._innovation_var is None:
            self.predict(log_nq)
        assert self._design is not None
        assert self._predicted is not None
        assert self._innovation_var is not None
        innovation = float(log_es - self._predicted)
        gain = self.cov @ self._design / self._innovation_var
        state = np.array([self.beta, self.alpha], dtype=float) + gain * innovation
        self.beta, self.alpha = float(state[0]), float(state[1])
        self.cov = self.cov - np.outer(gain, self._design) @ self.cov
        self._design = None
        self._predicted = None
        self._innovation_var = None
        return innovation


@dataclass(frozen=True)
class RollingEGResult:
    alpha: float
    beta: float
    pvalue: float


def rolling_engle_granger(log_es: np.ndarray, log_nq: np.ndarray) -> RollingEGResult:
    """Fit OLS and return the Engle–Granger residual p-value."""
    x = np.asarray(log_nq, dtype=float)
    y = np.asarray(log_es, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return RollingEGResult(float("nan"), float("nan"), float("nan"))
    beta, alpha = np.polyfit(x, y, 1)
    try:
        pvalue = engle_granger(pd.Series(y), pd.Series(x)).pvalue
    except Exception:
        pvalue = float("nan")
    return RollingEGResult(float(alpha), float(beta), float(pvalue))
