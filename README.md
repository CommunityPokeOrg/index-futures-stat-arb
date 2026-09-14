# index-futures-stat-arb

Research scaffold for statistical arbitrage between the E-mini S&P 500 (ES) and
E-mini Nasdaq-100 (NQ) index futures. This is a **research-only** project: there is
no live trading and no order placement anywhere in this codebase.

## Research goals

- Test whether the ES/NQ pair is cointegrated and the residual spread is mean-reverting.
- Estimate a stable hedge ratio and characterise the spread as an
  Ornstein–Uhlenbeck process.
- Evaluate a simple z-score entry/exit strategy strictly out of sample via a
  train/test walk-forward split.

## Method overview

1. **Cointegration** — Engle–Granger test on log prices; ADF test on residuals.
2. **Hedge ratio** — OLS or total least squares on log prices; rolling beta.
3. **Spread / z-score** — `spread = log(y) - alpha - beta * log(x)`, normalised
   by full-sample or rolling statistics.
4. **Ornstein–Uhlenbeck fit** — AR(1) calibration giving mean-reversion speed,
   long-run mean, and half-life.
5. **Out-of-sample backtest** — signals generated on the test window only;
   PnL computed on lagged positions with optional transaction costs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[data,notebook,dev]"
cp .env.example .env   # fill in DATABENTO_API_KEY if using Databento
```

## Usage

Open the research notebook:

```bash
jupyter notebook notebooks/es_nq_stat_arb_research.ipynb
```

Or use the library directly:

```python
from index_futures_stat_arb import load_prices, walk_forward_backtest

prices = load_prices(start="2018-01-01")          # yfinance ES=F / NQ=F
results = walk_forward_backtest(prices, split=0.7)
print(results["test"].metrics)
```

## Data

Two ingestion paths are provided:

- **yfinance proxies** (default): `ES=F` / `NQ=F` continuous front-month futures
  tickers, falling back to `SPY` / `QQQ` ETFs if the futures tickers are
  unavailable.
- **Databento**: real CME Globex data via `DATABENTO_API_KEY` and
  `DATABENTO_DATASET` (default `GLBX.MDP3`), using continuous contracts
  (`stype_in="continuous"`).

Caveats to keep in mind:

- Continuous contracts have **roll artifacts**; the PnL of a naive spread is not
  exactly tradable.
- yfinance data quality is best-effort; expect gaps and occasional bad prints.
- ETF proxies introduce **survivorship/tracking** differences vs futures.
- No tick data — daily bars only, so intraday execution is not modelled.

## Project layout

```
data/                       # local data artifacts (gitignored)
notebooks/                  # research notebooks
src/index_futures_stat_arb/ # library: data, cointegration, ou, signals, backtest
tests/                      # pytest suite (offline, synthetic data)
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

All tests run offline against synthetic cointegrated data; no network required.

## Risk disclaimer

This project is for research and educational purposes only. It is **not
financial advice** and must not be used for live trading. Futures trading
involves substantial risk of loss.

## License

MIT — see [LICENSE](LICENSE).
