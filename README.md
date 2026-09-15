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

The reproducible command-line pipeline is configured with TOML files:

```bash
ifsa ingest --config configs/ingest_example.toml --offline --no-resume
ifsa rolls --config configs/ingest_example.toml --rule volume \
  --out data/reference/roll_calendar
ifsa simulate --config configs/sim_synthetic.toml --offline --out data/results
```

The original research helpers and notebooks remain available for exploratory
cointegration, OU, and walk-forward work.

## Data

The ingestion pipeline requests per-contract CME `ohlcv-1m` data, normalizes
timestamps and prices, labels CME sessions/RTH, validates OHLC and duplicate
keys, and writes zstd-compressed Hive-partitioned Parquet:

```text
data/raw/source=<source>/dataset=<dataset>/schema=<schema>/
  product=ES/contract=ESH6/date=YYYY-MM-DD/part-0.parquet
```

Use `DATABENTO_API_KEY` in the environment for the real Databento path; keys
are never written to manifests or logs. Chunk requests are session-aligned and
resume from atomic checkpoints. Each completed dataset has a manifest containing
the deterministic dataset ID, row count, configuration, and SHA-256 for every
Parquet file. The offline `--offline` path uses clearly labelled deterministic
synthetic fixtures and never accesses the network.

## Roll calendar and continuous series

`ifsa rolls` builds explicit per-product calendars using calendar, volume,
open-interest, or fixed-k rules. Volume and open-interest decisions use only
the prior completed session, include an earliest-roll constraint, and have a
CME-Monday calendar guard. ES and NQ calendars are joined so the joint roll
date is visible in `es.parquet`, `nq.parquet`, and `joint.parquet`.

`build_continuous` supports none, Panama, and ratio adjustments, with
as-of-dated and forward-adjusted modes. Simulation uses forward adjustment so
historical prices do not change when later rolls are discovered.

## Simulation

`ifsa simulate` loads aligned ES/NQ bars, resamples 1-minute data (default
5-minute bars), optionally filters RTH, fits a log-price hedge ratio using only
completed sessions, and runs an incremental signal/execution loop. A decision
made at bar `i` close fills at bar `i + signal_lag_bars` open, then marks at the
bar close. Roll close/reopen trades occur on the first bar of a roll session.
Costs include configurable commission, exchange fees, tick slippage, and
half-spread assumptions. Fixed-contract, dollar-neutral, and volatility-target
sizers are available.

The engine has a cursor-based future-access guard and the test suite includes
deterministic no-lookahead checks. A run writes `report.json`, `trades.csv`,
`equity.csv`, and `daily.csv` below a timestamp/config-hash run directory.

Synthetic output is for pipeline and test validation only. It is **not evidence
of a tradeable edge**, realistic market impact, or expected live performance.

## Real data via Yahoo Finance (no API key)

Yahoo's continuous front-month ES=F/NQ=F series can be fetched without an API
key:

```bash
ifsa simulate-yahoo --config configs/sim_yahoo_daily.toml --out data/results
ifsa simulate-yahoo --config configs/sim_yahoo_5m.toml --out data/results
```

Supported intervals and Yahoo retention limits are:

| Interval | Maximum history | Maximum request span |
|---|---:|---:|
| 1m | 30 days | 7 days |
| 2m, 5m, 15m, 30m | 60 days | 60 days |
| 60m, 1h | 730 days | 730 days |
| 1d | unlimited | unlimited |

The adapter clips requests to these limits, splits long requests into bounded
windows, validates bars, and caches Parquet plus metadata sidecars. Yahoo does
not provide the per-contract volume needed for this project's explicit roll
calendar, so these are Yahoo's own continuous contracts. Prices are unadjusted
and may contain roll gaps, and Yahoo's roll timing is not necessarily the CME
calendar used by the per-contract pipeline. In particular, 1-minute history is
only approximately 30 days.

Reports identify the source explicitly with `data_source` and include the
effective range and fetch metadata. Synthetic and Yahoo results must not be
treated as interchangeable evidence of a tradeable edge.

## Refined execution methods

The execution engine supports fixed session OLS, a random-walk Kalman hedge,
and rolling Engle–Granger hedge estimates. Optional cointegration and OU
half-life gates block new entries when the fitted relationship is not usable;
exits, stops, roll handling, and delayed fills remain active. OU threshold mode
adapts the spread mean and stationary scale without tuning thresholds for
profitability. Reports record hedge estimates, gate reasons, OU diagnostics,
and entry-gating fractions. Volatility targeting uses the prior-bar hedge ratio
to estimate the PnL of a dollar-neutral ES/NQ unit.

Basis pairs can use `hedge_method = "unit"`, which fixes the log hedge ratio
at one and tracks only the log-basis level with a one-state Kalman filter.
This avoids the alpha/beta identification problem when the spot log price
barely moves. `max_leg_notional_usd` applies a proportional post-scaling
leverage cap to both legs.

Method sources: Kalman hedge (Elliott, van der Hoek & Malcolm 2005; Chan 2013), OU
s-scores and half-life thresholds (Avellaneda & Lee 2010; Bertram 2010; Leung & Li 2015,
arXiv:1411.5062), cointegration gating (Engle & Granger 1987; Vidyamurthy 2004), survey
(Krauss 2017). Real-data baseline-vs-refined evidence and limitations:
`docs/results/2026-09-14_refined_method_real_data.md`.

## Index-vs-ETF basis

The Yahoo path also supports cash-and-carry pairs with the index future as
leg A and its ETF as leg B: ES=F/SPY and NQ=F/QQQ. ES is $50 per index point
and NQ is $20 per point; SPY and QQQ are modelled as $1 per share with
$0.01 ticks. Dollar-neutral sizing converts futures notional into ETF shares
using both prices and multipliers, preserving fractional hedge units until
the final integer fill.

When enabled, carry uses a causal one-session-lagged ^IRX 13-week T-bill
discount-yield approximation and trailing per-share ETF dividends by ex-date:
`fair future = spot * exp((r-q)*tau)`, with ACT/365 time to the next quarterly
expiry. ETF fills use one tick of slippage plus a half-tick spread (1.5 cents)
and a $0.005/share commission. Fallback constants are 4% risk-free, 1.2% SPY
dividends, and 0.6% QQQ dividends when Yahoo rate or dividend data cannot be
fetched.

```bash
ifsa diagnose-yahoo --config configs/sim_yahoo_es_spy_daily_refined.toml \
  --out data/results/diag_es_spy_daily
ifsa simulate-yahoo --config configs/sim_yahoo_es_spy_daily_refined.toml \
  --out data/results
ifsa compare --runs data/results/<run1> data/results/<run2> \
  --out data/results/compare_etf.md
```

Limitations include Yahoo ES=F being an unadjusted continuous front-month
series; SPY daily closes at 16:00 ET versus the ES=F daily settlement label;
the ^IRX discount-yield approximation; dividends applied by ex-date; roughly
60-day 5-minute retention; no ETF borrow or financing cost; and no margin
modelling.

## Project layout

```text
configs/                    # ingestion, roll, and simulation TOML
data/                       # local artifacts, manifests, and checkpoints
src/index_futures_stat_arb/ # ingestion, contracts, rolls, continuous, execution
tests/                      # offline unit and integration tests
```

## Tests

```bash
pytest -q
ruff check src tests
ruff format --check src tests
mypy src
```

All tests are offline and use deterministic fixtures or hand-built bars.

## Limitations

- No live Databento run is performed in this environment.
- The CME holiday calendar is an explicit approximation; early closes are not
  modelled.
- The default one-tick slippage and half-spread values are assumptions.
- Intrabar fills, limit orders, and queue position are not modelled.
- Daily settlement differs from close-to-close marking and is not modelled.

## Risk disclaimer

This project is for research and educational purposes only. It is **not
financial advice** and must not be used for live trading. Futures trading
involves substantial risk of loss.

## License

MIT — see [LICENSE](LICENSE).
