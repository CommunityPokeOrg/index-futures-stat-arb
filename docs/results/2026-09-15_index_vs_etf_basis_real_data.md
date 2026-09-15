# Index-future vs ETF basis arbitrage on real Yahoo data (ES=F/SPY, NQ=F/QQQ)

Date: 2026-09-15. Commits `b8d0af4`, `3ffe6c7`, `97ee69c`, `e686e3a` on PR #2.
All runs below use **real Yahoo Finance data** (no synthetic fixtures) and exactly
**$1,000,000 initial capital**. Nothing in this document is a tradeable-edge claim.

## Data

| Series | Yahoo symbol | Interval | Rows | Range | Source |
|---|---|---|---:|---|---|
| ES future (continuous front month) | `ES=F` | 1d | 2941 | 2015-01-02..2026-09-11 | Yahoo cache `data/yahoo/yahoo/interval=1d/product=ES` |
| SPY | `SPY` | 1d | 2940 | 2015-01-02..2026-09-11 | Yahoo |
| NQ future (continuous front month) | `NQ=F` | 1d | 2941 | 2015-01-02..2026-09-11 | Yahoo |
| QQQ | `QQQ` | 1d | 2940 | 2015-01-02..2026-09-11 | Yahoo |
| 13-week T-bill discount yield | `^IRX` | 1d | 2939 | 2015-01-02..2026-09-11 (min −0.105%, max 5.348%) | Yahoo, `kind=rate` cache |
| SPY / QQQ dividends (ex-date) | `Ticker.dividends` | events | 46 / 46 | 2015..2026 | Yahoo |
| All four bar series | | 5m RTH | 3042 merged | 2026-07-20..2026-09-11 | Yahoo (60-day retention) |

Merged daily pair frames: 2938 rows each (inner join on timestamp). 5-minute pair
frames: 3042 rows each (10,643–10,644 raw bars per leg before RTH alignment).

Query parameters: `yf.download(symbol, start, end, interval, auto_adjust=False,
progress=False, threads=False)`; rates always fetched at `1d` (`fetch_yahoo_rate`);
dividends via `yf.Ticker(symbol).dividends`. Every refined run records
`carry_source = "yahoo_irx_dividends"`; the constant fallback (`r=4%`, `q=1.2%/0.6%`)
was **not** used in any reported run (a first attempt fell back because the
validator rejected the ≤0 bill yields of 2020–21; fixed in `3ffe6c7`).

## Method

* Legs: A = future (`ES`, $50/pt; `NQ`, $20/pt), B = ETF (`SPY`, `QQQ`, $1/share).
  Sizing is notional-matched: `shares_B = beta · n_A · F · mult_A / S`
  (≈500 SPY per ES, ≈820 QQQ per NQ at 2026 prices), fractional before vol scaling,
  rounded after; `max_leg_notional_usd = 2,000,000` caps either leg (2× capital).
* Carry: `carry_t = (r_t − q_t)·τ_t`, `r` = previous session's ^IRX/100,
  `q` = trailing-365-day dividends / spot (known by ex-date), `τ` = ACT/365 to the
  next quarterly third-Friday expiry. Regression variable `y = log F − carry`,
  `x = log S`; theoretical `y = x + const`.
* Hedge methods compared: `ols` (baseline, 120-session rolling OLS, fixed z),
  `kalman` (random-walk (β, α)), and `unit` (β ≡ 1, α tracked by a 1-state
  random-walk Kalman level filter, δ = 1e-5, obs var 1e-6). Predict-before /
  update-after ordering keeps every spread value free of current-bar hedge information.
* Refined thresholds: OU/AR(1) fit on the trailing residual window (excluding the
  current bar) → z = (s − μ)/σ_stationary, entry 2.0 / exit 0.5 / divergence stop 4.0,
  half-life gate 1–30 bars (daily) or 1–78 bars (5m), time stop at 2 × half-life,
  cointegration gate p < 0.10 (rolling EG for `kalman`, ADF of the unit residual for `unit`).
* Costs: futures $1.25 commission + $1.38 fees per contract, 1 tick slippage + 0.5
  tick half-spread; ETF $0.005/share, 1¢ slippage + 0.5¢ half-spread. Signal lag 1 bar.

Commands (all reproducible from the cached data):

```
ifsa diagnose-yahoo --config configs/sim_yahoo_{es_spy,nq_qqq}_{daily,5m}_refined.toml --out data/results/diag_...
ifsa simulate-yahoo --config configs/sim_yahoo_es_spy_daily.toml            --out data/results
ifsa simulate-yahoo --config configs/sim_yahoo_es_spy_daily_kalman.toml     --out data/results
ifsa simulate-yahoo --config configs/sim_yahoo_es_spy_daily_refined.toml    --out data/results
ifsa simulate-yahoo --config configs/sim_yahoo_nq_qqq_daily{,_kalman,_refined}.toml --out data/results
ifsa simulate-yahoo --config configs/sim_yahoo_{es_spy,nq_qqq}_5m{,_refined}.toml   --out data/results
ifsa compare --runs <6 daily dirs> --out data/results/compare_etf_daily.md
ifsa compare --runs <4 5m dirs>    --out data/results/compare_etf_5m.md
```

## Cointegration / stationarity diagnostics (`ifsa diagnose-yahoo`)

| | ES/SPY 1d | NQ/QQQ 1d | ES/SPY 5m | NQ/QQQ 5m |
|---|---:|---:|---:|---:|
| OLS β (y on x) / α | 1.0029 / 2.287 | 1.0010 / 3.709 | 0.9621 / 2.557 | 0.9861 / 3.808 |
| Engle–Granger p | 3.8e-24 | 1.3e-18 | 1.6e-05 | 2.0e-06 |
| ADF p, raw `log F − log S` | 7.8e-08 | 1.5e-07 | 0.937 | 0.942 |
| ADF p, carry-adjusted basis | 6.8e-17 | 2.4e-15 | 3.7e-06 | 4.0e-07 |
| OU half-life, raw basis (bars) | 3.44 | 2.44 | 3123 | 1496 |
| OU half-life, carry-adjusted (bars) | 1.05 | 0.52 | 79.2 | 60.4 |
| Rolling-window EG p<0.05 / p<0.10 | 57% / 66% | 85% / 94% | 0% / 0% | 0% / 0% |
| Rolling β min / median / max | 0.962 / 0.997 / 1.026 | 0.979 / 1.000 / 1.012 | 0.978 / 0.999 / 1.082 | 0.984 / 0.998 / 1.032 |
| Kalman (β,α) β min / median / max | 0.982 / 0.987 / **1.418** | 0.972 / 0.988 / **1.766** | **1.313 / 1.322 / 1.342** | **1.429 / 1.460 / 1.555** |
| Unit-hedge α min / median / max | 2.291 / 2.305 / 2.361 | 3.693 / 3.715 / 3.767 | 2.304 / 2.306 / 2.313 | 3.715 / 3.716 / 3.723 |

Findings:

1. Unlike ES/NQ (EG p ≈ 0.21), both future/ETF pairs are strongly cointegrated with
   β ≈ 1 in log space, as cash-and-carry implies.
2. Carry adjustment matters intraday: the raw 5m basis is non-stationary (ADF p ≈ 0.94,
   the (r−q)·τ term drifts within the 60-day window) while the carry-adjusted basis is
   stationary (p ≈ 1e-6) with a 60–80 bar (5–6.5 h) half-life.
3. On daily bars the carry-adjusted basis half-life is ≈ 0.5–1 session: deviations
   are gone by the next close, i.e. mostly close-time mismatch and bid/ask noise, not a
   tradeable signal with a one-bar lag.
4. The two-state Kalman (β, α) filter is **unidentified** on these pairs: `log S`
   barely moves within a window, so β and α trade off and β wanders to 1.3–1.8 while
   OLS/rolling β sits at 0.96–1.03. This is why the `unit` hedge (β ≡ 1) was added.

## Backtests — daily, $1,000,000, 2015-01-01..2026-09-14 (2941 bars/leg)

| Run (config) | Hedge / thresholds | Net PnL | Return | Ann. vol | Sharpe | Max DD | Round trips | Win rate | Avg hold (bars) | Fees | Slippage | Entries gated |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ES/SPY baseline (`sim_yahoo_es_spy_daily`) | OLS / fixed z | +$36,232 | 3.62% | 1.22% | 0.25 | 3.52% | 39 | 53.8% | 8.3 | $799 | $5,456 | 2.0% |
| NQ/QQQ baseline (`sim_yahoo_nq_qqq_daily`) | OLS / fixed z | −$7,968 | −0.80% | 1.51% | −0.05 | 6.02% | 59 | 59.3% | 6.6 | $1,587 | $6,225 | 2.0% |
| ES/SPY Kalman (`..._daily_kalman`) | Kalman(β,α) / OU | −$59,492 | −5.95% | 2.14% | −0.24 | 13.86% | 12 | 41.7% | 4.2 | $1,567 | $10,700 | 80.9% |
| NQ/QQQ Kalman (`..._daily_kalman`) | Kalman(β,α) / OU | −$97,369 | −9.74% | 2.73% | −0.31 | 13.77% | 9 | 66.7% | 3.0 | $1,385 | $5,432 | 88.7% |
| ES/SPY refined (`..._daily_refined`) | Unit / OU + carry | +$151,838 | 15.18% | 6.02% | 0.22 | 16.08% | 27 | 55.6% | 6.6 | $3,310 | $22,364 | 46.8% |
| NQ/QQQ refined (`..._daily_refined`) | Unit / OU + carry | +$113,344 | 11.33% | 3.00% | 0.32 | 5.35% | 13 | 46.2% | 2.7 | $2,341 | $9,186 | 77.6% |

Gate breakdown (refined daily): ES/SPY cointegration 14.0%, half-life 35.2%, hedge
history 2.0%; NQ/QQQ cointegration 2.0%, half-life 76.9%, hedge history 2.0%.
Max gross notional: $4.23M (ES/SPY), $4.08M (NQ/QQQ) — both legs at the $2M cap.
First refined ES/SPY trade: +18 ES @ 2109.50 / −9450 SPY @ 210.62 (≈ $1.90M per leg),
closed by time stop.

Run directories: baselines `20260915T000350441517Z-0caa3503`, `20260915T000402217525Z-be9ea4ea`;
Kalman `20260915T001808866562Z-7fcf5de1`, `20260915T001818498508Z-dfa04e1e`;
refined `20260915T001733672794Z-7e7603a3`, `20260915T001743479633Z-8ccf75a4`.
Table: `data/results/compare_etf_daily.md`.

## Backtests — 5-minute RTH, $1,000,000, 2026-07-20..2026-09-14

| Run | Hedge / thresholds | Net PnL | Return | Ann. vol | Sharpe | Max DD | Round trips | Win rate | Fees | Slippage | Entries gated |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ES/SPY baseline (`sim_yahoo_es_spy_5m`) | OLS / fixed z | −$9,506 | −0.95% | 0.32% | −18.98 | 0.97% | 61 | 13.1% | $1,256 | $8,583 | 12.8% |
| NQ/QQQ baseline (`sim_yahoo_nq_qqq_5m`) | OLS / fixed z | −$6,520 | −0.65% | 0.35% | −11.89 | 0.65% | 55 | 38.2% | $1,484 | $5,808 | 12.8% |
| ES/SPY refined (`..._5m_refined`) | Unit / OU + carry | −$756 | −0.08% | 0.16% | −3.02 | 0.08% | 4 | 0.0% | $52 | $354 | 100.0%* |
| NQ/QQQ refined (`..._5m_refined`) | Unit / OU + carry | −$898 | −0.09% | 0.12% | −5.04 | 0.10% | 6 | 33.3% | $140 | $548 | 100.0%* |

\* entry-gate fraction is rounded; a handful of entries passed (4 and 6 round trips,
all 1-contract, all closed by time stop). Gates: cointegration 76.9% / 59.0%,
half-life 86.9% / 89.1%, hedge history 12.8% / 12.8%.
Run directories: `20260915T000404981289Z-d8c1aabd`, `20260915T000410043040Z-621fc6cf`,
`20260915T001736929476Z-e5a7f731`, `20260915T001739788478Z-0f6a98e2`.
Table: `data/results/compare_etf_5m.md`.

## Interpretation (baseline vs refined)

* The **statistical premise now holds**: the carry-adjusted log basis is stationary
  with β ≈ 1, in contrast to the non-cointegrated ES/NQ pair.
* The **unconstrained Kalman hedge is wrong for this problem** and loses money
  (−6% / −10%) because its drifting β manufactures spurious residual moves; imposing
  the cash-and-carry restriction β = 1 removes that failure mode.
* The **refined unit-hedge runs are positive on daily data (+15.2%, +11.3% over 11.7 y)
  but not an edge**: Sharpe 0.22 / 0.32 with max drawdowns of 16% / 5%, 27 and 13
  round trips, win rates 56% / 46%, and slippage ($22k / $9k) 2–3× the PnL per
  trade scale. The daily half-life (≈1 bar) means the strategy is trading close-time
  noise with a one-bar lag; the sample is far too small to distinguish this from zero.
* On 5m data the refined gates correctly refuse to trade almost everywhere (60 days,
  half-life ≈ 60–80 bars → OU windows rarely qualify); the fixed-threshold baselines
  lose steadily to costs (−0.95%, −0.65%, Sharpe ≪ 0). Neither shows a positive edge.

## Limitations

* `ES=F`/`NQ=F` are Yahoo's unadjusted continuous front-month proxies; roll days
  inject basis jumps (unmasked here because the Yahoo path has no per-contract volume).
* Daily ES=F close is a futures settlement-time price while SPY/QQQ close at 16:00 ET;
  the timing gap contributes to the ≈1-bar half-life "noise" and would not be tradeable.
* ^IRX is a 13-week T-bill *discount* yield, used as a proxy for the funding rate of
  the implied repo; dividends use trailing-365-day realised payouts, not expected
  dividends to expiry. No ETF borrow/financing cost, no futures margin, no
  intraday-only ETF trading constraints (ES trades 23 h; SPY only RTH) are modelled.
* Yahoo 5m retention is ~60 days (1m ≈ 30 days), so intraday evidence is one summer.
* Slippage is a fixed 1 tick + half spread; 9,000–22,000-share ETF orders and 20–50
  contract futures orders would incur market impact beyond that.
* No out-of-sample / walk-forward split; parameters were fixed a priori but only one
  configuration per method was run (no sensitivity analysis).

## Validation evidence

```
pytest -q                     -> 127 passed, 1 deselected (network test deselected)
pytest -q -m network tests/test_yahoo.py -> live Yahoo test (earlier in session): 1 passed
ruff check src tests          -> All checks passed
ruff format --check src tests -> 47 files already formatted
mypy src                      -> Success: no issues found in 25 source files
```

New/changed tests: `tests/test_basis.py`, `tests/test_etf_pair.py`, additions in
`tests/test_hedge.py`, `tests/test_sizing.py`, `tests/test_yahoo.py`, `tests/test_cli.py`,
plus the `a_*/b_*` column rename across the engine tests.
