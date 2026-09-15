# Consolidated research findings, method specification, retail viability, and local continuation guide

Status: 2026-09-15, PR #2 (`devin/1789418721-databento-rolls-execution`).
Repository: https://github.com/CommunityPokeOrg/index-futures-stat-arb — PR: https://github.com/CommunityPokeOrg/index-futures-stat-arb/pull/2

This document consolidates everything produced so far so that research can continue locally
without re-deriving it. Every metric below is copied from an existing results document (cited
inline); nothing is new. **The honest conclusion stands: no demonstrated edge.** Stitched
walk-forward OOS on real Yahoo daily data: ES/SPY Sharpe −0.16; NQ/QQQ Sharpe 0.46 (t ≈ 1.4,
18 round trips); 5-minute streams uninformative (0 and 2 OOS trades on 60 days of retention).

Contents

1. Chronology and where each result lives
2. Consolidated metrics (real data, synthetic, walk-forward, Monte Carlo)
3. What was learned (diagnoses and defects fixed)
4. Method specification: Kalman filter + cross-sectional pairs engine (as implemented)
5. Retail viability analysis
6. Local continuation: environment, data, commands, reproducible experiments
7. Validation protocol and acceptance / rejection criteria
8. Assumptions and limitations register

---

## 1. Chronology and where each result lives

| Phase | Document | Data |
|---|---|---|
| Ingestion, rolls, execution harness; synthetic run | `docs/results/2026-09-14_synthetic_simulation.md` | **SYNTHETIC** fixture (no `DATABENTO_API_KEY`) |
| Keyless Yahoo adapter; naive ES/NQ z-score | `docs/results/2026-09-14_yahoo_real_data.md` | Real Yahoo `ES=F`, `NQ=F` |
| Root-cause diagnosis; Kalman / OU / risk controls on ES/NQ | `docs/results/2026-09-14_refined_method_real_data.md` | Real Yahoo |
| Refactor to index-future vs ETF basis (ES/SPY, NQ/QQQ) | `docs/results/2026-09-15_index_vs_etf_basis_real_data.md` | Real Yahoo incl. `^IRX`, SPY/QQQ dividends |
| Walk-forward hyperparameter evaluation | `docs/results/2026-09-15_walkforward_evaluation.md` | Real Yahoo daily + 5m |
| Monte Carlo / bootstrap / deflated Sharpe | `docs/results/2026-09-15_montecarlo_evaluation.md` | Resampled real PnL + **SYNTHETIC** paths |
| Next-steps validation protocol | `docs/research/next_steps_validation_protocol.md` | — |
| Data/roll methodology RFC (original) | `docs/research/01_data_ingestion_and_contract_roll_methods.md` | — |
| Interactive dashboard of walk-forward artefacts | `dashboard/README.md`; artefacts `data/results/walkforward/` | Real Yahoo |

## 2. Consolidated metrics

All backtests: $1,000,000 initial capital, 1-bar signal lag (decide at bar *i* close, fill at bar
*i+1* open), costs from §4.9. "Real" = Yahoo Finance via `yfinance` (unadjusted continuous
front-month futures; ETFs; `^IRX`; dividends). Yahoo retention: 1m ≈ 30 days, 5m ≈ 60 days.

### 2.1 Synthetic harness check (NOT market data) — `2026-09-14_synthetic_simulation.md`
298,080 synthetic bars; volume-rule rolls ES 2026-03-10, NQ 2026-03-12; PnL +$12,345, Sharpe
3.51, MaxDD 1.25 %, 82 round trips; falls to +$2,445 at 3-tick slippage. Purpose: harness
correctness only.

### 2.2 ES=F vs NQ=F, real daily 2015-01-01..2026-09-14 (2,941 rows/leg) — `2026-09-14_yahoo_real_data.md`, `2026-09-14_refined_method_real_data.md`

| Method | PnL | Return | Sharpe | MaxDD | Round trips | Win | Costs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive raw price-spread z (baseline) | −$73.7k | −29.5 %* | −0.40 | 32.6 % | 53 | 47 % | — |
| Baseline with z-window/residual bug fixed | −$1.5k | | | | | | |
| Refined: Kalman(β,α) on log prices, OU thresholds, EG gate p<0.10, 2×half-life stop, vol sizing | +$115.8k | 11.6 % | 0.57 | 2.8 % | 15 | 73 % | $9.3k |
| 5m RTH 2026-07-20..09-14 naive | −$10.2k | | −2.84 | | 24 | | |
| 5m refined | 0 trades (100 % gated) | | | | | | |

\*first run used $250k capital; all later runs use $1M.
Diagnosis: ES/NQ log prices are **not cointegrated** (EG p = 0.21 full sample; p > 0.10 in
21/24 rolling windows), β 0.42–1.12, log-ratio drift −5.3 %/yr, spread vol 8.5 %/yr. The
refined +$115.8k rests on 15 trades (t ≈ 2.0) with 88 % of sessions gated — not an edge claim.

### 2.3 ES=F/SPY and NQ=F/QQQ basis, real daily 2015-01-01..2026-09-14 — `2026-09-15_index_vs_etf_basis_real_data.md`

Cointegration (daily): EG p = 3.8e-24 (ES/SPY), 1.3e-18 (NQ/QQQ); OLS β = 1.0029 / 1.0010.
Carry adjustment is what makes the 5m basis stationary: raw ADF p = 0.937 / 0.942 →
carry-adjusted 3.7e-06 / 4.0e-07. Carry-adjusted daily OU half-life ≈ 1.05 / 0.52 sessions.
Unconstrained Kalman(β,α) drifts to β = 1.42 / 1.77 (unidentified for a β ≡ 1 relationship).

| Run | Hedge / thresholds | PnL | Return | Sharpe | MaxDD | Trips | Win | Fees | Slippage | Gated |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ES/SPY baseline | OLS / fixed z | +$36,232 | 3.62 % | 0.25 | 3.52 % | 39 | 53.8 % | $799 | $5,456 | 2.0 % |
| NQ/QQQ baseline | OLS / fixed z | −$7,968 | −0.80 % | −0.05 | 6.02 % | 59 | 59.3 % | $1,587 | $6,225 | 2.0 % |
| ES/SPY Kalman(β,α) | Kalman / OU | −$59,492 | −5.95 % | −0.24 | 13.86 % | 12 | 41.7 % | $1,567 | $10,700 | 80.9 % |
| NQ/QQQ Kalman(β,α) | Kalman / OU | −$97,369 | −9.74 % | −0.31 | 13.77 % | 9 | 66.7 % | $1,385 | $5,432 | 88.7 % |
| ES/SPY refined | Unit β≡1 + Kalman level / OU + carry | +$151,838 | 15.18 % | 0.22 | 16.08 % | 27 | 55.6 % | $3,310 | $22,364 | 46.8 % |
| NQ/QQQ refined | Unit + Kalman level / OU + carry | +$113,344 | 11.33 % | 0.32 | 5.35 % | 13 | 46.2 % | $2,341 | $9,186 | 77.6 % |

5m RTH (2026-07-20..09-14): baselines −$9,506 / −$6,520 (Sharpe −19 / −12, 61 / 55 trips);
refined −$756 / −$898 (4 / 6 trips, ≈100 % gated).

### 2.4 Walk-forward, real daily, OOS span 2016-12-29..2026-09-11 (2,438 sessions) — `2026-09-15_walkforward_evaluation.md`

48 seeded trials, 5 anchored folds (~488 sessions each); selection per fold from prior folds only
(median − ½·IQR of prior-fold Sharpe, min trades filter, trial 0 = base in fold 0).

| Stream | Model | OOS Sharpe | PnL | MaxDD | Win | Trips | Turnover | Fees+slip |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ES/SPY | **stitched OOS (honest)** | **−0.1614** | −$102,586 | 21.29 % | 34.6 % | 26 | 52.19 | $14,879 |
| ES/SPY | base config | −0.1571 | −$152,953 | 31.07 % | 39 % | 23 | 44.9 | $13,330 |
| ES/SPY | best in-sample, trial 47 (post-hoc) | 0.5144 | +$252,947 | 12.19 % | 62.5 % | 16 | 31.69 | $8,850 |
| NQ/QQQ | **stitched OOS (honest)** | **0.4559** | +$252,798 | 9.29 % | 61.1 % | 18 | 35.42 | $3,848 |
| NQ/QQQ | base config | 0.2748 | +$107,210 | 9.29 % | 56 % | 9 | 17.9 | $2,253 |
| NQ/QQQ | best in-sample, trial 46 (post-hoc) | 0.73 | +$320,184 | 6.1 % | 71 % | 14 | 26.2 | $2,430 |

t-stat for NQ/QQQ stitched: 0.456 × √(2438/252) ≈ 1.42. 5m: 24 trials / 3 folds on 60 days —
ES/SPY 0 OOS trades; NQ/QQQ 2 OOS trades (Sharpe 1.11 on +$553 — meaningless).

### 2.5 Monte Carlo (resampled / synthetic — harness only) — `2026-09-15_montecarlo_evaluation.md`

| Test | ES/SPY | NQ/QQQ |
|---|---:|---:|
| Block bootstrap (10,000 paths, block 14) Sharpe p5 / p50 / p95 | −0.580 / −0.156 / 0.317 | 0.057 / 0.468 / 0.929 |
| P(loss) / demeaned-null p | 71.8 % / 0.734 | 2.8 % / 0.040 |
| Stitched PSR vs 0 (daily units) | 0.301 | 0.914 |
| Best-trial Sharpe / E[max SR₄₈] / DSR | 0.942 / 0.742 / 0.687 | 0.961 / 0.710 / 0.757 |
| Minimum track record (95 %) | n/a (negative) | **3,525 sessions vs 2,438 observed** |
| Synthetic null (κ=0, 20 paths) false-positive rate | — | 0 / 20 |
| Synthetic power (κ=ln 2, 20 paths) | — | 55 % |

Daily PnL excess kurtosis ≈ 210 → bootstrap intervals under-cover. Feasibility: bootstrap 10k
paths ≈ 30 s / 450 MB; one full engine path ≈ 20 s (1,000 paths ≈ 1.4 h on 4 workers).

## 3. What was learned (diagnoses and defects)

1. **Raw price-spread z-score on ES vs NQ is invalid**: different multipliers ($50 vs $20/pt),
   index drift, β instability, non-stationarity (§2.2).
2. **Residual-window defect**: with a cross-session z-window, residuals from *previous* hedge
   fits were mixed with the current fit. Fixed (`recompute_z_window_on_refit`); legacy behaviour
   reproducible via `configs/sim_yahoo_daily_legacy.toml`. Effect alone: −$73.7k → −$1.5k.
3. **OU on Kalman innovations is wrong**: innovations are ≈ white noise → half-life ≈ 0 → all
   entries gated. OU is fitted on the *residual window under the current state* instead.
4. **Kalman(β,α) is unidentified for a cash-and-carry pair**: β drifts to 1.4–1.8 and loses.
   For F = S·e^{carry}, β ≡ 1 by construction; only the level (basis) needs filtering →
   `hedge_method="unit"` + `KalmanLevel`.
5. **Carry adjustment is essential intraday** (raw 5m basis is a random walk; adjusted is
   stationary), but the daily carry-adjusted half-life is ≈ 1 session, which is dominated by the
   ES=F (≈ 17:00 ET) vs SPY (16:00 ET) close-time mismatch and by slippage at a 1-bar lag.
6. **Notional cap must be re-applied after integer rounding**; fractional hedge units must
   survive vol scaling and be rounded last.
7. **Walk-forward win rate** must be per closed round trip, not per positive-PnL bar.
8. **PSR/DSR must use per-observation Sharpe units** (annualized SR / √252 with T sessions).

## 4. Method specification (as implemented in `src/index_futures_stat_arb/execution/engine.py`)

Legs: A = future (ES or NQ), B = hedge instrument (NQ for the ES/NQ pair; SPY/QQQ for basis
pairs). All logs are natural logs of close prices.

### 4.1 Observations
For bar *t* in session *s*:
- `log_a_t = ln(A_close_t) − carry_t` if `carry_adjust`, else `ln(A_close_t)`;
  `carry_t = (r_t − q_t)·τ_t` with `r` = `^IRX` discount yield (fallback 4 %), `q` = trailing
  12-month realised dividend yield (fallback 0.6 %), `τ` = years to the next quarterly futures
  expiry (`basis.py`). Rates/dividends are known at *t* (ex-date dividends only).
- `log_b_t = ln(B_close_t)`.

### 4.2 Hedge-ratio state (`hedge_method`)
- `ols`: per session, OLS of `log_a` on `log_b` over the last `hedge_lookback_sessions` (120)
  *completed* sessions, requiring ≥ `min_hedge_sessions` (60).
- `rolling_eg`: same OLS plus Engle–Granger p-value on the window.
- `kalman` — `KalmanHedge`, random-walk state x = (β, α):
  - state: `x_t = x_{t−1} + w_t`, `w ~ N(0, Q)`, `Q = (δ/(1−δ))·I₂`
  - observation: `log_a_t = [log_b_t, 1]·x_t + v_t`, `v ~ N(0, R)`, `R = kalman_obs_var`
  - initialised from the first valid session OLS (β₀, α₀), `P₀ = I₂`.
- `unit` — β ≡ 1, `KalmanLevel` on the basis level α:
  - state: `α_t = α_{t−1} + w_t`, `Q = δ/(1−δ)`; observation: `log_a_t − log_b_t = α_t + v_t`,
    `R = kalman_obs_var`; initialised from the mean basis over the lookback, `P₀ = 1`.
- Q/R are **not estimated**; they are configuration (`kalman_delta`, `kalman_obs_var`; defaults
  1e-5 / 1e-4, refined configs 1e-5 / 1e-6) and are swept in the walk-forward. Maximum-likelihood
  Q/R estimation is a listed next step (§7, H6), not implemented.

### 4.3 Causal update order per bar (no lookahead)
1. **Session boundary** (first bar of *s*): refit OLS/EG on completed prior sessions; create the
   Kalman state if absent; compute the session cointegration gate (`coint_pvalue_gate`: EG p for
   ols/rolling_eg, ADF p on the current-state residual window for kalman/unit).
2. **Execute pending orders** scheduled at *t − signal_lag_bars* at this bar's **open**
   (`_execute_target`), then mark PnL close-to-close, fees included.
3. **Predict**: `(ŷ, S) = predict(log_b_t)`; spread `e_t = log_a_t − α⁻ − β⁻·log_b_t` uses the
   *prior* state (β⁻, α⁻). Residual window `W_t` = last `z_window` residuals of history
   recomputed under (β⁻, α⁻), excluding *t*.
4. **OU fit on `W_t`** (`fit_ou`, AR(1): `x_{k+1} = a + b·x_k + ε`; θ = −ln b, μ = a/(1−b),
   σ_∞ = sd(ε)·√(−2 ln b/(1−b²)), half-life = ln 2/θ), requiring ≥ `ou_min_obs` (30).
   z-score `z_t = (e_t − μ)/σ_∞` (falls back to sample z on `W_t ∪ {e_t}` when OU is invalid).
   Threshold gate: `half_life_min_bars ≤ HL ≤ half_life_max_bars`, σ_∞ > 0.
5. **Append** `e_t` to history; **update** Kalman with `log_a_t` (state now (β⁺, α⁺)); this state
   is used for sizing at *t* and prediction at *t+1*.
6. **Entry filters** (entry only, flat → position): edge gate
   `|e_t − μ|·|units_A|·A_close·mult_A ≥ min_edge_cost_multiple × round-trip cost` (fees + 2×
   slippage per leg); OFI gate using the causal bar-derived proxy
   `Σ sign(close−open)·vol / Σ vol` over `ofi_window` bars, blocking longs when OFI < −θ and
   shorts when OFI > θ; roll-session mask.
7. **Signal**: long spread when `z ≤ −entry`, short when `z ≥ entry`, exit when `|z| ≤ exit`,
   stop when `|z| > stop` (re-entry blocked until `|z| < entry`), time stop when bars held >
   `max_holding_half_lives × HL` (HL fixed at entry).
8. **Sizing** (§4.8) at *t* close; order placed for `t + signal_lag_bars` (default 1).
9. Forced flat on the final bar. `_BarCursor.at(j > t)` raises `LookaheadError`;
   `assert_no_lookahead` checks that truncated/future-mutated runs reproduce the prefix.

### 4.4 Hedge / notional scaling
Unit hedge per 1 contract of A: `units_B = β · A_close · mult_A / (B_close · mult_B)`, with
`mult_ES = $50`, `mult_NQ = $20`, `mult_SPY = mult_QQQ = $1/share`. With β = 1 this is dollar
neutral (e.g. 1 ES ≈ 6,500 × 50 / 650 ≈ 500 SPY shares — illustrative). `DollarNeutralSizer`
rounds B to ≥ 1 unit and caps A at `max_units_a`.

### 4.5 Vol-targeted sizing (`VolTargetSizer`)
Realised daily PnL of *one* hedge unit over `lookback_sessions` (20) → `scale =
target_daily_vol_usd / realised` (refined: $5,000/day). Fractional units are scaled first, then
capped by `max_leg_notional_usd` ($2M/leg), then rounded to integers, then the cap is
re-checked after rounding.

### 4.6 Stops
Divergence stop `|z| > stop` (4.0); time stop at `max_holding_half_lives` (2.0) × entry
half-life; forced flat at end of data. No PnL-based stop.

### 4.7 Gates (all block *entries* only; open positions are managed by exits/stops)
`coint_pvalue_gate` (0.10), half-life bounds (refined daily 1–30 bars), `min_edge_cost_multiple`,
`ofi_threshold`, `hedge_history` (insufficient sessions), roll mask. Per-bar `gate_reason` is
written to `signals.csv`.

### 4.8 Default thresholds (refined daily configs)
`z_window` 60, `entry` 2.0, `exit` 0.5, `stop` 4.0, `hedge_lookback_sessions` 120,
`min_hedge_sessions` 60, `ou_min_obs` 30, `half_life_min/max` 1/30, `max_holding_half_lives` 2.

### 4.9 Cost model (`execution/costs.py`)
Fill = close ± (slippage_ticks + half_spread_ticks)·tick, rounded away from the trader to the
tick grid, applied at the next bar's open. Futures: tick 0.25 (ES $12.50, NQ $5.00), 1 tick
slippage + 0.5 tick half-spread, commission $1.25 + exchange fee $1.38 per contract per side.
ETFs: tick $0.01, 1 tick + 0.5 tick, commission $0.005/share, explicit fees $0. No market
impact, no borrow cost, no financing on the ETF leg, no margin interest.

### 4.10 Cross-sectional extension (not implemented)
The engine is a two-leg pair engine. A cross-sectional version (k futures vs k ETFs, or a
basket) would keep §4.3 verbatim per pair and add: a portfolio-level risk budget (sum of
per-pair `target_daily_vol_usd` ≤ portfolio target), pair-level gates evaluated independently,
and a common `SeedSequence`-driven trial sweep. Sharing Q/R across pairs is a bias-reducing
constraint (`[BIAS↓]`), fitting them per pair is `[FIT]`.

## 5. Retail viability analysis

Everything here is an assumption-laden estimate for a US retail account. Verify all broker,
exchange and tax parameters yourself; none of this is advice.

| Dimension | Finding / assumption | Implication |
|---|---|---|
| **Edge** | Not demonstrated (§2.4, §2.5). NQ/QQQ needs ~3,525 sessions of track record at Sharpe 0.46 to reach 95 % confidence; only 2,438 exist. | Do not trade this. Any live deployment is a *paper-trading experiment*, not an investment. |
| **Capital** | Backtests use $1M with a $2M/leg cap. One ES contract ≈ $300k+ notional (ES ≈ 6,000–6,500 × $50); hedging with ~500 SPY shares ≈ $300k+. Exchange initial margin for ES is on the order of $15–25k and for NQ $20–35k (**assumption**; check CME/broker), the ETF leg needs full cash or Reg-T margin (50 %). | Minimum realistic account for 1 hedge unit ≈ $200k+; the vol-target sizer at $5k/day needs several units. At $25–50k the strategy cannot be run with a cash ETF hedge; MES/MNQ micro contracts (1/10 size) would reduce the granularity problem but were not modelled. |
| **Costs** | Model: $2.63/contract/side + 1.5 ticks; ETF $0.005/share + $0.015/share. Real retail: commissions similar, but slippage at 500–2,000 share ETF orders and 1–10 ES contracts is realistic; *short* ETF legs incur borrow/hard-to-borrow fees and margin interest not modelled. | Fees+slippage were $3.8k–14.9k on $1M in the walk-forward; at retail scale costs are a larger share of a smaller PnL. Break-even slippage multiple (§7) has not been measured. |
| **Data quality** | Yahoo `ES=F` is an unadjusted continuous front-month series with an unverified close stamp (≈ 17:00 ET vs 16:00 ET for SPY). 1m retention ≈ 30 d, 5m ≈ 60 d. | The daily basis "signal" may be a timestamp artefact. Institutional bars (Databento GLBX.MDP3 + XNAS) are required before believing anything (§7). |
| **Execution** | Two-leg entry with 1-bar lag; legs are not filled atomically; no partial-fill or leg-risk model. | Retail leg risk (one leg fills, the other slips) is real and unmodelled; a synthetic-spread order type does not exist across CME and an equity venue. |
| **Leverage / margin** | Futures are inherently levered; the ETF hedge is not. Margin calls on the futures leg while the ETF leg is cash-heavy is an operational risk. Overnight margin > intraday. | Effective leverage on the futures leg ≈ 15–20×; a 5 % gap move on $300k notional is $15k against a $20k margin. |
| **Tax (US, assumption)** | Futures: IRC §1256 60/40 treatment, mark-to-market yearly. ETFs: short-term capital gains, wash-sale rules on the frequently-traded ETF leg. Mixed straddle rules may apply to offsetting positions. | Asymmetric tax treatment of the two legs can turn a small pre-tax gain into an after-tax loss; consult a professional. |
| **Operational** | Daily-bar strategy needs orders at/near the close on two venues; roll weeks (quarterly) require rolling the futures leg; dividends/ex-dates and rate changes shift the fair basis. | Automation or strict discipline required; roll and dividend handling were modelled but not stress-tested live. |
| **Statistical caveats** | 18–26 OOS trades; per-fold Sharpe SE ≈ 0.7; excess kurtosis ≈ 210 (a few sessions carry the PnL); best-trial DSR 0.69–0.76 after 48 trials. | Any "improvement" found by more sweeping on this sample is expected to be noise (E[max SR₄₈] ≈ 0.71–0.74 under the null). |

**Verdict for a retail participant:** not viable as an investment on current evidence. The
defensible retail use of this repository is educational: a reproducible research harness with
honest walk-forward and Monte Carlo tooling.

## 6. Local continuation

### 6.1 Environment
```bash
git clone https://github.com/CommunityPokeOrg/index-futures-stat-arb.git
cd index-futures-stat-arb
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # pandas, numpy, scipy, statsmodels, pyarrow, yfinance, pytest, ruff, mypy
cp .env.example .env               # DATABENTO_API_KEY optional
pytest -q                          # 138 passed at a830e95; a live-network Yahoo test is deselected by default
ruff check src tests && ruff format --check src tests && mypy src
cd dashboard && npm ci && npm test # builds and validates the static dashboard
```

### 6.2 Data acquisition
- **Yahoo (no key)**: fetched and cached automatically by any `simulate-yahoo` /
  `diagnose-yahoo` / `walkforward` run under `data/yahoo/` (parquet + sha256 sidecars). Daily
  history is unlimited; 5m ≈ 60 days; 1m ≈ 30 days.
- **Databento (key required)**: `ifsa ingest --config configs/ingest_example.toml` writes
  session-partitioned zstd Parquet + manifest with checksums, resumable via checkpoint. Use
  GLBX.MDP3 for ES/NQ; SPY/QQQ require an equities dataset (XNAS.ITCH or consolidated) — the
  adapter is generic over `BarClient`. Without a key the deterministic synthetic fixture is used
  and labelled `data_source = synthetic`.

### 6.3 Reproducible experiments (all deterministic; run ids = UTC timestamp + config hash)
```bash
# diagnostics: cointegration, ADF, OU half-life, rolling β, Kalman β drift
ifsa diagnose-yahoo --config configs/sim_yahoo_nq_qqq_daily_refined.toml --out data/results/nq_qqq_diag

# single backtests and baseline-vs-refined comparison
ifsa simulate-yahoo --config configs/sim_yahoo_nq_qqq_daily.toml         --out data/results/nq_qqq_base
ifsa simulate-yahoo --config configs/sim_yahoo_nq_qqq_daily_refined.toml --out data/results/nq_qqq_refined
# each simulate-yahoo call writes <out>/<run_id>/{report.json,trades.csv,signals.csv,equity.csv,daily.csv}
ifsa compare --runs data/results/nq_qqq_base/<run_id> data/results/nq_qqq_refined/<run_id> --out data/results/nq_qqq_compare

# walk-forward sweep (seeded), artefacts under data/results/walkforward/<stream>/
ifsa walkforward --config configs/wf_nq_qqq_daily.toml --out data/results/walkforward/nq_qqq_daily

# Monte Carlo harness (resampled / synthetic — never evidence of edge)
ifsa montecarlo bootstrap --walkforward-dir data/results/walkforward/nq_qqq_daily --n-paths 10000 --out data/results/montecarlo/nq_qqq_daily_bootstrap
ifsa montecarlo deflate   --walkforward-dir data/results/walkforward/nq_qqq_daily --out data/results/montecarlo/nq_qqq_daily_deflate
ifsa montecarlo synthetic --config configs/sim_yahoo_nq_qqq_daily_refined.toml --kappa 0 --sigma 0.0023935 --n-paths 20 --workers 4 --out data/results/montecarlo/nq_qqq_null
```
Exact flags used for the committed artefacts are recorded in the corresponding results docs.

### 6.4 Experiment hygiene
- Change one thing per run; keep the config file in the results directory (the engine writes
  `report.json` with the config hash and `data_source`).
- Never re-run the sweep on the same daily Yahoo sample to "find a better trial" — see §5
  statistical caveats and `next_steps_validation_protocol.md` §5.
- Record every trial executed (walk-forward `trials.csv` does this) so the deflated Sharpe uses
  the true trial count.

## 7. Validation protocol and acceptance / rejection criteria

The full protocol, with `[BIAS↓]` (bias/noise-reducing) vs `[FIT]` (historical-fit-improving)
labels and hypotheses H1–H8, is in `docs/research/next_steps_validation_protocol.md`. Summary:

1. **Untouched final holdout** `[BIAS↓]`: freeze the most recent ~2 years (≈ 500 sessions) now;
   never run a sweep on it; use it once per pre-registered model.
2. **Nested anchored walk-forward** `[BIAS↓]`: selection inside each training fold only (already
   the case for the stitched series); report only stitched OOS.
3. **Purging / embargo** `[BIAS↓]`: drop `z_window + hedge_lookback_sessions` (≈ 180 sessions)
   between train end and test start so estimator warm-up does not overlap; embargo ≥ 1 max
   holding period after each test fold.
4. **Multiple testing** `[BIAS↓]`: report DSR against `E[max SR_N]` for the *true* N (48 so far
   per stream); PSR/minTRL in daily units (`ifsa montecarlo deflate`).
5. **Parameter stability** `[BIAS↓]`: require a plateau — neighbouring trials in
   (`kalman_delta`, `entry`, `half_life_max_bars`) within ±50 % must keep the OOS Sharpe sign.
6. **Turnover / cost stress** `[BIAS↓]`: rerun at 1×, 2×, 3× slippage and with ETF borrow; report
   the break-even slippage multiple.
7. **Capacity** `[BIAS↓]`: √-impact model, notional at which Sharpe halves.
8. **Regime / sub-period** `[BIAS↓]`: by realised-vol tercile, rate regime, roll week, calendar
   year; buckets < 5 trades get no Sharpe.
9. **Institutional-data replication** `[BIAS↓]`: matched-timestamp Databento bars; **nothing is
   considered validated before this step**.
10. Anything that adds features or re-sweeps the same sample is `[FIT]` and is deferred.

**Acceptance (all required, pre-registered before the holdout is touched):** holdout t-stat ≥ 2
on ≥ 30 trades; DSR ≥ 0.95 with the true trial count; parameter plateau holds; break-even
slippage multiple ≥ 2; no single fold, year or trade contributes > 50 % of PnL; result
replicates on institutional data at matched timestamps.
**Rejection:** any one criterion fails → the hypothesis is recorded as falsified in
`docs/results/` with the exact metrics, and the sample is *not* re-swept. Sharpe > 1 is not a
target and is not promised anywhere in this repository.

## 8. Assumptions and limitations register

| # | Assumption / limitation | Where it bites |
|---|---|---|
| L1 | Yahoo `ES=F`/`NQ=F` are unadjusted continuous front-month series; roll/Panama adjustment cannot be applied to them. | All real-data backtests |
| L2 | Yahoo futures close stamp vs 16:00 ET ETF close is unverified. | Daily basis half-life ≈ 1 session may be an artefact |
| L3 | 5m ≈ 60 days, 1m ≈ 30 days of retention. | Intraday results uninformative |
| L4 | No live Databento run (no key); synthetic fixture is clearly labelled. | Databento adapter untested against the real API |
| L5 | Costs: fixed ticks, no impact, no ETF borrow/financing, no margin interest, legs not atomic. | Understates retail costs |
| L6 | `^IRX` discount yield as funding rate; realised (not expected) dividends; quarterly expiry approximation; CME holidays approximated. | Carry adjustment |
| L7 | Kalman Q/R are configured, not estimated; initial covariance is identity. | Filter tuning is part of the sweep (multiple testing) |
| L8 | 18–26 OOS trades; kurtosis ≈ 210; bootstrap treats the selected model as fixed. | Every significance statement |
| L9 | Retail margin, tax and borrow figures in §5 are order-of-magnitude assumptions. | Retail viability |
| L10 | Cross-sectional / basket version is specified (§4.10) but not implemented. | Scope |

*References:* Bailey & López de Prado (2012, 2014) PSR/DSR; López de Prado (2018) *Advances in
Financial Machine Learning* (purging, embargo, CPCV); Harvey & Liu (2015) multiple testing;
Lo (2002) Sharpe statistics; Politis & Romano (1992), Künsch (1989) block bootstrap; Engle &
Granger (1987); Avellaneda & Lee (2010) statistical arbitrage.
