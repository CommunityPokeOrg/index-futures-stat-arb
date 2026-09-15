# Walk-forward hyperparameter evaluation — ES=F/SPY and NQ=F/QQQ basis streams

Date: 2026-09-15. Commits `1a16954` (pipeline) and `c812d7b` (reporting fixes) on PR #2.
Real Yahoo data only (cached ES=F, NQ=F, SPY, QQQ, ^IRX, dividends; see
`2026-09-15_index_vs_etf_basis_real_data.md` for the data record). $1,000,000 capital.
Artifacts: `data/results/walkforward/<stream>/{trials.csv, folds.csv, selection.json,
summary.md, oos_sharpe_hist.png, stitched_equity.png, param_vs_oos.png}`.

## Pipeline (`src/index_futures_stat_arb/walkforward.py`, `ifsa walkforward`)

* **Trials.** Deterministic random search (`numpy.default_rng(seed)`), trial 0 = the base
  `*_refined` config. Sampled: Kalman `delta` (Q, log-uniform 1e-7..1e-3), `obs_var`
  (R, 1e-8..1e-4), `entry` z (1..3), `exit = exit_ratio·entry` (0..0.6), `stop = stop_ratio·entry`
  (1.5..3, the z-score bound), half-life cutoffs (`half_life_min/max_bars`), time stop
  `max_holding_half_lives` (1..4), `coint_pvalue_gate` (0.05..0.2), and two new entry-only
  execution filters: `min_edge_cost_multiple` (expected OU reversion in USD must exceed
  k × modelled round-trip fees+slippage+half-spread on both legs) and `ofi_threshold`
  (a **bar-derived order-flow-imbalance proxy**, Σ sign(close−open)·volume / Σ volume over
  `ofi_window` bars on the future leg — Yahoo has no order-book data; sampled "off" with p=0.3).
* **Causality.** Each trial is simulated once over the full history (state at any bar depends
  only on prior bars; `assert_no_lookahead` covers the engine) and its daily PnL is sliced
  into folds. Test: fold-k PnL sliced from the full run equals the PnL of a run truncated
  at fold k's end, exactly; `select(k)` is invariant to perturbing folds ≥ k.
* **Folds.** Anchored: sessions after `min_train_sessions` are split into `n_folds` equal
  contiguous test blocks; train_k = everything before test_k.
* **Selection rule (robust, no OOS peeking).** At fold k ≥ 1, candidates are trials with
  ≥ `min_trades_per_fold` in every earlier fold; score = median(train-fold Sharpe) − 0.5·IQR;
  argmax (ties → lowest id). Fold 0 uses the base config. The **stitched OOS** series is the
  honest performance of the whole procedure. Separately, a *recommended model* = argmax of
  median OOS-fold Sharpe among trials trading in ≥ ⌈n/2⌉ folds — this uses all OOS folds
  and is therefore a *description of the sample*, not an unbiased forecast; the overfit gap
  is shown by the "best in-sample" row.
* **Determinism.** Same seed ⇒ byte-identical `trials.csv`/`selection.json` (tested).

Commands:

```
ifsa walkforward --config configs/wf_es_spy_daily.toml --out data/results/walkforward/es_spy_daily   # 48 trials, 5 folds, 13m15s
ifsa walkforward --config configs/wf_nq_qqq_daily.toml --out data/results/walkforward/nq_qqq_daily   # 48 trials, 5 folds, 13m10s
ifsa walkforward --config configs/wf_es_spy_5m.toml    --out data/results/walkforward/es_spy_5m      # 24 trials, 3 folds, 33s
ifsa walkforward --config configs/wf_nq_qqq_5m.toml    --out data/results/walkforward/nq_qqq_5m      # 24 trials, 3 folds, 33s
```

## Daily streams (2015-01-02..2026-09-11; 500 warm-up sessions; 5 test folds × ~488 sessions, 2016-12-29..2026-09-11)

### Selection comparison (OOS span = all five test folds, 2438 sessions)

| stream | model | trial | Sharpe | PnL | MaxDD | win | round trips | turnover* | fees+slip |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ES/SPY | **stitched OOS (procedure)** | per fold: 0,46,47,9,47 | **−0.16** | −$102,586 | 21.3% | 34.6% | 26 | 52.2 | $14,879 |
| ES/SPY | base config OOS | 0 | −0.16 | −$152,953 | 31.1% | 39.1% | 23 | 44.9 | $13,330 |
| ES/SPY | recommended (uses OOS folds) | 47 | 0.51 | +$252,947 | 12.2% | 62.5% | 16 | 31.7 | $8,850 |
| ES/SPY | best in-sample (IS Sharpe 0.40) | 47 | 0.51 | | | | | | |
| NQ/QQQ | **stitched OOS (procedure)** | per fold: 0,46,19,33,33 | **0.46** | +$252,798 | 9.3% | 61.1% | 18 | 35.4 | $3,848 |
| NQ/QQQ | base config OOS | 0 | 0.27 | +$107,210 | 9.3% | 55.6% | 9 | 17.9 | $2,253 |
| NQ/QQQ | recommended (uses OOS folds) | 33 | 0.47 | +$250,217 | 12.1% | 66.7% | 18 | 35.5 | $4,910 |
| NQ/QQQ | best in-sample (IS Sharpe 0.67) | 46 | 0.73 | +$320,184 | 6.1% | 71.4% | 14 | 26.2 | $2,430 |

\* turnover = Σ|filled notional| / capital over the span.

### Per-fold results of the procedure

| fold | test range | ES/SPY selected → Sharpe / MaxDD / trips | NQ/QQQ selected → Sharpe / MaxDD / trips |
|---|---|---|---|
| 0 | 2016-12-29..2018-12-06 | 0 → −1.38 / 7.3% / 4 | 0 → 0.04 / 2.5% / 4 |
| 1 | 2018-12-07..2020-11-12 | 46 → −0.07 / 9.3% / 6 | 46 → 0.00 / 0% / 0 |
| 2 | 2020-11-13..2022-10-21 | 47 → 0.08 / 11.1% / 3 | 19 → 0.16 / 7.8% / 8 |
| 3 | 2022-10-24..2024-10-01 | 9 → 0.04 / 10.2% / 10 | 33 → 1.59 / 1.8% / 4 |
| 4 | 2024-10-02..2026-09-11 | 47 → −0.61 / 6.3% / 3 | 33 → 0.21 / 9.3% / 2 |

### Recommended-model parameters (for reference only)

| stream | trial | Q (delta) | R (obs var) | entry | exit/entry | stop/entry | HL min/max | max hold (HL) | edge mult | OFI thr | coint gate | fold Sharpes | top-quartile folds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| ES/SPY | 47 | 1.1e-5 | 6e-8 | 1.30 | 0.26 | 1.80 | 2.45 / 52.8 | 1.95 | 1.52 | 0.58 | 0.072 | 1.52, 0.94, 0.08, 1.38, −0.61 | 4/5 |
| NQ/QQQ | 33 | 3.8e-5 | 8e-6 | 1.79 | 0.45 | 1.85 | 0.71 / 24.4 | 2.56 | 1.28 | off | 0.079 | −0.50, 0.99, 0.96, 1.59, 0.21 | 3/5 |

## 5-minute streams (2026-07-20..2026-09-11, 3 test folds × 8 sessions)

| stream | stitched OOS Sharpe | PnL | round trips | note |
|---|---:|---:|---:|---|
| ES/SPY 5m | 0.00 | $0 | 0 | no trial produced an OOS entry (gates + 60-day sample); base IS Sharpe 1.14 on 4 trades |
| NQ/QQQ 5m | 1.11 | +$553 | 2 | two 1-lot trades (−$/+$), MaxDD 0.08%; meaningless sample size |

Yahoo's ~60-day 5m retention makes intraday walk-forward uninformative; both are reported
as executed, not as evidence.

## Findings

1. **No robust OOS edge on ES/SPY.** The honest procedure returns Sharpe −0.16 with a 21%
   drawdown; selected trials change every fold and the fold-4 pick (47) lost. Trial 47's
   0.51 "recommended" Sharpe is a post-hoc description of the same five folds.
2. **NQ/QQQ is mildly positive OOS** (stitched Sharpe 0.46, +$253k, 18 round trips, MaxDD
   9.3%) and the procedure beat the fixed base (0.27), but 18 trades over ~9.7 years give a
   t-stat ≈ 1.4; not significant.
3. **Overfit gap is visible**: best in-sample trial ≠ stitched OOS (ES/SPY 0.40 IS vs −0.16
   procedure OOS). Only 27 of 48 daily trials traded in ≥ 3 folds; medians of OOS fold Sharpes
   for the top-10 lie between 0.0 and 0.96 with per-fold ranges of ±1.5.
4. Parameter sensitivity (`param_vs_oos.png`): lower entry z (1.1–1.8) and moderate
   `min_edge_cost_multiple` (1–1.5) dominate the top-10 on both streams; Kalman Q/R have no
   visible monotone effect under the unit hedge (they only govern the level filter).

## Limitations

* 48 (daily) / 24 (5m) random trials; not an exhaustive grid. Seed 0; other seeds not run.
* 5 folds of ~2 years each with few trades per fold ⇒ fold Sharpes are noisy (±1.5).
* The recommended-model and best-in-sample rows use all OOS folds — only the stitched row
  is free of selection bias.
* OFI is a bar-derived proxy (signed volume imbalance), not order-book OFI.
* Same data caveats as the basis record: Yahoo continuous front-month futures, close-time
  mismatch (ES=F settlement vs 16:00 ET ETF close), ^IRX funding proxy, fixed-tick slippage
  without impact for 10–20k-share ETF orders, no margin/borrow.

## Validation

```
pytest -q                     -> 131 passed, 1 deselected
ruff check src tests          -> All checks passed
ruff format --check src tests -> 49 files already formatted
mypy src                      -> Success: no issues found in 26 source files
```

Tests added: `tests/test_walkforward.py` (sampling determinism/bounds/ordering, fold
construction, cross-fold causality, selection invariance, byte-identical artifacts, CLI on
offline fixture) and engine tests for the edge and OFI-proxy filters.
