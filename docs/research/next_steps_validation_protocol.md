# Next steps toward a higher Sharpe — without fooling ourselves

Status: research plan, 2026-09-15. Grounded in the real-Yahoo results in
`docs/results/2026-09-15_walkforward_evaluation.md` and
`docs/results/2026-09-15_index_vs_etf_basis_real_data.md`. Nothing here promises Sharpe > 1;
the plan is a set of falsifiable steps and a protocol that would let a Sharpe > 1 claim be
believed if it ever appeared.

## 1. Where we actually stand (exact produced numbers)

Stitched walk-forward OOS, real Yahoo daily bars, $1,000,000 capital, 2016-12-29..2026-09-11
(2,438 sessions, 5 anchored folds, 48 seeded trials, selection from prior folds only):

| stream | Sharpe | t ≈ Sharpe·√years | PnL | MaxDD | round trips | win (round trips) | fees+slip |
|---|---:|---:|---:|---:|---:|---:|---:|
| ES=F / SPY | **−0.16** | −0.5 | −$102,586 | 21.3% | 26 | 34.6% | $14,879 |
| NQ=F / QQQ | **0.46** | 1.4 | +$252,798 | 9.3% | 18 | 61.1% | $3,848 |

* ES/SPY: no edge. Selection chose four distinct trials over five folds (0, 46, 47, 9, 47);
  fold Sharpes −1.38, −0.07, 0.08, 0.04, −0.61.
* NQ/QQQ: mildly positive, not significant (t ≈ 1.4; 18 trades). Fold Sharpes 0.04, 0.00
  (0 trades), 0.16, 1.59, 0.21 — half the PnL comes from one fold.
* Post-hoc "recommended" models (ES/SPY trial 47: OOS 0.51; NQ/QQQ trial 33: 0.47) and
  "best in-sample" (NQ/QQQ trial 46: IS 0.67 → OOS 0.73) use all OOS folds to pick the trial
  and are descriptions of this sample, not forecasts.
* 5-minute streams: Yahoo keeps ~60 days; ES/SPY 0 OOS trades, NQ/QQQ 2 OOS trades
  (Sharpe 1.11 on +$553). Uninformative.
* The pairs are strongly cointegrated (daily EG p ≈ 1e-24 / 1e-18, β ≈ 1.00), and the
  carry adjustment is what makes the intraday basis stationary (5m ADF p 0.94 → 4e-6). The
  *relationship* is real; a *tradable, cost-covering* deviation at Yahoo's close-to-close
  resolution has not been shown.

Power check: to distinguish a true Sharpe of 0.5 from 0 at t = 2 you need ≈ 16 years of daily
data; for Sharpe 1.0, ≈ 4 years. With ~10 years and 18–26 trades, this dataset cannot
confirm a Sharpe near 0.5 and can only reject a very large one.

## 2. Two kinds of "improvement"

Every proposed change is tagged:

* **[BIAS↓]** — reduces estimation bias, selection bias, or noise in *how we measure*.
  These can make results look worse. They are the priority.
* **[FIT]** — changes the strategy or its parameters. On this sample size these are
  indistinguishable from curve-fitting unless they pass the protocol in §4 on data that
  was not used to conceive them.

## 3. Falsifiable hypotheses

Each hypothesis states what would refute it. Hypotheses that fail are recorded, not dropped.

| # | hypothesis | tag | refuted if |
|---|---|---|---|
| H1 | The daily basis half-life (~1 session) is close-time noise; using intraday fills (e.g. 15:45 ET snap vs 16:00 close) instead of Yahoo's mismatched futures/ETF closes changes the OOS Sharpe by more than its standard error. | BIAS↓ | With aligned timestamps (needs Databento/Polygon), stitched OOS Sharpe moves by < 1 SE on both streams. |
| H2 | Modelled costs are too optimistic for ETF legs (SPY/QQQ spread + impact for ~$1M notional). | BIAS↓ | Doubling ETF slippage changes NQ/QQQ OOS Sharpe by < 0.1. |
| H3 | Selection instability (4 distinct trials in 5 folds on ES/SPY) is a symptom of a flat objective, not of regime change. | BIAS↓ | Top-decile trials by prior-fold score share < 50% of their parameter ranges across folds (real regime dependence) rather than being interchangeable. |
| H4 | A minimum-edge-over-cost filter (`min_edge_cost_multiple`) adds value by *not trading*, not by better timing. | FIT | Stitched OOS Sharpe with the filter fixed at 1.5 is not higher than without it on the untouched holdout (§4.1). |
| H5 | The OFI *proxy* from bar data carries no information; the sampled `ofi_threshold` is pure noise. | BIAS↓ | Trials with the filter on vs off show no OOS difference (they do not today: NQ/QQQ recommended has it off). Drop it unless real order-book OFI is available. |
| H6 | The NQ/QQQ 0.46 is a fold-3 (2022-10..2024-10) artefact. | BIAS↓ | Leave-one-fold-out stitched Sharpe stays > 0.3 for every dropped fold. |
| H7 | Basis dislocations cluster around futures roll weeks and index rebalances; conditioning entries on days-to-expiry improves edge per trade. | FIT | Conditional OOS Sharpe on the holdout is not higher than unconditional within 1 SE. |
| H8 | Vol-targeting the position (already implemented) is what drives the drawdown difference between base and selected trials, not the signal. | BIAS↓ | Re-running selected trials with fixed sizing gives the same Sharpe ordering. |

## 4. Validation protocol (in order; each gate must pass before the next is worth running)

### 4.1 Untouched final holdout [BIAS↓]
Freeze `2024-10-02..2026-09-11` (fold 4, 488 sessions) plus everything that arrives after
today as a holdout. It has already been *seen* once in the walk-forward tables, so it is not
truly clean; treat the walk-forward as the development set and the *next* 2 years of new
data as the only truly untouched holdout. Rule: the holdout is evaluated exactly once per
pre-registered strategy version; the version and parameter grid are committed before the run.

### 4.2 Nested anchored walk-forward [BIAS↓]
Current design selects per fold from prior folds (outer loop) but the trial *grid* and the
selection *rule* were chosen by us with knowledge of the whole sample. Nest it: inside each
train fold, run an inner anchored split to choose the rule (median − ½ IQR vs. median vs. min)
and the trade-count gate. Report only the outer stitched series.

### 4.3 Purging and embargo [BIAS↓, relevant only for overlapping labels]
Train/test folds are contiguous with no overlap and positions are forced flat at fold
boundaries, so leakage is limited to Kalman/OU state carried across the boundary. Add an
embargo of `max(half_life_max_bars, z_window)` sessions (≈ 60 sessions) after each test
block before it may enter the next training set. Purging matters more once labels are
multi-day forward returns (H7) — apply de Prado's purged k-fold there.

### 4.4 Multiple-testing correction and deflated Sharpe [BIAS↓]
48 trials per stream, ~10 sampled parameters each. Report per stream: the number of trials
`N`, the variance of trial OOS Sharpes, and the **Deflated Sharpe Ratio** (Bailey & López de
Prado 2014) of the best trial against `E[max SR_N]`. Also a Bonferroni/Holm-adjusted t on the
stitched series. `ifsa montecarlo deflate` computes this from `trials.csv` (see §6). Acceptance:
DSR ≥ 0.95 *and* the stitched (not best-trial) t ≥ 2.

### 4.5 Parameter stability [BIAS↓]
For the selected trial in each fold, show the OOS Sharpe surface over ±25% perturbations of
`entry`, `exit_ratio`, `kalman_delta`. A defensible optimum is a plateau: median Sharpe over
the neighbourhood within 0.2 of the point estimate. A spike is discarded regardless of its
value.

### 4.6 Turnover and cost stress [BIAS↓]
Rerun stitched series at 1×, 1.5×, 2×, 3× modelled slippage and at ETF half-spread of
1, 2, 3 cents. Report the slippage multiple at which Sharpe crosses zero (break-even
multiple). Today: turnover 52 (ES/SPY) / 35 (NQ/QQQ) × $1M, fees+slip $14.9k / $3.8k.
Acceptance: break-even multiple ≥ 2.

### 4.7 Capacity [BIAS↓]
At $1M the ETF leg is on the order of 1,500–2,500 SPY/QQQ shares per unit — negligible vs. ADV.
Scale to $10M/$50M with a square-root impact model (`k·σ·√(shares/ADV)`) and report the
notional at which Sharpe halves. This is a stress test, not a forecast.

### 4.8 Regime and sub-period analysis [BIAS↓]
Split OOS by: VIX tercile (proxied by 21-day realized vol of SPY), rate regime (^IRX < 1% vs
> 3%), roll week vs. non-roll week, and calendar year. Report trades, Sharpe, and PnL per
bucket. Purpose is to *find where it breaks*; a bucket with < 5 trades gets no Sharpe.

### 4.9 Replication on institutional-quality data [BIAS↓]
Yahoo's ES=F/NQ=F daily "close" most likely reflects the CME session end (≈ 5pm ET) while the
SPY/QQQ close is 4pm ET; we have not verified Yahoo's exact stamping, but any such mismatch
can by itself generate a spurious mean-reverting "basis" at daily resolution. Replicate the
daily and (properly) the intraday study with Databento GLBX.MDP3 (ES/NQ) and XNAS/ARCX
(SPY/QQQ) bars at matched timestamps; the ingestion pipeline and roll logic for this already
exist in `ingest/databento.py` and `rolls.py` and need only a `DATABENTO_API_KEY`. Nothing in
this repository should be considered validated until this step is done.

## 5. What we will not do

* Sweep more parameters on the same daily Yahoo sample. Every additional trial raises
  `E[max SR_N]` and lowers the DSR of whatever we find.
* Report "recommended" or "best in-sample" trials as expected performance.
* Add features (macro, sentiment, cross-asset) before H1/H2 and §4.9 are settled — they cannot be
  evaluated on 18–26 trades.

## 6. Tooling in this repository that supports the protocol

* `ifsa walkforward` — anchored folds, prior-fold selection, deterministic trials
  (`src/index_futures_stat_arb/walkforward.py`).
* `ifsa montecarlo bootstrap|synthetic|deflate` — circular block bootstrap of realized daily
  PnL (confidence intervals, loss probabilities, demeaned null), synthetic OU/random-walk
  harness for power and false-positive rate, and Deflated Sharpe from `trials.csv`
  (`src/index_futures_stat_arb/montecarlo.py`; see `docs/results/2026-09-15_montecarlo_evaluation.md`).
  These are **harness/power/stress tools**, not evidence of edge.
* Dashboard (`dashboard/`, GitHub Pages) — renders the committed artefacts only; it will show
  holdout and stress results once they exist as artefacts.

## 7. Acceptance criteria for calling anything "an edge"

All of: stitched OOS t ≥ 2 on the pre-registered holdout; DSR ≥ 0.95; parameter plateau
(§4.5); break-even slippage multiple ≥ 2; no single fold or bucket contributing > 50% of
PnL; replicated on matched-timestamp institutional data. Until then the honest summary is:
ES/SPY OOS Sharpe −0.16 (no edge); NQ/QQQ 0.46 (t ≈ 1.4, not significant); 5m uninformative.

## References

* Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio. *J. Portfolio Mgmt*.
* Bailey, Borwein, López de Prado, Zhu (2014). Pseudo-Mathematics and Financial Charlatanism.
* López de Prado, M. (2018). *Advances in Financial Machine Learning* (purged k-fold, embargo, CPCV).
* Harvey, C. R., & Liu, Y. (2015). Backtesting. *J. Portfolio Mgmt* (multiple-testing haircuts).
* Lo, A. W. (2002). The Statistics of Sharpe Ratios. *FAJ* (SE ≈ √((1+SR²/2)/T)).
* Politis & Romano (1994). The Stationary Bootstrap; Künsch (1989) block bootstrap.
