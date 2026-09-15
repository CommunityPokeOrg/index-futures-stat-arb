# Monte Carlo / synthetic simulation evaluation — harness, power and stress tests only

Date: 2026-09-15. Commits `cd92ece`, `e35277d` on PR #2. Code: `src/index_futures_stat_arb/montecarlo.py`,
CLI `ifsa montecarlo bootstrap|synthetic|deflate`, tests `tests/test_montecarlo.py`.
Artifacts: `data/results/montecarlo/<run>/{montecarlo.json, montecarlo.md, paths.csv, sharpe_distribution.png}`.

**Every number in this document is a property of resampled or synthetic data. None of it is
evidence that the strategy has an edge.** The realized inputs are the stitched walk-forward OOS
daily PnL series (`data/results/walkforward/<stream>/stitched_oos_daily_pnl.csv`, real Yahoo data)
and the 48-trial `trials.csv` per stream.

## 1. Are large Monte Carlo / synthetic simulations useful here?

Yes for three narrow purposes, no for the question everyone actually cares about.

| purpose | can establish | cannot establish |
|---|---|---|
| **Block bootstrap of realized OOS PnL** | Sampling uncertainty of Sharpe/PnL/drawdown *given the trades that happened* and *treating the selected model as fixed*; probability of a loss over a same-length horizon; a null p-value against demeaned PnL. | Whether the trades would recur; anything about the 48-trial search that produced the series (selection bias is baked into the input). |
| **Synthetic OU / random-walk paths through the real engine** | That the harness has a controlled false-positive rate on a pure random walk (κ = 0) and non-trivial power when a known mean-reverting basis is injected; cost/turnover sensitivity under a parameterized spread model. | Anything about the *real* basis: the generator is our own model of the world. |
| **Deflated Sharpe from the trial distribution** | How much of the best trial's Sharpe is expected from searching 48 trials with the observed dispersion (Bailey & López de Prado 2014). | A corrected *forecast*; DSR only haircuts, it does not validate. |

What no simulation can do: increase the ~10 years / 18–26 trades of real information. The
minimum track record to accept NQ/QQQ's Sharpe 0.46 at 95% is **3,525 sessions (~14 years)**
versus 2,438 observed.

## 2. Executed runs (real inputs, resampled outputs)

### 2.1 Circular block bootstrap of stitched OOS PnL (10,000 paths, block = ⌈n^{1/3}⌉ = 14, seed 0)

```
ifsa montecarlo bootstrap --walkforward-dir data/results/walkforward/es_spy_daily --n-paths 10000 --out data/results/montecarlo/es_spy_daily_bootstrap
ifsa montecarlo bootstrap --walkforward-dir data/results/walkforward/nq_qqq_daily --n-paths 10000 --out data/results/montecarlo/nq_qqq_daily_bootstrap
```

| stream | observed Sharpe | Sharpe p5 / p50 / p95 | PnL p5 / p50 / p95 | MaxDD p50 / p95 | P(loss) | demeaned-null p |
|---|---:|---:|---:|---:|---:|---:|
| ES/SPY | −0.161 | −0.580 / −0.156 / 0.317 | −$398k / −$96k / +$173k | 24.3% / 48.7% | 71.8% | 0.734 |
| NQ/QQQ | 0.456 | 0.057 / 0.468 / 0.929 | +$31k / +$250k / +$499k | 9.3% / 16.8% | 2.8% | 0.040 |

Reading: NQ/QQQ's bootstrap interval barely excludes zero and the null p ≈ 0.04 — *conditional
on the selected trials and on these 18 trades being representative*. It ignores the 48-trial
search (see §2.2) and the daily PnL has excess kurtosis ≈ 210 (a handful of sessions carry the
result), so the block bootstrap under-covers. Treat as "not obviously luck", not "significant".

### 2.2 Deflated Sharpe (`ifsa montecarlo deflate`, daily-unit PSR/DSR, 48 trials, `median_oos_sharpe`)

| stream | stitched Sharpe | stitched PSR vs 0 | best-trial Sharpe | E[max SR₄₈] (ann.) | best-trial DSR | min track record (95%) |
|---|---:|---:|---:|---:|---:|---:|
| ES/SPY | −0.161 | 0.301 | 0.942 | 0.742 | 0.687 | n/a (negative) |
| NQ/QQQ | 0.456 | 0.914 | 0.961 | 0.710 | 0.757 | 3,525 sessions |

Reading: searching 48 trials with the observed dispersion is *expected* to produce a best
median-OOS Sharpe of ≈ 0.71–0.74 under the null; the observed best (0.94–0.96) is only
modestly above that (DSR 0.69–0.76, i.e. not close to the 0.95 acceptance bar). The stitched
NQ/QQQ PSR of 0.91 corresponds to the t ≈ 1.4 already reported.

### 2.3 Synthetic OU / random-walk harness (20 paths each, 4 workers, seed 0, NQ/QQQ refined config)

```
ifsa montecarlo synthetic --config configs/sim_yahoo_nq_qqq_daily_refined.toml --kappa 0 --sigma 0.0023935 --n-paths 20 --workers 4 --out data/results/montecarlo/nq_qqq_null
ifsa montecarlo synthetic --config configs/sim_yahoo_nq_qqq_daily_refined.toml --kappa 0.69314718056 --sigma 0.0023935 --n-paths 20 --workers 4 --out data/results/montecarlo/nq_qqq_power
```

Leg-B (QQQ) returns and volumes are jointly block-resampled from the real series (dependence
preserved); the futures leg is spot × exp(carry + synthetic basis). κ = 0 is a random-walk
basis (null); κ = ln 2 is a basis with half-life 1 session (matching the real daily estimate);
σ = 0.0023935 is the realized daily basis innovation.

| mode | paths | fraction Sharpe > 0 | mean trades | mean costs | mean Sharpe |
|---|---:|---:|---:|---:|---:|
| null (κ = 0) | 20 | **0%** (false-positive rate) | 10.5 | $13,713 | −0.53 |
| power (κ = ln 2) | 20 | **55%** | 2.8 | $3,650 | 0.02 |

Reading: the engine does not manufacture profits from a random walk (FPR 0/20), which is the
harness check we wanted. Power against a 1-session half-life basis at realistic costs is weak
(55% of paths positive, mean Sharpe ≈ 0) — consistent with the real-data finding that a
1-session half-life is too fast to trade at close-to-close with slippage. 20 paths is a
feasibility run; a real power curve needs ≥ 200 paths per (κ, σ) point.

## 3. Computational feasibility (measured, this VM)

| run | wall | peak RSS |
|---|---:|---:|
| bootstrap 10,000 paths × 2,438 sessions | 29–34 s | ~450 MB |
| deflate | 2–4 s | ~250 MB |
| synthetic, 20 full engine paths, 4 workers | 1:52–1:54 | ~285 MB |

Synthetic paths cost ≈ 20 s each through the real engine (single core); 1,000 paths ≈ 5.6 h
on one core, ≈ 1.4 h on 4 workers — a 10-point κ×σ grid at 200 paths is a ~3 h job on 4 cores.
Bootstrap is cheap: 10⁵ paths in ~5 min. Memory is not a constraint. Seeds derive from
`SeedSequence(seed).spawn(n_paths)`, so results are identical for `--workers 1` vs `--workers 3`
(tested), enabling chunked or distributed runs.

## 4. Acceptance criteria for proceeding with large simulations

Proceed only if all of: (a) matched-timestamp institutional data is available (§4.9 of
`docs/research/next_steps_validation_protocol.md`), because the synthetic generator's carry and
basis timing are calibrated to Yahoo closes; (b) the null FPR stays ≤ 5% at ≥ 200 paths; (c) a
pre-registered power target (e.g. ≥ 80% at κ = ln 2 / 5 sessions, realistic costs) is defined
*before* the run; (d) results are reported only as harness/power/stress figures.

## 5. Tests added (`tests/test_montecarlo.py`, 7 tests)

Worker-count independence of bootstrap output; contiguous-block preservation; demeaned null
centred at zero; single-trial DSR == PSR; hand-computed daily-unit PSR (SR_ann 0.5, T 2438 →
0.940); synthetic path prefix causality under extension; tiny synthetic run writes artifacts.

## References

Bailey & López de Prado (2014) *The Deflated Sharpe Ratio*; Bailey & López de Prado (2012)
*The Sharpe Ratio Efficient Frontier* (PSR, minTRL); Künsch (1989) and Politis & Romano (1992)
circular/stationary block bootstrap; Lo (2002) *The Statistics of Sharpe Ratios*.
