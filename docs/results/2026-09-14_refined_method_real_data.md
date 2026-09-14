# Methodological refinement on REAL Yahoo data — baseline vs refined (2026-09-14)

All runs below use **real Yahoo Finance data** (`ES=F`, `NQ=F`, continuous front-month
proxies, unadjusted, no API key), fetched via `ifsa simulate-yahoo` (`yfinance` 1.7.0,
`auto_adjust=False`), served from the local Parquet cache under `data/yahoo/`. No synthetic
fixtures are used in this document. Base capital is **$1,000,000** for every run.

Data:

| interval | requested range | rows ES | rows NQ | notes |
|---|---|---:|---:|---|
| `1d` | 2015-01-01 → 2026-09-14 | 2941 | 2941 | daily bars labelled 21:00 UTC |
| `5m` | 2026-07-20 → 2026-09-14 | 10643 | 10644 | Yahoo intraday retention is ~60 days for 5m; 1m is ~30 days, so 1m was **unavailable** for a meaningful test |

## 1. Diagnosis: why the baseline lost money

Diagnostics on the daily data (log prices, `statsmodels` ADF / Engle–Granger):

- Full-sample Engle–Granger of log(ES) on log(NQ): ADF p = **0.21** → no evidence of
  cointegration over 2015–2026.
- 120-session rolling OLS windows: hedge beta ranges **0.42 – 1.12**, ADF p > 0.10 in
  **21 of 24** windows; estimated AR(1) half-life ranges from 1.7 to 169 sessions.
- log(ES) − log(NQ) drifts **−5.3 %/yr** (NQ outperformance), σ ≈ 8.5 %/yr.
- Baseline yearly PnL (legacy run) was negative in 9 of 12 calendar years, so the loss is
  structural, not a single event.

Root causes:

1. **Non-stationary spread.** A static z-score of the OLS residual assumes a stationary,
   mean-reverting spread. ES/NQ has a persistent drift (sector composition, different
   multipliers $50 vs $20 handled only in sizing) and a time-varying beta, so "reversion"
   trades systematically fade a trend.
2. **Engine defect — residual mixing.** The engine refit (alpha, beta) once per session but,
   in daily mode (`z_reset_each_session=false`), kept the previous sessions' residuals —
   computed under *different* fits — in the z-window. The z-score therefore compared the
   current residual against a window of incomparable numbers. This alone accounts for most of
   the baseline loss: re-evaluating the window under the current fit changes the same rule
   from **−$73.7k to −$1.5k** (rows "Daily legacy" vs "Daily baseline" below).
3. **Hedge unit rounding.** The dollar-neutral sizer rounded the NQ leg to an integer
   *before* volatility scaling, so a 0.88 hedge became 1:1 and then scaled 26:26. Fixed by
   scaling the fractional hedged unit and rounding once.

## 2. Refined method (implemented in `execution/hedge.py`, `execution/engine.py`)

| component | implementation | literature |
|---|---|---|
| dynamic hedge | Kalman filter with random-walk state (β, α) on log prices; spread = innovation under the *prior* state (`predict` before `update`, no lookahead); `delta=1e-5`, `obs_var=1e-4` | Elliott, van der Hoek & Malcolm (2005); Chan (2013) ch. 3; Harlacher (2016) |
| cointegration gate | rolling Engle–Granger (OLS + ADF on residual) on the prior hedge window, or ADF on the current-state residual window for Kalman; entries blocked when p ≥ 0.10 | Engle & Granger (1987); Vidyamurthy (2004) |
| OU thresholds | AR(1)/OU fit on the trailing 60-bar residual window (excluding the current bar): μ, stationary σ, half-life; z = (spread − μ)/σ; entries blocked unless half-life ∈ [1, 30] bars | Avellaneda & Lee (2010); Bertram (2010); Leung & Li (2015) |
| risk controls | divergence stop at \|z\| > 4 (existing), **time stop** at 2 × half-life bars from entry, post-stop cooldown until \|z\| < 0.5 | Leung & Li (2015); Krauss (2017) |
| sizing | vol-target ($5k/day = 0.5 % of capital, 20-session realised vol of the hedged unit) over a fractional dollar-neutral unit (1 ES : β·P_ES·50/(P_NQ·20) NQ), cap 50/leg | standard risk-parity sizing |

Timing, costs and multipliers are unchanged: signal at close → fill at next bar open,
1 tick slippage + 0.5 tick half-spread, $1.25 + $1.38 per contract per side, ES $50/pt,
NQ $20/pt.

## 3. Executed results (real Yahoo data)

Commands (each a cache hit; run directories under `data/results/`):

```bash
ifsa simulate-yahoo --config configs/sim_yahoo_daily_legacy.toml     --out data/results  # 20260914T231807147379Z-c170a78d
ifsa simulate-yahoo --config configs/sim_yahoo_daily.toml            --out data/results  # 20260914T231828243102Z-4b17d0e6
ifsa simulate-yahoo --config configs/sim_yahoo_daily_refined.toml    --out data/results  # 20260914T231855571302Z-d5dd3efd
ifsa simulate-yahoo --config configs/sim_yahoo_daily_rolling_eg.toml --out data/results  # 20260914T231858135227Z-5aeaa83c
ifsa simulate-yahoo --config configs/sim_yahoo_5m.toml               --out data/results  # 20260914T231849318581Z-27833fe6
ifsa simulate-yahoo --config configs/sim_yahoo_5m_refined.toml       --out data/results  # 20260914T231852225666Z-1925a79b
ifsa compare --runs <the six dirs> --out data/results/compare_yahoo.md
```

| run | hedge | thresholds | net PnL | return | ann. vol | Sharpe | max DD | round trips | win rate | avg hold (bars) | fees | slippage | entries gated |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Daily legacy (pre-fix engine, `recompute_z_window_on_refit=false`) | OLS 120s | fixed z | **−$73,718** | −7.37 % | 1.59 % | −0.40 | 8.16 % | 53 | 47.2 % | 21.1 | $878 | $6,520 | 2.0 % |
| Daily baseline (residual-mixing fix only) | OLS 120s | fixed z | −$1,490 | −0.15 % | 1.56 % | −0.01 | 4.47 % | 61 | 63.9 % | 21.6 | $1,015 | $7,520 | 2.0 % |
| Daily rolling EG + gate | rolling EG, p<0.10 | fixed z | +$40,225 | 4.02 % | 1.13 % | 0.30 | 3.42 % | 15 | 73.3 % | 19.9 | $158 | $1,050 | 88.1 % |
| **Daily refined** | Kalman | OU adaptive + half-life gate + time stop, vol-target | **+$115,777** | 11.58 % | 1.73 % | 0.57 | 2.78 % | 15 | 73.3 % | 4.1 | $1,120 | $8,160 | 88.2 % |
| 5m baseline | OLS 10s | fixed z | −$10,224 | −1.02 % | 2.33 % | −2.84 | 1.14 % | 24 | 54.2 % | 16.8 | $379 | $2,880 | 12.8 % |
| 5m refined | Kalman | OU + gates | $0 | 0 % | — | — | — | 0 | — | — | $0 | $0 | **100 %** (no entries passed the cointegration/half-life gates) |

Refined-daily details (`20260914T231855571302Z-d5dd3efd`): profit factor 4.85, Sortino in
`report.json`, max gross 45 contracts, finite half-life median 8.0 sessions
(IQR 4.9–16.9); gate reasons over 2941 bars: cointegration 2173, cointegration+half-life 358,
none 348, hedge warm-up 60, half-life only 2. Trade exits: 20 legs time-stop, 8 exit, 2
divergence stop. Round-trip PnL: 11 winners / 4 losers, largest +$32.7k (2020-06),
largest −$20.6k (2021-02); positive in 2015, 2016, 2019, 2020, 2024, 2025, flat in years
with no gate pass, negative in 2022.

## 4. Interpretation — improvement is evidenced, edge is not

- The **loss cause is established**: the legacy rule mixed residuals from different hedge
  fits and traded a non-cointegrated spread as if it were stationary. Fixing the window alone
  removes ~98 % of the loss; the remaining baseline is ≈ zero net of costs.
- The **refined method mostly refuses to trade**: 88 % of sessions are gated (the pair fails
  the cointegration test most of the time), producing only **15 round trips in 11.7 years**.
  The positive PnL rests on that small sample: mean round trip ≈ +$7.7k with s.e. ≈ $3.8k
  (t ≈ 2.0); annualised Sharpe 0.57 over 11.7 years has t ≈ 1.95. This is *borderline*,
  not a demonstrated edge, and parameters (gate 0.10, half-life ≤ 30, 2 × half-life stop,
  delta 1e-5) were fixed a priori from the literature, not optimised — but only one
  configuration was run, so no out-of-sample claim is made.
- On 5-minute data the refined gates never open within 60 days of history: the honest result
  is **no trades**, not a profit.

## 5. Limitations

- Yahoo `ES=F`/`NQ=F` are unadjusted continuous front-month proxies: roll jumps enter the
  spread; per-contract volume/OI are unavailable, so the Databento roll calendar is not used.
- 1-minute Yahoo history (~30 days) is too short for the hedge warm-up; 5m (~60 days) barely
  covers it, hence the 100 % gating.
- Daily fills are modelled at the next day's open of a 21:00 UTC-labelled bar; intraday
  execution risk is not modelled.
- Slippage is a fixed 1 tick + 0.5 tick half-spread; large vol-targeted sizes (up to 45
  contracts) would incur market impact not modelled here.
- Johansen (multivariate) estimation was not implemented; with two series Engle–Granger is
  the standard single-equation approach.

## 6. Verification

```text
pytest -q                      116 passed, 1 deselected (network) 
pytest -q -m network tests/test_yahoo.py   1 passed (live Yahoo)
ruff check src tests           All checks passed!
ruff format --check src tests  44 files already formatted
mypy src                       Success: no issues found in 24 source files
```

## References

- Avellaneda, M., Lee, J.-H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance* 10(7).
- Bertram, W. K. (2010). Analytic solutions for optimal statistical arbitrage trading. *Physica A* 389(11).
- Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*, ch. 3 (Kalman hedge ratio). Wiley.
- Elliott, R. J., van der Hoek, J., Malcolm, W. P. (2005). Pairs trading. *Quantitative Finance* 5(3).
- Engle, R. F., Granger, C. W. J. (1987). Co-integration and error correction. *Econometrica* 55(2).
- Gatev, E., Goetzmann, W. N., Rouwenhorst, K. G. (2006). Pairs trading: performance of a relative-value arbitrage rule. *Review of Financial Studies* 19(3).
- Harlacher, M. (2016). Cointegration based algorithmic pairs trading. PhD thesis, Univ. St. Gallen.
- Krauss, C. (2017). Statistical arbitrage pairs trading strategies: review and outlook. *Journal of Economic Surveys* 31(2).
- Leung, T., Li, X. (2015). Optimal mean reversion trading with transaction costs and stop-loss exit. *IJTAF* 18(3); arXiv:1411.5062.
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
