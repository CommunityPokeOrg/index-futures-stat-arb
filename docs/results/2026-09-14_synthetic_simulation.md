# Validation & simulation record — 2026-09-14

All numbers below come from the **synthetic offline fixture** (`SyntheticBarClient`,
seed 42). No Databento credentials were available in the environment, so the live
`DatabentoClient` path was exercised only through its interface (fake/flaky clients
in tests). **Synthetic results are a harness check, not evidence of a tradeable edge**:
the fixture's ES/NQ spread is a constructed Ornstein–Uhlenbeck process, so a z-score
mean-reversion strategy is expected to be profitable on it by design.

## Commands run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q                      # 91 passed, 36 warnings in 17.62s
ruff check src tests           # All checks passed!
ruff format --check src tests  # 40 files already formatted
mypy src                       # Success: no issues found in 22 source files

ifsa ingest   --config configs/ingest_example.toml --offline --no-resume
#   data/manifests/synthetic-GLBX.MDP3-ohlcv-1m-ESH6,ESM6,NQH6,NQM6-2026-01-05-2026-03-20.json
#   row_count=298080   wall 11.2 s  (resume re-run: ~1.6 s, all chunks skipped)
ifsa rolls    --config configs/ingest_example.toml --rule volume --out data/reference/roll_calendar
ifsa simulate --config configs/sim_synthetic.toml --offline --out data/results
```

Warnings: NumPy "Degrees of freedom <= 0" from the first bars of each z-score window
(expected, z is NaN there) and the pre-existing statsmodels `adfuller` FutureWarning.

## Roll calendar (volume rule, prior-session volume, CME-Monday guard)

| product | from | to | roll session | decision basis | panama offset | ratio | joint roll |
|---|---|---|---|---|---|---|---|
| ES | ESH6 | ESM6 | 2026-03-10 | volume_prev_day from=164668 to=165755 | 18.208 | 1.004 | 2026-03-12 |
| NQ | NQH6 | NQM6 | 2026-03-12 | volume_prev_day from=164904 to=165949 | 63.758 | 1.004 | 2026-03-12 |

ESH6 expiry 2026-03-20 (third Friday); CME roll date 2026-03-16; both rolls fall inside
the `[expiry − 10 bd, CME Monday]` window. The joint roll is the later of the two legs.

## Simulation parameters (`configs/sim_synthetic.toml`)

- Window 2026-01-05 → 2026-03-20 (54 RTH sessions, includes the March roll), 5-minute bars, RTH only.
- Signal: log-spread z-score, window 78 bars, entry 2.0 / exit 0.5 / stop 4.0; hedge ratio refit
  each session by OLS on the prior 10 sessions (min 5); no entries on roll sessions.
- Timing: decide at bar close → fill at next bar open; 1 tick slippage + 0.5 tick half-spread,
  $1.25 commission + $1.38 exchange fees per contract per side.
- Sizing: dollar-neutral, 2 ES contracts, NQ contracts from the hedge ratio (cap 20).
- Prices forward-Panama-adjusted; roll trades booked at the open of the first roll-session bar.
- The recorded run used initial capital $250,000; the current default is $1,000,000; seed 0.

## Results

| metric | base (slip 1 tick) | slip 3 ticks | entry 2.5 |
|---|---|---|---|
| report | `20260914T204246Z-0a0635fd` | `20260914T204307Z-c2ae8a4d` | `20260914T204311Z-3f9b56a6` |
| total PnL (USD) | 12,345.26 | 2,445.26 | 9,966.72 |
| total return | 4.94 % | — | — |
| ann. return / vol | 25.2 % / 6.57 % | — | — |
| Sharpe (daily) | 3.51 | 0.70 | 3.29 |
| Sortino | 6.43 | — | — |
| max drawdown | $3,124.63 (1.25 %) | $4,065.97 | $1,699.54 |
| round trips / leg fills | 82 / 330 | 82 | 42 |
| win rate | 70.7 % | — | — |
| profit factor | 2.97 | — | — |
| fees / slippage (USD) | 1,309.74 / 9,809.92 | — / 19,709.92 | — / 5,106.95 |
| turnover (contracts) | 498 | — | — |
| exposure / avg hold | 30.8 % / 15.8 bars | — | — |

Slippage dominates costs: tripling assumed slippage removes ~80 % of the synthetic PnL, which
is the main sensitivity to calibrate against real TBBO data before drawing any conclusion.

Reports are reproducible: two runs with the same config produce identical metrics
(`tests/test_cli.py`). Output folders (`report.json`, `trades.csv`, `equity.csv`, `daily.csv`)
live under `data/results/<run_id>/` and are gitignored.

## Limitations

- No live Databento ingestion was run (no `DATABENTO_API_KEY`); adapter is thin and untested against the real API.
- CME holiday calendar is an approximation (full closures only; no early closes).
- Fills are at next-bar open ± fixed ticks; no intrabar/limit-order or queue modelling.
- Hedge ratio and z-score use forward-adjusted closes; daily settlement vs. close is not modelled.
- Synthetic fixture volumes/prices are stylised; the volume roll date is meaningful only as a rule check.
