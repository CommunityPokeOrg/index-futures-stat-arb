# Real-data validation (Yahoo Finance) — 2026-09-14

**Data source: REAL market data from Yahoo Finance** via the keyless `yfinance` adapter
(`src/index_futures_stat_arb/ingest/yahoo.py`). This is *not* the synthetic fixture used in
`2026-09-14_synthetic_simulation.md`; every report produced here carries
`"data_source": "yahoo"` plus per-symbol fetch metadata (`data_meta`) in `report.json`.

## Query parameters

| | daily run | 5-minute run |
|---|---|---|
| symbols | `ES=F`, `NQ=F` (Yahoo continuous front-month) | same |
| `yf.download` args | `interval="1d", auto_adjust=False, threads=False` | `interval="5m"`, same |
| requested range | 2015-01-01 → 2026-09-14 | 2026-07-20 → 2026-09-14 |
| effective range (after retention clip) | unchanged (`clipped=false`) | unchanged (`clipped=false`) |
| rows fetched | ES 2941, NQ 2941 | ES 10643, NQ 10644 (35 non-fatal gaps flagged per leg) |
| cache | `data/yahoo/interval=1d/product=ES/2015-01-01_2026-09-14.parquet` + `.json` sidecar (sha256) | `data/yahoo/interval=5m/...` |
| validation | `validate_bars` OHLC/dup/order checks passed | passed |

Yahoo retention limits encoded in `YAHOO_INTERVAL_LIMITS`: 1m ≤ 30 days history / 7 days per
request; 2m–30m ≤ 60 days; 60m ≤ 730 days; 1d unlimited. Requests are clipped and chunked
accordingly and the `clipped` flag is recorded.

## Commands

```bash
. .venv/bin/activate
pytest -q                                  # 101 passed, 1 deselected (network)
pytest -q -m network tests/test_yahoo.py   # 1 passed (live Yahoo fetch)
ruff check src tests && ruff format --check src tests && mypy src   # all clean
ifsa simulate-yahoo --config configs/sim_yahoo_daily.toml --out data/results
ifsa simulate-yahoo --config configs/sim_yahoo_5m.toml    --out data/results
```

## Strategy parameters

Daily (`configs/sim_yahoo_daily.toml`): log-spread z-score over 60 sessions (window not reset
per session), entry 2.0 / exit 0.5 / stop 4.0, hedge ratio OLS on prior 120 sessions (min 60),
signal at close → fill at next session's open, 1 tick slippage + 0.5 tick half-spread,
$1.25 + $1.38 per contract per side, dollar-neutral 2 ES vs hedge-ratio NQ, $250k capital.

5-minute (`configs/sim_yahoo_5m.toml`): RTH only, 78-bar z window reset each session,
hedge on prior 10 sessions (min 5), otherwise identical.

## Results (real data)

| metric | daily 2015-01-01 → 2026-09-14 | 5m 2026-07-20 → 2026-09-14 |
|---|---|---|
| report | `data/results/20260914T221320Z-17fbfa0f/` | `data/results/20260914T221323Z-29a6d9d3/` |
| sessions | 2941 | 39 |
| total PnL (USD) | −73,718.42 | −10,223.72 |
| total return | −29.49 % | −4.09 % |
| annualised return / vol | −2.95 % / 6.36 % | −23.65 % / 9.31 % |
| Sharpe / Sortino | −0.40 / −0.35 | −2.84 / −2.09 |
| max drawdown | $81,581.86 (32.6 %) | $11,419.82 (4.6 %) |
| Calmar | −0.09 | −5.18 |
| leg fills / round trips | 212 / 53 | 96 / 24 |
| win rate | 47.2 % | 54.2 % |
| avg round-trip PnL | −$1,333.57 | −$352.68 |
| profit factor | 0.54 | 0.49 |
| fees / slippage (USD) | 878.42 / 6,520.00 | 378.72 / 2,880.00 |
| turnover (contracts) | 334 | 144 |
| exposure / avg hold (bars) | 38.0 % / 21.1 | 13.2 % / 16.8 |
| max gross contracts | 5 | 3 |

**Interpretation.** On real ES/NQ data the naive z-score mean-reversion rule loses money net of
costs in both horizons, in contrast to the synthetic fixture (whose spread is OU by
construction). The drawdown in the daily run is dominated by extended ES/NQ divergence
(2020–2021 growth/value regime) where a fixed-window OLS hedge lags.

## Limitations of the Yahoo path

- `ES=F`/`NQ=F` are Yahoo's continuous front-month series: **unadjusted**, so price jumps at
  Yahoo's (undocumented) roll dates leak into the spread; no per-contract volume/OI, so the
  repo's roll calendar and Panama/ratio adjustment cannot be applied to this source.
- Daily bars carry no intraday timestamp; `ts_event` is set to 21:00 UTC (≈ 16:00 ET) as a label.
- Intraday history is short (1m ≈ 30 days, 5m ≈ 60 days); 5m bars showed 35 timestamp gaps per
  leg (reported, non-fatal).
- Yahoo data is unofficial and may be revised or rate-limited; the parquet cache + sha256
  sidecar make a given run reproducible but not the upstream source.
