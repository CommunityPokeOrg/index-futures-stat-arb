# RFC 01 — Data Ingestion & Continuous-Contract Representation for ES/NQ Stat-Arb

| | |
|---|---|
| Status | Draft for discussion (research only — no ingestion code) |
| As-of date | 2026-09-14 |
| Tracks | Issue #1, section "1. Data ingestion" |
| Scope | Data sources, continuous-contract construction, bar granularity, on-disk storage, schema/metadata design, follow-up plan |
| Out of scope | Live trading, order routing, production ingestion implementation, modeling/backtest harness (Tasks 2–3) |

Evidence convention used throughout: **[V]** = read directly on the cited primary page on the as-of date; **[A]** = assumption / common practice / secondary source, *not* verified on a primary page in this pass. Anything marked [A] is an open verification item, not a fact this RFC relies on.

---

## 0. Summary of recommendations

1. **Sources.** Use **Databento GLBX.MDP3** as the reference source for ES/NQ (continuous symbology `ES.v.0` / `NQ.v.0`, `ohlcv-1m`/`ohlcv-1s`, trades and MBP-1 available). Keep **yfinance `ES=F`/`NQ=F`** strictly as a zero-cost smoke-test path for daily bars and short intraday windows; treat its contract-selection semantics as undocumented. Alpha Vantage is *not* a viable futures source; Polygon/Massive and IBKR are viable but each carries constraints (see §1).
2. **Continuous contracts.** Store **per-contract, unadjusted** bars plus an explicit **roll calendar** as the canonical representation. Derive adjusted series *on read*. Research default: **volume-based roll (prior-day volume, Databento `v` rule), back-adjusted additively (Panama) for spread/PnL analysis, while cointegration/hedge-ratio fitting uses per-contract log prices within a roll window or a ratio-adjusted series** — see §2 for why the two questions want different adjustments.
3. **Granularity.** Ingest **1-minute OHLCV** as the primary research grain (aggregate to 5-min on read); keep a small **trades/TBBO** sample for cost/slippage calibration. Do not build the research pipeline on tick data.
4. **Storage.** **Zstd-compressed Parquet**, Hive-partitioned by `source/product/contract/date`, queried through **DuckDB** (and Polars/pandas). No Zarr/HDF5.

---

## 1. Data sources and access

### 1.1 Comparison matrix

| Criterion | yfinance (Yahoo) | Alpha Vantage | Databento (GLBX.MDP3) | Polygon.io → "Massive" | Interactive Brokers TWS API |
|---|---|---|---|---|---|
| ES/NQ futures coverage | `ES=F`, `NQ=F` tickers exist; contract semantics undocumented [A] | **No futures endpoint found** in documentation index [V] | Full CME futures & options from June 2010 [V] | Futures product exists (docs section present) but pricing/coverage page did not render — **unverified** [A] | Yes, incl. expired contracts ≤ 2 years past expiry [V] |
| Cost | Free, unofficial | Free tier; intraday/historical/"full" and index endpoints are premium [V] | Pay-as-you-go with $125 sign-up credit; Standard $199/mo, Plus $1,750/mo, Unlimited $4,500/mo (annual) [V] | Unverified for futures [A] | Requires brokerage account; market-data subscription requirement [A] |
| Rate limits | Undocumented; scraping-based [A] | Premium tiers advertised at 150/300/600/1200 req/min [V] | "Unlimited downloads/streaming/API calls" on subscription plans [V] | Unverified [A] | 50 simultaneous open requests; ≤ 60 requests per 10 min for bars ≤ 30 s; BID_ASK counts double [V] |
| Historical depth | Intraday cannot extend past last 60 days [V]; daily many years [A] | Stocks: `full` = 25+ years (premium) [V]; futures N/A | June 2010 → present for CME; full MBO granularity from May 2017 [V] | Unverified [A] | Bars ≤ 30 s: 6 months; expired futures: 2 years after expiry [V] |
| Granularities | 1m…1d,5d,1wk,1mo,3mo [V] | 1m…60m (premium for history) [V] | OHLCV 1s/1m/1h/1d, trades, TBBO, MBP-1, MBO [V] | Unverified [A] | 1 s … 1 month bar sizes [V] |
| Bid/ask | No [A] | No [A] | Yes — TBBO, MBP-1, MBO [V] | Unverified [A] | `whatToShow=BID/ASK/BID_ASK/MIDPOINT` bars (time-averaged) [V] |
| Timestamps | Intraday index converted to exchange tz (`ignore_tz=False` default); daily tz-naive [V] | Docs give US/Eastern by default [A] | `ts_recv`, `ts_event`, `ts_in_delta`, nanosecond precision [V] | Unverified [A] | Bars in the TWS login timezone; futures daily close may be exchange settlement arriving hours later; session dated by closing day [V] |
| Continuous contract | Implicit, opaque [A] | N/A | `ES.c.0` / `ES.v.0` / `ES.n.0` — unadjusted stitching, no back-adjustment [V] | Unverified [A] | `CONTFUT` secType exists [A]; behaviour unverified |
| Licensing / redistribution | Yahoo ToS; not licensed for redistribution [A] | Commercial use "contact sales" [V] | Self-service licensing; Plus plan includes external distribution [V] | Unverified [A] | Personal use under account agreement [A] |
| Survivorship / delisting | Not applicable to ES/NQ (contracts expire, they are not delisted); Yahoo silently rolls [A] | N/A | Every expired contract retained as `raw_symbol` e.g. `ESZ3` [V] | Unverified [A] | Expired contracts dropped after 2 years [V] |

### 1.2 Source notes

**yfinance / Yahoo Finance** — Documented in the yfinance API reference: valid intervals `1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo`; "Intraday data cannot extend last 60 days"; `auto_adjust=True` by default; `start` inclusive / `end` exclusive; `prepost=False` default; intraday index tz-converted unless `ignore_tz` [V] (<https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html>). Not verified: the frequently repeated "1m only for the last 7–8 days" and "1h up to 730 days" limits [A]; what contract `ES=F` maps to on a given day and how/when Yahoo rolls it [A]; whether `Adj Close` differs from `Close` for futures [A]. **Applicability:** fine for the daily-bar smoke path the scaffold already uses; unsuitable as a reference source for anything involving roll timing or intraday history > 60 days. Issue #1's "1-minute and 5-minute bars over 30–90 days" therefore **cannot** be fully satisfied from yfinance at 1m (60-day ceiling, likely less) — the 90-day end of the window requires Databento or IBKR.

**Alpha Vantage** — The documentation index lists stock, index, options, FX, crypto, commodities, economic indicators and technical indicators; no futures contract endpoint was found [V]. Intraday and full-history endpoints and the S&P 500 / Nasdaq-100 *index* endpoints are premium [V] (<https://www.alphavantage.co/documentation/>). **Applicability:** ES/NQ not covered → excluded. It could only serve SPX/NDX index levels as a cash proxy, which is a different research question (basis, dividends, financing).

**Databento GLBX.MDP3** — CME Group full depth-of-book dataset (CME, CBOT, NYMEX, COMEX), futures and options on futures from June 2010; MDP 3.0 full order-event granularity from May 2017, legacy FIX/FAST before that [V] (<https://databento.com/docs/venues-and-datasets/glbx-mdp3>). Symbology: `raw_symbol` (e.g. `ESZ3`), `instrument_id`, `parent` (`ES.FUT`), and `continuous` with notation `[ROOT].[ROLL_RULE].[RANK]`, zero-indexed rank, roll rules `c` (calendar/expiry), `n` (prior-day open interest), `v` (prior-day volume). Continuous series are **original, unadjusted prices** — Databento explicitly does not back-adjust rollover jumps [V] (<https://databento.com/docs/standards-and-conventions/symbology>). Schemas: `ohlcv-1s/1m/1h/1d`, `trades`, `tbbo`, `mbp-1`, `mbo` [V] (<https://databento.com/docs/schemas-and-data-formats>). Pricing: usage-based or subscription, $125 free credit at sign-up, plan history examples for one ES product: MBO 14 mo, MBP-1 12 mo, TBBO 15 mo, Trades 16 mo [V] (<https://databento.com/pricing>). Not verified: exact per-GB rates and ES/NQ byte-size estimates [A]. **Applicability:** best fit — the only evaluated source that gives per-contract history, a documented roll rule vocabulary, nanosecond timestamps and bid/ask at research-friendly cost. Note that Databento's continuous symbols are a *selection* rule, not an adjustment method; adjustment remains our job (§2).

**Polygon.io (now branded "Massive")** — The pricing page has a "Futures" product tab and the docs have a REST → Futures section, but neither the futures pricing nor the futures overview page rendered content in this pass; the old `polygon.io/docs/futures/...` URLs return 404 [V that the pages exist / 404; A for everything else] (<https://massive.com/pricing?product=futures>, <https://massive.com/docs/rest/futures/overview>). **Applicability:** potentially viable; requires a browser-rendered read of the futures plan (tiers, history, whether CME is included) before it can be scored. Open question OQ-1.

**Interactive Brokers TWS API** — From the (deprecated but still published) TWS API docs: `reqHistoricalData` bar sizes 1 s … 1 month; `whatToShow` TRADES/MIDPOINT/BID/ASK/BID_ASK/SCHEDULE; BID_ASK bars are time-averaged bid/ask with max ask/min bid; bar timestamps in the TWS login timezone; futures daily close may be the exchange settlement, which can arrive hours later (Friday settlement possibly Saturday); historical feed filters away-from-NBBO/block/combo trades so volume is lower than exchange volume [V] (<https://interactivebrokers.github.io/tws-api/historical_bars.html>). Limits: 50 simultaneous requests; for bars ≤ 30 s no identical request within 15 s, < 6 same-contract requests per 2 s, ≤ 60 requests per 10 min; bars ≤ 30 s only for 6 months; **expired futures unavailable > 2 years after expiry**; duration/bar-size step table [V] (<https://interactivebrokers.github.io/tws-api/historical_limitations.html>). Not verified: current market-data subscription requirements, `CONTFUT` behaviour [A]. **Applicability:** good for recent 1-min history and bid/ask bars if a funded account exists; the 2-year expired-contract cutoff makes it unsuitable for building a long per-contract archive. Session/timezone handling needs care (login-tz bars, settlement-vs-close).

### 1.3 ES/NQ contract facts used below

| Fact | Status | Source |
|---|---|---|
| NQ = $20 × Nasdaq-100 index, minimum tick 0.25 index points | [V] | <https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.contractSpecs.html> ("About" section) |
| ES = $50 × S&P 500, tick 0.25 = $12.50 | [A] — CME ES spec page renders client-side; numeric spec table not captured | <https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.contractSpecs.html> |
| Quarterly cycle Mar/Jun/Sep/Dec; expiration is the third Friday; **CME's customary equity-index roll date is the Monday prior to the third Friday** of the expiration month (e.g. Sep-2026: expiry 9/18, roll 9/14); after the roll date the second-nearest month is the "lead month" | [V] | <https://www.cmegroup.com/trading/equity-index/rolldates.html> |
| Trading hours Sun–Fri 6:00 pm – 5:00 pm ET with 4:15–4:30 pm maintenance break; daily settlement 3:00 pm CT | [A] — not captured from CME page | — |

The often-quoted "roll on the Thursday eight days before expiry" convention is **not** what CME publishes; CME publishes the Monday of expiry week. Both are candidate calendar rules; the RFC treats the CME-published date as the calendar default.

---

## 2. Continuous-contract construction

### 2.1 Why this matters for a pairs strategy

A spread \(s_t = \log ES_t - \beta \log NQ_t\) is only meaningful if both legs refer to instruments a trader could actually hold simultaneously. Every roll introduces a discontinuity equal to the calendar spread (front vs next), which for equity index futures is dominated by financing minus expected dividends and is typically tens of index points on ES. If ES and NQ roll on different days, or if their calendar spreads differ in sign/magnitude, the *pair* spread gets an artificial step that:

* biases Engle–Granger / ADF toward rejecting (or, worse, spuriously accepting) cointegration,
* distorts the hedge-ratio regression through a handful of high-leverage points,
* produces phantom z-score excursions exactly at roll dates (false entries/exits), and
* creates phantom PnL if the backtest marks a position across the roll without modelling the roll trade.

### 2.2 Methods compared

| Method | Construction | Preserves | Breaks | Effect on spread / z-score | Phantom PnL risk | Lookahead risk |
|---|---|---|---|---|---|---|
| **Unadjusted stitching** (Databento `c/v/n` series as delivered) | Concatenate front-contract prices; step at each roll | Actual tradable price levels; tick grid | Price continuity; returns across roll are wrong | Roll step enters spread directly unless both legs roll same day *and* their calendar spreads cancel (they don't) | High if PnL = Δprice across roll | None if roll rule uses prior-day data |
| **Panama / additive back-adjustment** | At each roll subtract (new − old) offset from all *earlier* prices (or add to later, forward-adjust) | Point differences → PnL in ticks × multiplier is exact | Absolute levels (can go negative far back); ratios/percent returns distorted; tick grid preserved only if offset is a tick multiple | Log-spread on adjusted levels is *not* what you traded; differences of the spread are correct | None for difference-based PnL | Offsets depend on future rolls → adjusted history changes each roll; must snapshot per as-of date |
| **Ratio / multiplicative adjustment** | Multiply earlier prices by (new/old) at each roll | Percent returns and log-price differences | Dollar PnL per contract, tick grid, absolute levels | Log-spread is continuous and returns-consistent → **best for cointegration/OLS on log prices** | Small if PnL computed in log-return space then scaled; wrong if computed as adjusted-price differences × multiplier | Same as Panama |
| **Calendar-month stitching** (fixed roll offset, e.g. CME Monday-of-expiry-week or N days before expiry) | Roll rule only; combine with any adjustment | Deterministic, reproducible, knowable in advance | May roll before/after liquidity actually migrates | Roll gap size depends on rule, not on liquidity | As per adjustment | **None** — rule is fully known ex-ante |
| **Volume / open-interest roll** (Databento `v` / `n`) | Switch when next contract's *prior-day* volume/OI exceeds front | Tracks liquidity → realistic fills | Roll day can differ between ES and NQ; occasionally flips back ("whipsaw") | Minimises stale-contract noise; needs hysteresis | As per adjustment | None if strictly prior-day (Databento's definition) [V]; **lookahead if same-day volume is used** |

### 2.3 Recommended approach

Canonical storage is **per-contract, unadjusted** (§5). Continuous series are *views* computed from (a) a roll calendar table and (b) an adjustment method, both recorded as metadata on every derived dataset.

* **Roll rule default:** prior-day volume (`v`), with a **one-way constraint** (never roll back) and a **calendar guard** (must roll no later than CME's published Monday-of-expiry-week; must not roll earlier than 10 business days before expiry). Alternatives kept as first-class: `c` (CME Monday), `n` (open interest), fixed `expiry − k` days.
* **Adjustment default depends on the consumer:**
  * *Cointegration test, hedge-ratio fit, OU fit on log spread* → **ratio-adjusted log prices** (or, cleaner, fit within a single-contract window and re-fit each roll; this is the leakage-free baseline recommended for Task 2).
  * *Backtest PnL, z-score thresholds in index points* → **Panama back-adjusted**, with PnL always computed as \(\text{position} \times \Delta\text{price} \times \text{multiplier}\) and an explicit **roll trade** (close old, open new) with its own cost on the roll date.
  * *Anything reported to a user as a price level* → unadjusted, with contract id shown.
* **Roll-day handling in signals:** mask signal generation on roll days (or require both legs to be on their new contracts) so the z-score is not evaluated on a mixed-contract spread.
* **Same-day roll for both legs:** ES and NQ share the quarterly cycle, so force a *joint* roll date (default: the later of the two legs' `v` dates, clipped by the calendar guard). Record both individual and joint dates.

### 2.4 Lookahead controls

1. Roll decisions use strictly *prior*-session volume/OI (matches Databento's `v`/`n` definition [V]).
2. Back-adjusted series are **as-of dated**: `adjusted_asof=YYYY-MM-DD` in metadata; a backtest run on date *D* may only use offsets from rolls ≤ *D*. Forward-adjustment (offsets applied to later data) avoids history rewriting and is the preferred variant for walk-forward tests.
3. Hedge ratio and OU parameters are fit on windows that end before the test window starts (already implemented in `walk_forward_backtest`) *and* never straddle a roll unless the series is ratio-adjusted.

### 2.5 Validation tests for the future implementation

* **Roll-calendar sanity:** every roll date lies within [expiry − 10 bd, CME Monday]; roll dates strictly increasing; exactly one roll per quarter; roll dates for `v` rule reproduce Databento `ES.v.0` instrument switches on a sampled period.
* **Adjustment invariants:** Panama: `diff(adjusted) == diff(unadjusted)` on all non-roll bars, and the roll-bar diff equals the *new* contract's own diff. Ratio: `diff(log(adjusted)) == diff(log(unadjusted))` on non-roll bars.
* **Roll-gap accounting:** sum of Panama offsets equals unadjusted[last] − adjusted[first] within float tolerance.
* **Lookahead:** rebuilding the adjusted series as-of date *D* must be identical for all bars ≤ *D* regardless of data after *D*.
* **Spread-level checks (Task 2):** ADF/EG statistic on the spread should not change materially when roll bars are excluded; a large change indicates roll artifacts driving the result. Z-score excursions concentrated on roll dates → fail.
* **Phantom PnL:** a buy-and-hold on the Panama series with the roll trade modelled must equal the sum of per-contract PnLs.

---

## 3. Sampling granularity

| Grain | Signal-to-noise for a daily/hourly-half-life spread | Microstructure noise | Execution realism | Data volume (order of magnitude, per product-year) | Verdict |
|---|---|---|---|---|---|
| Tick (trades / MBP-1) | Signal drowned in bid-ask bounce; realised-variance estimators biased upward [A — Roll 1984; Aït-Sahalia–Mykland–Zhang 2005, not re-read this pass] | Maximal | Best — enables queue/slippage modelling | Trades: tens of millions of rows; MBP-1: hundreds of millions [A] | Sample only, for cost calibration |
| 1-second OHLCV | Still noisy; many empty/flat bars overnight | High | Good | ~ 23 h × 3600 × 250 ≈ 20 M bars | Not needed for this strategy |
| **1-minute OHLCV** | Adequate for half-lives of hours–days; enables realistic entry timing | Moderate; bounce mostly averaged out in close-to-close but present in OHLC | Good — fill at next bar open, cost from TBBO sample | ≈ 350 k bars per product-year (~ 1,380 bars/session) — trivial | **Primary research grain** |
| 5-minute OHLCV | Cleaner; the classic "optimal sampling ~5 min" result [A — AMZ 2005] | Low | Coarser fill timing (+2.5 min average latency) | ≈ 70 k bars / product-year | Derived from 1-min on read; use for robustness checks |
| Daily | Highest SNR per bar, few observations (≈ 250/yr) | Negligible | Settlement vs close ambiguity (IB note [V]) | Trivial | Existing scaffold path; keep for long-history cointegration tests |

Decision: ingest **1-minute** bars for the Issue #1 30–90-day window; resample to 5-min in the loader (never store both — avoids parity bugs). Keep a **one-week trades + TBBO sample** per product from Databento to estimate half-spread and short-horizon impact for the backtest cost model. Storage/compute at these sizes is not a constraint; the binding constraint is data cost and the yfinance 60-day ceiling.

Volume-related facts to (re)verify before publishing numbers: ES/NQ typical messages/day, MBP-1 vs trades byte sizes (Databento cost estimator) — OQ-3.

---

## 4. Storage / retrieval

| Criterion | Parquet (zstd, PyArrow) | DuckDB native `.duckdb` | Zarr v3 / HDF5 (PyTables) |
|---|---|---|---|
| pandas / Polars slice-and-dice | Native in both; predicate & projection pushdown; row-group statistics [V — PyArrow docs] | Excellent SQL; reads Parquet directly (`read_parquet`, Hive partitioning) [A — DuckDB docs redirected, not read] | pandas `HDFStore` OK; Polars no native HDF5/Zarr [A] |
| Partitioning | Hive-style directories `key=value/`; `_metadata`/`_common_metadata` sidecars [V] | Internal; or just query partitioned Parquet | Chunking, not partitioning; time-series row-append is awkward for Zarr, fine for HDF5 tables [A] |
| Schema evolution | Add nullable columns per file; reader unions schemas; column rename requires rewrite | ALTER TABLE; but file-format version coupled to DuckDB release [A — storage-compat page not read] | HDF5: rigid table schema; Zarr: array-per-variable, metadata JSON — flexible but non-tabular |
| Compression | Snappy default; Brotli/Gzip/ZSTD/LZ4 supported; Snappy faster, Gzip smaller [V] | Internal, tunable | zlib/blosc/zstd via numcodecs / PyTables filters [A] |
| Portability / reproducibility | Best: language-agnostic, stable spec, content-hashable files | Single-file convenience; version-lock risk for archival | HDF5 portable but C-library dependent; Zarr designed for object stores |
| Fit for OHLCV time series | Ideal (tabular, append by partition) | Ideal as *query layer* | Designed for N-D arrays (gridded science data); tabular fit is poor |

Decision: **Parquet + DuckDB** — Parquet is the artifact of record (deterministic paths, SHA-256 in a manifest), DuckDB is the query engine (ad-hoc SQL over the Hive layout, aggregation to 5-min, joins ES↔NQ on `ts_event`). No `.duckdb` files committed or relied on for reproducibility. Zarr/HDF5 rejected for tabular bar data.

Compression: `zstd` (level 3) over Snappy — smaller files matter more than decode speed at our volumes; both verified as supported codecs [V] (<https://arrow.apache.org/docs/python/parquet.html>).

---

## 5. Proposed schema & contract-metadata representation (design only)

### 5.1 Layout

```
data/
  raw/                              # immutable vendor pulls, never rewritten
    source=databento/dataset=GLBX.MDP3/schema=ohlcv-1m/product=ES/contract=ESZ6/date=2026-09-14/part-0.parquet
    source=yfinance/schema=ohlcv-1d/product=ES/contract=UNKNOWN/date=.../part-0.parquet
  reference/
    contracts.parquet               # one row per listed contract (ES, NQ)
    roll_calendar/rule=v/asof=2026-09-14.parquet
  derived/                          # rebuildable; never the source of truth
    continuous/product=ES/rule=v/adjust=panama/asof=2026-09-14/bars-1m.parquet
  manifests/
    <dataset_id>.json               # provenance + checksums
```

### 5.2 Bar table (`ohlcv-*`)

| column | type | notes |
|---|---|---|
| `ts_event` | timestamp[ns, UTC] | bar **open** time; UTC always; source tz recorded in manifest |
| `ts_recv` | timestamp[ns, UTC] nullable | Databento only |
| `source` | dictionary<string> | `databento`, `yfinance`, `ibkr`, … |
| `product` | dictionary<string> | `ES`, `NQ` |
| `contract` | dictionary<string> | vendor `raw_symbol` e.g. `ESZ6`; `UNKNOWN` when the vendor hides it (yfinance) |
| `instrument_id` | int64 nullable | vendor numeric id |
| `open, high, low, close` | float64 | index points, unadjusted |
| `volume` | int64 | contracts |
| `interval` | dictionary<string> | `1m`, `1s`, `1d` |
| `session_date` | date32 | CME trade date (session dated by its *closing* day, 5 pm ET boundary) — derived, but stored to avoid re-deriving tz rules |
| `is_rth` | bool | 9:30–16:00 ET flag, derived |

Sorted by (`product`, `contract`, `ts_event`), unique on the same key. Prices stored as float64 in index points (Databento fixed-point 1e-9 ints converted at ingest; conversion recorded in manifest).

### 5.3 Contract reference (`contracts.parquet`)

| column | type | notes |
|---|---|---|
| `product` | string | `ES` |
| `contract` | string | `ESZ6` |
| `month_code`, `year` | string, int16 | `Z`, 2026 |
| `first_trade_date`, `last_trade_date`, `expiration_ts` | date/timestamp | from vendor definitions or CME calendar |
| `multiplier_usd` | float64 | 50 / 20 |
| `tick_size` | float64 | 0.25 |
| `tick_value_usd` | float64 | 12.50 / 5.00 |
| `exchange`, `currency` | string | `XCME`, `USD` |
| `source`, `asof` | string, date | provenance |

### 5.4 Roll calendar (`roll_calendar/rule=…`)

| column | notes |
|---|---|
| `product`, `rule` (`c`/`v`/`n`/`fixed_k`) | |
| `from_contract`, `to_contract` | |
| `roll_session_date` | first session on which `to_contract` is the front |
| `decision_basis` | e.g. `volume_prev_day`, and the two volumes/OIs used |
| `panama_offset`, `ratio_factor` | `to.close − from.close` and `to.close / from.close` at roll decision close (previous session) |
| `joint_roll_session_date` | for the ES/NQ pair |
| `asof` | date the calendar was generated |

### 5.5 Manifest (`manifests/<dataset_id>.json`)

```json
{
  "dataset_id": "databento-GLBX.MDP3-ohlcv-1m-ES,NQ-20260615-20260912",
  "source": "databento", "dataset": "GLBX.MDP3", "schema": "ohlcv-1m",
  "stype_in": "continuous", "symbols": ["ES.v.0", "NQ.v.0"],
  "start": "2026-06-15T00:00:00Z", "end": "2026-09-13T00:00:00Z",
  "retrieved_at": "2026-09-14T12:00:00Z", "client_version": "databento==x.y.z",
  "price_scale": 1e-9, "timezone_in": "UTC",
  "files": [{"path": "...", "rows": 123456, "sha256": "..."}],
  "row_count": 123456, "notes": "credentials via DATABENTO_API_KEY env; not stored"
}
```

Deterministic `dataset_id` → deterministic paths → reproducible fixtures (Issue #1 acceptance criterion). Fixtures small enough for tests (a few sessions) can be committed; anything larger lives outside git with the manifest committed.

---

## 6. Decision matrix

Scores 1 (poor) – 5 (good); weights reflect a research project whose priority is *correctness and reproducibility*, then cost.

| Option | Coverage / correctness (×3) | Reproducibility (×3) | Cost (×2) | Effort (×1) | Weighted |
|---|---|---|---|---|---|
| Databento GLBX.MDP3 | 5 | 5 | 3 | 4 | **40** |
| IBKR TWS API | 4 | 2 (2-yr expiry cutoff, login-tz bars) | 4 | 2 | 28 |
| yfinance | 2 | 2 | 5 | 5 | 27 |
| Polygon/Massive | ? (unverified) | ? | ? | 3 | n/a — OQ-1 |
| Alpha Vantage | 0 (no futures) | – | – | – | excluded |

| Continuous method (research default) | Cointegration validity | PnL validity | Lookahead safety | Simplicity | Choice |
|---|---|---|---|---|---|
| Per-contract windows (no stitching) | 5 | 5 | 5 | 3 | **baseline for Task 2** |
| Ratio-adjusted, `v` roll | 5 | 3 | 4 (as-of dated) | 4 | **for log-spread fitting across rolls** |
| Panama-adjusted, `v` roll | 3 | 5 | 4 (as-of dated) | 4 | **for backtest PnL** |
| Unadjusted stitching | 1 | 1 | 5 | 5 | display only |
| Calendar `c` roll (any adjust) | 4 | 4 | 5 | 5 | alternative / guard |

| Storage | Verdict |
|---|---|
| Parquet (zstd) + DuckDB query layer | **chosen** |
| DuckDB native file as artifact | rejected (version coupling) |
| Zarr / HDF5 | rejected (non-tabular / dependency weight) |

---

## 7. Open questions

| ID | Question | Why it matters | How to resolve |
|---|---|---|---|
| OQ-1 | Polygon/Massive futures plan: does it include CME ES/NQ, at what history and price? | Could be a cheaper alternative to Databento | Browser-render <https://massive.com/pricing?product=futures> and the futures REST docs |
| OQ-2 | What contract does Yahoo `ES=F` represent on any given day, and when does it roll? | Determines whether yfinance data can be used for anything roll-sensitive | Compare `ES=F` daily closes to Databento `ES.c.0`/`ES.v.0` over several rolls |
| OQ-3 | ES/NQ message and byte volumes for trades/TBBO/MBP-1 | Cost estimate for the microstructure sample | Databento cost estimator (`metadata.get_cost`) |
| OQ-4 | Confirm ES spec numbers ($50, 0.25 tick, hours, 3 pm CT settlement) on a rendered CME page | Needed for `contracts.parquet` | Browser read of the CME ES spec page or CME rulebook Ch. 358 |
| OQ-5 | Do ES and NQ `v`-rule roll dates ever differ by more than one session? | Determines whether the joint-roll rule is a no-op or material | Compute from Databento definitions + daily volume once ingested |
| OQ-6 | IBKR market-data subscription and `CONTFUT` semantics; is an account available to this project? | Decides whether IBKR is a realistic secondary source | IBKR Campus docs + account check |
| OQ-7 | Primary references for adjustment methods (Panama/ratio) and optimal-sampling literature | Citation hygiene | Re-read Roll (1984, J. Finance), Aït-Sahalia–Mykland–Zhang (2005, RFS), Bandi–Russell (2008, RES) from publisher pages |
| OQ-8 | DuckDB storage-version compatibility policy and `read_parquet` Hive options | Confirms the "query layer only" decision | DuckDB docs (page redirected in this pass) |

---

## 8. Follow-up implementation plan (not started)

| Step | Deliverable | Depends on |
|---|---|---|
| I-1 | `src/index_futures_stat_arb/schema.py`: PyArrow schemas for bars, contracts, roll calendar; manifest dataclass + JSON round-trip | implemented (this PR) |
| I-2 | `ingest/databento.py`: pull `ohlcv-1m` for `ES.v.0`/`NQ.v.0` **and** per-contract `raw_symbol` for the window; write Hive-partitioned zstd Parquet + manifest; env-var key only | implemented (this PR) |
| I-3 | `ingest/yfinance.py`: adapt existing `fetch_yfinance` to the same schema (`contract=UNKNOWN`), 1d + ≤ 60-day 1m | implemented (this PR) |
| I-4 | `reference/rolls.py`: roll-calendar builder for `c`, `v`, `n`, `fixed_k` with one-way + calendar guard; joint ES/NQ roll | implemented (this PR) |
| I-5 | `continuous.py`: Panama / ratio / unadjusted views, as-of dated, forward- and back-adjust | implemented (this PR) |
| I-6 | Loader: DuckDB/Polars read with 1m → 5m resampling, RTH filter, ES↔NQ alignment on `ts_event` | implemented (this PR) |
| I-7 | Tests from §2.5 plus Issue #1 ingestion tests (date-range, missing/duplicate bars, sorting, tz normalisation, source parity, Parquet round-trip, checksum) — using small committed fixtures | implemented (this PR) |
| I-8 | Resolve OQ-1…OQ-8; update this RFC's [A] items to [V] or drop them | — |

Estimated effort: I-1…I-7 is roughly one focused session once Databento credentials are provisioned (`DATABENTO_API_KEY` in `.env`, never committed).

Yahoo Finance is also supported as a keyless continuous-front-month source for
real-data smoke tests; it does not provide per-contract volume for this roll
calendar.

---

## 9. Sources

Primary pages read on 2026-09-14 ([V]):

* yfinance `download` API — <https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html>
* Alpha Vantage documentation — <https://www.alphavantage.co/documentation/>
* Databento GLBX.MDP3 — <https://databento.com/docs/venues-and-datasets/glbx-mdp3>
* Databento symbology (continuous contracts) — <https://databento.com/docs/standards-and-conventions/symbology>
* Databento schemas — <https://databento.com/docs/schemas-and-data-formats>
* Databento pricing — <https://databento.com/pricing>
* IBKR TWS API historical bars — <https://interactivebrokers.github.io/tws-api/historical_bars.html>
* IBKR TWS API historical limitations — <https://interactivebrokers.github.io/tws-api/historical_limitations.html>
* CME Equity Index Roll Dates — <https://www.cmegroup.com/trading/equity-index/rolldates.html>
* CME E-mini Nasdaq-100 contract page — <https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.contractSpecs.html>
* PyArrow Parquet — <https://arrow.apache.org/docs/python/parquet.html>
* Zarr user guide — <https://zarr.readthedocs.io/en/stable/user-guide/>

Attempted, not rendered / not verified ([A]):

* CME E-mini S&P 500 spec table — <https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.contractSpecs.html>
* Massive (Polygon) futures pricing/docs — <https://massive.com/pricing?product=futures>, <https://massive.com/docs/rest/futures/overview>
* DuckDB Parquet & storage docs — <https://duckdb.org/docs/stable/data/parquet/overview>, <https://duckdb.org/docs/stable/internals/storage>
* Roll (1984); Aït-Sahalia, Mykland & Zhang (2005); Bandi & Russell (2008) — cited from memory, to be re-read (OQ-7)
