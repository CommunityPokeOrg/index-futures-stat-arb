# Research documents index

Start here: **`02_consolidated_findings_and_local_continuation.md`** — consolidated metrics,
method specification, retail viability, local setup/commands, and acceptance criteria.

| Document | Purpose |
|---|---|
| `01_data_ingestion_and_contract_roll_methods.md` | Original RFC: Databento ingestion, contract selection, roll rules, continuous series |
| `02_consolidated_findings_and_local_continuation.md` | Consolidation of all findings to date; Kalman/pairs engine spec; retail viability; how to continue locally |
| `next_steps_validation_protocol.md` | Falsifiable hypotheses H1–H8, `[BIAS↓]` vs `[FIT]` labels, validation protocol and acceptance criteria |

Executed results (chronological) live in `../results/`:
`2026-09-14_synthetic_simulation.md` (SYNTHETIC), `2026-09-14_yahoo_real_data.md`,
`2026-09-14_refined_method_real_data.md`, `2026-09-15_index_vs_etf_basis_real_data.md`,
`2026-09-15_walkforward_evaluation.md`, `2026-09-15_montecarlo_evaluation.md` (resampled/synthetic).

Current conclusion (unchanged): no demonstrated edge. Stitched walk-forward OOS on real Yahoo
daily data — ES/SPY Sharpe −0.16; NQ/QQQ Sharpe 0.46 (t ≈ 1.4, 18 trades); 5-minute streams
uninformative.
