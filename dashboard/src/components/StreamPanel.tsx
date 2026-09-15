import { useMemo, useState } from 'react'
import type { Metrics, Stream, Trial } from '../types'
import { fmtInt, fmtParam, fmtPct, fmtRate, fmtSharpe, fmtUsd, tStat } from '../format'
import { FoldChart } from './FoldChart'

interface Props {
  stream: Stream
}

const PLOT_TITLES: Record<string, string> = {
  'stitched_equity.png': 'Stitched OOS equity (selected-per-fold vs base trial 0; fold boundaries marked)',
  'oos_sharpe_hist.png': 'Distribution of median OOS Sharpe across trials',
  'param_vs_oos.png': 'Sampled parameter vs median OOS Sharpe',
}

const tone = (v: number): 'pos' | 'neg' | undefined => (v > 0 ? 'pos' : v < 0 ? 'neg' : undefined)

function MetricsRow({ label, trial, metrics, note, highlight }: {
  label: string
  trial: string
  metrics: Metrics
  note?: string
  highlight?: boolean
}) {
  return (
    <tr className={highlight ? 'highlight' : undefined}>
      <td>
        <strong>{label}</strong>
        {note && <div className="muted small">{note}</div>}
      </td>
      <td>{trial}</td>
      <td className={tone(metrics.sharpe)}>{fmtSharpe(metrics.sharpe)}</td>
      <td className={tone(metrics.pnl)}>{fmtUsd(metrics.pnl)}</td>
      <td>{fmtPct(metrics.max_dd_pct)}</td>
      <td>{fmtRate(metrics.win_rate)}</td>
      <td>{fmtInt(metrics.round_trips)}</td>
      <td>{metrics.turnover.toFixed(1)}</td>
      <td>{fmtUsd(metrics.fees_slippage).replace('+', '')}</td>
    </tr>
  )
}

export function StreamPanel({ stream }: Props) {
  const { selection, folds, trials } = stream
  const [showAll, setShowAll] = useState(false)
  const stitched = selection.stitched_oos
  const bestTrial = trials.find((t) => t.trial_id === selection.best_in_sample.trial_id)
  const t = tStat(stitched.sharpe, stitched.sessions)
  const years = stitched.sessions / 252

  const ranked = useMemo(() => {
    const active = trials.filter((tr) => tr.oos_round_trips > 0)
    const sorted = [...active].sort(
      (a, b) => b.median_oos_sharpe - a.median_oos_sharpe || a.trial_id - b.trial_id,
    )
    return showAll ? sorted : sorted.slice(0, 10)
  }, [trials, showAll])
  const activeCount = trials.filter((tr) => tr.oos_round_trips > 0).length
  const churn = new Set(selection.selected_trials.map((s) => s.trial_id)).size
  const sparse = stitched.round_trips < 30
  const intraday = stream.interval === '5m'

  return (
    <section className="panel" id={stream.id}>
      <header className="panel-head">
        <h2>{stream.label}</h2>
        <div className="kpis">
          <Kpi label="Stitched OOS Sharpe" value={fmtSharpe(stitched.sharpe)} tone={tone(stitched.sharpe)} />
          <Kpi label="OOS PnL ($1M capital)" value={fmtUsd(stitched.pnl)} tone={tone(stitched.pnl)} />
          <Kpi label="Max drawdown" value={fmtPct(stitched.max_dd_pct)} />
          <Kpi label="Round trips" value={fmtInt(stitched.round_trips)} />
          <Kpi label="t-stat (Sharpe·√years)" value={t.toFixed(2)} tone={Math.abs(t) < 2 ? 'warn' : undefined} />
        </div>
      </header>

      <div className="callouts">
        {intraday ? (
          <p className="callout warn">
            <strong>Uninformative:</strong> Yahoo retains only ~60 days of 5-minute bars ({fmtInt(stitched.sessions)} OOS sessions over
            {folds.length} folds). {stitched.round_trips === 0 ? 'No trial produced an out-of-sample entry.' : `${fmtInt(stitched.round_trips)} OOS round trips.`}{' '}
            Reported as executed, not as evidence.
          </p>
        ) : (
          <p className={`callout ${stitched.sharpe > 0 ? 'note' : 'warn'}`}>
            <strong>{stitched.sharpe > 0 ? 'No demonstrated edge (not significant).' : 'No edge.'}</strong>{' '}
            The honest number is the stitched OOS row: {fmtSharpe(stitched.sharpe)} Sharpe over {years.toFixed(1)} years and{' '}
            {fmtInt(stitched.round_trips)} round trips (t ≈ {t.toFixed(1)}). Selection picked {churn} distinct trials across{' '}
            {folds.length} folds{churn >= folds.length - 1 ? ' — parameter choice is unstable.' : '.'}
          </p>
        )}
        {sparse && !intraday && (
          <p className="callout muted-box">
            Sample size: fold Sharpes rest on {Math.min(...folds.map((f) => f.round_trips))}–{Math.max(...folds.map((f) => f.round_trips))} trades each; per-fold
            Sharpe standard error is ≥ 0.7 (≈ 1/√years per fold). "Recommended" and "best in-sample" rows below use all OOS folds and are descriptions of this
            sample, not forecasts.
          </p>
        )}
      </div>

      <h3>Selection comparison (OOS span: {folds[0]?.test_start} → {folds[folds.length - 1]?.test_end}, {fmtInt(stitched.sessions)} sessions)</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Trial</th>
              <th>Sharpe</th>
              <th>PnL</th>
              <th>MaxDD</th>
              <th>Win</th>
              <th>Round trips</th>
              <th>Turnover</th>
              <th>Fees + slippage</th>
            </tr>
          </thead>
          <tbody>
            <MetricsRow
              label="Stitched OOS (procedure)"
              note="Only row free of selection bias"
              trial={selection.selected_trials.map((s) => s.trial_id).join(', ')}
              metrics={stitched}
              highlight
            />
            <MetricsRow label="Base config OOS" note="Trial 0, fixed parameters" trial={String(selection.base.trial_id)} metrics={selection.base.oos} />
            <MetricsRow
              label="Recommended model"
              note="argmax median OOS Sharpe — uses all OOS folds (biased)"
              trial={String(selection.recommended.trial_id)}
              metrics={selection.recommended.oos_over_test_span}
            />
          </tbody>
        </table>
      </div>
      <p className="small muted">
        Best in-sample trial: #{selection.best_in_sample.trial_id} (IS Sharpe {fmtSharpe(selection.best_in_sample.in_sample_sharpe)}, OOS over
        test span {fmtSharpe(selection.best_in_sample.oos_sharpe)}
        {bestTrial ? `, ${fmtInt(bestTrial.oos_round_trips)} OOS round trips` : ''}). Selection rule: <code>{selection.rule}</code>.
        Recommended-model rank stability: {selection.recommended.rank_stability}/{folds.length} folds in the top quartile.
      </p>

      <h3>Selection behaviour per fold</h3>
      <div className="two-col">
        <FoldChart folds={folds} />
        <div className="table-wrap">
          <table className="compact">
            <thead>
              <tr>
                <th>Fold</th>
                <th>Train</th>
                <th>Test</th>
                <th>Selected</th>
                <th>Sharpe</th>
                <th>MaxDD</th>
                <th>Win</th>
                <th>Trips</th>
              </tr>
            </thead>
            <tbody>
              {folds.map((f) => (
                <tr key={f.fold}>
                  <td>{f.fold}</td>
                  <td className="nowrap small">{f.train_start} → {f.train_end}</td>
                  <td className="nowrap small">{f.test_start} → {f.test_end}</td>
                  <td>{f.selected_trial}</td>
                  <td className={tone(f.sharpe)}>{fmtSharpe(f.sharpe)}</td>
                  <td>{fmtPct(f.max_dd_pct)}</td>
                  <td>{fmtRate(f.win_rate)}</td>
                  <td>{fmtInt(f.round_trips)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <h3>
        Trials ranked by median OOS Sharpe ({activeCount} of {trials.length} trials traded out of sample)
        <button className="link" onClick={() => setShowAll((v) => !v)}>
          {showAll ? 'show top 10' : 'show all'}
        </button>
      </h3>
      <div className="table-wrap">
        <table className="compact">
          <thead>
            <tr>
              <th>#</th>
              <th>Q (δ)</th>
              <th>R</th>
              <th>Entry</th>
              <th>Exit/entry</th>
              <th>Stop/entry</th>
              <th>HL min</th>
              <th>HL max</th>
              <th>Max hold (HL)</th>
              <th>Edge×cost</th>
              <th>OFI thr</th>
              <th>Coint p</th>
              <th>IS Sharpe</th>
              {folds.map((f) => (
                <th key={f.fold}>F{f.fold}</th>
              ))}
              <th>Median OOS</th>
              <th>Min OOS</th>
              <th>OOS trips</th>
            </tr>
          </thead>
          <tbody>
            {ranked.length === 0 && (
              <tr>
                <td colSpan={17 + folds.length} className="muted">
                  No trial produced an out-of-sample round trip on this stream.
                </td>
              </tr>
            )}
            {ranked.map((tr) => (
              <TrialRow key={tr.trial_id} trial={tr} recommended={tr.trial_id === selection.recommended.trial_id} />
            ))}
          </tbody>
        </table>
      </div>

      <h3>Plots (generated by <code>ifsa walkforward</code>)</h3>
      <div className="plots">
        {stream.plots.map((p) => (
          <figure key={p}>
            <img src={`${import.meta.env.BASE_URL}plots/${stream.id}/${p}`} alt={PLOT_TITLES[p] ?? p} loading="lazy" />
            <figcaption>{PLOT_TITLES[p] ?? p}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  )
}

function Kpi({ label, value, tone }: { label: string; value: string; tone?: 'pos' | 'neg' | 'warn' }) {
  return (
    <div className={`kpi ${tone ?? ''}`}>
      <div className="kpi-value">{value}</div>
      <div className="kpi-label">{label}</div>
    </div>
  )
}

function TrialRow({ trial, recommended }: { trial: Trial; recommended: boolean }) {
  return (
    <tr className={recommended ? 'highlight' : undefined}>
      <td>{trial.trial_id}{trial.trial_id === 0 ? ' (base)' : ''}</td>
      <td>{fmtParam(trial.kalman_delta)}</td>
      <td>{fmtParam(trial.kalman_obs_var)}</td>
      <td>{fmtParam(trial.entry)}</td>
      <td>{fmtParam(trial.exit_ratio)}</td>
      <td>{fmtParam(trial.stop_ratio)}</td>
      <td>{fmtParam(trial.half_life_min_bars)}</td>
      <td>{fmtParam(trial.half_life_max_bars)}</td>
      <td>{fmtParam(trial.max_holding_half_lives)}</td>
      <td>{fmtParam(trial.min_edge_cost_multiple)}</td>
      <td>{fmtParam(trial.ofi_threshold)}</td>
      <td>{fmtParam(trial.coint_pvalue_gate)}</td>
      <td>{fmtSharpe(trial.in_sample_sharpe)}</td>
      {trial.fold_sharpes.map((s, i) => (
        <td key={i} className={s !== null && s < 0 ? 'neg' : undefined}>
          {fmtSharpe(s)}
        </td>
      ))}
      <td>{fmtSharpe(trial.median_oos_sharpe)}</td>
      <td>{fmtSharpe(trial.min_oos_sharpe)}</td>
      <td>{fmtInt(trial.oos_round_trips)}</td>
    </tr>
  )
}
