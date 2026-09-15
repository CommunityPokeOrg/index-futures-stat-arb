import payload from './generated/walkforward.json'
import type { Payload } from './types'
import { StreamPanel } from './components/StreamPanel'
import { fmtSharpe, tStat } from './format'

const data = payload as Payload
const REPO = 'https://github.com/CommunityPokeOrg/index-futures-stat-arb'

export default function App() {
  const daily = data.streams.filter((s) => s.interval === '1d')
  const intraday = data.streams.filter((s) => s.interval === '5m')
  const dailyTrips = daily.flatMap((s) => s.folds.map((f) => f.round_trips))
  const trialCounts = [...new Set(data.streams.map((s) => s.trials.length))].join('/')
  const foldCounts = [...new Set(data.streams.map((s) => s.folds.length))].join('/')

  return (
    <div className="page">
      <header className="hero">
        <p className="eyebrow">index-futures-stat-arb · walk-forward evaluation</p>
        <h1>ES=F/SPY and NQ=F/QQQ basis stat-arb — out-of-sample results</h1>
        <p className="lede">
          Real Yahoo Finance daily and 5-minute bars, $1,000,000 initial capital, β≡1 cash-and-carry basis with a Kalman level
          filter, realistic ES/NQ/ETF commissions and slippage. Every number below is read from the committed artifacts in{' '}
          <code>{data.generated_from}</code>; nothing is recomputed in the browser.
        </p>
        <div className="verdict">
          <h2>Honest conclusion: no demonstrated edge</h2>
          <ul>
            {daily.map((s) => {
              const m = s.selection.stitched_oos
              return (
                <li key={s.id}>
                  <strong>{s.label}</strong>: stitched OOS Sharpe {fmtSharpe(m.sharpe)} (t ≈ {tStat(m.sharpe, m.sessions).toFixed(1)},{' '}
                  {m.round_trips} round trips, {(m.sessions / 252).toFixed(1)} years) — {m.sharpe > 0 ? 'mildly positive, not significant' : 'negative'}.
                </li>
              )
            })}
            <li>
              <strong>5-minute streams</strong>: Yahoo retains ~60 days; {intraday.map((s) => `${s.label.split(' — ')[0]} ${s.selection.stitched_oos.round_trips} OOS trips`).join(', ')} — uninformative.
            </li>
            <li>
              "Recommended"/"best in-sample" rows are chosen using all OOS folds and are <em>post-hoc descriptions</em>, not unbiased
              forecasts. {trialCounts} seeded trials, {foldCounts} anchored folds, {Math.min(...dailyTrips)}–{Math.max(...dailyTrips)} round trips per daily fold: per-fold Sharpe standard error is ≈ 1/√years ≥ 0.7.
            </li>
          </ul>
        </div>
        <nav className="toc">
          {data.streams.map((s) => (
            <a key={s.id} href={`#${s.id}`}>{s.label}</a>
          ))}
          <a href="#method">Method</a>
        </nav>
      </header>

      <main>
        {data.streams.map((s) => (
          <StreamPanel key={s.id} stream={s} />
        ))}

        <section className="panel" id="method">
          <h2>Method and selection rule</h2>
          <ol>
            <li>Trials ({trialCounts} per stream) sampled with a seeded NumPy RNG over Kalman Q/R, OU entry/exit/stop ratios, half-life bounds, max holding period, min edge-over-cost multiple, bar-derived OFI-proxy threshold, and cointegration p-value gate. Trial 0 is the hand-set base configuration.</li>
            <li>Each trial is simulated once, causally, over the full sample; PnL is sliced into anchored contiguous train/test folds (no shuffling, no future information in state).</li>
            <li>Fold 0 always uses trial 0. For fold k ≥ 1 the trial maximising median(prior-fold Sharpe) − ½·IQR is selected, restricted to trials with a minimum trade count in every prior fold; ties go to the lowest trial id.</li>
            <li>The <em>stitched OOS</em> series concatenates each fold's test PnL under its pre-selected trial. It is the only performance figure here that is free of selection bias.</li>
          </ol>
          <p className="small muted">
            Data: Yahoo Finance ES=F, NQ=F, SPY, QQQ, ^IRX (financing) and SPY/QQQ dividends. Yahoo intraday retention is ~60 days for 5m and
            ~30 days for 1m; daily history starts 2015 for the futures. Costs: ES $50/pt, NQ $20/pt, $1.25 commission + $1.38 fee per
            contract, 1 tick slippage + ½ tick half-spread; ETF $0.005/share, $0.01 tick. Reproduce with{' '}
            <code>ifsa walkforward --config configs/wf_&lt;stream&gt;.toml --out data/results/walkforward/&lt;stream&gt;</code>. Source, docs and
            methodology: <a href={REPO}>{REPO.replace('https://', '')}</a>.
          </p>
        </section>
      </main>
      <footer className="small muted">
        Research artefact. Not investment advice. Synthetic and bootstrap simulations in the repository are harness/power tests, never
        evidence of edge.
      </footer>
    </div>
  )
}
