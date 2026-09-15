import type { Fold } from '../types'
import { fmtSharpe } from '../format'

interface Props {
  folds: Fold[]
}

// Per-fold OOS Sharpe of the trial the selection rule picked, as an SVG bar chart.
export function FoldChart({ folds }: Props) {
  const width = 560
  const height = 230
  const pad = { l: 40, r: 12, t: 18, b: 52 }
  const values = folds.map((f) => f.sharpe)
  const maxAbs = Math.max(1, ...values.map((v) => Math.abs(v)))
  const innerW = width - pad.l - pad.r
  const innerH = height - pad.t - pad.b
  const zeroY = pad.t + innerH / 2
  const scale = innerH / 2 / maxAbs
  const slot = innerW / folds.length
  const barW = slot * 0.6

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Out-of-sample Sharpe per fold for the selected trial"
      className="fold-chart"
    >
      <line x1={pad.l} x2={width - pad.r} y1={zeroY} y2={zeroY} className="axis" />
      {[maxAbs, -maxAbs].map((tick) => (
        <text key={tick} x={pad.l - 6} y={zeroY - tick * scale + 4} textAnchor="end" className="tick">
          {tick.toFixed(1)}
        </text>
      ))}
      <text x={pad.l - 6} y={zeroY + 4} textAnchor="end" className="tick">
        0
      </text>
      {folds.map((f, i) => {
        const x = pad.l + i * slot + (slot - barW) / 2
        const h = Math.abs(f.sharpe) * scale
        const y = f.sharpe >= 0 ? zeroY - h : zeroY
        return (
          <g key={f.fold}>
            <rect x={x} y={y} width={barW} height={Math.max(h, 0.5)} className={f.sharpe >= 0 ? 'bar pos' : 'bar neg'} />
            <text
              x={x + barW / 2}
              y={f.sharpe >= 0 ? Math.max(y - 4, pad.t) : Math.min(y + h + 12, height - pad.b - 2)}
              textAnchor="middle"
              className="value"
            >
              {fmtSharpe(f.sharpe)}
            </text>
            <text x={x + barW / 2} y={height - pad.b + 16} textAnchor="middle" className="tick">
              fold {f.fold} · trial {f.selected_trial}
            </text>
            <text x={x + barW / 2} y={height - pad.b + 30} textAnchor="middle" className="tick muted">
              {f.test_start.slice(0, 7)}→{f.test_end.slice(0, 7)}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
