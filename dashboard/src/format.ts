export const fmtSharpe = (v: number | null | undefined): string =>
  v === null || v === undefined ? '—' : v.toFixed(2)

export const fmtPct = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined ? '—' : `${v.toFixed(digits)}%`

export const fmtRate = (v: number | null | undefined): string =>
  v === null || v === undefined ? '—' : `${(v * 100).toFixed(0)}%`

export const fmtUsd = (v: number | null | undefined): string => {
  if (v === null || v === undefined) return '—'
  const sign = v < 0 ? '−' : v > 0 ? '+' : ''
  return `${sign}$${Math.abs(Math.round(v)).toLocaleString('en-US')}`
}

export const fmtParam = (v: number | null | undefined): string => {
  if (v === null || v === undefined) return 'off'
  if (Math.abs(v) < 1e-3) return v.toExponential(1)
  return Number(v.toPrecision(3)).toString()
}

export const fmtInt = (v: number | null | undefined): string =>
  v === null || v === undefined ? '—' : Math.round(v).toLocaleString('en-US')

export const tStat = (sharpe: number, sessions: number): number =>
  sharpe * Math.sqrt(sessions / 252)
