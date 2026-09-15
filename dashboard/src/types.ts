export interface Metrics {
  fees_slippage: number
  max_dd_pct: number
  max_dd_usd: number
  pnl: number
  round_trips: number
  sessions: number
  sharpe: number
  turnover: number
  win_rate: number
}

export interface Selection {
  base: { oos: Metrics; trial_id: number }
  best_in_sample: { in_sample_sharpe: number; oos_sharpe: number; trial_id: number }
  recommended: {
    oos_over_test_span: Metrics
    params: Record<string, number | null>
    rank_stability: number
    trial_id: number
  }
  rule: string
  selected_trials: { fold: number; trial_id: number }[]
  stitched_oos: Metrics
}

export interface Fold {
  fold: number
  selected_trial: number
  sharpe: number
  max_dd_pct: number
  pnl: number
  round_trips: number
  trades: number
  turnover: number
  win_rate: number
  fees_slippage: number
  train_start: string
  train_end: string
  test_start: string
  test_end: string
}

export interface Trial {
  trial_id: number
  kalman_delta: number
  kalman_obs_var: number
  entry: number
  exit_ratio: number
  stop_ratio: number
  half_life_min_bars: number
  half_life_max_bars: number | null
  max_holding_half_lives: number
  min_edge_cost_multiple: number | null
  ofi_threshold: number | null
  coint_pvalue_gate: number
  in_sample_sharpe: number
  in_sample_pnl: number
  median_oos_sharpe: number
  min_oos_sharpe: number
  fold_sharpes: (number | null)[]
  fold_round_trips: (number | null)[]
  oos_round_trips: number
}

export interface Stream {
  id: string
  label: string
  interval: '1d' | '5m'
  selection: Selection
  folds: Fold[]
  trials: Trial[]
  plots: string[]
}

export interface Payload {
  generated_from: string
  streams: Stream[]
}
