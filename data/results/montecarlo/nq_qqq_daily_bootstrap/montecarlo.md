# Monte Carlo harness — SYNTHETIC / RESAMPLED — harness test, not evidence of edge

```json
{
  "block_len": 14,
  "disclaimer": "SYNTHETIC / RESAMPLED \u2014 harness test, not evidence of edge",
  "distribution": {
    "max_dd_pct_percentiles": {
      "5": 4.571672168786609,
      "50": 9.288401245117187,
      "95": 16.826707612384027
    },
    "pnl_percentiles": {
      "5": 31373.453581787136,
      "50": 249688.448838501,
      "95": 498834.03447869857
    },
    "probability_of_loss": 0.028,
    "sharpe_percentiles": {
      "5": 0.0572818478884916,
      "50": 0.468387840106206,
      "95": 0.9288212661943919
    }
  },
  "mode": "bootstrap",
  "n_observations": 2438,
  "n_paths": 10000,
  "null": {
    "null_sharpe_mean": -0.0042433113017110585,
    "null_sharpe_percentiles": {
      "5": -0.4376930596846749,
      "50": -0.003967642671755221,
      "95": 0.4298453500827858
    },
    "p_value_observed_sharpe": 0.0399
  },
  "observed": {
    "loss_probability": 0.0,
    "max_dd_pct": 9.288401245117187,
    "pnl": 252798.01,
    "sharpe": 0.4559306164518576
  }
}
```
