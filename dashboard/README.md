# Walk-forward results dashboard

Static Vite + React showcase of the committed walk-forward artefacts under
`../data/results/walkforward/<stream>/` (ES=F/SPY and NQ=F/QQQ, daily and 5m). It renders
only what is in those files — stitched OOS metrics, per-fold selection behaviour, trial
tables, the generated PNG plots — plus explicit warnings about sample size, Yahoo's 60-day
5-minute retention, and the absence of a demonstrated edge. Nothing is recomputed in the
browser.

## Commands

```bash
cd dashboard
npm ci
npm run dev        # package data from ../data/results/walkforward, start Vite dev server
npm run build      # package data -> tsc -b -> vite build (dist/)
npm run preview    # serve dist/ locally
npm test           # build + scripts/validate-dist.mjs
npm run lint       # oxlint
```

`scripts/package-data.mjs` converts `selection.json`, `folds.csv`, `trials.csv` into a
single `src/generated/walkforward.json` (sorted keys, fixed stream order — deterministic for
identical inputs) and copies the plots into `public/plots/<stream>/`. Both outputs are
git-ignored and regenerated on every build, so the dashboard can never drift from the
artefacts. Regenerate the artefacts themselves with `ifsa walkforward` (see the repo README).

`scripts/validate-dist.mjs` fails the build if `dist/index.html` uses absolute asset URLs
(which would break at a Pages subpath), if any plot is missing, or if the exact stitched OOS
Sharpe from each `selection.json` is not embedded in the bundle.

## GitHub Pages

`vite.config.ts` uses `base: './'`, so the bundle works at `https://<org>.github.io/<repo>/`
or any other subpath. `.github/workflows/pages.yml` runs lint + `npm test` on pull requests
and pushes touching `dashboard/**` or the artefacts, uploads `dashboard/dist` with
`actions/upload-pages-artifact`, and deploys with `actions/deploy-pages` on the default
branch. One-time repository setting: **Settings → Pages → Source: GitHub Actions**. The
deployed URL appears as the `github-pages` environment URL on the workflow run.
