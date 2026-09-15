// Packages walk-forward artifacts from ../data/results/walkforward into
// src/generated/walkforward.json (deterministic: sorted keys, fixed stream order)
// and copies the PNG plots into public/plots/<stream>/.
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(here, '..', '..')
const source = resolve(repoRoot, 'data', 'results', 'walkforward')
const outJson = resolve(here, '..', 'src', 'generated', 'walkforward.json')
const outPlots = resolve(here, '..', 'public', 'plots')

const STREAMS = [
  { id: 'es_spy_daily', label: 'ES=F / SPY — daily', interval: '1d' },
  { id: 'nq_qqq_daily', label: 'NQ=F / QQQ — daily', interval: '1d' },
  { id: 'es_spy_5m', label: 'ES=F / SPY — 5-minute', interval: '5m' },
  { id: 'nq_qqq_5m', label: 'NQ=F / QQQ — 5-minute', interval: '5m' },
]
const PLOTS = ['stitched_equity.png', 'oos_sharpe_hist.png', 'param_vs_oos.png']

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/)
  const header = lines[0].split(',')
  return lines.slice(1).map((line) => {
    const cells = line.split(',')
    const row = {}
    header.forEach((key, i) => {
      const v = cells[i]
      if (v === undefined || v === '' || v === 'nan') row[key] = null
      else if (/^-?\d{4}-\d{2}-\d{2}$/.test(v)) row[key] = v
      else if (!Number.isNaN(Number(v))) row[key] = Number(v)
      else row[key] = v
    })
    return row
  })
}

function sortKeys(value) {
  if (Array.isArray(value)) return value.map(sortKeys)
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((k) => [k, sortKeys(value[k])]),
    )
  }
  return value
}

const streams = []
for (const stream of STREAMS) {
  const dir = join(source, stream.id)
  if (!existsSync(dir)) {
    console.warn(`missing ${dir}; skipping`)
    continue
  }
  const selection = JSON.parse(readFileSync(join(dir, 'selection.json'), 'utf8'))
  const folds = parseCsv(readFileSync(join(dir, 'folds.csv'), 'utf8'))
  const trialsRaw = parseCsv(readFileSync(join(dir, 'trials.csv'), 'utf8'))
  const foldCount = folds.length
  const trials = trialsRaw.map((row) => {
    const out = {}
    for (const [k, v] of Object.entries(row)) if (!k.startsWith('fold_')) out[k] = v
    out.fold_sharpes = Array.from({ length: foldCount }, (_, i) => row[`fold_${i}_sharpe`] ?? null)
    out.fold_round_trips = Array.from({ length: foldCount }, (_, i) => row[`fold_${i}_round_trips`] ?? null)
    out.oos_round_trips = out.fold_round_trips.reduce((a, b) => a + (b ?? 0), 0)
    return out
  })
  const plots = PLOTS.filter((p) => existsSync(join(dir, p)))
  mkdirSync(join(outPlots, stream.id), { recursive: true })
  for (const p of plots) copyFileSync(join(dir, p), join(outPlots, stream.id, p))
  streams.push({ ...stream, selection, folds, trials, plots })
}

const payload = sortKeys({ generated_from: 'data/results/walkforward', streams })
mkdirSync(dirname(outJson), { recursive: true })
writeFileSync(outJson, JSON.stringify(payload, null, 2) + '\n')
console.log(`packaged ${streams.length} streams -> ${outJson}`)
for (const dir of readdirSync(outPlots)) console.log(`plots: ${dir}`)
