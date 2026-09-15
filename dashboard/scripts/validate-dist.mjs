// Build validation: dist must exist, use relative asset URLs (Pages subpath safe),
// ship every plot, and embed the exact stitched OOS Sharpe values from the source artifacts.
import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const dist = resolve(here, '..', 'dist')
const source = resolve(here, '..', '..', 'data', 'results', 'walkforward')
const fail = (msg) => { console.error(`FAIL: ${msg}`); process.exit(1) }

if (!existsSync(join(dist, 'index.html'))) fail('dist/index.html missing')
const html = readFileSync(join(dist, 'index.html'), 'utf8')
for (const m of html.matchAll(/(?:src|href)="([^"]+)"/g)) {
  if (m[1].startsWith('/')) fail(`absolute asset URL in index.html: ${m[1]}`)
}
const js = readdirSync(join(dist, 'assets')).filter((f) => f.endsWith('.js')).map((f) => readFileSync(join(dist, 'assets', f), 'utf8')).join('\n')

const streams = readdirSync(source).filter((d) => existsSync(join(source, d, 'selection.json')))
if (streams.length === 0) fail('no source streams found')
for (const s of streams) {
  for (const p of ['stitched_equity.png', 'oos_sharpe_hist.png', 'param_vs_oos.png']) {
    if (!existsSync(join(dist, 'plots', s, p))) fail(`missing plot ${s}/${p}`)
  }
  const sel = JSON.parse(readFileSync(join(source, s, 'selection.json'), 'utf8'))
  const sharpe = sel.stitched_oos.sharpe
  const embedded = [...js.matchAll(/sharpe:(-?\d*\.?\d+(?:e-?\d+)?)/g)].some((m) => Number(m[1]) === sharpe)
  if (!embedded) fail(`stitched OOS sharpe ${sharpe} for ${s} not embedded in bundle`)
}
console.log(`OK: ${streams.length} streams, relative assets, plots present, metrics embedded`)
