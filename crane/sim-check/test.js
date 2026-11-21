const C = require('./core.js'); const { CFG } = C;
const jobs = C.makeJobs(7); const rng = C.mulberry(3);
const t0 = Date.now(); let ctx = C.commissioning(11); console.log('commission ms', Date.now() - t0, 'rows', ctx.length);
const strat = {
  slow: c => C.plan(0.4, 0),
  full: c => C.plan(1.0, 0),
  physics: c => C.plan(1.0, c.L1 + 0.5),
};
const N = +process.argv[2] || 15;
for (let p = 0; p < 4; p++) {
  const out = {};
  for (const k of [...Object.keys(strat), 'tabpfn']) out[k] = { cyc: 0, hit: 0, near: 0, set: 0, v: 0 };
  let cover = 0;
  for (let i = 0; i < N; i++) {
    const c = jobs[p][i];
    for (const k in strat) { const pl = strat[k](c), o = C.simulateMove(c, pl); const R = out[k]; R.cyc += pl.T + o.settle + CFG.SETDOWN + (o.hit ? CFG.HIT_PENALTY : 0); R.hit += o.hit; R.near += !o.hit && o.amp > CFG.NEAR * CFG.CLEAR; R.set += o.settle; }
    const d = C.decide(ctx, c, c.meas, rng), b = d.best, pl = C.plan(b.v, c.L1 + b.tu), o = C.simulateMove(c, pl);
    const R = out.tabpfn; R.cyc += pl.T + o.settle + CFG.SETDOWN + (o.hit ? CFG.HIT_PENALTY : 0); R.hit += o.hit; R.near += !o.hit && o.amp > CFG.NEAR * CFG.CLEAR; R.set += o.settle; R.v += b.v;
    cover += o.amp >= b.lo && o.amp <= b.hi;
    ctx.push({ x: C.rowFeat(c, c.meas, b.v, b.tu), ...o });
  }
  console.log('phase', p, 'coverage90', (cover / N).toFixed(2));
  for (const k in out) { const R = out[k]; console.log(' ', k.padEnd(8), 'cycle', (R.cyc / N).toFixed(1), 'settle', (R.set / N).toFixed(1), 'hits', R.hit, 'near', R.near, k === 'tabpfn' ? 'avgV ' + (R.v / N).toFixed(2) : ''); }
}
