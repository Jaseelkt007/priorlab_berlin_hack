const C = require('./core.js'); const { CFG } = C;
// calibration on held-out random jobs at random actions, using context = commissioning + 150 random phase-3 rows
const r = C.mulberry(5), jobs = C.makeJobs(21);
let ctx = C.commissioning(11);
for (let i = 0; i < 150; i++) { const c = jobs[3][i], v = CFG.SPEEDS[Math.floor(r()*7)], tu = CFG.TUNES[Math.floor(r()*4)]; ctx.push({ x: C.rowFeat(c, c.meas, v, tu), ...C.simulateMove(c, C.plan(v, c.L1 + tu)) }); }
const zs = []; let se = 0, sd = 0;
for (let i = 150; i < 270; i++) { const c = jobs[3][i], v = CFG.SPEEDS[Math.floor(r()*7)], tu = CFG.TUNES[Math.floor(r()*4)]; const p = C.predict(ctx, C.rowFeat(c, c.meas, v, tu)), o = C.simulateMove(c, C.plan(v, c.L1 + tu)); zs.push((o.amp - p.amp) / p.sdA); se += (o.amp - p.amp) ** 2; sd += p.sdA; }
const n = zs.length; console.log('rmse', Math.sqrt(se / n).toFixed(3), 'mean sd', (sd / n).toFixed(3), 'cov90', (zs.filter(z => Math.abs(z) < 1.645).length / n).toFixed(2), 'cov50', (zs.filter(z => Math.abs(z) < 0.674).length / n).toFixed(2));
