// ===== CORE (pure; no DOM). Copied verbatim into the page between CORE markers. =====
const G = 9.81;
const CFG = {
  D: 10,          // travel per move, m
  A: 0.4,         // trolley/bridge acceleration limit, m/s^2
  VMAX: 1.2,      // rated travel speed, m/s (72 m/min)
  ZETA: 0.06,     // effective sway damping ratio (rope losses + operator corrections), assumption
  TOL: 0.05,      // set-down tolerance: |load offset| below this, m
  HOLD: 1.0,      // must stay within tolerance this long, s
  CLEAR: 0.45,    // clearance to the racks either side of the set-down bay, m
  NEAR: 0.6,      // near-miss = residual amplitude above NEAR * CLEAR
  SETTLE_CAP: 30, // operator steadies the load by hand after this, s
  HIT_PENALTY: 20,// inspection stop after a collision, s
  SETDOWN: 4,     // lower, unhook, hook next load, s
  DT: 1 / 200,    // physics step, s
  SPEEDS: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
  TUNES: [0.5, 1.5, 2.5, 3.5], // sling length the shaper assumes, m (tuning length = L1 + this)
  RISK: 0.01,     // max accepted collision probability per move
};

function mulberry(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function gauss(r) { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
function erf(x) { const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; }
const Phi = z => 0.5 * (1 + erf(z / Math.SQRT2));

// Linearised double pendulum: the two sway frequencies (rad/s). R = payload / hook mass.
function modes(L1, L2, R) {
  const s = (1 + R) * (1 / L1 + 1 / L2);
  const d = Math.sqrt(Math.max(0, s * s - 4 * (1 + R) / (L1 * L2)));
  return [Math.sqrt(G / 2 * (s - d)), Math.sqrt(G / 2 * (s + d))];
}

// Full nonlinear double pendulum hanging from a trolley with acceleration a.
// st = [th1, w1, th2, w2]; c = {L1, L2, m1, m2}. Damping as a ratio on each mode's own scale.
function deriv(st, a, c, out) {
  const [t1, w1, t2, w2] = st, { L1, L2, m1, m2 } = c, M = m1 + m2, d = t1 - t2;
  const cd = Math.cos(d), sd = Math.sin(d);
  const k1 = 2 * CFG.ZETA * Math.sqrt(G / L1), k2 = 2 * CFG.ZETA * Math.sqrt(G / L2);
  const a11 = M * L1, a12 = m2 * L2 * cd, a21 = m2 * L1 * cd, a22 = m2 * L2;
  const b1 = -m2 * L2 * w2 * w2 * sd - M * G * Math.sin(t1) - M * a * Math.cos(t1) - M * L1 * k1 * w1;
  const b2 = m2 * L1 * w1 * w1 * sd - m2 * G * Math.sin(t2) - m2 * a * Math.cos(t2) - m2 * L2 * k2 * (w2 - w1);
  const det = a11 * a22 - a12 * a21;
  out[0] = w1; out[1] = (b1 * a22 - a12 * b2) / det; out[2] = w2; out[3] = (a11 * b2 - a21 * b1) / det;
}
const _k = [new Float64Array(4), new Float64Array(4), new Float64Array(4), new Float64Array(4)], _s = new Float64Array(4);
function rk4(st, a, c, h) {
  deriv(st, a, c, _k[0]);
  for (let i = 0; i < 4; i++) _s[i] = st[i] + h / 2 * _k[0][i]; deriv(_s, a, c, _k[1]);
  for (let i = 0; i < 4; i++) _s[i] = st[i] + h / 2 * _k[1][i]; deriv(_s, a, c, _k[2]);
  for (let i = 0; i < 4; i++) _s[i] = st[i] + h * _k[2][i]; deriv(_s, a, c, _k[3]);
  for (let i = 0; i < 4; i++) st[i] += h / 6 * (_k[0][i] + 2 * _k[1][i] + 2 * _k[2][i] + _k[3][i]);
}
const offset = (st, c) => c.L1 * Math.sin(st[0]) + c.L2 * Math.sin(st[2]); // load vs trolley, m

// A move plan: trapezoid at speed v, optionally ZV-shaped for a pendulum of length Lt.
// ZV shaper (Singer & Seering): two impulses 1/(1+K), K/(1+K) at 0 and pi/wd; with zeta assumed 0, 0.5 and 0.5 half a period apart.
function plan(vFrac, Lt) {
  const v = vFrac * CFG.VMAX, A = vFrac * CFG.A, ta = v / A, T = CFG.D / v + ta;
  const dz = Lt ? Math.PI / Math.sqrt(G / Lt) : 0, amps = Lt ? [[0, 0.5], [dz, 0.5]] : [[0, 1]];
  const aRef = t => t < 0 ? 0 : t < ta ? A : t < T - ta ? 0 : t < T ? -A : 0;
  const vRef = t => t <= 0 ? 0 : t < ta ? A * t : t < T - ta ? v : t < T ? A * (T - t) : 0;
  return { v, a: A, vFrac, Lt, T: T + dz, acc: t => amps.reduce((s, [d, w]) => s + w * aRef(t - d), 0), vel: t => amps.reduce((s, [d, w]) => s + w * vRef(t - d), 0) };
}

// Run one whole move offline: travel, then wait until the load settles. Returns what the log records.
function simulateMove(c, p, h = 1 / 100) {
  const st = new Float64Array(4); let t = 0;
  for (; t < p.T; t += h) rk4(st, p.acc(t), c, h);
  let amp = 0, inTol = 0, s = 0;
  for (; s < CFG.SETTLE_CAP; s += h) {
    rk4(st, 0, c, h); const o = Math.abs(offset(st, c)); amp = Math.max(amp, o);
    inTol = o < CFG.TOL ? inTol + h : 0; if (inTol >= CFG.HOLD) break;
  }
  return { amp, settle: Math.min(s, CFG.SETTLE_CAP), hit: amp > CFG.CLEAR };
}

// Conditions: L1 hoist rope (m), L2 sling (m), m1 hook (kg), m2 payload (kg)
function condInfo(c) { const [w1] = modes(c.L1, c.L2, c.m2 / c.m1); return { period: 2 * Math.PI / w1 }; }

// ---- Stand-in for TabPFN: kernel-weighted in-context predictor over the log (no training step). ----
// Features (summaries, not raw lengths): measured lift-off sway period (s, may be NaN), ln(payload/hook mass),
// speed fraction, and the shaper's tuning error = assumed period / measured period - 1 (NaN if no measurement).
const FSCALE = [1.0, 0.7, 0.12, 0.08];
const NF = FSCALE.length;
const shaperPeriod = (L1, tune) => 2 * Math.PI * Math.sqrt((L1 + tune) / G);
function rowFeat(c, meas, vFrac, tune) { return [meas, Math.log(c.m2 / c.m1), vFrac, Number.isNaN(meas) ? NaN : shaperPeriod(c.L1, tune) / meas - 1]; }
function predict(ctx, x) {
  // weak prior worth W0 of a row: faster = riskier, wide spread
  const pm = 0.1 + 0.4 * x[2], ps = 0.3, sm = 6 + 14 * x[2], ss = 10;
  const W0 = 0.2; let sw = W0, sa = W0 * pm, sS = W0 * sm; const ws = [];
  for (const r of ctx) {
    let d2 = 0;
    for (let j = 0; j < NF; j++) { if (Number.isNaN(x[j]) || Number.isNaN(r.x[j])) { d2 += 1; continue; } const z = (x[j] - r.x[j]) / FSCALE[j]; d2 += z * z; }
    const w = Math.exp(-d2 / 2); if (w > 1e-4) { ws.push([w, r]); sw += w; sa += w * r.amp; sS += w * r.settle; }
  }
  const ma = sa / sw, ms = sS / sw, neff = sw - W0;
  let vA = W0 * ((pm - ma) ** 2 + ps * ps), vS = W0 * ((sm - ms) ** 2 + ss * ss);
  for (const [w, r] of ws) { vA += w * (r.amp - ma) ** 2; vS += w * (r.settle - ms) ** 2; }
  const sdA = Math.sqrt(vA / sw + (ps / (1 + neff)) ** 2), sdS = Math.sqrt(vS / sw + (ss / (1 + neff)) ** 2);
  return { amp: ma, sdA, settle: ms, sdS, neff, risk: 1 - Phi((CFG.CLEAR - ma) / sdA) };
}
// Decide: score every (speed, tuning) candidate in one batch; fastest expected cycle under the risk limit.
function decide(ctx, c, meas, rng) {
  const cands = [];
  for (const v of CFG.SPEEDS) for (const tu of CFG.TUNES) {
    const p = plan(v, c.L1 + tu), pr = predict(ctx, rowFeat(c, meas, v, tu));
    const lo = Math.max(0, pr.amp - 1.645 * pr.sdA), hi = pr.amp + 1.645 * pr.sdA;
    cands.push({ v, tu, moveT: p.T, ...pr, lo, hi, total: p.T + pr.settle + pr.risk * CFG.HIT_PENALTY });
  }
  let ok = cands.filter(k => k.risk <= CFG.RISK), mode = 'safe';
  if (!ok.length) { ok = [cands.filter(k => k.v === CFG.SPEEDS[0]).sort((a, b) => a.risk - b.risk)[0]]; mode = 'fallback'; }
  let best = ok.reduce((a, b) => b.total < a.total ? b : a);
  // guarded exploration: now and then try a cheaper candidate whose risk is still small
  if (rng() < 0.2) { const ex = cands.filter(k => k.risk <= 0.05 && k.total < best.total); if (ex.length) { best = ex.reduce((a, b) => b.total < a.total ? b : a); mode = 'explore'; } }
  return { best, cands, mode };
}

function makeJobs(seed) {
  const r = mulberry(seed), J = [];
  const P = [
    () => ({ L1: 4 + 0.6 * (r() - 0.5), L2: 0.5, m1: 250, m2: 2500 + 800 * (r() - 0.5) }),        // coils, short sling
    () => ({ L1: 7 + 0.8 * (r() - 0.5), L2: 0.5, m1: 250, m2: 2500 + 800 * (r() - 0.5) }),        // high lift
    () => ({ L1: 4 + 0.6 * (r() - 0.5), L2: 2.6 + 0.4 * (r() - 0.5), m1: 250, m2: 400 + 200 * (r() - 0.5) }), // pallets, long slings
    () => ({ L1: 3 + 5 * r(), L2: 0.3 + 2.8 * r(), m1: 250, m2: 300 + 3200 * r() }),            // mixed shift
  ];
  for (let p = 0; p < 4; p++) { J.push([]); for (let i = 0; i < 400; i++) { const c = P[p](); c.meas = r() < 0.15 ? NaN : condInfo(c).period * (1 + 0.03 * gauss(r)); J[p].push(c); } }
  return J;
}
function commissioning(seed, n = 90) { // pre-existing logs: phase-0-like jobs at random speeds and tunings
  const r = mulberry(seed), rows = [];
  for (let i = 0; i < n; i++) {
    const c = { L1: 3.2 + 1.6 * r(), L2: 0.5, m1: 250, m2: 2000 + 1000 * r() }; const meas = condInfo(c).period * (1 + 0.03 * gauss(r));
    const v = CFG.SPEEDS[Math.floor(r() * CFG.SPEEDS.length)], tu = CFG.TUNES[Math.floor(r() * CFG.TUNES.length)];
    const o = simulateMove(c, plan(v, c.L1 + tu)); rows.push({ x: rowFeat(c, meas, v, tu), ...o, src: 'fleet' });
  }
  return rows;
}
// ===== /CORE =====
if (typeof module !== 'undefined') module.exports = { CFG, modes, plan, simulateMove, condInfo, predict, decide, makeJobs, commissioning, rowFeat, mulberry, offset, rk4 };
