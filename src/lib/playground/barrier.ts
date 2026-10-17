// Log-barrier central path for a two-variable linear program: minimize c·x subject to a_i·x ≤ b_i.
// Pure functions (no DOM) so tests can check stationarity and the exact duality gap m/t.
export type Vec = [number, number];
export interface Polytope { A: Vec[]; b: number[]; vertices: Vec[] }

/** Convex polygon from counter-clockwise vertices: one unit-normal constraint per edge. */
export function polygon(vertices: Vec[]): Polytope {
  const A: Vec[] = [], b: number[] = [];
  vertices.forEach((v, i) => {
    const w = vertices[(i + 1) % vertices.length], dx = w[0] - v[0], dy = w[1] - v[1], len = Math.hypot(dx, dy);
    const n: Vec = [dy / len, -dx / len];
    A.push(n); b.push(n[0] * v[0] + n[1] * v[1]);
  });
  return { A, b, vertices };
}
export const slacks = (p: Polytope, x: Vec) => p.A.map((a, i) => p.b[i] - a[0] * x[0] - a[1] * x[1]);
const barrierValue = (p: Polytope, c: Vec, t: number, x: Vec) => {
  const s = slacks(p, x);
  return s.some(v => v <= 0) ? Infinity : t * (c[0] * x[0] + c[1] * x[1]) - s.reduce((acc, v) => acc + Math.log(v), 0);
};

/** Minimizer of t·c·x − Σ log(b_i − a_i·x), by damped Newton from a strictly feasible start. */
export function centralPoint(p: Polytope, c: Vec, t: number, start: Vec): { x: Vec; iterations: number; gradientNorm: number } {
  let x: Vec = [...start], it = 0, gnorm = Infinity;
  for (; it < 100; it++) {
    const s = slacks(p, x);
    let g0 = t * c[0], g1 = t * c[1], h00 = 0, h01 = 0, h11 = 0;
    p.A.forEach((a, i) => { g0 += a[0] / s[i]; g1 += a[1] / s[i]; const w = 1 / (s[i] * s[i]); h00 += w * a[0] * a[0]; h01 += w * a[0] * a[1]; h11 += w * a[1] * a[1]; });
    gnorm = Math.hypot(g0, g1);
    if (gnorm / Math.max(t, 1e-3) < 1e-11) break; // dual residual |c + Σ λ_i a_i| = |gradient| / t
    const det = h00 * h11 - h01 * h01, d0 = -(h11 * g0 - h01 * g1) / det, d1 = -(-h01 * g0 + h00 * g1) / det;
    const decrement = -(g0 * d0 + g1 * d1);
    if (decrement / 2 < 1e-22) break;
    let step = 1; const f0 = barrierValue(p, c, t, x);
    while (step > 1e-12) {
      const y: Vec = [x[0] + step * d0, x[1] + step * d1];
      if (barrierValue(p, c, t, y) <= f0 - 0.25 * step * decrement) break;
      step /= 2;
    }
    x = [x[0] + step * d0, x[1] + step * d1];
  }
  return { x, iterations: it, gradientNorm: gnorm };
}
/** The point that minimizes the barrier alone (t = 0 limit): the analytic center. */
export function analyticCenter(p: Polytope, start: Vec): Vec {
  return centralPoint(p, [0, 0], 1, start).x;
}
/** Optimal vertex and value of the linear program, by enumerating vertices. */
export function optimum(p: Polytope, c: Vec): { vertex: Vec; value: number } {
  let best = p.vertices[0];
  for (const v of p.vertices) if (c[0] * v[0] + c[1] * v[1] < c[0] * best[0] + c[1] * best[1]) best = v;
  return { vertex: best, value: c[0] * best[0] + c[1] * best[1] };
}
/** Everything the lab shows for one t: the central point, the dual point λ_i = 1/(t s_i), and the gaps. */
export function barrierState(p: Polytope, c: Vec, t: number, start: Vec) {
  const { x, gradientNorm } = centralPoint(p, c, t, start), s = slacks(p, x);
  const lambda = s.map(v => 1 / (t * v)), primal = c[0] * x[0] + c[1] * x[1];
  const dual = -lambda.reduce((acc, l, i) => acc + l * p.b[i], 0), { value } = optimum(p, c);
  return { x, slack: s, lambda, primal, dual, gap: primal - dual, suboptimality: primal - value, bound: p.A.length / t, gradientNorm, minSlack: Math.min(...s) };
}
/** Central path samples for log10 t from `lo` to `hi`, warm-started along increasing t. */
export function centralPath(p: Polytope, c: Vec, lo: number, hi: number, count: number, start: Vec): Vec[] {
  const out: Vec[] = []; let x = start;
  for (let k = 0; k < count; k++) { x = centralPoint(p, c, 10 ** (lo + (hi - lo) * k / (count - 1)), x).x; out.push(x); }
  return out;
}
