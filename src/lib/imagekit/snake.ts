// Closed active contour (snake) of Kass, Witkin and Terzopoulos (1988), in the semi-implicit form of the DIP 11 chapter and of
// its figure code (scripts/figures/dip_ch11.py, `snake_matrix` and `run_snake`):
//
//   gamma (x_t - x_{t-1}) = -A x_t + f_x(x_{t-1}, y_{t-1})   =>   x_t = (A + gamma I)^-1 (gamma x_{t-1} + f_x),   same for y,
//
// where A is the symmetric circulant pentadiagonal matrix with rows [beta, -alpha-4beta, 2alpha+6beta, -alpha-4beta, beta] (unit
// point spacing), and the external force is f = kappa * grad(e) for the edge map e = |grad(G_sigma * image)| scaled to a maximum
// of 1, sampled by bilinear interpolation at the points. (A + gamma I) is inverted once per parameter set. Points are clipped to
// the image after every step. There is no resampling and no cap on the step length.
//
// Energy (for display and for checking descent): E = E_int + E_ext with
//   E_int = 1/2 sum_i [ alpha |v_{i+1} - v_i|^2 + beta |v_{i+1} - 2 v_i + v_{i-1}|^2 ],   E_ext = -kappa sum_i e(v_i).
// A x is exactly the gradient of E_int with respect to the x coordinates, so each step is a semi-implicit descent step on E.
import { gaussianGradient } from './harris.ts';

/** Edge map and the two components of its gradient, all in double precision, row-major. */
export interface EdgeField { w: number; h: number; e: Float64Array; gx: Float64Array; gy: Float64Array }

/** `numpy.gradient` of a 2-D array with unit spacing: central differences inside, one-sided differences on the border. */
export function numpyGradient(a: ArrayLike<number>, w: number, h: number): { gx: Float64Array; gy: Float64Array } {
  const gx = new Float64Array(w * h), gy = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    gx[i] = x === 0 ? a[i + 1] - a[i] : x === w - 1 ? a[i] - a[i - 1] : (a[i + 1] - a[i - 1]) / 2;
    gy[i] = y === 0 ? a[i + w] - a[i] : y === h - 1 ? a[i] - a[i - w] : (a[i + w] - a[i - w]) / 2;
  }
  return { gx, gy };
}
/** e = |grad(G_sigma * f)| (Gaussian derivatives, reflected borders) divided by its maximum, and the gradient of e. */
export function edgeField(src: Float32Array, w: number, h: number, sigma: number): EdgeField {
  const { ix, iy } = gaussianGradient(src, w, h, sigma), e = new Float64Array(w * h);
  let max = 0;
  for (let i = 0; i < e.length; i++) { e[i] = Math.hypot(ix[i], iy[i]); if (e[i] > max) max = e[i]; }
  if (max > 0) for (let i = 0; i < e.length; i++) e[i] /= max;
  return { w, h, e, ...numpyGradient(e, w, h) };
}
/** Bilinear interpolation; coordinates outside the image are clamped to its border. */
export function bilinear(a: ArrayLike<number>, w: number, h: number, x: number, y: number): number {
  const cx = x < 0 ? 0 : x > w - 1 ? w - 1 : x, cy = y < 0 ? 0 : y > h - 1 ? h - 1 : y;
  const x0 = Math.min(Math.floor(cx), w - 2), y0 = Math.min(Math.floor(cy), h - 2), fx = cx - x0, fy = cy - y0, i = y0 * w + x0;
  return (1 - fy) * ((1 - fx) * a[i] + fx * a[i + 1]) + fy * ((1 - fx) * a[i + w] + fx * a[i + w + 1]);
}

/** The points of a circle, i = 0 … n-1 at angle 2 pi i / n (numpy.linspace(0, 2 pi, n, endpoint=False)). */
export function circlePoints(cx: number, cy: number, r: number, n: number): { x: Float64Array; y: Float64Array } {
  const x = new Float64Array(n), y = new Float64Array(n);
  for (let i = 0; i < n; i++) { const t = (2 * Math.PI * i) / n; x[i] = cx + r * Math.cos(t); y[i] = cy + r * Math.sin(t); }
  return { x, y };
}

/** The circulant pentadiagonal matrix A of the chapter (unit spacing), as a dense row-major n × n array. */
export function internalMatrix(n: number, alpha: number, beta: number): Float64Array {
  if (n < 5) throw new Error('A closed snake needs at least 5 points');
  const a = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    const at = (d: number) => i * n + (((i + d) % n) + n) % n;
    a[at(0)] += 2 * alpha + 6 * beta; a[at(-1)] += -alpha - 4 * beta; a[at(1)] += -alpha - 4 * beta; a[at(-2)] += beta; a[at(2)] += beta;
  }
  return a;
}
/** Inverse of a dense square matrix by Gauss-Jordan elimination with partial pivoting (row-major, n × n). */
export function invert(m: Float64Array, n: number): Float64Array {
  const a = Float64Array.from(m), inv = new Float64Array(n * n);
  for (let i = 0; i < n; i++) inv[i * n + i] = 1;
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r * n + c]) > Math.abs(a[p * n + c])) p = r;
    if (Math.abs(a[p * n + c]) < 1e-300) throw new Error('Singular matrix');
    if (p !== c) for (let k = 0; k < n; k++) {
      let t = a[c * n + k]; a[c * n + k] = a[p * n + k]; a[p * n + k] = t;
      t = inv[c * n + k]; inv[c * n + k] = inv[p * n + k]; inv[p * n + k] = t;
    }
    const d = 1 / a[c * n + c];
    for (let k = 0; k < n; k++) { a[c * n + k] *= d; inv[c * n + k] *= d; }
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = a[r * n + c];
      if (f === 0) continue;
      for (let k = 0; k < n; k++) { a[r * n + k] -= f * a[c * n + k]; inv[r * n + k] -= f * inv[c * n + k]; }
    }
  }
  return inv;
}
const matrixCache = new Map<string, Float64Array>();
/** (A + gamma I)^-1, computed once per (n, alpha, beta, gamma) and then reused. */
export function snakeMatrix(n: number, alpha: number, beta: number, gamma: number): Float64Array {
  const key = `${n}|${alpha}|${beta}|${gamma}`, hit = matrixCache.get(key);
  if (hit) return hit;
  const a = internalMatrix(n, alpha, beta);
  for (let i = 0; i < n; i++) a[i * n + i] += gamma;
  const inv = invert(a, n);
  if (matrixCache.size > 32) matrixCache.clear();
  matrixCache.set(key, inv);
  return inv;
}

export interface SnakeEnergy { internal: number; external: number }
/** The energy defined at the top of this file for the closed contour (x_i, y_i) on the edge map `e`. */
export function snakeEnergy(field: EdgeField, x: ArrayLike<number>, y: ArrayLike<number>, alpha: number, beta: number, kappa: number): SnakeEnergy {
  const n = x.length;
  let stretch = 0, bend = 0, edge = 0;
  for (let i = 0; i < n; i++) {
    const p = (i + 1) % n, m = (i + n - 1) % n;
    stretch += (x[p] - x[i]) ** 2 + (y[p] - y[i]) ** 2;
    bend += (x[p] - 2 * x[i] + x[m]) ** 2 + (y[p] - 2 * y[i] + y[m]) ** 2;
    edge += bilinear(field.e, field.w, field.h, x[i], y[i]);
  }
  return { internal: 0.5 * (alpha * stretch + beta * bend), external: -kappa * edge };
}

export interface SnakeOptions {
  alpha: number; beta: number; gamma: number; kappa: number;
  /** Stop when no point moves farther than this many pixels in one step. */
  tol: number;
  maxIter: number;
  /** Called with iteration 0 (the start) and after every step, with the current points (do not keep the arrays). */
  observe?: (iteration: number, x: Float64Array, y: Float64Array) => void;
}
export interface SnakeResult {
  x: Float64Array; y: Float64Array;
  /** Number of steps taken. */
  iterations: number;
  /** True when the last step moved every point by less than `tol`; false when `maxIter` ran out first. */
  converged: boolean;
  /** Largest point movement in the last step, in pixels. */
  lastMove: number;
  /** Energy after iteration 0, 1, …, `iterations` (length iterations + 1). */
  internal: Float64Array; external: Float64Array;
}
export function runSnake(field: EdgeField, x0: ArrayLike<number>, y0: ArrayLike<number>, o: SnakeOptions): SnakeResult {
  const n = x0.length, { w, h } = field, inv = snakeMatrix(n, o.alpha, o.beta, o.gamma);
  let x = Float64Array.from(x0), y = Float64Array.from(y0), xn = new Float64Array(n), yn = new Float64Array(n);
  const rx = new Float64Array(n), ry = new Float64Array(n), internal: number[] = [], external: number[] = [];
  const record = (it: number) => {
    const en = snakeEnergy(field, x, y, o.alpha, o.beta, o.kappa);
    internal.push(en.internal); external.push(en.external);
    o.observe?.(it, x, y);
  };
  record(0);
  let iterations = 0, move = Infinity;
  while (iterations < o.maxIter) {
    for (let i = 0; i < n; i++) {
      rx[i] = o.gamma * x[i] + o.kappa * bilinear(field.gx, w, h, x[i], y[i]);
      ry[i] = o.gamma * y[i] + o.kappa * bilinear(field.gy, w, h, x[i], y[i]);
    }
    move = 0;
    for (let i = 0; i < n; i++) {
      let sx = 0, sy = 0;
      for (let j = 0; j < n; j++) { const m = inv[i * n + j]; sx += m * rx[j]; sy += m * ry[j]; }
      xn[i] = Math.min(w - 1, Math.max(0, sx)); yn[i] = Math.min(h - 1, Math.max(0, sy));
      const d = Math.hypot(xn[i] - x[i], yn[i] - y[i]);
      if (d > move) move = d;
    }
    [x, xn] = [xn, x]; [y, yn] = [yn, y];
    iterations++;
    record(iterations);
    if (move < o.tol) break;
  }
  return { x, y, iterations, converged: move < o.tol, lastMove: move, internal: Float64Array.from(internal), external: Float64Array.from(external) };
}

/** Least-squares circle through points (Kasa fit): minimises the sum of (x² + y² + a x + b y + c)². */
export function fitCircle(xs: ArrayLike<number>, ys: ArrayLike<number>): { cx: number; cy: number; r: number } {
  const n = xs.length;
  // Normal equations for [x y 1] [p q s]^T = x² + y², with cx = p / 2, cy = q / 2, r² = s + cx² + cy².
  let sxx = 0, sxy = 0, sx = 0, syy = 0, sy = 0, sb = 0, sxb = 0, syb = 0;
  for (let i = 0; i < n; i++) {
    const x = xs[i], y = ys[i], b = x * x + y * y;
    sxx += x * x; sxy += x * y; sx += x; syy += y * y; sy += y; sxb += x * b; syb += y * b; sb += b;
  }
  const m = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]], r = [sxb, syb, sb];
  const det3 = (a: number[][]) => a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  const d = det3(m), col = (k: number) => det3(m.map((row, i) => row.map((v, j) => (j === k ? r[i] : v))));
  const p = col(0) / d, q = col(1) / d, s = col(2) / d, cx = p / 2, cy = q / 2;
  return { cx, cy, r: Math.sqrt(s + cx * cx + cy * cy) };
}
