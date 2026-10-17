import type { Lang } from '../../../i18n/ui';
import type { LabDef } from './types.ts';
import { L, pick, fmt, num } from '../types.ts';
import { plotColors, startPlot, label, line } from './plot2d.ts';
import mlpData from './data/mlp-data.json' with { type: 'json' };
import { normalField } from '../../imagekit/rng.ts';

/**
 * A tiny multilayer perceptron trained live in the browser: 2 inputs, m tanh hidden units, 1 sigmoid output, binary
 * cross-entropy, full-batch gradient descent with a fixed learning rate. Same notation as ML 02:
 *   z1 = W1 x + b1,  a1 = tanh(z1),  z2 = W2 a1 + b2,  p = sigmoid(z2),  L = mean of -[y log p + (1 - y) log(1 - p)].
 * Parameters are packed in one Float64Array: [ W1 (m x 2, row-major) | b1 (m) | W2 (m) | b2 (1) ], 4m + 1 numbers.
 * Everything here is pure (no DOM), so Node tests can call it.
 */

// ------------------------------------------------------------------------------------------------ data

export type SetKey = 'moons' | 'circles';
export const SET_KEYS: SetKey[] = ['moons', 'circles'];
export interface Points { n: number; x: Float64Array; y: Float64Array }
function points(p: { x: number[][]; y: number[] }): Points {
  const n = p.y.length, x = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) { x[2 * i] = p.x[i][0]; x[2 * i + 1] = p.x[i][1]; }
  return { n, x, y: Float64Array.from(p.y) };
}
/** scikit-learn make_moons (noise 0.2) and make_circles (noise 0.1, factor 0.5): 200 training and 200 held-out test points each. */
export const SETS: Record<SetKey, { train: Points; test: Points }> = {
  moons: { train: points(mlpData.moons.train), test: points(mlpData.moons.test) },
  circles: { train: points(mlpData.circles.train), test: points(mlpData.circles.test) },
};

// ------------------------------------------------------------------------------------------------ network

/** Seed of the initial weights (the kit's mulberry32 + Box-Muller normal field), the same for every width and learning rate. */
export const SEED = 5;
export const H_MAX = 8;
export const MAX_STEPS = 2000;
export const paramCount = (m: number) => 4 * m + 1;

/**
 * Xavier (Glorot) normal initialisation, the one ML 02 pairs with tanh: Var(W) = 2 / (n_in + n_out), biases zero.
 * Unit j always uses the same raw normal numbers, so adding a hidden unit never changes the others' starting directions.
 */
export function initParams(m: number, seed = SEED): Float64Array {
  const z = normalField(3 * H_MAX, seed), p = new Float64Array(paramCount(m));
  const s1 = Math.sqrt(2 / (2 + m)), s2 = Math.sqrt(2 / (m + 1));
  for (let j = 0; j < m; j++) {
    p[2 * j] = z[2 * j] * s1; p[2 * j + 1] = z[2 * j + 1] * s1;
    p[3 * m + j] = z[2 * H_MAX + j] * s2;
  }
  return p;
}

/** tanh through exp: about twice as fast as Math.tanh in V8 and equal to it within 1e-15, which matters at 2000 steps on every slider move. */
const tanh = (x: number) => 1 - 2 / (Math.exp(2 * x) + 1);

/** Logit z2 for every point. */
export function logits(p: Float64Array, m: number, pts: Points, out = new Float64Array(pts.n)): Float64Array {
  const b2 = p[4 * m];
  for (let i = 0; i < pts.n; i++) {
    const x0 = pts.x[2 * i], x1 = pts.x[2 * i + 1];
    let z2 = b2;
    for (let j = 0; j < m; j++) z2 += p[3 * m + j] * tanh(p[2 * j] * x0 + p[2 * j + 1] * x1 + p[2 * m + j]);
    out[i] = z2;
  }
  return out;
}
/** Binary cross-entropy from a logit, stable for large |z|: max(z, 0) - z y + log(1 + exp(-|z|)). */
export const bce = (z: number, y: number) => Math.max(z, 0) - z * y + Math.log1p(Math.exp(-Math.abs(z)));
export const sigmoid = (z: number) => z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));

export interface Score { loss: number; accuracy: number }
/** Mean loss and accuracy on a point set (forward pass only). */
export function score(p: Float64Array, m: number, pts: Points): Score {
  const b2 = p[4 * m], n = pts.n;
  let loss = 0, right = 0;
  for (let i = 0; i < n; i++) {
    const x0 = pts.x[2 * i], x1 = pts.x[2 * i + 1], y = pts.y[i];
    let z = b2;
    for (let j = 0; j < m; j++) z += p[3 * m + j] * tanh(p[2 * j] * x0 + p[2 * j + 1] * x1 + p[2 * m + j]);
    loss += bce(z, y); if ((z > 0) === (y === 1)) right++;
  }
  return { loss: loss / n, accuracy: right / n };
}

/**
 * Mean loss and (optionally) its gradient by backpropagation, in the notation of ML 02:
 * delta2 = p - y, dW2 = delta2 a1, delta1 = (W2^T delta2) * (1 - a1^2), dW1 = delta1 x^T, with the mean over the points.
 */
export function lossAndGrad(p: Float64Array, m: number, pts: Points, grad: Float64Array | null, scratch: Float64Array = new Float64Array(m)): Score {
  if (grad) grad.fill(0);
  const a = scratch, n = pts.n, b2 = p[4 * m];
  let loss = 0, right = 0;
  for (let i = 0; i < n; i++) {
    const x0 = pts.x[2 * i], x1 = pts.x[2 * i + 1], y = pts.y[i];
    let z2 = b2;
    for (let j = 0; j < m; j++) { const t = tanh(p[2 * j] * x0 + p[2 * j + 1] * x1 + p[2 * m + j]); a[j] = t; z2 += p[3 * m + j] * t; }
    const e = Math.exp(-Math.abs(z2));
    loss += Math.max(z2, 0) - z2 * y + Math.log1p(e); if ((z2 > 0) === (y === 1)) right++;
    if (grad) {
      const d2 = ((z2 >= 0 ? 1 : e) / (1 + e) - y) / n;   // (sigmoid(z2) - y) / n, from the exp already computed
      grad[4 * m] += d2;
      for (let j = 0; j < m; j++) {
        const d1 = d2 * p[3 * m + j] * (1 - a[j] * a[j]);
        grad[3 * m + j] += d2 * a[j];
        grad[2 * j] += d1 * x0; grad[2 * j + 1] += d1 * x1; grad[2 * m + j] += d1;
      }
    }
  }
  return { loss: loss / n, accuracy: right / n };
}

// ------------------------------------------------------------------------------------------------ training trajectory

/** The held-out loss curve is sampled every TEST_STRIDE updates (the training loss comes free with the gradient, every update). */
export const TEST_STRIDE = 4;
/**
 * Gradient descent from the seeded initial weights. Index k describes the network after k updates (k = 0 is the
 * untrained network), so `trainLoss[1]` is the loss after the first step. The trajectory is extended on demand, never
 * recomputed: moving the "steps" slider only reads it. `testLoss[i]` belongs to update i * TEST_STRIDE.
 */
export interface Trajectory {
  set: SetKey; m: number; lr: number;
  /** Most updates this trajectory can hold (MAX_STEPS in the lab; tests build longer ones). */
  capacity: number;
  /** Number of updates performed so far; the training arrays are valid for indices 0..done. */
  done: number;
  params: Float64Array;
  trainLoss: Float64Array; trainAcc: Float64Array; testLoss: Float64Array;
  cur: Float64Array; grad: Float64Array; scratch: Float64Array;
}
export function newTrajectory(set: SetKey, m: number, lr: number, seed = SEED, capacity = MAX_STEPS): Trajectory {
  const P = paramCount(m), n = capacity + 1;
  const t: Trajectory = {
    set, m, lr, capacity, done: 0, params: new Float64Array(n * P),
    trainLoss: new Float64Array(n), trainAcc: new Float64Array(n), testLoss: new Float64Array(Math.floor(capacity / TEST_STRIDE) + 1),
    cur: initParams(m, seed), grad: new Float64Array(P), scratch: new Float64Array(m),
  };
  t.params.set(t.cur, 0);
  t.testLoss[0] = score(t.cur, m, SETS[set].test).loss;
  const a = lossAndGrad(t.cur, m, SETS[set].train, null, t.scratch);
  t.trainLoss[0] = a.loss; t.trainAcc[0] = a.accuracy;
  return t;
}
/** Run gradient descent until `steps` updates have been performed (no-op if already there). */
export function extend(t: Trajectory, steps: number): Trajectory {
  const target = Math.min(t.capacity, Math.max(0, Math.round(steps))), { train, test } = SETS[t.set], P = t.cur.length;
  if (t.done >= target) return t;
  const measure = (grad: Float64Array | null) => {
    // One pass over the training set gives the loss and accuracy at the current weights (and the gradient, when asked for).
    const a = lossAndGrad(t.cur, t.m, train, grad, t.scratch);
    t.trainLoss[t.done] = a.loss; t.trainAcc[t.done] = a.accuracy;
    if (t.done % TEST_STRIDE === 0) t.testLoss[t.done / TEST_STRIDE] = score(t.cur, t.m, test).loss;
  };
  while (t.done < target) {
    measure(t.grad);
    for (let i = 0; i < P; i++) t.cur[i] -= t.lr * t.grad[i];
    t.done++;
    t.params.set(t.cur, t.done * P);
  }
  measure(null);
  return t;
}
/** Held-out score of the network after k updates, computed from the stored parameters. */
export const testScore = (t: Trajectory, k: number): Score => score(paramsAt(t, k), t.m, SETS[t.set].test);
/** Parameters after k updates (a view into the stored trajectory). */
export const paramsAt = (t: Trajectory, k: number) => t.params.subarray(k * paramCount(t.m), (k + 1) * paramCount(t.m));
/** Training-loss rises within the first k updates. */
export function risesUpTo(t: Trajectory, k: number): number {
  let r = 0;
  for (let i = 1; i <= k; i++) if (t.trainLoss[i] > t.trainLoss[i - 1]) r++;
  return r;
}

const cache = new Map<string, Trajectory>();
/** Cached trajectory for (set, m, lr), extended to at least `steps` updates. At most 12 trajectories are kept. */
export function trajectory(set: SetKey, m: number, lr: number, steps: number): Trajectory {
  const key = `${set}|${m}|${lr}`;
  let t = cache.get(key);
  if (!t) {
    t = newTrajectory(set, m, lr);
    if (cache.size >= 12) cache.delete(cache.keys().next().value as string);
    cache.set(key, t);
  } else { cache.delete(key); cache.set(key, t); }
  return extend(t, steps);
}

// ------------------------------------------------------------------------------------------------ what a straight line can do

/**
 * The most points any straight line can classify correctly (one class on each side). With one hidden unit the network's
 * decision boundary is such a line, so this is a hard ceiling for m = 1, however long it trains.
 * Exact: an optimal line can be moved until it touches two points; those two points can then be put on either side
 * independently (rotate or shift a hair), so the best count is max over point pairs and orientations of
 * (others classified correctly) + 2. Assumes no three points are collinear, which `collinearTriples` lets a test confirm.
 */
export function bestLineCorrect(pts: Points): number {
  const n = pts.n, x = pts.x, y = pts.y;
  let best = 0;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const nx = -(x[2 * j + 1] - x[2 * i + 1]), ny = x[2 * j] - x[2 * i];
    let c = 0, others = 0;
    for (let k = 0; k < n; k++) {
      if (k === i || k === j) continue;
      const s = nx * (x[2 * k] - x[2 * i]) + ny * (x[2 * k + 1] - x[2 * i + 1]);
      others++;
      if ((s > 0) === (y[k] === 1)) c++;
    }
    best = Math.max(best, c + 2, others - c + 2);
  }
  return best;
}
/** Number of point triples that are exactly collinear within 1e-12 (the exact line search above needs none). */
export function collinearTriples(pts: Points): number {
  let c = 0;
  for (let i = 0; i < pts.n; i++) for (let j = i + 1; j < pts.n; j++) for (let k = j + 1; k < pts.n; k++) {
    const a = (pts.x[2 * j] - pts.x[2 * i]) * (pts.x[2 * k + 1] - pts.x[2 * i + 1]) - (pts.x[2 * j + 1] - pts.x[2 * i + 1]) * (pts.x[2 * k] - pts.x[2 * i]);
    if (Math.abs(a) < 1e-12) c++;
  }
  return c;
}

// ------------------------------------------------------------------------------------------------ decision map

export interface Box { x0: number; x1: number; y0: number; y1: number }
/** Logits of the network on a w x h grid of cell centres covering `box` (row 0 is the top, y1). */
export function logitGrid(p: Float64Array, m: number, box: Box, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h);
  const b2 = p[4 * m];
  for (let r = 0; r < h; r++) {
    const y = box.y1 - (r + 0.5) * (box.y1 - box.y0) / h;
    for (let c = 0; c < w; c++) {
      const x = box.x0 + (c + 0.5) * (box.x1 - box.x0) / w;
      let z = b2;
      for (let j = 0; j < m; j++) z += p[3 * m + j] * tanh(p[2 * j] * x + p[2 * j + 1] * y + p[2 * m + j]);
      out[r * w + c] = z;
    }
  }
  return out;
}

// ------------------------------------------------------------------------------------------------ text

export const LEARNING_RATES = [0.03, 0.1, 0.3, 1, 3, 10, 30];
export const DEFAULTS = { m: 3, lr: 1, steps: 1000 };
/** Share of the steps at which the training loss went up that we call "overshooting". */
export const OVERSHOOT = 0.05;
/** Verified facts quoted in the lab text (scripts/tests/lab-mlp.test.ts asserts every one of them). */
export const BEST_LINE = { moons: { train: 175 }, circles: { train: 133 } };

export interface SetView {
  trainAcc: number; testAcc: number; trainLoss: number; testLoss: number; rises: number; peak: number;
}
export function viewOf(t: Trajectory, k: number): SetView {
  const test = testScore(t, k);
  let peak = 0;
  for (let i = 0; i <= k; i++) peak = Math.max(peak, t.trainLoss[i]);
  return { trainAcc: t.trainAcc[k], testAcc: test.accuracy, trainLoss: t.trainLoss[k], testLoss: test.loss, rises: risesUpTo(t, k), peak };
}
const pct = (a: number) => `${(100 * a).toFixed(1)}%`;

/** Plain-language reading of the current state (pure, so tests can check it). */
export function describe(lang: Lang, m: number, lr: number, steps: number, moons: SetView, circles: SetView): string {
  if (steps === 0) return pick(lang,
    `Step 0: the untrained network with its random starting weights (m = ${m}). Moons: ${pct(moons.trainAcc)} right on the training points, circles: ${pct(circles.trainAcc)}. Move the steps slider to train.`,
    `第 0 步：還沒訓練的網路，只有隨機的起始權重（m = ${m}）。月牙形的訓練點答對 ${pct(moons.trainAcc)}，圓環是 ${pct(circles.trainAcc)}。拖曳步數滑桿開始訓練。`);
  const head = pick(lang,
    `After ${steps} steps with m = ${m} and η = ${lr}: the moons are ${pct(moons.trainAcc)} right on the training points and ${pct(moons.testAcc)} on the held-out test points; the circles ${pct(circles.trainAcc)} and ${pct(circles.testAcc)}.`,
    `m = ${m}、η = ${lr}，訓練 ${steps} 步後：月牙形的訓練點答對 ${pct(moons.trainAcc)}、測試點 ${pct(moons.testAcc)}；圓環是 ${pct(circles.trainAcc)} 與 ${pct(circles.testAcc)}。`);
  const hints: string[] = [];
  if (m === 1) hints.push(pick(lang,
    `With one hidden unit the boundary is a straight line, however long it trains: at best ${BEST_LINE.moons.train} of the ${SETS.moons.train.n} training moons and ${BEST_LINE.circles.train} of the ${SETS.circles.train.n} training circle points are right.`,
    `只有一個隱藏單元時，不管訓練多久，邊界都是一條直線：在 ${SETS.moons.train.n} 個月牙形訓練點中最多答對 ${BEST_LINE.moons.train} 個，在 ${SETS.circles.train.n} 個圓環訓練點中最多答對 ${BEST_LINE.circles.train} 個。`));
  if (m === 2) hints.push(pick(lang,
    'Two hidden units give one bent curve. It can bend, but it cannot close around the inner circle; try m = 3.',
    '兩個隱藏單元只能畫出一條彎曲的線，可以彎，卻無法把內圈圍起來；試試 m = 3。'));
  if (steps >= 20 && Math.max(moons.rises, circles.rises) > OVERSHOOT * steps) hints.push(pick(lang,
    `The training loss went up on ${moons.rises} of the ${steps} steps for the moons and on ${circles.rises} for the circles: the steps overshoot the valley, so η is too large for this network.`,
    `訓練損失在月牙形的 ${steps} 步中有 ${moons.rises} 步上升、在圓環有 ${circles.rises} 步上升：每一步都跨過谷底，對這個網路來說 η 太大了。`));
  if (!hints.length) hints.push(Math.min(moons.trainAcc, circles.trainAcc) >= 0.95
    ? pick(lang, 'Both curved boundaries are learned, and the test points agree with the training points.', '兩條彎曲的邊界都學會了，測試點的表現與訓練點相近。')
    : pick(lang, 'The accuracy is still low. With this many hidden units that means the network has not finished: more steps or a larger η may help.', '準確率還很低。隱藏單元有這麼多，代表網路還沒訓練完：增加步數或調大 η 也許有幫助。'));
  const hint = hints.join(' ');
  return `${head} ${hint}`;
}

// ------------------------------------------------------------------------------------------------ drawing (DOM)

const W = 400, H = 300;
const PX = 38, PY = 8, PW = 352, PH = 264;          // plot area of a map (aspect 4:3, like the boxes below)
const GW = 176, GH = 132;                            // decision-map grid (2 px cells)
const BOXES: Record<SetKey, Box> = {
  moons: { x0: -1.6, x1: 2.6, y0: 0.2 - 2.1 * 0.75, y1: 0.2 + 2.1 * 0.75 },
  circles: { x0: -1.6, x1: 1.6, y0: -1.2, y1: 1.2 },
};
const CLASS_RGB = [[36, 102, 194], [195, 91, 37]];
const CLASS_CSS = CLASS_RGB.map(c => `rgb(${c[0]},${c[1]},${c[2]})`);

function shape(ctx: CanvasRenderingContext2D, cls: number, x: number, y: number, r: number) {
  ctx.beginPath();
  if (cls === 0) ctx.arc(x, y, r, 0, 2 * Math.PI);
  else { ctx.moveTo(x, y - r - 1); ctx.lineTo(x + r + 1, y + r); ctx.lineTo(x - r - 1, y + r); ctx.closePath(); }
}
function drawMap(canvas: HTMLCanvasElement, off: HTMLCanvasElement, set: SetKey, p: Float64Array, m: number) {
  const c = plotColors(canvas), ctx = startPlot(canvas, W, H, c), box = BOXES[set];
  const z = logitGrid(p, m, box, GW, GH), cc = off.getContext('2d')!, im = cc.createImageData(GW, GH);
  for (let i = 0; i < GW * GH; i++) {
    const strength = 0.62 * Math.abs(Math.tanh(z[i] / 2)), col = CLASS_RGB[z[i] > 0 ? 1 : 0];   // |2 sigmoid(z) - 1| = |tanh(z / 2)|
    for (let k = 0; k < 3; k++) im.data[4 * i + k] = Math.round(255 + (col[k] - 255) * strength);
    im.data[4 * i + 3] = 255;
  }
  cc.putImageData(im, 0, 0);
  ctx.imageSmoothingEnabled = true; ctx.drawImage(off, PX, PY, PW, PH);
  // The 0.5 probability contour: cells whose sign differs from the cell to the right or below.
  ctx.fillStyle = '#111111';
  const cw = PW / GW, ch = PH / GH;
  for (let r = 0; r < GH - 1; r++) for (let q = 0; q < GW - 1; q++) {
    const s = z[r * GW + q] > 0;
    if (s !== (z[r * GW + q + 1] > 0) || s !== (z[(r + 1) * GW + q] > 0)) ctx.fillRect(PX + q * cw + 0.5, PY + r * ch + 0.5, cw, ch);
  }
  ctx.lineWidth = 1; ctx.strokeStyle = c.muted; ctx.strokeRect(PX + 0.5, PY + 0.5, PW, PH);
  const gx = (x: number) => PX + (x - box.x0) / (box.x1 - box.x0) * PW, gy = (y: number) => PY + (box.y1 - y) / (box.y1 - box.y0) * PH;
  for (let v = Math.ceil(box.x0); v <= box.x1; v++) { line(ctx, gx(v), PY + PH, gx(v), PY + PH + 4, c.muted); label(ctx, String(v), gx(v), PY + PH + 19, { size: 13, align: 'center', color: c.muted }); }
  for (let v = Math.ceil(box.y0); v <= box.y1; v++) { line(ctx, PX - 4, gy(v), PX, gy(v), c.muted); label(ctx, String(v), PX - 7, gy(v) + 4, { size: 13, align: 'right', color: c.muted }); }
  label(ctx, 'x₁', PX + PW - 3, PY + PH + 19, { size: 13, align: 'right', color: c.ink }); label(ctx, 'x₂', PX + 5, PY + 15, { size: 13, color: '#111111', halo: 'rgba(255,255,255,0.85)' });
  const { train, test } = SETS[set], zt = logits(p, m, train), zs = logits(p, m, test);
  // Test points are hollow and drawn first, so the filled training points stay readable on top.
  for (let i = 0; i < test.n; i++) {
    const cls = test.y[i], x = gx(test.x[2 * i]), y = gy(test.x[2 * i + 1]);
    shape(ctx, cls, x, y, 3.2); ctx.fillStyle = '#ffffff'; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = CLASS_CSS[cls]; ctx.stroke();
    if ((zs[i] > 0) !== (cls === 1)) { ctx.beginPath(); ctx.arc(x, y, 7.5, 0, 2 * Math.PI); ctx.lineWidth = 1.2; ctx.setLineDash([2, 2]); ctx.strokeStyle = '#111111'; ctx.stroke(); ctx.setLineDash([]); }
  }
  for (let i = 0; i < train.n; i++) {
    const cls = train.y[i], x = gx(train.x[2 * i]), y = gy(train.x[2 * i + 1]);
    shape(ctx, cls, x, y, 3.8); ctx.fillStyle = CLASS_CSS[cls]; ctx.fill(); ctx.lineWidth = 1; ctx.strokeStyle = '#ffffff'; ctx.stroke();
    if ((zt[i] > 0) !== (cls === 1)) { ctx.beginPath(); ctx.arc(x, y, 7.5, 0, 2 * Math.PI); ctx.lineWidth = 1.8; ctx.strokeStyle = '#111111'; ctx.stroke(); }
  }
}

const Y_MAX = 1;
function drawLoss(canvas: HTMLCanvasElement, lang: Lang, t: Trajectory, step: number, v: SetView) {
  const c = plotColors(canvas), ctx = startPlot(canvas, W, H, c);
  const x0 = 48, x1 = 388, y0 = 38, y1 = 252, gx = (k: number) => x0 + k / MAX_STEPS * (x1 - x0), gy = (loss: number) => y1 - Math.min(loss, Y_MAX) / Y_MAX * (y1 - y0);
  for (const g of [0, 0.25, 0.5, 0.75, 1]) { line(ctx, x0, gy(g), x1, gy(g), c.grid, 1, g === 0 ? [] : [2, 4]); label(ctx, g.toFixed(2), x0 - 6, gy(g) + 4, { size: 13, align: 'right', color: c.muted }); }
  for (const k of [0, 500, 1000, 1500, 2000]) { line(ctx, gx(k), y1, gx(k), y1 + 4, c.muted); label(ctx, String(k), gx(k), y1 + 19, { size: 13, align: k === 0 ? 'left' : k === MAX_STEPS ? 'right' : 'center', color: c.muted }); }
  label(ctx, pick(lang, 'step', '步數'), x1, y1 + 36, { size: 13, align: 'right', color: c.ink });
  // The loss of a model that always answers 50/50 is ln 2.
  line(ctx, x0, gy(Math.LN2), x1, gy(Math.LN2), c.muted, 1.2, [6, 4]);
  label(ctx, pick(lang, 'ln 2 = 0.693: always answering 50/50', 'ln 2 = 0.693：永遠回答 50/50'), x1 - 2, gy(Math.LN2) - 5, { size: 12, align: 'right', color: c.muted });
  // Test loss (sampled every TEST_STRIDE steps, plus the exact value at the chosen step), then the training loss on top.
  ctx.lineJoin = 'round'; ctx.lineWidth = 2; ctx.strokeStyle = c.b; ctx.setLineDash([6, 3]); ctx.beginPath();
  for (let k = 0; k <= step; k += TEST_STRIDE) { const X = gx(k), Y = gy(t.testLoss[k / TEST_STRIDE]); if (k === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); }
  ctx.lineTo(gx(step), gy(v.testLoss)); ctx.stroke(); ctx.setLineDash([]);
  ctx.lineWidth = 2.4; ctx.strokeStyle = c.a; ctx.beginPath();
  for (let k = 0; k <= step; k++) { const X = gx(k), Y = gy(t.trainLoss[k]); if (k === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); }
  ctx.stroke();
  // The current step.
  line(ctx, gx(step), y0, gx(step), y1, c.ink, 1, [3, 3]);
  for (const [val, col] of [[v.testLoss, c.b], [v.trainLoss, c.a]] as const) { ctx.beginPath(); ctx.arc(gx(step), gy(val), 4.5, 0, 2 * Math.PI); ctx.fillStyle = col; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = c.bg; ctx.stroke(); }
  // Legend with the current values.
  line(ctx, x0, 14, x0 + 22, 14, c.a, 2.4); label(ctx, `${pick(lang, 'train', '訓練')} ${fmt(v.trainLoss, 3)}`, x0 + 27, 19, { size: 14, color: c.ink });
  line(ctx, x0 + 150, 14, x0 + 172, 14, c.b, 2, [6, 3]); label(ctx, `${pick(lang, 'test', '測試')} ${fmt(v.testLoss, 3)}`, x0 + 177, 19, { size: 14, color: c.ink });
  if (v.peak > Y_MAX) label(ctx, pick(lang, `peak ${fmt(v.peak, 1)}, above the chart`, `最高 ${fmt(v.peak, 1)}，超出圖外`), x1, 34, { size: 12, align: 'right', color: c.ink });
}

// ------------------------------------------------------------------------------------------------ the lab

export const lab: LabDef = {
  kind: 'image', id: 'mlp-training', track: 'ml',
  title: L('ML 02 · Can a tiny network learn a curved boundary?', 'ML 02・小小的網路學得會彎曲的邊界嗎？'),
  description: L(
    'Train a network with 2 inputs, m tanh hidden units and 1 sigmoid output on the two moons and the two circles, one gradient-descent step at a time. Watch the decision map and the training and test loss: too few hidden units, too small a learning rate and too large a learning rate each fail in their own way.',
    '在雙月形與雙圓環兩個資料集上，一步一步用梯度下降訓練一個有 2 個輸入、m 個 tanh 隱藏單元與 1 個 sigmoid 輸出的網路，觀察決策圖與訓練、測試損失。隱藏單元太少、學習率太小、學習率太大，各有各的失敗方式。'),
  challenge: L(
    'Start from the default (m = 3, η = 1, 1000 steps), where both curved boundaries are learned. (1) Drag the steps slider up from 0: after 100 steps the moons are only 85.0% right (close to one tilted straight line, the loss on a plateau near 0.30), after 300 steps 94.5%, after 500 steps 98.0%. (2) Set η to 0.3 and lower m to 2, then 1: at 1000 steps the moons stay at 86.0% and 85.0%, the circles at 77.5% and 61.5%. (3) Go back to m = 3 and try η = 0.03, 0.1 and 0.3: at 1000 steps the moons are 84.5%, 85.0% and 94.5% right. A small η is not wrong, only slow: run offline for 5000 steps, beyond the 2000-step limit of this lab, η = 0.1 reaches 98.0% on the moons. (4) Try η = 3, 10 and 30: the loss now goes up on many steps, and with the largest η the training does not settle.',
    '從預設值開始（m = 3、η = 1、1000 步），兩條彎曲的邊界都學會了。(1) 把步數滑桿從 0 往右拖：100 步時月牙形只答對 85.0%（幾乎是一條斜直線，損失停在 0.30 附近的平原），300 步時 94.5%，500 步時 98.0%。(2) 把 η 設為 0.3，再把 m 降成 2，然後 1：訓練 1000 步後，月牙形分別停在 86.0% 與 85.0%，圓環停在 77.5% 與 61.5%。(3) 回到 m = 3，試試 η = 0.03、0.1 與 0.3：1000 步時月牙形分別答對 84.5%、85.0% 與 94.5%。學習率小並不是錯，只是慢：離線跑 5000 步（超出本實驗 2000 步的上限），η = 0.1 在月牙形上可達 98.0%。(4) 試試 η = 3、10 與 30：損失在許多步上升，η 最大時訓練更是無法穩定下來。'),
  note: L(
    'Data: scikit-learn make_moons (noise 0.2, random_state 0) and make_circles (noise 0.1, factor 0.5, random_state 0), 200 training points each, plus a held-out test set of 200 points each (random_state 1) that is never used for training. Network and loss as in ML 02: z₁ = W₁x + b₁, a₁ = tanh(z₁), z₂ = W₂a₁ + b₂, p = σ(z₂), binary cross-entropy averaged over the training points; the gradient comes from backpropagation (checked against finite differences and against an independent NumPy run). Training is plain full-batch gradient descent, θ ← θ − η∇L, with a fixed η: no momentum, mini-batches or regularization. Initial weights: Xavier normal (variance 2 / (fan-in + fan-out)) with zero biases, from one fixed seed (5), so a slider never re-rolls them; unit j always gets the same random numbers. Another seed changes the details: over seeds 1 to 20 (η = 1, 2000 steps), 3 hidden units reached at least 95% training accuracy on both datasets in all 20 runs, 1 unit never did, and 2 units reached it on the moons in at most 1 of the 20 runs and on the circles never. A step is one parameter update; step 0 is the untrained network. When the loss goes up and down the iteration is chaotic, so the curves at large η can differ in their details between browsers, though not in their story. The test-loss curve is sampled every 4 steps.',
    '資料：scikit-learn 的 make_moons（noise 0.2、random_state 0）與 make_circles（noise 0.1、factor 0.5、random_state 0），各 200 個訓練點，另有各 200 個保留的測試點（random_state 1），測試點從不用於訓練。網路與損失採用 ML 02 的記號：z₁ = W₁x + b₁、a₁ = tanh(z₁)、z₂ = W₂a₁ + b₂、p = σ(z₂)，損失是訓練點上的二元交叉熵平均；梯度由反向傳播計算（已用有限差分及獨立的 NumPy 計算核對）。訓練是最單純的全批次梯度下降 θ ← θ − η∇L，η 固定，沒有動量、小批次或正則化。初始權重為 Xavier 常態分布（變異數 2 /（輸入數＋輸出數））、偏差為 0，來自固定種子（5），所以拉動滑桿不會重新抽籤；第 j 個單元永遠拿到同樣的隨機數。換種子細節會變：種子 1 到 20（η = 1、2000 步）中，3 個隱藏單元在兩個資料集上的訓練準確率 20 次都達到 95% 以上，1 個單元從未達到，2 個單元在月牙形上 20 次中至多 1 次達到、在圓環上從未達到。一步是一次參數更新，第 0 步是尚未訓練的網路。損失忽上忽下時，迭代是混沌的，所以 η 很大時的曲線細節可能因瀏覽器而異，但故事不變。測試損失曲線每 4 步取樣一次。'),
  related: ['ml/ml-02-neural-networks'],
  controls: [
    { key: 'm', label: L('Hidden units m', '隱藏單元數 m'), kind: 'range', value: DEFAULTS.m, min: 1, max: H_MAX, step: 1 },
    { key: 'lr', label: L('Learning rate η', '學習率 η'), kind: 'select', value: String(DEFAULTS.lr), options: LEARNING_RATES.map(v => ({ value: String(v), label: L(`η = ${v}`, `η = ${v}`) })) },
    { key: 'steps', label: L('Training steps (gradient-descent updates)', '訓練步數（梯度下降的更新次數）'), kind: 'range', value: DEFAULTS.steps, min: 0, max: MAX_STEPS, step: 10 },
  ],
  predict: {
    question: L('Set the hidden units to 1 and leave everything else as it is. Can the network separate the two moons?', '把隱藏單元數設為 1，其他都不動。網路能把兩個月牙形分開嗎？'),
    options: [
      { value: 'yes', label: L('Yes: trained long enough, it finds the curve and reaches 95% or more', '能：訓練夠久就找得到彎曲的邊界，準確率可達 95% 以上') },
      { value: 'line', label: L('Not well: its boundary is still a straight line, so it cannot beat 87.5%', '不太能：邊界仍是一條直線，所以不可能超過 87.5%') },
      { value: 'chance', label: L('Not at all: it stays at chance, around 50%', '完全不能：停在隨機猜測的 50% 左右') },
    ],
    answer: 'line',
    explain: L(
      'Not well. With one hidden unit the network computes a single number, tanh(w·x + b), and the output is a rising or falling function of it, so the decision boundary is a straight line however long you train. Checking every line through two of the 200 training points, the best straight line gets 175 of them right (87.5%), and a one-unit network cannot beat that. The default three hidden units bend the boundary: 97.5% of the training points and 95.5% of the held-out test points are right. On the circles the best straight line gets only 133 of 200 (66.5%).',
      '不太能。只有一個隱藏單元時，網路先算出一個數字 tanh(w·x + b)，輸出只是這個數字的遞增或遞減函數，所以不論訓練多久，決策邊界都是一條直線。檢查 200 個訓練點中所有通過兩點的直線，最好的直線也只能答對 175 個（87.5%），一個單元的網路不可能超過它。預設的三個隱藏單元把邊界彎起來：訓練點答對 97.5%，保留的測試點答對 95.5%。在圓環上，最好的直線 200 個點只能答對 133 個（66.5%）。'),
  },
  assets: {},
  paneMin: 340,
  panes: [
    { key: 'moonsMap', kind: 'canvas', label: L('Two moons: decision map at the chosen step (blue circles = class 0, orange triangles = class 1; filled = training, hollow = test; ringed = wrong; black line = 0.5 boundary)', '雙月形：所選步數的決策圖（藍圓＝類別 0，橙三角＝類別 1；實心＝訓練點，空心＝測試點；加圈＝答錯；黑線＝0.5 邊界）'), width: W, height: H },
    { key: 'moonsLoss', kind: 'canvas', label: L('Two moons: cross-entropy loss against the step (solid = training, dashed = test)', '雙月形：交叉熵損失隨步數的變化（實線＝訓練，虛線＝測試）'), width: W, height: H },
    { key: 'circlesMap', kind: 'canvas', label: L('Two circles: decision map at the chosen step (same symbols)', '雙圓環：所選步數的決策圖（符號相同）'), width: W, height: H },
    { key: 'circlesLoss', kind: 'canvas', label: L('Two circles: cross-entropy loss against the step (solid = training, dashed = test)', '雙圓環：交叉熵損失隨步數的變化（實線＝訓練，虛線＝測試）'), width: W, height: H },
  ],
  sources: [
    { title: 'scikit-learn documentation, sklearn.datasets.make_moons (the two-moons data, generated here, no download)', url: 'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html' },
    { title: 'scikit-learn documentation, sklearn.datasets.make_circles (the two-circles data)', url: 'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html' },
    { title: 'F. Pedregosa et al., Scikit-learn: Machine Learning in Python, Journal of Machine Learning Research, 12, 2825-2830, 2011', url: 'https://jmlr.org/papers/v12/pedregosa11a.html' },
    { title: 'D. E. Rumelhart, G. E. Hinton and R. J. Williams, Learning representations by back-propagating errors, Nature, 323, 533-536, 1986 (backpropagation)', url: 'https://doi.org/10.1038/323533a0' },
    { title: 'X. Glorot and Y. Bengio, Understanding the difficulty of training deep feedforward neural networks, Proceedings of AISTATS, PMLR 9, 249-256, 2010 (Xavier initialisation)', url: 'https://proceedings.mlr.press/v9/glorot10a.html' },
    { title: 'I. Goodfellow, Y. Bengio and A. Courville, Deep Learning, MIT Press, 2016 (multilayer perceptrons, cross-entropy, gradient descent)', url: 'https://www.deeplearningbook.org/' },
  ],
  create({ lang }) {
    const offscreen = { moons: document.createElement('canvas'), circles: document.createElement('canvas') };
    for (const k of SET_KEYS) { offscreen[k].width = GW; offscreen[k].height = GH; }
    return {
      render(state, canvases) {
        const m = Math.max(1, Math.min(H_MAX, Math.round(num(state, 'm')))), lr = num(state, 'lr'), steps = Math.max(0, Math.min(MAX_STEPS, Math.round(num(state, 'steps'))));
        const views = {} as Record<SetKey, SetView>;
        for (const set of SET_KEYS) {
          const t = trajectory(set, m, lr, steps);
          views[set] = viewOf(t, steps);
          drawMap(canvases[`${set}Map`], offscreen[set], set, paramsAt(t, steps), m);
          drawLoss(canvases[`${set}Loss`], lang, t, steps, views[set]);
        }
        const a = views.moons, b = views.circles;
        return {
          summary: describe(lang, m, lr, steps, a, b),
          metrics: [
            { label: pick(lang, 'Moons · train accuracy', '月牙形・訓練準確率'), value: pct(a.trainAcc) },
            { label: pick(lang, 'Moons · test accuracy', '月牙形・測試準確率'), value: pct(a.testAcc) },
            { label: pick(lang, 'Moons · training loss', '月牙形・訓練損失'), value: fmt(a.trainLoss, 3) },
            { label: pick(lang, 'Circles · train accuracy', '圓環・訓練準確率'), value: pct(b.trainAcc) },
            { label: pick(lang, 'Circles · test accuracy', '圓環・測試準確率'), value: pct(b.testAcc) },
            { label: pick(lang, 'Circles · training loss', '圓環・訓練損失'), value: fmt(b.trainLoss, 3) },
            { label: pick(lang, 'Steps where the loss rose (moons | circles)', '損失上升的步數（月牙形｜圓環）'), value: `${a.rises} | ${b.rises}` },
          ],
        };
      },
    };
  },
};
