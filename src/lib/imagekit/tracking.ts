// Tracking by detection, the way the DLCV 06 chapter writes it: a constant-velocity Kalman filter on [cx, cy, w, h, vx, vy],
// a cost matrix C = lambda * (1 - IoU) + (1 - lambda) * (1 - cosine of colour embeddings), the Hungarian algorithm, and identity switches.
// Pure functions on plain arrays, no DOM, so Node tests can run them against NumPy and SciPy references.
import type { Rgb } from './types.ts';

/** A box as [cx, cy, w, h]: centre and size in pixels (the chapter's convention). */
export interface Box { cx: number; cy: number; w: number; h: number }

/** Intersection over union of two boxes, with the same 1e-9 in the denominator as the chapter's `iou_matrix`. */
export function iouBox(a: Box, b: Box): number {
  const w = Math.max(0, Math.min(a.cx + a.w / 2, b.cx + b.w / 2) - Math.max(a.cx - a.w / 2, b.cx - b.w / 2));
  const h = Math.max(0, Math.min(a.cy + a.h / 2, b.cy + b.h / 2) - Math.max(a.cy - a.h / 2, b.cy - b.h / 2));
  const inter = w * h;
  return inter / (a.w * a.h + b.w * b.h - inter + 1e-9);
}

// ---------------------------------------------------------------------------------------------- Hungarian algorithm

/**
 * Minimum-cost one-to-one assignment (the Hungarian algorithm in its shortest-augmenting-path form with row and column
 * potentials, O(n^2 m) for an n x m matrix with n <= m). Returns, for every row, the column it is assigned to, or -1 when
 * there are more rows than columns and the row is left out. Same optimum as `scipy.optimize.linear_sum_assignment`; when two
 * assignments cost exactly the same, which one is returned may differ.
 */
export function hungarian(cost: number[][]): number[] {
  const rows = cost.length, cols = rows ? cost[0].length : 0;
  if (rows === 0 || cols === 0) return new Array(rows).fill(-1);
  if (rows > cols) {
    const t = Array.from({ length: cols }, (_, j) => cost.map(r => r[j]));
    const colToRow = hungarian(t), out: number[] = new Array(rows).fill(-1);
    colToRow.forEach((i, j) => { if (i >= 0) out[i] = j; });
    return out;
  }
  const n = rows, m = cols, u = new Array(n + 1).fill(0), v = new Array(m + 1).fill(0), p = new Array(m + 1).fill(0), way = new Array(m + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array(m + 1).fill(Infinity), used = new Array(m + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = Infinity, j1 = 0;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= m; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const out: number[] = new Array(n).fill(-1);
  for (let j = 1; j <= m; j++) if (p[j]) out[p[j] - 1] = j - 1;
  return out;
}
/** Total cost of an assignment returned by `hungarian`. */
export function assignmentCost(cost: number[][], rowToCol: number[]): number {
  let s = 0;
  rowToCol.forEach((j, i) => { if (j >= 0) s += cost[i][j]; });
  return s;
}

// ---------------------------------------------------------------------------------------------- Kalman filter

type Mat = number[][];
const matMul = (a: Mat, b: Mat): Mat => a.map(row => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)));
const transpose = (a: Mat): Mat => a[0].map((_, j) => a.map(r => r[j]));
const matAdd = (a: Mat, b: Mat): Mat => a.map((r, i) => r.map((v, j) => v + b[i][j]));
const eye = (n: number): Mat => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
/** Gauss-Jordan inverse with partial pivoting (matrices here are 4 x 4 and well conditioned). */
function inverse(a: Mat): Mat {
  const n = a.length, m = a.map((r, i) => [...r, ...eye(n)[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[piv][c])) piv = r;
    [m[c], m[piv]] = [m[piv], m[c]];
    const d = m[c][c];
    for (let j = 0; j < 2 * n; j++) m[c][j] /= d;
    for (let r = 0; r < n; r++) if (r !== c) { const f = m[r][c]; if (f) for (let j = 0; j < 2 * n; j++) m[r][j] -= f * m[c][j]; }
  }
  return m.map(r => r.slice(n));
}

/**
 * The chapter's `KalmanBox`: state x = [cx, cy, w, h, vx, vy], measurement z = [cx, cy, w, h], one frame per step.
 * P starts at r on the box entries and 100 on the unknown velocities; Q = q * diag(1, 1, 0.1, 0.1, 0.5, 0.5); R = r * I.
 */
export class KalmanBox {
  x: number[];
  P: Mat;
  private readonly F: Mat;
  private readonly Q: Mat;
  private readonly r: number;
  constructor(z: Box, q = 1, r = 4) {
    this.x = [z.cx, z.cy, z.w, z.h, 0, 0];
    this.P = [r, r, r, r, 100, 100].map((v, i) => Array.from({ length: 6 }, (_, j) => (i === j ? v : 0)));
    this.F = eye(6); this.F[0][4] = 1; this.F[1][5] = 1;
    this.Q = [1, 1, 0.1, 0.1, 0.5, 0.5].map((v, i) => Array.from({ length: 6 }, (_, j) => (i === j ? q * v : 0)));
    this.r = r;
  }
  get box(): Box { return { cx: this.x[0], cy: this.x[1], w: this.x[2], h: this.x[3] }; }
  /** Time update: x <- F x, P <- F P F^T + Q. Returns the predicted box. */
  predict(): Box {
    this.x = this.F.map(row => row.reduce((s, v, k) => s + v * this.x[k], 0));
    this.P = matAdd(matMul(matMul(this.F, this.P), transpose(this.F)), this.Q);
    return this.box;
  }
  /** Measurement update with a matched detection z. */
  update(z: Box): void {
    const zv = [z.cx, z.cy, z.w, z.h];
    // H = [I4 0]: S is the top-left 4 x 4 block of P plus R, and P H^T is the first four columns of P.
    const S: Mat = Array.from({ length: 4 }, (_, i) => Array.from({ length: 4 }, (_, j) => this.P[i][j] + (i === j ? this.r : 0)));
    const PHt: Mat = this.P.map(row => row.slice(0, 4));
    const K = matMul(PHt, inverse(S));
    const innov = zv.map((v, i) => v - this.x[i]);
    this.x = this.x.map((v, i) => v + K[i].reduce((s, k, j) => s + k * innov[j], 0));
    // P <- (I - K H) P
    const KH: Mat = K.map(row => [...row, 0, 0]);
    this.P = matMul(eye(6).map((row, i) => row.map((v, j) => v - KH[i][j])), this.P);
  }
}

// ---------------------------------------------------------------------------------------------- appearance

/** Pixel range [lo, hi) covered by a box along one axis, rounded half up and clamped to the image. */
export function pixelRange(centre: number, size: number, limit: number): [number, number] {
  const lo = Math.floor(centre - size / 2 + 0.5), hi = Math.floor(centre + size / 2 + 0.5);
  return [Math.max(0, lo), Math.min(limit, hi)];
}
/**
 * Appearance embedding of a detection: the colour histogram of the pixels inside its box, `bins` bins per channel
 * (bins^3 numbers), scaled to unit length so that 1 - dot product is the cosine distance of the chapter. Empty box: zeros.
 */
export function colourEmbedding(img: Rgb, box: Box, bins = 4): Float64Array {
  const [x0, x1] = pixelRange(box.cx, box.w, img.w), [y0, y1] = pixelRange(box.cy, box.h, img.h);
  const hist = new Float64Array(bins * bins * bins);
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
    const i = 3 * (y * img.w + x);
    const r = Math.min(bins - 1, Math.floor(img.data[i] * bins)), g = Math.min(bins - 1, Math.floor(img.data[i + 1] * bins)), b = Math.min(bins - 1, Math.floor(img.data[i + 2] * bins));
    hist[(r * bins + g) * bins + b]++;
  }
  return unit(hist);
}
export function unit(v: Float64Array): Float64Array {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const n = Math.sqrt(s), out = new Float64Array(v.length);
  if (n > 0) for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}
export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
/** Track memory update m <- alpha m + (1 - alpha) e, renormalised (the chapter uses alpha = 0.9). */
export function updateMemory(m: Float64Array, e: Float64Array, alpha = 0.9): Float64Array {
  const out = new Float64Array(m.length);
  for (let i = 0; i < m.length; i++) out[i] = alpha * m[i] + (1 - alpha) * e[i];
  return unit(out);
}

// ---------------------------------------------------------------------------------------------- association and tracker

export interface CostMatrices { iou: number[][]; app: number[][]; total: number[][] }
/** C[k][j] = lambda (1 - IoU(predicted box k, detection j)) + (1 - lambda)(1 - m_k . e_j); also returns the two parts. */
export function associationCost(predicted: Box[], memory: Float64Array[], dets: Box[], embeddings: Float64Array[], lambda: number): CostMatrices {
  const iou = predicted.map(p => dets.map(d => 1 - iouBox(p, d))), app = memory.map(m => embeddings.map(e => 1 - dot(m, e)));
  return { iou, app, total: iou.map((row, k) => row.map((v, j) => lambda * v + (1 - lambda) * app[k][j])) };
}

/** A detection: its box, its embedding, and the true object that produced it (used only to evaluate, never by the tracker). */
export interface Detection { box: Box; embedding: Float64Array; truth: number }
export interface FrameResult {
  /** For each track, the index (in this frame's detection list) of the detection it took. */
  assign: number[];
  /** Predicted boxes (before the update) and filtered boxes (after it), per track. */
  predicted: Box[];
  boxes: Box[];
  costs: CostMatrices | null;
}
export interface TrackerRun {
  frames: FrameResult[];
  /** identity[f][o]: the id of the track that holds true object o in frame f. */
  identity: number[][];
}
export interface TrackerParams { lambda: number; q?: number; r?: number; alpha?: number }

/**
 * Online tracking by detection over a list of frames of detections. Frame 0 starts one track per detection (track k takes the
 * detection whose true object is k, so ids equal object numbers at the start); later frames predict, build the cost matrix and solve it with
 * the Hungarian algorithm; matched tracks update their Kalman filter and appearance memory. Every frame has as many
 * detections as tracks here (no misses, no false alarms), so there is no gating, birth or death.
 */
export function runTracker(detections: Detection[][], p: TrackerParams): TrackerRun {
  const { lambda, q = 1, r = 4, alpha = 0.9 } = p, K = detections[0].length;
  const filters: KalmanBox[] = [], memory: Float64Array[] = [];
  const frames: FrameResult[] = [], identity: number[][] = [];
  detections[0].forEach(d => { filters[d.truth] = new KalmanBox(d.box, q, r); memory[d.truth] = d.embedding; });
  {
    const assign = new Array(K).fill(0);
    detections[0].forEach((d, j) => { assign[d.truth] = j; });
    frames.push({ assign, predicted: filters.map(f => f.box), boxes: filters.map(f => f.box), costs: null });
    identity.push(detections[0].reduce((a, d) => { a[d.truth] = d.truth; return a; }, new Array<number>(K).fill(0)));
  }
  for (let f = 1; f < detections.length; f++) {
    const dets = detections[f], predicted = filters.map(k => k.predict());
    const costs = associationCost(predicted, memory, dets.map(d => d.box), dets.map(d => d.embedding), lambda);
    const assign = hungarian(costs.total), id = new Array<number>(K).fill(-1);
    assign.forEach((j, k) => {
      filters[k].update(dets[j].box);
      memory[k] = updateMemory(memory[k], dets[j].embedding, alpha);
      id[dets[j].truth] = k;
    });
    frames.push({ assign, predicted, boxes: filters.map(k => k.box), costs });
    identity.push(id);
  }
  return { frames, identity };
}

export interface SwitchCount {
  /** Identity switches in the chapter's sense: an object is matched to a different track id than in the previous frame. */
  switches: number;
  /** First frame in which a switch happens, or -1. */
  firstFrame: number;
  /** Object-frames in which the track holding the object is not the track that started on it. */
  wrongFrames: number;
  /** Objects times frames. */
  total: number;
}
export function countSwitches(identity: number[][]): SwitchCount {
  let switches = 0, firstFrame = -1, wrongFrames = 0, total = 0;
  identity.forEach((row, f) => row.forEach((id, o) => {
    total++;
    if (id !== o) wrongFrames++;
    if (f > 0 && id !== identity[f - 1][o]) { switches++; if (firstFrame < 0) firstFrame = f; }
  }));
  return { switches, firstFrame, wrongFrames, total };
}
/** MOTA of the chapter, 1 - (FN + FP + IDSW) / GT, summed over frames. Here FN = FP = 0 (every object is detected once per frame). */
export const mota = (fn: number, fp: number, idsw: number, gt: number) => 1 - (fn + fp + idsw) / gt;
/** Mean IoU between each track's filtered box and the true box of the object that track started on (frames 1 onward). */
export function meanTrackIou(run: TrackerRun, truth: Box[][]): number {
  let s = 0, n = 0;
  for (let f = 1; f < run.frames.length; f++) run.frames[f].boxes.forEach((b, k) => { s += iouBox(b, truth[f][k]); n++; });
  return n ? s / n : 0;
}
