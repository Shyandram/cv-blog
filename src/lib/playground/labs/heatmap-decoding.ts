import type { LabDef } from './types.ts';
import { L, pick, fmt, num } from '../types.ts';
import { plotColors, startPlot, label, line } from './plot2d.ts';
import { mulberry32 } from '../../imagekit/rng.ts';
import type { Lang } from '../../../i18n/ui';

// ---------------------------------------------------------------------------------------------- the simulation
// 500 keypoints fall at random sub-pixel positions in a 256 x 256 input (16 px margin). The "network" is perfect up to a
// little noise: its heatmap at output stride s is the Gaussian of the chapter, exp(-|p - x/s|^2 / (2 sigma^2)) with p an integer
// cell, plus N(0, 0.01^2) noise per cell, floored at 1e-6. Three decoders read it back (the chapter's code, ported):
//   argmax  : s * (brightest cell)
//   quarter : the brightest cell moved by 1/4 cell toward its brighter neighbour on each axis (hourglass, SimpleBaseline)
//   taylor  : one Newton step on log(heatmap) at the brightest cell (DARK)
// Only a window of (2R + 1)^2 cells around the true cell is generated, R = ceil(2 sigma) + 2. The Gaussian is at least 0.77 at the
// brightest cell and below exp(-2) = 0.14 at every cell outside the window, 65 noise standard deviations apart, so the rest of the
// map cannot change the brightest cell or its 3 x 3 neighbourhood. This keeps a recompute to a few hundred cells per keypoint.
export const INPUT = 256, MARGIN = 16, N_KEYPOINTS = 500, NOISE = 0.01, FLOOR = 1e-6;
export const STRIDES = [1, 2, 4, 8] as const;
export const DECODERS = ['argmax', 'quarter', 'taylor'] as const;
export type Decoder = (typeof DECODERS)[number];
export const KEYPOINT_SEED = 2024, NOISE_SEED = 77;
export const DEFAULTS = { stride: 4, decoder: 'argmax' as Decoder, sigma: 2 };
/** The keypoint drawn in the zoomed grid, in input pixels. */
export const EXAMPLE = { x: 102.6, y: 78.9 };
/** Mean distance from the centre of a unit square to a random point of it: (sqrt 2 + ln(1 + sqrt 2)) / 6. */
export const SQUARE_MEAN_DIST = (Math.SQRT2 + Math.log(1 + Math.SQRT2)) / 6;

export interface Pt { x: number; y: number }
export function keypoints(n = N_KEYPOINTS, seed = KEYPOINT_SEED): Pt[] {
  const r = mulberry32(seed), span = INPUT - 2 * MARGIN, out: Pt[] = [];
  for (let i = 0; i < n; i++) out.push({ x: MARGIN + span * r(), y: MARGIN + span * r() });
  return out;
}
export const windowRadius = (sigma: number) => Math.ceil(2 * sigma) + 2;

/** A square window of heatmap cells. Cell (i, j) of `vals` is the map cell (ix0 + i, iy0 + j), whose centre is at that integer position. */
export interface Window { vals: Float64Array; side: number; ix0: number; iy0: number }
function normals(count: number, seed: number): Float64Array {
  const r = mulberry32(seed), out = new Float64Array(count);
  for (let i = 0; i < count; i++) out[i] = Math.sqrt(-2 * Math.log(Math.max(1e-12, r()))) * Math.cos(2 * Math.PI * r());
  return out;
}
/** Gaussian heatmap (exact, unrounded centre) with additive noise, restricted to the window around the keypoint. */
export function heatmapWindow(pt: Pt, stride: number, sigma: number, noise: number, noiseSeed: number): Window {
  const cx = pt.x / stride, cy = pt.y / stride, R = windowRadius(sigma), side = 2 * R + 1;
  const ix0 = Math.round(cx) - R, iy0 = Math.round(cy) - R, vals = new Float64Array(side * side);
  const z = noise > 0 ? normals(side * side, noiseSeed) : null, inv = 1 / (2 * sigma * sigma);
  for (let j = 0; j < side; j++) for (let i = 0; i < side; i++) {
    const dx = ix0 + i - cx, dy = iy0 + j - cy;
    vals[j * side + i] = Math.max(FLOOR, Math.exp(-(dx * dx + dy * dy) * inv) + (z ? noise * z[j * side + i] : 0));
  }
  return { vals, side, ix0, iy0 };
}
/** Brightest cell (first in row-major order on ties, like numpy). */
export function peak(w: Window): { i: number; j: number } {
  let best = -Infinity, bi = 0, bj = 0;
  for (let j = 0; j < w.side; j++) for (let i = 0; i < w.side; i++) { const v = w.vals[j * w.side + i]; if (v > best) { best = v; bi = i; bj = j; } }
  return { i: bi, j: bj };
}
/** Decode a window to input pixels. The 3 x 3 neighbourhood of the peak always lies inside the window here. */
export function decode(w: Window, stride: number, method: Decoder): Pt {
  const { i, j } = peak(w), s = w.side, v = (a: number, b: number) => w.vals[b * s + a];
  let px = w.ix0 + i, py = w.iy0 + j;
  const inside = i >= 1 && i <= s - 2 && j >= 1 && j <= s - 2;
  if (method === 'quarter' && inside) {
    px += 0.25 * Math.sign(v(i + 1, j) - v(i - 1, j));
    py += 0.25 * Math.sign(v(i, j + 1) - v(i, j - 1));
  } else if (method === 'taylor' && inside) {
    const L = (a: number, b: number) => Math.log(Math.max(v(a, b), 1e-10));
    const gx = (L(i + 1, j) - L(i - 1, j)) / 2, gy = (L(i, j + 1) - L(i, j - 1)) / 2;
    const hxx = L(i + 1, j) - 2 * L(i, j) + L(i - 1, j), hyy = L(i, j + 1) - 2 * L(i, j) + L(i, j - 1);
    const hxy = (L(i + 1, j + 1) - L(i - 1, j + 1) - L(i + 1, j - 1) + L(i - 1, j - 1)) / 4;
    const det = hxx * hyy - hxy * hxy;
    if (det > 0 && hxx < 0) { px += -(hyy * gx - hxy * gy) / det; py += -(-hxy * gx + hxx * gy) / det; }
  }
  return { x: px * stride, y: py * stride };
}

export type Sweep = Record<number, Record<Decoder, Float64Array>>;
/** Localization error in input pixels of every decoder at every stride, for the same keypoints and the same noise at one stride. */
export function sweep(sigma: number, n = N_KEYPOINTS, noise = NOISE): Sweep {
  const pts = keypoints(n), out: Sweep = {};
  for (const s of STRIDES) {
    const e = { argmax: new Float64Array(n), quarter: new Float64Array(n), taylor: new Float64Array(n) };
    for (let k = 0; k < n; k++) {
      const w = heatmapWindow(pts[k], s, sigma, noise, NOISE_SEED + 7919 * k + 104729 * s);
      for (const d of DECODERS) { const p = decode(w, s, d); e[d][k] = Math.hypot(p.x - pts[k].x, p.y - pts[k].y); }
    }
    out[s] = e;
  }
  return out;
}
export const mean = (a: ArrayLike<number>) => { let t = 0; for (let i = 0; i < a.length; i++) t += a[i]; return a.length ? t / a.length : 0; };

// ---------------------------------------------------------------------------------------------- drawing
const GRID_R = 4, GRID_N = 2 * GRID_R + 1, CELL = 40, GRID_W = GRID_N * CELL, STRIP = 40, GRID_H = GRID_W + STRIP;
const STOPS: [number, number, number][] = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
function ramp(v: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, v)) * (STOPS.length - 1), k = Math.min(STOPS.length - 2, Math.floor(t)), f = t - k;
  return [0, 1, 2].map(c => Math.round(STOPS[k][c] * (1 - f) + STOPS[k + 1][c] * f)) as [number, number, number];
}
const DEC_COLOUR: Record<Decoder, string> = { argmax: '--plot-b', quarter: '--plot-c', taylor: '--plot-a' };
const shortDecoder = (lang: Lang, d: Decoder) => d === 'argmax' ? 'argmax' : d === 'quarter' ? pick(lang, 'quarter', '¼ 偏移') : 'Taylor';
const decoderName = (lang: Lang, d: Decoder) => d === 'argmax' ? 'argmax' : d === 'quarter' ? pick(lang, 'quarter offset', '四分之一偏移') : pick(lang, 'Taylor step', 'Taylor 步驟');

function drawGrid(canvas: HTMLCanvasElement, lang: Lang, stride: number, sigma: number, decoder: Decoder) {
  const col = plotColors(canvas), ctx = startPlot(canvas, GRID_W, GRID_H, col);
  const w = heatmapWindow(EXAMPLE, stride, sigma, NOISE, NOISE_SEED + 1), cx = EXAMPLE.x / stride, cy = EXAMPLE.y / stride;
  const ox = Math.round(cx) - GRID_R, oy = Math.round(cy) - GRID_R; // grid origin in map cells
  ctx.font = '600 12px system-ui, -apple-system, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (let j = 0; j < GRID_N; j++) for (let i = 0; i < GRID_N; i++) {
    const v = w.vals[(oy + j - w.iy0) * w.side + (ox + i - w.ix0)], [r, g, b] = ramp(v);
    ctx.fillStyle = `rgb(${r},${g},${b})`; ctx.fillRect(i * CELL, j * CELL, CELL, CELL);
    ctx.fillStyle = (0.299 * r + 0.587 * g + 0.114 * b) > 140 ? '#101010' : '#ffffff';
    ctx.fillText(v.toFixed(2).replace(/^0/, ''), i * CELL + CELL / 2, j * CELL + CELL / 2);
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1;
  for (let k = 0; k <= GRID_N; k++) { ctx.beginPath(); ctx.moveTo(k * CELL, 0); ctx.lineTo(k * CELL, GRID_W); ctx.moveTo(0, k * CELL); ctx.lineTo(GRID_W, k * CELL); ctx.stroke(); }
  const toX = (c: number) => (c - ox + 0.5) * CELL, toY = (c: number) => (c - oy + 0.5) * CELL;
  // The brightest cell.
  const pk = peak(w), px = w.ix0 + pk.i, py = w.iy0 + pk.j;
  ctx.setLineDash([5, 3]); ctx.lineWidth = 2.5; ctx.strokeStyle = '#ffffff';
  ctx.strokeRect((px - ox) * CELL + 2, (py - oy) * CELL + 2, CELL - 4, CELL - 4); ctx.setLineDash([]);
  const dec = decode(w, stride, decoder), dx = dec.x / stride, dy = dec.y / stride, err = Math.hypot(dec.x - EXAMPLE.x, dec.y - EXAMPLE.y);
  // Error segment, true point (cross) and decoded point (disc).
  line(ctx, toX(cx), toY(cy), toX(dx), toY(dy), 'rgba(0,0,0,0.85)', 5); line(ctx, toX(cx), toY(cy), toX(dx), toY(dy), '#ffffff', 2);
  const cross = (x: number, y: number, r: number, c: string, lw: number) => { ctx.strokeStyle = c; ctx.lineWidth = lw; ctx.beginPath(); ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x - r, y + r); ctx.lineTo(x + r, y - r); ctx.stroke(); };
  cross(toX(cx), toY(cy), 8, '#000000', 6); cross(toX(cx), toY(cy), 8, '#ffffff', 3);
  ctx.beginPath(); ctx.arc(toX(dx), toY(dy), 7, 0, 2 * Math.PI); ctx.fillStyle = '#ff7a1a'; ctx.fill(); ctx.lineWidth = 2.5; ctx.strokeStyle = '#000000'; ctx.stroke();
  // Legend strip.
  ctx.fillStyle = col.bg; ctx.fillRect(0, GRID_W, GRID_W, STRIP);
  label(ctx, pick(lang, `1 cell = ${stride} px · × true · ● ${decoderName(lang, decoder)} · ▢ brightest`, `1 格 = ${stride} 像素 · × 真實 · ● ${decoderName(lang, decoder)} · ▢ 最亮格`), 6, GRID_W + 17, { color: col.muted, size: 13 });
  label(ctx, pick(lang, `error of this keypoint: ${fmt(err, 2)} px`, `這個關鍵點的誤差：${fmt(err, 2)} 像素`), 6, GRID_W + 34, { color: col.ink, size: 14, bold: true });
  return err;
}

const CURVE_W = 360, CURVE_H = 300;
function drawCurve(canvas: HTMLCanvasElement, lang: Lang, sw: Sweep, stride: number, decoder: Decoder) {
  const col = plotColors(canvas), ctx = startPlot(canvas, CURVE_W, CURVE_H, col), cs = getComputedStyle(canvas);
  const colour = (d: Decoder) => cs.getPropertyValue(DEC_COLOUR[d]).trim() || '#888';
  const x0 = 46, x1 = CURVE_W - 76, y0 = CURVE_H - 46, y1 = 44, ymax = 3.5;
  const X = (i: number) => x0 + (i / (STRIDES.length - 1)) * (x1 - x0), Y = (v: number) => y0 - (Math.min(v, ymax) / ymax) * (y0 - y1);
  label(ctx, pick(lang, 'Mean error against stride', '平均誤差與步幅'), 8, 20, { bold: true, color: col.ink, size: 15 });
  for (let t = 0; t <= 3; t++) { line(ctx, x0, Y(t), x1 + 6, Y(t), col.grid, 1); label(ctx, String(t), x0 - 8, Y(t) + 5, { align: 'right', color: col.muted, size: 13 }); }
  label(ctx, pick(lang, 'px', '像素'), x0 - 8, y1 - 8, { align: 'right', color: col.muted, size: 13 });
  STRIDES.forEach((s, i) => label(ctx, String(s), X(i), y0 + 20, { align: 'center', color: s === stride ? col.ink : col.muted, size: 14, bold: s === stride }));
  label(ctx, pick(lang, 'output stride (input pixels per cell)', '輸出步幅（每格的輸入像素數）'), (x0 + x1) / 2, y0 + 40, { align: 'center', color: col.muted, size: 13 });
  const si = STRIDES.indexOf(stride as 1 | 2 | 4 | 8);
  if (si >= 0) { ctx.fillStyle = col.grid; ctx.globalAlpha = 0.35; ctx.fillRect(X(si) - 16, y1 - 4, 32, y0 - y1 + 4); ctx.globalAlpha = 1; }
  // Theory for argmax without noise.
  ctx.setLineDash([5, 4]); ctx.lineWidth = 1.6; ctx.strokeStyle = col.muted; ctx.beginPath();
  STRIDES.forEach((s, i) => { const x = X(i), y = Y(SQUARE_MEAN_DIST * s); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.stroke(); ctx.setLineDash([]);
  const order: Decoder[] = ['argmax', 'quarter', 'taylor'].filter(d => d !== decoder).concat(decoder) as Decoder[];
  for (const d of order) {
    const sel = d === decoder, c = colour(d);
    ctx.globalAlpha = sel ? 1 : 0.8; ctx.lineWidth = sel ? 3.4 : 1.8; ctx.strokeStyle = c; ctx.beginPath();
    STRIDES.forEach((s, i) => { const x = X(i), y = Y(mean(sw[s][d])); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.stroke();
    STRIDES.forEach((s, i) => { ctx.beginPath(); ctx.arc(X(i), Y(mean(sw[s][d])), s === stride ? 6 : 3.4, 0, 2 * Math.PI); ctx.fillStyle = c; ctx.fill(); if (s === stride) { ctx.lineWidth = 2; ctx.strokeStyle = col.ink; ctx.stroke(); } });
    ctx.globalAlpha = 1;
    label(ctx, shortDecoder(lang, d), x1 + 12, Y(mean(sw[8][d])) + 5, { color: c, size: 13, bold: sel });
  }
  label(ctx, pick(lang, 'dashed: 0.383 × stride', '虛線：0.383 × 步幅'), x0 + 14, y1 - 8, { color: col.muted, size: 13 });
}

const HIST_W = 360, HIST_H = 262, BINS = 32, HIST_MAX = 8; // errors above HIST_MAX px fall into the last bin
function histogram(errors: Float64Array): Int32Array {
  const h = new Int32Array(BINS);
  for (let k = 0; k < errors.length; k++) h[Math.min(BINS - 1, Math.floor((errors[k] / HIST_MAX) * BINS))]++;
  return h;
}
function drawHist(canvas: HTMLCanvasElement, lang: Lang, sw: Sweep, stride: number, decoder: Decoder) {
  const col = plotColors(canvas), ctx = startPlot(canvas, HIST_W, HIST_H, col), cs = getComputedStyle(canvas);
  const colour = (d: Decoder) => cs.getPropertyValue(DEC_COLOUR[d]).trim() || '#888';
  const x0 = 40, x1 = HIST_W - 12, y0 = HIST_H - 46, y1 = 68, bw = (x1 - x0) / BINS;
  const sel = histogram(sw[stride][decoder]), arg = histogram(sw[stride].argmax), top = Math.max(1, ...sel, ...arg), Y = (c: number) => y0 - (c / top) * (y0 - y1);
  label(ctx, pick(lang, `Error of ${sw[stride][decoder].length} keypoints, stride ${stride}`, `${sw[stride][decoder].length} 個關鍵點的誤差，步幅 ${stride}`), 8, 20, { bold: true, color: col.ink, size: 15 });
  line(ctx, x0, y0, x1, y0, col.muted, 1.2);
  ctx.fillStyle = colour(decoder); ctx.globalAlpha = 0.9;
  for (let b = 0; b < BINS; b++) if (sel[b]) ctx.fillRect(x0 + b * bw + 0.5, Y(sel[b]), bw - 1, y0 - Y(sel[b]));
  ctx.globalAlpha = 1;
  if (decoder !== 'argmax') { // outline of the plain argmax for comparison
    ctx.strokeStyle = colour('argmax'); ctx.lineWidth = 1.8; ctx.beginPath();
    for (let b = 0; b < BINS; b++) { const xa = x0 + b * bw, xb = xa + bw, y = Y(arg[b]); if (b === 0) ctx.moveTo(xa, y); else ctx.lineTo(xa, y); ctx.lineTo(xb, y); } ctx.stroke();
  }
  for (let t = 0; t <= HIST_MAX; t += 2) { const x = x0 + (t / HIST_MAX) * (x1 - x0); line(ctx, x, y0, x, y0 + 5, col.muted, 1); label(ctx, String(t), x, y0 + 20, { align: 'center', color: col.muted, size: 13 }); }
  label(ctx, pick(lang, 'error (input pixels)', '誤差（輸入像素）'), (x0 + x1) / 2, y0 + 38, { align: 'center', color: col.muted, size: 13 });
  const m = mean(sw[stride][decoder]), xm = x0 + (Math.min(m, HIST_MAX) / HIST_MAX) * (x1 - x0);
  line(ctx, xm, 46, xm, y0, col.ink, 2, [5, 3]);
  label(ctx, pick(lang, `mean ${fmt(m, 2)} px`, `平均 ${fmt(m, 2)} 像素`), xm > x1 - 110 ? xm - 6 : xm + 6, 42, { color: col.ink, size: 13, bold: true, align: xm > x1 - 110 ? 'right' : 'left' });
  if (decoder !== 'argmax') label(ctx, pick(lang, 'orange outline: plain argmax', '橘色外框：單純 argmax'), x1, 60, { color: col.muted, size: 13, align: 'right' });
}

// ---------------------------------------------------------------------------------------------- the lab
const pxs = (a: ArrayLike<number>[]) => a.map(x => fmt(mean(x), 2)).join(' / ');
export const lab: LabDef = {
  kind: 'image', id: 'heatmap-decoding', track: 'deep-learning',
  title: L('DLCV 05 · Is the argmax of a heatmap the keypoint?', 'DLCV 05・熱圖的 argmax 就是關鍵點嗎？'),
  description: L('Render a Gaussian heatmap at output stride 1, 2, 4 or 8 for 500 random sub-pixel keypoints and compare three ways to read the position back: the brightest cell, a quarter-cell offset and a Taylor step.', '對 500 個隨機的子像素關鍵點，以步幅 1、2、4 或 8 繪製高斯熱圖，比較三種讀回位置的方法：最亮的格子、四分之一格偏移與 Taylor 步驟。'),
  challenge: L(
    'At the default setting (stride 4, plain argmax) the 500 keypoints are off by 1.58 px on average, and the table of means shows the error doubling with every doubling of the stride: 0.37, 0.78, 1.58 and 3.12 px at strides 1, 2, 4 and 8, close to 0.383 times the stride. Switch the decoding to the quarter offset (about half of that) and then to the Taylor step: at stride 8 it brings 3.12 px down to 0.41 px. Now keep the Taylor step and raise σ from 2 to 4 cells: at stride 8 its error grows from 0.41 to 1.58 px. A wider blob should be easier to read, so why does the same noise hurt more?',
    '預設設定（步幅 4、單純 argmax）下，500 個關鍵點平均偏離 1.58 像素；平均誤差表顯示步幅每加倍，誤差也加倍：步幅 1、2、4、8 分別是 0.37、0.78、1.58 與 3.12 像素，接近步幅的 0.383 倍。把解碼改成四分之一偏移（約減半），再改成 Taylor 步驟：在步幅 8，它把 3.12 像素降到 0.41 像素。接著保持 Taylor 步驟，把 σ 從 2 格提高到 4 格：在步幅 8，它的誤差從 0.41 增加到 1.58 像素。較寬的峰應該比較容易讀，為什麼同樣的雜訊卻傷害更大？'),
  note: L(
    'A toy that mirrors the chapter’s code, with its own random numbers (so the means differ in the second decimal from the chapter’s run). 500 keypoints are drawn uniformly in [16, 240] × [16, 240] pixels of a 256 × 256 input. The heatmap at stride s is exp(−‖p − x/s‖² / (2σ²)) on integer cells p, centred on the exact unrounded position, plus Gaussian noise of standard deviation 0.01 per cell, floored at 10⁻⁶; a cell’s centre is the integer position, as in the chapter, so the brightest cell i maps to s · i. Argmax decodes s · (brightest cell). Quarter offset moves the brightest cell by ¼ cell on each axis toward the brighter neighbour. The Taylor step is one Newton step on the logarithm of the heatmap at the brightest cell, with central differences, kept only if the 2 × 2 Hessian is negative definite. The error is the Euclidean distance to the true position in input pixels. Only the (2R + 1)² cells around the true cell, R = ⌈2σ⌉ + 2, are generated; outside them the Gaussian is below 0.14 (against a peak of at least 0.77 and noise of 0.01), so the rest of the map cannot change the answer. The grid shows one fixed keypoint at (102.6, 78.9), the same noise for every decoder. Real networks add errors that this toy leaves out; the lab isolates the error that comes from the grid alone.',
    '一個模仿章節程式碼的玩具，使用自己的亂數（因此平均值與章節那次執行相差在小數第二位）。500 個關鍵點在 256 × 256 輸入的 [16, 240] × [16, 240] 像素內均勻抽樣。步幅 s 的熱圖是整數格 p 上的 exp(−‖p − x/s‖² / (2σ²))，以未四捨五入的精確位置為中心，再加上每格標準差 0.01 的高斯雜訊，下限為 10⁻⁶；與章節相同，格子的中心就是整數位置，所以最亮的格子 i 對應到 s · i。Argmax 解碼為 s · （最亮的格子）。四分之一偏移是在每個軸上，把最亮的格子朝較亮的鄰格移動 ¼ 格。Taylor 步驟是在最亮的格子上，對熱圖的對數做一次牛頓步，使用中央差分，且只有在 2 × 2 的 Hessian 為負定時才採用。誤差是與真實位置的歐氏距離（輸入像素）。這裡只產生真實格子周圍的 (2R + 1)² 個格子，R = ⌈2σ⌉ + 2；視窗之外高斯值低於 0.14（峰值至少 0.77，雜訊只有 0.01），所以其餘的熱圖不會改變答案。格線圖顯示固定的一個關鍵點 (102.6, 78.9)，所有解碼器使用相同的雜訊。真實網路還有這個玩具省略的其他誤差；本實驗只隔離出格子本身造成的誤差。'),
  related: ['deep-learning/dl-06a-pose-keypoints'],
  controls: [
    { key: 'stride', label: L('Output stride (input pixels per heatmap cell)', '輸出步幅（每個熱圖格子的輸入像素數）'), kind: 'select', value: String(DEFAULTS.stride), options: STRIDES.map(s => ({ value: String(s), label: L(String(s), String(s)) })) },
    {
      key: 'decoder', label: L('Decoding', '解碼方式'), kind: 'select', value: DEFAULTS.decoder, options: [
        { value: 'argmax', label: L('Argmax (brightest cell)', 'Argmax（最亮的格子）') },
        { value: 'quarter', label: L('Argmax + quarter-cell offset', 'Argmax + 四分之一格偏移') },
        { value: 'taylor', label: L('Sub-pixel: Taylor step on log heatmap', '子像素：對數熱圖的 Taylor 步驟') },
      ],
    },
    { key: 'sigma', label: L('Blob width σ (heatmap cells)', '峰的寬度 σ（熱圖格數）'), kind: 'range', value: DEFAULTS.sigma, min: 1, max: 4, step: 0.5 },
  ],
  predict: {
    question: L(
      'A network returns an almost perfect Gaussian heatmap at stride 4 (one cell = 4 input pixels) for a keypoint at a random sub-pixel position. You report the position of the brightest cell as the keypoint. On average, how far off is it?',
      '網路對一個隨機子像素位置的關鍵點，在步幅 4（一格 = 4 個輸入像素）回傳幾乎完美的高斯熱圖。你把最亮格子的位置當作關鍵點。平均會差多遠？'),
    options: [
      { value: 'zero', label: L('About 0 px: the peak of the heatmap is the keypoint', '約 0 像素：熱圖的峰值就是關鍵點') },
      { value: 'one-half', label: L('About 1.5 px: the grid alone costs about 0.38 of a cell', '約 1.5 像素：光是格子就損失約 0.38 格') },
      { value: 'four', label: L('About 4 px: a whole cell', '約 4 像素：整整一格') },
    ],
    answer: 'one-half',
    explain: L(
      'About 1.5 px. The peak can only sit on a grid with 4 px spacing, but the keypoint can fall anywhere between grid points, so the brightest cell is the nearest grid point and is off by up to half a cell on each axis. The mean distance is 0.383 of a cell: 1.58 px in this run (0.37, 0.78, 1.58, 3.12 px at strides 1, 2, 4, 8), before the network makes any mistake. A Taylor step on the log of the heatmap cuts this to 0.21 px.',
      '約 1.5 像素。峰值只能落在間距 4 像素的格點上，但關鍵點可以落在格點之間的任何位置，所以最亮的格子只是最近的格點，每個軸最多差半格。平均距離是 0.383 格：這次執行是 1.58 像素（步幅 1、2、4、8 分別為 0.37、0.78、1.58、3.12 像素），而且這還沒算網路自己的錯誤。對熱圖的對數做 Taylor 步驟，可以把它降到 0.21 像素。'),
  },
  assets: {},
  paneMin: 330,
  panes: [
    { key: 'grid', kind: 'canvas', label: L('Heatmap cells around one keypoint (values 0 to 1): true position, decoded position and the brightest cell', '一個關鍵點周圍的熱圖格子（數值 0 到 1）：真實位置、解碼位置與最亮的格子'), width: GRID_W, height: GRID_H },
    { key: 'curve', kind: 'canvas', label: L('Mean error of the three decoders at each stride, over 500 keypoints', '三種解碼方式在各步幅下，500 個關鍵點的平均誤差'), width: CURVE_W, height: CURVE_H },
    { key: 'hist', kind: 'canvas', label: L('Histogram of the 500 errors for the chosen decoding and stride', '所選解碼方式與步幅下，500 個誤差的直方圖'), width: HIST_W, height: HIST_H },
  ],
  sources: [
    { title: 'J. Tompson, A. Jain, Y. LeCun and C. Bregler, Joint Training of a Convolutional Network and a Graphical Model for Human Pose Estimation, NeurIPS, 2014 (heatmap targets: a 2D Gaussian at the joint)', url: 'https://arxiv.org/abs/1406.2984' },
    { title: 'A. Newell, K. Yang and J. Deng, Stacked Hourglass Networks for Human Pose Estimation, ECCV, 2016 (quarter-pixel offset toward the next highest neighbour)', url: 'https://doi.org/10.1007/978-3-319-46484-8_29' },
    { title: 'B. Xiao, H. Wu and Y. Wei, Simple Baselines for Human Pose Estimation and Tracking, ECCV, 2018 (quarter offset from the highest to the second highest response)', url: 'https://doi.org/10.1007/978-3-030-01231-1_29' },
    { title: 'F. Zhang, X. Zhu, H. Dai, M. Ye and C. Zhu, Distribution-Aware Coordinate Representation for Human Pose Estimation, CVPR, 2020 (Taylor-expansion decoding of the heatmap)', url: 'https://doi.org/10.1109/CVPR42600.2020.00712' },
  ],
  create({ lang }) {
    const cache = new Map<number, Sweep>();
    return {
      render(state, canvases) {
        const stride = Math.round(num(state, 'stride')), sigma = Math.max(1, Math.min(4, num(state, 'sigma')));
        const decoder = (DECODERS as readonly string[]).includes(String(state.decoder)) ? (state.decoder as Decoder) : 'argmax';
        const key = Math.round(sigma * 100);
        if (!cache.has(key)) { if (cache.size > 6) cache.clear(); cache.set(key, sweep(sigma)); }
        const sw = cache.get(key)!, errs = sw[stride] ?? sw[4];
        const exErr = drawGrid(canvases.grid, lang, stride, sigma, decoder);
        drawCurve(canvases.curve, lang, sw, stride, decoder);
        drawHist(canvases.hist, lang, sw, stride, decoder);
        const mSel = mean(errs[decoder]), mArg = mean(errs.argmax);
        return {
          summary: pick(lang,
            `At stride ${stride} the ${decoderName(lang, decoder)} decoding is off by ${fmt(mSel, 2)} px on average over ${errs.argmax.length} keypoints; plain argmax is off by ${fmt(mArg, 2)} px, which is ${fmt(mArg / stride, 2)} of a cell.`,
            `在步幅 ${stride}，${decoderName(lang, decoder)}解碼對 ${errs.argmax.length} 個關鍵點平均偏離 ${fmt(mSel, 2)} 像素；單純 argmax 偏離 ${fmt(mArg, 2)} 像素，約為 ${fmt(mArg / stride, 2)} 格。`),
          metrics: [
            { label: pick(lang, `Mean error, ${decoderName(lang, decoder)}, stride ${stride}`, `平均誤差，${decoderName(lang, decoder)}，步幅 ${stride}`), value: `${fmt(mSel, 2)} px` },
            { label: pick(lang, 'Argmax at stride 1 / 2 / 4 / 8', 'Argmax，步幅 1 / 2 / 4 / 8'), value: `${pxs(STRIDES.map(s => sw[s].argmax))} px` },
            { label: pick(lang, 'Taylor step at stride 1 / 2 / 4 / 8', 'Taylor 步驟，步幅 1 / 2 / 4 / 8'), value: `${pxs(STRIDES.map(s => sw[s].taylor))} px` },
            { label: pick(lang, 'Error of the keypoint in the grid', '格線圖中那個關鍵點的誤差'), value: `${fmt(exErr, 2)} px` },
          ],
        };
      },
    };
  },
};
