import type { LabDef } from './types.ts';
import { L, pick, fmt, num } from '../types.ts';
import { plotColors, startPlot, label, line } from './plot2d.ts';
import type { Gray } from '../../imagekit/types.ts';
import { otsuThreshold } from '../../imagekit/threshold.ts';
import { labelComponents } from '../../imagekit/label.ts';
import { edgeField, circlePoints, runSnake, snakeEnergy, fitCircle, type EdgeField, type SnakeResult } from '../../imagekit/snake.ts';
import type { Lang } from '../../../i18n/ui';

/**
 * The lab runs the snake of the IP 11 chapter (semi-implicit Kass-Witkin-Terzopoulos step, see src/lib/imagekit/snake.ts) on a
 * 144 x 144 crop of skimage.data.coins around one coin. Constants follow the chapter's figure code (dip_ch11.py): n = 100 points,
 * gamma = 1, kappa = 4, alpha = beta (the smoothness slider; the chapter's defaults are 0.05) and e = |grad(G_sigma * f)| scaled to 1.
 */
export const N_POINTS = 100;
export const GAMMA = 1;
export const KAPPA = 4;
/** The snake has converged when no point moves farther than this (pixels) in one iteration. */
export const TOL = 0.05;
export const MAX_ITER = 1500;
/** The start circle is centred on this pixel, the middle of the 144 x 144 crop. */
export const CENTER = 72;
export const DEFAULTS = { radius: 40, smooth: 0.05, sigma: 3 };
export const RADIUS_RANGE = { min: 8, max: 64 };
/** A snake counts as having found the coin when its mean distance to the coin boundary is at most this many pixels. */
export const FOUND_PX = 2;
/** Iterations at which an intermediate contour is drawn (when the run lasts that long). */
export const SNAPSHOTS = [10, 30, 100, 300];
/** A final point counts as caught by a neighbouring coin when it lies within this distance of that coin's boundary. */
export const NEAR_PX = 3;

export interface CoinTruth {
  w: number; h: number;
  /** Circle fitted to the boundary pixels of the coin at the image centre (Otsu mask, holes filled). */
  cx: number; cy: number; r: number;
  threshold: number; area: number; boundary: number;
  /** Boundary pixels (x, y) of the other coins in the crop (components of the same filled Otsu mask). */
  others: { x: number; y: number }[];
}

/** Background pixels (4-connected) that do not touch the border are holes: they are filled. */
export function fillHoles(mask: Uint8Array, w: number, h: number): Uint8Array {
  const inv = new Uint8Array(w * h);
  for (let i = 0; i < inv.length; i++) inv[i] = mask[i] ? 0 : 1;
  const { labels, count } = labelComponents(inv, w, h, 4), touches = new Uint8Array(count + 1);
  for (let x = 0; x < w; x++) { touches[labels[x]] = 1; touches[labels[(h - 1) * w + x]] = 1; }
  for (let y = 0; y < h; y++) { touches[labels[y * w]] = 1; touches[labels[y * w + w - 1]] = 1; }
  const out = Uint8Array.from(mask);
  for (let i = 0; i < out.length; i++) if (labels[i] && !touches[labels[i]]) out[i] = 1;
  return out;
}
/** Pixels of the component that have a 4-neighbour outside it (the image border counts as outside). */
function boundaryOf(inside: (i: number) => boolean, w: number, h: number): { x: number; y: number }[] {
  const pts: { x: number; y: number }[] = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    if (!inside(i)) continue;
    if (x === 0 || y === 0 || x === w - 1 || y === h - 1 || !inside(i - 1) || !inside(i + 1) || !inside(i - w) || !inside(i + w)) pts.push({ x, y });
  }
  return pts;
}
/** The coin's "true" boundary: Otsu mask of the crop, holes filled, the 8-connected component at the centre, least-squares circle. */
export function coinTruth(image: Gray): CoinTruth {
  const { w, h } = image, threshold = otsuThreshold(image.data), mask = new Uint8Array(w * h);
  for (let i = 0; i < mask.length; i++) mask[i] = image.data[i] > threshold ? 1 : 0;
  const filled = fillHoles(mask, w, h), { labels, areas } = labelComponents(filled, w, h, 8), mine = labels[CENTER * w + CENTER];
  if (!mine) throw new Error('The centre of the crop is not on a coin');
  const own = boundaryOf(i => labels[i] === mine, w, h), others = boundaryOf(i => labels[i] > 0 && labels[i] !== mine, w, h);
  const c = fitCircle(own.map(p => p.x), own.map(p => p.y));
  return { w, h, ...c, threshold, area: areas[mine - 1], boundary: own.length, others };
}

export interface Params { radius: number; smooth: number; sigma: number }
export type Verdict = 'found' | 'outside' | 'inside';
export interface Run {
  params: Params; result: SnakeResult;
  start: { x: Float64Array; y: Float64Array };
  snapshots: { iteration: number; x: Float64Array; y: Float64Array }[];
  /** Mean distance of the points to the true coin boundary, in pixels, after every iteration (index 0 = the start). */
  distanceTrace: Float64Array;
  distance: number;
  /** Mean distance of the final points from their own centroid (the snake's own mean radius). */
  ownRadius: number;
  /** E_int + E_ext after every iteration, and the energy of the fitted coin circle (same alpha, beta, kappa, n points). */
  energy: Float64Array;
  ringEnergy: number;
  verdict: Verdict;
  /** Final points within NEAR_PX of the boundary of another coin. */
  nearNeighbours: number;
}
export const meanDistance = (x: ArrayLike<number>, y: ArrayLike<number>, t: CoinTruth) => {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += Math.abs(Math.hypot(x[i] - t.cx, y[i] - t.cy) - t.r);
  return s / x.length;
};
export function ownRadiusOf(x: ArrayLike<number>, y: ArrayLike<number>): number {
  let mx = 0, my = 0;
  for (let i = 0; i < x.length; i++) { mx += x[i]; my += y[i]; }
  mx /= x.length; my /= x.length;
  let s = 0;
  for (let i = 0; i < x.length; i++) s += Math.hypot(x[i] - mx, y[i] - my);
  return s / x.length;
}
export function nearNeighbourCount(t: CoinTruth, x: ArrayLike<number>, y: ArrayLike<number>): number {
  let count = 0;
  for (let i = 0; i < x.length; i++) {
    let near = false;
    for (const p of t.others) if (Math.hypot(x[i] - p.x, y[i] - p.y) <= NEAR_PX) { near = true; break; }
    if (near) count++;
  }
  return count;
}
export function verdictOf(distance: number, ownRadius: number, t: CoinTruth): Verdict {
  return distance <= FOUND_PX ? 'found' : ownRadius < t.r ? 'inside' : 'outside';
}

const fields = new WeakMap<Gray, Map<number, EdgeField>>();
/** Edge map and force field of the image for one sigma (cached per image object and sigma). */
export function fieldFor(image: Gray, sigma: number): EdgeField {
  let perImage = fields.get(image);
  if (!perImage) { perImage = new Map(); fields.set(image, perImage); }
  let f = perImage.get(sigma);
  if (!f) { f = edgeField(image.data, image.w, image.h, sigma); perImage.set(sigma, f); }
  return f;
}
export function runContour(field: EdgeField, truth: CoinTruth, p: Params): Run {
  const start = circlePoints(CENTER, CENTER, p.radius, N_POINTS), distanceTrace: number[] = [], snapshots: Run['snapshots'] = [];
  const result = runSnake(field, start.x, start.y, {
    alpha: p.smooth, beta: p.smooth, gamma: GAMMA, kappa: KAPPA, tol: TOL, maxIter: MAX_ITER,
    observe(it, x, y) {
      distanceTrace.push(meanDistance(x, y, truth));
      if (SNAPSHOTS.includes(it)) snapshots.push({ iteration: it, x: Float64Array.from(x), y: Float64Array.from(y) });
    },
  });
  const energy = new Float64Array(result.internal.length);
  for (let i = 0; i < energy.length; i++) energy[i] = result.internal[i] + result.external[i];
  const ring = circlePoints(truth.cx, truth.cy, truth.r, N_POINTS), re = snakeEnergy(field, ring.x, ring.y, p.smooth, p.smooth, KAPPA);
  const distance = meanDistance(result.x, result.y, truth), own = ownRadiusOf(result.x, result.y);
  return {
    params: p, result, start, snapshots, distanceTrace: Float64Array.from(distanceTrace), distance, ownRadius: own, energy,
    ringEnergy: re.internal + re.external, verdict: verdictOf(distance, own, truth), nearNeighbours: nearNeighbourCount(truth, result.x, result.y),
  };
}
/** Run from the lab's three controls. */
export function runFromState(image: Gray, truth: CoinTruth, p: Params): Run { return runContour(fieldFor(image, p.sigma), truth, p); }

// ---------------------------------------------------------------------------------------------- text (pure, so tests can read it)

const VERDICT_WORD: Record<Verdict, { en: string; zh: string }> = {
  found: { en: 'Found the coin', zh: '找到硬幣' },
  outside: { en: 'Stopped outside the coin', zh: '停在硬幣外面' },
  inside: { en: 'Stopped inside the coin', zh: '停在硬幣裡面' },
};
export const verdictLabel = (lang: Lang, v: Verdict) => pick(lang, VERDICT_WORD[v].en, VERDICT_WORD[v].zh);
const px = (n: number) => `${fmt(n, 1)} px`;

export function describe(lang: Lang, run: Run, truth: CoinTruth): string {
  const d = px(run.distance), it = run.result.iterations, e = fmt(run.energy[run.energy.length - 1], 1), ring = fmt(run.ringEnergy, 1);
  let text: string;
  if (run.verdict === 'found') text = pick(lang,
    `The snake found the coin: after ${it} iterations its points lie ${d} from the coin boundary on average, on the rim where the edge map is strongest. Its energy (${e}) is as low as that of a circle on the coin (${ring}) or lower.`,
    `蛇模型找到了硬幣：經過 ${it} 次迭代，它的點平均距離硬幣邊界 ${d}，落在邊緣圖最強的邊緣上。它的能量（${e}）與硬幣上一個圓的能量（${ring}）相當或更低。`);
  else if (run.verdict === 'outside') {
    const caught = run.nearNeighbours >= 30;
    text = pick(lang,
      `The snake stopped outside the coin, ${d} from its boundary on average. ${caught ? `${run.nearNeighbours} of its ${N_POINTS} points lie within ${NEAR_PX} px of a neighbouring coin's boundary: other edges caught it. ` : 'Other edges or flat background hold it there. '}${run.energy[run.energy.length - 1] > run.ringEnergy ? `Its energy (${e}) is higher than the energy of a circle on the coin (${ring}), so this is a local minimum, not the best answer.` : `Its energy is ${e}.`}`,
      `蛇模型停在硬幣外面，平均距離硬幣邊界 ${d}。${caught ? `它的 ${N_POINTS} 個點中有 ${run.nearNeighbours} 個離鄰近硬幣的邊界不到 ${NEAR_PX} 像素：其他邊緣把它攔住了。` : '其他邊緣或平坦的背景把它留在那裡。'}${run.energy[run.energy.length - 1] > run.ringEnergy ? `它的能量（${e}）高於硬幣上一個圓的能量（${ring}），所以這是局部極小，不是最好的答案。` : `它的能量是 ${e}。`}`);
  } else text = pick(lang,
    `The snake stopped inside the coin (its own mean radius is ${px(run.ownRadius)}, the coin's is ${px(truth.r)}), ${d} from the boundary. Tension only pulls a snake inwards and this lab has no balloon force, so it settled on the coin's own relief or shrank to a small curve.`,
    `蛇模型停在硬幣裡面（它自己的平均半徑是 ${px(run.ownRadius)}，硬幣的是 ${px(truth.r)}），距離邊界 ${d}。張力只會把蛇往內拉，而本實驗沒有氣球力，所以它停在硬幣自己的浮雕上，或縮成一小圈。`);
  if (!run.result.converged) text += pick(lang,
    ` It did not converge: after ${MAX_ITER} iterations it still moves by up to ${px(run.result.lastMove)} per step.`,
    ` 它沒有收斂：經過 ${MAX_ITER} 次迭代，每一步仍會移動最多 ${px(run.result.lastMove)}。`);
  return text;
}

// ---------------------------------------------------------------------------------------------- drawing (DOM)

export const PANE_PX = 360;
const SCALE = PANE_PX / 144;
const at = (v: number) => (v + 0.5) * SCALE;

function stroke(ctx: CanvasRenderingContext2D, color: string, width: number, dash: number[] = [], halo = 'rgba(0,0,0,0.65)') {
  ctx.lineJoin = 'round';
  if (halo) { ctx.setLineDash(dash); ctx.lineWidth = width + 2.4; ctx.strokeStyle = halo; ctx.stroke(); }
  ctx.setLineDash(dash); ctx.lineWidth = width; ctx.strokeStyle = color; ctx.stroke(); ctx.setLineDash([]);
}
function closedPath(ctx: CanvasRenderingContext2D, x: ArrayLike<number>, y: ArrayLike<number>) {
  ctx.beginPath();
  for (let i = 0; i < x.length; i++) { if (i) ctx.lineTo(at(x[i]), at(y[i])); else ctx.moveTo(at(x[i]), at(y[i])); }
  ctx.closePath();
}
function ringPath(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) { ctx.beginPath(); ctx.arc(at(cx), at(cy), r * SCALE, 0, 2 * Math.PI); }

const grayLayers = new WeakMap<Gray, HTMLCanvasElement>();
function grayLayer(image: Gray): HTMLCanvasElement {
  const hit = grayLayers.get(image);
  if (hit) return hit;
  const c = document.createElement('canvas'); c.width = image.w; c.height = image.h;
  const ctx = c.getContext('2d')!, im = ctx.createImageData(image.w, image.h);
  for (let i = 0; i < image.w * image.h; i++) { const v = Math.round(255 * image.data[i]); im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = v; im.data[4 * i + 3] = 255; }
  ctx.putImageData(im, 0, 0);
  grayLayers.set(image, c);
  return c;
}
function drawScene(canvas: HTMLCanvasElement, image: Gray, run: Run, truth: CoinTruth) {
  canvas.width = PANE_PX; canvas.height = PANE_PX;
  const ctx = canvas.getContext('2d')!;
  ctx.imageSmoothingEnabled = true; ctx.drawImage(grayLayer(image), 0, 0, PANE_PX, PANE_PX);
  ringPath(ctx, truth.cx, truth.cy, truth.r); stroke(ctx, '#27d3ff', 1.8, [5, 4]);
  closedPath(ctx, run.start.x, run.start.y); stroke(ctx, '#ffffff', 1.8, [6, 4]);
  for (const s of run.snapshots) { closedPath(ctx, s.x, s.y); stroke(ctx, 'rgba(255,160,60,0.9)', 1.2, [], 'rgba(0,0,0,0.35)'); }
  closedPath(ctx, run.result.x, run.result.y); stroke(ctx, '#ffd400', 2.6);
}
let edgeLayer: HTMLCanvasElement | undefined, edgeKey: EdgeField | undefined;
function edgeImage(field: EdgeField): HTMLCanvasElement {
  if (edgeLayer && edgeKey === field) return edgeLayer;
  const c = document.createElement('canvas'); c.width = field.w; c.height = field.h;
  const ctx = c.getContext('2d')!, im = ctx.createImageData(field.w, field.h);
  for (let i = 0; i < field.w * field.h; i++) {
    const t = Math.pow(field.e[i], 0.8), r = 255 + (24 - 255) * t, g = 255 + (62 - 255) * t, b = 255 + (120 - 255) * t;
    im.data[4 * i] = r; im.data[4 * i + 1] = g; im.data[4 * i + 2] = b; im.data[4 * i + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
  edgeLayer = c; edgeKey = field;
  return c;
}
function drawEdge(canvas: HTMLCanvasElement, field: EdgeField, run: Run) {
  canvas.width = PANE_PX; canvas.height = PANE_PX;
  const ctx = canvas.getContext('2d')!;
  ctx.imageSmoothingEnabled = true; ctx.drawImage(edgeImage(field), 0, 0, PANE_PX, PANE_PX);
  closedPath(ctx, run.start.x, run.start.y); stroke(ctx, '#333333', 1.6, [6, 4], 'rgba(255,255,255,0.8)');
  closedPath(ctx, run.result.x, run.result.y); stroke(ctx, '#e0561e', 2.6, [], 'rgba(255,255,255,0.85)');
}
const niceStep = (span: number, ticks: number) => {
  const raw = span / ticks, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag;
};
interface Frame { ox: number; oy: number; w: number; h: number }
function axes(ctx: CanvasRenderingContext2D, f: Frame, x0: number, x1: number, y0: number, y1: number, c: ReturnType<typeof plotColors>, fmtY: (v: number) => string) {
  ctx.lineWidth = 1; ctx.strokeStyle = c.grid; ctx.strokeRect(f.ox, f.oy, f.w, f.h);
  const sy = niceStep(y1 - y0, 4), sx = niceStep(x1 - x0, 4);
  for (let v = Math.ceil(y0 / sy) * sy; v <= y1 + 1e-9; v += sy) {
    const y = f.oy + f.h - ((v - y0) / (y1 - y0)) * f.h;
    line(ctx, f.ox, y, f.ox + f.w, y, c.grid, 0.6);
    label(ctx, fmtY(v), f.ox - 6, y + 4, { size: 12, align: 'right', color: c.muted });
  }
  for (let v = Math.ceil(x0 / sx) * sx; v <= x1 + 1e-9; v += sx) {
    const x = f.ox + ((v - x0) / (x1 - x0)) * f.w;
    line(ctx, x, f.oy + f.h, x, f.oy + f.h + 4, c.muted, 1);
    label(ctx, String(Math.round(v)), x, f.oy + f.h + 18, { size: 12, align: 'center', color: c.muted });
  }
}
function polyline(ctx: CanvasRenderingContext2D, f: Frame, data: ArrayLike<number>, x1: number, y0: number, y1: number, color: string, width = 2.4) {
  ctx.beginPath();
  const n = data.length, step = Math.max(1, Math.floor(n / 400));
  for (let i = 0; i < n; i += step) {
    const x = f.ox + (i / x1) * f.w, y = f.oy + f.h - ((data[i] - y0) / (y1 - y0)) * f.h;
    if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
  }
  const last = n - 1, lx = f.ox + (last / x1) * f.w, ly = f.oy + f.h - ((data[last] - y0) / (y1 - y0)) * f.h;
  ctx.lineTo(lx, ly);
  ctx.lineJoin = 'round'; ctx.lineWidth = width; ctx.strokeStyle = color; ctx.stroke();
  return { x: lx, y: ly };
}
function drawEnergy(canvas: HTMLCanvasElement, lang: Lang, run: Run) {
  const c = plotColors(canvas), ctx = startPlot(canvas, PANE_PX, PANE_PX, c), f: Frame = { ox: 62, oy: 40, w: 280, h: 250 };
  label(ctx, pick(lang, 'Contour energy at each iteration', '每次迭代的輪廓能量'), 14, 22, { bold: true, color: c.ink });
  const e = run.energy, last = e.length - 1;
  let lo = Math.min(...e, run.ringEnergy), hi = Math.max(...e, run.ringEnergy);
  const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
  const x1 = Math.max(last, 10);
  axes(ctx, f, 0, x1, lo, hi, c, v => fmt(v, Math.abs(v) >= 100 ? 0 : 1));
  const yRing = f.oy + f.h - ((run.ringEnergy - lo) / (hi - lo)) * f.h;
  line(ctx, f.ox, yRing, f.ox + f.w, yRing, c.b, 2, [6, 4]);
  const end = polyline(ctx, f, e, x1, lo, hi, c.a);
  ctx.beginPath(); ctx.arc(end.x, end.y, 5, 0, 2 * Math.PI); ctx.fillStyle = c.a; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = c.bg; ctx.stroke();
  label(ctx, pick(lang, 'iteration', '迭代次數'), f.ox + f.w, f.oy + f.h + 36, { size: 13, align: 'right', color: c.ink });
  label(ctx, pick(lang, 'orange dashed: a circle on the coin', '橘色虛線：硬幣上的圓'), 14, 336, { size: 13, color: c.b });
  label(ctx, pick(lang, 'blue: the snake (lower is better)', '藍色：蛇模型（越低越好）'), 14, 354, { size: 13, color: c.a });
}
function drawDistance(canvas: HTMLCanvasElement, lang: Lang, run: Run) {
  const c = plotColors(canvas), ctx = startPlot(canvas, PANE_PX, PANE_PX, c), f: Frame = { ox: 62, oy: 40, w: 280, h: 250 };
  label(ctx, pick(lang, 'Distance to the coin boundary', '到硬幣邊界的距離'), 14, 22, { bold: true, color: c.ink });
  const d = run.distanceTrace, last = d.length - 1, hi = Math.max(5, Math.ceil(Math.max(...d) / 5) * 5), x1 = Math.max(last, 10);
  axes(ctx, f, 0, x1, 0, hi, c, v => fmt(v, 0));
  const yFound = f.oy + f.h - (FOUND_PX / hi) * f.h;
  line(ctx, f.ox, yFound, f.ox + f.w, yFound, c.b, 2, [6, 4]);
  const end = polyline(ctx, f, d, x1, 0, hi, c.a);
  ctx.beginPath(); ctx.arc(end.x, end.y, 5, 0, 2 * Math.PI); ctx.fillStyle = c.a; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = c.bg; ctx.stroke();
  label(ctx, pick(lang, 'iteration', '迭代次數'), f.ox + f.w, f.oy + f.h + 36, { size: 13, align: 'right', color: c.ink });
  label(ctx, pick(lang, 'pixels, mean over the 100 points', '單位：像素，100 個點的平均'), 14, 336, { size: 13, color: c.a });
  label(ctx, pick(lang, `orange dashed: ${FOUND_PX} px, “found”`, `橘色虛線：${FOUND_PX} 像素，視為「找到」`), 14, 354, { size: 13, color: c.b });
}

/** A click or drag on an image pane sets the start radius so that the circle passes through the pointer (the circle stays on the coin). */
function attachRadius(canvases: Record<string, HTMLCanvasElement>, setControl: (key: string, value: string | number) => void) {
  for (const pane of ['image', 'edge']) {
    const cv = canvases[pane];
    if (!cv) continue;
    cv.style.cursor = 'crosshair';
    const move = (e: PointerEvent) => {
      const r = cv.getBoundingClientRect(), x = ((e.clientX - r.left) / r.width) * 144 - 0.5, y = ((e.clientY - r.top) / r.height) * 144 - 0.5;
      setControl('radius', Math.max(RADIUS_RANGE.min, Math.min(RADIUS_RANGE.max, Math.round(Math.hypot(x - CENTER, y - CENTER)))));
    };
    cv.addEventListener('pointerdown', e => { move(e); if (e.pointerType !== 'touch') cv.setPointerCapture(e.pointerId); });
    cv.addEventListener('pointermove', e => { if (e.pointerType !== 'touch' && (e.buttons & 1)) move(e); });
  }
}

export const lab: LabDef = {
  kind: 'image', id: 'contour-energy', track: 'image-processing',
  title: L('IP 11 · Does the snake find the edge, or get stuck?', 'IP 11・蛇模型會找到邊緣，還是卡住？'),
  description: L('Run a real snake, the Kass–Witkin–Terzopoulos contour of IP 11, on a photograph of coins: start a circle around one coin and see how far from its rim the circle may begin before the snake settles in the wrong place.', '在硬幣的照片上執行真正的蛇模型（IP 11 的 Kass–Witkin–Terzopoulos 輪廓）：在一枚硬幣外畫一個起始圓，看看圓最遠可以離硬幣邊緣多遠，蛇模型才不會停在錯的地方。'),
  challenge: L(
    'The default circle has radius 40 and the coin about 23: the snake ends 17.7 px from the coin boundary, caught by the neighbouring coins. (1) Lower the start radius to 28: it finds the rim (0.5 px). (2) Back at radius 40, raise the smoothness to 1.0: now it finds the coin too (0.4 px). Why does a stiffer, more contracting snake reach the coin from farther away? (3) Keep smoothness 2.0 and start inside the coin at radius 12: the snake shrinks to a small curve (17.4 px). (4) With smoothness 0.05, a start at radius 12 reaches the rim at σ = 3 (1.1 px) but stays inside at σ = 2 (10.2 px). What does the blur remove?',
    '預設的圓半徑為 40，而硬幣約為 23：蛇模型最後離硬幣邊界 17.7 像素，被旁邊的硬幣攔住。(1) 把起始半徑降到 28：它找到了邊緣（0.5 像素）。(2) 回到半徑 40，把平滑度提高到 1.0：這次它也找到了硬幣（0.4 像素）。為什麼較硬、收縮較強的蛇可以從更遠的地方抵達硬幣？(3) 保持平滑度 2.0，從硬幣內部、半徑 12 起始：蛇縮成一小圈（17.4 像素）。(4) 平滑度 0.05 時，半徑 12 的起點在 σ = 3 能到達邊緣（1.1 像素），在 σ = 2 卻停在裡面（10.2 像素）。模糊濾掉了什麼？'),
  note: L(
    'Real image: a 144 × 144 crop of skimage.data.coins, centred on one coin and showing parts of the eight coins around it (credit below). The snake follows IP 11 and its figure code: a closed contour of 100 points, the semi-implicit step x(t) = (A + γI)⁻¹(γ x(t−1) + f) with A built from [β, −α−4β, 2α+6β, −α−4β, β], γ = 1, α = β = the smoothness slider, and the force f = κ∇e with κ = 4, where the edge map e = |∇(G_σ ∗ image)| is scaled to a maximum of 1 and sampled by bilinear interpolation. There is no resampling, no balloon force and no step cap; points stay inside the image. It stops when no point moves more than 0.05 px in one step, or after 1500 iterations. This is not scikit-image’s active_contour: that function builds the same matrix A, but takes a Sobel gradient of the image you pass in, a spline for the force and caps every step at one pixel. The true boundary is the circle fitted by least squares to the boundary pixels of the coin’s Otsu mask (holes filled); the coin is not a perfect circle, so a snake sitting on the rim is still about half a pixel from it. Distance is the mean over the 100 points of |distance from the circle’s centre − its radius|; “found” means at most 2 px. Energy is E = ½ Σ(α|v(i+1) − v(i)|² + β|v(i+1) − 2v(i) + v(i−1)|²) − κ Σ e(v(i)), in arbitrary units; the orange dashed line is E for the fitted circle itself. A click or drag on an image sets the start radius so that the circle passes through the pointer; the circle always stays centred on the coin. Click positions are rounded to whole pixels.',
    '真實影像：skimage.data.coins 的 144 × 144 裁切，以一枚硬幣為中心，並露出周圍八枚硬幣的一部分（來源見下方）。蛇模型依照 IP 11 與其圖形程式碼：100 個點的封閉輪廓，半隱式步驟 x(t) = (A + γI)⁻¹(γ x(t−1) + f)，其中 A 由 [β, −α−4β, 2α+6β, −α−4β, β] 建構，γ = 1，α = β = 平滑度滑桿，外力 f = κ∇e，κ = 4；邊緣圖 e = |∇(G_σ ∗ 影像)| 縮放到最大值為 1，並以雙線性內插取樣。沒有重新取樣、沒有氣球力、也沒有步長上限；點會停留在影像內。當一步之內沒有任何點移動超過 0.05 像素就停止，或在 1500 次迭代後停止。這不是 scikit-image 的 active_contour：該函式建立相同的矩陣 A，但對你傳入的影像取 Sobel 梯度，以樣條函數表示外力，並把每一步限制在一個像素以內。真實邊界是以最小平方法擬合硬幣 Otsu 遮罩（已填補孔洞）邊界像素得到的圓；硬幣並非完美的圓，所以停在邊緣上的蛇仍與它相差約半個像素。距離是 100 個點「到圓心的距離 − 圓半徑」的絕對值平均；「找到」指不超過 2 像素。能量為 E = ½ Σ(α|v(i+1) − v(i)|² + β|v(i+1) − 2v(i) + v(i−1)|²) − κ Σ e(v(i))，單位任意；橘色虛線是擬合出的圓本身的 E。在影像上按一下或拖曳，會設定起始半徑，使圓通過指標位置；圓始終以硬幣為中心。點擊位置會四捨五入到整數像素。'),
  related: ['image-processing/dip-11-active-contours'],
  controls: [
    { key: 'radius', label: L('Start radius of the circle (pixels; the coin’s is about 23)', '起始圓的半徑（像素；硬幣約為 23）'), kind: 'range', value: DEFAULTS.radius, min: RADIUS_RANGE.min, max: RADIUS_RANGE.max, step: 1 },
    { key: 'smooth', label: L('Smoothness α = β (tension and rigidity)', '平滑度 α = β（張力與剛性）'), kind: 'range', value: DEFAULTS.smooth, min: 0.05, max: 3, step: 0.05 },
    { key: 'sigma', label: L('Edge blur σ (pixels)', '邊緣模糊 σ（像素）'), kind: 'range', value: DEFAULTS.sigma, min: 1, max: 6, step: 0.5 },
  ],
  predict: {
    question: L('The start circle has radius 40 pixels; the coin has radius about 23. With the default settings, will the snake still find the coin?', '起始圓的半徑是 40 像素；硬幣的半徑約為 23。使用預設設定時，蛇模型還找得到硬幣嗎？'),
    options: [
      { value: 'found', label: L('Yes: it shrinks until it reaches the coin’s rim', '找得到：它一直收縮，直到碰到硬幣的邊緣') },
      { value: 'neighbours', label: L('No: it stops on the edges of the neighbouring coins', '找不到：它停在旁邊硬幣的邊緣上') },
      { value: 'collapse', label: L('No: it shrinks to a small curve inside the coin', '找不到：它縮成硬幣裡面的一小圈') },
    ],
    answer: 'neighbours',
    explain: L(
      'The snake does not find the coin. Tension pulls the circle inwards only a little before the edges of the neighbouring coins catch it: 74 of its 100 points end within 3 px of a neighbour’s boundary, and the mean distance to the coin boundary is 17.7 px. It has stopped moving (after 871 iterations), so it sits in a local minimum of the energy, but a poor one: its energy, −266.2, is higher than −283.4, the energy of a circle on the coin. A snake that does find the coin (start radius 28) reaches −298.9. Energies are in arbitrary units.',
      '蛇模型找不到硬幣。張力只把圓往內拉了一點，就被旁邊硬幣的邊緣攔住：它的 100 個點中有 74 個離鄰近硬幣的邊界不到 3 像素，到硬幣邊界的平均距離是 17.7 像素。它已經不再移動（經過 871 次迭代），所以停在能量的局部極小，但那是個不好的極小：它的能量 −266.2 高於硬幣上一個圓的能量 −283.4。真正找到硬幣的蛇（起始半徑 28）能達到 −298.9。能量單位任意。'),
  },
  assets: { 'coin-crop-144': 'gray' },
  paneMin: 340,
  panes: [
    { key: 'image', kind: 'canvas', label: L('Coins with the start circle (white dashed), snapshots at 10, 30, 100 and 300 iterations (orange), the final snake (yellow) and the true coin boundary (cyan dashed). Click or drag to set the start radius', '硬幣影像，含起始圓（白色虛線）、第 10、30、100、300 次迭代的快照（橘色）、最終的蛇（黃色）與真實硬幣邊界（青色虛線）。按一下或拖曳可設定起始半徑'), width: PANE_PX, height: PANE_PX },
    { key: 'edge', kind: 'canvas', label: L('Edge map e = |∇(G_σ ∗ image)| (dark blue: strong edge) with the start circle (dashed) and the final snake (orange)', '邊緣圖 e = |∇(G_σ ∗ 影像)|（深藍：強邊緣），含起始圓（虛線）與最終的蛇（橘色）'), width: PANE_PX, height: PANE_PX },
    { key: 'energy', kind: 'canvas', label: L('Snake energy against iteration', '蛇模型能量對迭代次數'), width: PANE_PX, height: PANE_PX },
    { key: 'distance', kind: 'canvas', label: L('Mean distance to the true coin boundary against iteration', '到真實硬幣邊界的平均距離對迭代次數'), width: PANE_PX, height: PANE_PX },
  ],
  sources: [
    { title: 'Michael Kass, Andrew Witkin and Demetri Terzopoulos, Snakes: Active contour models, International Journal of Computer Vision, 1988', url: 'https://doi.org/10.1007/BF00133570' },
    { title: 'Nobuyuki Otsu, A Threshold Selection Method from Gray-Level Histograms, IEEE Transactions on Systems, Man, and Cybernetics, 1979 (the coin’s true boundary)', url: 'https://doi.org/10.1109/TSMC.1979.4310076' },
    { title: 'scikit-image documentation, skimage.segmentation.active_contour (same matrix A; the lab’s step and force differ, see the note)', url: 'https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.active_contour' },
  ],
  create({ lang, images }) {
    const image = images['coin-crop-144'] as Gray, truth = coinTruth(image);
    return {
      attach: (canvases, setControl) => attachRadius(canvases, setControl),
      render(state, canvases) {
        const p: Params = { radius: Math.round(num(state, 'radius')), smooth: num(state, 'smooth'), sigma: num(state, 'sigma') };
        const field = fieldFor(image, p.sigma), run = runContour(field, truth, p);
        drawScene(canvases.image, image, run, truth);
        drawEdge(canvases.edge, field, run);
        drawEnergy(canvases.energy, lang, run);
        drawDistance(canvases.distance, lang, run);
        const finalE = run.energy[run.energy.length - 1];
        return {
          summary: describe(lang, run, truth),
          metrics: [
            { label: pick(lang, 'Result', '結果'), value: verdictLabel(lang, run.verdict) },
            { label: pick(lang, 'Mean distance to the coin boundary', '到硬幣邊界的平均距離'), value: px(run.distance) },
            { label: pick(lang, 'Converged', '是否收斂'), value: run.result.converged ? pick(lang, `Yes, after ${run.result.iterations} iterations`, `是，經過 ${run.result.iterations} 次迭代`) : pick(lang, `No, still moving after ${MAX_ITER} iterations`, `否，經過 ${MAX_ITER} 次迭代仍在移動`) },
            { label: pick(lang, 'Final energy (a circle on the coin)', '最終能量（硬幣上的圓）'), value: `${fmt(finalE, 1)} (${fmt(run.ringEnergy, 1)})` },
          ],
        };
      },
    };
  },
};
