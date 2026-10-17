import type { LabDef } from './types.ts';
import { L, pick, fmt, num } from '../types.ts';
import { plotColors, startPlot, label, line } from './plot2d.ts';
import type { Gray } from '../../imagekit/types.ts';
import { reflectIndex } from '../../imagekit/filter.ts';
import { drawGray } from '../../imagekit/draw.ts';
import { gaussianGradient, structureTensorMaps, harrisResponse, eigenSym2, windowWeights } from '../../imagekit/harris.ts';
import { attachProbe } from './pointer.ts';
import type { Lang } from '../../../i18n/ui';

/** Constants shared with the IP 12 chapter code: Gaussian derivative sigma_d = 1 pixel, Harris k = 0.05. */
export const SIGMA_D = 1;
export const HARRIS_K = 0.05;
/** Gradients are shown in gray levels per pixel (0 to 255 scale), so eigenvalues are in squared gray levels per pixel. */
export const GRAY = 255;
const LAMBDA_SCALE = GRAY * GRAY;
/** An eigenvalue is called "large" when it is at least this fraction of the largest first eigenvalue anywhere in the image. */
export const LARGE = 0.05;
/** Axis range of the gradient scatter (the largest |Ix| or |Iy| on this image is 74 gray levels per pixel). */
export const AXIS = 80;

export const DEFAULT_PROBE = { x: 89, y: 103, win: 9 };
/** The three locations of the predict question: sky, a straight edge and a true corner (verified in the tests). */
export const MARKS = [
  { key: 'A', x: 200, y: 20, kind: 'flat' },
  { key: 'B', x: 80, y: 52, kind: 'corner' },
  { key: 'C', x: 30, y: 80, kind: 'edge' },
] as const;

export type Verdict = 'flat' | 'edge' | 'corner';

export interface Gradients { w: number; h: number; ix: Float32Array; iy: Float32Array; image: Gray }
export interface HarrisField {
  w: number; h: number; win: number; radius: number; sigma: number;
  xx: Float32Array; yy: Float32Array; xy: Float32Array; R: Float32Array; l1: Float32Array; l2: Float32Array;
  rmax: number; l1max: number; weights: Float64Array;
}
/** Window of `win` x `win` pixels (win odd) with Gaussian weights of sigma = win / 6. */
export const windowSigma = (win: number) => win / 6;

export function gradientsOf(image: Gray): Gradients {
  const { ix, iy } = gaussianGradient(image.data, image.w, image.h, SIGMA_D);
  return { w: image.w, h: image.h, ix, iy, image };
}
export function harrisField(g: Gradients, win: number): HarrisField {
  const radius = (win - 1) >> 1, sigma = windowSigma(win), weights = windowWeights(radius, sigma);
  const { xx, yy, xy } = structureTensorMaps(g.ix, g.iy, g.w, g.h, weights);
  const R = harrisResponse(xx, yy, xy, HARRIS_K), n = g.w * g.h, l1 = new Float32Array(n), l2 = new Float32Array(n);
  let rmax = 0, l1max = 0;
  for (let i = 0; i < n; i++) {
    const e = eigenSym2(xx[i], yy[i], xy[i]);
    l1[i] = e.l1; l2[i] = e.l2;
    if (R[i] > rmax) rmax = R[i];
    if (e.l1 > l1max) l1max = e.l1;
  }
  return { w: g.w, h: g.h, win, radius, sigma, xx, yy, xy, R, l1, l2, rmax, l1max, weights };
}
/** Flat: no large eigenvalue. Edge: only the first is large. Corner: both are large (the chapter's rule for lambda1 and lambda2). */
export function verdictOf(l1: number, l2: number, l1max: number): Verdict {
  const big = LARGE * l1max;
  return l1 < big ? 'flat' : l2 < big ? 'edge' : 'corner';
}
/** Share of image pixels whose window is flat, an edge or a corner under `verdictOf`. */
export function verdictShares(f: HarrisField): Record<Verdict, number> {
  const n = f.w * f.h, c = { flat: 0, edge: 0, corner: 0 };
  for (let i = 0; i < n; i++) c[verdictOf(f.l1[i], f.l2[i], f.l1max)]++;
  return { flat: c.flat / n, edge: c.edge / n, corner: c.corner / n };
}

export interface Probe {
  x: number; y: number; win: number;
  /** One gradient vector per window pixel, in gray levels per pixel, with its Gaussian weight. */
  vectors: { dx: number; dy: number; ix: number; iy: number; weight: number; value: number }[];
  /** Tensor entries from the field's maps (squared gray levels per pixel). */
  xx: number; yy: number; xy: number;
  l1: number; l2: number; angle: number; R: number; rRel: number; verdict: Verdict; ratio: number;
}
export function probeAt(g: Gradients, f: HarrisField, x: number, y: number): Probe {
  const { w, h } = g, r = f.radius, vectors: Probe['vectors'] = [];
  for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
    const i = reflectIndex(y + dy, h) * w + reflectIndex(x + dx, w);
    vectors.push({ dx, dy, ix: g.ix[i] * GRAY, iy: g.iy[i] * GRAY, weight: f.weights[dx + r] * f.weights[dy + r], value: g.image.data[i] });
  }
  const i = y * w + x, e = eigenSym2(f.xx[i], f.yy[i], f.xy[i]);
  return {
    x, y, win: f.win, vectors,
    xx: f.xx[i] * LAMBDA_SCALE, yy: f.yy[i] * LAMBDA_SCALE, xy: f.xy[i] * LAMBDA_SCALE,
    l1: e.l1 * LAMBDA_SCALE, l2: e.l2 * LAMBDA_SCALE, angle: e.angle, R: f.R[i],
    rRel: f.rmax > 0 ? f.R[i] / f.rmax : 0, verdict: verdictOf(e.l1, e.l2, f.l1max), ratio: e.l1 > 0 ? e.l2 / e.l1 : 0,
  };
}
/** The same tensor computed directly from the probe's scattered gradient vectors: M = sum of weight * g g^T. */
export function tensorFromVectors(p: Probe): { xx: number; yy: number; xy: number } {
  let xx = 0, yy = 0, xy = 0;
  for (const v of p.vectors) { xx += v.weight * v.ix * v.ix; yy += v.weight * v.iy * v.iy; xy += v.weight * v.ix * v.iy; }
  return { xx, yy, xy };
}

const VERDICT_WORD: Record<Verdict, { en: string; zh: string }> = {
  flat: { en: 'flat', zh: '平坦' }, edge: { en: 'edge', zh: '邊緣' }, corner: { en: 'corner', zh: '角點' },
};
export const verdictLabel = (lang: Lang, v: Verdict) => pick(lang, VERDICT_WORD[v].en, VERDICT_WORD[v].zh);

const f0 = (n: number) => fmt(n, Math.abs(n) >= 100 ? 0 : Math.abs(n) >= 10 ? 1 : 2);
/** Plain-language reading of the current probe (pure, so tests can check it). */
export function describe(lang: Lang, p: Probe, l1max: number): string {
  const big = LARGE * l1max * LAMBDA_SCALE, where = pick(lang, `At (${p.x}, ${p.y}) with a ${p.win}-pixel window`, `在 (${p.x}, ${p.y})、視窗 ${p.win} 像素`);
  if (p.verdict === 'flat') return pick(lang,
    `${where}, the gradient dots huddle at the origin: λ1 = ${f0(p.l1)} is far below ${f0(big)}, a twentieth of the largest λ1 in the image, so nothing changes under a small shift. Verdict: flat.`,
    `${where}，梯度點擠在原點附近：λ1 = ${f0(p.l1)}，遠小於 ${f0(big)}（影像中最大 λ1 的二十分之一），所以視窗稍微移動也看不出變化。判定：平坦。`);
  if (p.verdict === 'edge') return pick(lang,
    `${where}, the gradient dots lie along one line through the origin: λ1 = ${f0(p.l1)} is large but λ2 = ${f0(p.l2)} is tiny (the cutoff is ${f0(big)}), so the image changes in only one direction. Verdict: edge.`,
    `${where}，梯度點排成一條通過原點的直線：λ1 = ${f0(p.l1)} 很大，λ2 = ${f0(p.l2)} 卻很小（門檻為 ${f0(big)}），所以影像只在一個方向上變化。判定：邊緣。`);
  return pick(lang,
    `${where}, the gradient dots spread in two directions: λ1 = ${f0(p.l1)} and λ2 = ${f0(p.l2)} both exceed the cutoff ${f0(big)}, so any shift changes the window. Harris response R is ${fmt(100 * p.rRel, 0)}% of the largest R in the image. Verdict: corner.`,
    `${where}，梯度點分散在兩個方向：λ1 = ${f0(p.l1)} 與 λ2 = ${f0(p.l2)} 都超過門檻 ${f0(big)}，所以往任何方向移動都會改變視窗。Harris 響應 R 是影像中最大 R 的 ${fmt(100 * p.rRel, 0)}%。判定：角點。`);
}

// ---------------------------------------------------------------------------------------------- drawing (DOM)

function overlayMarks(ctx: CanvasRenderingContext2D) {
  ctx.font = 'bold 14px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  for (const m of MARKS) {
    for (const [lw, col] of [[3.5, 'rgba(0,0,0,0.85)'], [1.6, '#27d3ff']] as const) { ctx.lineWidth = lw; ctx.strokeStyle = col; ctx.beginPath(); ctx.arc(m.x + 0.5, m.y + 0.5, 5, 0, 2 * Math.PI); ctx.stroke(); }
    ctx.lineWidth = 3.5; ctx.strokeStyle = 'rgba(0,0,0,0.85)'; ctx.strokeText(m.key, m.x + 8, m.y - 6);
    ctx.fillStyle = '#27d3ff'; ctx.fillText(m.key, m.x + 8, m.y - 6);
  }
}
function overlayProbe(ctx: CanvasRenderingContext2D, x: number, y: number, r: number) {
  for (const [lw, col] of [[3.5, 'rgba(0,0,0,0.85)'], [1.5, '#ffd400']] as const) {
    ctx.lineWidth = lw; ctx.strokeStyle = col; ctx.strokeRect(x - r, y - r, 2 * r + 1, 2 * r + 1);
  }
}
function drawHeat(canvas: HTMLCanvasElement, f: HarrisField) {
  if (canvas.width !== f.w) canvas.width = f.w;
  if (canvas.height !== f.h) canvas.height = f.h;
  const ctx = canvas.getContext('2d')!, im = ctx.createImageData(f.w, f.h), px = im.data;
  for (let i = 0; i < f.w * f.h; i++) {
    const v = f.rmax > 0 ? f.R[i] / f.rmax : 0, t = Math.min(1, Math.sqrt(Math.abs(v))), pos = v >= 0;
    const [r, g, b] = pos ? [214, 39, 40] : [36, 100, 194];
    px[4 * i] = Math.round(255 + (r - 255) * t); px[4 * i + 1] = Math.round(255 + (g - 255) * t); px[4 * i + 2] = Math.round(255 + (b - 255) * t); px[4 * i + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}
const WINDOW_PX = 252;
function drawWindow(canvas: HTMLCanvasElement, p: Probe) {
  if (canvas.width !== WINDOW_PX) canvas.width = WINDOW_PX;
  if (canvas.height !== WINDOW_PX) canvas.height = WINDOW_PX;
  canvas.style.imageRendering = 'pixelated';
  const ctx = canvas.getContext('2d')!, n = p.win, cell = WINDOW_PX / n, ref = 30;
  p.vectors.forEach((v, k) => {
    const cx = (k % n) * cell, cy = Math.floor(k / n) * cell, g = Math.round(v.value * 255);
    ctx.fillStyle = `rgb(${g},${g},${g})`; ctx.fillRect(cx, cy, cell + 0.6, cell + 0.6);
  });
  ctx.lineCap = 'round';
  p.vectors.forEach((v, k) => {
    const mag = Math.hypot(v.ix, v.iy);
    if (mag < 1) return;
    const cx = ((k % n) + 0.5) * cell, cy = (Math.floor(k / n) + 0.5) * cell, len = Math.min(1, mag / ref) * cell * 0.9, ux = v.ix / mag, uy = v.iy / mag;
    const x0 = cx - ux * len / 2, y0 = cy - uy * len / 2, x1 = cx + ux * len / 2, y1 = cy + uy * len / 2;
    ctx.strokeStyle = 'rgba(0,0,0,0.75)'; ctx.lineWidth = 3.4; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.strokeStyle = '#ff9a1f'; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.fillStyle = '#ff9a1f'; ctx.beginPath(); ctx.arc(x1, y1, 2.2, 0, 2 * Math.PI); ctx.fill();
  });
  ctx.strokeStyle = '#ffd400'; ctx.lineWidth = 3; ctx.strokeRect(1.5, 1.5, WINDOW_PX - 3, WINDOW_PX - 3);
}

// ---------------------------------------------------------------------------------------------- plots (canvas)

export const PLOT_PX = 360;
/** Density of all windows in the plane of (sqrt lambda1, sqrt lambda2), in gray levels per pixel, in 2.5-wide bins. */
export const CLOUD_BIN = 2.5;
export function cloudCounts(f: HarrisField): Uint32Array {
  const nb = Math.ceil(AXIS / CLOUD_BIN), counts = new Uint32Array(nb * nb);
  for (let i = 0; i < f.w * f.h; i++) {
    const a = Math.min(nb - 1, Math.floor(Math.sqrt(f.l1[i] * LAMBDA_SCALE) / CLOUD_BIN)), b = Math.min(nb - 1, Math.floor(Math.sqrt(f.l2[i] * LAMBDA_SCALE) / CLOUD_BIN));
    counts[b * nb + a]++;
  }
  return counts;
}
function drawScatter(canvas: HTMLCanvasElement, lang: Lang, p: Probe) {
  const c = plotColors(canvas), ctx = startPlot(canvas, PLOT_PX, PLOT_PX, c), cx = 180, cy = 176, half = 130, k = half / AXIS;
  label(ctx, pick(lang, 'Gradients in the window', '視窗內的梯度'), 14, 24, { bold: true, color: c.ink });
  ctx.lineWidth = 1; ctx.strokeStyle = c.grid; ctx.strokeRect(cx - half, cy - half, 2 * half, 2 * half);
  line(ctx, cx - half, cy, cx + half, cy, c.grid); line(ctx, cx, cy - half, cx, cy + half, c.grid);
  label(ctx, 'Ix', cx + half - 4, cy - 6, { align: 'right', color: c.ink }); label(ctx, 'Iy', cx + 6, cy - half + 16, { color: c.ink });
  label(ctx, `−${AXIS}`, cx - half, cy + half + 17, { size: 13, color: c.muted }); label(ctx, `${AXIS}`, cx + half, cy + half + 17, { size: 13, align: 'right', color: c.muted });
  const wmax = Math.max(...p.vectors.map(v => v.weight));
  ctx.fillStyle = c.a;
  for (const v of p.vectors) { ctx.globalAlpha = 0.22 + 0.7 * v.weight / wmax; ctx.beginPath(); ctx.arc(cx + v.ix * k, cy - v.iy * k, 3.6, 0, 2 * Math.PI); ctx.fill(); }
  ctx.globalAlpha = 1;
  // The ellipse with semi-axes sqrt(lambda1), sqrt(lambda2) along the eigenvectors: the footprint of the dots (M is their second-moment matrix).
  ctx.beginPath(); ctx.ellipse(cx, cy, Math.max(Math.sqrt(p.l1) * k, 1.5), Math.max(Math.sqrt(p.l2) * k, 1.5), -p.angle, 0, 2 * Math.PI);
  ctx.globalAlpha = 0.14; ctx.fillStyle = c.b; ctx.fill(); ctx.globalAlpha = 1; ctx.lineWidth = 2.6; ctx.strokeStyle = c.b; ctx.stroke();
  label(ctx, pick(lang, `Ellipse axes √λ1 = ${fmt(Math.sqrt(p.l1), 1)}, √λ2 = ${fmt(Math.sqrt(p.l2), 1)}`, `橢圓半軸 √λ1 = ${fmt(Math.sqrt(p.l1), 1)}，√λ2 = ${fmt(Math.sqrt(p.l2), 1)}`), cx, cy + half + 40, { align: 'center', color: c.b });
}
function drawPlane(canvas: HTMLCanvasElement, lang: Lang, p: Probe, f: HarrisField, counts: Uint32Array) {
  const c = plotColors(canvas), ctx = startPlot(canvas, PLOT_PX, PLOT_PX, c), ox = 62, oy = 308, side = 270, ku = side / AXIS, px = (v: number) => ox + v * ku, py = (v: number) => oy - v * ku;
  label(ctx, pick(lang, 'Eigenvalues vs all windows', '特徵值與所有視窗的比較'), 14, 24, { bold: true, color: c.ink });
  const nb = Math.ceil(AXIS / CLOUD_BIN), top = Math.log1p(counts.reduce((m, v) => Math.max(m, v), 0));
  ctx.fillStyle = c.muted;
  for (let b = 0; b < nb; b++) for (let a = 0; a < nb; a++) {
    const n = counts[b * nb + a];
    if (n) { ctx.globalAlpha = 0.15 + 0.7 * Math.log1p(n) / top; ctx.fillRect(ox + a * CLOUD_BIN * ku, oy - (b + 1) * CLOUD_BIN * ku, CLOUD_BIN * ku + 0.5, CLOUD_BIN * ku + 0.5); }
  }
  ctx.globalAlpha = 1; ctx.lineWidth = 1; ctx.strokeStyle = c.grid; ctx.strokeRect(ox, oy - side, side, side);
  const t = Math.sqrt(LARGE * f.l1max * LAMBDA_SCALE);
  line(ctx, px(t), oy, px(t), py(AXIS), c.muted, 1.2, [5, 4]); line(ctx, ox, py(t), px(AXIS), py(t), c.muted, 1.2, [5, 4]);
  const halo = { color: c.ink, halo: c.bg, bold: true, align: 'center' as const };
  label(ctx, pick(lang, 'flat', '平坦'), px(t / 2), oy - 6, { ...halo, size: 13 });
  label(ctx, pick(lang, 'edge', '邊緣'), px((t + AXIS) / 2), oy - 6, halo);
  label(ctx, pick(lang, 'corner', '角點'), px((t + AXIS) / 2), py((t + AXIS) / 2) - 30, halo);
  label(ctx, '√λ1', px(AXIS), oy + 20, { align: 'right', color: c.ink }); label(ctx, '√λ2', ox - 8, oy - side + 14, { align: 'right', color: c.ink });
  ctx.beginPath(); ctx.arc(px(Math.sqrt(p.l1)), py(Math.sqrt(p.l2)), 7, 0, 2 * Math.PI); ctx.fillStyle = c.b; ctx.fill(); ctx.lineWidth = 2.5; ctx.strokeStyle = c.bg; ctx.stroke();
  label(ctx, pick(lang, 'grey: all windows, dot: probe', '灰：所有視窗，圓點：探測點'), 14, 348, { size: 13, color: c.muted });
}

export const lab: LabDef = {
  kind: 'image', id: 'corner-tensor', track: 'image-processing',
  title: L('IP 12 · Probe a pixel: flat, edge or corner?', 'IP 12・探測像素：平坦、邊緣還是角點？'),
  description: L('Slide a small window over a real photograph and watch the window’s gradient vectors, its structure-tensor ellipse and the Harris response decide whether the spot is flat, an edge or a corner.', '在真實照片上移動一個小視窗，觀察視窗內的梯度向量、結構張量橢圓與 Harris 響應，如何判定該處是平坦、邊緣還是角點。'),
  challenge: L(
    'Place the probe on the three marked points with the sliders (A: x 200, y 20; B: x 80, y 52; C: x 30, y 80) and compare the dots: a tiny huddle at the origin, a cloud in two directions and a thin line. Then try the left tripod leg (x 122, y 232) and the right leg (x 183, y 220): the line turns, yet λ2/λ1 stays below 0.0025 for both, because both are edges. Finally raise the window from 9 to 25 pixels: the share of the image labelled corner grows from 1.4% to 8.9%. Why does a bigger window find more corners?',
    '用滑桿把探測點放到三個標記處（A：x 200, y 20；B：x 80, y 52；C：x 30, y 80），比較梯度點的樣子：擠在原點的小團、朝兩個方向散開的雲、細長的一條線。接著試試左側腳架腳（x 122, y 232）與右側腳（x 183, y 220）：直線的方向會轉動，但兩處的 λ2/λ1 都低於 0.0025，因為兩者都是邊緣。最後把視窗從 9 增加到 25 像素：被判為角點的影像面積比例從 1.4% 增加到 8.9%。為什麼較大的視窗會找到較多角點？'),
  note: L(
    'Real photograph: skimage.data.camera at 256 × 256 (see the credit below). Gradients are Gaussian derivatives with σ = 1 pixel, as in the IP 12 code. The window is W × W pixels with Gaussian weights of σ = W/6, so W = 9 gives the chapter’s integration scale σ = 1.5 (the chapter code truncates the Gaussian at 4σ instead of at the box). M is the weighted sum of the outer products of the gradient vectors, R = det M − 0.05 · tr(M)². Gradients are in gray levels per pixel (0 to 255) and eigenvalues in their squares. The flat, edge and corner labels use a cutoff chosen for this lab, not part of Harris and Stephens: an eigenvalue counts as large when it is at least 5% of the largest λ1 anywhere in the image at that window size. The Harris detector proper keeps local maxima of R above a threshold. The heat map shows R divided by its image maximum, compressed by a square root: red positive, blue negative, white near zero.',
    '真實照片：skimage.data.camera，縮為 256 × 256（來源見下方）。梯度是 σ = 1 像素的高斯導數，與 IP 12 的程式碼相同。視窗是 W × W 像素，以 σ = W/6 的高斯加權，所以 W = 9 對應章節中的積分尺度 σ = 1.5（章節程式碼把高斯截在 4σ，而不是視窗邊緣）。M 是梯度向量外積的加權和，R = det M − 0.05 · tr(M)²。梯度單位是每像素灰階值（0 到 255），特徵值單位是其平方。平坦、邊緣、角點的標籤使用本實驗自訂的門檻，並非 Harris 與 Stephens 原文的一部分：特徵值若不小於該視窗大小下影像中最大 λ1 的 5%，就視為「大」。真正的 Harris 偵測器則是取 R 超過門檻的局部最大值。熱圖顯示 R 除以影像中的最大值後再開平方：紅色為正、藍色為負、白色接近零。'),
  related: ['image-processing/dip-12-feature-extraction'],
  controls: [
    { key: 'x', label: L('Probe x (pixels from the left)', '探測點 x（距左邊的像素數）'), kind: 'range', value: DEFAULT_PROBE.x, min: 0, max: 255, step: 1 },
    { key: 'y', label: L('Probe y (pixels from the top)', '探測點 y（距上邊的像素數）'), kind: 'range', value: DEFAULT_PROBE.y, min: 0, max: 255, step: 1 },
    { key: 'win', label: L('Window size (pixels)', '視窗大小（像素）'), kind: 'range', value: DEFAULT_PROBE.win, min: 5, max: 25, step: 2 },
  ],
  predict: {
    question: L('Three places are marked on the photograph: A, B and C. Which one is a corner, a point where shifting the window in any direction changes what it sees?', '照片上標了三個位置：A、B、C。哪一個是角點，也就是把視窗往任何方向移動都會改變所見內容的位置？'),
    options: MARKS.map(m => ({ value: m.key, label: L(`${m.key} (x ${m.x}, y ${m.y})`, `${m.key}（x ${m.x}, y ${m.y}）`) })),
    answer: 'B',
    explain: L(
      'B is the corner: where the hair meets the dark coat against the sky, the gradients point in two directions. With the default 9-pixel window its tensor has λ1 = 1759 and λ2 = 360, both large. At C, a straight shoulder edge, all gradients point the same way: λ1 = 2206 but λ2 = 1.85, so only one direction changes. In the sky at A both eigenvalues are tiny (λ1 = 0.13). Eigenvalues are in squared gray levels per pixel.',
      'B 是角點：頭髮與深色外套在天空前相接之處，梯度指向兩個方向。使用預設的 9 像素視窗時，它的張量 λ1 = 1759、λ2 = 360，兩者都很大。C 是一段筆直的肩線邊緣，所有梯度指向同一方向：λ1 = 2206，但 λ2 = 1.85，所以只有一個方向會改變。A 在天空中，兩個特徵值都極小（λ1 = 0.13）。特徵值的單位是每像素灰階值的平方。'),
  },
  assets: { 'camera-256': 'gray' },
  paneMin: 250,
  panes: [
    { key: 'image', kind: 'canvas', label: L('Photograph with the probe window (yellow) and marks A, B, C. Click or drag to move the probe', '照片、探測視窗（黃框）與標記 A、B、C。按一下或拖曳可移動探測點'), width: 256, height: 256 },
    { key: 'window', kind: 'canvas', label: L('The window magnified, with a gradient arrow in each pixel (orange)', '放大後的視窗，每個像素上有一支梯度箭頭（橙色）'), width: WINDOW_PX, height: WINDOW_PX },
    { key: 'heat', kind: 'canvas', label: L('Harris response R over the whole image (red: corner, blue: edge, white: flat)', '整張影像的 Harris 響應 R（紅：角點，藍：邊緣，白：平坦）'), width: 256, height: 256 },
    { key: 'scatter', kind: 'canvas', label: L('One dot per window pixel: its gradient (Ix, Iy). The orange ellipse is the structure tensor', '視窗中每個像素一個點：它的梯度 (Ix, Iy)。橙色橢圓就是結構張量'), width: PLOT_PX, height: PLOT_PX },
    { key: 'plane', kind: 'canvas', label: L('The two eigenvalues of this window (dot) among all windows of the image (grey)', '這個視窗的兩個特徵值（圓點）與影像所有視窗（灰色）的比較'), width: PLOT_PX, height: PLOT_PX },
  ],
  sources: [
    { title: 'Chris Harris and Mike Stephens, A Combined Corner and Edge Detector, Proceedings of the Alvey Vision Conference, 1988', url: 'https://doi.org/10.5244/C.2.23' },
    { title: 'SciPy documentation, scipy.ndimage.gaussian_filter (Gaussian derivatives with reflected borders, as in the IP 12 code)', url: 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html' },
  ],
  create({ lang, images }) {
    const image = images['camera-256'] as Gray, grads = gradientsOf(image);
    let field: HarrisField | undefined, counts: Uint32Array = new Uint32Array(0);
    return {
      attach: (canvases, setControl) => attachProbe(canvases, ['image', 'heat'], setControl),
      render(state, canvases) {
        const win = num(state, 'win'), x = Math.max(0, Math.min(image.w - 1, Math.round(num(state, 'x')))), y = Math.max(0, Math.min(image.h - 1, Math.round(num(state, 'y'))));
        if (!field || field.win !== win) { field = harrisField(grads, win); counts = cloudCounts(field); }
        const p = probeAt(grads, field, x, y);
        drawGray(canvases.image, image);
        overlayMarks(canvases.image.getContext('2d')!);
        overlayProbe(canvases.image.getContext('2d')!, x, y, field.radius);
        drawHeat(canvases.heat, field);
        overlayMarks(canvases.heat.getContext('2d')!);
        overlayProbe(canvases.heat.getContext('2d')!, x, y, field.radius);
        drawWindow(canvases.window, p);
        drawScatter(canvases.scatter, lang, p);
        drawPlane(canvases.plane, lang, p, field, counts);
        const shares = verdictShares(field);
        return {
          summary: describe(lang, p, field.l1max),
          metrics: [
            { label: pick(lang, 'Verdict', '判定'), value: verdictLabel(lang, p.verdict) },
            { label: pick(lang, 'λ1 (large eigenvalue)', 'λ1（較大特徵值）'), value: f0(p.l1) },
            { label: pick(lang, 'λ2 (small eigenvalue)', 'λ2（較小特徵值）'), value: f0(p.l2) },
            { label: pick(lang, 'Harris R / largest R in image', 'Harris R ÷ 影像中最大的 R'), value: fmt(p.rRel, 2) },
            { label: pick(lang, 'Image labelled corner', '被判為角點的影像比例'), value: `${fmt(100 * shares.corner, 1)}%` },
          ],
        };
      },
    };
  },
};
