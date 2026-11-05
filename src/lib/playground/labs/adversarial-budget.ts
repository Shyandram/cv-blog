// Facts for the `adversarial-budget` lab (DLCV 14): a logistic-regression classifier trained here, in TypeScript, on synthetic noisy
// stripe images of 8 x 8, 16 x 16 and 32 x 32 pixels, attacked with FGSM (sign of the input gradient of the loss) and with random signs
// of the same size. Pure computation is exported so that scripts/tests/lab-adversarial.test.ts can check it.
import type { LabDef } from './types.ts';
import { L, pick, fmt, num } from '../types.ts';
import { plotColors, startPlot, label, line } from './plot2d.ts';
import { mulberry32, normalField } from '../../imagekit/rng.ts';
import type { Lang } from '../../../i18n/ui';

// ---------------------------------------------------------------------------------------------- data
/** Image sides offered by the lab; the number of pixel values is the side squared. */
export const SIDES = [8, 16, 32] as const;
/** Gaussian noise per pixel (on a 0..1 scale; 0.2 is 51 of 255 levels). */
export const SIGMA = 0.2;
/**
 * The two classes differ by a stripe pattern of total size ||s a||_2 = Z0 * SIGMA, spread over all d pixels, so the per-pixel
 * amplitude is s = Z0 * SIGMA / sqrt(d). With this choice the best possible classifier is equally accurate at every size, and the
 * only thing that changes is how many pixels the evidence (and the attacker) can use.
 */
export const Z0 = 3;
export const N_TRAIN = 300, N_TEST = 300, SEED_TRAIN = 11, SEED_TEST = 12;
/** Full-batch gradient descent on the logistic loss: iterations, step size and weight decay. */
export const ITERS = 50, LR = 8, L2 = 1e-3;
export const EPS_MAX = 16;
export const DEFAULTS = { eps: 8, dim: 1024, attack: 'fgsm' };

export interface Dataset { n: number; d: number; count: number; X: Float64Array; y: Uint8Array; s: number }
export interface Model { n: number; d: number; w: Float64Array; b: number }

/** Oblique stripes (two periods across, one down), scaled to unit root-mean-square so that s is the amplitude per pixel. */
export function stripePattern(n: number): Float64Array {
  const a = new Float64Array(n * n);
  let ss = 0;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const v = Math.cos(2 * Math.PI * (2 * (x + 0.5) / n + (y + 0.5) / n));
    a[y * n + x] = v; ss += v * v;
  }
  const rms = Math.sqrt(ss / (n * n));
  for (let i = 0; i < a.length; i++) a[i] /= rms;
  return a;
}
/** Class A (label 1) is 0.5 + s·a, class B (label 0) is 0.5 − s·a, plus Gaussian noise, clipped to [0, 1]. Labels alternate. */
export function makeDataset(n: number, count: number, seed: number): Dataset {
  const d = n * n, s = Z0 * SIGMA / Math.sqrt(d), a = stripePattern(n), z = normalField(count * d, seed);
  const X = new Float64Array(count * d), y = new Uint8Array(count);
  for (let k = 0; k < count; k++) {
    y[k] = k % 2;
    const sign = y[k] ? 1 : -1;
    for (let i = 0; i < d; i++) X[k * d + i] = Math.min(1, Math.max(0, 0.5 + sign * s * a[i] + SIGMA * z[k * d + i]));
  }
  return { n, d, count, X, y, s };
}

// ---------------------------------------------------------------------------------------------- model and attacks
/** Numerically stable logistic function. */
export const sigmoid = (z: number) => z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
/** Logit z = w·(x − 0.5) + b; z > 0 means class A. */
export function logit(m: Model, x: ArrayLike<number>, offset = 0): number {
  let z = m.b;
  for (let i = 0; i < m.d; i++) z += m.w[i] * (x[offset + i] - 0.5);
  return z;
}
/** Train a logistic-regression classifier by full-batch gradient descent with weight decay (deterministic, no randomness). */
export function trainLogistic(ds: Dataset, iters = ITERS, lr = LR, l2 = L2): Model {
  const { d, count: N, X, y } = ds, w = new Float64Array(d), g = new Float64Array(d), m: Model = { n: ds.n, d, w, b: 0 };
  for (let it = 0; it < iters; it++) {
    g.fill(0);
    let gb = 0;
    for (let k = 0; k < N; k++) {
      const e = sigmoid(logit(m, X, k * d)) - y[k];
      gb += e;
      for (let i = 0; i < d; i++) g[i] += e * (X[k * d + i] - 0.5);
    }
    for (let i = 0; i < d; i++) w[i] -= lr * (g[i] / N + l2 * w[i]);
    m.b -= lr * gb / N;
  }
  return m;
}
export const l1Norm = (w: ArrayLike<number>) => { let s = 0; for (let i = 0; i < w.length; i++) s += Math.abs(w[i]); return s; };
export const l2Norm = (w: ArrayLike<number>) => { let s = 0; for (let i = 0; i < w.length; i++) s += w[i] * w[i]; return Math.sqrt(s); };

/**
 * dL/dx_i of the cross-entropy loss at (x, y): the logit is linear in x, so it equals (p − y)·w_i. The factor p − y is computed in
 * a stable form (it is −σ(−z) for y = 1 and σ(z) for y = 0) so that a very confident prediction still has a gradient direction.
 */
export function lossGradient(m: Model, x: ArrayLike<number>, y: number, offset = 0): Float64Array {
  const z = logit(m, x, offset), f = y ? -sigmoid(-z) : sigmoid(z), g = new Float64Array(m.d);
  for (let i = 0; i < m.d; i++) g[i] = f * m.w[i];
  return g;
}
/** The FGSM direction: sign of the input gradient of the loss, entry by entry (−1, 0 or +1). */
export function fgsmDirection(m: Model, x: ArrayLike<number>, y: number, offset = 0): Int8Array {
  const g = lossGradient(m, x, y, offset), out = new Int8Array(m.d);
  for (let i = 0; i < m.d; i++) out[i] = g[i] > 0 ? 1 : g[i] < 0 ? -1 : 0;
  return out;
}
/** A fixed random-sign direction (±1 per pixel) for sample `k`: the same draw every time, so a slider never makes it flicker. */
export function randomDirection(d: number, k: number): Int8Array {
  const r = mulberry32(7919 * (k + 1)), out = new Int8Array(d);
  for (let i = 0; i < d; i++) out[i] = r() < 0.5 ? -1 : 1;
  return out;
}
/** x' = clip(x + ε·direction) with ε in 0..1 pixel units (a level of 8-bit brightness is 1/255). */
export function perturb(x: ArrayLike<number>, offset: number, dir: Int8Array, eps: number): Float64Array {
  const out = new Float64Array(dir.length);
  for (let i = 0; i < dir.length; i++) out[i] = Math.min(1, Math.max(0, x[offset + i] + eps * dir[i]));
  return out;
}

export interface Bundle {
  n: number; d: number; train: Dataset; test: Dataset; model: Model;
  /** Clean logits of the test set. */
  z0: Float64Array;
  /** FGSM and random-sign directions of every test sample, stored flat (count × d). */
  fgsm: Int8Array; rand: Int8Array;
  l1: number; l2: number;
  curveFgsm?: Float64Array; curveRand?: Float64Array;
}
const bundles = new Map<number, Bundle>();
/** Data, trained model and attack directions for an n × n image (trained once per size, then cached). */
export function bundleFor(n: number): Bundle {
  const hit = bundles.get(n);
  if (hit) return hit;
  const train = makeDataset(n, N_TRAIN, SEED_TRAIN), test = makeDataset(n, N_TEST, SEED_TEST), model = trainLogistic(train), d = n * n;
  const z0 = new Float64Array(N_TEST), fgsm = new Int8Array(N_TEST * d), rand = new Int8Array(N_TEST * d);
  for (let k = 0; k < N_TEST; k++) {
    z0[k] = logit(model, test.X, k * d);
    fgsm.set(fgsmDirection(model, test.X, test.y[k], k * d), k * d);
    rand.set(randomDirection(d, k), k * d);
  }
  const b: Bundle = { n, d, train, test, model, z0, fgsm, rand, l1: l1Norm(model.w), l2: l2Norm(model.w) };
  bundles.set(n, b);
  return b;
}
export interface Attacked { acc: number; clean: number; meanShift: number; rmsShift: number }
/** Accuracy of the model on the test set under a perturbation of ε levels (0..255 scale) along the FGSM or random directions. */
export function attackResult(b: Bundle, levels: number, kind: 'fgsm' | 'random'): Attacked {
  const { d, test, model, z0 } = b, dirs = kind === 'fgsm' ? b.fgsm : b.rand, eps = levels / 255;
  let ok = 0, clean = 0, shift = 0, shift2 = 0;
  for (let k = 0; k < test.count; k++) {
    const sgn = test.y[k] ? 1 : -1;
    let z = model.b;
    for (let i = 0; i < d; i++) z += model.w[i] * (Math.min(1, Math.max(0, test.X[k * d + i] + eps * dirs[k * d + i])) - 0.5);
    if (sgn * z > 0) ok++;
    if (sgn * z0[k] > 0) clean++;
    const sh = sgn * (z0[k] - z);
    shift += sh; shift2 += sh * sh;
  }
  const N = test.count;
  return { acc: ok / N, clean: clean / N, meanShift: shift / N, rmsShift: Math.sqrt(shift2 / N) };
}
/** Accuracy at every integer ε from 0 to EPS_MAX levels (cached in the bundle). */
export function accuracyCurve(b: Bundle, kind: 'fgsm' | 'random'): Float64Array {
  const key = kind === 'fgsm' ? 'curveFgsm' : 'curveRand';
  if (!b[key]) b[key] = Float64Array.from({ length: EPS_MAX + 1 }, (_, e) => attackResult(b, e, kind).acc);
  return b[key]!;
}
/** Smallest ε (interpolated between integer levels) at which the accuracy first drops to 50% or below, or null if it never does. */
export function halfAccuracyLevel(curve: ArrayLike<number>): number | null {
  for (let e = 1; e < curve.length; e++) if (curve[e] <= 0.5) {
    const a = curve[e - 1], c = curve[e];
    return a === c ? e : e - 1 + (a - 0.5) / (a - c);
  }
  return null;
}
/** The example shown: a correctly classified class-A test image whose margin is closest to the median margin. */
export function exampleIndex(b: Bundle): number {
  const margins: number[] = [], idx: number[] = [];
  for (let k = 0; k < b.test.count; k++) if (b.test.y[k] === 1 && b.z0[k] > 0) { margins.push(b.z0[k]); idx.push(k); }
  const sorted = [...margins].sort((p, q) => p - q), med = sorted[Math.floor(sorted.length / 2)];
  let best = 0;
  for (let j = 1; j < idx.length; j++) if (Math.abs(margins[j] - med) < Math.abs(margins[best] - med)) best = j;
  return idx[best];
}
/** Median of the signed clean margins y·z over the test set (correct predictions have positive margin). */
export function medianMargin(b: Bundle): number {
  const m = Array.from({ length: b.test.count }, (_, k) => (b.test.y[k] ? 1 : -1) * b.z0[k]).sort((p, q) => p - q);
  return m[Math.floor(m.length / 2)];
}

// ---------------------------------------------------------------------------------------------- drawing
const STRIP_W = 600, STRIP_H = 262, PLOT_W = 420, PLOT_H = 330;
const pct = (v: number, digits = 1) => `${(100 * v).toFixed(digits)}%`;
const signed = (v: number) => { const r = Math.round(v * 10) / 10; return r === 0 ? '0.0' : `${r > 0 ? '+' : '−'}${Math.abs(r).toFixed(1)}`; };
const className = (z: number) => z > 0 ? 'A' : 'B';

/** Three images side by side on one canvas (so that they stay in one row on a phone): clean, the change, perturbed. */
function drawImages(canvas: HTMLCanvasElement, tmp: HTMLCanvasElement, lang: Lang, n: number, imgs: ArrayLike<number>[], captions: { title: string; sub: string; colour: string }[]) {
  const col = plotColors(canvas), ctx = startPlot(canvas, STRIP_W, STRIP_H, col), size = 180, gap = (STRIP_W - 3 * size) / 4;
  tmp.width = n; tmp.height = n;
  const t = tmp.getContext('2d')!;
  imgs.forEach((gray, k) => {
    const im = t.createImageData(n, n);
    for (let i = 0; i < n * n; i++) { const v = Math.round(255 * Math.min(1, Math.max(0, gray[i]))); im.data[4 * i] = v; im.data[4 * i + 1] = v; im.data[4 * i + 2] = v; im.data[4 * i + 3] = 255; }
    t.putImageData(im, 0, 0);
    const x = gap + k * (size + gap);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, x, 34, size, size);
    ctx.lineWidth = 1; ctx.strokeStyle = col.grid; ctx.strokeRect(x - 0.5, 33.5, size + 1, size + 1);
    label(ctx, captions[k].title, x + size / 2, 24, { align: 'center', bold: true, color: col.ink, size: 17 });
    label(ctx, captions[k].sub, x + size / 2, 34 + size + 22, { align: 'center', bold: true, color: captions[k].colour, size: 16 });
  });
  void lang;
}
function drawCurves(canvas: HTMLCanvasElement, lang: Lang, cur: Bundle, levels: number) {
  const col = plotColors(canvas), ctx = startPlot(canvas, PLOT_W, PLOT_H, col);
  const x0 = 52, x1 = PLOT_W - 16, y0 = 40, y1 = PLOT_H - 96, X = (e: number) => x0 + (e / EPS_MAX) * (x1 - x0), Y = (a: number) => y1 - a * (y1 - y0);
  label(ctx, pick(lang, 'Accuracy under attack', '受攻擊下的準確率'), 12, 22, { bold: true, color: col.ink, size: 17 });
  for (const a of [0, 0.5, 1]) {
    line(ctx, x0, Y(a), x1, Y(a), a === 0.5 ? col.muted : col.grid, 1, a === 0.5 ? [4, 4] : []);
    label(ctx, `${100 * a}%`, x0 - 6, Y(a) + 5, { align: 'right', color: col.muted, size: 14 });
  }
  for (const e of [0, 4, 8, 12, 16]) { line(ctx, X(e), y1, X(e), y1 + 5, col.muted); label(ctx, String(e), X(e), y1 + 21, { align: 'center', color: col.muted, size: 14 }); }
  label(ctx, pick(lang, 'ε: change per pixel, in levels of 255', 'ε：每個像素的改變量（255 階中的幾階）'), (x0 + x1) / 2, y1 + 42, { align: 'center', color: col.muted, size: 14 });
  const poly = (curve: ArrayLike<number>, colour: string, width: number, alpha: number) => {
    ctx.globalAlpha = alpha; ctx.lineWidth = width; ctx.strokeStyle = colour; ctx.lineJoin = 'round'; ctx.beginPath();
    for (let e = 0; e <= EPS_MAX; e++) { if (e === 0) ctx.moveTo(X(e), Y(curve[e])); else ctx.lineTo(X(e), Y(curve[e])); }
    ctx.stroke(); ctx.globalAlpha = 1;
  };
  // FGSM at the other sizes (thin), with a dot where each one reaches 50%; then the current size (thick) and random signs.
  for (const n of SIDES) {
    if (n === cur.n) continue;
    const c = accuracyCurve(bundleFor(n), 'fgsm');
    poly(c, col.b, 1.6, 0.55);
    const h = halfAccuracyLevel(c);
    if (h != null) { ctx.fillStyle = col.b; ctx.globalAlpha = 0.75; ctx.beginPath(); ctx.arc(X(h), Y(0.5), 3.5, 0, 2 * Math.PI); ctx.fill(); ctx.globalAlpha = 1; label(ctx, String(n * n), X(h) + (h > 12 ? -7 : 7), Y(0.5) - 8, { color: col.b, size: 14, bold: true, align: h > 12 ? 'right' : 'left', halo: col.bg }); }
  }
  const cf = accuracyCurve(cur, 'fgsm'), cr = accuracyCurve(cur, 'random');
  poly(cr, col.a, 3, 1); poly(cf, col.b, 3, 1);
  const h = halfAccuracyLevel(cf);
  if (h != null) { ctx.fillStyle = col.b; ctx.beginPath(); ctx.arc(X(h), Y(0.5), 4.5, 0, 2 * Math.PI); ctx.fill(); label(ctx, String(cur.d), X(h) + (h > 12 ? -8 : 8), Y(0.5) - 9, { color: col.b, size: 15, bold: true, align: h > 12 ? 'right' : 'left', halo: col.bg }); }
  // The current ε.
  line(ctx, X(levels), y0 - 6, X(levels), y1, col.ink, 1.5, [5, 3]);
  for (const [c, colour] of [[cf, col.b], [cr, col.a]] as const) { const e = Math.round(levels); ctx.fillStyle = colour; ctx.strokeStyle = col.bg; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(e), Y(c[e]), 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); }
  // Legend under the axis.
  const ly = PLOT_H - 36;
  line(ctx, 14, ly, 40, ly, col.b, 3); label(ctx, pick(lang, `FGSM, ${cur.n} × ${cur.n}`, `FGSM，${cur.n} × ${cur.n}`), 46, ly + 5, { color: col.ink, size: 14 });
  line(ctx, 190, ly, 216, ly, col.a, 3); label(ctx, pick(lang, 'random signs, same ε', '隨機正負號，相同 ε'), 222, ly + 5, { color: col.ink, size: 14 });
  line(ctx, 14, ly + 22, 40, ly + 22, col.b, 1.6); label(ctx, pick(lang, 'FGSM, other sizes (dot: 50% accuracy)', 'FGSM，其他尺寸（圓點：準確率 50%）'), 46, ly + 27, { color: col.muted, size: 14 });
}

// ---------------------------------------------------------------------------------------------- text
export function describe(lang: Lang, b: Bundle, levels: number, fg: Attacked, rd: Attacked): string {
  const medM = medianMargin(b);
  return pick(lang,
    `At ε = ${levels} of 255 levels (${(100 * levels / 255).toFixed(1)}% of the brightness range) every one of the ${b.d.toLocaleString('en')} pixels moves by at most ${levels} levels. Accuracy goes from ${pct(fg.clean)} to ${pct(fg.acc)} when the signs follow the gradient (FGSM) and to ${pct(rd.acc)} when they are random. FGSM moves the logit by ${signed(fg.meanShift)} on average, against a median clean margin of ${medM.toFixed(1)}; random signs move it by ${signed(rd.meanShift)} on average.`,
    `在 ε = ${levels}（255 階中的 ${levels} 階，約占亮度範圍的 ${(100 * levels / 255).toFixed(1)}%）時，${b.d.toLocaleString('en')} 個像素每一個最多只改變 ${levels} 階。正負號沿著梯度（FGSM）時，準確率由 ${pct(fg.clean)} 降到 ${pct(fg.acc)}；正負號隨機時則是 ${pct(rd.acc)}。FGSM 平均讓 logit 移動 ${signed(fg.meanShift)}，而乾淨影像的邊際（margin）中位數是 ${medM.toFixed(1)}；隨機正負號平均只移動 ${signed(rd.meanShift)}。`);
}

// ---------------------------------------------------------------------------------------------- the lab
export const lab: LabDef = {
  kind: 'image', id: 'adversarial-budget', track: 'deep-learning',
  title: L('DLCV 14 · Why can an invisible change flip a prediction?', 'DLCV 14・為什麼看不出來的改變就能翻轉預測？'),
  description: L('Train a linear classifier on noisy stripe images of 64, 256 or 1,024 pixels, then change every pixel by the same few levels, once along the loss gradient (FGSM) and once with random signs.', '在 64、256 或 1,024 個像素的雜訊條紋影像上訓練線性分類器，再把每個像素都改變相同的幾階：一次沿著損失梯度（FGSM），一次用隨機正負號。'),
  challenge: L(
    'At the default setting (32 × 32 pixels, ε = 8, FGSM) accuracy falls from 99.3% to 2.7%, while random signs of the same size leave it at 99.3%. The mean logit shift is +5.8, against a median clean margin of 3.5. Now switch the size to 16 × 16 and then to 8 × 8 at the same ε: FGSM leaves 76.3% and 94.7%. Next find the ε at which FGSM first brings accuracy below 50%: ε = 5 for 32 × 32, ε = 11 for 16 × 16, and none up to ε = 16 for 8 × 8 (70.7% at ε = 16). The classes differ by the same total amount at every size, so why does the same ε matter more when the image has more pixels?',
    '預設設定（32 × 32 像素、ε = 8、FGSM）下，準確率由 99.3% 掉到 2.7%；同樣大小的隨機正負號則維持在 99.3%。logit 平均位移是 +5.8，而乾淨影像的邊際（margin）中位數是 3.5。接著把大小改成 16 × 16、再改成 8 × 8，ε 不變：FGSM 之後分別剩 76.3% 與 94.7%。再找出 FGSM 第一次讓準確率低於 50% 的 ε：32 × 32 是 ε = 5，16 × 16 是 ε = 11，8 × 8 到 ε = 16 都沒有（ε = 16 時仍有 70.7%）。不論大小，兩個類別相差的總量都一樣，那麼為什麼影像的像素越多，同樣的 ε 影響就越大？'),
  note: L(
    'Synthetic and tiny on purpose: no photographs. Each image is 0.5 plus or minus an oblique cosine stripe pattern (two periods across, one down) plus Gaussian noise with standard deviation 0.2 (51 of 255 levels), clipped to [0, 1]; class A has the stripes in one phase and class B in the opposite phase. The stripe amplitude per pixel is 3 × 0.2 / √d for d pixels, so the two classes differ by the same total amount at every size (before clipping, the best possible classifier is equally accurate at every size); only the number of pixels that carry the evidence changes. The classifier is a logistic regression with one weight per pixel, trained in your browser by 50 steps of full-batch gradient descent (step 8, weight decay 0.001) on 300 images and tested on 300 other images (seeds 11 and 12). Its logit is z = w·(x − 0.5) + b, so the gradient of the cross-entropy loss with respect to the pixels is (p − y)·w and FGSM, x′ = clip(x + ε·sign(∇ₓL), 0, 1), moves every pixel by ε toward the wrong class. With no clipping the logit moves by exactly ε‖w‖₁; random signs move it by about ε‖w‖₂ and cancel on average. Here ‖w‖₂ is 7.5, 7.5 and 6.9 for 64, 256 and 1,024 pixels, while ‖w‖₁ is 54.4, 102.8 and 186.2. ε is counted in levels of 8-bit brightness (1 level = 1/255). This is a linear toy that isolates the argument of Goodfellow et al.; it is not the digits experiment of the chapter, a deep network is not linear, and one draw of data and training is shown, so read the trend, not the exact percentages.',
    '刻意做得合成又小：沒有使用任何照片。每張影像是 0.5 加減一個斜向的餘弦條紋圖樣（橫向兩個週期、縱向一個週期），再加上標準差 0.2（255 階中的 51 階）的高斯雜訊，並截在 [0, 1] 之內；A 類的條紋是一種相位，B 類是相反的相位。d 個像素時，每個像素的條紋振幅是 3 × 0.2 / √d，所以不論大小，兩個類別相差的總量都一樣（不考慮截斷時，最佳分類器在每種大小下的準確率相同）；改變的只有承載證據的像素數量。分類器是每個像素一個權重的邏輯迴歸，在你的瀏覽器裡以 50 步全批次梯度下降（步長 8、權重衰減 0.001）用 300 張影像訓練，並在另外 300 張影像上測試（種子 11 與 12）。它的 logit 是 z = w·(x − 0.5) + b，所以交叉熵損失對像素的梯度是 (p − y)·w，而 FGSM，x′ = clip(x + ε·sign(∇ₓL), 0, 1)，把每個像素都朝錯誤的類別推 ε。不截斷時，logit 恰好移動 ε‖w‖₁；隨機正負號則只移動約 ε‖w‖₂，且平均會互相抵消。這裡 64、256 與 1,024 個像素的 ‖w‖₂ 分別是 7.5、7.5 與 6.9，‖w‖₁ 則是 54.4、102.8 與 186.2。ε 以 8 位元亮度的階數計算（1 階 = 1/255）。這是一個線性的玩具，用來單獨呈現 Goodfellow 等人的論點；它不是章節中的手寫數字實驗，深度網路也不是線性的，而且這裡只呈現一次資料與訓練的結果，所以請看趨勢，不要看確切的百分比。'),
  related: ['deep-learning/dl-11-robustness'],
  controls: [
    { key: 'eps', label: L('Attack size ε (levels of 255)', '攻擊強度 ε（255 階中的幾階）'), kind: 'range', value: DEFAULTS.eps, min: 0, max: EPS_MAX, step: 1 },
    { key: 'dim', label: L('Image size (pixels)', '影像大小（像素數）'), kind: 'select', value: String(DEFAULTS.dim), options: SIDES.map(n => ({ value: String(n * n), label: L(`${n} × ${n} = ${n * n}`, `${n} × ${n} = ${n * n}`) })) },
    { key: 'attack', label: L('Perturbation shown', '顯示的擾動'), kind: 'select', value: DEFAULTS.attack, options: [
      { value: 'fgsm', label: L('FGSM: signs of the loss gradient', 'FGSM：損失梯度的正負號') },
      { value: 'random', label: L('Random signs, same size', '隨機正負號，相同大小') },
    ] },
  ],
  predict: {
    question: L(
      'Every pixel of a 32 × 32 test image is changed by at most 8 of 255 levels, far below the pixel noise of 51 levels. Which change brings the accuracy of the trained classifier close to zero: signs that follow the loss gradient (FGSM), random signs, both, or neither?',
      '32 × 32 測試影像的每個像素最多只改變 255 階中的 8 階，遠小於像素雜訊的 51 階。哪一種改變會讓訓練好的分類器準確率接近零：正負號沿著損失梯度（FGSM）、隨機正負號、兩者，還是兩者都不會？'),
    options: [
      { value: 'both', label: L('Both', '兩者都會') },
      { value: 'fgsm', label: L('Only FGSM', '只有 FGSM') },
      { value: 'random', label: L('Only random signs', '只有隨機正負號') },
      { value: 'neither', label: L('Neither', '兩者都不會') },
    ],
    answer: 'fgsm',
    explain: L(
      'Only FGSM: accuracy 2.7% against 99.3% for random signs. The logit is a sum over 1,024 pixels. If all 1,024 changes of 8 levels have the sign that raises the loss, the logit moves by ε‖w‖₁ = 5.8, more than the median margin of 3.5. Random signs push half the pixels each way and cancel: the typical shift is about ε‖w‖₂ = 0.2. The change per pixel is the same; what differs is whether the 1,024 small changes add up.',
      '只有 FGSM：準確率 2.7%，隨機正負號則是 99.3%。logit 是 1,024 個像素的總和。如果 1,024 個 8 階的改變都取讓損失上升的正負號，logit 會移動 ε‖w‖₁ = 5.8，超過邊際（margin）中位數 3.5。隨機正負號讓一半像素往一邊、一半往另一邊而互相抵消：典型的位移只有約 ε‖w‖₂ = 0.2。每個像素的改變量一樣，差別在於 1,024 個小改變是否加在一起。'),
  },
  assets: {},
  panes: [
    { key: 'images', kind: 'canvas', label: L('Left: a clean test image (noisy stripes). Middle: the change, stretched to the full gray range for display (white = +ε, black = −ε). Right: the perturbed image. Under each, the logit and the class it gives.', '左：乾淨的測試影像（雜訊條紋）。中：改變量，為了顯示而拉伸到整個灰階範圍（白 = +ε，黑 = −ε）。右：加上擾動後的影像。每張下方是它的 logit 與判定的類別。'), width: STRIP_W, height: STRIP_H, wide: true },
    { key: 'curve', kind: 'canvas', label: L('Accuracy on 300 test images as ε grows', '300 張測試影像的準確率隨 ε 變化'), width: PLOT_W, height: PLOT_H, wide: true },
  ],
  sources: [
    { title: 'I. J. Goodfellow, J. Shlens and C. Szegedy, Explaining and Harnessing Adversarial Examples, ICLR, 2015 (the linear explanation and the fast gradient sign method)', url: 'https://arxiv.org/abs/1412.6572' },
  ],
  create({ lang }) {
    const tmp = document.createElement('canvas');
    return {
      render(state, canvases) {
        const levels = Math.max(0, Math.min(EPS_MAX, Math.round(num(state, 'eps'))));
        const dim = Number(state.dim), n = SIDES.find(s => s * s === dim) ?? 32, b = bundleFor(n), kind = state.attack === 'random' ? 'random' : 'fgsm';
        const fg = attackResult(b, levels, 'fgsm'), rd = attackResult(b, levels, 'random');
        const k = exampleIndex(b), dirs = kind === 'fgsm' ? b.fgsm : b.rand, eps = levels / 255;
        const adv = perturb(b.test.X, k * b.d, dirs.subarray(k * b.d, (k + 1) * b.d), eps), clean = b.test.X.subarray(k * b.d, (k + 1) * b.d);
        const z1 = logit(b.model, adv), z0 = b.z0[k];
        const delta = Float64Array.from(adv, (v, i) => levels === 0 ? 0.5 : 0.5 + 0.5 * (v - clean[i]) / eps);
        const ok1 = z1 > 0 === z0 > 0, green = '#1a8f3c', red = '#d62728';
        drawImages(canvases.images, tmp, lang, n, [clean, delta, adv], [
          { title: pick(lang, 'clean', '乾淨'), sub: `${signed(z0)} → ${className(z0)}`, colour: green },
          { title: pick(lang, 'the change', '改變量'), sub: pick(lang, `±${levels} of 255`, `±${levels} / 255`), colour: '#6b6b6b' },
          { title: pick(lang, 'perturbed', '擾動後'), sub: `${signed(z1)} → ${className(z1)}`, colour: ok1 ? green : red },
        ]);
        drawCurves(canvases.curve, lang, b, levels);
        return {
          summary: describe(lang, b, levels, fg, rd),
          metrics: [
            { label: pick(lang, 'Accuracy, clean', '準確率（乾淨）'), value: pct(fg.clean) },
            { label: pick(lang, 'Accuracy under FGSM', '準確率（FGSM）'), value: pct(fg.acc) },
            { label: pick(lang, 'Accuracy under random signs', '準確率（隨機正負號）'), value: pct(rd.acc) },
            { label: pick(lang, 'Mean logit shift toward the wrong class: FGSM / random', '朝錯誤類別的平均 logit 位移：FGSM／隨機'), value: `${signed(fg.meanShift)} / ${signed(rd.meanShift)}` },
            { label: pick(lang, 'Linear prediction ε·‖w‖₁ of the FGSM shift', 'FGSM 位移的線性預測 ε·‖w‖₁'), value: fmt(eps * b.l1, 1) },
            { label: pick(lang, 'Median clean margin', '乾淨影像邊際的中位數'), value: fmt(medianMargin(b), 1) },
            { label: pick(lang, 'Logit of the shown image: clean → perturbed', '所示影像的 logit：乾淨 → 擾動後'), value: `${signed(z0)} → ${signed(z1)}` },
          ],
        };
      },
    };
  },
};
