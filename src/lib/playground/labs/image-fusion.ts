import type { LabDef } from './types.ts';
import { L, pick, fmt, num } from '../types.ts';
import type { Gray } from '../../imagekit/types.ts';
import { gaussianBlur, reflectIndex } from '../../imagekit/filter.ts';
import { drawGray, drawRgb } from '../../imagekit/draw.ts';
import { psnr } from '../../imagekit/metrics.ts';

/** How "how much fine detail is around this pixel" is measured. */
export type Measure = 'laplacian' | 'variance';

/** Five-point Laplacian (the stencil of scipy.ndimage.laplace: left + right + up + down − 4 · centre), with 'reflect' borders. */
export function laplace(src: Float32Array, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const xm = reflectIndex(x - 1, w), xp = reflectIndex(x + 1, w), ym = reflectIndex(y - 1, h), yp = reflectIndex(y + 1, h);
    out[y * w + x] = src[y * w + xm] + src[y * w + xp] + src[ym * w + x] + src[yp * w + x] - 4 * src[y * w + x];
  }
  return out;
}
/**
 * Focus measure: how much fine detail surrounds each pixel, averaged with a Gaussian window of standard deviation `sigma`.
 * laplacian: G_σ * (∇²I)²; variance: G_σ * I² − (G_σ * I)².
 */
export function focusMeasure(img: Float32Array, w: number, h: number, kind: Measure, sigma: number): Float32Array {
  if (kind === 'laplacian') {
    const l = laplace(img, w, h);
    for (let i = 0; i < l.length; i++) l[i] *= l[i];
    return gaussianBlur(l, w, h, sigma);
  }
  const sq = new Float32Array(img.length);
  for (let i = 0; i < sq.length; i++) sq[i] = img[i] * img[i];
  const m = gaussianBlur(img, w, h, sigma), m2 = gaussianBlur(sq, w, h, sigma), out = new Float32Array(img.length);
  for (let i = 0; i < out.length; i++) out[i] = m2[i] - m[i] * m[i];
  return out;
}
/** Decision map: 1 where shot A has the larger focus measure (ties go to B), 0 where shot B does. */
export function decisionMap(a: Float32Array, b: Float32Array, w: number, h: number, kind: Measure, sigma: number): Uint8Array {
  const ma = focusMeasure(a, w, h, kind, sigma), mb = focusMeasure(b, w, h, kind, sigma), d = new Uint8Array(w * h);
  for (let i = 0; i < d.length; i++) d[i] = ma[i] > mb[i] ? 1 : 0;
  return d;
}
/** Weight of shot A: the decision map itself (smooth = 0) or the map blurred with a Gaussian of standard deviation `smooth`. */
export function weights(d: Uint8Array, w: number, h: number, smooth: number): Float32Array {
  const f = Float32Array.from(d);
  return smooth > 0 ? gaussianBlur(f, w, h, smooth) : f;
}
/** F = w · A + (1 − w) · B, clipped to [0, 1]. */
export function blend(a: Float32Array, b: Float32Array, wt: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < out.length; i++) { const v = wt[i] * a[i] + (1 - wt[i]) * b[i]; out[i] = v < 0 ? 0 : v > 1 ? 1 : v; }
  return out;
}
export function average(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < out.length; i++) { const v = 0.5 * (a[i] + b[i]); out[i] = v < 0 ? 0 : v > 1 ? 1 : v; }
  return out;
}
/** Pixels where the choice between the shots matters: smoothed |Laplacian| of the sharp truth above 0.01 (the chapter's definition). */
export function texturedPixels(truth: Float32Array, w: number, h: number): Uint8Array {
  const l = laplace(truth, w, h);
  for (let i = 0; i < l.length; i++) l[i] = Math.abs(l[i]);
  const g = gaussianBlur(l, w, h, 2), out = new Uint8Array(l.length);
  for (let i = 0; i < out.length; i++) out[i] = g[i] > 0.01 ? 1 : 0;
  return out;
}
/** Fraction of the pixels in `where` (all pixels when omitted) on which the decision map equals the true focus map. */
export function agreement(d: Uint8Array, focus: Uint8Array, where?: Uint8Array): number {
  let hit = 0, n = 0;
  for (let i = 0; i < d.length; i++) { if (where && !where[i]) continue; n++; if (d[i] === focus[i]) hit++; }
  return n ? hit / n : NaN;
}
/** Root-mean-square of the detail (image minus its Gaussian blur, σ = 2) over the pixels in `where`: a measure of how much fine contrast survives. */
export function detailContrast(img: Float32Array, w: number, h: number, where: Uint8Array): number {
  const low = gaussianBlur(img, w, h, 2);
  let s = 0, n = 0;
  for (let i = 0; i < img.length; i++) if (where[i]) { const d = img[i] - low[i]; s += d * d; n++; }
  return Math.sqrt(s / n);
}
const toMask = (g: Gray) => Uint8Array.from(g.data, v => (v > 0.5 ? 1 : 0));

export interface Fusion { avg: Float32Array; fused: Float32Array; d: Uint8Array; wt: Float32Array; psnrAvg: number; psnrFused: number; agree: number }
/** One full run: both fusion rules, the decision map and the three numbers. */
export function run(a: Gray, b: Gray, truth: Gray, focus: Uint8Array, tex: Uint8Array, kind: Measure, sigma: number, smooth: number): Fusion {
  const d = decisionMap(a.data, b.data, a.w, a.h, kind, sigma), wt = weights(d, a.w, a.h, smooth);
  const avg = average(a.data, b.data), fused = blend(a.data, b.data, wt);
  return { avg, fused, d, wt, psnrAvg: psnr(truth.data, avg), psnrFused: psnr(truth.data, fused), agree: agreement(d, focus, tex) };
}

export const lab: LabDef = {
  kind: 'image', id: 'image-fusion', track: 'deep-learning',
  title: L('DLCV 05B · Does averaging two photos fuse them?', 'DLCV 05B・把兩張照片取平均，就算融合了嗎？'),
  description: L('Two shots of one scene are each sharp in a different region. Average them, or pick every pixel from the shot with more local detail, and score both against the all-in-focus truth.', '同一個場景的兩張照片，各自在不同區域是清晰的。把它們取平均，或逐像素選取局部細節較多的那一張，並以全清晰的真實影像評分。'),
  challenge: L('Start from the defaults: the average is blurry everywhere while the focus rule is sharp. Now lower the window to 0.5 and look at the decision map: why does it fill with specks, and where do they hurt the picture? Switch the detail measure to local variance at window 0.5 (it is far less forgiving), then raise the window and smooth the map by 2. What does smoothing buy, and what does a very large window cost at the circle’s edge?', '從預設值開始：平均後的影像處處模糊，而依清晰度選取的結果是清晰的。現在把視窗降到 0.5，觀察決策圖：為什麼它充滿雜點？這些雜點在哪裡傷害了影像？再把細節量測改成局部變異數、視窗維持 0.5（它對雜點的容忍度低得多），然後把視窗調大、並把決策圖平滑 2。平滑換來了什麼？視窗太大時，圓的邊緣又付出了什麼代價？'),
  note: L('Synthetic multi-focus pair (scripts/figures/playground_image_fusion.py), the recipe of the chapter’s snippet at half the size: the camera test image (256 × 256) is the all-in-focus truth; shot A keeps it inside a disk and is blurred (Gaussian, σ = 2) outside, shot B the other way round, each with Gaussian noise of standard deviation 0.01. Because we made the blur, the true decision map (the disk) is known, which real multi-focus pairs never provide. The average is (A + B)/2. The focus rule scores each shot at each pixel with a detail measure averaged over a Gaussian window of standard deviation σ (the “window”): the Laplacian energy G_σ * (∇²I)², with ∇²I = I(x−1,y) + I(x+1,y) + I(x,y−1) + I(x,y+1) − 4 I(x,y), or the local variance G_σ * I² − (G_σ * I)². The decision map marks the pixels where shot A scores higher, and F = w A + (1 − w) B with w the decision map, optionally blurred by a Gaussian of the chosen σ (0 keeps it binary). PSNR = 10 log₁₀(1/MSE) in dB against the truth, pixel values in [0, 1]. Map agreement is the share of pixels, counted only where the picture has texture (smoothed |Laplacian| of the truth above 0.01), on which the decision equals the true disk. These numbers belong to this toy.', '合成的多焦點影像對（scripts/figures/playground_image_fusion.py），做法與章節程式的相同，只是尺寸縮小一半：camera 測試影像（256 × 256）是全清晰的真實影像；照片 A 在圓盤內保留原圖、圓盤外做高斯模糊（σ = 2），照片 B 則相反，兩者各加上標準差 0.01 的高斯雜訊。由於模糊是我們自己做的，真實的決策圖（圓盤）完全已知，而真實的多焦點照片永遠無法提供這一點。平均是 (A + B)/2。選取規則在每個像素替兩張照片打分，分數是在標準差 σ 的高斯視窗（即「視窗」）內平均的細節量測：拉普拉斯能量 G_σ * (∇²I)²，其中 ∇²I = I(x−1,y) + I(x+1,y) + I(x,y−1) + I(x,y+1) − 4 I(x,y)；或局部變異數 G_σ * I² − (G_σ * I)²。決策圖標出照片 A 分數較高的像素，F = w A + (1 − w) B，w 是決策圖，可再以所選 σ 的高斯模糊（0 表示維持二值）。PSNR = 10 log₁₀(1/MSE)，單位 dB，以真實影像為基準，像素值在 [0, 1]。圖的一致率是在有紋理的像素上（真實影像的平滑 |拉普拉斯| 大於 0.01）決策與真實圓盤相同的比例。這些數字只屬於這個玩具範例。'),
  related: ['deep-learning/dl-05b-image-fusion'],
  controls: [
    { key: 'measure', label: L('Detail measure', '細節量測'), kind: 'select', value: 'laplacian', options: [
      { value: 'laplacian', label: L('Laplacian energy', '拉普拉斯能量') }, { value: 'variance', label: L('Local variance', '局部變異數') }] },
    { key: 'window', label: L('Window σ (pixels)', '視窗 σ（像素）'), kind: 'range', value: 2, min: 0.5, max: 6, step: 0.5 },
    { key: 'smooth', label: L('Smooth the decision map by σ', '決策圖平滑 σ'), kind: 'range', value: 0, min: 0, max: 4, step: 0.5 },
  ],
  predict: {
    question: L('Shot A is sharp inside the disk and shot B is sharp outside it. Inside the disk, what does the plain average (A + B)/2 look like?', '照片 A 在圓盤內清晰，照片 B 在圓盤外清晰。在圓盤內，單純取平均 (A + B)/2 看起來如何？'),
    options: [
      { value: 'sharp', label: L('Sharp: A is sharp there, and B only adds a little blur', '清晰：那裡 A 是清晰的，B 只多加一點模糊') },
      { value: 'weak', label: L('Weakened: A’s fine contrast shrinks to about 60% and a soft ghost of B is mixed in', '被削弱：A 的細微對比縮小到約 60%，並混入 B 的柔和重影') },
      { value: 'blur', label: L('As blurry as B', '和 B 一樣模糊') },
    ],
    answer: 'weak',
    explain: L('Inside the disk A equals the truth T and B is a blurred copy T′, so (A + B)/2 = (T + T′)/2. The blurred copy has very little fine detail, so the fine detail of T survives at only about half amplitude (a little more, because T′ keeps some detail and the noise adds some), on top of a soft ghost. Measuring the fine contrast as the root-mean-square of the image minus its Gaussian blur (σ = 2) inside the disk, the average keeps 61% of the truth’s and the focus-rule result 101%. Averaging cannot choose; selecting by local detail can.', '在圓盤內 A 等於真實影像 T，B 是模糊的複本 T′，所以 (A + B)/2 = (T + T′)/2。模糊複本只剩很少的細微細節，因此 T 的細節只以約一半的振幅留下來（略多一些，因為 T′ 還保有一點細節，雜訊也貢獻了一些），還疊在一層柔和的重影上。把細微對比定義為影像減去其高斯模糊（σ = 2）後在圓盤內的均方根，平均影像只保留真實影像的 61%，選取規則的結果則是 101%。平均無法做選擇，依局部細節挑選才能。'),
  },
  assets: { 'image-fusion-a': 'gray', 'image-fusion-b': 'gray', 'image-fusion-truth': 'gray', 'image-fusion-focus': 'gray' },
  paneMin: 200,
  panes: [
    { key: 'a', kind: 'canvas', label: L('Shot A (sharp inside the disk)', '照片 A（圓盤內清晰）'), width: 256, height: 256 },
    { key: 'b', kind: 'canvas', label: L('Shot B (sharp outside the disk)', '照片 B（圓盤外清晰）'), width: 256, height: 256 },
    { key: 'avg', kind: 'canvas', label: L('Average (A + B)/2', '平均 (A + B)/2'), width: 256, height: 256 },
    { key: 'fused', kind: 'canvas', label: L('Fused by local detail', '依局部細節融合'), width: 256, height: 256 },
    { key: 'map', kind: 'canvas', label: L('Decision map (white: take A; red line: true boundary)', '決策圖（白：取 A；紅線：真實邊界）'), width: 256, height: 256 },
    { key: 'err', kind: 'canvas', label: L('Error of the fused image (4×, bright: wrong)', '融合影像的誤差（放大 4 倍，亮：錯誤）'), width: 256, height: 256 },
  ],
  sources: [
    { title: 'scikit-image documentation, skimage.metrics.peak_signal_noise_ratio (the PSNR used here)', url: 'https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.peak_signal_noise_ratio' },
  ],
  create({ lang, images }) {
    const a = images['image-fusion-a'] as Gray, b = images['image-fusion-b'] as Gray, truth = images['image-fusion-truth'] as Gray;
    const focus = toMask(images['image-fusion-focus'] as Gray), tex = texturedPixels(truth.data, truth.w, truth.h), { w, h } = a;
    // The true boundary of the focus disk: in-focus pixels with an out-of-focus 4-neighbour.
    const edge = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!focus[i]) continue;
      if ((x > 0 && !focus[i - 1]) || (x < w - 1 && !focus[i + 1]) || (y > 0 && !focus[i - w]) || (y < h - 1 && !focus[i + w])) edge[i] = 1;
    }
    return {
      render(state, canvases) {
        const kind: Measure = state.measure === 'variance' ? 'variance' : 'laplacian', sigma = num(state, 'window'), smooth = num(state, 'smooth');
        const r = run(a, b, truth, focus, tex, kind, sigma, smooth);
        drawGray(canvases.a, a); drawGray(canvases.b, b);
        drawGray(canvases.avg, { w, h, data: r.avg }); drawGray(canvases.fused, { w, h, data: r.fused });
        const map = new Float32Array(3 * w * h), err = new Float32Array(w * h);
        for (let i = 0; i < w * h; i++) {
          const e = edge[i];
          map[3 * i] = e ? 0.9 : r.wt[i]; map[3 * i + 1] = e ? 0.1 : r.wt[i]; map[3 * i + 2] = e ? 0.1 : r.wt[i];
          err[i] = Math.min(1, 4 * Math.abs(r.fused[i] - truth.data[i]));
        }
        drawRgb(canvases.map, { w, h, data: map }); drawGray(canvases.err, { w, h, data: err });
        const gain = r.psnrFused - r.psnrAvg, name = kind === 'laplacian' ? pick(lang, 'Laplacian energy', '拉普拉斯能量') : pick(lang, 'local variance', '局部變異數');
        const summary = gain >= 3
          ? pick(lang, `Averaging scores ${fmt(r.psnrAvg, 2)} dB against the sharp truth: every edge is soft. Choosing by ${name} (window σ ${fmt(sigma, 1)}${smooth > 0 ? `, map smoothed by ${fmt(smooth, 1)}` : ''}) scores ${fmt(r.psnrFused, 2)} dB, ${fmt(gain, 1)} dB higher, and the decision map matches the true disk on ${fmt(100 * r.agree, 1)}% of textured pixels.`,
                  `平均後與清晰的真實影像相比為 ${fmt(r.psnrAvg, 2)} dB：每條邊緣都變軟了。依${name}挑選（視窗 σ ${fmt(sigma, 1)}${smooth > 0 ? `，決策圖平滑 ${fmt(smooth, 1)}` : ''}）得到 ${fmt(r.psnrFused, 2)} dB，高出 ${fmt(gain, 1)} dB，決策圖在有紋理的像素上有 ${fmt(100 * r.agree, 1)}% 與真實圓盤相同。`)
          : pick(lang, `With ${name} at window σ ${fmt(sigma, 1)}${smooth > 0 ? ` and the map smoothed by ${fmt(smooth, 1)}` : ''} the focus rule scores ${fmt(r.psnrFused, 2)} dB, only ${fmt(gain, 1)} dB above averaging (${fmt(r.psnrAvg, 2)} dB): the decision map is so noisy (${fmt(100 * r.agree, 1)}% right on textured pixels) that the result is a patchwork of sharp and blurred pieces.`,
                  `以${name}、視窗 σ ${fmt(sigma, 1)}${smooth > 0 ? `、決策圖平滑 ${fmt(smooth, 1)}` : ''}，選取規則得到 ${fmt(r.psnrFused, 2)} dB，只比平均（${fmt(r.psnrAvg, 2)} dB）高 ${fmt(gain, 1)} dB：決策圖雜訊太多（在有紋理的像素上只有 ${fmt(100 * r.agree, 1)}% 正確），結果成了清晰與模糊碎片拼成的馬賽克。`);
        return {
          summary,
          metrics: [
            { label: pick(lang, 'PSNR, average', 'PSNR，平均'), value: `${fmt(r.psnrAvg, 2)} dB` },
            { label: pick(lang, 'PSNR, fused by local detail', 'PSNR，依局部細節融合'), value: `${fmt(r.psnrFused, 2)} dB` },
            { label: pick(lang, 'Decision map right on textured pixels', '決策圖在有紋理像素上的正確率'), value: `${fmt(100 * r.agree, 1)}%` },
          ],
        };
      },
    };
  },
};
