import type { LabDef } from './types.ts';
import { L, pick, fmt, svg, text, path, num } from '../types.ts';
import type { Gray } from '../../imagekit/types.ts';
import { fft2d, freqIndex } from '../../imagekit/fft.ts';
import { normalField } from '../../imagekit/rng.ts';
import { psnr } from '../../imagekit/metrics.ts';
import { drawGray } from '../../imagekit/draw.ts';

/** Seed of the fixed white-noise field: the noise level slider only scales it, so nothing flickers. */
export const SEED = 7;
/** Frequencies with |H| below this are not divided by when K = 0 (the convention of the chapter's `inverse_filter`). */
export const EPS = 1e-6;
/** The K slider holds log10 K; its leftmost value means K = 0. */
export const LOGK_OFF = -7;
export const LOGK_MAX = 0;
export const LOGK_STEP = 0.25;
export const kOf = (logk: number) => (logk <= LOGK_OFF ? 0 : 10 ** logk);
/** The grid of log10 K values on which the PSNR curve and the "best K" are computed (including the K = 0 end). */
export const LOGK_GRID: number[] = Array.from({ length: Math.round((LOGK_MAX - LOGK_OFF) / LOGK_STEP) + 1 }, (_, i) => LOGK_OFF + i * LOGK_STEP);

/** The image spectrum and the spectrum of one fixed unit-variance noise field, both computed once. */
export interface Problem { w: number; h: number; img: Float32Array; Fre: Float64Array; Fim: Float64Array; Zre: Float64Array; Zim: Float64Array }
export function makeProblem(img: Gray, seed = SEED): Problem {
  const { w, h } = img, Fre = Float64Array.from(img.data), Fim = new Float64Array(w * h);
  const Zre = Float64Array.from(normalField(w * h, seed)), Zim = new Float64Array(w * h);
  fft2d(Fre, Fim, w, h); fft2d(Zre, Zim, w, h);
  return { w, h, img: img.data, Fre, Fim, Zre, Zim };
}

/**
 * Transfer function H(u, v) of a Gaussian point spread function with standard deviation `sigma` pixels, for periodic
 * images: H = exp(−2π²σ²(fu² + fv²)) with fu = u/w and fv = v/h in cycles per pixel (real, because the PSF is symmetric).
 */
export function gaussianOtf(w: number, h: number, sigma: number): Float64Array {
  const H = new Float64Array(w * h);
  for (let v = 0; v < h; v++) for (let u = 0; u < w; u++) {
    const fu = freqIndex(u, w) / w, fv = freqIndex(v, h) / h;
    H[v * w + u] = Math.exp(-2 * Math.PI ** 2 * sigma * sigma * (fu * fu + fv * fv));
  }
  return H;
}
/**
 * The restoration filter W(u, v) = conj(H) / (|H|² + K) (the constant-K Wiener filter; H is real here). For K = 0 this is the plain
 * inverse 1/H, except that frequencies with |H| < EPS are set to 0.
 */
export function restorationGain(H: Float64Array, K: number): Float64Array {
  const W = new Float64Array(H.length);
  for (let i = 0; i < H.length; i++) W[i] = K > 0 ? H[i] / (H[i] * H[i] + K) : (Math.abs(H[i]) < EPS ? 0 : 1 / H[i]);
  return W;
}
export const peakGain = (W: Float64Array) => { let m = 0; for (let i = 0; i < W.length; i++) m = Math.max(m, Math.abs(W[i])); return m; };

/** G = H·F + N, where N is the transform of `noise` times the fixed unit noise field (the blur is periodic, as in the chapter's code). */
export function degradedSpectrum(p: Problem, H: Float64Array, noise: number): { re: Float64Array; im: Float64Array } {
  const re = new Float64Array(p.w * p.h), im = new Float64Array(p.w * p.h);
  for (let i = 0; i < re.length; i++) { re[i] = H[i] * p.Fre[i] + noise * p.Zre[i]; im[i] = H[i] * p.Fim[i] + noise * p.Zim[i]; }
  return { re, im };
}
/** Real part of the inverse transform of W·G (not clipped). */
export function applyGain(p: Problem, G: { re: Float64Array; im: Float64Array }, W: Float64Array): Float32Array {
  const re = new Float64Array(p.w * p.h), im = new Float64Array(p.w * p.h);
  for (let i = 0; i < re.length; i++) { re[i] = W[i] * G.re[i]; im[i] = W[i] * G.im[i]; }
  fft2d(re, im, p.w, p.h, true);
  return Float32Array.from(re);
}
export const clip01 = (a: Float32Array) => a.map(v => (v < 0 ? 0 : v > 1 ? 1 : v));

export interface Evaluation { degraded: Float32Array; restored: Float32Array; psnrDegraded: number; psnrRestored: number; gain: number }
/** The whole pipeline for one setting. Both images are clipped to [0, 1] before PSNR, as they are before display. */
export function evaluate(p: Problem, sigma: number, noise: number, K: number): Evaluation {
  const H = gaussianOtf(p.w, p.h, sigma), G = degradedSpectrum(p, H, noise), W = restorationGain(H, K);
  const degraded = clip01(applyGain(p, G, new Float64Array(H.length).fill(1))), restored = clip01(applyGain(p, G, W));
  return { degraded, restored, psnrDegraded: psnr(p.img, degraded), psnrRestored: psnr(p.img, restored), gain: peakGain(W) };
}
/** PSNR of the restored image on LOGK_GRID for fixed blur and noise (one inverse FFT per grid point). */
export function psnrCurve(p: Problem, sigma: number, noise: number): number[] {
  const H = gaussianOtf(p.w, p.h, sigma), G = degradedSpectrum(p, H, noise);
  return LOGK_GRID.map(lk => psnr(p.img, clip01(applyGain(p, G, restorationGain(H, kOf(lk))))));
}
export function bestOnGrid(curve: number[]): { index: number; logk: number; psnr: number } {
  let index = 0;
  curve.forEach((v, i) => { if (v > curve[index]) index = i; });
  return { index, logk: LOGK_GRID[index], psnr: curve[index] };
}

const fmtGain = (g: number) => (g >= 1000 ? `×${g.toExponential(1).replace('e+', '·10^')}` : `×${g.toFixed(1)}`);
const fmtDb = (v: number) => (v > 100 ? '> 100 dB' : `${v.toFixed(1)} dB`);

export const lab: LabDef = {
  kind: 'image', id: 'inverse-regularization', track: 'image-processing',
  title: L('IP 05 · Why does undoing a blur amplify the noise?', 'IP 05・為什麼還原模糊會放大雜訊？'),
  description: L('Blur a real photo with a Gaussian PSF, add a little noise, then divide by the blur in the frequency domain. See the plain inverse filter fail, and find the Wiener constant K that rescues it.', '用高斯 PSF 模糊一張真實照片，加入少量雜訊，再於頻域除以模糊的轉移函數。觀察單純的逆濾波如何失敗，並找出能救回影像的 Wiener 常數 K。'),
  challenge: L('Leave the noise at 0.005 and raise log10 K from −7 in small steps. The restored PSNR climbs, peaks, then falls: where is the peak, and what does the image look like on each side of it? Then set the noise to 0: is K = 0 still a disaster? Finally raise the noise to 0.02 and see where the best K moves.', '保持雜訊 0.005，把 log10 K 從 −7 一小步一小步提高。還原後的 PSNR 會上升、到達高峰、再下降：高峰在哪裡？高峰兩側的影像各是什麼樣子？接著把雜訊設為 0：K = 0 還是災難嗎？最後把雜訊提高到 0.02，看最佳的 K 往哪裡移動。'),
  note: L('Real 256 × 256 camera image. The degradation follows the chapter, G = H·F + N, in the frequency domain. H is the transform of a Gaussian PSF of standard deviation σ pixels, H = exp(−2π²σ²(u² + v²)) with u and v in cycles per pixel; the blur is periodic (circular), as in the chapter’s code, so the filter matches the blur exactly and no padding is used. N is a fixed seeded white Gaussian field scaled by the noise level (a standard deviation as a fraction of the 0 to 1 range, so 0.005 is about 1.3 gray levels of 255); it is added to the blurred image and not rounded to 8 bits. The restoration is the constant-K Wiener filter of the chapter, F̂ = H*/(|H|² + K) · G. Here K is a tuning constant that you choose, not an estimate of the noise-to-signal power ratio. K = 0 is the plain inverse 1/H, where frequencies with |H| below 10^−6 are set to 0 (the convention of the chapter’s code); that is why its peak gain stops at about a million and why no-noise restoration is not perfect for large σ. PSNR compares each image, clipped to 0 to 1, with the original. The gain is the largest value of |H|/(|H|² + K) over all frequencies; for K greater than 0 it never exceeds 1/(2√K).', '使用真實的 256 × 256 相機影像。退化過程依章節的模型，在頻域寫成 G = H·F + N。H 是標準差為 σ 像素的高斯 PSF 的轉移函數，H = exp(−2π²σ²(u² + v²))，其中 u、v 的單位是每像素週期數；模糊採週期性（循環）模型，與章節程式碼相同，因此濾波器與模糊完全吻合，也不做補零。N 是固定種子的白色高斯雜訊場，乘上雜訊強度（標準差，以 0 到 1 的範圍為單位，所以 0.005 約為 255 階中的 1.3 階），直接加在模糊影像上，不量化成 8 位元。還原使用章節中的常數 K Wiener 濾波器，F̂ = H*/(|H|² + K) · G。這裡的 K 是由你自己調整的常數，並不是雜訊對訊號功率比的估計值。K = 0 就是單純的逆濾波 1/H，其中 |H| 小於 10^−6 的頻率直接設為 0（與章節程式碼相同的慣例），所以它的最大增益停在約一百萬，σ 較大時即使沒有雜訊也無法完美還原。PSNR 是把各影像截斷到 0 到 1 之後與原影像比較。增益是 |H|/(|H|² + K) 在所有頻率上的最大值；K 大於 0 時，它不會超過 1/(2√K)。'),
  related: ['image-processing/dip-05-restoration-reconstruction'],
  controls: [
    { key: 'sigma', label: L('Blur: Gaussian PSF σ (pixels)', '模糊：高斯 PSF 的 σ（像素）'), kind: 'range', value: 2, min: 1, max: 3, step: 0.5 },
    { key: 'noise', label: L('Noise: standard deviation (fraction of the 0 to 1 range)', '雜訊：標準差（以 0 到 1 的範圍為單位）'), kind: 'range', value: 0.005, min: 0, max: 0.02, step: 0.001 },
    { key: 'logk', label: L('Wiener constant: log10 K (−7 means K = 0, the plain inverse)', 'Wiener 常數：log10 K（−7 代表 K = 0，即單純逆濾波）'), kind: 'range', value: LOGK_OFF, min: LOGK_OFF, max: LOGK_MAX, step: LOGK_STEP },
  ],
  predict: {
    question: L('The camera image is blurred (σ = 2) and a barely visible amount of noise is added (0.005, about one gray level in 255). The plain inverse filter (K = 0) divides every frequency by the blur’s transfer function. What does the result look like?', '相機影像被模糊（σ = 2），再加上幾乎看不見的雜訊（0.005，約 255 階中的 1 階）。單純逆濾波（K = 0）把每個頻率都除以模糊的轉移函數。結果會是什麼樣子？'),
    options: [
      { value: 'sharp', label: L('Sharper than the blurred image: it undoes the blur', '比模糊影像更清晰：它把模糊還原了') },
      { value: 'noisy', label: L('Worse than the blurred image: buried in amplified noise', '比模糊影像更差：被放大的雜訊淹沒') },
      { value: 'same', label: L('About the same as the blurred image', '和模糊影像差不多') },
    ],
    answer: 'noisy',
    explain: L('Blur makes the transfer function tiny at high frequencies, so dividing by it multiplies whatever is there by a huge number, and noise is spread evenly over all frequencies. With this noise the blurred image scores 23.95 dB, almost the same as without noise (23.98 dB), while the plain inverse scores 4.79 dB: worse than the blurred image, and it looks like random black and white dots. On a noise-free image the same filter reaches 34.94 dB, and noise of only 0.00001 already drops it to 6.1 dB. The failure comes from the noise, not from the blur.', '模糊使轉移函數在高頻變得極小，所以除以它會把該處的內容乘上一個巨大的數字，而雜訊平均分布在所有頻率。在這個雜訊下，模糊影像的 PSNR 是 23.95 dB，與沒有雜訊時的 23.98 dB 幾乎相同；單純逆濾波卻只有 4.79 dB，比模糊影像更差，看起來是隨機的黑白點。同一個濾波器用在沒有雜訊的影像上可達 34.94 dB，而只要雜訊有 0.00001，PSNR 就掉到 6.1 dB。失敗來自雜訊，而不是模糊。'),
  },
  assets: { 'camera-256': 'gray' },
  panes: [
    { key: 'original', kind: 'canvas', label: L('Original (known)', '原始影像（已知）'), width: 256, height: 256 },
    { key: 'degraded', kind: 'canvas', label: L('Blurred, plus noise', '模糊並加上雜訊'), width: 256, height: 256 },
    { key: 'restored', kind: 'canvas', label: L('Restored with H*/(|H|² + K)', '以 H*/(|H|² + K) 還原'), width: 256, height: 256 },
    { key: 'curve', kind: 'svg', label: L('PSNR of the restored image against K (the same blur and noise)', '還原影像的 PSNR 對 K 的曲線（相同的模糊與雜訊）'), width: 640, height: 280, wide: true },
  ],
  sources: [
    { title: 'Rafael C. Gonzalez & Richard E. Woods, Digital Image Processing, 4th ed., Pearson, 2018 (image restoration: inverse and Wiener filtering)', url: 'https://www.imageprocessingplace.com/DIP-4E/dip4e_main_page.htm' },
  ],
  create({ lang, images }) {
    const img = images['camera-256'] as Gray, p = makeProblem(img);
    let curveKey = '', curve: number[] = [];
    return {
      render(state, canvases) {
        const sigma = num(state, 'sigma'), noise = num(state, 'noise'), logk = num(state, 'logk'), K = kOf(logk);
        const ev = evaluate(p, sigma, noise, K);
        drawGray(canvases.original, img);
        drawGray(canvases.degraded, { w: p.w, h: p.h, data: ev.degraded });
        drawGray(canvases.restored, { w: p.w, h: p.h, data: ev.restored });
        // The curve depends on blur and noise only, so moving K just moves the marker.
        const key = `${sigma}|${noise}`;
        if (key !== curveKey) { curve = psnrCurve(p, sigma, noise); curveKey = key; }
        const best = bestOnGrid(curve);
        const X0 = 56, X1 = 620, Y0 = 28, Y1 = 226, top = Math.max(10, Math.ceil((Math.max(...curve, ev.psnrDegraded) + 1) / 5) * 5);
        const px = (lk: number) => X0 + ((lk - LOGK_OFF) / (LOGK_MAX - LOGK_OFF)) * (X1 - X0), py = (v: number) => Y1 - (Math.max(0, v) / top) * (Y1 - Y0);
        let plot = path([[X0, Y0 - 6], [X0, Y1], [X1, Y1]], 'stroke="var(--plot-grid)"');
        for (let v = 0; v <= top; v += 5) plot += path([[X0 - 4, py(v)], [X1, py(v)]], 'stroke="var(--plot-grid)" stroke-width="0.6"') + text(X0 - 8, py(v) + 4, v, 'text-anchor="end"');
        for (let lk = -6; lk <= 0; lk += 1) plot += text(px(lk), Y1 + 17, lk, 'text-anchor="middle"');
        plot += text(px(LOGK_OFF), Y1 + 17, 'K=0', 'text-anchor="middle"');
        plot += path([[X0, py(ev.psnrDegraded)], [X1, py(ev.psnrDegraded)]], 'stroke="var(--plot-muted)" stroke-dasharray="5 4" stroke-width="1.5"')
          + text(X0 + 12, py(ev.psnrDegraded) - 6, pick(lang, `blurred image: ${fmtDb(ev.psnrDegraded)}`, `模糊影像：${fmtDb(ev.psnrDegraded)}`), 'fill="var(--plot-muted)"');
        plot += path(LOGK_GRID.map((lk, i) => [px(lk), py(curve[i])] as [number, number]), 'stroke="var(--plot-a)" stroke-width="2.5"');
        plot += `<circle cx="${px(best.logk)}" cy="${py(best.psnr)}" r="5" fill="none" stroke="var(--plot-c)" stroke-width="2.5"/>`
          + text(Math.min(px(best.logk) + 9, X1 - 150), py(best.psnr) - 9, pick(lang, `best on grid: ${fmtDb(best.psnr)}`, `網格上的最佳值：${fmtDb(best.psnr)}`), 'fill="var(--plot-c)"');
        plot += path([[px(Math.max(logk, LOGK_OFF)), Y0 - 4], [px(Math.max(logk, LOGK_OFF)), Y1]], 'stroke="var(--plot-b)" stroke-width="2"')
          + `<circle cx="${px(logk)}" cy="${py(ev.psnrRestored)}" r="5.5" fill="var(--plot-b)"/>`;
        plot += text((X0 + X1) / 2, 268, pick(lang, 'log10 K (left end: K = 0, the plain inverse filter)', 'log10 K（最左端：K = 0，單純逆濾波）'), 'text-anchor="middle"')
          + text(12, 18, 'PSNR (dB)');
        const kLabel = K === 0 ? 'K = 0' : `K = 10^${fmt(logk, 2)}`;
        const bestLabel = best.logk <= LOGK_OFF ? 'K = 0' : `K = 10^${fmt(best.logk, 2)}`;
        const summary = K === 0
          ? pick(lang, `Plain inverse (K = 0): the filter multiplies some frequencies by up to ${fmtGain(ev.gain)}, so the restored image scores ${fmtDb(ev.psnrRestored)} against ${fmtDb(ev.psnrDegraded)} for the blurred one. The best K on the grid (${bestLabel}) reaches ${fmtDb(best.psnr)}.`,
              `單純逆濾波（K = 0）：濾波器把部分頻率放大到最多 ${fmtGain(ev.gain)}，因此還原影像的 PSNR 是 ${fmtDb(ev.psnrRestored)}，模糊影像則是 ${fmtDb(ev.psnrDegraded)}。網格上最好的 K（${bestLabel}）可達 ${fmtDb(best.psnr)}。`)
          : pick(lang, `${kLabel}: the peak gain is ${fmtGain(ev.gain)} and the restored image scores ${fmtDb(ev.psnrRestored)}, ${ev.psnrRestored >= ev.psnrDegraded ? 'better' : 'worse'} than the blurred image (${fmtDb(ev.psnrDegraded)}). The best K on the grid is ${bestLabel} with ${fmtDb(best.psnr)}.`,
              `${kLabel}：最大增益為 ${fmtGain(ev.gain)}，還原影像的 PSNR 是 ${fmtDb(ev.psnrRestored)}，${ev.psnrRestored >= ev.psnrDegraded ? '優於' : '不及'}模糊影像（${fmtDb(ev.psnrDegraded)}）。網格上最好的 K 是 ${bestLabel}，PSNR 為 ${fmtDb(best.psnr)}。`);
        return {
          summary,
          metrics: [
            { label: pick(lang, 'PSNR, blurred and noisy', 'PSNR，模糊且有雜訊'), value: fmtDb(ev.psnrDegraded) },
            { label: pick(lang, 'PSNR, restored', 'PSNR，還原後'), value: fmtDb(ev.psnrRestored) },
            { label: pick(lang, 'Noise gain, max |H|/(|H|² + K)', '雜訊增益，最大 |H|/(|H|² + K)'), value: fmtGain(ev.gain) },
          ],
          plots: { curve: svg(plot, 280) },
        };
      },
    };
  },
};
