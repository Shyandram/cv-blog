import type { LabDef } from './types.ts';
import { L, pick, fmt, num } from '../types.ts';
import type { Gray } from '../../imagekit/types.ts';
import { blockDct, blockIdct, haar2d, ihaar2d, quantizeUniform } from '../../imagekit/transform.ts';
import { psnr } from '../../imagekit/metrics.ts';
import { sobel } from '../../imagekit/filter.ts';
import { drawGray } from '../../imagekit/draw.ts';

export type TransformKind = 'dct' | 'haar';
/** Decomposition depth of the Haar transform (as the 5-level wavelet comparison in IP 08). */
export const HAAR_LEVELS = 5;
/** Side of the zoom box in pixels (a multiple of the 8-pixel DCT block) and its magnification. */
export const ZOOM = 32, SCALE = 6, RULER = 8;
/** Top-left corners of the three zoom boxes; all are multiples of 8 so that they follow the DCT block grid. */
export const REGIONS: Record<string, { x: number; y: number }> = {
  sky: { x: 128, y: 8 },
  edge: { x: 128, y: 64 },
  grass: { x: 168, y: 152 },
};

export const forward = (kind: TransformKind, data: Float32Array, w: number, h: number) => (kind === 'dct' ? blockDct(data, w, h) : haar2d(data, w, h, HAAR_LEVELS));
export const inverse = (kind: TransformKind, coef: Float32Array, w: number, h: number) => (kind === 'dct' ? blockIdct(coef, w, h) : ihaar2d(coef, w, h, HAAR_LEVELS));

export interface Coded {
  /** Quantized coefficients (what the decoder receives, times the step). */
  q: Float32Array;
  /** Fraction of coefficients that are not zero after quantization. */
  nonzero: number;
  /** Decoder output before clipping to [0, 1]. */
  recon: Float32Array;
  /** recon minus original. */
  error: Float32Array;
  psnr: number;
}
/** Transform (or reuse `coef`), quantize every coefficient with one uniform step, invert and score against the original. */
export function codeImage(kind: TransformKind, image: Gray, step: number, coef?: Float32Array): Coded {
  const c = coef ?? forward(kind, image.data, image.w, image.h);
  const { q, nonzero } = quantizeUniform(c, step);
  const recon = inverse(kind, q, image.w, image.h);
  const error = new Float32Array(recon.length);
  for (let i = 0; i < error.length; i++) error[i] = recon[i] - image.data[i];
  return { q, nonzero, recon, error, psnr: psnr(image.data, recon) };
}

/** 1 for the half of the pixels with the larger Sobel gradient magnitude (ties broken by position), 0 for the smoother half. */
export function busyMask(image: Gray): Uint8Array {
  const { ix, iy } = sobel(image.data, image.w, image.h), n = image.data.length, mag = new Float64Array(n);
  for (let i = 0; i < n; i++) mag[i] = Math.hypot(ix[i], iy[i]);
  const order = Array.from(mag.keys()).sort((a, b) => mag[a] - mag[b] || a - b), mask = new Uint8Array(n);
  for (let k = n >> 1; k < n; k++) mask[order[k]] = 1;
  return mask;
}
/** Mean absolute value of `values` over the pixels where `mask` equals `which`. */
export function meanAbsWhere(values: Float32Array, mask: Uint8Array, which: 0 | 1): number {
  let s = 0, n = 0;
  for (let i = 0; i < values.length; i++) if (mask[i] === which) { s += Math.abs(values[i]); n++; }
  return n ? s / n : 0;
}
/** Mean absolute value of `values` inside a square box of `size` pixels. */
export function meanAbsInBox(values: Float32Array, w: number, x0: number, y0: number, size = ZOOM): number {
  let s = 0;
  for (let y = y0; y < y0 + size; y++) for (let x = x0; x < x0 + size; x++) s += Math.abs(values[y * w + x]);
  return s / (size * size);
}
/**
 * How strongly the 8 × 8 grid shows: the mean absolute difference between neighbours that straddle a block edge divided by the
 * same quantity for neighbours inside a block, pooled over rows and columns. Close to 1 for a natural image.
 */
export function seamRatio(data: Float32Array, w: number, h: number): number {
  let edge = 0, nEdge = 0, inner = 0, nInner = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w - 1; x++) {
    const d = Math.abs(data[y * w + x + 1] - data[y * w + x]);
    if (x % 8 === 7) { edge += d; nEdge++; } else { inner += d; nInner++; }
  }
  for (let y = 0; y < h - 1; y++) for (let x = 0; x < w; x++) {
    const d = Math.abs(data[(y + 1) * w + x] - data[y * w + x]);
    if (y % 8 === 7) { edge += d; nEdge++; } else { inner += d; nInner++; }
  }
  return edge / nEdge / (inner / nInner);
}
/** Display values for the quantized coefficients: black for zero, otherwise 0.3 to 1 on a log scale of |q| / step. */
export function coefficientMap(q: Float32Array, step: number): Float32Array {
  let top = 0;
  for (const v of q) top = Math.max(top, Math.abs(v));
  const span = Math.log(top / step), out = new Float32Array(q.length);
  for (let i = 0; i < q.length; i++) out[i] = q[i] === 0 ? 0 : 0.3 + 0.7 * (span > 0 ? Math.log(Math.abs(q[i]) / step) / span : 1);
  return out;
}
/** Error image: signed error times `gain` around mid-gray, clamped to [-1, 1] (drawGray maps it with `signed`). */
export function errorView(error: Float32Array, gain = 10): Float32Array {
  return error.map(v => Math.max(-1, Math.min(1, v * gain)));
}
/** One ZOOM × ZOOM crop magnified SCALE times under a ruler whose cells are 8 source pixels wide (ruler only for the DCT). */
export function zoomCrop(src: Float32Array, w: number, x0: number, y0: number, ruler: boolean): Gray {
  const side = ZOOM * SCALE, H = side + RULER, data = new Float32Array(side * H).fill(0.5);
  for (let y = 0; y < side; y++) for (let x = 0; x < side; x++) data[(RULER + y) * side + x] = Math.max(0, Math.min(1, src[(y0 + Math.floor(y / SCALE)) * w + x0 + Math.floor(x / SCALE)]));
  if (ruler) for (let y = 0; y < RULER; y++) for (let x = 0; x < side; x++) data[y * side + x] = Math.floor(x / (8 * SCALE)) % 2 ? 0.9 : 0.1;
  return { w: side, h: H, data };
}

const GRAY_LEVELS = 255;
/** Seam ratio from which the summary talks about visible block seams (about twice a typical jump). */
export const SEAM_VISIBLE = 2;
export const lab: LabDef = {
  kind: 'image', id: 'transform-coding', track: 'image-processing',
  title: L('IP 08 · How many coefficients can you throw away?', 'IP 08・可以丟掉多少係數？'),
  description: L('Transform a real photo with an 8 × 8 block DCT or a Haar wavelet, round every coefficient to a coarse step, and see where the lost information shows up.', '用 8 × 8 區塊 DCT 或 Haar 小波轉換一張真實照片，把每個係數捨入到較粗的步長，觀察遺失的資訊出現在哪裡。'),
  challenge: L('Set the DCT step to 0.6 and look at the sky in the zoom box. Then switch to Haar at the same step: does the staircase go away, and which transform keeps fewer nonzero coefficients at about the same PSNR? Finally move the zoom box to the grass: how much of the texture is left?', '把 DCT 的步長設為 0.6，看放大框裡的天空。接著在相同步長下切換成 Haar：階梯狀方塊消失了嗎？在 PSNR 差不多時，哪一種轉換保留的非零係數較少？最後把放大框移到草地：還剩下多少紋理？'),
  note: L('A real 256 × 256 version of the camera test image (a 2 × 2 average of skimage.data.camera), gray values in [0, 1]. Both transforms are orthonormal: an 8 × 8 block DCT (each block coded on its own) and a 5-level 2-D Haar wavelet transform. The quantizer is uniform: every coefficient c becomes round(c / step) × step, with the same step for all coefficients including the block averages, so a coefficient survives only if |c| is at least step / 2. This is simpler than the frequency-dependent table of JPEG in IP 08, so the pictures show what the transform and the rounding do on their own. "Nonzero" counts surviving coefficients. It is not a bit rate: no entropy coder is implemented, and the real cost also depends on which coefficients survive and how they are coded. PSNR has peak 1 and uses the decoder output before clipping. Because both transforms are orthonormal, the squared error in pixels equals the squared error in the coefficients: what the quantizer removes is exactly the error, and the decoder cannot bring it back, because many different images give the same quantized coefficients. A bottleneck autoencoder (ML 06A) loses detail in the same way, except that its basis is learned from data instead of fixed. The error image shows (reconstruction − original) × 10 around mid-gray. "Smoothest" and "busiest" halves are the pixels with the smaller and larger Sobel gradient of the original. Haar is the simplest wavelet and is piecewise constant, so it also produces blocks, of several sizes; smoother wavelets such as the one in IP 08’s wavelet comparison blur instead.', '使用 camera 測試影像的真實 256 × 256 版本（skimage.data.camera 以 2 × 2 平均縮小），灰階值在 [0, 1]。兩種轉換都是正交的：8 × 8 區塊 DCT（每個區塊各自編碼）與 5 層二維 Haar 小波轉換。量化器是均勻的：每個係數 c 變成 round(c / 步長) × 步長，所有係數（包含區塊平均值）使用相同步長，因此只有 |c| 至少為步長的一半時係數才會留下。這比 IP 08 中 JPEG 隨頻率變化的量化表更簡單，所以畫面呈現的是轉換與捨入本身造成的效果。「非零」是計算留下的係數。這不是位元率：此處沒有實作熵編碼，真正的成本還取決於哪些係數留下以及如何編碼。PSNR 的峰值為 1，並使用截斷前的解碼輸出。因為兩種轉換都是正交的，像素的平方誤差等於係數的平方誤差：量化器移除的部分正是誤差，而解碼器無法把它找回來，因為許多不同的影像會得到相同的量化係數。瓶頸自編碼器（ML 06A）以同樣的方式遺失細節，差別只在它的基底是從資料學得，而不是固定的。誤差影像顯示（重建 − 原始）× 10，以中間灰色為零。「最平滑」與「最繁雜」兩半是指原始影像 Sobel 梯度較小與較大的像素。Haar 是最簡單的小波，由分段常數組成，所以它也會產生方塊，只是大小不一；較平滑的小波（例如 IP 08 小波比較中所用的）則是變模糊。'),
  related: ['image-processing/dip-08-compression-watermarking', 'image-processing/dip-07-wavelets-transforms', 'ml/ml-05a-factor-models-autoencoders'],
  controls: [
    { key: 'transform', label: L('Transform', '轉換'), kind: 'select', value: 'dct', options: [
      { value: 'dct', label: L('8 × 8 block DCT', '8 × 8 區塊 DCT') }, { value: 'haar', label: L('Haar wavelet, 5 levels', 'Haar 小波，5 層') }] },
    { key: 'step', label: L('Quantizer step (larger keeps fewer coefficients)', '量化步長（越大，留下的係數越少）'), kind: 'range', value: 0.2, min: 0.02, max: 0.8, step: 0.02 },
    { key: 'zoom', label: L('Zoom box', '放大框'), kind: 'select', value: 'sky', options: [
      { value: 'sky', label: L('Sky (smooth)', '天空（平滑）') }, { value: 'edge', label: L('Camera (edges)', '相機（邊緣）') }, { value: 'grass', label: L('Grass (texture)', '草地（紋理）') }] },
  ],
  predict: {
    question: L('You keep the default setting, where most coefficients are rounded to zero. Where in the picture will the reconstruction error be largest?', '維持預設設定，大多數係數都被捨入為零。重建誤差在畫面的哪些地方最大？'),
    options: [
      { value: 'flat', label: L('In smooth areas such as the sky, because they cover the most area', '在天空這類平滑區域，因為它們的面積最大') },
      { value: 'busy', label: L('At edges and in texture, where the picture changes quickly', '在邊緣與紋理，也就是畫面變化很快的地方') },
      { value: 'even', label: L('Evenly everywhere, because every coefficient is rounded with the same step', '到處一樣，因為每個係數都用相同步長捨入') },
    ],
    answer: 'busy',
    explain: L('A smooth patch is described almost completely by its average, so rounding costs little there. An edge or a texture spreads its energy over many coefficients, and many of them are smaller than half a step, so they become zero and the detail they carried is gone. At the default setting (DCT, step 0.2) the mean absolute error is 2.8 gray levels on the smoothest half of the pixels and 7.5 on the busiest half, about 2.7 times more. The smooth half is not error-free: rounding still shifts each block average a little, which is what draws the 8 × 8 seams at coarse steps.', '平滑的小區塊幾乎只靠平均值就能描述，所以在那裡捨入的代價很小。邊緣或紋理會把能量分散到許多係數，其中很多比半個步長還小，因此變成零，它們攜帶的細節也就消失了。在預設設定（DCT，步長 0.2）下，最平滑的一半像素平均絕對誤差為 2.8 個灰階，最繁雜的一半為 7.5，約為 2.7 倍。平滑的一半並非沒有誤差：捨入仍會讓每個區塊的平均值稍微偏移，這就是步長較粗時出現 8 × 8 接縫的原因。'),
  },
  assets: { 'camera-256': 'gray' },
  paneMin: 230,
  panes: [
    { key: 'original', kind: 'canvas', label: L('Original (yellow box = zoom)', '原始影像（黃框 = 放大處）'), width: 256, height: 256 },
    { key: 'recon', kind: 'canvas', label: L('Reconstruction from the quantized coefficients', '由量化後係數重建的影像'), width: 256, height: 256 },
    { key: 'coef', kind: 'canvas', label: L('Surviving coefficients, log scale, zeros black (DCT: one 8 × 8 tile per block, block average top-left; Haar: coarsest average top-left, finer details outward)', '留下的係數（對數尺度，零為黑色）。DCT：每個區塊一個 8 × 8 方格，區塊平均值在左上；Haar：最粗的平均值在左上，越細的細節越往外'), width: 256, height: 256 },
    { key: 'error', kind: 'canvas', label: L('Error: reconstruction − original, × 10, gray = 0', '誤差：重建 − 原始，× 10，灰色 = 0'), width: 256, height: 256 },
    { key: 'zoomOriginal', kind: 'canvas', label: L('Zoom (32 × 32 pixels, 6 ×): original', '放大（32 × 32 像素的 6 倍）：原始'), width: ZOOM * SCALE, height: ZOOM * SCALE + RULER },
    { key: 'zoomRecon', kind: 'canvas', label: L('Zoom: reconstruction (DCT: the ruler marks the 8-pixel blocks)', '放大：重建（DCT：尺標標示 8 像素的區塊）'), width: ZOOM * SCALE, height: ZOOM * SCALE + RULER },
  ],
  sources: [
    { title: 'Rafael C. Gonzalez & Richard E. Woods, Digital Image Processing, 4th ed., Pearson, 2018 (compression, transforms)', url: 'https://www.imageprocessingplace.com/DIP-4E/dip4e_main_page.htm' },
    { title: 'N. Ahmed, T. Natarajan and K. R. Rao, Discrete Cosine Transform, IEEE Transactions on Computers, C-23(1), 1974', url: 'https://doi.org/10.1109/T-C.1974.223784' },
    { title: 'Alfred Haar, Zur Theorie der orthogonalen Funktionensysteme, Mathematische Annalen 69(3), 1910', url: 'https://doi.org/10.1007/BF01456326' },
    { title: 'Gregory K. Wallace, The JPEG still picture compression standard, IEEE Transactions on Consumer Electronics 38(1), 1992', url: 'https://doi.org/10.1109/30.125072' },
  ],
  create({ lang, images }) {
    const image = images['camera-256'] as Gray, { w, h } = image, busy = busyMask(image);
    const coefs: Partial<Record<TransformKind, Float32Array>> = {};
    return {
      render(state, canvases) {
        const kind: TransformKind = state.transform === 'haar' ? 'haar' : 'dct', step = num(state, 'step');
        const region = REGIONS[String(state.zoom)] ?? REGIONS.sky;
        const coef = (coefs[kind] ??= forward(kind, image.data, w, h));
        const r = codeImage(kind, image, step, coef);
        drawGray(canvases.original, image);
        const ctx = canvases.original.getContext('2d');
        if (ctx) { ctx.strokeStyle = '#ffd400'; ctx.lineWidth = 2; ctx.strokeRect(region.x - 1, region.y - 1, ZOOM + 2, ZOOM + 2); }
        drawGray(canvases.recon, { w, h, data: r.recon });
        drawGray(canvases.coef, { w, h, data: coefficientMap(r.q, step) });
        drawGray(canvases.error, { w, h, data: errorView(r.error) }, true);
        drawGray(canvases.zoomOriginal, zoomCrop(image.data, w, region.x, region.y, kind === 'dct'));
        drawGray(canvases.zoomRecon, zoomCrop(r.recon, w, region.x, region.y, kind === 'dct'));
        const smooth = meanAbsWhere(r.error, busy, 0) * GRAY_LEVELS, busier = meanAbsWhere(r.error, busy, 1) * GRAY_LEVELS;
        const box = meanAbsInBox(r.error, w, region.x, region.y) * GRAY_LEVELS, seam = seamRatio(r.recon, w, h);
        const ratio = smooth > 0 ? busier / smooth : NaN, pct = (100 * r.nonzero).toFixed(1);
        const name = kind === 'dct' ? pick(lang, '8 × 8 DCT', '8 × 8 DCT') : pick(lang, 'Haar', 'Haar');
        let summary = pick(lang,
          `${name}, step ${fmt(step, 2)}: ${pct}% of the coefficients survive and PSNR is ${fmt(r.psnr, 1)} dB. The error is ${fmt(ratio, 1)} times larger on the busiest half of the pixels than on the smoothest half. `,
          `${name}，步長 ${fmt(step, 2)}：留下 ${pct}% 的係數，PSNR 為 ${fmt(r.psnr, 1)} dB。最繁雜的一半像素，誤差是最平滑一半的 ${fmt(ratio, 1)} 倍。`);
        if (kind === 'dct') summary += seam >= SEAM_VISIBLE
          ? pick(lang, `Block seams now show: a jump across an 8-pixel block edge is ${fmt(seam, 1)} times a typical jump inside a block (the sky in the zoom shows it best).`, `區塊接縫現在很明顯：跨越 8 像素區塊邊界的跳變是區塊內一般跳變的 ${fmt(seam, 1)} 倍（放大框裡的天空最容易看出來）。`)
          : pick(lang, `The block seams are still faint (edge jump ${fmt(seam, 1)} times the inner jump); raise the step to see them.`, `區塊接縫仍不明顯（邊界跳變是內部跳變的 ${fmt(seam, 1)} 倍）；提高步長就會看到。`);
        else summary += seam >= SEAM_VISIBLE
          ? pick(lang, 'Haar pieces are constant on blocks of 2, 4, 8, ... pixels, so a coarse step leaves a staircase of blocks of several sizes instead of one 8 × 8 grid.', 'Haar 的基底在 2、4、8…… 像素的區塊上是常數，所以步長較粗時會留下大小不一的階梯狀方塊，而不是單一的 8 × 8 格線。')
          : pick(lang, 'At this step the Haar pieces are too fine to see as blocks; raise the step.', '在這個步長下，Haar 的片段太細，看不出方塊；請提高步長。');
        return {
          summary,
          metrics: [
            { label: pick(lang, 'Nonzero coefficients (not a bit rate)', '非零係數比例（不是位元率）'), value: `${pct}%` },
            { label: 'PSNR', value: `${fmt(r.psnr, 1)} dB` },
            { label: pick(lang, 'Mean |error|, smoothest / busiest half (gray levels)', '平均絕對誤差，最平滑／最繁雜的一半（灰階）'), value: `${smooth.toFixed(1)} / ${busier.toFixed(1)}` },
            { label: pick(lang, 'Mean |error| inside the zoom box (gray levels)', '放大框內的平均絕對誤差（灰階）'), value: box.toFixed(1) },
          ],
        };
      },
    };
  },
};
