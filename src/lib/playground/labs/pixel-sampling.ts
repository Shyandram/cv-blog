import type { LabDef } from './types.ts';
import { L, pick, fmt, num } from '../types.ts';
import type { Gray } from '../../imagekit/types.ts';
import { gaussianBlur } from '../../imagekit/filter.ts';
import { drawGray } from '../../imagekit/draw.ts';
import { psnr } from '../../imagekit/metrics.ts';

/**
 * Geometry of the synthetic zone plate (scripts/figures/playground_photo_assets.py): I(r) = 0.5 + 0.5 cos(k r^2) with r measured
 * from the centre (127.5, 127.5) and k = pi / (2 r_max), r_max being the distance to a corner. Local frequency at radius r is k r / pi
 * cycles per pixel, reaching 0.5 (the Nyquist limit of the full grid) exactly at the corner.
 */
export const ZONE = { size: 256, centre: 127.5, k: Math.PI / (2 * Math.SQRT2 * 127.5) };
/** Radius inside which a grid that keeps every `stride`-th pixel (Nyquist limit 1 / (2 stride) cycles per pixel) can still show the true pattern. */
export const aliasRadius = (stride: number) => Math.PI / (2 * stride * ZONE.k);

/** Gaussian pre-filter width used by the lab: half the stride in pixels (no filtering at stride 1). */
export const prefilterSigma = (stride: number) => (stride > 1 ? 0.5 * stride : 0);

/** Uniform quantizer used in the Images as Arrays chapter: keep `bits` bits and return the centre of each bin, for byte values 0..255. */
export function quantizeByte(v: number, bits: number): number {
  const step = 2 ** (8 - bits);
  return Math.floor(v / step) * step + Math.floor(step / 2);
}
export interface Sampled {
  /** Kept samples (bytes) on the coarse grid. */ small: Uint8Array; sw: number; sh: number;
  /** Nearest-neighbour enlargement back to the original size, in [0, 1]. */ up: Float32Array;
}
/** Optional Gaussian anti-alias filter, keep every `stride`-th pixel (row 0, column 0 of each block), quantize to `bits`, enlarge by pixel replication. */
export function samplePhoto(img: Gray, stride: number, bits: number, prefilter: boolean): Sampled {
  const { w, h } = img, s = Math.max(1, Math.round(stride));
  const filtered = prefilter && s > 1 ? gaussianBlur(img.data, w, h, prefilterSigma(s)) : img.data;
  const sw = Math.ceil(w / s), sh = Math.ceil(h / s), small = new Uint8Array(sw * sh);
  for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
    const v = Math.max(0, Math.min(255, Math.round(255 * filtered[y * s * w + x * s])));
    small[y * sw + x] = quantizeByte(v, bits);
  }
  const up = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) up[y * w + x] = small[Math.floor(y / s) * sw + Math.floor(x / s)] / 255;
  return { small, sw, sh, up };
}
/**
 * Contrast (standard deviation, in [0, 1] units) of the kept samples that lie where the zone plate is finer than the coarse grid can show.
 * A faithful low-resolution picture of that region is flat gray, so any contrast there is a false pattern. Returns NaN at stride 1 (no such region).
 */
export function falsePatternContrast(s: Sampled, stride: number): number {
  const rc = aliasRadius(stride), vals: number[] = [];
  for (let y = 0; y < s.sh; y++) for (let x = 0; x < s.sw; x++) {
    if (Math.hypot(x * stride - ZONE.centre, y * stride - ZONE.centre) > rc) vals.push(s.small[y * s.sw + x] / 255);
  }
  if (!vals.length) return NaN;
  const m = vals.reduce((a, b) => a + b, 0) / vals.length;
  return Math.sqrt(vals.reduce((a, b) => a + (b - m) ** 2, 0) / vals.length);
}
const db = (v: number, lang: 'en' | 'zh-tw') => (Number.isFinite(v) ? `${v.toFixed(1)} dB` : pick(lang, '∞ (identical)', '∞（完全相同）'));

export const lab: LabDef = {
  kind: 'image', id: 'pixel-sampling', track: 'image-processing',
  title: L('IP 02 · What do you lose with fewer pixels, and what with fewer gray levels?', 'IP 02・減少像素與減少灰階，各會失去什麼？'),
  description: L('Keep every s-th pixel and fewer gray levels of a real photo and of a synthetic zone plate, and see false patterns (aliasing) appear unless you filter first.', '對真實照片與合成的波帶片，只保留每隔 s 個像素與較少的灰階，觀察不先濾波時出現的假圖樣（混疊）。'),
  challenge: L('Set the stride to 1 and the bits to 3 (8 gray levels): the photo shows bands, yet its PSNR is higher than with stride 4 at 8 bits, and even higher than with stride 2. Which loss does PSNR punish more? Then set the bits back to 8 and compare the zone plate at strides 2, 4 and 8: where do the false rings begin, and how does that radius shrink with the stride? Finally turn the Gaussian prefilter on.', '把取樣間距設為 1、位元數設為 3（8 個灰階）：照片會出現色帶，但它的 PSNR 仍高於間距 4、8 位元的結果，甚至高於間距 2。PSNR 對哪一種損失的懲罰比較重？接著把位元數調回 8，比較波帶片在間距 2、4、8 時的結果：假圓環從哪裡開始？這個半徑如何隨間距縮小？最後打開高斯前置濾波。'),
  note: L('Photo: the camera test image, 256 × 256. Zone plate: synthetic, 256 × 256, I(r) = 0.5 + 0.5 cos(k r²) with r the distance from the centre and k = π / (2 r_max), so its local frequency rises linearly to 0.5 cycles per pixel at the corner. Sampling keeps pixel (0, 0) of every stride × stride block (a[::s, ::s]); gray levels use the chapter’s rule of rounding to the centre of one of 2^bits bins; the result is enlarged by pixel replication (nearest neighbour) so single samples stay visible. The optional prefilter is a Gaussian with σ = stride / 2 applied before sampling. PSNR compares the enlarged result with the original 8-bit image (peak 255). False-pattern contrast is the standard deviation of the kept zone-plate samples that lie beyond radius π / (2 · stride · k), where the true pattern is finer than the coarse grid can show; a faithful result is flat gray there, so a large value means false rings. PSNR cannot tell a false pattern from honest blur, which is why both numbers are shown.', '照片：相機測試影像，256 × 256。波帶片：合成影像，256 × 256，I(r) = 0.5 + 0.5 cos(k r²)，r 為到中心的距離，k = π / (2 r_max)，因此它的局部頻率線性上升，在角落達到每像素 0.5 週期。取樣保留每個間距 × 間距區塊的 (0, 0) 像素（a[::s, ::s]）；灰階依章節的規則，四捨五入到 2^bits 個區間之一的中心；結果以像素複製（最近鄰）放大，讓單一取樣點仍然清楚可見。可選的前置濾波是取樣前套用 σ = 間距 / 2 的高斯濾波器。PSNR 是把放大後的結果與原始 8 位元影像（峰值 255）比較。假圖樣對比是半徑超過 π / (2 · 間距 · k) 處、保留下來的波帶片取樣點的標準差，這個範圍內真實圖樣比粗網格能表示的更細；忠實的結果在那裡應為均勻灰色，所以數值大代表出現假圓環。PSNR 分不出假圖樣與誠實的模糊，所以兩個數字都會顯示。'),
  related: ['foundations/images-as-arrays', 'image-processing/dip-02-fundamentals', 'deep-learning/dl-04a-super-resolution', 'deep-learning/dl-04b-video-super-resolution'],
  controls: [
    { key: 'stride', label: L('Keep every s-th pixel (stride s)', '每隔 s 個像素保留一個（間距 s）'), kind: 'range', value: 4, min: 1, max: 8, step: 1 },
    { key: 'bits', label: L('Bits per pixel', '每像素位元數'), kind: 'range', value: 8, min: 1, max: 8, step: 1 },
    { key: 'prefilter', label: L('Before sampling', '取樣之前'), kind: 'select', value: 'none', options: [
      { value: 'none', label: L('Nothing (plain point sampling)', '不處理（單純點取樣）') }, { value: 'gaussian', label: L('Gaussian blur, σ = stride / 2', '高斯模糊，σ = 間距 / 2') }] },
  ],
  predict: {
    question: L('You keep every 4th pixel in each direction. Does a blur before sampling help or hurt?', '你在水平與垂直方向每隔 4 個像素保留一個。取樣前先模糊，會有幫助還是有害？'),
    options: [
      { value: 'hurts', label: L('It hurts: blurring throws detail away, so PSNR drops on both images', '有害：模糊會丟掉細節，所以兩張影像的 PSNR 都下降') },
      { value: 'trade', label: L('A trade-off: PSNR drops, but the false rings disappear', '有得有失：PSNR 下降，但假圓環消失') },
      { value: 'helps', label: L('It helps: PSNR rises on both images and the false rings nearly vanish', '有幫助：兩張影像的 PSNR 都上升，假圓環幾乎消失') },
    ],
    answer: 'helps',
    explain: L('Point sampling keeps one pixel and ignores the other fifteen in its block, so fine detail is not removed but misread as a coarse pattern. A Gaussian with σ = 2 first replaces each pixel by a local average, which is a better guess for the whole block. On the photo the PSNR rises from 19.6 to 21.6 dB, on the zone plate from 6.4 to 9.2 dB, and the contrast of the false rings (beyond radius 45 px, where the true pattern is finer than a stride-4 grid can show) falls from 0.35 to 0.02. Honest limits: the blur restores no detail, so the zone plate is still poor, and the picture is softer. Across strides 2 to 8 and 2 to 8 bits the prefilter raised PSNR on both images in every case; only at 1 bit did it lower the PSNR of the zone plate slightly.', '點取樣只保留區塊內的一個像素，忽略其他十五個，所以細節不是被移除，而是被誤讀成粗大的圖樣。先用 σ = 2 的高斯濾波器，會把每個像素換成局部平均，這對整個區塊來說是比較好的估計。照片的 PSNR 從 19.6 升到 21.6 dB，波帶片從 6.4 升到 9.2 dB，假圓環（半徑超過 45 像素處，那裡的真實圖樣比間距 4 的網格能表示的更細）的對比從 0.35 降到 0.02。誠實的限制：模糊不會找回任何細節，所以波帶片仍然很差，畫面也變得較柔和。在間距 2 到 8、位元數 2 到 8 的所有組合中，前置濾波都讓兩張影像的 PSNR 上升；只有在 1 位元時，波帶片的 PSNR 略為下降。'),
  },
  assets: { 'camera-256': 'gray', 'zoneplate-256': 'gray' },
  paneMin: 200,
  panes: [
    { key: 'photo', kind: 'canvas', label: L('Photo: original', '照片：原始'), width: 256, height: 256 },
    { key: 'photoOut', kind: 'canvas', label: L('Photo: kept samples, enlarged', '照片：保留的取樣點，放大後'), width: 256, height: 256 },
    { key: 'zone', kind: 'canvas', label: L('Zone plate (synthetic): original', '波帶片（合成）：原始'), width: 256, height: 256 },
    { key: 'zoneOut', kind: 'canvas', label: L('Zone plate: kept samples, enlarged', '波帶片：保留的取樣點，放大後'), width: 256, height: 256 },
  ],
  sources: [
    { title: 'Rafael C. Gonzalez & Richard E. Woods, Digital Image Processing, 4th ed., Pearson, 2018, Chapter 2 (sampling and quantization)', url: 'https://www.pearson.com/en-us/subject-catalog/p/digital-image-processing/P200000003249/9780133356724' },
  ],
  create({ lang, images }) {
    const photo = images['camera-256'] as Gray, zone = images['zoneplate-256'] as Gray;
    let styled = false;
    return {
      render(state, canvases) {
        if (!styled) { for (const c of Object.values(canvases)) c.style.imageRendering = 'pixelated'; styled = true; }
        const stride = Math.max(1, Math.min(8, Math.round(num(state, 'stride')))), bits = Math.max(1, Math.min(8, Math.round(num(state, 'bits')))), prefilter = state.prefilter === 'gaussian';
        const a = samplePhoto(photo, stride, bits, prefilter), b = samplePhoto(zone, stride, bits, prefilter);
        drawGray(canvases.photo, photo); drawGray(canvases.photoOut, { w: photo.w, h: photo.h, data: a.up });
        drawGray(canvases.zone, zone); drawGray(canvases.zoneOut, { w: zone.w, h: zone.h, data: b.up });
        const pa = psnr(photo.data, a.up), pb = psnr(zone.data, b.up), fp = falsePatternContrast(b, stride), kept = a.sw * a.sh;
        const rc = aliasRadius(stride);
        const where = stride === 1
          ? pick(lang, 'At stride 1 every pixel is kept, so only the gray levels matter.', '間距為 1 時保留每個像素，所以只有灰階數有影響。')
          : pick(lang, `Beyond radius ${fmt(rc, 0)} px the zone plate is finer than a stride-${stride} grid can show, so any rings there are false. ${prefilter ? `With the Gaussian prefilter their contrast is ${fmt(fp, 2)}.` : `Without a prefilter their contrast is ${fmt(fp, 2)}.`}`, `半徑超過 ${fmt(rc, 0)} 像素後，波帶片比間距 ${stride} 的網格能表示的更細，所以那裡的圓環都是假的。${prefilter ? `使用高斯前置濾波時，它們的對比是 ${fmt(fp, 2)}。` : `沒有前置濾波時，它們的對比是 ${fmt(fp, 2)}。`}`);
        return {
          summary: pick(lang, `Keeping ${a.sw} × ${a.sh} samples at ${2 ** bits} gray levels gives PSNR ${db(pa, lang)} on the photo and ${db(pb, lang)} on the zone plate. ${where}`, `保留 ${a.sw} × ${a.sh} 個取樣點、${2 ** bits} 個灰階時，照片的 PSNR 為 ${db(pa, lang)}，波帶片為 ${db(pb, lang)}。${where}`),
          metrics: [
            { label: pick(lang, 'PSNR, photo', 'PSNR，照片'), value: db(pa, lang) },
            { label: pick(lang, 'PSNR, zone plate', 'PSNR，波帶片'), value: db(pb, lang) },
            { label: pick(lang, 'False-pattern contrast, zone plate', '假圖樣對比，波帶片'), value: Number.isFinite(fp) ? fmt(fp, 2) : '—' },
            { label: pick(lang, 'Gray levels', '灰階數'), value: String(2 ** bits) },
            { label: pick(lang, 'Samples kept', '保留的取樣點'), value: `${a.sw} × ${a.sh} = ${kept.toLocaleString('en-US')}` },
          ],
        };
      },
    };
  },
};
