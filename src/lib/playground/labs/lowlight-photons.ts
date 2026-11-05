import type { LabDef } from './types.ts';
import { L, pick, fmt, num } from '../types.ts';
import type { Gray } from '../../imagekit/types.ts';
import { clip01 } from '../../imagekit/types.ts';
import { uniformField, normalField, poissonFromUniform } from '../../imagekit/rng.ts';
import { gaussianBlur, median3 } from '../../imagekit/filter.ts';
import { drawGray } from '../../imagekit/draw.ts';
import { psnr, meanOf, stdOf } from '../../imagekit/metrics.ts';

/** Electrons that correspond to white at digital gain 1 (the fixed full-scale reference of the simulated camera). */
export const FULL_SCALE = 100;
export const READ_NOISE = 2;
export const SEEDS = { poisson: 11, read: 12 };

export interface Capture { electrons: Float32Array; display: Float32Array; snrDb: number; clipped: number }
/** Poisson photon noise plus Gaussian read noise, then digital gain. Fixed noise fields: moving a slider never re-rolls the dice. */
export function capture(img: Float32Array, photons: number, gain: number, u: Float32Array, z: Float32Array): Capture {
  const electrons = new Float32Array(img.length), display = new Float32Array(img.length), residual = new Float32Array(img.length);
  let clipped = 0;
  for (let i = 0; i < img.length; i++) {
    const mean = img[i] * photons;
    electrons[i] = poissonFromUniform(mean, u[i]) + READ_NOISE * z[i];
    residual[i] = electrons[i] - mean;
    const v = gain * electrons[i] / FULL_SCALE;
    if (v > 1) clipped++;
    display[i] = clip01(v);
  }
  // Gain multiplies signal and noise alike, so the ratio is computed in electrons and does not depend on it.
  return { electrons, display, snrDb: 20 * Math.log10((meanOf(img) * photons) / stdOf(residual)), clipped: clipped / img.length };
}
export function enhance(kind: string, display: Float32Array, w: number, h: number): Float32Array {
  return kind === 'gaussian' ? gaussianBlur(display, w, h, 1.5) : kind === 'median' ? median3(display, w, h) : display;
}

export const lab: LabDef = {
  kind: 'image', id: 'lowlight-photons', track: 'deep-learning',
  title: L('DLCV 09C · Gain cannot create missing photons', 'DLCV 09C・增益無法創造缺少的光子'),
  description: L('Simulate a dim photo with photon and read noise, then compare more light, more digital gain and a denoiser.', '模擬帶有光子雜訊與讀出雜訊的昏暗照片，比較增加光量、增加數位增益與使用去雜訊器的差別。'),
  challenge: L('At 8 photons and gain 12 the picture is as bright as the scene, yet very noisy. Lower the gain to 1: the picture goes dark, but does the signal-to-noise ratio change? Now keep gain 12 and raise the photons to 100. What changes instead, and what happens to the highlights?', '在 8 個光子、增益 12 時，畫面和場景一樣亮，卻有很多雜訊。把增益降到 1：畫面變暗，但訊雜比有變嗎？接著保持增益 12，把光子數提高到 100。這次改變了什麼？亮部又會發生什麼事？'),
  note: L('Camera test image, 256 × 256. Each pixel collects Poisson(photons × scene) electrons plus Gaussian read noise (σ = 2 electrons), then the output is clip(gain × electrons / 100). Noise fields are fixed, so sliders never re-roll them. SNR is the mean signal over the standard deviation of the noise, measured before clipping, in dB. The enhancers are a Gaussian blur (σ = 1.5) and a 3 × 3 median, stand-ins for a learned denoiser; they are not one. Large photon counts use a normal approximation of the Poisson distribution.', '相機測試影像，256 × 256。每個像素收集 Poisson(光子數 × 場景) 個電子，再加上高斯讀出雜訊（σ = 2 個電子），輸出為 clip(增益 × 電子數 / 100)。雜訊場固定不變，所以拉動滑桿不會重新抽樣。SNR 是截斷前的平均訊號除以雜訊標準差，單位為 dB。增強器是高斯模糊（σ = 1.5）與 3 × 3 中值濾波，用來代替學習式去雜訊器，但它們並不是。光子數很大時使用 Poisson 分布的常態近似。'),
  related: ['deep-learning/dl-05-low-light-enhancement'],
  controls: [
    { key: 'photons', label: L('Exposure: photons per pixel at full white', '曝光量：全白處每像素的光子數'), kind: 'range', value: 8, min: 2, max: 100, step: 1 },
    { key: 'gain', label: L('Digital gain', '數位增益'), kind: 'range', value: 12, min: 1, max: 32, step: 1 },
    { key: 'enhancer', label: L('Enhancer', '增強器'), kind: 'select', value: 'none', options: [
      { value: 'none', label: L('None', '無') }, { value: 'gaussian', label: L('Gaussian blur σ = 1.5', '高斯模糊 σ = 1.5') }, { value: 'median', label: L('Median 3 × 3', '中值 3 × 3') }] },
  ],
  predict: {
    question: L('You double the digital gain while nothing clips. What happens to the signal-to-noise ratio?', '在沒有任何像素被截斷的情況下，你把數位增益加倍。訊雜比會怎麼變？'),
    options: [{ value: 'better', label: L('It improves', '變好') }, { value: 'same', label: L('It stays the same', '不變') }, { value: 'worse', label: L('It gets worse', '變差') }],
    answer: 'same',
    explain: L('Gain multiplies the signal and the noise by the same factor, so their ratio does not move; the picture only looks brighter and its noise looks larger. The ratio improves only when more photons are collected: signal grows with the photon count n while photon noise grows only with its square root.', '增益把訊號與雜訊乘上同一個倍數，兩者的比值不會改變；畫面只是變亮，雜訊也看起來更大。只有收集到更多光子時比值才會改善：訊號隨光子數 n 成長，而光子雜訊只隨其平方根成長。'),
  },
  assets: { 'camera-256': 'gray' },
  panes: [
    { key: 'scene', kind: 'canvas', label: L('The scene (what we want)', '場景（我們想要的）'), width: 256, height: 256 },
    { key: 'capture', kind: 'canvas', label: L('Sensor output after gain', '增益後的感光元件輸出'), width: 256, height: 256 },
    { key: 'enhanced', kind: 'canvas', label: L('After the enhancer', '增強器處理後'), width: 256, height: 256 },
  ],
  sources: [
    { title: 'Alessandro Foi, Mejdi Trimeche, Vladimir Katkovnik & Karen Egiazarian, Practical Poissonian-Gaussian Noise Modeling and Fitting for Single-Image Raw-Data, IEEE Transactions on Image Processing, 2008', url: 'https://doi.org/10.1109/TIP.2008.2001399' },
  ],
  create({ lang, images }) {
    const img = images['camera-256'] as Gray, { w, h } = img;
    const u = uniformField(w * h, SEEDS.poisson), z = normalField(w * h, SEEDS.read);
    return {
      render(state, canvases) {
        const photons = num(state, 'photons'), gain = num(state, 'gain'), kind = String(state.enhancer);
        const cap = capture(img.data, photons, gain, u, z), out = enhance(kind, cap.display, w, h);
        drawGray(canvases.scene, img);
        drawGray(canvases.capture, { w, h, data: cap.display });
        drawGray(canvases.enhanced, { w, h, data: out });
        // Compare in scene units: undo the overall brightness scale so only noise and clipping count.
        const scale = gain * photons / FULL_SCALE, back = out.map(v => clip01(v / scale));
        const p = psnr(img.data, back);
        const bright = scale < 0.9 ? pick(lang, 'The picture is darker than the scene: raise the gain to match it.', '畫面比場景暗：提高增益可以讓它一樣亮。') : scale > 1.1 ? pick(lang, 'The picture is brighter than the scene, so highlights clip.', '畫面比場景亮，亮部會被截斷。') : pick(lang, 'The brightness matches the scene.', '亮度與場景相符。');
        return {
          summary: `${bright} ${pick(lang, `Signal-to-noise ratio ${fmt(cap.snrDb, 1)} dB comes from the ${photons} photons collected; gain ${gain} scales signal and noise together.`, `訊雜比 ${fmt(cap.snrDb, 1)} dB 來自收集到的 ${photons} 個光子；增益 ${gain} 會把訊號與雜訊一起放大。`)}`,
          metrics: [
            { label: pick(lang, 'Signal-to-noise (dB)', '訊雜比 (dB)'), value: fmt(cap.snrDb, 1) },
            { label: pick(lang, 'Clipped at white', '白端被截斷'), value: `${fmt(100 * cap.clipped, 1)}%` },
            { label: pick(lang, 'PSNR vs scene (dB)', '與場景的 PSNR (dB)'), value: fmt(p, 1) },
          ],
        };
      },
    };
  },
};
