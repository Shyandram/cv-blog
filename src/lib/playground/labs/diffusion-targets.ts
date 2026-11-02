import type { LabDef } from './types.ts';
import { L, pick, fmt, num } from '../types.ts';
import type { Rgb } from '../../imagekit/types.ts';
import { normalField } from '../../imagekit/rng.ts';
import { psnr } from '../../imagekit/metrics.ts';
import { drawRgb } from '../../imagekit/draw.ts';

/** Fixed seeds: the noise field ε and the field δ that shapes the network's error never change while sliders move. */
export const SEEDS = { eps: 7, err: 8 };
export type Target = 'eps' | 'x0' | 'v';
export const TARGETS: Target[] = ['eps', 'x0', 'v'];
/** Display mapping shared by every panel: gray = 0.5 + u / 2.5, so u = -1.25 is black and +1.25 is white (values beyond are clipped). */
export const DISPLAY_SPAN = 2.5;
/** Gain on the error panel (in the same display mapping): mid-gray is zero error. */
export const ERROR_GAIN = 3;

/** Signal level ᾱ from the signal-to-noise ratio of x_t in dB: SNR = ᾱ / (1 − ᾱ), as in the chapter. */
export const alphaFromSnr = (snrDb: number) => 1 / (1 + Math.pow(10, -snrDb / 10));
/** Image in [0, 1] to the diffusion range [-1, 1]. */
export function toSignal(img: Rgb): Float32Array { return Float32Array.from(img.data, v => 2 * v - 1); }

export interface Fields { x0: Float32Array; eps: Float32Array; xt: Float32Array; v: Float32Array }
/** Forward mix x_t = √ᾱ x₀ + √(1−ᾱ) ε and the velocity target v = √ᾱ ε − √(1−ᾱ) x₀ (the chapter's ML 08 convention). */
export function forward(x0: Float32Array, eps: Float32Array, alpha: number): Fields {
  const a = Math.sqrt(alpha), b = Math.sqrt(1 - alpha), xt = new Float32Array(x0.length), v = new Float32Array(x0.length);
  for (let i = 0; i < x0.length; i++) { xt[i] = a * x0[i] + b * eps[i]; v[i] = a * eps[i] - b * x0[i]; }
  return { x0, eps, xt, v };
}
export const targetField = (f: Fields, kind: Target) => (kind === 'eps' ? f.eps : kind === 'x0' ? f.x0 : f.v);
/** The network's output: the exact target plus an error s·δ. */
export function perturbed(target: Float32Array, delta: Float32Array, s: number): Float32Array {
  const out = new Float32Array(target.length);
  for (let i = 0; i < out.length; i++) out[i] = target[i] + s * delta[i];
  return out;
}
/** Convert a prediction of the chosen target into an estimate of x₀: (x_t − b ε̂)/a, x̂₀, or a x_t − b v̂. */
export function recoverX0(kind: Target, pred: Float32Array, xt: Float32Array, alpha: number): Float32Array {
  const a = Math.sqrt(alpha), b = Math.sqrt(1 - alpha), out = new Float32Array(pred.length);
  for (let i = 0; i < out.length; i++) out[i] = kind === 'eps' ? (xt[i] - b * pred[i]) / a : kind === 'x0' ? pred[i] : a * xt[i] - b * pred[i];
  return out;
}
/** Error multiplier from an error in the network output to the error in x̂₀: b/a for ε, 1 for x₀, b for v. */
export function amplification(kind: Target, alpha: number): number {
  const a = Math.sqrt(alpha), b = Math.sqrt(1 - alpha);
  return kind === 'eps' ? b / a : kind === 'x0' ? 1 : b;
}
/** An image decoder clips to the valid range. */
export const clipSignal = (x: Float32Array) => Float32Array.from(x, v => (v < -1 ? -1 : v > 1 ? 1 : v));
/** PSNR in dB between the clean image and the clipped estimate, both mapped back to [0, 1] (peak 1). */
export function recoveryPsnr(x0: Float32Array, x0hat: Float32Array): number {
  return psnr(Float32Array.from(x0, v => (v + 1) / 2), Float32Array.from(clipSignal(x0hat), v => (v + 1) / 2));
}
export function pearson(p: ArrayLike<number>, q: ArrayLike<number>): number {
  const n = p.length;
  let mp = 0, mq = 0;
  for (let i = 0; i < n; i++) { mp += p[i]; mq += q[i]; }
  mp /= n; mq /= n;
  let spq = 0, spp = 0, sqq = 0;
  for (let i = 0; i < n; i++) { const dp = p[i] - mp, dq = q[i] - mq; spq += dp * dq; spp += dp * dp; sqq += dq * dq; }
  return spq / Math.sqrt(spp * sqq);
}
export function rms(p: ArrayLike<number>, q: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < p.length; i++) s += (p[i] - q[i]) ** 2;
  return Math.sqrt(s / p.length);
}
export interface Evaluation { kind: Target; x0hat: Float32Array; psnr: number; amplification: number; measured: number }
/** Everything the lab reports for one target at one signal level and one output-error size. */
export function evaluate(f: Fields, delta: Float32Array, alpha: number, s: number, kind: Target): Evaluation {
  const out = perturbed(targetField(f, kind), delta, s), x0hat = recoverX0(kind, out, f.xt, alpha);
  const e = rms(out, targetField(f, kind));
  return { kind, x0hat, psnr: recoveryPsnr(f.x0, x0hat), amplification: amplification(kind, alpha), measured: e > 0 ? rms(x0hat, f.x0) / e : NaN };
}
function display(u: ArrayLike<number>, w: number, h: number, gain = 1): Rgb {
  const data = new Float32Array(u.length);
  for (let i = 0; i < data.length; i++) { const g = 0.5 + gain * u[i] / DISPLAY_SPAN; data[i] = g < 0 ? 0 : g > 1 ? 1 : g; }
  return { w, h, data };
}
const NAMES: Record<Target, { en: string; zh: string; symbol: string }> = {
  eps: { en: 'the noise ε', zh: '雜訊 ε', symbol: 'ε' }, x0: { en: 'the clean image x₀', zh: '乾淨影像 x₀', symbol: 'x₀' }, v: { en: 'the velocity v', zh: '速度 v', symbol: 'v' },
};
const dB = (v: number) => (v >= 100 ? '≥ 100' : fmt(v, 1));

export const lab: LabDef = {
  kind: 'image', id: 'diffusion-targets', track: 'ml',
  title: L('ML 08 · What does a network see when it predicts noise, the image or a velocity?', 'ML 08・網路預測雜訊、影像或速度時，看到的是什麼？'),
  description: L('Mix the astronaut photograph with fixed noise, look at the noise, clean-image and velocity targets side by side, then give the network a small error and see which conversion back to the clean image survives.', '把太空人照片與固定雜訊混合，並排檢視雜訊、乾淨影像與速度三種目標，再讓網路帶一點誤差，看看哪一種轉換回乾淨影像的方式承受得住。'),
  challenge: L('Start at −20 dB with an error of 0.05 in the noise output. Slide the signal level up to 0 dB: where do the three recoveries become equally good? Then go to +20 dB: which target is now the most error-sensitive, and why does the answer flip?', '從 −20 dB、雜訊輸出誤差 0.05 開始。把訊號強度往上拉到 0 dB：三種還原方式在哪裡一樣好？再拉到 +20 dB：現在哪一種目標對誤差最敏感？為什麼答案反過來了？'),
  note: L('Image: the astronaut photograph (256 × 256, RGB), with pixel values mapped to [-1, 1] as in the chapter. The mix is x_t = √ᾱ x₀ + √(1−ᾱ) ε with a fixed seeded standard-normal ε, and v = √ᾱ ε − √(1−ᾱ) x₀ is the velocity of the ML 08 chapter (the diffusion velocity, not the flow-matching transport velocity). The slider sets the signal-to-noise ratio ᾱ/(1−ᾱ) of x_t in dB. The network is not trained: its output is the exact target plus a fixed seeded error, and the same error size (in the units of the [-1, 1] pixel scale) is applied whichever target is predicted. A real network does not make equal errors for every target, so this isolates the conversion, not the learning. x̂₀ is clipped to [-1, 1] as an image decoder would, and PSNR is measured against x₀ with both mapped to [0, 1] (peak 1). All panels share one gray mapping (gray = 0.5 + u/2.5).', '影像：太空人照片（256 × 256、RGB），像素值如章節所述對應到 [-1, 1]。混合式為 x_t = √ᾱ x₀ + √(1−ᾱ) ε，其中 ε 是固定種子的標準常態雜訊，v = √ᾱ ε − √(1−ᾱ) x₀ 是 ML 08 章節的速度（擴散的速度，不是流匹配的傳輸速度）。滑桿設定的是 x_t 的訊雜比 ᾱ/(1−ᾱ)（單位 dB）。網路並未訓練：它的輸出是精確目標加上固定種子的誤差，且不論預測哪一種目標，都使用相同大小的誤差（以 [-1, 1] 像素尺度為單位）。真實的網路對每種目標的誤差並不相同，所以這個實驗隔離的是轉換本身，而不是學習。x̂₀ 會像影像解碼器一樣裁切到 [-1, 1]，PSNR 是把 x₀ 與 x̂₀ 都對應到 [0, 1]（峰值 1）後相對於 x₀ 計算。所有面板使用同一個灰階對應（灰階 = 0.5 + u/2.5）。'),
  related: ['ml/ml-07-diffusion-flow-matching'],
  controls: [
    { key: 'snr', label: L('Signal level: signal-to-noise ratio of x_t (dB)', '訊號強度：x_t 的訊雜比（dB）'), kind: 'range', value: -20, min: -30, max: 20, step: 1 },
    { key: 'target', label: L('The network predicts (with an error)', '網路預測（帶有誤差）'), kind: 'select', value: 'eps', options: [
      { value: 'eps', label: L('Noise ε', '雜訊 ε') }, { value: 'x0', label: L('Clean image x₀', '乾淨影像 x₀') }, { value: 'v', label: L('Velocity v', '速度 v') }] },
    { key: 'err', label: L('Error in the network output (ε has standard deviation 1)', '網路輸出的誤差（ε 的標準差為 1）'), kind: 'range', value: 0.05, min: 0, max: 0.2, step: 0.01 },
  ],
  predict: {
    question: L('The picture x_t is almost pure noise (−20 dB). A network makes the same small error, 0.05, whichever target it predicts. After converting each prediction back to a clean image x̂₀, which one is the most error-sensitive?', 'x_t 幾乎只剩雜訊（−20 dB）。不論預測哪一種目標，網路都犯同樣的小誤差 0.05。把每個預測轉換回乾淨影像 x̂₀ 之後，哪一種對誤差最敏感？'),
    options: [
      { value: 'eps', label: L('Predicting the noise ε: x̂₀ = (x_t − √(1−ᾱ) ε̂)/√ᾱ', '預測雜訊 ε：x̂₀ = (x_t − √(1−ᾱ) ε̂)/√ᾱ') },
      { value: 'x0', label: L('Predicting the clean image x₀ directly', '直接預測乾淨影像 x₀') },
      { value: 'v', label: L('Predicting the velocity v: x̂₀ = √ᾱ x_t − √(1−ᾱ) v̂', '預測速度 v：x̂₀ = √ᾱ x_t − √(1−ᾱ) v̂') },
      { value: 'equal', label: L('None: the conversions are exact, so all three are equally sensitive', '都不是：轉換是精確的，所以三者同樣敏感') },
    ],
    answer: 'eps',
    explain: L('The conversions are exact, but they do not treat errors alike. The noise route divides by √ᾱ, so an error in ε̂ is multiplied by √(1−ᾱ)/√ᾱ, which is 10.0 at −20 dB; the clean-image route passes the error through unchanged (×1), and the velocity route multiplies it by √(1−ᾱ), which is 0.995. With the same error of 0.05 the recovered image has a PSNR of 13.5 dB from the noise route, against 32.4 dB from the clean-image route and 32.4 dB from the velocity route. Ironically the noise is the target that is easiest to read off the picture at this level: ε is almost a copy of x_t (correlation 0.998), while v is nearly the negative of the clean image (correlation −0.988). And the ranking flips near the clean end: at +20 dB the clean-image route is the most sensitive (32.4 dB, against 52.3 dB for the other two).', '轉換是精確的，但對誤差並不一視同仁。雜訊路徑要除以 √ᾱ，所以 ε̂ 的誤差會被乘上 √(1−ᾱ)/√ᾱ，在 −20 dB 時是 10.0；乾淨影像路徑原封不動地傳遞誤差（×1），速度路徑則乘上 √(1−ᾱ)，也就是 0.995。同樣是 0.05 的誤差，雜訊路徑還原出的影像 PSNR 為 13.5 dB，乾淨影像路徑是 32.4 dB，速度路徑也是 32.4 dB。耐人尋味的是，在這個程度下雜訊反而是最容易從畫面上「讀出來」的目標：ε 幾乎就是 x_t 的複本（相關係數 0.998），而 v 幾乎是乾淨影像的負片（相關係數 −0.988）。而且排名在接近乾淨的那一端會翻轉：在 +20 dB 時，乾淨影像路徑最敏感（32.4 dB，另外兩種是 52.3 dB）。'),
  },
  assets: { 'astronaut-256': 'rgb' },
  paneMin: 200,
  panes: [
    { key: 'xt', kind: 'canvas', label: L('x_t, the network input: the only picture it gets to see', 'x_t，網路的輸入：它唯一看得到的畫面'), width: 256, height: 256 },
    { key: 'eps', kind: 'canvas', label: L('ε, the noise that was added', 'ε，被加入的雜訊'), width: 256, height: 256 },
    { key: 'x0', kind: 'canvas', label: L('x₀, the clean image', 'x₀，乾淨影像'), width: 256, height: 256 },
    { key: 'v', kind: 'canvas', label: L('v = √ᾱ·ε − √(1−ᾱ)·x₀, the velocity', 'v = √ᾱ·ε − √(1−ᾱ)·x₀，速度'), width: 256, height: 256 },
    { key: 'rec', kind: 'canvas', label: L('x̂₀ recovered from the network’s imperfect output of the chosen target', '由網路對所選目標的不完美輸出還原出的 x̂₀'), width: 256, height: 256 },
    { key: 'err', kind: 'canvas', label: L('Error x̂₀ − x₀ (gain ×3, mid-gray = 0)', '誤差 x̂₀ − x₀（增益 ×3，中灰 = 0）'), width: 256, height: 256 },
  ],
  sources: [
    { title: 'Jonathan Ho, Ajay Jain & Pieter Abbeel, Denoising Diffusion Probabilistic Models, NeurIPS 2020 (forward mix x_t and noise prediction)', url: 'https://arxiv.org/abs/2006.11239' },
    { title: 'Tim Salimans & Jonathan Ho, Progressive Distillation for Fast Sampling of Diffusion Models, ICLR 2022 (the velocity target v)', url: 'https://arxiv.org/abs/2202.00512' },
  ],
  create({ lang, images }) {
    const img = images['astronaut-256'] as Rgb, { w, h } = img;
    const x0 = toSignal(img), eps = normalField(x0.length, SEEDS.eps), delta = normalField(x0.length, SEEDS.err);
    return {
      render(state, canvases) {
        const snr = num(state, 'snr'), s = num(state, 'err'), kind: Target = TARGETS.includes(state.target as Target) ? (state.target as Target) : 'eps';
        const alpha = alphaFromSnr(snr), f = forward(x0, eps, alpha);
        const runs = TARGETS.map(t => evaluate(f, delta, alpha, s, t)), chosen = runs[TARGETS.indexOf(kind)];
        drawRgb(canvases.xt, display(f.xt, w, h)); drawRgb(canvases.eps, display(f.eps, w, h));
        drawRgb(canvases.x0, display(f.x0, w, h)); drawRgb(canvases.v, display(f.v, w, h));
        drawRgb(canvases.rec, display(clipSignal(chosen.x0hat), w, h));
        drawRgb(canvases.err, display(Float32Array.from(chosen.x0hat, (v, i) => v - x0[i]), w, h, ERROR_GAIN));
        const name = pick(lang, NAMES[kind].en, NAMES[kind].zh), mult = fmt(chosen.amplification, 2);
        const summary = pick(lang,
          `At ${snr} dB the signal weight √ᾱ is ${fmt(Math.sqrt(alpha), 3)} and the noise weight √(1−ᾱ) is ${fmt(Math.sqrt(1 - alpha), 3)}. An error of ${fmt(s, 2)} in the network’s output for ${name} reaches x̂₀ multiplied by ${mult}, so the recovered image has a PSNR of ${dB(chosen.psnr)} dB. At the same error the three routes give ${runs.map(r => dB(r.psnr)).join(', ')} dB for ε, x₀, v.`,
          `在 ${snr} dB 時，訊號權重 √ᾱ 為 ${fmt(Math.sqrt(alpha), 3)}，雜訊權重 √(1−ᾱ) 為 ${fmt(Math.sqrt(1 - alpha), 3)}。網路對${name}的輸出若有 ${fmt(s, 2)} 的誤差，傳到 x̂₀ 時會乘上 ${mult} 倍，因此還原影像的 PSNR 為 ${dB(chosen.psnr)} dB。同樣的誤差下，ε、x₀、v 三條路徑分別得到 ${runs.map(r => dB(r.psnr)).join('、')} dB。`);
        return {
          summary,
          metrics: [
            { label: pick(lang, 'Signal level ᾱ', '訊號強度 ᾱ'), value: fmt(alpha, 4) },
            { label: pick(lang, 'PSNR of x̂₀, chosen target (dB)', '所選目標還原的 x̂₀ PSNR (dB)'), value: dB(chosen.psnr) },
            { label: pick(lang, 'PSNR if ε / x₀ / v (dB)', '若預測 ε / x₀ / v 的 PSNR (dB)'), value: runs.map(r => dB(r.psnr)).join(' / ') },
            { label: pick(lang, 'Error multiplier, output → x̂₀', '誤差倍數，輸出 → x̂₀'), value: `×${mult}` },
          ],
        };
      },
    };
  },
};
