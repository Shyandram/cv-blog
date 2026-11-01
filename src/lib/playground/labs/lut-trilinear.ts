import type { LabDef } from './types.ts';
import { L, pick, fmt, num } from '../types.ts';
import type { Rgb } from '../../imagekit/types.ts';
import { drawRgb } from '../../imagekit/draw.ts';

/** The "true" enhancement: a gamma lift (exponent), an S-curve (blend with smoothstep), a saturation gain about luma and a hue turn about the gray axis. */
export const GAMMA = 0.9, S_BLEND = 0.5, SATURATION = 1.15, HUE_DEG = 8;
const LUMA = [0.299, 0.587, 0.114];

/** Per-channel tone curve: g = t^GAMMA, then (1 − a) g + a · smoothstep(g). Monotone, maps [0, 1] onto [0, 1]. */
export function toneCurve(t: number): number {
  const g = Math.pow(t < 0 ? 0 : t, GAMMA);
  return (1 - S_BLEND) * g + S_BLEND * (3 * g * g - 2 * g * g * g);
}
/** 3 × 3 colour matrix: rotate by HUE_DEG about the gray axis (1, 1, 1)/√3 after scaling the distance from the luma gray by SATURATION. Row-major. */
export function colourMatrix(): number[] {
  const th = (HUE_DEG * Math.PI) / 180, c = Math.cos(th), s = Math.sin(th), k = 1 / Math.sqrt(3), oc = 1 - c;
  const rot = [c + oc * k * k, oc * k * k - s * k, oc * k * k + s * k, oc * k * k + s * k, c + oc * k * k, oc * k * k - s * k, oc * k * k - s * k, oc * k * k + s * k, c + oc * k * k];
  const sat = [0, 1, 2].flatMap(i => [0, 1, 2].map(j => (i === j ? SATURATION : 0) + (1 - SATURATION) * LUMA[j]));
  const m: number[] = [];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { let v = 0; for (let t = 0; t < 3; t++) v += rot[3 * i + t] * sat[3 * t + j]; m.push(v); }
  return m;
}
const MATRIX = colourMatrix();
const clip01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
/** The exact transform of one colour: tone curve on every channel, then the colour matrix, then clipping to [0, 1]. */
export function exactColour(r: number, g: number, b: number): [number, number, number] {
  const x = [toneCurve(r), toneCurve(g), toneCurve(b)];
  return [0, 1, 2].map(i => clip01(MATRIX[3 * i] * x[0] + MATRIX[3 * i + 1] * x[1] + MATRIX[3 * i + 2] * x[2])) as [number, number, number];
}
/** Apply the exact transform to an interleaved RGB array in [0, 1]. */
export function exactImage(src: Float32Array): Float32Array {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i += 3) { const [r, g, b] = exactColour(src[i], src[i + 1], src[i + 2]); out[i] = r; out[i + 1] = g; out[i + 2] = b; }
  return out;
}
/** Bake the exact transform into an M × M × M lattice. Node (i, j, k) (i along red, k along blue) is at ((i · M + j) · M + k) · 3. */
export function bakeLut(M: number): Float64Array {
  const lut = new Float64Array(M * M * M * 3), d = M - 1;
  for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) for (let k = 0; k < M; k++) {
    const [r, g, b] = exactColour(i / d, j / d, k / d), o = ((i * M + j) * M + k) * 3;
    lut[o] = r; lut[o + 1] = g; lut[o + 2] = b;
  }
  return lut;
}
export type Interpolation = 'nearest' | 'trilinear';
/** Look every pixel up in the table: the nearest node, or the trilinear blend of the 8 corners of its cell (the formula of the chapter). */
export function applyLut(lut: Float64Array, M: number, src: Float32Array, mode: Interpolation): Float32Array {
  const out = new Float32Array(src.length), d = M - 1;
  for (let p = 0; p < src.length; p += 3) {
    const ur = clip01(src[p]) * d, ug = clip01(src[p + 1]) * d, ub = clip01(src[p + 2]) * d;
    if (mode === 'nearest') {
      const o = ((Math.round(ur) * M + Math.round(ug)) * M + Math.round(ub)) * 3;
      out[p] = lut[o]; out[p + 1] = lut[o + 1]; out[p + 2] = lut[o + 2];
      continue;
    }
    const i = Math.min(Math.floor(ur), M - 2), j = Math.min(Math.floor(ug), M - 2), k = Math.min(Math.floor(ub), M - 2);
    const dr = ur - i, dg = ug - j, db = ub - k;
    let r = 0, g = 0, b = 0;
    for (let a = 0; a < 2; a++) for (let bb = 0; bb < 2; bb++) for (let c = 0; c < 2; c++) {
      const w = (a ? dr : 1 - dr) * (bb ? dg : 1 - dg) * (c ? db : 1 - db), o = (((i + a) * M + (j + bb)) * M + (k + c)) * 3;
      r += w * lut[o]; g += w * lut[o + 1]; b += w * lut[o + 2];
    }
    out[p] = r; out[p + 1] = g; out[p + 2] = b;
  }
  return out;
}
/** 8-bit codes (what a screen shows): round(255 · clip(v)). */
export function toLevels(a: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(a.length);
  for (let i = 0; i < out.length; i++) out[i] = Math.round(255 * clip01(a[i]));
  return out;
}
/** Mean and maximum absolute difference of two 8-bit images, in levels (all channels). */
export function levelError(a: Uint8Array, b: Uint8Array): { mean: number; max: number } {
  let s = 0, m = 0;
  for (let i = 0; i < a.length; i++) { const e = Math.abs(a[i] - b[i]); s += e; if (e > m) m = e; }
  return { mean: s / a.length, max: m };
}
/** A smooth synthetic picture with a known answer: bilinear blend of four corner colours (dark blue, teal, orange, pale peach). */
export function gradientImage(w = 256, h = 256): Rgb {
  const tl = [0.05, 0.10, 0.35], tr = [0.10, 0.55, 0.60], bl = [0.90, 0.45, 0.20], br = [0.98, 0.85, 0.75], data = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const u = x / (w - 1), v = y / (h - 1);
    for (let c = 0; c < 3; c++) data[3 * (y * w + x) + c] = (1 - v) * ((1 - u) * tl[c] + u * tr[c]) + v * ((1 - u) * bl[c] + u * br[c]);
  }
  return { w, h, data };
}
/** Number of nodes and of stored numbers of an M × M × M table. */
export const lutSize = (M: number) => ({ nodes: M ** 3, values: 3 * M ** 3 });
const sep = (n: number) => n.toLocaleString('en-US');

export interface LutRun { exact: Uint8Array; nearest: Uint8Array; trilinear: Uint8Array; eNearest: { mean: number; max: number }; eTrilinear: { mean: number; max: number } }
/** Everything the lab shows for one image and one table size (`exact` may be passed in when it is already known). */
export function run(src: Float32Array, M: number, exact: Uint8Array = toLevels(exactImage(src))): LutRun {
  const lut = bakeLut(M), nearest = toLevels(applyLut(lut, M, src, 'nearest')), trilinear = toLevels(applyLut(lut, M, src, 'trilinear'));
  return { exact, nearest, trilinear, eNearest: levelError(nearest, exact), eTrilinear: levelError(trilinear, exact) };
}
const fromLevels = (a: Uint8Array) => Float32Array.from(a, v => v / 255);
const GAIN = 8;

export const lab: LabDef = {
  kind: 'image', id: 'lut-trilinear', track: 'deep-learning',
  title: L('DLCV 05A · How big must a 3D LUT be?', 'DLCV 05A・3D 查找表要多大才夠？'),
  description: L('Bake one smooth color enhancement into an M × M × M table and apply it with a nearest-node lookup or with trilinear interpolation. Compare both with the exact transform.', '把一個平滑的色彩增強烘焙進 M × M × M 的表中，分別以最近節點查表與三線性內插套用，並與精確轉換比較。'),
  challenge: L('At the default (M = 9, the smooth gradient) the nearest lookup shows clear bands and the trilinear result looks like the exact one. Switch the error map to trilinear: where is the error largest, and why there? Then raise M to 33 for the nearest lookup: are the bands gone? Finally set M = 3 and compare the two interpolations.', '預設（M = 9、平滑漸層）下，最近節點查表出現明顯的色帶，三線性的結果則看起來與精確結果一樣。把誤差圖切換成三線性：誤差最大的地方在哪裡？為什麼在那裡？接著把最近節點查表的 M 調到 33：色帶消失了嗎？最後把 M 設為 3，比較兩種內插。'),
  note: L('The true enhancement is defined in code (src/lib/playground/labs/lut-trilinear.ts) and is the lab’s own toy: on each channel t, a tone curve g = t^0.9 followed by the S-curve (1 − a) g + a (3g² − 2g³) with a = 0.5; then a saturation gain of 1.15 that scales each color’s distance from its luma gray (luma 0.299 R + 0.587 G + 0.114 B); then a hue turn of 8° (a rotation of the RGB cube about its gray axis); then clipping to [0, 1]. The table has one entry per node of an M × M × M lattice over the RGB cube: node (i, j, k) stores the exact transform of the color (i/(M−1), j/(M−1), k/(M−1)), so it holds 3M³ numbers. A nearest lookup returns the stored color of the closest node. Trilinear interpolation blends the 8 corners of the cell around the color with weights that are products of the offsets, as in the chapter. Both results and the exact transform are rounded to 8-bit levels (0 to 255) before the errors are measured, and the error map is magnified 8 times. The photo is the astronaut test image reduced to 256 × 256; the gradient is a smooth blend of four corner colors computed in the lab, so it has no texture to hide banding.', '真實的增強轉換寫在程式碼裡（src/lib/playground/labs/lut-trilinear.ts），是這個實驗自己的玩具範例：對每個通道 t，先做色調曲線 g = t^0.9，再套用 S 曲線 (1 − a) g + a (3g² − 2g³)（a = 0.5）；接著把每個顏色到其亮度灰點（亮度 0.299 R + 0.587 G + 0.114 B）的距離乘上 1.15 的飽和度增益；然後做 8° 的色相轉動（RGB 立方體繞灰軸旋轉）；最後裁切到 [0, 1]。查找表在 RGB 立方體上對 M × M × M 格點的每個節點存一個項目：節點 (i, j, k) 存放顏色 (i/(M−1), j/(M−1), k/(M−1)) 經精確轉換後的結果，因此共有 3M³ 個數字。最近節點查表回傳最接近節點所儲存的顏色。三線性內插則以偏移量的乘積為權重，混合顏色所在格子的 8 個角點，與章節相同。兩種結果與精確轉換在量測誤差之前，都先四捨五入成 8 位元階（0 到 255），誤差圖放大 8 倍。照片是縮小為 256 × 256 的 astronaut 測試影像；漸層是在實驗中計算出的四個角落顏色的平滑混合，沒有紋理可以遮住色帶。'),
  related: ['deep-learning/dl-05a-lightweight-enhancement'],
  controls: [
    { key: 'm', label: L('Nodes per axis M', '每軸節點數 M'), kind: 'select', value: '9', options: ['2', '3', '5', '9', '17', '33'].map(v => ({ value: v, label: L(v, v) })) },
    { key: 'image', label: L('Image', '影像'), kind: 'select', value: 'gradient', options: [
      { value: 'gradient', label: L('Smooth gradient (shows banding)', '平滑漸層（顯示色帶）') }, { value: 'photo', label: L('Photo (astronaut)', '照片（astronaut）') }] },
    { key: 'view', label: L('Error map of', '誤差圖顯示'), kind: 'select', value: 'nearest', options: [
      { value: 'nearest', label: L('Nearest-node lookup', '最近節點查表') }, { value: 'trilinear', label: L('Trilinear interpolation', '三線性內插') }] },
  ],
  predict: {
    question: L('A nearest-node table with M = 33 has about 49 times as many entries as a trilinear table with M = 9 (35,937 against 729 nodes). Which one reproduces the exact transform more closely on the smooth gradient?', '最近節點查表 M = 33 的項目數約為三線性 M = 9 的 49 倍（35,937 對 729 個節點）。在平滑漸層上，哪一個更接近精確轉換？'),
    options: [
      { value: 'nearest33', label: L('The nearest lookup with M = 33: far more entries', '最近節點查表 M = 33：項目多得多') },
      { value: 'tri9', label: L('The trilinear table with M = 9: it blends between nodes', '三線性 M = 9：它在節點之間做混合') },
      { value: 'same', label: L('About the same', '差不多') },
    ],
    answer: 'tri9',
    explain: L('A nearest lookup replaces every color by the stored color of one node, so the output is a staircase: even with M = 33 the steps are several levels tall, and they show up as bands on a smooth gradient. Trilinear interpolation turns the staircase into a continuous, piecewise-linear surface that follows a smooth transform closely even with 9 nodes. On the gradient the nearest table with M = 33 has a mean error of 2.46 levels and a maximum of 7; the trilinear table with M = 9 has a mean error of 0.32 levels and a maximum of 4.', '最近節點查表會把每個顏色換成某個節點儲存的顏色，所以輸出是階梯狀：即使 M = 33，每階仍有好幾個階高，在平滑漸層上就成了色帶。三線性內插把階梯變成連續、分段線性的曲面，即使只有 9 個節點，也能緊貼平滑的轉換。在漸層上，M = 33 的最近節點查表平均誤差為 2.46 階、最大為 7；M = 9 的三線性查表平均誤差為 0.32 階、最大為 4。'),
  },
  assets: { 'astronaut-256': 'rgb' },
  paneMin: 220,
  panes: [
    { key: 'exact', kind: 'canvas', label: L('Exact transform (the target)', '精確轉換（目標）'), width: 256, height: 256 },
    { key: 'nearest', kind: 'canvas', label: L('LUT, nearest node', '查找表，最近節點'), width: 256, height: 256 },
    { key: 'trilinear', kind: 'canvas', label: L('LUT, trilinear', '查找表，三線性'), width: 256, height: 256 },
    { key: 'err', kind: 'canvas', label: L('Error map of the chosen lookup (8×, bright: wrong)', '所選查表方式的誤差圖（放大 8 倍，亮：錯誤）'), width: 256, height: 256 },
  ],
  sources: [
    { title: 'Hui Zeng, Jianrui Cai, Lida Li, Zisheng Cao & Lei Zhang, Learning Image-adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time, IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022', url: 'https://doi.org/10.1109/TPAMI.2020.3026740' },
  ],
  create({ lang, images }) {
    const photo = images['astronaut-256'] as Rgb, grad = gradientImage(256, 256);
    const sources = { gradient: grad, photo }, cache = new Map<string, { exact: Uint8Array }>();
    const exactOf = (name: 'gradient' | 'photo') => { let e = cache.get(name); if (!e) { e = { exact: toLevels(exactImage(sources[name].data)) }; cache.set(name, e); } return e.exact; };
    return {
      render(state, canvases) {
        const M = Math.round(num(state, 'm')), name = state.image === 'photo' ? 'photo' : 'gradient', src = sources[name], view = state.view === 'trilinear' ? 'trilinear' : 'nearest';
        const r = run(src.data, M, exactOf(name)), { exact, nearest, trilinear } = r, en = r.eNearest, et = r.eTrilinear, shown = view === 'nearest' ? nearest : trilinear, err = new Float32Array(shown.length);
        for (let i = 0; i < err.length; i++) err[i] = Math.min(1, (GAIN * Math.abs(shown[i] - exact[i])) / 255);
        const { w, h } = src;
        drawRgb(canvases.exact, { w, h, data: fromLevels(exact) }); drawRgb(canvases.nearest, { w, h, data: fromLevels(nearest) }); drawRgb(canvases.trilinear, { w, h, data: fromLevels(trilinear) });
        drawRgb(canvases.err, { w, h, data: err });
        const size = lutSize(M);
        const summary = pick(lang,
          `With M = ${M} the table holds ${sep(size.nodes)} nodes (${sep(size.values)} numbers). The nearest lookup is off by ${fmt(en.mean, 2)} levels on average and by up to ${en.max}; trilinear interpolation is off by ${fmt(et.mean, 2)} on average and by up to ${et.max}.${en.mean > 3 * et.mean ? ' The nearest lookup turns the smooth transform into steps, which shows as bands; interpolation follows it.' : ''}`,
          `M = ${M} 時，表有 ${sep(size.nodes)} 個節點（${sep(size.values)} 個數字）。最近節點查表平均誤差 ${fmt(en.mean, 2)} 階、最大 ${en.max} 階；三線性內插平均誤差 ${fmt(et.mean, 2)} 階、最大 ${et.max} 階。${en.mean > 3 * et.mean ? '最近節點查表把平滑的轉換變成階梯，呈現為色帶；內插則能緊貼轉換。' : ''}`);
        return {
          summary,
          metrics: [
            { label: pick(lang, 'Table size', '表的大小'), value: pick(lang, `${sep(size.nodes)} nodes, ${sep(size.values)} values`, `${sep(size.nodes)} 個節點，${sep(size.values)} 個數字`) },
            { label: pick(lang, 'Nearest: mean / max error (levels)', '最近節點：平均 / 最大誤差（階）'), value: `${fmt(en.mean, 2)} / ${en.max}` },
            { label: pick(lang, 'Trilinear: mean / max error (levels)', '三線性：平均 / 最大誤差（階）'), value: `${fmt(et.mean, 2)} / ${et.max}` },
          ],
        };
      },
    };
  },
};
