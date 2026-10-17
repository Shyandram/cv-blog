import type { LabDef } from './types.ts';
import { L, pick, fmt, svg, text, path, num } from '../types.ts';
import type { Rgb } from '../../imagekit/types.ts';
import { drawRgb } from '../../imagekit/draw.ts';
import { rgbToHsv, hueDistance } from '../../imagekit/color.ts';

/** How a pixel whose gained channels exceed 1 is brought back into range. */
export type Overflow = 'clip' | 'scale' | 'tone';
/** A pixel has a usable hue only if it is neither near-gray nor near-black (HSV saturation and value at least this). */
export const HUE_MIN = 0.1;
/** Slack on that threshold: a saturation that is exactly 0.1 in decimal (for example (1, 0.9, 0.9)) must not flip with float rounding. */
const SLACK = 1e-6;
/** Tolerance so that a channel landing exactly on 1.0 (for example 170/255 × 1.5) is not counted as overflow because of float rounding. */
export const EPS = 1e-6;
const luma = (r: number, g: number, b: number) => 0.299 * r + 0.587 * g + 0.114 * b;

/**
 * Multiply every channel by `gain`, then handle overflow:
 * clip: min(1, gain · c) per channel;
 * scale: divide the whole pixel by its largest channel when that exceeds 1, so ratios (and hue) are kept;
 * tone: Reinhard-style extended operator on luma, L_d = L (1 + L / g²) / (1 + L) with L the gained luma and the white point g the gain,
 * applied as a common factor L_d / L to the three channels (ratios kept), then any channel still above 1 is clipped.
 * Returns the output (interleaved RGB in [0, 1]) and the fraction of pixels that had to be hard-clipped after the handling.
 */
export function applyGain(src: Float32Array, gain: number, method: Overflow): { out: Float32Array; clipped: number } {
  const n = src.length / 3, out = new Float32Array(src.length);
  let clippedPixels = 0;
  for (let i = 0; i < n; i++) {
    const r = src[3 * i] * gain, g = src[3 * i + 1] * gain, b = src[3 * i + 2] * gain, m = Math.max(r, g, b);
    let k = 1;
    if (method === 'scale') k = m > 1 ? 1 / m : 1;
    else if (method === 'tone') { const l = luma(r, g, b); k = l > 0 ? (1 + l / (gain * gain)) / (1 + l) : 1; }
    const vr = r * k, vg = g * k, vb = b * k;
    if (Math.max(vr, vg, vb) > 1 + EPS) clippedPixels++;
    out[3 * i] = Math.min(1, vr); out[3 * i + 1] = Math.min(1, vg); out[3 * i + 2] = Math.min(1, vb);
  }
  return { out, clipped: clippedPixels / n };
}
export interface HueReport {
  /** Pixels with some gained channel above 1 (before any handling). */ overflow: Uint8Array;
  /** Hue change in degrees on overflow pixels that have a hue before and after, NaN elsewhere. */ dh: Float32Array;
  /** Overflow pixels that were coloured before and whose saturation fell below HUE_MIN. */ lost: Uint8Array;
  overflowFraction: number; meanHueChange: number; lostFraction: number; satBefore: number; satAfter: number; coloured: number;
  hueBefore: number[]; hueAfter: number[];
}
/** Compare input and output hue and saturation on the pixels that overflowed. Hue bins are 10 degrees wide. */
export function hueReport(src: Float32Array, out: Float32Array, gain: number): HueReport {
  const n = src.length / 3, overflow = new Uint8Array(n), lost = new Uint8Array(n), dh = new Float32Array(n).fill(NaN);
  const hueBefore = new Array(36).fill(0), hueAfter = new Array(36).fill(0);
  let ov = 0, coloured = 0, sumDh = 0, nDh = 0, nLost = 0, sIn = 0, sOut = 0;
  for (let i = 0; i < n; i++) {
    if (Math.max(src[3 * i], src[3 * i + 1], src[3 * i + 2]) * gain <= 1 + EPS) continue;
    overflow[i] = 1; ov++;
    const [h0, s0, v0] = rgbToHsv(src[3 * i], src[3 * i + 1], src[3 * i + 2]);
    if (s0 < HUE_MIN - SLACK || v0 < HUE_MIN - SLACK) continue;
    const [h1, s1, v1] = rgbToHsv(out[3 * i], out[3 * i + 1], out[3 * i + 2]);
    coloured++; sIn += s0; sOut += s1; hueBefore[Math.min(35, Math.floor(h0 / 10))]++;
    if (s1 < HUE_MIN - SLACK || v1 < HUE_MIN - SLACK) { lost[i] = 1; nLost++; continue; }
    dh[i] = hueDistance(h0, h1); sumDh += dh[i]; nDh++; hueAfter[Math.min(35, Math.floor(h1 / 10))]++;
  }
  return {
    overflow, dh, lost, overflowFraction: ov / n, meanHueChange: nDh ? sumDh / nDh : NaN, lostFraction: coloured ? nLost / coloured : NaN,
    satBefore: coloured ? sIn / coloured : NaN, satAfter: coloured ? sOut / coloured : NaN, coloured, hueBefore, hueAfter,
  };
}
/** Gray copy of the image with red where the hue moved (stronger for bigger changes) and blue where the colour was lost. */
export function hueMap(src: Rgb, rep: HueReport): Rgb {
  const n = src.w * src.h, data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const g = 0.25 + 0.6 * luma(src.data[3 * i], src.data[3 * i + 1], src.data[3 * i + 2]);
    let a = 0, tint: [number, number, number] = [0.86, 0.2, 0.16];
    if (rep.lost[i]) { a = 0.75; tint = [0.16, 0.35, 0.9]; }
    else if (rep.dh[i] >= 1) a = Math.min(1, 0.25 + rep.dh[i] / 40);
    for (let c = 0; c < 3; c++) data[3 * i + c] = g * (1 - a) + tint[c] * a;
  }
  return { w: src.w, h: src.h, data };
}

const pct = (x: number) => (Number.isFinite(x) ? `${fmt(100 * x, 1)}%` : '—');
const deg = (x: number) => (Number.isFinite(x) ? `${fmt(x, 1)}°` : '—');

export const lab: LabDef = {
  kind: 'image', id: 'color-clipping', track: 'image-processing',
  title: L('IP 06 · Why does brightening turn reds and oranges yellow, then white?', 'IP 06・為什麼增亮會讓紅色與橙色變黃、再變白？'),
  description: L('Multiply a real color photo by a gain and compare three ways of handling the channels that overflow: clip each one, rescale the pixel, or tone-map it.', '把一張真實彩色照片乘上增益，比較處理溢位通道的三種方式：各通道分別裁切、整個像素一起縮放，或做色調映射。'),
  challenge: L('Set the gain to 2 with per-channel clipping and look at the hue map: the wooden table and the coffee are red (hue moved) and the bright cup walls are blue (colour lost). Raise the gain to 4: why do the orange bars of the hue histogram pile up in one bin at 60° (yellow), and where do the other pixels go? Now switch to rescaling: what does not change, and what does the picture lose instead?', '把增益設為 2、使用各通道分別裁切，並看色相圖：木桌與咖啡呈紅色（色相移動），明亮的杯壁呈藍色（失去顏色）。把增益提高到 4：為什麼色相直方圖中橘色的長條會堆在 60°（黃色）那一格，其他像素又去了哪裡？再切換成整體縮放：什麼沒有改變？畫面反而失去了什麼？'),
  note: L('Photo: the coffee test image reduced to 256 × 171 (CC0, Rachel Michetti). The gain multiplies the stored 8-bit sRGB values directly, like np.clip(g * x, 0, 1) in the Images as Arrays chapter; a real exposure change acts on linear light, so the numbers would differ but the story would not. Overflow means some gained channel exceeds 255. Clip: each channel is cut at 255. Rescale: an overflowing pixel is divided by its largest channel, so the channel ratios, hence the hue, survive, but its largest channel always ends at 255. Tone map: a Reinhard-style operator L(1 + L/g²)/(1 + L) on the luma L of the gained pixel (white point g, the gain), applied as one common factor to R, G and B, followed by clipping of any channel still above 255. Hue is the HSV hue angle in degrees (red 0°, green 120°, blue 240°), close to but not identical with the chapter’s HSI hue. Pixels with HSV saturation or value below 0.1 have no usable hue and are left out of the hue statistics; a pixel that was coloured but ends below 0.1 saturation counts as colour lost. Saturation is HSV saturation, averaged over coloured pixels that overflow.', '照片：咖啡測試影像，縮小為 256 × 171（CC0，Rachel Michetti）。增益直接乘在儲存的 8 位元 sRGB 值上，就像「影像就是陣列」章節中的 np.clip(g * x, 0, 1)；真正的曝光變化是作用在線性光上，數字會不同，但道理一樣。溢位是指增益後有某個通道超過 255。裁切：每個通道在 255 處截斷。縮放：溢位的像素整個除以它最大的通道，因此通道比例（也就是色相）得以保留，但它最大的通道一定會落在 255。色調映射：對增益後像素的亮度 L 套用類似 Reinhard 的運算子 L(1 + L/g²)/(1 + L)（白點為增益 g），以同一個係數乘在 R、G、B 上，之後再裁切仍高於 255 的通道。色相是 HSV 色相角（紅 0°、綠 120°、藍 240°），與章節的 HSI 色相接近但不完全相同。HSV 飽和度或明度低於 0.1 的像素沒有可用的色相，不列入色相統計；原本有顏色、最後飽和度低於 0.1 的像素，算作失去顏色。飽和度是 HSV 飽和度，對溢位且有顏色的像素取平均。'),
  related: ['image-processing/dip-06-color'],
  controls: [
    { key: 'gain', label: L('Gain (multiply every channel)', '增益（乘在每個通道上）'), kind: 'range', value: 2, min: 1, max: 4, step: 0.1 },
    { key: 'method', label: L('When a channel overflows', '通道溢位時'), kind: 'select', value: 'clip', options: [
      { value: 'clip', label: L('Clip each channel at 255', '每個通道在 255 處裁切') },
      { value: 'scale', label: L('Rescale the pixel (keeps hue)', '整個像素縮放（保留色相）') },
      { value: 'tone', label: L('Reinhard-style tone map', '類似 Reinhard 的色調映射') }] },
    { key: 'view', label: L('Show', '顯示'), kind: 'select', value: 'result', options: [
      { value: 'result', label: L('The result', '運算結果') }, { value: 'hue', label: L('Pixels whose hue changed', '色相改變的像素') }] },
  ],
  predict: {
    question: L('Multiply an orange pixel (R = 200, G = 100, B = 40) by 2 with per-channel clipping: what happens to its hue?', '把一個橙色像素（R = 200、G = 100、B = 40）乘以 2，並各通道分別裁切：它的色相會怎麼變？'),
    options: [
      { value: 'same', label: L('It stays the same: all three channels were doubled', '不變：三個通道都乘了 2') },
      { value: 'yellow', label: L('It moves toward yellow', '往黃色移動') },
      { value: 'red', label: L('It moves toward red', '往紅色移動') },
    ],
    answer: 'yellow',
    explain: L('Doubling gives (400, 200, 80), but a byte stops at 255, so R is cut and the pixel becomes (255, 200, 80). The ratio R : G was 2 : 1 and is now about 1.3 : 1, so the red part shrank relative to the green part and the hue angle grows from 22.5° to 41.1°, toward yellow at 60°. Saturation falls from 0.80 to 0.69. Brightening more cuts G as well and the pixel approaches (255, 255, 255), yellow first and then white. Rescaling the pixel by its largest channel instead gives (255, 127.5, 51), which keeps the 22.5° hue.', '乘以 2 得到 (400, 200, 80)，但一個位元組最多只到 255，所以 R 被截斷，像素變成 (255, 200, 80)。R : G 原本是 2 : 1，現在約為 1.3 : 1，紅的部分相對綠的部分縮小，色相角從 22.5° 增加到 41.1°，往 60° 的黃色靠近。飽和度從 0.80 降到 0.69。再增亮會連 G 也被截斷，像素逐漸趨近 (255, 255, 255)：先變黃，再變白。若改成以最大通道縮放整個像素，則得到 (255, 127.5, 51)，色相維持 22.5°。'),
  },
  assets: { 'coffee-256': 'rgb' },
  paneMin: 300,
  panes: [
    { key: 'original', kind: 'canvas', label: L('Original photo', '原始照片'), width: 256, height: 171 },
    { key: 'result', kind: 'canvas', label: L('Result, or hue map (red: hue moved, blue: color lost, gray: unchanged)', '結果，或色相圖（紅：色相移動，藍：失去顏色，灰：未改變）'), width: 256, height: 171 },
    { key: 'hist', kind: 'svg', label: L('Hue of the overflowing colored pixels, before and after (10° bins)', '溢位且有顏色的像素，處理前後的色相分布（每格 10°）'), width: 640, height: 250, wide: true },
  ],
  sources: [
    { title: 'Rafael C. Gonzalez & Richard E. Woods, Digital Image Processing, 4th ed., Pearson, 2018, Chapter 6', url: 'https://www.pearson.com/en-us/subject-catalog/p/digital-image-processing/P200000003249/9780133356724' },
    { title: 'Erik Reinhard, Michael Stark, Peter Shirley & James Ferwerda, Photographic Tone Reproduction for Digital Images, ACM Transactions on Graphics (SIGGRAPH), 2002', url: 'https://doi.org/10.1145/566654.566575' },
    { title: 'Alvy Ray Smith, Color Gamut Transform Pairs, ACM SIGGRAPH Computer Graphics, 1978 (the HSV model)', url: 'https://doi.org/10.1145/965139.807361' },
    { title: 'scikit-image test image coffee: CC0 by the photographer, Rachel Michetti (courtesy of Pikolo Espresso Bar)', url: 'https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coffee' },
  ],
  create({ lang, images }) {
    const img = images['coffee-256'] as Rgb;
    return {
      render(state, canvases) {
        const gain = Math.round(10 * num(state, 'gain')) / 10, method = String(state.method) as Overflow, view = String(state.view);
        const { out, clipped } = applyGain(img.data, gain, method), rep = hueReport(img.data, out, gain);
        drawRgb(canvases.original, img);
        drawRgb(canvases.result, view === 'hue' ? hueMap(img, rep) : { w: img.w, h: img.h, data: out });
        // Hue histogram: bars before (blue) and after (orange), over a strip showing the hue of each bin.
        const top = Math.max(1, ...rep.hueBefore, ...rep.hueAfter), bw = 570 / 36, X = (i: number) => 50 + i * bw;
        let plot = path([[50, 30], [50, 170], [620, 170]], 'stroke="var(--plot-grid)"');
        for (let i = 0; i < 36; i++) {
          const hb = 130 * rep.hueBefore[i] / top, ha = 130 * rep.hueAfter[i] / top;
          plot += `<rect x="${X(i) + 1}" y="${170 - hb}" width="${bw / 2 - 1}" height="${hb}" fill="var(--plot-a)"/><rect x="${X(i) + bw / 2}" y="${170 - ha}" width="${bw / 2 - 1}" height="${ha}" fill="var(--plot-b)"/>`;
          plot += `<rect x="${X(i)}" y="176" width="${bw}" height="10" fill="hsl(${i * 10 + 5} 85% 50%)"/>`;
        }
        plot += text(50, 206, '0° ' + pick(lang, 'red', '紅')) + text(50 + 570 / 6, 206, '60°', 'text-anchor="middle"') + text(50 + 570 / 3, 206, '120°', 'text-anchor="middle"')
          + text(50 + 570 / 2, 206, '180°', 'text-anchor="middle"') + text(50 + 570 * 2 / 3, 206, '240°', 'text-anchor="middle"') + text(50 + 570 * 5 / 6, 206, '300°', 'text-anchor="middle"') + text(620, 206, '360°', 'text-anchor="end"')
          + text(55, 20, pick(lang, 'Blue: hue before the gain · Orange: hue after (pixels that turned gray are not drawn)', '藍：增益前的色相 · 橘：處理後的色相（變成灰色的像素不畫出）'), 'fill="var(--plot-ink)"')
          + text(50 + 570 / 6, 228, pick(lang, 'yellow', '黃'), 'text-anchor="middle"') + text(50 + 570 / 3, 228, pick(lang, 'green', '綠'), 'text-anchor="middle"')
          + text(50 + 570 * 2 / 3, 228, pick(lang, 'blue', '藍'), 'text-anchor="middle"');
        let summary: string;
        if (rep.overflowFraction === 0) summary = pick(lang, 'No channel exceeds 255 at this gain, so nothing needs handling and the picture is simply brighter. Raise the gain.', '在這個增益下沒有通道超過 255，不需要任何處理，畫面只是變亮。請提高增益。');
        else if (method === 'clip') summary = pick(lang, `${pct(rep.overflowFraction)} of the pixels have a channel above 255 at gain ${fmt(gain, 1)}. Cutting each channel separately changes their channel ratios: the hue of the coloured ones moves ${deg(rep.meanHueChange)} on average, and ${pct(rep.lostFraction)} of them end up nearly gray or white.`, `增益 ${fmt(gain, 1)} 時，有 ${pct(rep.overflowFraction)} 的像素出現超過 255 的通道。各通道分別截斷會改變通道比例：有顏色的像素色相平均移動 ${deg(rep.meanHueChange)}，其中 ${pct(rep.lostFraction)} 最後變得接近灰色或白色。`);
        else if (method === 'scale') summary = pick(lang, `${pct(rep.overflowFraction)} of the pixels overflow at gain ${fmt(gain, 1)}. Dividing each by its largest channel keeps the ratios, so the hue change is ${deg(rep.meanHueChange)} and saturation stays ${fmt(rep.satAfter, 2)}. The price: all those pixels now have their largest channel at 255, so brightness differences among them are flattened.`, `增益 ${fmt(gain, 1)} 時有 ${pct(rep.overflowFraction)} 的像素溢位。把每個像素除以它最大的通道會保留比例，所以色相變化為 ${deg(rep.meanHueChange)}，飽和度維持 ${fmt(rep.satAfter, 2)}。代價是：這些像素最大的通道都落在 255，彼此之間的亮度差異被壓平。`);
        else summary = pick(lang, `${pct(rep.overflowFraction)} of the pixels overflow at gain ${fmt(gain, 1)}. The tone map compresses luma and keeps channel ratios, but ${pct(clipped)} of the pixels still have a channel above 255 afterwards and are cut; the mean hue change is ${deg(rep.meanHueChange)}.`, `增益 ${fmt(gain, 1)} 時有 ${pct(rep.overflowFraction)} 的像素溢位。色調映射壓縮亮度並保留通道比例，但之後仍有 ${pct(clipped)} 的像素有通道超過 255 而被截斷；平均色相變化為 ${deg(rep.meanHueChange)}。`);
        return {
          summary,
          metrics: [
            { label: pick(lang, 'Pixels with a channel above 255', '有通道超過 255 的像素'), value: pct(rep.overflowFraction) },
            { label: pick(lang, 'Mean hue change (coloured pixels)', '平均色相變化（有顏色的像素）'), value: deg(rep.meanHueChange) },
            { label: pick(lang, 'Colour lost (turned gray or white)', '失去顏色（變成灰或白）'), value: pct(rep.lostFraction) },
            { label: pick(lang, 'Mean saturation before → after', '平均飽和度 處理前 → 處理後'), value: Number.isFinite(rep.satBefore) ? `${fmt(rep.satBefore, 2)} → ${fmt(rep.satAfter, 2)}` : '—' },
          ],
          plots: { hist: svg(plot, 250) },
        };
      },
    };
  },
};
