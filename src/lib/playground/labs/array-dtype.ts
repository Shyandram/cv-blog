import type { LabDef } from './types.ts';
import { L, pick, fmt, svg, text, path, num } from '../types.ts';
import type { Gray } from '../../imagekit/types.ts';
import { drawGray, drawMask } from '../../imagekit/draw.ts';

/** How the addition is carried out: `wrap` is NumPy's uint8 `a + b`, `saturate` is OpenCV's `cv2.add`, `float` computes in floating point and clips. */
export type Arithmetic = 'wrap' | 'saturate' | 'float';

/** Gray image in [0, 1] to the 8-bit values the browser decoded (the PNG stores whole bytes). */
export function toBytes(g: Gray): Uint8Array {
  return Uint8Array.from(g.data, v => Math.round(v * 255));
}
/** The mathematically correct brightened image: integer sum clipped to 0..255. */
export function correct(src: Uint8Array, amount: number): Uint8Array {
  return Uint8Array.from(src, v => Math.max(0, Math.min(255, v + amount)));
}
/**
 * Add `amount` to every pixel with the chosen arithmetic. For `wrap` the assignment into a Uint8Array reduces the sum
 * modulo 256, which is exactly what NumPy does for `gray + np.uint8(60)` (and for `gray - np.uint8(n)` when the amount is negative).
 */
export function brighten(src: Uint8Array, amount: number, mode: Arithmetic): Uint8Array {
  const out = new Uint8Array(src.length);
  if (mode === 'wrap') { for (let i = 0; i < src.length; i++) out[i] = src[i] + amount; return out; }
  if (mode === 'saturate') return correct(src, amount);
  const f = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) f[i] = src[i] + amount;
  for (let i = 0; i < src.length; i++) out[i] = Math.round(Math.max(0, Math.min(255, f[i])));
  return out;
}
/** Pixels whose exact sum falls outside 0..255, so the arithmetic has to do something about them. */
export function overflowMask(src: Uint8Array, amount: number): Uint8Array {
  return Uint8Array.from(src, v => (v + amount < 0 || v + amount > 255 ? 1 : 0));
}
export interface DtypeStats { overflow: number; wrapped: number; clipped: number; meanIn: number; meanOut: number; psnr: number; darker: number; brighter: number }
export function stats(src: Uint8Array, out: Uint8Array, amount: number, mode: Arithmetic): DtypeStats {
  const ref = correct(src, amount), n = src.length;
  let over = 0, sumIn = 0, sumOut = 0, se = 0, darker = 0, brighter = 0;
  for (let i = 0; i < n; i++) {
    if (src[i] + amount < 0 || src[i] + amount > 255) over++;
    sumIn += src[i]; sumOut += out[i];
    const d = out[i] - ref[i]; se += d * d;
    if (out[i] < src[i]) darker++; else if (out[i] > src[i]) brighter++;
  }
  const mse = se / n;
  return {
    overflow: over / n, wrapped: mode === 'wrap' ? over / n : 0, clipped: mode === 'wrap' ? 0 : over / n,
    meanIn: sumIn / n, meanOut: sumOut / n, psnr: mse === 0 ? Infinity : 10 * Math.log10(255 * 255 / mse),
    darker: darker / n, brighter: brighter / n,
  };
}

const pct = (x: number) => `${fmt(100 * x, 1)}%`;

export const lab: LabDef = {
  kind: 'image', id: 'array-dtype', track: 'foundations',
  title: L('Images as Arrays · Why does brightening make bright pixels dark?', '影像就是陣列・為什麼增亮反而讓亮的像素變暗？'),
  description: L('Add the same number to every pixel of a real photo and compare 8-bit wraparound with saturating and floating-point arithmetic.', '對一張真實照片的每個像素加上同一個數，並比較 8 位元回繞、飽和運算與浮點運算。'),
  challenge: L('With the amount at 60 and the view on overflow, see where the red pixels sit. Now switch the arithmetic to saturate: the same pixels turn white instead of black, and the PSNR against the correct result becomes infinite. Last, set the amount to −100 with wraparound: which pixels flip, and in which direction?', '把增加量設為 60，並把檢視切換到溢位，看紅色像素在哪裡。接著把運算切換為飽和：同一批像素會變成白色而不是黑色，與正確結果相比的 PSNR 也變成無限大。最後把增加量設為 −100 並使用回繞：哪些像素會翻轉？往哪個方向翻？'),
  note: L('Camera test image reduced to 256 × 256 by 2 × 2 averaging, so the fractions differ slightly from the chapter, which counts on the 512 × 512 original. The pixels are real bytes: wraparound is done in a Uint8Array, so the sum is reduced modulo 256 exactly as NumPy does for gray + np.uint8(60). Saturate clamps the integer sum to 0..255 (what cv2.add does); float adds in floating point, then clips and rounds, which gives the same bytes for whole-number amounts. A negative amount corresponds to subtracting. PSNR compares the result with the correct clipped result, with peak 255. Brightness here is the mean pixel value, 0 to 255; this shows arithmetic on stored values, not a physical exposure model.', '相機測試影像以 2 × 2 平均縮小為 256 × 256，因此各項比例與章節（以 512 × 512 原圖計算）略有不同。像素是真實的位元組：回繞在 Uint8Array 中進行，總和對 256 取餘數，與 NumPy 執行 gray + np.uint8(60) 時完全相同。飽和是把整數總和限制在 0..255（即 cv2.add 的行為）；浮點是以浮點數相加，再裁切並四捨五入，在增加量為整數時得到相同的位元組。負的增加量相當於做減法。PSNR 是把結果與正確的裁切結果比較，峰值為 255。這裡的亮度是像素值的平均（0 到 255）；它示範的是對儲存值的運算，不是物理曝光模型。'),
  related: ['foundations/images-as-arrays'],
  controls: [
    { key: 'amount', label: L('Add to every pixel', '加到每個像素的值'), kind: 'range', value: 60, min: -100, max: 100, step: 1 },
    { key: 'mode', label: L('Arithmetic', '運算方式'), kind: 'select', value: 'wrap', options: [
      { value: 'wrap', label: L('uint8 wraparound (NumPy a + b)', 'uint8 回繞（NumPy a + b）') },
      { value: 'saturate', label: L('uint8 saturate (OpenCV cv2.add)', 'uint8 飽和（OpenCV cv2.add）') },
      { value: 'float', label: L('float, then clip', '浮點運算後再裁切') }] },
    { key: 'view', label: L('Show', '顯示'), kind: 'select', value: 'result', options: [
      { value: 'result', label: L('The result', '運算結果') }, { value: 'overflow', label: L('Pixels that overflowed (red)', '溢位的像素（紅色）') }] },
  ],
  predict: {
    question: L('Add 60 to every pixel of an 8-bit photo: which pixels change the most?', '對一張 8 位元照片的每個像素都加 60：哪些像素的值變化最大？'),
    options: [
      { value: 'dark', label: L('The darkest pixels: they gain the full 60', '最暗的像素：它們剛好增加 60') },
      { value: 'same', label: L('None in particular: every pixel changes by +60', '沒有特別的：每個像素都增加 60') },
      { value: 'bright', label: L('The brightest pixels: they wrap around and end up dark', '最亮的像素：它們會回繞，變得很暗') },
    ],
    answer: 'bright',
    explain: L('A byte cannot hold 256 or more, so any pixel at 196 or above overflows and starts again from 0. On this photo that is 27.2% of the pixels (17,855 of 65,536; the chapter counts 27.5% on the 512 × 512 original). Each of them falls by exactly 196 levels, from at least 196 to at most 59, while every other pixel rises by 60. The mean brightness even drops, from 129.1 to 119.3, although every pixel was asked to get brighter. With saturating or floating-point arithmetic the same pixels stop at 255 and the mean rises to 185.7.', '一個位元組放不下 256 以上的數，所以值在 196 以上的像素都會溢位，並從 0 重新開始。在這張照片上，這佔像素的 27.2%（65,536 個中的 17,855 個；章節以 512 × 512 原圖計算為 27.5%）。這些像素每一個都恰好下降 196 個灰階，從至少 196 掉到至多 59，而其他像素都上升 60。平均亮度甚至從 129.1 降到 119.3，雖然每個像素原本都被要求變亮。若改用飽和或浮點運算，同一批像素會停在 255，平均亮度升到 185.7。'),
  },
  assets: { 'camera-256': 'gray' },
  paneMin: 250,
  panes: [
    { key: 'original', kind: 'canvas', label: L('Original photo (8-bit)', '原始照片（8 位元）'), width: 256, height: 256 },
    { key: 'result', kind: 'canvas', label: L('Your arithmetic (see Show)', '你選的運算（見「顯示」）'), width: 256, height: 256 },
    { key: 'correct', kind: 'canvas', label: L('Correct result: clip(pixel + amount, 0, 255)', '正確結果：clip(像素 + 增加量, 0, 255)'), width: 256, height: 256 },
    { key: 'curve', kind: 'svg', label: L('Output value against input value', '輸出值對輸入值'), width: 640, height: 230, wide: true },
  ],
  sources: [
    { title: 'NumPy developers, Data types, NumPy user guide (integer overflow and wraparound)', url: 'https://numpy.org/doc/stable/user/basics.types.html' },
  ],
  create({ lang, images }) {
    const img = images['camera-256'] as Gray, { w, h } = img, src = toBytes(img);
    const asGray = (b: Uint8Array) => ({ w, h, data: Float32Array.from(b, v => v / 255) });
    return {
      render(state, canvases) {
        const amount = Math.round(num(state, 'amount')), mode = String(state.mode) as Arithmetic, view = String(state.view);
        const out = brighten(src, amount, mode), ref = correct(src, amount), over = overflowMask(src, amount), s = stats(src, out, amount, mode);
        drawGray(canvases.original, img);
        if (view === 'overflow') drawMask(canvases.result, w, h, over, img.data); else drawGray(canvases.result, asGray(out));
        drawGray(canvases.correct, asGray(ref));
        // Transfer curve: what each input byte becomes. Wraparound draws as saw-teeth, the correct result as a flat-topped line.
        const X = (v: number) => 50 + (v / 255) * 570, Y = (v: number) => 190 - (v / 255) * 160;
        const ramp = Uint8Array.from({ length: 256 }, (_, v) => v), lut = brighten(ramp, amount, mode), ideal = correct(ramp, amount);
        let plot = path([[50, 30], [50, 190], [620, 190]], 'stroke="var(--plot-grid)"');
        plot += path(Array.from(ideal, (o, v) => [X(v), Y(o)] as [number, number]), 'stroke="var(--plot-c)" stroke-width="2" stroke-dasharray="5 4"');
        let seg: [number, number][] = [];
        for (let v = 0; v < 256; v++) {
          const o = lut[v];
          if (seg.length && Math.abs(o - lut[v - 1]) > 128) { plot += path(seg, 'stroke="var(--plot-b)" stroke-width="3"'); seg = []; }
          seg.push([X(v), Y(o)]);
        }
        plot += path(seg, 'stroke="var(--plot-b)" stroke-width="3"');
        if (amount !== 0) {
          const edge = amount > 0 ? 256 - amount : -amount;
          plot += path([[X(edge), 30], [X(edge), 190]], 'stroke="var(--plot-ink)" stroke-width="1" stroke-dasharray="2 4"')
            + (X(edge) + 170 > 620
              ? text(620, 58, pick(lang, `${amount > 0 ? 'input ≥' : 'input <'} ${edge} overflows`, `輸入${amount > 0 ? ' ≥ ' : ' < '}${edge} 會溢位`), 'fill="var(--plot-ink)" text-anchor="end"')
              : text(X(edge) + 6, 58, pick(lang, `${amount > 0 ? 'input ≥' : 'input <'} ${edge} overflows`, `輸入${amount > 0 ? ' ≥ ' : ' < '}${edge} 會溢位`), 'fill="var(--plot-ink)"'));
        }
        plot += text(50, 212, '0') + text(620, 212, pick(lang, '255 (input)', '255（輸入）'), 'text-anchor="end"') + text(44, 36, '255', 'text-anchor="end"')
          + text(300, 20, pick(lang, 'Solid: your arithmetic · Dashed: correct (clipped)', '實線：你選的運算 · 虛線：正確（裁切）結果'), 'fill="var(--plot-ink)"');
        const thr = amount > 0 ? 256 - amount : -amount;
        let summary: string;
        if (amount === 0) summary = pick(lang, 'Adding 0 changes nothing. Move the amount away from 0.', '加 0 不會改變任何東西。請把增加量移離 0。');
        else if (mode === 'wrap') summary = amount > 0
          ? pick(lang, `Pixels with value ${thr} or more (${pct(s.overflow)} of the photo) go past 255 and wrap to the dark end, so the mean brightness moves from ${fmt(s.meanIn, 1)} to ${fmt(s.meanOut, 1)} while the correct result would be higher. ${pct(s.darker)} of the pixels ended up darker.`, `值在 ${thr} 以上的像素（佔照片的 ${pct(s.overflow)}）超過 255 後回繞到暗的一端，平均亮度由 ${fmt(s.meanIn, 1)} 變成 ${fmt(s.meanOut, 1)}，而正確結果應該更亮。有 ${pct(s.darker)} 的像素反而變暗。`)
          : pick(lang, `Pixels below ${thr} (${pct(s.overflow)} of the photo) go under 0 and wrap to the bright end, so the darkest parts of the photo become bright. Mean brightness goes from ${fmt(s.meanIn, 1)} to ${fmt(s.meanOut, 1)}.`, `值低於 ${thr} 的像素（佔照片的 ${pct(s.overflow)}）低於 0 後回繞到亮的一端，照片最暗的部分反而變亮。平均亮度由 ${fmt(s.meanIn, 1)} 變成 ${fmt(s.meanOut, 1)}。`);
        else summary = s.overflow === 0
          ? pick(lang, 'No pixel leaves 0..255 at this amount, so every arithmetic gives the same picture. Raise the amount to see overflow.', '在這個增加量下沒有像素超出 0..255，所以每種運算的結果都一樣。提高增加量就能看到溢位。')
          : pick(lang, `${pct(s.overflow)} of the pixels would leave 0..255 and are held at the limit instead, so the picture matches the correct result (PSNR ∞). Mean brightness moves from ${fmt(s.meanIn, 1)} to ${fmt(s.meanOut, 1)}.`, `有 ${pct(s.overflow)} 的像素會超出 0..255，改為停在極限值，因此畫面與正確結果相同（PSNR ∞）。平均亮度由 ${fmt(s.meanIn, 1)} 變成 ${fmt(s.meanOut, 1)}。`);
        return {
          summary,
          metrics: [
            { label: pick(lang, 'Wrapped to the other end', '回繞到另一端的像素'), value: pct(s.wrapped) },
            { label: pick(lang, 'Clipped at 0 or 255', '被截在 0 或 255 的像素'), value: pct(s.clipped) },
            { label: pick(lang, 'Mean brightness (0–255)', '平均亮度（0–255）'), value: fmt(s.meanOut, 1) },
            { label: pick(lang, 'PSNR vs correct result', '與正確結果的 PSNR'), value: Number.isFinite(s.psnr) ? `${s.psnr.toFixed(1)} dB` : pick(lang, '∞ (identical)', '∞（完全相同）') },
          ],
          plots: { curve: svg(plot, 230) },
        };
      },
    };
  },
};
