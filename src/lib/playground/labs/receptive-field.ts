import type { LabDef } from './types.ts';
import { L, pick, fmt, num } from '../types.ts';
import { plotColors, startPlot, label, line } from './plot2d.ts';
import type { Gray } from '../../imagekit/types.ts';
import { drawGray } from '../../imagekit/draw.ts';
import { attachProbe } from './pointer.ts';
import type { Lang } from '../../../i18n/ui';

/** One convolution layer: kernel size k, stride s, dilation d. Padding is (effective kernel - 1) / 2 on each side (zeros). */
export interface Layer { k: number; s: number; d: number }
export const effectiveKernel = (l: Layer) => (l.k - 1) * l.d + 1;
const conv = (n: number, s = 1, d = 1): Layer[] => Array.from({ length: n }, () => ({ k: 3, s, d }));
const at = (layers: Layer[], i: number, patch: Partial<Layer>) => layers.map((l, j) => (j === i ? { ...l, ...patch } : l));

export interface Stack { key: string; layers: Layer[]; label: { en: string; zh: string } }
/** The named stacks of the select control. Keep the keys in sync with scripts/figures/playground_goldens_receptive.py. */
export const STACKS: Stack[] = [
  { key: 'plain3', layers: conv(3), label: { en: 'Three 3×3 layers', zh: '三層 3×3' } },
  { key: 'plain4', layers: conv(4), label: { en: 'Four 3×3 layers', zh: '四層 3×3' } },
  { key: 'plain6', layers: conv(6), label: { en: 'Six 3×3 layers', zh: '六層 3×3' } },
  { key: 'stride-first', layers: at(conv(3), 0, { s: 2 }), label: { en: 'Three 3×3 layers, stride 2 in the first', zh: '三層 3×3，第一層步幅 2' } },
  { key: 'stride-last', layers: at(conv(3), 2, { s: 2 }), label: { en: 'Three 3×3 layers, stride 2 in the last', zh: '三層 3×3，最後一層步幅 2' } },
  { key: 'dilate-first', layers: at(conv(3), 0, { d: 2 }), label: { en: 'Three 3×3 layers, dilation 2 in the first', zh: '三層 3×3，第一層膨脹率 2' } },
  { key: 'stride-every-2nd', layers: [1, 2, 1, 2, 1, 2].map(s => ({ k: 3, s, d: 1 })), label: { en: 'Six 3×3 layers, stride 2 in every second', zh: '六層 3×3，每隔一層步幅 2' } },
  { key: 'dilated-1248', layers: [1, 2, 4, 8].map(d => ({ k: 3, s: 1, d })), label: { en: 'Four 3×3 layers, dilations 1, 2, 4, 8', zh: '四層 3×3，膨脹率 1、2、4、8' } },
];
export const stackByKey = (key: string) => STACKS.find(s => s.key === key) ?? STACKS[0];
export const DEFAULT = { stack: 'stride-every-2nd', x: 120, y: 72 };

/** The standard recurrence: r += (k_eff - 1) * j, then j *= s, starting from r = j = 1. Row l is the state after layer l. */
export function recurrence(layers: Layer[]): { r: number; j: number; rows: { r: number; j: number }[] } {
  let r = 1, j = 1;
  const rows = [{ r, j }];
  for (const l of layers) { r += (effectiveKernel(l) - 1) * j; j *= l.s; rows.push({ r, j }); }
  return { r, j, rows };
}
/** Number of units along one axis after each layer for an input of n pixels (index 0 is the input). */
export function gridSizes(layers: Layer[], n: number): number[] {
  const out = [n];
  for (const l of layers) { const e = effectiveKernel(l), pad = (e - 1) >> 1; out.push(Math.floor((out[out.length - 1] + 2 * pad - e) / l.s) + 1); }
  return out;
}
/**
 * Exact trace: push the index interval [unit, unit] of one unit of layer `upTo` back through layers upTo..1.
 * A unit u of a layer reads its input at s·u − pad, ..., s·u − pad + (k_eff − 1); the interval of inputs it can touch is
 * [s·lo − pad, s·hi − pad + (k_eff − 1)]. Returns input-pixel coordinates (negative values lie in the zero padding).
 */
export function traceInterval(layers: Layer[], upTo: number, unit: number): { lo: number; hi: number } {
  let lo = unit, hi = unit;
  for (let i = upTo; i >= 1; i--) {
    const l = layers[i - 1], e = effectiveKernel(l), pad = (e - 1) >> 1;
    lo = l.s * lo - pad; hi = l.s * hi - pad + (e - 1);
  }
  return { lo, hi };
}
/** The exact set of input indices (not only the interval) that one unit reads: shows holes that dilation can leave. */
export function traceSet(layers: Layer[], upTo: number, unit: number): number[] {
  let cur = new Set([unit]);
  for (let i = upTo; i >= 1; i--) {
    const l = layers[i - 1], e = effectiveKernel(l), pad = (e - 1) >> 1, next = new Set<number>();
    for (const u of cur) for (let t = 0; t < l.k; t++) next.add(l.s * u - pad + t * l.d);
    cur = next;
  }
  return [...cur].sort((a, b) => a - b);
}

export interface Unit {
  /** Unit indices along x and y on the output grid, and the input-pixel position of the unit's center (index · jump). */
  ux: number; uy: number; cx: number; cy: number; grid: number; jump: number;
  /** Unclipped receptive-field interval (may leave the image: zero padding) and the part inside the image. */
  x: { lo: number; hi: number }; y: { lo: number; hi: number }; r: number;
  clipped: { x0: number; x1: number; y0: number; y1: number };
  /** Fraction of the image's pixels inside the receptive field. */
  covered: number;
}
/** Snap the requested input position to the nearest output unit and trace its receptive field exactly. */
export function unitAt(layers: Layer[], n: number, x: number, y: number): Unit {
  const { j } = recurrence(layers), grid = gridSizes(layers, n)[layers.length], clamp = (v: number) => Math.max(0, Math.min(grid - 1, Math.round(v / j)));
  const ux = clamp(x), uy = clamp(y), tx = traceInterval(layers, layers.length, ux), ty = traceInterval(layers, layers.length, uy);
  const clipped = { x0: Math.max(0, tx.lo), x1: Math.min(n - 1, tx.hi), y0: Math.max(0, ty.lo), y1: Math.min(n - 1, ty.hi) };
  return {
    ux, uy, cx: ux * j, cy: uy * j, grid, jump: j, x: tx, y: ty, r: tx.hi - tx.lo + 1, clipped,
    covered: ((clipped.x1 - clipped.x0 + 1) * (clipped.y1 - clipped.y0 + 1)) / (n * n),
  };
}

export const pct = (v: number) => `${fmt(100 * v, v >= 0.01 ? 1 : 2)}%`;
/** Plain-language reading of the current unit (pure, so tests can check it). */
export function describe(lang: Lang, stack: Stack, u: Unit): string {
  const name = stack.label[lang === 'zh-tw' ? 'zh' : 'en'];
  return pick(lang,
    `${name}: one output unit sees a ${u.r} × ${u.r} pixel box, ${pct(u.covered)} of the image. Neighboring units are ${u.jump} pixel${u.jump === 1 ? '' : 's'} apart (the jump), so this layer is a ${u.grid} × ${u.grid} grid of units.`,
    `${name}：一個輸出單元看到 ${u.r} × ${u.r} 像素的方框，占影像的 ${pct(u.covered)}。相鄰單元相距 ${u.jump} 像素（跳距），所以這一層是 ${u.grid} × ${u.grid} 的單元網格。`);
}

// ---------------------------------------------------------------------------------------------- drawing

function drawImageWithBox(canvas: HTMLCanvasElement, image: Gray, u: Unit) {
  drawGray(canvas, image);
  const ctx = canvas.getContext('2d')!, { x0, x1, y0, y1 } = u.clipped, w = image.w, h = image.h;
  ctx.fillStyle = 'rgba(0,0,0,0.5)';
  ctx.fillRect(0, 0, w, y0); ctx.fillRect(0, y1 + 1, w, h - y1 - 1); ctx.fillRect(0, y0, x0, y1 - y0 + 1); ctx.fillRect(x1 + 1, y0, w - x1 - 1, y1 - y0 + 1);
  for (const [lw, col] of [[4, 'rgba(0,0,0,0.85)'], [2, '#ffd400']] as const) { ctx.lineWidth = lw; ctx.strokeStyle = col; ctx.strokeRect(x0 - 0.5, y0 - 0.5, x1 - x0 + 2, y1 - y0 + 2); }
  ctx.fillStyle = '#ff4d4d'; ctx.beginPath(); ctx.arc(u.cx + 0.5, u.cy + 0.5, 2.2, 0, 2 * Math.PI); ctx.fill();
}
const PATCH_PX = 252;
/** What the unit sees: its receptive field resampled (nearest neighbor) to PATCH_PX; zero padding is drawn dark blue. */
function drawPatch(canvas: HTMLCanvasElement, image: Gray, u: Unit) {
  if (canvas.width !== PATCH_PX) canvas.width = PATCH_PX;
  if (canvas.height !== PATCH_PX) canvas.height = PATCH_PX;
  canvas.style.imageRendering = 'pixelated';
  const ctx = canvas.getContext('2d')!, im = ctx.createImageData(PATCH_PX, PATCH_PX), px = im.data;
  for (let v = 0; v < PATCH_PX; v++) for (let t = 0; t < PATCH_PX; t++) {
    const sx = u.x.lo + Math.floor((t * u.r) / PATCH_PX), sy = u.y.lo + Math.floor((v * u.r) / PATCH_PX), o = 4 * (v * PATCH_PX + t);
    if (sx < 0 || sy < 0 || sx >= image.w || sy >= image.h) { px[o] = 24; px[o + 1] = 32; px[o + 2] = 64; }
    else { const g = Math.round(image.data[sy * image.w + sx] * 255); px[o] = px[o + 1] = px[o + 2] = g; }
    px[o + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}

export const STACK_W = 420, STACK_H = 404, VIEW = 40;
/** The stack drawn row by row: for the chosen unit, the horizontal extent of the receptive field of its ancestor in each layer. */
function drawStack(canvas: HTMLCanvasElement, lang: Lang, stack: Stack, u: Unit, n: number) {
  const c = plotColors(canvas), ctx = startPlot(canvas, STACK_W, STACK_H, c), layers = stack.layers, rec = recurrence(layers), rowH = 44, top = 62, rows = layers.length + 1;
  const lo = u.cx - VIEW, hi = u.cx + VIEW, x0 = 12, x1 = STACK_W - 12, X = (v: number) => x0 + ((v - lo) / (hi - lo)) * (x1 - x0), bottom = top + rows * rowH - 8;
  label(ctx, pick(lang, 'Receptive field of the chosen unit’s', '所選單元在每一層的祖先單元的'), 12, 20, { bold: true, color: c.ink });
  label(ctx, pick(lang, 'ancestor in each layer, along x', '感受野，沿 x 方向（輸入像素）'), 12, 40, { bold: true, color: c.ink });
  // Outside the image (x below 0 or above n - 1) the unit reads zero padding: shade it.
  ctx.fillStyle = c.muted; ctx.globalAlpha = 0.18;
  if (lo < 0) ctx.fillRect(x0, top - 8, Math.min(x1, X(-0.5)) - x0, bottom - top + 8);
  if (hi > n - 1) ctx.fillRect(Math.max(x0, X(n - 0.5)), top - 8, x1 - Math.max(x0, X(n - 0.5)), bottom - top + 8);
  ctx.globalAlpha = 1;
  for (let l = layers.length; l >= 0; l--) {
    const y = top + (layers.length - l) * rowH, jl = rec.rows[l].j, t = traceInterval(layers, l, Math.round(u.cx / jl)), a = X(t.lo - 0.5), b = X(t.hi + 0.5);
    const i0 = Math.max(a, X(-0.5)), i1 = Math.min(b, X(n - 0.5));
    ctx.fillStyle = c.muted; ctx.globalAlpha = 0.35; ctx.fillRect(a, y + 8, Math.max(2.5, b - a), 14);
    if (i1 > i0) { ctx.fillStyle = c.a; ctx.globalAlpha = 0.45 + 0.5 * l / Math.max(1, layers.length); ctx.fillRect(i0, y + 8, Math.max(2.5, i1 - i0), 14); }
    ctx.globalAlpha = 1;
    const k = l === 0 ? null : layers[l - 1];
    const cap = k === null ? pick(lang, 'Input pixels', '輸入像素') : `L${l} · ${k.k}×${k.k}${k.s > 1 ? pick(lang, `, stride ${k.s}`, `，步幅 ${k.s}`) : ''}${k.d > 1 ? pick(lang, `, dilation ${k.d}`, `，膨脹率 ${k.d}`) : ''}`;
    label(ctx, cap, x0, y + 2, { size: 14, color: c.ink });
    label(ctx, `r = ${rec.rows[l].r}, j = ${jl}`, x1, y + 2, { size: 14, align: 'right', color: c.b });
  }
  line(ctx, X(u.cx), top - 8, X(u.cx), bottom, c.b, 1.5, [4, 3]);
  label(ctx, `x = ${lo}`, x0, bottom + 20, { size: 13, color: c.muted }); label(ctx, `${u.cx}`, X(u.cx), bottom + 20, { size: 13, align: 'center', color: c.b }); label(ctx, `${hi}`, x1, bottom + 20, { size: 13, align: 'right', color: c.muted });
  label(ctx, pick(lang, `${2 * VIEW + 1} input pixels shown${lo < 0 || hi > n - 1 ? '; shaded: zero padding' : ''}`, `顯示 ${2 * VIEW + 1} 個輸入像素${lo < 0 || hi > n - 1 ? '；灰底：零填補' : ''}`), x0, bottom + 40, { size: 13, color: c.muted });
}

export const lab: LabDef = {
  kind: 'image', id: 'receptive-field', track: 'deep-learning',
  title: L('DLCV 01 · How much of the image can one output unit see?', 'DLCV 01・一個輸出單元能看到多少影像？'),
  description: L('Pick a stack of convolution layers and a spot in a real photograph: the exact box of input pixels that one output unit depends on is drawn on the image, with its size, jump and the share of the image it covers.', '選一組卷積層堆疊與真實照片上的一個位置：影像上會畫出某個輸出單元所依賴的輸入像素方框，並顯示其大小、跳距與占整張影像的比例。'),
  challenge: L(
    'Choose “Six 3×3 layers, stride 2 in every second” and click the image: the unit sees 29 × 29 pixels, 1.3% of the picture, and its neighbors are 8 pixels apart. Now choose “Four 3×3 layers, dilations 1, 2, 4, 8”: a similar field, 31 × 31 pixels (1.5%), but the jump stays 1, so every pixel keeps its own output unit. Finally click the top-left corner: why is the covered share smaller there than in the middle of the photograph?',
    '選擇「六層 3×3，每隔一層步幅 2」並點擊影像：這個單元看到 29 × 29 像素，占影像的 1.3%，相鄰單元相距 8 像素。接著選「四層 3×3，膨脹率 1、2、4、8」：感受野差不多，是 31 × 31 像素（1.5%），但跳距維持 1，所以每個像素仍有自己的輸出單元。最後點擊左上角：為什麼那裡被涵蓋的比例比照片中央小？'),
  note: L(
    'Real photograph: skimage.data.camera at 256 × 256 (credit below). Each layer is a square convolution with kernel k, stride s and dilation d, padded with zeros by (k_eff − 1)/2 on each side, where the effective kernel is k_eff = (k − 1)·d + 1. The size and jump come from the recurrence r ← r + (k_eff − 1)·j and j ← j·s (r = j = 1 at the input), and the box on the image from tracing the index interval of one output unit back through every layer: both agree for every stack here, and an independent SciPy computation of which input pixels influence one unit gives the same boxes. The unit snaps to the nearest position on the output grid, whose spacing is the jump. Where the box leaves the image, the part outside is zero padding and the covered share counts only the part inside. This is the theoretical receptive field: the set of pixels that can change the unit. Measured influence is not uniform; it is concentrated near the center of the box, so the effective receptive field is smaller. That is not computed here, and no weights are trained.',
    '真實照片：skimage.data.camera，256 × 256（來源見下方）。每一層是方形卷積，卷積核 k、步幅 s、膨脹率 d，兩側各以零填補 (k_eff − 1)/2，其中有效卷積核 k_eff = (k − 1)·d + 1。大小與跳距由遞迴 r ← r + (k_eff − 1)·j、j ← j·s 得出（輸入處 r = j = 1），影像上的方框則由一個輸出單元的索引區間逐層往回追蹤得到：這裡每一種堆疊兩種算法都一致，另以 SciPy 獨立計算哪些輸入像素會影響一個單元，也得到相同的方框。單元會對齊到輸出網格上最近的位置，網格間距就是跳距。方框超出影像的部分是零填補，被涵蓋的比例只計算影像內的部分。這是理論感受野：可能改變該單元的像素集合。實際影響並不均勻，而是集中在方框中央，所以有效感受野比較小；這裡沒有計算有效感受野，也沒有訓練任何權重。'),
  related: ['deep-learning/dl-01-deep-learning-for-image-processing', 'ml/ml-03-convolutional-networks'],
  controls: [
    { key: 'stack', label: L('Layer stack', '層的堆疊'), kind: 'select', value: DEFAULT.stack, options: STACKS.map(s => ({ value: s.key, label: L(s.label.en, s.label.zh) })) },
    { key: 'x', label: L('Output unit x (snaps to the unit grid)', '輸出單元 x（會對齊單元網格）'), kind: 'range', value: DEFAULT.x, min: 0, max: 255, step: 1 },
    { key: 'y', label: L('Output unit y (snaps to the unit grid)', '輸出單元 y（會對齊單元網格）'), kind: 'range', value: DEFAULT.y, min: 0, max: 255, step: 1 },
  ],
  predict: {
    question: L('Three stacked 3×3 layers give each output unit a 7 × 7 pixel receptive field. Which single change makes it grow the most?', '三層 3×3 卷積讓每個輸出單元的感受野為 7 × 7 像素。哪一種單一改動能讓它增加最多？'),
    options: [
      { value: 'layer', label: L('Add a fourth 3×3 layer', '再加第四層 3×3') },
      { value: 'stride', label: L('Give the first layer stride 2', '讓第一層的步幅為 2') },
      { value: 'dilate', label: L('Give the first layer dilation 2', '讓第一層的膨脹率為 2') },
      { value: 'same', label: L('All three grow it by the same amount', '三種增加的量相同') },
    ],
    answer: 'stride',
    explain: L(
      'Stride 2 in the first layer wins: the field grows from 7 to 11, against 9 for a fourth layer and 9 for dilation 2 in the first layer. A stride doubles the jump, the pixel spacing between neighboring units, for every later layer, so each of the two remaining 3×3 layers now adds 4 pixels instead of 2. Dilation 2 on a layer fed by the input (jump 1) adds only 2 pixels, the same as one more layer. Where the stride sits matters: in the last layer it changes nothing, and the field stays 7.',
      '第一層步幅 2 勝出：感受野由 7 增加到 11，第四層與第一層膨脹率 2 都只增加到 9。步幅會讓之後每一層的跳距（相鄰單元間的像素間距）加倍，所以剩下兩層 3×3 各增加 4 像素，而不是 2 像素。膨脹率 2 作用在直接接收輸入（跳距 1）的層上，只增加 2 像素，與多加一層相同。步幅放的位置很重要：放在最後一層完全沒有影響，感受野仍是 7。'),
  },
  assets: { 'camera-256': 'gray' },
  paneMin: 250,
  panes: [
    { key: 'image', kind: 'canvas', label: L('Photograph with the receptive field of one output unit (yellow box). Click or drag to move the unit', '照片與一個輸出單元的感受野（黃色方框）。按一下或拖曳可移動單元'), width: 256, height: 256 },
    { key: 'patch', kind: 'canvas', label: L('Everything this unit can see, enlarged', '這個單元能看到的全部內容（放大）'), width: PATCH_PX, height: PATCH_PX },
    { key: 'stack', kind: 'canvas', label: L('The stack, layer by layer: r is the receptive field and j the jump after that layer; bars are traced exactly', '逐層畫出的堆疊：r 是該層之後的感受野，j 是跳距；長條是精確追蹤的結果'), width: STACK_W, height: STACK_H },
  ],
  sources: [
    { title: 'André Araujo, Wade Norris and Jack Sim, Computing Receptive Fields of Convolutional Neural Networks, Distill, 2019', url: 'https://distill.pub/2019/computing-receptive-fields/' },
    { title: 'Wenjie Luo, Yujia Li, Raquel Urtasun and Richard Zemel, Understanding the Effective Receptive Field in Deep Convolutional Neural Networks, Advances in Neural Information Processing Systems 29 (NIPS 2016)', url: 'https://proceedings.neurips.cc/paper/2016/hash/c8067ad1937f728f51288b3eb986afaa-Abstract.html' },
  ],
  create({ lang, images }) {
    const image = images['camera-256'] as Gray, n = image.w;
    return {
      attach: (canvases, setControl) => attachProbe(canvases, ['image'], setControl),
      render(state, canvases) {
        const stack = stackByKey(String(state.stack)), u = unitAt(stack.layers, n, num(state, 'x'), num(state, 'y'));
        drawImageWithBox(canvases.image, image, u);
        drawPatch(canvases.patch, image, u);
        drawStack(canvases.stack, lang, stack, u, n);
        return {
          summary: describe(lang, stack, u),
          metrics: [
            { label: pick(lang, 'Receptive field (pixels)', '感受野（像素）'), value: `${u.r} × ${u.r}` },
            { label: pick(lang, 'Jump (cumulative stride)', '跳距（累積步幅）'), value: String(u.jump) },
            { label: pick(lang, 'Share of the image covered', '涵蓋的影像比例'), value: pct(u.covered) },
            { label: pick(lang, 'Output grid', '輸出網格'), value: `${u.grid} × ${u.grid}` },
          ],
        };
      },
    };
  },
};
