import type { LabDef } from './types.ts';
import { L, pick, fmt, num } from '../types.ts';
import type { Lang } from '../../../i18n/ui';
import { plotColors, startPlot, label } from './plot2d.ts';

// ---------------------------------------------------------------------------------------------- the synthetic label map
// A 128 x 128 label map drawn by code (no photographs, no random numbers): a large background, one disk and one block of
// medium size, a 3-pixel-wide pole and five 6 x 6 squares. Class 4 (the squares) is the smallest class by pixel count.
export const W = 128, H = 128, K = 5;
export const STRIDES = [1, 2, 4, 8, 16, 32];
export const DEFAULT_STRIDE = 8;
export const CLASS_NAMES = [
  { en: 'background', zh: '背景' }, { en: 'disk', zh: '圓盤' }, { en: 'block', zh: '方塊' }, { en: 'thin pole', zh: '細桿' }, { en: 'small squares', zh: '小方塊' },
];
export const CLASS_COLORS: [number, number, number][] = [[226, 226, 222], [36, 102, 194], [224, 138, 30], [42, 157, 92], [196, 40, 120]];
const DISK = { cx: 38, cy: 66, r: 21 }, BLOCK = { x0: 80, x1: 104, y0: 20, y1: 46 }, POLE = { x0: 118, x1: 121, y0: 30, y1: 118 };
const SQUARES = { size: 6, at: [[18, 100], [50, 108], [84, 84], [97, 110], [66, 76]] as [number, number][] };

/** The ground-truth label map, row-major, classes 0..4. */
export function makeScene(): Uint8Array {
  const g = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if ((x - DISK.cx) ** 2 + (y - DISK.cy) ** 2 < DISK.r ** 2) g[y * W + x] = 1;
  for (let y = BLOCK.y0; y < BLOCK.y1; y++) for (let x = BLOCK.x0; x < BLOCK.x1; x++) g[y * W + x] = 2;
  for (let y = POLE.y0; y < POLE.y1; y++) for (let x = POLE.x0; x < POLE.x1; x++) g[y * W + x] = 3;
  for (const [sx, sy] of SQUARES.at) for (let y = sy; y < sy + SQUARES.size; y++) for (let x = sx; x < sx + SQUARES.size; x++) g[y * W + x] = 4;
  return g;
}

// ---------------------------------------------------------------------------------------------- the simulated network
/**
 * Ideal stride-s score map: for every s x s cell, the fraction of its pixels that belong to each class. This is the best a head
 * at output stride s can say about a cell, and the same construction as the chapter's toy experiment. Layout [class][cell row][cell col].
 */
export function coarseScores(labels: Uint8Array, w: number, h: number, k: number, s: number): { scores: Float64Array; gh: number; gw: number } {
  if (w % s || h % s) throw new Error('the stride must divide the image size');
  const gh = h / s, gw = w / s, scores = new Float64Array(k * gh * gw), unit = 1 / (s * s);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) scores[(labels[y * w + x] * gh + Math.floor(y / s)) * gw + Math.floor(x / s)] += unit;
  return { scores, gh, gw };
}
function argmaxAt(scores: Float64Array, k: number, plane: number, i: number): number {
  let best = 0;
  for (let c = 1; c < k; c++) if (scores[c * plane + i] > scores[best * plane + i]) best = c; // ties go to the lowest class index, as numpy.argmax
  return best;
}
/** Decide at low resolution (argmax per cell), then enlarge the label map by nearest-neighbour: every cell becomes an s x s block. */
export function upsampleNearest(scores: Float64Array, gh: number, gw: number, k: number, s: number): Uint8Array {
  const w = gw * s, h = gh * s, plane = gh * gw, cell = new Uint8Array(plane), out = new Uint8Array(w * h);
  for (let i = 0; i < plane; i++) cell[i] = argmaxAt(scores, k, plane, i);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = cell[Math.floor(y / s) * gw + Math.floor(x / s)];
  return out;
}
/**
 * Enlarge the class scores bilinearly (pixel-centre alignment, borders replicated: scipy.ndimage.zoom(order=1, mode="nearest",
 * grid_mode=True)), then decide per pixel by argmax.
 */
export function upsampleBilinear(scores: Float64Array, gh: number, gw: number, k: number, s: number): Uint8Array {
  const w = gw * s, h = gh * s, plane = gh * gw, out = new Uint8Array(w * h), z = new Float64Array(k);
  for (let y = 0; y < h; y++) {
    const fy = Math.min(gh - 1, Math.max(0, (y + 0.5) / s - 0.5)), y0 = Math.floor(fy), y1 = Math.min(y0 + 1, gh - 1), b = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(gw - 1, Math.max(0, (x + 0.5) / s - 0.5)), x0 = Math.floor(fx), x1 = Math.min(x0 + 1, gw - 1), a = fx - x0;
      let best = 0;
      for (let c = 0; c < k; c++) {
        const o = c * plane;
        z[c] = (1 - a) * (1 - b) * scores[o + y0 * gw + x0] + a * (1 - b) * scores[o + y0 * gw + x1] + (1 - a) * b * scores[o + y1 * gw + x0] + a * b * scores[o + y1 * gw + x1];
        if (c > 0 && z[c] > z[best]) best = c;
      }
      out[y * w + x] = best;
    }
  }
  return out;
}
export type Upsampling = 'nearest' | 'bilinear';
/** The prediction of a network with output stride `s`, whose scores are enlarged by the chosen method. Stride 1 returns the ground truth. */
export function predict(labels: Uint8Array, w: number, h: number, k: number, s: number, mode: Upsampling): Uint8Array {
  if (s === 1) return Uint8Array.from(labels);
  const { scores, gh, gw } = coarseScores(labels, w, h, k, s);
  return mode === 'bilinear' ? upsampleBilinear(scores, gh, gw, k, s) : upsampleNearest(scores, gh, gw, k, s);
}

// ---------------------------------------------------------------------------------------------- the metrics (as in the chapter)
/** K x K confusion matrix, rows = true class, columns = predicted class (flattened). */
export function confusion(gt: Uint8Array, pred: Uint8Array, k: number): Float64Array {
  const c = new Float64Array(k * k);
  for (let i = 0; i < gt.length; i++) c[gt[i] * k + pred[i]]++;
  return c;
}
export interface SegScores { pixelAcc: number; iou: number[]; mIoU: number; counts: number[]; wrong: number }
/**
 * FCN's definitions: pixel accuracy = sum_i n_ii / sum_i t_i, IoU_i = n_ii / (t_i + sum_j n_ji - n_ii), mIoU = the mean of the IoU
 * over all K classes (the background is a class like any other).
 */
export function segScores(gt: Uint8Array, pred: Uint8Array, k: number): SegScores {
  const c = confusion(gt, pred, k), iou: number[] = [], counts: number[] = [];
  let tp = 0, total = 0;
  for (let i = 0; i < k; i++) {
    let t = 0, p = 0;
    for (let j = 0; j < k; j++) { t += c[i * k + j]; p += c[j * k + i]; }
    iou.push(c[i * k + i] / (t + p - c[i * k + i])); counts.push(t); tp += c[i * k + i]; total += t;
  }
  return { pixelAcc: tp / total, iou, mIoU: iou.reduce((a, b) => a + b, 0) / k, counts, wrong: total - tp };
}
/** Index of the class with the fewest ground-truth pixels. */
export const smallestClass = (counts: number[]) => counts.reduce((best, v, i) => (v < counts[best] ? i : best), 0);

// ---------------------------------------------------------------------------------------------- drawing
const DRAW = 2; // each label pixel is drawn as a 2 x 2 block, so the canvas stays sharp when the page scales it
function paint(canvas: HTMLCanvasElement, labels: Uint8Array, pixel: (i: number) => [number, number, number]) {
  const w = W * DRAW, h = H * DRAW;
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  const ctx = canvas.getContext('2d')!, im = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const [r, g, b] = pixel(Math.floor(y / DRAW) * W + Math.floor(x / DRAW)), o = 4 * (y * w + x);
    im.data[o] = r; im.data[o + 1] = g; im.data[o + 2] = b; im.data[o + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
  void labels;
}
const WRONG: [number, number, number] = [214, 39, 40];
const pale = (c: [number, number, number]): [number, number, number] => [Math.round(0.35 * c[0] + 0.65 * 255), Math.round(0.35 * c[1] + 0.65 * 255), Math.round(0.35 * c[2] + 0.65 * 255)];
const className = (lang: Lang, i: number) => pick(lang, CLASS_NAMES[i].en, CLASS_NAMES[i].zh);

const CH_W = 400, CH_H = 262;
function drawChart(canvas: HTMLCanvasElement, lang: Lang, r: SegScores, small: number) {
  const col = plotColors(canvas), ctx = startPlot(canvas, CH_W, CH_H, col), x0 = 134, span = 196, rowH = 32;
  const rows: { name: string; value: number; color: string; bold?: boolean }[] = [
    { name: pick(lang, 'pixel accuracy', '像素準確率'), value: r.pixelAcc, color: col.ink },
    { name: 'mIoU', value: r.mIoU, color: col.muted },
    ...r.iou.map((v, i) => ({ name: className(lang, i), value: v, color: `rgb(${CLASS_COLORS[i].join(',')})`, bold: i === small })),
  ];
  label(ctx, pick(lang, 'Score (0 to 1)', '分數（0 到 1）'), 8, 20, { color: col.muted, size: 14 });
  for (const t of [0, 0.5, 1]) {
    const x = x0 + t * span;
    ctx.strokeStyle = col.grid; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, 30); ctx.lineTo(x, 30 + rows.length * rowH + 14); ctx.stroke();
    label(ctx, String(t), x, 24, { color: col.muted, size: 13, align: 'center' });
  }
  rows.forEach((row, i) => {
    const y = 38 + i * rowH + (i >= 2 ? 14 : 0), bw = Math.max(0, row.value) * span;
    label(ctx, row.name, 8, y + 17, { color: col.ink, size: 14, bold: row.bold });
    ctx.fillStyle = row.color; ctx.fillRect(x0, y + 2, bw, rowH - 8);
    if (i >= 2) { ctx.lineWidth = 1; ctx.strokeStyle = col.ink; ctx.strokeRect(x0 + 0.5, y + 2.5, Math.max(bw - 1, 0), rowH - 9); }
    label(ctx, fmt(row.value, 3), x0 + bw + 6, y + 18, { color: col.ink, size: 14, bold: row.bold });
  });
  label(ctx, pick(lang, 'IoU per class (thick name = smallest class)', '各類別的 IoU（粗體 = 最小的類別）'), 8, 38 + 2 * rowH + 8, { color: col.muted, size: 13 });
}

// ---------------------------------------------------------------------------------------------- the lab
export const lab: LabDef = {
  kind: 'image', id: 'segmentation-metrics', track: 'deep-learning',
  title: L('DLCV 03B · Does high pixel accuracy mean a good segmentation?', 'DLCV 03B・像素準確率高，就代表分割得好嗎？'),
  description: L('Simulate a segmentation network that predicts at a coarse output stride and compare pixel accuracy, mean IoU and the IoU of the smallest class.', '模擬一個以較粗的輸出步幅（stride）做預測的分割網路，比較像素準確率、平均 IoU 與最小類別的 IoU。'),
  challenge: L(
    'At the default output stride 8 the prediction gets about 95% of the pixels right, yet the small squares have an IoU of 0. Lower the stride to 4, then 2: at which stride does the thin pole come back, and why does it need a finer stride than the squares? Then raise the stride to 32: how many of the five classes still have a non-zero IoU? Finally switch the enlarging method to bilinear at stride 8: which numbers change, and which one stays at 0?',
    '預設的輸出步幅為 8，預測約有 95% 的像素正確，但小方塊的 IoU 卻是 0。把步幅降到 4，再降到 2：細桿在哪個步幅才回來？為什麼它比小方塊需要更細的步幅？接著把步幅升到 32：五個類別中還有幾個的 IoU 不是 0？最後在步幅 8 把放大方式改成雙線性：哪些數字改變了，哪一個仍然是 0？'),
  note: L(
    'The label map is synthetic (128 × 128, drawn by code, no random numbers): a background of 85% of the pixels, a disk, a block, a 3-pixel pole and five 6 × 6 squares, all in separate classes. The network is idealized: at output stride s its score for each class in an s × s cell is the fraction of the cell’s pixels that belong to the class, which is the best a stride-s head can report. With nearest enlargement the class with the largest fraction labels the whole cell (ties go to the lowest class number); with bilinear enlargement the fractions are interpolated per class and the largest score labels each pixel. It is the construction of the chapter’s toy experiment on a smaller scene, so its numbers are this lab’s own and not the chapter’s. The scores follow the chapter: pixel accuracy is the share of correct pixels, the IoU of a class is true positives divided by true positives plus misses plus false alarms, and mIoU is the mean of the IoU over all five classes, the background included, computed from one confusion matrix.',
    '標籤圖是合成的（128 × 128，由程式畫出，沒有使用隨機數）：約 85% 的像素是背景，另有一個圓盤、一個方塊、一根 3 像素寬的細桿與五個 6 × 6 的小方塊，各自是獨立的類別。網路是理想化的：在輸出步幅 s 下，每個 s × s 格子對各類別的分數，就是格內屬於該類別的像素比例，這已是步幅 s 的預測頭所能給出的最好結果。最近鄰放大時，比例最大的類別決定整個格子的標籤（平手時取編號最小的類別）；雙線性放大則逐類別對比例做內插，再由最大的分數決定每個像素。它是章節玩具實驗在較小場景上的相同做法，因此數字屬於這個實驗本身，不是章節中的數字。各項分數依章節定義：像素準確率是預測正確的像素比例，類別的 IoU 是真陽性除以（真陽性 + 漏判 + 誤判），mIoU 是五個類別（含背景）IoU 的平均，由同一個混淆矩陣算出。'),
  related: ['deep-learning/dl-03b-image-segmentation'],
  controls: [
    { key: 'stride', label: L('Output stride of the network (pixels per score cell)', '網路的輸出步幅（每個分數格子的像素數）'), kind: 'select', value: String(DEFAULT_STRIDE), options: STRIDES.map(s => ({ value: String(s), label: L(s === 1 ? '1 (full resolution)' : String(s), s === 1 ? '1（完整解析度）' : String(s)) })) },
    { key: 'up', label: L('How the coarse map is enlarged', '粗糙分數圖的放大方式'), kind: 'select', value: 'nearest', options: [
      { value: 'nearest', label: L('Nearest: enlarge the labels', '最近鄰：放大標籤') },
      { value: 'bilinear', label: L('Bilinear: enlarge the scores', '雙線性：放大分數') },
    ] },
  ],
  predict: {
    question: L('A network predicts at output stride 8, so one score cell covers 8 × 8 pixels. Its cell scores are the best possible (the exact class fractions). The thin pole is 3 pixels wide and the small squares are 6 × 6. What do the scores look like?', '一個網路在輸出步幅 8 下預測，因此一個分數格子涵蓋 8 × 8 個像素，而且格子的分數是最理想的（精確的類別比例）。細桿寬 3 像素，小方塊為 6 × 6。各項分數會是什麼樣子？'),
    options: [
      { value: 'fall-together', label: L('Pixel accuracy falls as far as the small classes do: all of them end up below 0.6', '像素準確率和小類別一起下降：全部都低於 0.6') },
      { value: 'accuracy-high', label: L('Pixel accuracy stays above 0.9, while the thin pole and the small squares have an IoU near 0', '像素準確率仍高於 0.9，但細桿與小方塊的 IoU 接近 0') },
      { value: 'all-fine', label: L('Everything stays above 0.9, including the IoU of the small classes', '所有分數都高於 0.9，包括小類別的 IoU') },
    ],
    answer: 'accuracy-high',
    explain: L(
      'Pixel accuracy stays high while the small classes disappear. In an 8 × 8 cell the pole fills at most 3 × 8 = 24 of the 64 pixels, and none of the five squares fills more than 24 of 64 either, because they straddle cell borders. The background holds the largest share of every such cell, so the whole cell is labelled background. The pole and the squares together hold only about 2.7% of the pixels, so losing them costs little pixel accuracy, but every class counts one fifth of mIoU. Pixel accuracy mostly measures the large background.',
      '像素準確率仍然很高，小類別卻消失了。在 8 × 8 的格子裡，細桿最多只占 64 個像素中的 3 × 8 = 24 個，五個小方塊也沒有任何一個在格子裡超過 24 個，因為它們都跨過格子邊界。在這些格子裡，背景的占比最大，所以整個格子被標成背景。細桿與小方塊合計只占約 2.7% 的像素，所以失去它們對像素準確率影響很小，但每個類別在 mIoU 中各占五分之一。像素準確率主要衡量的是大面積的背景。'),
  },
  assets: {},
  paneMin: 300,
  panes: [
    { key: 'truth', kind: 'canvas', label: L('Ground truth (5 classes)', '真實標籤（5 個類別）'), width: W * DRAW, height: H * DRAW },
    { key: 'pred', kind: 'canvas', label: L('Prediction of the simulated network', '模擬網路的預測'), width: W * DRAW, height: H * DRAW },
    { key: 'wrong', kind: 'canvas', label: L('Wrong pixels in red', '預測錯誤的像素（紅色）'), width: W * DRAW, height: H * DRAW },
    { key: 'chart', kind: 'canvas', label: L('Pixel accuracy, mIoU and the IoU of every class', '像素準確率、mIoU 與各類別的 IoU'), width: CH_W, height: CH_H },
  ],
  sources: [
    { title: 'J. Long, E. Shelhamer and T. Darrell, Fully Convolutional Networks for Semantic Segmentation, CVPR, 2015 (pixel accuracy, mean IoU and output stride)', url: 'https://doi.org/10.1109/CVPR.2015.7298965' },
    { title: 'L.-C. Chen et al., DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs, IEEE TPAMI, 2018 (output stride 8 and 16)', url: 'https://doi.org/10.1109/TPAMI.2017.2699184' },
  ],
  create({ lang }) {
    const gt = makeScene(), cache = new Map<string, { pred: Uint8Array; scores: SegScores }>();
    const truthScores = segScores(gt, gt, K), small = smallestClass(truthScores.counts);
    let truthDrawn = false;
    return {
      render(state, canvases) {
        const s = STRIDES.includes(num(state, 'stride')) ? num(state, 'stride') : DEFAULT_STRIDE, mode: Upsampling = state.up === 'bilinear' ? 'bilinear' : 'nearest', key = `${s}-${mode}`;
        if (!cache.has(key)) { const pred = predict(gt, W, H, K, s, mode); cache.set(key, { pred, scores: segScores(gt, pred, K) }); }
        const { pred, scores: r } = cache.get(key)!;
        if (!truthDrawn) { paint(canvases.truth, gt, i => CLASS_COLORS[gt[i]]); truthDrawn = true; }
        paint(canvases.pred, pred, i => CLASS_COLORS[pred[i]]);
        paint(canvases.wrong, pred, i => (pred[i] === gt[i] ? pale(CLASS_COLORS[gt[i]]) : WRONG));
        drawChart(canvases.chart, lang, r, small);
        const name = className(lang, small), share = fmt(100 * r.counts[small] / (W * H), 1), iouSmall = r.iou[small];
        const summary = s === 1
          ? pick(lang, 'At output stride 1 the prediction is the ground truth, so every score is 1. Raise the stride to see which scores fall first.', '輸出步幅為 1 時，預測就是真實標籤，所有分數都是 1。提高步幅，看看哪些分數最先下降。')
          : iouSmall < 0.2
            ? pick(lang, `At output stride ${s}, ${fmt(100 * r.pixelAcc, 1)}% of the pixels are right (pixel accuracy ${fmt(r.pixelAcc, 3)}), yet mIoU is only ${fmt(r.mIoU, 3)} and the ${name} score an IoU of ${fmt(iouSmall, 3)}. They fill ${share}% of the image, so losing them barely moves pixel accuracy, but every class has the same weight 1/${K} in mIoU.`, `輸出步幅 ${s} 時，${fmt(100 * r.pixelAcc, 1)}% 的像素是對的（像素準確率 ${fmt(r.pixelAcc, 3)}），但 mIoU 只有 ${fmt(r.mIoU, 3)}，而${name}的 IoU 只有 ${fmt(iouSmall, 3)}。它們只占畫面的 ${share}%，所以失去它們幾乎不影響像素準確率，但每個類別在 mIoU 中的權重都是 1/${K}。`)
            : pick(lang, `At output stride ${s}, pixel accuracy is ${fmt(r.pixelAcc, 3)} and mIoU ${fmt(r.mIoU, 3)}; the ${name} (${share}% of the image) reach an IoU of ${fmt(iouSmall, 3)}. The smaller a class, the earlier its IoU falls as the stride grows.`, `輸出步幅 ${s} 時，像素準確率為 ${fmt(r.pixelAcc, 3)}，mIoU 為 ${fmt(r.mIoU, 3)}；${name}（占畫面 ${share}%）的 IoU 為 ${fmt(iouSmall, 3)}。類別越小，隨著步幅變大，它的 IoU 越早下降。`);
        return {
          summary,
          metrics: [
            { label: pick(lang, 'Pixel accuracy', '像素準確率'), value: fmt(r.pixelAcc, 3) },
            { label: 'mIoU', value: fmt(r.mIoU, 3) },
            { label: pick(lang, `IoU of the smallest class (${name})`, `最小類別（${name}）的 IoU`), value: fmt(iouSmall, 3) },
            { label: pick(lang, 'Wrong pixels', '預測錯誤的像素'), value: `${fmt(100 * r.wrong / (W * H), 1)}%` },
          ],
        };
      },
    };
  },
};
