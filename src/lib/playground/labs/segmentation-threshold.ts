import type { LabDef } from './types.ts';
import { L, pick, fmt, svg, text, path, num } from '../types.ts';
import type { Gray } from '../../imagekit/types.ts';
import { otsuThreshold, histogram } from '../../imagekit/threshold.ts';
import { gaussianBlur } from '../../imagekit/filter.ts';
import { drawGray, drawMask } from '../../imagekit/draw.ts';
import { iou } from '../../imagekit/metrics.ts';

/** Lighting that falls off linearly to the right: multiplies the page by 1 − ramp · x / (w − 1). */
export function lit(doc: Gray, ramp: number): Float32Array {
  const out = new Float32Array(doc.data.length);
  for (let y = 0; y < doc.h; y++) for (let x = 0; x < doc.w; x++) out[y * doc.w + x] = doc.data[y * doc.w + x] * (1 - ramp * x / (doc.w - 1));
  return out;
}
/** Ink if darker than (1 − c) times the Gaussian-weighted local average (window of `block` pixels). */
export function localMask(im: Float32Array, w: number, h: number, block: number, c: number): Uint8Array {
  const mean = gaussianBlur(im, w, h, (block - 1) / 6), out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = im[i] < mean[i] * (1 - c) ? 1 : 0;
  return out;
}
export function globalMask(im: Float32Array): { mask: Uint8Array; threshold: number } {
  const threshold = otsuThreshold(im), mask = new Uint8Array(im.length);
  for (let i = 0; i < mask.length; i++) mask[i] = im[i] < threshold ? 1 : 0;
  return { mask, threshold };
}

export const lab: LabDef = {
  kind: 'image', id: 'segmentation-threshold', track: 'image-processing',
  title: L('IP 10 · Where does one global threshold fail?', 'IP 10・單一全域門檻在哪裡失效？'),
  description: L('Dim one side of a page and watch Otsu’s single threshold lose the text, while a threshold that follows the local light keeps it.', '讓頁面一側變暗，觀察 Otsu 的單一門檻如何丟失文字，而隨局部光線調整的門檻仍能保留文字。'),
  challenge: L('Start with the lighting ramp at 0, then raise it slowly. At what ramp does the global mask start to lose text on the dim side? Then shrink the local window to 11: why does the local mask get worse?', '先把光線漸變設為 0，再慢慢提高。漸變到多少時，全域遮罩開始在較暗的一側丟失文字？接著把局部視窗縮小到 11：為什麼局部遮罩變差了？'),
  note: L('The page is synthetic (scripts/figures/playground_images.py), so the true text mask is known exactly and IoU is a real score, not a proxy. Lighting multiplies the page by 1 − ramp · x/(w − 1). Global: Otsu’s threshold on the 256-bin histogram. Local: a pixel is ink if it is darker than (1 − c) times the Gaussian-weighted average of its window (σ = (window − 1)/6). IoU is the overlap of the ink mask with the true mask divided by their union.', '頁面是合成的（scripts/figures/playground_images.py），因此真實的文字遮罩完全已知，IoU 是真正的分數而不是代理指標。光線將頁面乘以 1 − 漸變 · x/(w − 1)。全域：以 256 格直方圖計算的 Otsu 門檻。局部：像素若比其視窗高斯加權平均的 (1 − c) 倍更暗就判為墨跡（σ = (視窗 − 1)/6）。IoU 是墨跡遮罩與真實遮罩的交集除以聯集。'),
  related: ['image-processing/dip-10-segmentation'],
  controls: [
    { key: 'ramp', label: L('Lighting falls off to the right by', '光線向右減弱的幅度'), kind: 'range', value: 0.5, min: 0, max: 0.9, step: 0.05 },
    { key: 'block', label: L('Local window (pixels)', '局部視窗（像素）'), kind: 'range', value: 41, min: 11, max: 81, step: 10 },
    { key: 'c', label: L('Local rule: ink if darker than local average by', '局部規則：比局部平均更暗多少即為墨跡'), kind: 'range', value: 0.25, min: 0.05, max: 0.5, step: 0.05 },
  ],
  predict: {
    question: L('The lamp is much stronger on the left than on the right (ramp 0.5). Can one global threshold still separate ink from paper across the whole page?', '左側的燈光比右側強很多（漸變 0.5）。單一全域門檻還能在整頁上分開墨跡與紙張嗎？'),
    options: [
      { value: 'yes', label: L('Yes: ink is always darker than paper', '可以：墨跡永遠比紙張暗') },
      { value: 'part', label: L('Only on part of the page: the dim paper becomes darker than the cutoff', '只有部分頁面可以：較暗一側的紙張會比門檻更暗') },
      { value: 'no', label: L('No: it fails everywhere', '不行：整頁都失敗') },
    ],
    answer: 'part',
    explain: L('Ink is darker than paper only when you compare pixels under the same light. On the dim side the paper reflects less light than the ink on the bright side, so no single number separates the two. The global mask turns the dim paper into ink while the local rule, which compares each pixel with its own surroundings, is unaffected.', '只有在相同光線下比較像素時，墨跡才一定比紙張暗。在較暗的一側，紙張反射的光比明亮一側的墨跡還少，所以沒有任何單一數字能分開兩者。全域遮罩會把暗側的紙張當成墨跡，而逐一與周圍像素比較的局部規則則不受影響。'),
  },
  assets: { doc: 'gray', 'doc-truth': 'gray' },
  paneMin: 460,
  panes: [
    { key: 'scene', kind: 'canvas', label: L('The page under uneven light', '光線不均的頁面'), width: 384, height: 192 },
    { key: 'truth', kind: 'canvas', label: L('True text mask (known exactly)', '真實文字遮罩（完全已知）'), width: 384, height: 192 },
    { key: 'global', kind: 'canvas', label: L('One global Otsu threshold', '單一全域 Otsu 門檻'), width: 384, height: 192 },
    { key: 'local', kind: 'canvas', label: L('Local threshold', '局部門檻'), width: 384, height: 192 },
    { key: 'histogram', kind: 'svg', label: L('Histogram of the lit page and the Otsu threshold', '受光頁面的直方圖與 Otsu 門檻'), width: 640, height: 230, wide: true },
  ],
  sources: [
    { title: 'Nobuyuki Otsu, A Threshold Selection Method from Gray-Level Histograms, IEEE Transactions on Systems, Man, and Cybernetics, 1979', url: 'https://doi.org/10.1109/TSMC.1979.4310076' },
    { title: 'scikit-image documentation, Thresholding (global Otsu and local thresholds)', url: 'https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_thresholding.html' },
  ],
  create({ lang, images }) {
    const doc = images.doc as Gray, truthImg = images['doc-truth'] as Gray, { w, h } = doc;
    const truth = Uint8Array.from(truthImg.data, v => (v > 0.5 ? 1 : 0));
    return {
      render(state, canvases) {
        const ramp = num(state, 'ramp'), block = num(state, 'block'), c = num(state, 'c');
        const im = lit(doc, ramp), g = globalMask(im), l = localMask(im, w, h, block, c);
        drawGray(canvases.scene, { w, h, data: im });
        drawMask(canvases.truth, w, h, truth);
        drawMask(canvases.global, w, h, g.mask);
        drawMask(canvases.local, w, h, l);
        const ig = iou(g.mask, truth), il = iou(l, truth);
        const bins = histogram(im, 64, 0, 1), top = Math.sqrt(Math.max(...bins)), bx = (i: number) => 50 + (i / 64) * 570;
        let plot = path([[50, 30], [50, 190], [620, 190]], 'stroke="var(--plot-grid)"');
        bins.forEach((v, i) => { const bh = 150 * Math.sqrt(v) / top; plot += `<rect x="${bx(i) + 1}" y="${190 - bh}" width="${570 / 64 - 2}" height="${bh}" fill="var(--plot-a)" opacity="0.85"/>`; });
        const tx = 50 + g.threshold * 570;
        plot += path([[tx, 28], [tx, 190]], 'stroke="var(--plot-b)" stroke-width="2.5"') + text(Math.min(tx + 6, 470), 44, `Otsu t = ${fmt(g.threshold, 2)}`, 'fill="var(--plot-b)"')
          + text(50, 212, '0 (black)') + text(620, 212, pick(lang, '1 (white)', '1（白）'), 'text-anchor="end"')
          + text(55, 20, pick(lang, 'Bar height: square root of the pixel count', '長條高度：像素數的平方根'));
        const summary = ramp < 0.05
          ? pick(lang, `With even light the global threshold already scores IoU ${fmt(ig, 2)} and the local rule ${fmt(il, 2)}. Raise the lighting ramp to see them part.`, `光線均勻時，全域門檻的 IoU 已是 ${fmt(ig, 2)}，局部規則則是 ${fmt(il, 2)}。提高光線漸變，看它們如何拉開差距。`)
          : pick(lang, `At ramp ${fmt(ramp, 2)} the paper on the dim side is darker than the Otsu threshold ${fmt(g.threshold, 2)}, so the global mask scores IoU ${fmt(ig, 2)}. The local rule compares each pixel with its own window and scores ${fmt(il, 2)}.`, `漸變 ${fmt(ramp, 2)} 時，較暗一側的紙張比 Otsu 門檻 ${fmt(g.threshold, 2)} 更暗，因此全域遮罩的 IoU 為 ${fmt(ig, 2)}。局部規則逐一與自己的視窗比較，IoU 為 ${fmt(il, 2)}。`);
        return {
          summary,
          metrics: [
            { label: pick(lang, 'IoU, global Otsu', 'IoU，全域 Otsu'), value: fmt(ig, 2) },
            { label: pick(lang, 'IoU, local rule', 'IoU，局部規則'), value: fmt(il, 2) },
            { label: pick(lang, 'Otsu threshold', 'Otsu 門檻'), value: fmt(g.threshold, 2) },
          ],
          plots: { histogram: svg(plot, 230) },
        };
      },
    };
  },
};
