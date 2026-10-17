import type { LabDef } from './types.ts';
import { L, pick, fmt, num } from '../types.ts';
import type { Gray } from '../../imagekit/types.ts';
import { structuringElement, elementGrid, elementSize, applyOperation, type ElementShape, type Operation } from '../../imagekit/morphology.ts';
import { labelComponents, countHoles } from '../../imagekit/label.ts';

/** Connected pieces smaller than this many pixels count as clutter (specks and scratches); the smallest coin has over a thousand. */
export const SMALL = 200;
export const SHAPES: ElementShape[] = ['disk', 'square', 'hline', 'vline'];
export const OPERATIONS: Operation[] = ['erosion', 'dilation', 'opening', 'closing', 'gradient'];

export const maskFromGray = (g: Gray): Uint8Array => Uint8Array.from(g.data, v => (v > 0.5 ? 1 : 0));

/** What the lab measures about the input mask; computed once. */
export interface InputStats { area: number; components: number; labels: Int32Array; small: number[]; holes: number }
export function inputStats(mask: Uint8Array, w: number, h: number): InputStats {
  const lab = labelComponents(mask, w, h, 8);
  let area = 0;
  for (const v of mask) area += v;
  const small: number[] = [];
  lab.areas.forEach((a, i) => { if (a < SMALL) small.push(i + 1); });
  return { area, components: lab.count, labels: lab.labels, small, holes: countHoles(mask, w, h, 8) };
}
export interface Analysis { output: Uint8Array; areaOut: number; areaChangePct: number; componentsOut: number; smallGone: number; holesOut: number; seSize: number; seSide: number }
/** Run one operation and measure it: foreground area change, components (8-connected), clutter that disappeared, holes. */
export function analyse(mask: Uint8Array, w: number, h: number, st: InputStats, op: Operation, shape: ElementShape, r: number): Analysis {
  const se = structuringElement(shape, r), output = applyOperation(op, mask, w, h, se);
  let areaOut = 0;
  for (const v of output) areaOut += v;
  const kept = new Uint8Array(st.components + 1);
  for (let i = 0; i < output.length; i++) if (output[i] && st.labels[i]) kept[st.labels[i]] = 1;
  const smallGone = st.small.filter(id => !kept[id]).length;
  return {
    output, areaOut, areaChangePct: (100 * (areaOut - st.area)) / st.area, componentsOut: labelComponents(output, w, h, 8).count,
    smallGone, holesOut: countHoles(output, w, h, 8), seSize: elementSize(se), seSide: 2 * r + 1,
  };
}

const names: Record<Operation, [string, string]> = { erosion: ['Erosion', '侵蝕'], dilation: ['Dilation', '膨脹'], opening: ['Opening', '開運算'], closing: ['Closing', '閉運算'], gradient: ['Gradient', '梯度'] };
const shapeNames: Record<ElementShape, [string, string]> = { disk: ['disk', '圓盤'], square: ['square', '方形'], hline: ['horizontal line', '水平線'], vline: ['vertical line', '垂直線'] };

function paintMasks(canvas: HTMLCanvasElement, w: number, h: number, pixel: (i: number) => [number, number, number]) {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d')!, im = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) { const [r, g, b] = pixel(i); im.data[4 * i] = r; im.data[4 * i + 1] = g; im.data[4 * i + 2] = b; im.data[4 * i + 3] = 255; }
  ctx.putImageData(im, 0, 0);
}
const INK: [number, number, number] = [28, 32, 48], PAPER: [number, number, number] = [250, 250, 248], LOST: [number, number, number] = [214, 39, 40], GAINED: [number, number, number] = [36, 110, 214];

/** Draw the structuring element as a grid of squares, one per pixel, with its origin marked. */
function drawElement(canvas: HTMLCanvasElement, w: number, h: number, shape: ElementShape, r: number, label: string) {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d')!, se = structuringElement(shape, r), grid = elementGrid(se), n = 2 * r + 1;
  const cell = Math.max(3, Math.floor((Math.min(w, h) - 70) / n)), x0 = Math.round((w - cell * n) / 2), y0 = Math.round((h - 24 - cell * n) / 2) + 6;
  ctx.fillStyle = '#fafaf8'; ctx.fillRect(0, 0, w, h);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    if (grid[j * n + i]) { ctx.fillStyle = '#1c2030'; ctx.fillRect(x0 + i * cell, y0 + j * cell, cell, cell); }
  }
  if (cell >= 7) {
    ctx.strokeStyle = 'rgba(250,250,248,0.55)'; ctx.lineWidth = 1;
    for (let k = 0; k <= n; k++) {
      ctx.beginPath(); ctx.moveTo(x0 + k * cell + 0.5, y0); ctx.lineTo(x0 + k * cell + 0.5, y0 + n * cell); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x0, y0 + k * cell + 0.5); ctx.lineTo(x0 + n * cell, y0 + k * cell + 0.5); ctx.stroke();
    }
  }
  ctx.strokeStyle = '#8a929c'; ctx.lineWidth = 1; ctx.strokeRect(x0 - 0.5, y0 - 0.5, n * cell + 1, n * cell + 1);
  ctx.fillStyle = '#e8590c'; ctx.beginPath(); ctx.arc(x0 + (r + 0.5) * cell, y0 + (r + 0.5) * cell, Math.max(2, cell * 0.28), 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = '#4a5260'; ctx.font = '14px sans-serif'; ctx.textAlign = 'center'; ctx.fillText(label, w / 2, h - 10);
}

export const lab: LabDef = {
  kind: 'image', id: 'morphology-shapes', track: 'image-processing',
  title: L('IP 09 · Which shapes does a structuring element keep?', 'IP 09・結構元素會留下哪些形狀？'),
  description: L('Clean a thresholded photo of coins that carries specks, scratches and holes. Change the operation, the shape and the size of the structuring element, and see what each choice removes, fills or keeps.', '清理一張經過門檻處理、帶有孤立點、刮痕與孔洞的硬幣照片。改變運算、結構元素的形狀與大小，觀察各種選擇會移除、填補或保留什麼。'),
  challenge: L('With opening and a disk of size 4, all clutter goes. Now switch the element to a horizontal line of size 4: which pieces survive, and why? Then raise the disk size with opening until a coin disappears. Finally switch to closing: what does it fix that opening cannot?', '在開運算、圓盤大小 4 時，所有雜物都消失了。現在把元素換成大小 4 的水平線：哪些部分會留下來？為什麼？接著用開運算逐步加大圓盤，直到有一枚硬幣消失。最後換成閉運算：它能修好哪些開運算修不好的缺陷？'),
  note: L('The photograph is skimage’s coins image (303 × 384). The clean mask smooths it (Gaussian, σ = 2), subtracts a wide Gaussian background (σ = 50) so the bright corner does not count, and thresholds with Otsu’s method; that gives exactly 24 solid coins. Clutter is then added with a fixed seed so the answer is known: 40 specks (blobs of radius 0 to 3 pixels), 10 thin scratches (5 horizontal, 5 vertical) and 18 round holes of radius 1 to 3 pixels inside the coins (scripts/figures/playground_restore_assets.py). Notation follows the chapter: A is the foreground, B the structuring element, opening = erosion then dilation, closing = dilation then erosion, gradient = dilation minus erosion. Size r means a disk of radius r (all offsets with dx² + dy² ≤ r²), a square of (2r + 1) × (2r + 1) pixels, or a line of 2r + 1 pixels. Pixels outside the image count as background. Components are 8-connected; a hole is a background region that does not touch the border; clutter means a connected piece under 200 pixels (the smallest coin has more than a thousand). In the output picture, black pixels stay foreground, red were foreground and are gone, blue are new.', '照片是 skimage 的 coins 影像（303 × 384）。乾淨遮罩的做法是先平滑（高斯，σ = 2），再減去寬的高斯背景（σ = 50），使明亮的角落不被算進去，最後用 Otsu 方法取門檻；結果恰好是 24 枚實心硬幣。接著以固定種子加入雜物，使答案已知：40 個孤立點（半徑 0 到 3 像素的小塊）、10 條細刮痕（5 條水平、5 條垂直）以及 18 個位於硬幣內、半徑 1 到 3 像素的圓形孔洞（scripts/figures/playground_restore_assets.py）。符號依照章節：A 是前景，B 是結構元素，開運算 = 先侵蝕再膨脹，閉運算 = 先膨脹再侵蝕，梯度 = 膨脹減侵蝕。大小 r 的意思是：圓盤為半徑 r（所有滿足 dx² + dy² ≤ r² 的位移）、方形為 (2r + 1) × (2r + 1) 像素、線段為 2r + 1 像素。影像之外的像素視為背景。連通元件採 8 連通；孔洞是沒有碰到邊界的背景區域；雜物指小於 200 像素的連通塊（最小的硬幣超過一千像素）。輸出圖中，黑色是仍為前景的像素，紅色是原本為前景而消失的像素，藍色是新增的像素。'),
  related: ['image-processing/dip-09-morphology'],
  controls: [
    { key: 'op', label: L('Operation', '運算'), kind: 'select', value: 'opening', options: OPERATIONS.map(o => ({ value: o, label: L(names[o][0], names[o][1]) })) },
    { key: 'shape', label: L('Structuring element', '結構元素'), kind: 'select', value: 'disk', options: SHAPES.map(s => ({ value: s, label: L(shapeNames[s][0][0].toUpperCase() + shapeNames[s][0].slice(1), shapeNames[s][1]) })) },
    { key: 'size', label: L('Size r (the element spans 2r + 1 pixels)', '大小 r（元素寬 2r + 1 像素）'), kind: 'range', value: 4, min: 1, max: 20, step: 1 },
  ],
  predict: {
    question: L('Open the mask with a disk of radius 4. Every speck and scratch is thinner than the disk (at most 7 pixels wide), and every coin is far wider. What happens?', '用半徑 4 的圓盤對遮罩做開運算。每個孤立點與刮痕都比圓盤細（寬度最多 7 像素），每枚硬幣則寬得多。會發生什麼事？'),
    options: [
      { value: 'clean', label: L('The clutter disappears and the coins keep their shape (only the corners round off)', '雜物消失，硬幣保持原本的形狀（只有角落稍微變圓）') },
      { value: 'shrunk', label: L('The clutter disappears but every coin shrinks by about 4 pixels all around', '雜物消失，但每枚硬幣都向內縮小約 4 像素') },
      { value: 'both', label: L('The specks, scratches and coins all survive', '孤立點、刮痕與硬幣全部留下') },
    ],
    answer: 'clean',
    explain: L('Erosion removes everything the disk does not fit inside, and the dilation that follows grows back exactly what survived, so opening keeps every place where the disk fits. The mask starts with 74 components (24 coins and 50 small ones); after the opening 24 remain and all 50 small pieces are gone. The coins lose only 0.10% of their pixels. The 18 holes are still there: opening removes foreground, it does not fill background.', '侵蝕會移除圓盤放不進去的所有地方，接著的膨脹會把留下的部分恰好長回來，所以開運算保留圓盤能放得進去的每個位置。遮罩一開始有 74 個連通元件（24 枚硬幣與 50 個小塊）；開運算後剩下 24 個，50 個小塊全部消失。硬幣只損失 0.10% 的像素。18 個孔洞仍然存在：開運算移除的是前景，不會填補背景。'),
  },
  assets: { 'coins-mask': 'gray' },
  paneMin: 300,
  panes: [
    { key: 'input', kind: 'canvas', label: L('Input mask: coins with specks, scratches and holes', '輸入遮罩：帶有孤立點、刮痕與孔洞的硬幣'), width: 384, height: 303 },
    { key: 'output', kind: 'canvas', label: L('Output (black stays, red removed, blue added)', '輸出（黑色保留、紅色被移除、藍色為新增）'), width: 384, height: 303 },
    { key: 'element', kind: 'canvas', label: L('The structuring element (each square is one pixel; the orange dot is the origin)', '結構元素（每個方格是一個像素；橘色點是原點）'), width: 384, height: 303 },
  ],
  sources: [
    { title: 'Rafael C. Gonzalez & Richard E. Woods, Digital Image Processing, 4th ed., Pearson, 2018 (morphological image processing)', url: 'https://www.imageprocessingplace.com/DIP-4E/dip4e_main_page.htm' },
    { title: 'Nobuyuki Otsu, A Threshold Selection Method from Gray-Level Histograms, IEEE Transactions on Systems, Man, and Cybernetics, 1979', url: 'https://doi.org/10.1109/TSMC.1979.4310076' },
    { title: 'scikit-image documentation, skimage.morphology (footprints, erosion, dilation, opening, closing)', url: 'https://scikit-image.org/docs/stable/api/skimage.morphology.html' },
    { title: 'scikit-image documentation, skimage.measure.label (connected components)', url: 'https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label' },
  ],
  create({ lang, images }) {
    const src = images['coins-mask'] as Gray, { w, h } = src, mask = maskFromGray(src), st = inputStats(mask, w, h);
    return {
      render(state, canvases) {
        const op = (OPERATIONS.includes(state.op as Operation) ? state.op : 'opening') as Operation, shape = (SHAPES.includes(state.shape as ElementShape) ? state.shape : 'disk') as ElementShape, r = Math.round(num(state, 'size'));
        const a = analyse(mask, w, h, st, op, shape, r);
        paintMasks(canvases.input, w, h, i => (mask[i] ? INK : PAPER));
        paintMasks(canvases.output, w, h, i => (mask[i] && a.output[i] ? INK : mask[i] ? LOST : a.output[i] ? GAINED : PAPER));
        const seLabel = shape === 'hline' ? `${a.seSide} × 1` : shape === 'vline' ? `1 × ${a.seSide}` : `${a.seSide} × ${a.seSide}`;
        drawElement(canvases.element, w, h, shape, r, pick(lang, `${shapeNames[shape][0]}, ${seLabel} box, ${a.seSize} pixels`, `${shapeNames[shape][1]}，${seLabel} 範圍，${a.seSize} 個像素`));
        const pct = `${a.areaChangePct > 0 ? '+' : ''}${fmt(a.areaChangePct, 1)}%`;
        const summary = pick(lang,
          `${names[op][0]} with a ${shapeNames[shape][0]} of size ${r}: the foreground changes by ${pct}, the components go from ${st.components} to ${a.componentsOut}, ${a.smallGone} of the ${st.small.length} small pieces disappear and the holes go from ${st.holes} to ${a.holesOut}.`,
          `以${shapeNames[shape][1]}（大小 ${r}）做${names[op][1]}：前景變化 ${pct}，連通元件從 ${st.components} 個變成 ${a.componentsOut} 個，${st.small.length} 個小塊中有 ${a.smallGone} 個消失，孔洞從 ${st.holes} 個變成 ${a.holesOut} 個。`);
        return {
          summary,
          metrics: [
            { label: pick(lang, 'Foreground area change', '前景面積變化'), value: pct },
            { label: pick(lang, 'Components, before → after', '連通元件，前 → 後'), value: `${st.components} → ${a.componentsOut}` },
            { label: pick(lang, 'Small pieces gone (under 200 px)', '消失的小塊（小於 200 像素）'), value: `${a.smallGone} / ${st.small.length}` },
            { label: pick(lang, 'Holes, before → after', '孔洞，前 → 後'), value: `${st.holes} → ${a.holesOut}` },
          ],
        };
      },
    };
  },
};
