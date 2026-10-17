import type { Experiment, State } from './types.ts';
import { L, pick, fmt, svg, text, path } from './types.ts';

const bounded = (state: State, key: string, fallback: number, lo: number, hi: number) => {
  const n = Number(state[key]);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : fallback;
};
const mse = (a: number[], b: number[]) => a.reduce((sum, v, i) => sum + (v - b[i]) ** 2, 0) / a.length;
const TAU = 2 * Math.PI;

/** Point sampling followed by nearest-neighbor display and uniform scalar quantization. */
export function samplePixels(stride: number, bits: number) {
  const side = 24;
  const original = Array.from({ length: side * side }, (_, i) => {
    const x = i % side, y = Math.floor(i / side);
    return Math.max(0, Math.min(1, 0.12 + 0.56 * x / (side - 1) + 0.12 * Math.sin(TAU * y / 5) + (x > 8 && x < 16 && y > 6 && y < 18 ? 0.25 : 0)));
  });
  const levels = 2 ** bits;
  const output = original.map((_, i) => {
    const x = Math.floor((i % side) / stride) * stride;
    const y = Math.floor(Math.floor(i / side) / stride) * stride;
    return Math.round(original[y * side + x] * (levels - 1)) / (levels - 1);
  });
  return { original, output, side, levels, error: mse(original, output) };
}
const pixelGrid = (values: number[], side: number, x0: number, y0: number) => values.map((v, i) => {
  const gray = Math.round(v * 255);
  return `<rect x="${x0 + i % side * 9}" y="${y0 + Math.floor(i / side) * 9}" width="9.1" height="9.1" fill="rgb(${gray},${gray},${gray})"/>`;
}).join('');

const signalPath = (values: number[], color: string, scale: number, center = 146, height = 72, dashed = false) => path(values.map((v, i) => [38 + i * 564 / (values.length - 1), center - height * v / scale]), `stroke="var(--plot-${color})" stroke-width="2.5"${dashed ? ' stroke-dasharray="7 4"' : ''}`);
const baseline = (y: number) => `<line x1="38" y1="${y}" x2="602" y2="${y}" stroke="currentColor" opacity="0.25"/>`;

/** Fixed seeded Gaussian realization: the noise is unchanged when controls move. */
const fixedNoise = () => {
  let seed = 79421;
  const uniform = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return (seed + 0.5) / 4294967296; };
  return Array.from({ length: 48 }, () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(TAU * uniform()));
};
export function diffusionSignal(alpha: number, target: string) {
  const x0 = Array.from({ length: 48 }, (_, n) => 0.85 * Math.sin(TAU * n / 48) + 0.3 * Math.cos(TAU * 3 * n / 48));
  const noise = fixedNoise();
  const a = Math.sqrt(alpha), b = Math.sqrt(1 - alpha);
  const xt = x0.map((x, i) => a * x + b * noise[i]);
  const values = target === 'epsilon' ? noise : target === 'v' ? noise.map((e, i) => a * e - b * x0[i]) : x0;
  // At alpha=0, x_t=epsilon carries no information about x_0; epsilon alone cannot invert it.
  const reconstructed = target === 'epsilon' && alpha === 0 ? null : xt.map((x, i) => target === 'epsilon' ? (x - b * values[i]) / a : target === 'v' ? a * x - b * values[i] : values[i]);
  return { x0, noise, xt, values, reconstructed, error: reconstructed ? mse(x0, reconstructed) : null };
}

export interface Box { id: string; x: number; y: number; width: number; height: number; score: number }
export function intersectionOverUnion(a: Box, b: Box) {
  const intersection = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x)) * Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  const union = a.width * a.height + b.width * b.height - intersection;
  return union > 0 ? intersection / union : 0;
}
export function suppressBoxes(boxes: Box[], threshold: number, scoreThreshold: number) {
  const ordered = boxes.filter(b => b.score >= scoreThreshold).sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
  const kept: Box[] = [];
  const status: Record<string, 'kept' | 'suppressed' | 'filtered'> = Object.fromEntries(boxes.map(b => [b.id, 'filtered']));
  for (const box of ordered) {
    const suppress = kept.some(k => intersectionOverUnion(k, box) > threshold);
    status[box.id] = suppress ? 'suppressed' : 'kept';
    if (!suppress) kept.push(box);
  }
  return { kept, status };
}

export const visionExperiments: Experiment[] = [
  {
    id: 'pixel-sampling', track: 'foundations',
    title: L('Sampling and gray levels', '取樣與灰階'),
    description: L('Which details disappear when you keep fewer spatial samples or fewer brightness levels?', '保留較少空間取樣點或亮度階數時，哪些細節會消失？'),
    challenge: L('Set the stride to 1, then change only the bit depth. Can extra gray levels recover detail lost at stride 4?', '先將取樣間距設為 1，只改變位元深度。增加灰階數能找回間距為 4 時遺失的細節嗎？'),
    note: L('Synthetic 24 × 24 grayscale image, values in [0, 1]. Point sampling has no antialiasing filter; output uses nearest-neighbor display. Error is measured against the original grid.', '使用值域為 [0, 1] 的合成 24 × 24 灰階影像。點取樣前未使用抗混疊濾波器，輸出以最近鄰方式顯示。誤差以原始網格為基準。'),
    related: ['foundations/images-as-arrays', 'image-processing/dip-02-fundamentals'],
    controls: [
      { key: 'stride', label: L('Sampling stride', '取樣間距'), kind: 'range', value: 2, min: 1, max: 4, step: 1 },
      { key: 'bits', label: L('Bits per pixel', '每像素位元數'), kind: 'range', value: 3, min: 1, max: 8, step: 1 },
    ],
    compute(state, lang) {
      const stride = Math.round(bounded(state, 'stride', 2, 1, 4)), bits = Math.round(bounded(state, 'bits', 3, 1, 8));
      const r = samplePixels(stride, bits);
      return { svg: svg(text(52, 27, pick(lang, 'Original', '原始影像')) + text(370, 27, pick(lang, 'Sampled + quantized', '取樣與量化後')) + pixelGrid(r.original, r.side, 52, 43) + pixelGrid(r.output, r.side, 370, 43), 282),
        summary: pick(lang, `Keep one sample per ${stride} × ${stride} block and round brightness to ${r.levels} levels. Spatial detail and brightness precision are separate choices.`, `每個 ${stride} × ${stride} 區塊保留一個取樣點，亮度四捨五入至 ${r.levels} 階。空間細節與亮度精度是不同的選擇。`),
        metrics: [{ label: pick(lang, 'Available gray levels', '可用灰階數'), value: String(r.levels) }, { label: pick(lang, 'Mean squared error', '均方誤差'), value: fmt(r.error, 6) }, { label: pick(lang, 'Retained samples', '保留取樣點數'), value: String(Math.ceil(r.side / stride) ** 2) }] };
    },
  },
  {
    id: 'diffusion-targets', track: 'ml',
    title: L('Diffusion: signal, noise and prediction targets', '擴散：訊號、雜訊與預測目標'),
    description: L('Mix one clean signal with fixed noise. Compare the ε, x₀ and velocity targets at the same noise level.', '將乾淨訊號與固定雜訊混合，在相同雜訊程度下比較 ε、x₀ 與速度預測目標。'),
    challenge: L('Compare all three targets at α=0 and α=1. Why can an exact ε target fail to recover x₀ when α=0?', '在 α=0 與 α=1 時比較三個目標。為什麼 α=0 時，即使知道精確的 ε 也無法還原 x₀？'),
    note: L('xₜ=√α x₀+√(1−α) ε; v=√α ε−√(1−α) x₀. Here α is the cumulative signal-power weight. Targets use the known clean signal and a fixed seeded Gaussian noise sample: this is oracle algebra, with no trained generator or reverse sampling.', 'xₜ=√α x₀+√(1−α) ε；v=√α ε−√(1−α) x₀。此處 α 是累積訊號功率權重。目標由已知的乾淨訊號與固定種子的高斯雜訊算出；這是已知真值的代數示範，不含訓練過的生成模型或反向取樣。'),
    related: ['ml/ml-07-diffusion-flow-matching'],
    controls: [
      { key: 'alpha', label: L('Signal-power weight α', '訊號功率權重 α'), kind: 'range', value: 0.5, min: 0, max: 1, step: 0.01 },
      { key: 'target', label: L('Prediction target', '預測目標'), kind: 'select', value: 'epsilon', options: [{ value: 'epsilon', label: L('Noise ε', '雜訊 ε') }, { value: 'x0', label: L('Clean signal x₀', '乾淨訊號 x₀') }, { value: 'v', label: L('Velocity v', '速度 v') }] },
    ],
    compute(state, lang) {
      const alpha = bounded(state, 'alpha', 0.5, 0, 1), target = ['epsilon', 'x0', 'v'].includes(String(state.target)) ? String(state.target) : 'epsilon';
      const r = diffusionSignal(alpha, target), scale = Math.max(1, ...r.xt.map(Math.abs), ...r.values.map(Math.abs));
      const drawing = text(38, 24, pick(lang, 'Dashed: clean x₀ · Solid: mixed xₜ', '虛線：乾淨 x₀ · 實線：混合 xₜ')) + baseline(112) + signalPath(r.x0, 'a', scale, 112, 66, true) + signalPath(r.xt, 'b', scale, 112, 66) + text(38, 211, pick(lang, `Oracle target: ${target === 'epsilon' ? 'ε' : target === 'x0' ? 'x₀' : 'v'}`, `已知真值的目標：${target === 'epsilon' ? 'ε' : target === 'x0' ? 'x₀' : 'v'}`)) + baseline(293) + signalPath(r.values, 'c', scale, 293, 66);
      const formula = target === 'epsilon' ? 'x₀=(xₜ−√(1−α) ε)/√α' : target === 'v' ? 'x₀=√α xₜ−√(1−α) v' : 'x₀=target';
      return { svg: svg(drawing, 377), summary: r.error === null ? pick(lang, 'At α=0, xₜ equals ε. The ε inversion divides by zero, so the clean signal is not identifiable from these two values.', 'α=0 時，xₜ 等於 ε。由 ε 反推的公式會除以零，因此這兩個值無法決定乾淨訊號。') : pick(lang, `Oracle reconstruction: ${formula}. The near-zero residual checks the identity; it does not measure a learned model. Both panels use the same vertical scale.`, `已知真值時的還原公式：${formula}。接近零的殘差驗證代數恆等式，並非評估學習模型。上下圖使用相同的縱軸尺度。`), metrics: [{ label: pick(lang, 'Signal weight √α', '訊號權重 √α'), value: fmt(Math.sqrt(alpha)) }, { label: pick(lang, 'Noise weight √(1−α)', '雜訊權重 √(1−α)'), value: fmt(Math.sqrt(1 - alpha)) }, { label: pick(lang, 'Oracle reconstruction MSE', '真值還原均方誤差'), value: r.error === null ? pick(lang, 'Not identifiable', '無法唯一決定') : r.error.toExponential(1) }] };
    },
  },
  {
    id: 'detection-nms', track: 'deep-learning',
    title: L('Boxes, overlap and suppression', '邊界框、重疊與抑制'),
    description: L('When do overlapping detections represent one object? Inspect how greedy suppression makes that decision.', '重疊的偵測框何時會被當成同一個物件？觀察貪婪抑制如何作出決定。'),
    challenge: L('Move B onto A, then raise the IoU threshold. Find a setting that keeps both. Can the rule tell duplicates from two nearby objects?', '將 B 移到 A 上，再提高 IoU 門檻。找出同時保留兩框的設定。這項規則能分辨重複偵測與兩個相鄰物件嗎？'),
    note: L('Three fixed-score boxes of the same class. First keep scores ≥ the score threshold; then visit boxes by descending score and suppress any box whose intersection-over-union with a kept box is strictly greater than the IoU threshold. Scores are toy inputs, not model accuracy.', '三個同類別的框具有固定分數。先保留分數 ≥ 分數門檻的框，再依分數由高到低處理；若與已保留框的交集聯集比（IoU）嚴格大於 IoU 門檻，就抑制該框。分數是示範輸入，不代表模型準確率。'),
    related: ['deep-learning/dl-03-object-detection'],
    predict: { question: L("At the default positions, boxes A (score 0.92) and B (score 0.84) overlap with IoU about 0.55 and the IoU threshold is 0.5. How many of the three boxes survive?", "預設位置下，A 框（分數 0.92）與 B 框（分數 0.84）的 IoU 約為 0.55，IoU 門檻是 0.5。三個框會保留幾個？"), options: [{ value: "3", label: L("All three", "三個都保留") }, { value: "2", label: L("Two: A and C", "兩個：A 與 C") }, { value: "1", label: L("Only A", "只剩 A") }], answer: "2", explain: L("B overlaps the higher-scoring A by more than the threshold, so it is suppressed. C is far from both and stays. Move B away or raise the threshold to keep it.", "B 與分數較高的 A 重疊超過門檻，因此被抑制；C 離兩者都很遠，所以保留。把 B 移開或提高門檻，就能保留 B。") },
    controls: [
      { key: 'offset', label: L('Box B horizontal offset', 'B 框水平位移'), kind: 'range', value: 20, min: 0, max: 280, step: 5 },
      { key: 'iou', label: L('IoU threshold', 'IoU 門檻'), kind: 'range', value: 0.5, min: 0, max: 1, step: 0.05 },
      { key: 'score', label: L('Score threshold', '分數門檻'), kind: 'range', value: 0.5, min: 0, max: 1, step: 0.05 },
    ],
    compute(state, lang) {
      const boxes: Box[] = [{ id: 'A', x: 60, y: 58, width: 150, height: 100, score: 0.92 }, { id: 'B', x: 60 + bounded(state, 'offset', 20, 0, 280), y: 76, width: 150, height: 100, score: 0.84 }, { id: 'C', x: 420, y: 145, width: 150, height: 100, score: 0.65 }];
      const r = suppressBoxes(boxes, bounded(state, 'iou', 0.5, 0, 1), bounded(state, 'score', 0.5, 0, 1));
      let drawing = text(30, 27, pick(lang, 'Solid: kept · Dashed: suppressed · Dotted: score-filtered', '實線：保留 · 虛線：抑制 · 點線：分數不足'));
      boxes.forEach((b, i) => { const status = r.status[b.id]; drawing += `<rect x="${b.x}" y="${b.y}" width="${b.width}" height="${b.height}" fill="var(--plot-${['a', 'b', 'c'][i]})" fill-opacity="0.08" stroke="var(--plot-${['a', 'b', 'c'][i]})" stroke-width="3"${status === 'kept' ? '' : ` stroke-dasharray="${status === 'suppressed' ? '9 5' : '2 5'}"`}/>` + text(b.x + 5, b.y + 23, `${b.id}: ${b.score}`); });
      const statusText = boxes.map(b => `${b.id}: ${pick(lang, r.status[b.id], r.status[b.id] === 'kept' ? '保留' : r.status[b.id] === 'suppressed' ? '抑制' : '分數不足')}`).join(' · ');
      return { svg: svg(drawing + text(30, 286, statusText), 309), summary: pick(lang, 'IoU = intersection area / union area. A suppressed box never suppresses another box; only previously kept boxes participate.', 'IoU = 交集面積 / 聯集面積。已遭抑制的框不會再抑制其他框；比較對象僅限先前保留的框。') + ` ${statusText}.`, metrics: [{ label: 'IoU(A, B)', value: fmt(intersectionOverUnion(boxes[0], boxes[1])) }, { label: 'IoU(B, C)', value: fmt(intersectionOverUnion(boxes[1], boxes[2])) }, { label: pick(lang, 'Kept boxes', '保留框數'), value: String(r.kept.length) }] };
    },
  },
];
