import type { Experiment, State } from './types.ts';
import { L, pick, fmt, svg, text } from './types.ts';

const bounded = (state: State, key: string, fallback: number, lo: number, hi: number) => {
  const n = Number(state[key]);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : fallback;
};

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
