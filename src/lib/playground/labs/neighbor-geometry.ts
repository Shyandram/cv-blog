import type { LabDef } from './types.ts';
import { L, pick, fmt, svg, text, path, num, type Localized } from '../types.ts';
import irisData from './data/iris.json' with { type: 'json' };

/** Real data: Fisher's iris measurements as bundled with scikit-learn (150 flowers, 3 species, 4 lengths in cm). */
export const N: number = irisData.y.length;
export const LABELS: number[] = irisData.y;
export const CLASS_NAMES: string[] = irisData.classes;
/** The source rounds every measurement to 0.1 cm. Internally we keep integer tenths of a centimetre, so that distances
 *  between samples are exact integers and ties between equal distances are real ties, not floating-point accidents. */
const TENTHS: number[][] = irisData.x.map(row => row.map(v => Math.round(v * 10)));

export type PairKey = 'width' | 'sepal' | 'petal';
export const PAIRS: Record<PairKey, { cols: [number, number]; name: Localized; names: [Localized, Localized] }> = {
  width: { cols: [1, 3], name: L('sepal width and petal width', '花萼寬度與花瓣寬度'), names: [L('sepal width', '花萼寬度'), L('petal width', '花瓣寬度')] },
  sepal: { cols: [0, 1], name: L('sepal length and sepal width', '花萼長度與花萼寬度'), names: [L('sepal length', '花萼長度'), L('sepal width', '花萼寬度')] },
  petal: { cols: [2, 3], name: L('petal length and petal width', '花瓣長度與花瓣寬度'), names: [L('petal length', '花瓣長度'), L('petal width', '花瓣寬度')] },
};
export function pairColumns(pair: PairKey): { a: number[]; b: number[] } {
  const [ca, cb] = PAIRS[pair].cols;
  return { a: TENTHS.map(r => r[ca]), b: TENTHS.map(r => r[cb]) };
}

/** Scale applied to the SECOND feature (the first stays in centimetres), as the fraction p/q. 'z' standardizes both. */
export const SCALES: Record<string, { p: number; q: number; factor: string; unit: Localized }> = {
  m: { p: 1, q: 100, factor: '×0.01', unit: L('metres', '公尺') },
  dm: { p: 1, q: 10, factor: '×0.1', unit: L('decimetres', '公寸') },
  cm: { p: 1, q: 1, factor: '×1', unit: L('centimetres, as measured', '公分（原始量測）') },
  mm: { p: 10, q: 1, factor: '×10', unit: L('millimetres', '公釐') },
  tenth: { p: 100, q: 1, factor: '×100', unit: L('tenths of a millimetre', '0.1 公釐') },
};
export const SCALE_KEYS = ['m', 'dm', 'cm', 'mm', 'tenth', 'z'] as const;
export const REFERENCE_SCALE = 'cm';

/** Squared distance is wa·Δa² + wb·Δb² with Δ in tenths of a centimetre. Scaling the second feature by p/q gives (q², p²), all integers. */
export interface Metric { wa: number; wb: number }
export function metricFor(pair: PairKey, scale: string): Metric {
  if (scale === 'z') {
    const { a, b } = pairColumns(pair), v = (x: number[]) => { const m = x.reduce((s, t) => s + t, 0) / x.length; return x.reduce((s, t) => s + (t - m) ** 2, 0) / x.length; };
    return { wa: 1 / v(a), wb: 1 / v(b) };
  }
  const s = SCALES[scale];
  return { wa: s.q * s.q, wb: s.p * s.p };
}
export const scaleOf = (m: Metric) => Math.sqrt(m.wb / m.wa);

/** Fixed random order used only to break exact ties between equal distances (the file lists the species one after another). */
export const TIE_RANK: number[] = irisData.tieRank;
/**
 * The k training samples nearest to the query (qa, qb), in tenths of a centimetre, nearest first. Order is (distance,
 * tie rank): of two samples at exactly the same distance, the one with the smaller rank comes first. `skip` leaves one
 * sample out (leave-one-out).
 */
export function nearest(a: ArrayLike<number>, b: ArrayLike<number>, m: Metric, qa: number, qb: number, k: number, skip = -1, rank: ArrayLike<number> = TIE_RANK): number[] {
  return ranked(a, b, m, qa, qb, k, skip, rank).idx;
}
/** Same as `nearest`, with the squared distances of the returned samples. */
export function ranked(a: ArrayLike<number>, b: ArrayLike<number>, m: Metric, qa: number, qb: number, k: number, skip = -1, rank: ArrayLike<number> = TIE_RANK): { idx: number[]; dist: number[] } {
  const idx: number[] = [], dist: number[] = [];
  for (let j = 0; j < a.length; j++) {
    if (j === skip) continue;
    const da = a[j] - qa, db = b[j] - qb, d = m.wa * da * da + m.wb * db * db;
    const before = (p: number) => dist[p] < d || (dist[p] === d && rank[idx[p]] < rank[j]); // is the sample stored at p ahead of j?
    if (idx.length === k && before(k - 1)) continue;
    let p = idx.length < k ? idx.length : k - 1;
    while (p > 0 && !before(p - 1)) p--;
    idx.splice(p, 0, j); dist.splice(p, 0, d);
    if (idx.length > k) { idx.pop(); dist.pop(); }
  }
  return { idx, dist };
}
/** Votes per class among the neighbours, and whether the winner is ambiguous (two classes tie). */
export function voteTied(labels: ArrayLike<number>, idx: number[]): boolean {
  const votes = [0, 0, 0];
  for (const j of idx) votes[labels[j]]++;
  const top = Math.max(...votes);
  return votes.filter(v => v === top).length > 1;
}
/**
 * Majority vote among the neighbours (given nearest first). A tie between classes goes to the tied class whose member is
 * nearest, so the answer never depends on how the classes are numbered.
 */
export function majority(labels: ArrayLike<number>, idx: number[]): number {
  const votes = [0, 0, 0];
  for (const j of idx) votes[labels[j]]++;
  const top = Math.max(...votes);
  for (const j of idx) if (votes[labels[j]] === top) return labels[j];
  return 0;
}
export interface Loo { pred: number[]; nn: number[]; loose: number[]; correct: number; accuracy: number; tieDependent: number }
/**
 * Leave-one-out k-NN: every flower is classified by the other 149. `nn[i]` is its single nearest other flower;
 * `loose[i]` is 1 when the answer rests on a tie (the k-th and (k+1)-th nearest distances are equal, or two classes tie in the vote).
 */
export function leaveOneOut(pair: PairKey, m: Metric, k: number): Loo {
  const { a, b } = pairColumns(pair), pred: number[] = [], nn: number[] = [], loose: number[] = [];
  let correct = 0, tieDependent = 0;
  for (let i = 0; i < N; i++) {
    const r = ranked(a, b, m, a[i], b[i], k + 1, i), nb = r.idx.slice(0, k);
    pred.push(majority(LABELS, nb)); nn.push(r.idx[0]);
    const tied = r.dist[k] === r.dist[k - 1] || voteTied(LABELS, nb);
    loose.push(tied ? 1 : 0); if (tied) tieDependent++;
    if (pred[i] === LABELS[i]) correct++;
  }
  return { pred, nn, loose, correct, accuracy: correct / N, tieDependent };
}
/** How many of the 150 flowers have a different nearest neighbour under `m` than under `ref`. */
export function nearestChanged(pair: PairKey, m: Metric, ref: Metric): number {
  const { a, b } = pairColumns(pair);
  let changed = 0;
  for (let i = 0; i < N; i++) if (nearest(a, b, m, a[i], b[i], 1, i)[0] !== nearest(a, b, ref, a[i], b[i], 1, i)[0]) changed++;
  return changed;
}
export interface Box { a0: number; a1: number; b0: number; b1: number }
/** Plot window in tenths of a centimetre: the data range padded by 3 tenths on every side. */
export function boxFor(pair: PairKey): Box {
  const { a, b } = pairColumns(pair);
  return { a0: Math.min(...a) - 3, a1: Math.max(...a) + 3, b0: Math.min(...b) - 3, b1: Math.max(...b) + 3 };
}
/** k-NN decision map: the class of every cell centre of a res × res grid over `box` (row 0 is the top, i.e. the largest second feature). */
export function decisionMap(pair: PairKey, m: Metric, k: number, box: Box, res = 64): Uint8Array {
  const { a, b } = pairColumns(pair), out = new Uint8Array(res * res);
  for (let gy = 0; gy < res; gy++) for (let gx = 0; gx < res; gx++) {
    const qa = box.a0 + (gx + 0.5) * (box.a1 - box.a0) / res, qb = box.b1 - (gy + 0.5) * (box.b1 - box.b0) / res;
    out[gy * res + gx] = majority(LABELS, nearest(a, b, m, qa, qb, k));
  }
  return out;
}
/** Second-feature multipliers of the accuracy curve (powers of ten and their ×3 neighbours), as fractions p/q. */
export const CURVE: { p: number; q: number }[] = [[1, 100], [3, 100], [1, 10], [3, 10], [1, 1], [3, 1], [10, 1], [30, 1], [100, 1]].map(([p, q]) => ({ p, q }));
export function accuracyCurve(pair: PairKey, k: number): { s: number; accuracy: number; correct: number }[] {
  return CURVE.map(({ p, q }) => { const r = leaveOneOut(pair, { wa: q * q, wb: p * p }, k); return { s: p / q, accuracy: r.accuracy, correct: r.correct }; });
}

const RGB: [number, number, number][] = [[0, 114, 178], [230, 159, 0], [0, 158, 115]];
const tint = (c: [number, number, number], w = 0.72) => c.map(v => Math.round(v * (1 - w) + 255 * w));
const GRID = 64, PLOT = 320, LEFT = 56, TOP = 12;
const f3 = (v: number) => v.toFixed(3);

export const lab: LabDef = {
  kind: 'image', id: 'neighbor-geometry', track: 'pattern-recognition',
  title: L('PR 04 · Does changing a feature’s unit change its neighbours?', 'PR 04・改變特徵的單位，鄰居會跟著改變嗎？'),
  description: L('Run leave-one-out k-nearest-neighbour classification on the real iris measurements, then change the unit of one feature: no flower moves, yet the neighbours and the accuracy do.', '對真實的鳶尾花量測資料做留一法 k 近鄰分類，再改變其中一個特徵的單位：沒有任何一朵花移動，鄰居與準確率卻改變了。'),
  challenge: L('Start with sepal width and petal width, 5 neighbours, petal width in decimetres (×0.1). Step the unit to centimetres (×1) and millimetres (×10): how do the accuracy and the number of changed nearest neighbours move? Then switch to petal length and petal width. Is the effect as large? Finally compare with the standardized setting.', '從花萼寬度與花瓣寬度、5 個鄰居、花瓣寬度以公寸（×0.1）為單位開始。把單位依序改成公分（×1）與公釐（×10）：準確率與最近鄰改變的數量如何變化？接著換成花瓣長度與花瓣寬度，影響一樣大嗎？最後與標準化的設定比較。'),
  note: L('Data: the 150 iris flowers bundled with scikit-learn (three species, 50 each), two of the four measurements at a time, in centimetres. The classifier is the plain k-NN rule with Euclidean distance. Only the second feature is rescaled (the first stays in centimetres); "standardized" divides both features by their standard deviation. Accuracy is leave-one-out: each flower is classified by the other 149, and the map is trained on all 150. The measurements are rounded to 0.1 cm, so many distances tie exactly, and the fifth and sixth neighbour of a flower are often equally far away. Ties are settled in a fixed way: equal distances are ordered by a fixed random ranking of the flowers (the file lists the species one after another, so ordering by position would favour setosa), and a tie between classes goes to the tied class with the nearest member. The last metric counts the answers that rest on such a tie. scikit-learn orders equal distances in its own way: on the 108 settings checked in the tests (3 feature pairs, 6 units, 6 values of k) it agrees with this lab on every answer that does not rest on a tie, and its leave-one-out count differs from this one by at most 7 flowers. Rescaling is reversible, so it adds no information; the accuracy is a single 150-flower sample, not a general result.', '資料：scikit-learn 內建的 150 朵鳶尾花（三個品種，各 50 朵），每次取四項量測中的兩項，單位為公分。分類器是使用歐氏距離的標準 k 近鄰規則。只有第二個特徵會被縮放（第一個維持公分）；「標準化」則把兩個特徵都除以各自的標準差。準確率採留一法：每朵花由其餘 149 朵分類，決策圖則使用全部 150 朵訓練。量測值四捨五入到 0.1 公分，所以許多距離會剛好相等，一朵花的第五與第六個鄰居常常一樣遠。平手以固定的方式處理：距離相同時依一個固定的隨機排名決定先後（資料檔依品種連續排列，若依位置排序會偏袒 setosa）；類別得票相同時選擇「最近成員所屬」的那個類別。最後一項指標計算有多少答案取決於這類平手。scikit-learn 對距離相同的樣本有自己的排序方式：在測試檢查的 108 種設定（3 組特徵、6 種單位、6 個 k 值）中，凡是不取決於平手的答案，它與這裡完全一致，而兩者的留一法答對數最多相差 7 朵。縮放可以還原，所以不會增加任何資訊；準確率只是 150 朵花的單一樣本，不是一般性的結論。'),
  related: ['pattern-recognition/pr-04-kernels-nonlinear-classifiers', 'image-processing/dip-13-pattern-classification'],
  controls: [
    { key: 'pair', label: L('Two features', '兩個特徵'), kind: 'select', value: 'width', options: (Object.keys(PAIRS) as PairKey[]).map(key => ({ value: key, label: PAIRS[key].name })) },
    { key: 'scale', label: L('Unit of the feature on the vertical axis', '縱軸那個特徵的單位'), kind: 'select', value: 'dm', options: [
      ...(['m', 'dm', 'cm', 'mm', 'tenth'] as const).map(key => ({ value: key, label: L(`${SCALES[key].factor}: ${SCALES[key].unit.en}`, `${SCALES[key].factor}：${SCALES[key].unit['zh-tw']}`) })),
      { value: 'z', label: L('Standardized (both features)', '標準化（兩個特徵）') }] },
    { key: 'k', label: L('Neighbours k', '鄰居數 k'), kind: 'range', value: 5, min: 1, max: 25, step: 2 },
  ],
  predict: {
    question: L('Every number in one feature column is multiplied by 10, for example decimetres to centimetres or centimetres to millimetres. The flowers do not move and the column can be divided back by 10. What happens to a k-nearest-neighbour classifier?', '某一個特徵欄的每個數字都乘以 10，例如公寸改成公分、或公分改成公釐。花朵沒有移動，而且除以 10 就能還原。k 近鄰分類器會發生什麼事？'),
    options: [
      { value: 'more', label: L('It gets more information (bigger numbers), so accuracy can only go up', '它得到更多資訊（數字更大），所以準確率只會上升') },
      { value: 'neighbours', label: L('It gets no new information, but different neighbours, so accuracy can go up or down', '它沒有得到新資訊，但鄰居不同，所以準確率可能上升也可能下降') },
      { value: 'same', label: L('Nothing changes: a unit is only a label', '什麼都不變：單位只是標籤') },
    ],
    answer: 'neighbours',
    explain: L('Multiplying a column by 10 can be undone by dividing by 10, so the data carry exactly the same information, and multiplying both columns by 10 changes no answer at all. The distance, however, adds squared differences, so the column with the larger numbers gets a larger say and the nearest neighbours change. With 5 neighbours, sepal width with petal width scores 0.860 (129 of 150) when petal width is in decimetres and 0.953 (143) in centimetres: one factor of 10 helps a lot. Sepal length with sepal width scores 0.747 (112) in centimetres but 0.647 (97) with sepal width in tenths of a millimetre, a factor of 100: it hurts. Which way it moves depends on which feature carries the information, not on the unit.', '把一欄乘以 10，只要再除以 10 就能還原，所以資料所含的資訊完全相同；而且兩欄同時乘以 10，任何答案都不會改變。但距離是把差的平方相加，數字較大的那一欄話語權變大，最近鄰因此改變。取 5 個鄰居時，花萼寬度搭配花瓣寬度：花瓣寬度用公寸，準確率是 0.860（150 朵中 129 朵）；用公分是 0.953（143 朵）：乘以 10 一次幫助很大。花萼長度搭配花萼寬度：用公分是 0.747（112 朵），把花萼寬度改用 0.1 公釐（相差 100 倍）則是 0.647（97 朵）：傷害很大。往哪個方向走，取決於哪個特徵帶有資訊，而不是取決於單位。'),
  },
  assets: {},
  paneMin: 400,
  panes: [
    { key: 'map', kind: 'canvas', label: L('k-NN decision map on the iris flowers (axes in the original centimetres; ring = misclassified in leave-one-out)', '鳶尾花的 k 近鄰決策圖（座標軸為原始公分；圈起來的是留一法分錯的花）'), width: 400, height: 400 },
    { key: 'curve', kind: 'svg', label: L('Leave-one-out accuracy as the unit of the second feature changes (the data are the same in every column)', '第二個特徵的單位改變時的留一法準確率（每一欄用的資料都相同）'), width: 640, height: 300 },
  ],
  sources: [
    { title: 'R. A. Fisher, The Use of Multiple Measurements in Taxonomic Problems, Annals of Eugenics, 7(2), 179-188, 1936 (the iris measurements)', url: 'https://doi.org/10.1111/j.1469-1809.1936.tb02137.x' },
    { title: 'R. A. Fisher, Iris, UCI Machine Learning Repository, 1936, doi:10.24432/C56C76 (donated 1988)', url: 'https://archive.ics.uci.edu/dataset/53/iris' },
    { title: 'scikit-learn documentation, sklearn.datasets.load_iris (bundled copy of the data used here, no download)', url: 'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html' },
    { title: 'T. Cover and P. Hart, Nearest Neighbor Pattern Classification, IEEE Transactions on Information Theory, 13(1), 21-27, 1967 (the k-NN rule)', url: 'https://doi.org/10.1109/TIT.1967.1053964' },
    { title: 'M. Stone, Cross-Validatory Choice and Assessment of Statistical Predictions, Journal of the Royal Statistical Society B, 36(2), 111-133, 1974 (leave-one-out)', url: 'https://doi.org/10.1111/j.2517-6161.1974.tb00994.x' },
  ],
  create({ lang }) {
    return {
      render(state, canvases) {
        const pair = (state.pair in PAIRS ? state.pair : 'width') as PairKey, scale = String(state.scale), k = Math.max(1, Math.round(num(state, 'k')));
        const m = metricFor(pair, scale), ref = metricFor(pair, REFERENCE_SCALE);
        const { a, b } = pairColumns(pair), box = boxFor(pair);
        const loo = leaveOneOut(pair, m, k), looRef = scale === REFERENCE_SCALE ? loo : leaveOneOut(pair, ref, k);
        const changed = nearestChanged(pair, m, ref), grid = decisionMap(pair, m, k, box, GRID);
        draw(canvases.map, lang, pair, grid, a, b, box, loo.pred);
        const curve = accuracyCurve(pair, k), z = leaveOneOut(pair, metricFor(pair, 'z'), k);
        const second = pick(lang, PAIRS[pair].names[1].en, PAIRS[pair].names[1]['zh-tw']);
        const unit = scale === 'z' ? pick(lang, 'standardized units (both features)', '標準化單位（兩個特徵）') : pick(lang, `${SCALES[scale].unit.en} (${SCALES[scale].factor})`, `${SCALES[scale].unit['zh-tw']}（${SCALES[scale].factor}）`);
        const delta = Math.round(loo.correct - looRef.correct);
        const summary = scale === REFERENCE_SCALE
          ? pick(lang, `With ${second} in centimetres, as measured, ${k}-NN gets ${loo.correct} of ${N} flowers right in leave-one-out (${f3(loo.accuracy)}). Change the unit of ${second} to see the same flowers get different neighbours.`, `${second}使用原始公分時，${k} 近鄰在留一法中答對 ${loo.correct} / ${N} 朵（${f3(loo.accuracy)}）。改變${second}的單位，就會看到同樣的花得到不同的鄰居。`)
          : pick(lang, `With ${second} in ${unit}, ${k}-NN gets ${loo.correct} of ${N} right (${f3(loo.accuracy)}), ${delta === 0 ? 'the same count as' : `${Math.abs(delta)} ${delta > 0 ? 'more' : 'fewer'} than`} in centimetres (${f3(looRef.accuracy)}). ${changed} flowers have a different nearest neighbour. No flower moved and no information was added: only the ruler for ${second} changed.`, `${second}使用${unit}時，${k} 近鄰答對 ${loo.correct} / ${N} 朵（${f3(loo.accuracy)}），${delta === 0 ? '與公分時（' + f3(looRef.accuracy) + '）相同' : `比公分時（${f3(looRef.accuracy)}）${delta > 0 ? '多' : '少'} ${Math.abs(delta)} 朵`}。有 ${changed} 朵花的最近鄰換了人。沒有任何一朵花移動，也沒有增加資訊：只是${second}的尺換了。`);
        return {
          summary,
          metrics: [
            { label: pick(lang, 'Leave-one-out accuracy', '留一法準確率'), value: `${f3(loo.accuracy)} (${loo.correct}/${N})` },
            { label: pick(lang, 'Same, in centimetres (as measured)', '同設定，原始公分'), value: `${f3(looRef.accuracy)} (${looRef.correct}/${N})` },
            { label: pick(lang, 'Nearest neighbour changed', '最近鄰改變的花'), value: `${changed} / ${N}` },
            { label: pick(lang, 'Answers resting on a tie', '答案取決於平手的花'), value: `${loo.tieDependent} / ${N}` },
          ],
          plots: { curve: svg(curvePlot(lang, curve, z.accuracy, scale, loo.accuracy, k), 300) },
        };
      },
    };
  },
};

function draw(canvas: HTMLCanvasElement, lang: 'en' | 'zh-tw', pair: PairKey, grid: Uint8Array, a: number[], b: number[], box: Box, errors: number[]) {
  const W = canvas.width, H = canvas.height, ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, W, H);
  const cell = document.createElement('canvas'); cell.width = GRID; cell.height = GRID;
  const cc = cell.getContext('2d')!, im = cc.createImageData(GRID, GRID);
  for (let i = 0; i < GRID * GRID; i++) { const c = tint(RGB[grid[i]]); im.data[4 * i] = c[0]; im.data[4 * i + 1] = c[1]; im.data[4 * i + 2] = c[2]; im.data[4 * i + 3] = 255; }
  cc.putImageData(im, 0, 0);
  ctx.imageSmoothingEnabled = false; ctx.drawImage(cell, LEFT, TOP, PLOT, PLOT);
  ctx.strokeStyle = '#555555'; ctx.lineWidth = 1; ctx.strokeRect(LEFT + 0.5, TOP + 0.5, PLOT, PLOT);
  const px = (v: number) => LEFT + (v - box.a0) / (box.a1 - box.a0) * PLOT, py = (v: number) => TOP + (box.b1 - v) / (box.b1 - box.b0) * PLOT;
  ctx.fillStyle = '#222222'; ctx.font = '13px sans-serif';
  // Ticks every 1 cm (10 tenths) on x, and at a spacing that keeps about 5 labels on y.
  ctx.textAlign = 'center';
  for (let t = Math.ceil(box.a0 / 10) * 10; t <= box.a1; t += 10) { ctx.fillText(String(t / 10), px(t), TOP + PLOT + 17); ctx.fillRect(px(t), TOP + PLOT, 1, 4); }
  ctx.textAlign = 'right';
  const yStep = (box.b1 - box.b0) > 40 ? 10 : 5;
  for (let t = Math.ceil(box.b0 / yStep) * yStep; t <= box.b1; t += yStep) { ctx.fillText(String(t / 10), LEFT - 8, py(t) + 4); ctx.fillRect(LEFT - 4, py(t), 4, 1); }
  ctx.textAlign = 'center';
  ctx.fillText(`${pick(lang, PAIRS[pair].names[0].en, PAIRS[pair].names[0]['zh-tw'])} (cm)`, LEFT + PLOT / 2, TOP + PLOT + 37);
  ctx.save(); ctx.translate(14, TOP + PLOT / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText(`${pick(lang, PAIRS[pair].names[1].en, PAIRS[pair].names[1]['zh-tw'])} (cm)`, 0, 0); ctx.restore();
  // Samples: colour and shape (circle, square, triangle) so the classes do not rely on colour alone; a black ring marks a leave-one-out error.
  for (let i = 0; i < N; i++) {
    const x = px(a[i]), y = py(b[i]), c = RGB[LABELS[i]], r = 4.2;
    ctx.beginPath();
    if (LABELS[i] === 0) ctx.arc(x, y, r, 0, 2 * Math.PI);
    else if (LABELS[i] === 1) ctx.rect(x - r, y - r, 2 * r, 2 * r);
    else { ctx.moveTo(x, y - r - 1); ctx.lineTo(x + r + 1, y + r); ctx.lineTo(x - r - 1, y + r); ctx.closePath(); }
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.globalAlpha = 0.9; ctx.fill(); ctx.globalAlpha = 1;
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1; ctx.stroke();
    if (errors[i] !== LABELS[i]) { ctx.beginPath(); ctx.arc(x, y, r + 4, 0, 2 * Math.PI); ctx.strokeStyle = '#111111'; ctx.lineWidth = 1.6; ctx.stroke(); }
  }
  // Legend.
  ctx.textAlign = 'left'; ctx.fillStyle = '#222222';
  const names = CLASS_NAMES;
  let lx = 14;
  for (let c = 0; c < 3; c++) {
    const col = RGB[c], y = H - 12;
    ctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`; ctx.beginPath();
    if (c === 0) ctx.arc(lx + 5, y - 4, 4.5, 0, 2 * Math.PI); else if (c === 1) ctx.rect(lx + 0.5, y - 8.5, 9, 9); else { ctx.moveTo(lx + 5, y - 9); ctx.lineTo(lx + 10.5, y); ctx.lineTo(lx - 0.5, y); ctx.closePath(); }
    ctx.fill(); ctx.fillStyle = '#222222'; ctx.fillText(names[c], lx + 15, y);
    lx += 15 + ctx.measureText(names[c]).width + 14;
  }
  ctx.beginPath(); ctx.arc(lx + 5, H - 16, 7, 0, 2 * Math.PI); ctx.strokeStyle = '#111111'; ctx.lineWidth = 1.6; ctx.stroke();
  ctx.fillStyle = '#222222'; ctx.fillText(pick(lang, 'error', '分錯'), lx + 17, H - 12);
}

function curvePlot(lang: 'en' | 'zh-tw', curve: { s: number; accuracy: number; correct: number }[], zAcc: number, scale: string, acc: number, k: number): string {
  const x0 = 64, x1 = 612, y0 = 30, y1 = 214, lo = 0.5, hi = 1;
  const gx = (s: number) => x0 + (Math.log10(s) + 2) / 4 * (x1 - x0), gy = (v: number) => y1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo) * (y1 - y0);
  let out = path([[x0, y0], [x0, y1], [x1, y1]], 'stroke="var(--plot-grid)"');
  for (const v of [0.5, 0.6, 0.7, 0.8, 0.9, 1]) out += path([[x0, gy(v)], [x1, gy(v)]], 'stroke="var(--plot-grid)" stroke-dasharray="2 4"') + text(x0 - 8, gy(v) + 4, fmt(v, 1), 'text-anchor="end"');
  out += path(curve.map(c => [gx(c.s), gy(c.accuracy)] as [number, number]), 'stroke="var(--plot-a)" stroke-width="2.5"');
  for (const c of curve) out += `<circle cx="${gx(c.s)}" cy="${gy(c.accuracy)}" r="3.5" fill="var(--plot-a)"/>`;
  out += path([[x0, gy(zAcc)], [x1, gy(zAcc)]], 'stroke="var(--plot-c)" stroke-width="2" stroke-dasharray="7 4"')
    + text(x0 + 8, gy(zAcc) - 7, pick(lang, `standardized: ${zAcc.toFixed(3)}`, `標準化：${zAcc.toFixed(3)}`), 'fill="var(--plot-c)"');
  const sel = scale === 'z' ? null : SCALES[scale].p / SCALES[scale].q;
  if (sel !== null) out += `<circle cx="${gx(sel)}" cy="${gy(acc)}" r="8" fill="none" stroke="var(--plot-b)" stroke-width="3"/>` + text(gx(sel), gy(acc) + (acc > 0.9 ? 26 : -14), acc.toFixed(3), 'text-anchor="middle" fill="var(--plot-b)"');
  const ticks: [number, string][] = [[0.01, '×0.01'], [0.1, '×0.1'], [1, '×1'], [10, '×10'], [100, '×100']];
  for (const [s, label] of ticks) out += path([[gx(s), y1], [gx(s), y1 + 5]], 'stroke="var(--plot-muted)"') + text(gx(s), y1 + 21, label, 'text-anchor="middle"');
  out += text(x0 + (x1 - x0) / 2, y1 + 44, pick(lang, 'Multiplier on the second feature (cm → m, dm, cm, mm, 0.1 mm)', '第二個特徵的倍率（公分 → 公尺、公寸、公分、公釐、0.1 公釐）'), 'text-anchor="middle"')
    + text(x0, 18, pick(lang, `Leave-one-out accuracy, ${k}-NN (ring = current setting)`, `留一法準確率，${k} 近鄰（圈起來的是目前設定）`));
  return out;
}
