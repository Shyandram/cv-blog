import type { LabDef } from './types.ts';
import { L, pick, num } from '../types.ts';
import { plotColors, startPlot, label, line } from './plot2d.ts';
import {
  BOOTSTRAP_SEED, bootstrapSamples, fitBagging, fitTree, leafCount, memberPredictions, predictTree, voteCounts, voteError, errorOf, ensembleCurve, disagreement,
  type Tree, type Curve,
} from './trees.ts';
import moons from './data/moons.json' with { type: 'json' };
import type { Lang } from '../../../i18n/ui';

/** Real toy data: sklearn.datasets.make_moons (noise 0.3). 300 training points and 1000 held-out test points from a different seed. */
export const TRAIN_X: number[][] = moons.train.x;
export const TRAIN_Y: number[] = moons.train.y;
export const TEST_X: number[][] = moons.test.x;
export const TEST_Y: number[] = moons.test.y;
export const N = TRAIN_Y.length;
export const MAX_M = 100;
/** 14 grows every tree of this lab (single tree and all 100 bootstrap trees) until its leaves are pure. */
export const MAX_DEPTH = 14;
export const DEFAULT = { m: 50, depth: MAX_DEPTH, view: 'bagged' };
/** Map window and resolution (cell centres, row 0 on top). */
export const BOX = { a0: -2, a1: 3, b0: -1.5, b1: 2 };
export const GRID_W = 80, GRID_H = 56;
export const SEED = BOOTSTRAP_SEED;
/**
 * Error of the best possible classifier on the distribution make_moons samples (angles uniform along each half circle, Gaussian
 * noise 0.3), found by numerical integration of min(p0, p1) / 2; the test file repeats the integration.
 */
export const BAYES_ERROR = 0.0854;

/** Bootstrap samples of all MAX_M members: the same for every depth and every M (the first M members form the M-tree ensemble). */
export const SAMPLES: number[][] = bootstrapSamples(N, MAX_M);

export interface Run {
  depth: number;
  /** One tree per bootstrap sample. */
  trees: Tree[];
  /** The tree grown on all N training points. */
  single: Tree;
  /** Class predictions of every member on the test points. */
  testPreds: Uint8Array[];
  singleError: number;
  /** Test error of the vote, mean member test error and mean pairwise disagreement for M = 1..MAX_M (index M - 1). */
  curve: Curve;
}
/** Grow the single tree and all bootstrap trees at one depth limit and score them on the test set. */
export function runFor(depth: number): Run {
  const trees = fitBagging(TRAIN_X, TRAIN_Y, SAMPLES, depth);
  const single = fitTree(TRAIN_X, TRAIN_Y, Array.from({ length: N }, (_, i) => i), depth);
  const testPreds = memberPredictions(trees, TEST_X);
  const singleError = errorOf(memberPredictions([single], TEST_X)[0], TEST_Y);
  return { depth, trees, single, testPreds, singleError, curve: ensembleCurve(testPreds, TEST_Y) };
}
/** 1 for the training rows that a bootstrap sample does not contain (about 37% of them). */
export function outOfBag(sample: number[]): Uint8Array {
  const seen = new Uint8Array(N);
  for (const i of sample) seen[i] = 1;
  return seen.map(v => 1 - v) as Uint8Array;
}
/** Test points that more than half of the first m members classify wrongly (they stay wrong in the vote, however many trees are added). */
export function majorityWrong(preds: Uint8Array[], m: number): number {
  const v = voteCounts(preds, m);
  let n = 0;
  for (let j = 0; j < TEST_Y.length; j++) if (2 * (TEST_Y[j] === 1 ? m - v[j] : v[j]) > m) n++;
  return n;
}

/** Cell centre coordinates. */
export const cellX = (gx: number) => BOX.a0 + (gx + 0.5) * (BOX.a1 - BOX.a0) / GRID_W;
export const cellY = (gy: number) => BOX.b1 - (gy + 0.5) * (BOX.b1 - BOX.b0) / GRID_H;
/** Number of the given trees voting class 1 in every cell of the map. */
export function mapVotes(trees: Tree[]): Uint16Array {
  const out = new Uint16Array(GRID_W * GRID_H);
  for (const t of trees) for (let gy = 0; gy < GRID_H; gy++) for (let gx = 0; gx < GRID_W; gx++) out[gy * GRID_W + gx] += predictTree(t, cellX(gx), cellY(gy));
  return out;
}
export { voteCounts, voteError, disagreement, leafCount };

type View = 'bagged' | 'single' | 'member';
const VIEWS: View[] = ['bagged', 'single', 'member'];
const pct = (v: number) => `${(v * 100).toFixed(2)}%`;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.round(v)));

const RGB: [number, number, number][] = [[0, 114, 178], [230, 159, 0]];
/** Cell colour for a fraction f of votes for class 1: white at a tie, the class colour when unanimous. */
export function shade(f: number): [number, number, number] {
  const s = Math.abs(2 * f - 1), c = RGB[f > 0.5 ? 1 : 0], w = 0.3 * Math.sqrt(s);
  return [Math.round(255 + (c[0] - 255) * w), Math.round(255 + (c[1] - 255) * w), Math.round(255 + (c[2] - 255) * w)];
}

const MAP_W = 440, MAP_H = 316, PX = 36, PY = 8, PW = 400, PH = 280;
function drawMap(canvas: HTMLCanvasElement, lang: Lang, votes: Uint16Array, count: number, hollow: Uint8Array | null) {
  const ctx = canvas.getContext('2d')!;
  if (canvas.width !== MAP_W) canvas.width = MAP_W;
  if (canvas.height !== MAP_H) canvas.height = MAP_H;
  ctx.setLineDash([]); ctx.globalAlpha = 1; ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, MAP_W, MAP_H);
  const cell = document.createElement('canvas'); cell.width = GRID_W; cell.height = GRID_H;
  const cc = cell.getContext('2d')!, im = cc.createImageData(GRID_W, GRID_H);
  for (let i = 0; i < GRID_W * GRID_H; i++) { const c = shade(votes[i] / count); im.data[4 * i] = c[0]; im.data[4 * i + 1] = c[1]; im.data[4 * i + 2] = c[2]; im.data[4 * i + 3] = 255; }
  cc.putImageData(im, 0, 0);
  ctx.imageSmoothingEnabled = false; ctx.drawImage(cell, PX, PY, PW, PH);
  ctx.strokeStyle = '#555555'; ctx.lineWidth = 1; ctx.strokeRect(PX + 0.5, PY + 0.5, PW, PH);
  const px = (v: number) => PX + (v - BOX.a0) / (BOX.a1 - BOX.a0) * PW, py = (v: number) => PY + (BOX.b1 - v) / (BOX.b1 - BOX.b0) * PH;
  ctx.fillStyle = '#222222'; ctx.font = '14px system-ui, sans-serif'; ctx.textAlign = 'center';
  for (let t = BOX.a0; t <= BOX.a1; t++) { ctx.fillText(String(t), px(t), PY + PH + 18); ctx.fillRect(px(t), PY + PH, 1, 4); }
  ctx.textAlign = 'right';
  for (let t = Math.ceil(BOX.b0); t <= BOX.b1; t++) { ctx.fillText(String(t), PX - 7, py(t) + 5); ctx.fillRect(PX - 4, py(t), 4, 1); }
  // Training points: circle = class 0, square = class 1 (shape as well as colour); hollow = not in this member's bootstrap sample.
  for (let i = 0; i < N; i++) {
    const x = px(TRAIN_X[i][0]), y = py(TRAIN_X[i][1]), c = RGB[TRAIN_Y[i]], r = 3.9, open = hollow !== null && hollow[i] === 1;
    ctx.beginPath();
    if (TRAIN_Y[i] === 0) ctx.arc(x, y, r, 0, 2 * Math.PI); else ctx.rect(x - r, y - r, 2 * r, 2 * r);
    const col = `rgb(${c[0]},${c[1]},${c[2]})`;
    if (open) { ctx.fillStyle = '#ffffff'; ctx.fill(); ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.stroke(); }
    else { ctx.fillStyle = col; ctx.fill(); ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1; ctx.stroke(); }
  }
  void lang;
}

const CW = 420, CH = 310;
function drawCurve(canvas: HTMLCanvasElement, lang: Lang, run: Run, m: number) {
  const c = plotColors(canvas), ctx = startPlot(canvas, CW, CH, c);
  const x0 = 50, x1 = CW - 12, y0 = 84, y1 = CH - 44, curve = run.curve;
  const all = [...curve.bagged, ...curve.member, run.singleError, BAYES_ERROR];
  const lo = Math.max(0, Math.floor(Math.min(...all) * 100) - 1), hi = Math.ceil(Math.max(...all) * 100) + 1;
  const gx = (k: number) => x0 + (k - 1) / (MAX_M - 1) * (x1 - x0), gy = (v: number) => y1 - (v * 100 - lo) / (hi - lo) * (y1 - y0);
  const step = hi - lo > 12 ? 4 : hi - lo > 6 ? 2 : 1;
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    line(ctx, x0, gy(v / 100), x1, gy(v / 100), c.grid, 1, [2, 4]);
    label(ctx, String(v), x0 - 8, gy(v / 100) + 5, { size: 14, align: 'right', color: c.muted });
  }
  line(ctx, x0, y0, x0, y1, c.muted, 1); line(ctx, x0, y1, x1, y1, c.muted, 1);
  for (const k of [1, 25, 50, 75, 100]) { line(ctx, gx(k), y1, gx(k), y1 + 4, c.muted, 1); label(ctx, String(k), gx(k), y1 + 20, { size: 14, align: 'center', color: c.muted }); }
  label(ctx, pick(lang, 'Number of trees M', '樹的數量 M'), (x0 + x1) / 2, CH - 8, { size: 14, align: 'center', color: c.ink });
  label(ctx, pick(lang, 'Test error (%)', '測試錯誤率（%）'), 6, 18, { size: 14, bold: true, color: c.ink });
  // Reference lines: best possible, one tree on all data, average member.
  line(ctx, x0, gy(BAYES_ERROR), x1, gy(BAYES_ERROR), c.muted, 1.6, [2, 3]);
  line(ctx, x0, gy(run.singleError), x1, gy(run.singleError), c.b, 2, [8, 5]);
  const trace = (ys: number[], color: string, width: number, dash: number[]) => {
    ctx.setLineDash(dash); ctx.lineWidth = width; ctx.strokeStyle = color; ctx.beginPath();
    ys.forEach((v, i) => { const X = gx(i + 1), Y = gy(v); if (i) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); });
    ctx.stroke(); ctx.setLineDash([]);
  };
  trace(curve.member, c.muted, 2, [3, 3]);
  trace(curve.bagged, c.a, 2.8, []);
  // Current M.
  line(ctx, gx(m), y0, gx(m), y1, c.ink, 1, [4, 3]);
  ctx.beginPath(); ctx.arc(gx(m), gy(curve.bagged[m - 1]), 6, 0, 2 * Math.PI); ctx.fillStyle = c.bg; ctx.fill(); ctx.lineWidth = 2.6; ctx.strokeStyle = c.a; ctx.stroke();
  // Legend (two rows).
  const items: [string, string, number[], number][] = [
    [pick(lang, 'bagged vote', '袋裝投票'), c.a, [], 2.8],
    [pick(lang, 'average member', '成員平均'), c.muted, [3, 3], 2],
    [pick(lang, 'one tree, all data', '單棵樹（全部資料）'), c.b, [8, 5], 2],
    [pick(lang, `best possible ${(BAYES_ERROR * 100).toFixed(1)}%`, `理論最佳 ${(BAYES_ERROR * 100).toFixed(1)}%`), c.muted, [2, 3], 1.6],
  ];
  items.forEach(([text, color, dash, w], i) => {
    const lx = 10 + (i % 2) * 196, ly = 44 + Math.floor(i / 2) * 22;
    line(ctx, lx, ly - 5, lx + 24, ly - 5, color, w, dash);
    label(ctx, text, lx + 30, ly, { size: 14, color: c.ink });
  });
}

export const lab: LabDef = {
  kind: 'image', id: 'bagging-trees', track: 'pattern-recognition',
  title: L('PR 09 · Does averaging many unstable trees beat one tree?', 'PR 09・平均許多不穩定的樹，會比一棵樹更好嗎？'),
  description: L(
    'Grow real decision trees on 300 noisy two-moons points, bootstrap them, let them vote, and watch the test error fall and then flatten as trees are added.',
    '在 300 個帶雜訊的雙月形資料點上訓練真正的決策樹，用自助抽樣（bootstrap）做出許多棵讓它們投票，觀察測試誤差如何隨樹數下降，然後趨於平緩。'),
  challenge: L(
    'Start with 50 trees at the fully grown depth 14 and drag M from 1 to 100: the vote falls from 14.60% to 11.10% over the first ten trees, then barely moves (10.25% at both 50 and 100). Switch the view to One member and drag M to see how ragged single bootstrap trees are, then back to the vote. Lower the depth to 2: the members now disagree on only 4.88% of the test points per pair, and the vote (10.70%) is no better than one tree (10.50%). Finally try depth 5: one tree already scores 9.30% against 10.15% for 50 bagged trees, a gap of less than one percentage point, about the noise of a 1000-point test set, so bagging does not replace choosing a sensible depth. Why does a vote of similar trees gain nothing?',
    '從 50 棵、完全長成的深度 14 開始，把 M 從 1 拉到 100：前十棵樹讓投票的錯誤率從 14.60% 降到 11.10%，之後幾乎不動（50 棵與 100 棵都是 10.25%）。把檢視切換成「單一成員」，拖動 M 看看單一自助樹有多凌亂，再切回投票。把深度降到 2：成員之間任兩棵只在 4.88% 的測試點上意見不同，投票（10.70%）並沒有比單棵樹（10.50%）好。最後試試深度 5：單棵樹就有 9.30%，50 棵袋裝樹是 10.15%，差距不到 1 個百分點，約等於 1000 個測試點的雜訊，所以袋裝法不能取代選一個合理的深度。為什麼一群相似的樹投票卻毫無進步？'),
  note: L(
    'Data: the two-moons set from scikit-learn make_moons with Gaussian noise 0.3, generated once and stored in the page: 300 training points (150 per class, data seed 2) and a separate held-out test set of 1000 points (seed 1002). The data seed is the one of 30 random draws whose bagging gain is closest to the average gain, so this is a typical case, not a favourable one. Trees: CART with Gini impurity and axis-aligned splits at the midpoint between neighbouring values; the depth limit comes from the slider (14 grows every tree here until its leaves are pure). Bagging: member b is trained on 300 rows drawn with replacement from one fixed seeded stream (seed 20260618), so moving M only adds or removes trees and the curve does not change. The vote is a plain majority; for an even M an exact tie counts as half an error. Test error is measured on the 1000 held-out points; its standard error is about 1 percentage point at an error of 10%, so differences smaller than that between two different settings are weak evidence on their own. Disagreement: for every pair of the M members, the share of the 1000 test points on which the two predict different classes, averaged over all pairs (0 means identical members; the closer to 0, the more alike the members, which is a proxy for how correlated their mistakes are). Ties between equally good splits are broken by a fixed rule (feature 1 before feature 2, lower threshold first), while scikit-learn visits the features in random order: in the tests, trees without such a tie give predictions identical to DecisionTreeClassifier, and the vote error of the full 100-tree ensembles differs from the same ensembles built with scikit-learn trees by at most 0.7 percentage points. Best possible 8.5% is the error of the ideal classifier on the distribution that make_moons samples, computed by numerical integration of the two class densities; the lab does not use it. Real bagging also gives each tree out-of-bag points for free error estimates; this lab uses the separate test set instead.',
    '資料：scikit-learn make_moons 產生的雙月形資料，高斯雜訊 0.3，只產生一次並存在頁面中：300 個訓練點（每類 150 個，資料種子 2）與另外 1000 個測試點（種子 1002）。資料種子是 30 次隨機抽樣中，袋裝法的進步幅度最接近平均值的那一個，所以這是典型的情形，不是特別有利的情形。樹：使用吉尼不純度（Gini impurity）的 CART，沿座標軸切分，門檻取相鄰兩個值的中點；深度上限由滑桿決定（14 時這裡的每棵樹都會長到葉節點純淨為止）。袋裝法：第 b 個成員用來自同一條固定種子亂數序列（種子 20260618）、可重複抽取的 300 列訓練，所以移動 M 只是增加或減少樹，曲線不會改變。投票採簡單多數決；M 為偶數時剛好平手算半個錯誤。測試誤差用 1000 個保留的測試點計算；錯誤率約 10% 時，它的標準誤約為 1 個百分點，所以兩個不同設定之間比這更小的差距，單獨看證據很弱。不一致率：對 M 個成員的每一對，計算兩者在 1000 個測試點中預測不同類別的比例，再對所有成員對取平均（0 代表成員完全相同；越接近 0，成員越相像，這是它們犯的錯有多相關的一個代理指標）。分數相同的切分以固定規則決定（先特徵 1 再特徵 2，門檻由低到高），而 scikit-learn 以隨機順序拜訪特徵：測試中，沒有這種平手的樹與 DecisionTreeClassifier 的預測完全相同，100 棵樹的完整集成其投票錯誤率與用 scikit-learn 的樹組成的同一集成最多相差 0.7 個百分點。「理論最佳 8.5%」是理想分類器在 make_moons 所取樣的分布上的錯誤率，由兩個類別密度的數值積分求得；本實驗並不使用它。真正的袋裝法還能免費利用每棵樹沒抽到的點（out-of-bag）估計誤差；這裡改用獨立的測試集。'),
  related: ['pattern-recognition/pr-09-ensembles-validation'],
  controls: [
    { key: 'm', label: L('Trees M (vote size; in the One member view, which tree)', '樹數 M（投票規模；在「單一成員」檢視中是第幾棵）'), kind: 'range', value: DEFAULT.m, min: 1, max: MAX_M, step: 1 },
    { key: 'depth', label: L('Depth limit of every tree (14 = fully grown)', '每棵樹的深度上限（14 = 完全長成）'), kind: 'range', value: DEFAULT.depth, min: 1, max: MAX_DEPTH, step: 1 },
    { key: 'view', label: L('View', '檢視'), kind: 'select', value: DEFAULT.view, options: [
      { value: 'bagged', label: L('Bagged vote of M trees', 'M 棵樹的袋裝投票') },
      { value: 'single', label: L('One tree trained on all 300 points', '用全部 300 點訓練的單棵樹') },
      { value: 'member', label: L('One member: the M-th bootstrap tree', '單一成員：第 M 棵自助樹') },
    ] },
  ],
  predict: {
    question: L(
      'Fully grown trees (depth 14) are trained on 300 noisy moon points. Fifty more copies are trained on bootstrap resamples of the same points and vote on 1000 fresh test points. Compared with one fully grown tree trained on all 300 points, the vote will…',
      '在 300 個帶雜訊的月形點上訓練完全長成的樹（深度 14）。另外對同一批點做自助重抽樣，訓練 50 棵樹，讓它們對 1000 個全新的測試點投票。和一棵用全部 300 點訓練、完全長成的樹相比，投票的結果會……'),
    options: [
      { value: 'better', label: L('Have a clearly lower test error, but nowhere near zero', '測試誤差明顯較低，但離零還很遠') },
      { value: 'zero', label: L('Drop close to zero error, because the trees’ mistakes cancel out', '降到接近零，因為各棵樹的錯誤會互相抵銷') },
      { value: 'same', label: L('Be no better, because each member sees fewer different points than the single tree', '沒有比較好，因為每個成員看到的不同點比單棵樹少') },
    ],
    answer: 'better',
    explain: L(
      'Yes, at the lab’s seed: the single fully grown tree misclassifies 12.90% of the 1000 test points and the vote of 50 bagged trees 10.25%. Each member alone is worse than that tree (14.73% on average), because it sees only a resample of the points; the vote wins because the members do not make the same mistakes: a random pair of them disagrees on 13.70% of the test points. It cannot fall near zero. For 100 test points, more than half of the 50 trees are wrong, and adding trees cannot fix those: every tree learns from the same 300 noisy points, so they share the same bias and many of the same mistakes. This is the limit in the chapter’s formula, where the variance of the average of M predictors with pairwise correlation ρ is σ²[ρ + (1 − ρ)/M]: more trees only shrink the second term. Going from 50 to 100 trees changes nothing (10.25% both times), and even the best possible classifier would still make about 8.5% errors on this noisy data.',
      '是的（以本實驗的種子而言）：一棵完全長成的單棵樹在 1000 個測試點上錯 12.90%，而 50 棵袋裝樹的投票只錯 10.25%。每個成員單獨看都比那棵樹差（平均 14.73%），因為它只看到資料點的一份重抽樣；投票之所以贏，是因為各成員犯的錯並不相同：隨機取兩個成員，它們在 13.70% 的測試點上意見不同。但誤差不可能降到接近零。有 100 個測試點被 50 棵樹中的多數答錯，增加樹也無法修正它們：所有的樹都從同一批 300 個帶雜訊的點學習，共享同樣的偏差與許多相同的錯誤。這正是本章公式的極限：M 個兩兩相關係數為 ρ 的預測器，平均後的變異數是 σ²[ρ + (1 − ρ)/M]，增加樹只會縮小第二項。從 50 棵增加到 100 棵毫無改變（兩次都是 10.25%），而且在這份帶雜訊的資料上，就連理論最佳的分類器也約有 8.5% 的錯誤。'),
  },
  assets: {},
  paneMin: 400,
  panes: [
    { key: 'map', kind: 'canvas', label: L('Decision map of the chosen view over the 300 training points (circle = class 0, square = class 1; paler colour = the trees disagree; hollow points = missing from that member’s bootstrap sample)', '所選檢視的決策圖與 300 個訓練點（圓形 = 類別 0，方形 = 類別 1；顏色越淡代表樹之間意見越分歧；空心點 = 該成員的自助樣本沒抽到的點）'), width: MAP_W, height: MAP_H },
    { key: 'curve', kind: 'canvas', label: L('Test error of the bagged vote for every M from 1 to 100 (1000 held-out points); the ring marks the current M', '從 1 到 100 的每個 M，袋裝投票在 1000 個保留測試點上的錯誤率；圓圈標示目前的 M'), width: CW, height: CH },
  ],
  sources: [
    { title: 'L. Breiman, Bagging Predictors, Machine Learning, 24(2), 123-140, 1996 (bootstrap aggregating)', url: 'https://doi.org/10.1007/BF00058655' },
    { title: 'L. Breiman, J. H. Friedman, R. A. Olshen and C. J. Stone, Classification and Regression Trees, Chapman and Hall/CRC (Wadsworth 1984), 2017 edition (CART, Gini impurity)', url: 'https://doi.org/10.1201/9781315139470' },
    { title: 'F. Pedregosa et al., Scikit-learn: Machine Learning in Python, Journal of Machine Learning Research, 12, 2825-2830, 2011 (make_moons and the DecisionTreeClassifier used for the golden tests)', url: 'https://jmlr.org/papers/v12/pedregosa11a.html' },
    { title: 'scikit-learn documentation, sklearn.datasets.make_moons (the generated two-moons data, no download)', url: 'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html' },
  ],
  create({ lang }) {
    let run: Run | undefined;
    return {
      render(state, canvases) {
        const m = clamp(num(state, 'm'), 1, MAX_M), depth = clamp(num(state, 'depth'), 1, MAX_DEPTH);
        const view = (VIEWS as string[]).includes(String(state.view)) ? (state.view as View) : 'bagged';
        if (!run || run.depth !== depth) run = runFor(depth);
        const curve = run.curve, shown = view === 'bagged' ? run.trees.slice(0, m) : view === 'single' ? [run.single] : [run.trees[m - 1]];
        const hollow = view === 'member' ? outOfBag(SAMPLES[m - 1]) : null;
        drawMap(canvases.map, lang, mapVotes(shown), shown.length, hollow);
        drawCurve(canvases.curve, lang, run, m);
        const bagged = curve.bagged[m - 1], single = run.singleError, member = curve.member[m - 1], dis = curve.disagreement[m - 1];
        const gap = (single - bagged) * 100;
        const verdict = Math.abs(gap) < 0.5
          ? pick(lang, `about the same as one tree (${Math.abs(gap).toFixed(2)} points apart)`, `和單棵樹差不多（相差 ${Math.abs(gap).toFixed(2)} 個百分點）`)
          : gap > 0 ? pick(lang, `${gap.toFixed(2)} points better than one tree`, `比單棵樹好 ${gap.toFixed(2)} 個百分點`) : pick(lang, `${(-gap).toFixed(2)} points worse than one tree`, `比單棵樹差 ${(-gap).toFixed(2)} 個百分點`);
        let summary: string;
        if (view === 'bagged') {
          summary = pick(lang,
            `${m} bagged ${m === 1 ? 'tree' : 'trees'} (depth limit ${depth}) misclassify ${pct(bagged)} of the 1000 test points: ${verdict}. One tree trained on all 300 points gets ${pct(single)}; the average member alone ${pct(member)}.${m >= 2 ? ` Two members chosen at random disagree on ${pct(dis)} of the test points.` : ''}`,
            `${m} 棵袋裝樹（深度上限 ${depth}）在 1000 個測試點上錯 ${pct(bagged)}：${verdict}。用全部 300 點訓練的單棵樹是 ${pct(single)}；成員單獨的平均是 ${pct(member)}。${m >= 2 ? `隨機挑兩個成員，它們在 ${pct(dis)} 的測試點上意見不同。` : ''}`);
        } else if (view === 'single') {
          summary = pick(lang,
            `One tree trained on all 300 points (depth limit ${depth}, ${leafCount(run.single)} leaves) misclassifies ${pct(single)} of the 1000 test points. Its map has hard edges that follow the individual noisy points.`,
            `用全部 300 點訓練的單棵樹（深度上限 ${depth}，${leafCount(run.single)} 個葉節點）在 1000 個測試點上錯 ${pct(single)}。它的決策圖邊界很硬，緊貼著各個帶雜訊的點。`);
        } else {
          const own = errorOf(run.testPreds[m - 1], TEST_Y), missing = hollow!.reduce((a, v) => a + v, 0);
          summary = pick(lang,
            `Member ${m}: a tree trained on a bootstrap sample that misses ${missing} of the 300 points (hollow). It misclassifies ${pct(own)} of the test points, against ${pct(single)} for the tree trained on all the data. Drag M to meet other members.`,
            `第 ${m} 個成員：用一份自助樣本訓練的樹，300 個點中有 ${missing} 個沒被抽到（空心點）。它在測試點上錯 ${pct(own)}，用全部資料訓練的樹則是 ${pct(single)}。拖動 M 可以看看其他成員。`);
        }
        return {
          summary,
          metrics: [
            { label: pick(lang, `Bagged vote, ${m} ${m === 1 ? 'tree' : 'trees'}: test error`, `${m} 棵袋裝投票：測試錯誤率`), value: pct(bagged) },
            { label: pick(lang, 'One tree on all 300 points: test error', '全部 300 點的單棵樹：測試錯誤率'), value: pct(single) },
            { label: pick(lang, 'Average single member: test error', '成員單獨的平均測試錯誤率'), value: pct(member) },
            { label: pick(lang, 'Disagreement between two members', '兩個成員間的不一致率'), value: m >= 2 ? pct(dis) : '—' },
          ],
        };
      },
    };
  },
};
