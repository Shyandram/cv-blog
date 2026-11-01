import type { LabDef } from './types.ts';
import { L, pick, fmt, num } from '../types.ts';
import type { Lang } from '../../../i18n/ui';
import { plotColors, startPlot, label, line } from './plot2d.ts';
import { normalField } from '../../imagekit/rng.ts';
import { gaussianBlur } from '../../imagekit/filter.ts';

// ---------------------------------------------------------------------------------------------- the synthetic rectified pair
// Left image 128 x 96. Background: a fronto-parallel textured plane at disparity 4. Foreground: a textured disc at disparity 12, brighter than
// the background so that its outline is a strong intensity edge. A textureless rectangle sits on the background plane. The right image is
// made by shifting every left pixel left by its disparity (the nearer surface wins), with fresh texture where nothing lands. Fixed seeds, so
// nothing flickers when a slider moves.
export const W = 128, H = 96, D = 16, D_BG = 4, D_FG = 12;
export const FG = { cx: 86, cy: 44, r: 25 };
export const FLAT = { x0: 20, x1: 46, y0: 62, y1: 84 };
export const WINDOWS = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31];
export const DEFAULTS = { win: 3, lr: 'off', noise: 0.03 };
export const NOISE_MAX = 0.08;
const BG_MEAN = 0.36, BG_AMP = 0.1, FG_MEAN = 0.64, FG_AMP = 0.14, FLAT_VALUE = 0.5, TEXTURE_SEED = 31, FILL_SEED = 32, NOISE_SEED_L = 33, NOISE_SEED_R = 34;

export interface Scene {
  left: Float32Array; right: Float32Array;
  /** True disparity of every left pixel. */
  gt: Uint8Array;
  /** 1 where the left pixel has no partner in the right image (hidden behind the disc there, or shifted out of the picture). */
  occluded: Uint8Array;
  /** 1 on the textureless rectangle. */
  flat: Uint8Array;
  /** Pixels used for scoring: visible in both views and at least D columns from the left border (where the larger disparities have no partner). */
  evaluated: Uint8Array;
  /** Noise-free texture field values are inside `left`/`right` already; `noiseL`/`noiseR` are unit-variance fields, scaled by the noise slider. */
  noiseL: Float32Array; noiseR: Float32Array;
}
function texture(seed: number): Float32Array {
  const t = gaussianBlur(normalField(W * H, seed), W, H, 1.2);
  let m = 0, v = 0;
  for (let i = 0; i < t.length; i++) m += t[i];
  m /= t.length;
  for (let i = 0; i < t.length; i++) v += (t[i] - m) ** 2;
  const sd = Math.sqrt(v / t.length), out = new Float32Array(t.length);
  for (let i = 0; i < t.length; i++) out[i] = (t[i] - m) / sd;
  return out;
}
let sceneCache: Scene | undefined;
export function makeScene(): Scene {
  if (sceneCache) return sceneCache;
  const tex = texture(TEXTURE_SEED), fill = texture(FILL_SEED);
  const gt = new Uint8Array(W * H), left = new Float32Array(W * H), flat = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, inDisc = (x - FG.cx) ** 2 + (y - FG.cy) ** 2 < FG.r ** 2;
    flat[i] = !inDisc && x >= FLAT.x0 && x < FLAT.x1 && y >= FLAT.y0 && y < FLAT.y1 ? 1 : 0;
    gt[i] = inDisc ? D_FG : D_BG;
    left[i] = inDisc ? FG_MEAN + FG_AMP * tex[i] : flat[i] ? FLAT_VALUE : BG_MEAN + BG_AMP * tex[i];
  }
  const right = new Float32Array(W * H), occluded = new Uint8Array(W * H), evaluated = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    const owner = new Int16Array(W).fill(-1), row = y * W;
    for (let i = 0; i < W; i++) right[row + i] = BG_MEAN + BG_AMP * fill[row + i];
    // Painter's algorithm: the smaller disparity first, so a nearer pixel overwrites what it covers.
    for (const d of [D_BG, D_FG]) for (let x = 0; x < W; x++) if (gt[row + x] === d && x - d >= 0) { right[row + x - d] = left[row + x]; owner[x - d] = x; }
    for (let x = 0; x < W; x++) {
      const d = gt[row + x], xr = x - d;
      occluded[row + x] = xr < 0 || owner[xr] !== x ? 1 : 0;
      evaluated[row + x] = !occluded[row + x] && x >= D ? 1 : 0;
    }
  }
  sceneCache = { left, right, gt, occluded, flat, evaluated, noiseL: normalField(W * H, NOISE_SEED_L), noiseR: normalField(W * H, NOISE_SEED_R) };
  return sceneCache;
}
/** The pair as the matcher sees it: the clean images plus sensor noise of standard deviation `sigma` (a fixed field, only scaled). */
export function noisyPair(sc: Scene, sigma: number): { left: Float32Array; right: Float32Array } {
  const left = new Float32Array(W * H), right = new Float32Array(W * H);
  for (let i = 0; i < left.length; i++) { left[i] = sc.left[i] + sigma * sc.noiseL[i]; right[i] = sc.right[i] + sigma * sc.noiseR[i]; }
  return { left, right };
}

// ---------------------------------------------------------------------------------------------- block matching
/**
 * SAD block matching with winner-take-all for several window sizes at once. The cost of a window is the mean absolute difference over the
 * pixels of the window that have a partner (so windows that stick out of the picture, or whose partner would lie left of the right image, are
 * not penalized); the disparity of a pixel is the d in 0..D-1 with the lowest cost (ties: the smallest d). `ref` selects the reference image:
 * 'left' matches left(x) with right(x - d), 'right' matches right(x) with left(x + d). Returns one disparity map per window size.
 */
export function blockMatch(left: Float32Array, right: Float32Array, w: number, h: number, nd: number, windows: number[], ref: 'left' | 'right'): Uint8Array[] {
  const n = w * h, W1 = w + 1;
  const best = windows.map(() => new Float32Array(n).fill(Infinity)), arg = windows.map(() => new Uint8Array(n));
  const sumAd = new Float64Array(W1 * (h + 1)), sumOk = new Float64Array(W1 * (h + 1));
  const sign = ref === 'left' ? -1 : 1, a = ref === 'left' ? left : right, b = ref === 'left' ? right : left;
  for (let d = 0; d < nd; d++) {
    // Summed-area tables of |a(x) - b(x + sign * d)| and of the indicator that the partner exists.
    for (let y = 0; y < h; y++) {
      let ra = 0, ro = 0;
      for (let x = 0; x < w; x++) {
        const xp = x + sign * d, ok = xp >= 0 && xp < w;
        if (ok) { ra += Math.abs(a[y * w + x] - b[y * w + xp]); ro += 1; }
        sumAd[(y + 1) * W1 + x + 1] = sumAd[y * W1 + x + 1] + ra;
        sumOk[(y + 1) * W1 + x + 1] = sumOk[y * W1 + x + 1] + ro;
      }
    }
    for (let k = 0; k < windows.length; k++) {
      const r = (windows[k] - 1) >> 1, bk = best[k], ak = arg[k];
      for (let y = 0; y < h; y++) {
        const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
        for (let x = 0; x < w; x++) {
          const xc = x + sign * d;
          if (xc < 0 || xc >= w) continue; // the centre pixel itself has no partner at this disparity
          const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1);
          const ok = sumOk[y1 * W1 + x1] - sumOk[y0 * W1 + x1] - sumOk[y1 * W1 + x0] + sumOk[y0 * W1 + x0];
          const c = (sumAd[y1 * W1 + x1] - sumAd[y0 * W1 + x1] - sumAd[y1 * W1 + x0] + sumAd[y0 * W1 + x0]) / ok;
          if (c < bk[y * w + x]) { bk[y * w + x] = c; ak[y * w + x] = d; }
        }
      }
    }
  }
  return arg;
}
export const INVALID = 255;
/**
 * Left-right consistency check: a left pixel keeps its disparity d only if the right image, matched back, gives a disparity within
 * `tol` pixels of d at the position x - d (as in the chapter's snippet). Otherwise it becomes INVALID (a hole, not a guess).
 */
export function leftRightCheck(dl: Uint8Array, dr: Uint8Array, w: number, h: number, tol = 1): Uint8Array {
  const out = new Uint8Array(dl.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, d = dl[i], xr = x - d;
    out[i] = xr >= 0 && Math.abs(d - dr[y * w + xr]) <= tol ? d : INVALID;
  }
  return out;
}
export interface StereoScores {
  /** Percentage of evaluated pixels (kept ones only, if the check is on) with |d - d_gt| > 1: the 'bad-1' of the chapter. */
  bad1: number;
  /** Bad-1 over the part of the textureless rectangle that is evaluated (kept pixels only, if the check is on). */
  badFlat: number;
  /** Visible background pixels that were given the foreground disparity (within 1), as a percentage of the disc's true area: how much the disc has fattened. */
  fgSpill: number;
  /** Percentage of the evaluated pixels that the check invalidated (0 when the check is off). */
  invalid: number;
  /** Percentage of the occluded pixels that the check invalidated (0 when the check is off). */
  invalidOccluded: number;
}
export function scoreDisparity(sc: Scene, disp: Uint8Array): StereoScores {
  let ev = 0, kept = 0, bad = 0, flatKept = 0, flatBad = 0, fgTrue = 0, occ = 0, occInvalid = 0, spill = 0;
  for (let i = 0; i < disp.length; i++) {
    const ok = disp[i] !== INVALID;
    if (sc.gt[i] === D_FG) fgTrue++;
    if (ok && Math.abs(disp[i] - D_FG) <= 1 && sc.evaluated[i] && sc.gt[i] !== D_FG) spill++;
    if (sc.occluded[i]) { occ++; if (!ok) occInvalid++; }
    if (!sc.evaluated[i]) continue;
    ev++;
    if (!ok) continue;
    kept++;
    const wrong = Math.abs(disp[i] - sc.gt[i]) > 1;
    if (wrong) bad++;
    if (sc.flat[i]) { flatKept++; if (wrong) flatBad++; }
  }
  return {
    bad1: kept ? 100 * bad / kept : NaN, badFlat: flatKept ? 100 * flatBad / flatKept : NaN,
    fgSpill: 100 * spill / fgTrue, invalid: ev ? 100 * (ev - kept) / ev : 0, invalidOccluded: occ ? 100 * occInvalid / occ : 0,
  };
}
export interface MatchResult { left: Uint8Array[]; right: Uint8Array[] }
/** Disparity maps of both references for every window in WINDOWS at one noise level. */
export function matchPair(sc: Scene, sigma: number): MatchResult {
  const { left, right } = noisyPair(sc, sigma);
  return { left: blockMatch(left, right, W, H, D, WINDOWS, 'left'), right: blockMatch(left, right, W, H, D, WINDOWS, 'right') };
}
/** The disparity map of window index k, with or without the left-right check. */
export function finalDisparity(m: MatchResult, k: number, lr: boolean): Uint8Array {
  return lr ? leftRightCheck(m.left[k], m.right[k], W, H) : m.left[k];
}

// ---------------------------------------------------------------------------------------------- scores for every window (the plot)
export interface Sweep { scores: StereoScores[]; best: number }
/** Scores of every window size at this noise level; `best` is the index of the window with the lowest bad-1. */
export function sweep(sc: Scene, m: MatchResult, lr: boolean): Sweep {
  const scores = WINDOWS.map((_, k) => scoreDisparity(sc, finalDisparity(m, k, lr)));
  let best = 0;
  scores.forEach((s, k) => { if (s.bad1 < scores[best].bad1) best = k; });
  return { scores, best };
}

// ---------------------------------------------------------------------------------------------- drawing
const DRAW = 2, CW = W * DRAW, CH = H * DRAW;
const STOPS: [number, number, number][] = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
export const INVALID_COLOR: [number, number, number] = [255, 45, 149];
/** A viridis-like color for disparity 0..D-1. */
export function disparityColor(d: number): [number, number, number] {
  const t = Math.min(1, Math.max(0, d / (D - 1))) * (STOPS.length - 1), i = Math.min(STOPS.length - 2, Math.floor(t)), f = t - i;
  return [0, 1, 2].map(c => Math.round(STOPS[i][c] * (1 - f) + STOPS[i + 1][c] * f)) as [number, number, number];
}
/** Pixels of the true depth edge: where the true disparity changes to the right or below. */
export function trueEdge(sc: Scene): Uint8Array {
  const e = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    if ((x + 1 < W && sc.gt[i] !== sc.gt[i + 1]) || (y + 1 < H && sc.gt[i] !== sc.gt[i + W])) e[i] = 1;
  }
  return e;
}
function paint(canvas: HTMLCanvasElement, pixel: (i: number) => [number, number, number]) {
  if (canvas.width !== CW) canvas.width = CW;
  if (canvas.height !== CH) canvas.height = CH;
  const ctx = canvas.getContext('2d')!, im = ctx.createImageData(CW, CH);
  for (let y = 0; y < CH; y++) for (let x = 0; x < CW; x++) {
    const [r, g, b] = pixel(Math.floor(y / DRAW) * W + Math.floor(x / DRAW)), o = 4 * (y * CW + x);
    im.data[o] = r; im.data[o + 1] = g; im.data[o + 2] = b; im.data[o + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}
/** Washes out the left D columns, which are not scored (the larger disparities have no partner there). */
const unscored = (i: number, c: [number, number, number]): [number, number, number] => (i % W < D ? [Math.round(0.4 * c[0] + 153), Math.round(0.4 * c[1] + 153), Math.round(0.4 * c[2] + 153)] : c);

const PL_W = 400, PL_H = 320;
function drawSweep(canvas: HTMLCanvasElement, lang: Lang, sw: Sweep, k: number) {
  const col = plotColors(canvas), ctx = startPlot(canvas, PL_W, PL_H, col), x0 = 44, x1 = 384, y0 = 28, y1 = 214;
  const series = [
    { name: pick(lang, 'bad-1, all scored pixels', 'bad-1，所有評分像素'), color: col.ink, get: (s: StereoScores) => s.bad1, dash: [] as number[] },
    { name: pick(lang, 'bad-1 in the textureless patch', 'bad-1，無紋理區塊'), color: col.a, get: (s: StereoScores) => s.badFlat, dash: [] as number[] },
    { name: pick(lang, 'disc fattening (% of disc area)', '圓盤變胖（占圓盤面積的 %）'), color: col.b, get: (s: StereoScores) => s.fgSpill, dash: [6, 4] },
  ];
  const top = Math.max(20, Math.ceil(Math.max(...series.flatMap(s => sw.scores.map(v => s.get(v))).filter(Number.isFinite)) / 10) * 10);
  const X = (i: number) => x0 + (i / (WINDOWS.length - 1)) * (x1 - x0), Y = (v: number) => y1 - (v / top) * (y1 - y0);
  label(ctx, pick(lang, 'Error (%) against window size', '誤差（%）與視窗大小'), 8, 18, { color: col.ink, size: 15, bold: true });
  for (const v of [0, top / 2, top]) { line(ctx, x0, Y(v), x1, Y(v), col.grid, 1); label(ctx, String(v), x0 - 6, Y(v) + 5, { color: col.muted, size: 13, align: 'right' }); }
  for (const w of [3, 11, 21, 31]) { const i = WINDOWS.indexOf(w); label(ctx, `${w}`, X(i), y1 + 18, { color: col.muted, size: 13, align: 'center' }); line(ctx, X(i), y1, X(i), y1 + 4, col.muted, 1); }
  label(ctx, pick(lang, 'window size (pixels per side)', '視窗大小（每邊像素數）'), (x0 + x1) / 2, y1 + 36, { color: col.muted, size: 13, align: 'center' });
  line(ctx, X(k), y0 - 4, X(k), y1, col.ink, 1.5, [3, 3]);
  series.forEach(s => {
    ctx.beginPath(); ctx.setLineDash(s.dash); ctx.lineWidth = 2.4; ctx.strokeStyle = s.color; ctx.lineJoin = 'round';
    let started = false;
    sw.scores.forEach((v, i) => { const y = s.get(v); if (!Number.isFinite(y)) return; if (!started) { ctx.moveTo(X(i), Y(y)); started = true; } else ctx.lineTo(X(i), Y(y)); });
    ctx.stroke(); ctx.setLineDash([]);
    const yk = s.get(sw.scores[k]);
    if (Number.isFinite(yk)) { ctx.beginPath(); ctx.arc(X(k), Y(yk), 4.5, 0, 2 * Math.PI); ctx.fillStyle = s.color; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = col.bg; ctx.stroke(); }
  });
  series.forEach((s, i) => {
    const y = y1 + 62 + i * 20;
    ctx.beginPath(); ctx.setLineDash(s.dash); ctx.lineWidth = 2.4; ctx.strokeStyle = s.color; ctx.moveTo(10, y - 4); ctx.lineTo(40, y - 4); ctx.stroke(); ctx.setLineDash([]);
    label(ctx, s.name, 48, y, { color: col.ink, size: 14 });
  });
}

// ---------------------------------------------------------------------------------------------- the lab
const pct = (v: number) => (Number.isFinite(v) ? `${fmt(v, 1)}%` : '—');
export const lab: LabDef = {
  kind: 'image', id: 'stereo-matching', track: 'deep-learning',
  title: L('DLCV 08A · Is a bigger matching window always better?', 'DLCV 08A・匹配視窗越大一定越好嗎？'),
  description: L('Run block matching on a synthetic stereo pair with known disparity, and see small windows turn noisy while large windows fatten the foreground.', '在已知視差的合成立體影像對上執行區塊匹配，看小視窗變得雜亂，而大視窗讓前景變胖。'),
  challenge: L(
    'At the default 3 × 3 window the map is noisy, most of all inside the textureless patch. Grow the window one step at a time and watch the three curves: where is the all-pixel error lowest? What happens to the white outline of the disc once the window reaches 31 × 31? Then raise the noise to 0.06 and see how much a 3 × 3 window suffers. Finally switch the left–right check on: which pixels turn pink, and what happens to bad-1?',
    '預設的 3 × 3 視窗會讓視差圖很雜亂，無紋理區塊裡最嚴重。一次放大一格視窗，觀察三條曲線：所有像素的誤差在哪裡最低？視窗到 31 × 31 時，圓盤的白色輪廓發生了什麼事？接著把雜訊調到 0.06，看 3 × 3 視窗受影響多大。最後打開左右一致性檢查：哪些像素變成粉紅色？bad-1 又怎麼變？'),
  note: L(
    'Synthetic and tiny on purpose: the left image is 128 × 96 and is drawn by code, with fixed seeds so nothing flickers. A textured background plane has disparity 4, a brighter textured disc has disparity 12, and a flat gray rectangle lies on the background. The right image shifts every left pixel left by its disparity (the nearer surface wins) and fills the uncovered pixels with fresh texture; Gaussian sensor noise with the chosen standard deviation is added to both images. The matcher is the one of the chapter: for every disparity 0 to 15 the cost of a window is the mean absolute difference (SAD, normalized by the number of pixels that have a partner), and the winner-take-all rule takes the lowest cost. Bad-1 is the percentage of scored pixels whose disparity is off by more than 1 pixel. Scored pixels are those visible in both views and at least 16 columns from the left border (washed out in the pictures). The check matches again with the right image as reference and invalidates a pixel whose two answers differ by more than 1; bad-1 and the flat-patch score are then taken over the kept pixels. Disc fattening counts the visible background pixels labelled with the disc’s disparity (within 1), as a percentage of the disc’s true area. These are the lab’s own numbers on its own toy scene, not those of the chapter’s scene.',
    '刻意做成合成而小巧：左影像為 128 × 96，由程式畫出，使用固定的隨機種子，因此拖動滑桿時畫面不會閃動。有紋理的背景平面視差為 4，較亮、有紋理的圓盤視差為 12，背景上另有一塊平坦的灰色矩形。右影像把每個左影像像素依其視差向左平移（較近的表面勝出），未被覆蓋的像素以新的紋理填補；兩張影像都加上標準差為所選值的高斯感測器雜訊。匹配方法與章節相同：對每個視差 0 到 15，視窗的成本是絕對差的平均（SAD，除以有對應像素的數量），再由贏者全拿（winner-take-all）取成本最低者。bad-1 是評分像素中視差誤差超過 1 像素者的百分比。評分像素是兩個視角都看得到、且距左邊界至少 16 欄的像素（圖中左側褪色的部分不評分）。檢查會改以右影像為參考再匹配一次，兩次答案相差超過 1 的像素即作廢；此時 bad-1 與平坦區塊的分數只計算保留下來的像素。圓盤變胖是指被標成圓盤視差（誤差 1 以內）的可見背景像素，占圓盤真實面積的百分比。這些是本實驗在自己的玩具場景上的數字，不是章節場景的數字。'),
  related: ['deep-learning/dl-08a-depth-optical-flow'],
  controls: [
    { key: 'win', label: L('Matching window (pixels per side)', '匹配視窗（每邊像素數）'), kind: 'range', value: DEFAULTS.win, min: 3, max: 31, step: 2 },
    { key: 'lr', label: L('Left–right consistency check', '左右一致性檢查'), kind: 'select', value: DEFAULTS.lr, options: [
      { value: 'off', label: L('Off: keep every winner', '關閉：保留每個贏家') },
      { value: 'on', label: L('On: invalidate pixels whose two answers disagree', '開啟：兩次答案不一致的像素作廢') },
    ] },
    { key: 'noise', label: L('Image noise (standard deviation)', '影像雜訊（標準差）'), kind: 'range', value: DEFAULTS.noise, min: 0, max: NOISE_MAX, step: 0.01 },
  ],
  predict: {
    question: L('At the default noise a 3 × 3 window gives a noisy disparity map. Suppose you enlarge the window step by step up to 31 × 31. What happens to the bad-pixel rate (bad-1 over all scored pixels)?', '在預設雜訊下，3 × 3 視窗得到的視差圖很雜亂。如果把視窗一步步放大到 31 × 31，bad 像素率（所有評分像素的 bad-1）會怎麼變？'),
    options: [
      { value: 'falls', label: L('It keeps falling: a bigger window always averages more noise away', '一路下降：視窗越大，平均掉的雜訊越多') },
      { value: 'valley', label: L('It falls at first, then rises again: the best window lies in between', '先下降，之後又上升：最好的視窗在中間') },
      { value: 'rises', label: L('It rises right away: any window larger than 3 × 3 only blurs the depth edge', '立刻上升：任何大於 3 × 3 的視窗都只會把深度邊緣弄糊') },
    ],
    answer: 'valley',
    explain: L(
      'It falls, then rises. A small window has too little texture to decide on (and sees mostly noise), so most of the textureless patch is wrong. A large window fixes the patch but mixes pixels of both depths near the edge, and there the textured, brighter disc wins: it spreads into the background. Bad-1 is lowest for windows around 7 × 7 to 11 × 11 here, and 31 × 31 is worse than those although it is still better than 3 × 3. The all-pixel error hides that two different errors trade off against each other.',
      '先下降，再上升。小視窗的紋理太少，無法判斷（看到的大多是雜訊），所以無紋理區塊大多是錯的。大視窗能修好那一塊，但在邊緣附近會把兩種深度的像素混在一起，而有紋理、較亮的圓盤會勝出：它向背景擴張。這裡 bad-1 在約 7 × 7 到 11 × 11 的視窗最低，31 × 31 比它們差，但仍比 3 × 3 好。所有像素的誤差把兩種互相抵換的誤差藏在一起了。'),
  },
  assets: {},
  paneMin: 300,
  panes: [
    { key: 'left', kind: 'canvas', label: L('Left image (with noise)', '左影像（含雜訊）'), width: CW, height: CH },
    { key: 'truth', kind: 'canvas', label: L('True disparity (dark: far, light: near; left strip not scored)', '真實視差（暗：遠，亮：近；左側色帶不評分）'), width: CW, height: CH },
    { key: 'est', kind: 'canvas', label: L('Estimated disparity (pink: invalidated; white line: true depth edge)', '估計的視差（粉紅：已作廢；白線：真實的深度邊緣）'), width: CW, height: CH },
    { key: 'plot', kind: 'canvas', label: L('Error against window size at this noise level', '此雜訊下，誤差與視窗大小的關係'), width: PL_W, height: PL_H },
  ],
  sources: [
    { title: 'D. Scharstein and R. Szeliski, A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms, International Journal of Computer Vision, 47(1-3), 7-42, 2002 (SAD windows with winner-take-all, the percentage of bad pixels with threshold 1, and the window-size trade-off)', url: 'https://doi.org/10.1023/A:1014573219977' },
  ],
  create({ lang }) {
    const sc = makeScene(), edge = trueEdge(sc), matches = new Map<number, MatchResult>(), sweeps = new Map<string, Sweep>();
    let truthDrawn = false;
    return {
      render(state, canvases) {
        const win = Math.max(WINDOWS[0], Math.min(WINDOWS[WINDOWS.length - 1], Math.round(num(state, 'win')))), k = WINDOWS.reduce((b, w, i) => (Math.abs(w - win) < Math.abs(WINDOWS[b] - win) ? i : b), 0);
        const sigma = Math.max(0, Math.min(NOISE_MAX, num(state, 'noise'))), nk = Math.round(sigma * 100), lr = state.lr === 'on';
        if (!matches.has(nk)) { if (matches.size > 4) { matches.clear(); sweeps.clear(); } matches.set(nk, matchPair(sc, nk / 100)); }
        const m = matches.get(nk)!, sk = `${nk}-${lr}`;
        if (!sweeps.has(sk)) sweeps.set(sk, sweep(sc, m, lr));
        const sw = sweeps.get(sk)!, r = sw.scores[k], disp = finalDisparity(m, k, lr), pair = noisyPair(sc, nk / 100);
        paint(canvases.left, i => { const v = Math.round(255 * Math.min(1, Math.max(0, pair.left[i]))); return [v, v, v]; });
        if (!truthDrawn) { paint(canvases.truth, i => unscored(i, disparityColor(sc.gt[i]))); truthDrawn = true; }
        paint(canvases.est, i => (edge[i] ? [255, 255, 255] : unscored(i, disp[i] === INVALID ? INVALID_COLOR : disparityColor(disp[i]))));
        drawSweep(canvases.plot, lang, sw, k);
        const b = WINDOWS[sw.best], bs = sw.scores[sw.best];
        const head = pick(lang, `With a ${win} × ${win} window, ${pct(r.bad1)} of the scored pixels are off by more than 1 pixel (bad-1). At this noise level the lowest bad-1, ${pct(bs.bad1)}, is reached with a ${b} × ${b} window.`, `使用 ${win} × ${win} 視窗時，${pct(r.bad1)} 的評分像素視差誤差超過 1 像素（bad-1）。在這個雜訊下，bad-1 最低的是 ${b} × ${b} 視窗，為 ${pct(bs.bad1)}。`);
        const tail = win < b
          ? pick(lang, ` The window is smaller than the best one: it sees too little texture, and ${pct(r.badFlat)} of the textureless patch is wrong.`, ` 這個視窗比最好的視窗小：看到的紋理太少，無紋理區塊有 ${pct(r.badFlat)} 是錯的。`)
          : win > b
            ? pick(lang, ` The window is larger than the best one: the flat patch is fine (${pct(r.badFlat)} wrong), but the disc has grown by ${fmt(r.fgSpill, 1)}% of its area into the background.`, ` 這個視窗比最好的視窗大：平坦區塊沒問題（錯 ${pct(r.badFlat)}），但圓盤已向背景多長出相當於其面積 ${fmt(r.fgSpill, 1)}% 的範圍。`)
            : pick(lang, ` This is the best window: the textureless patch is still ${pct(r.badFlat)} wrong, and the disc has grown by ${fmt(r.fgSpill, 1)}% of its area.`, ` 這是最好的視窗：無紋理區塊仍有 ${pct(r.badFlat)} 是錯的，圓盤則多長出相當於其面積 ${fmt(r.fgSpill, 1)}% 的範圍。`);
        const lrNote = lr ? pick(lang, ` The check removed ${pct(r.invalid)} of the scored pixels and ${pct(r.invalidOccluded)} of the occluded ones.`, ` 檢查作廢了 ${pct(r.invalid)} 的評分像素，以及 ${pct(r.invalidOccluded)} 的遮擋像素。`) : '';
        return {
          summary: head + tail + lrNote,
          metrics: [
            { label: pick(lang, 'Bad-1, all scored pixels', 'bad-1，所有評分像素'), value: pct(r.bad1) },
            { label: pick(lang, 'Bad-1 in the textureless patch', 'bad-1，無紋理區塊'), value: pct(r.badFlat) },
            { label: pick(lang, 'Disc fattening (% of its area)', '圓盤變胖（占其面積的 %）'), value: pct(r.fgSpill) },
            { label: pick(lang, 'Invalidated by the check', '被檢查作廢'), value: lr ? pct(r.invalid) : '—' },
          ],
        };
      },
    };
  },
};
