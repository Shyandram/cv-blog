import type { LabDef } from './types.ts';
import { L, pick, fmt, num } from '../types.ts';
import type { State } from '../types.ts';
import type { Lang } from '../../../i18n/ui';
import type { Gray } from '../../imagekit/types.ts';
import { det2, svd2, rotation2, polygonArea, unitSquareImage, ellipsePoint } from '../../imagekit/matrix2.ts';
import type { Mat2, Svd2 } from '../../imagekit/matrix2.ts';
import { centredFrame, warpMatrix, overBackground } from '../../imagekit/warp.ts';
import type { Warped } from '../../imagekit/warp.ts';
import { label } from './plot2d.ts';

/** The photograph is 256 x 256 pixels; the plane origin is its centre and 64 pixels are one unit, so the photo spans -2 to 2 in x and y (y points up). */
export const SIZE = 256;
export const UNIT_PX = 64;
export const FRAME = centredFrame(SIZE, SIZE, UNIT_PX, true);
/** Canvas panes are drawn at 1.5 times the photo size so that arrows and labels stay crisp: 96 canvas pixels per unit, origin at the centre. */
export const VIEW = 384;
export const VIEW_UNIT = (VIEW / SIZE) * UNIT_PX;
/** Tips can be placed anywhere in this square (a little inside the canvas). */
export const TIP_LIMIT = 1.95;

export type Preset = 'shear' | 'stretch' | 'rotate' | 'reflect' | 'project' | 'free';
export const PRESETS: Preset[] = ['shear', 'stretch', 'rotate', 'reflect', 'project', 'free'];
export const DEFAULTS = { preset: 'shear' as Preset, k: 1, theta: 30 };
/** The matrix a learner starts from when "Your own" is chosen before any tip has ever been moved. */
export const FREE_START: Mat2 = [1.2, 0.5, -0.3, 0.9];

/** Round tiny floating-point residue (cos 90 degrees = 6e-17) so that special matrices show exact zeros and ones. */
const clean = (v: number) => Math.round(v * 1e12) / 1e12 + 0;
const cleanMat = (m: Mat2): Mat2 => [clean(m[0]), clean(m[1]), clean(m[2]), clean(m[3])];

/**
 * The matrix of each named family. With u = (cos t, sin t) and v = (-sin t, cos t) (t is the angle in degrees):
 *   shear    I + k u v^T         moves points along u by k times their distance from the line along u (det = 1)
 *   stretch  I + (k - 1) u u^T   multiplies lengths along u by k and leaves v alone (det = k, so k = 1 is the identity)
 *   rotate   k Rot(t)            rotates by t and scales by k (det = k^2)
 *   reflect  k (2 u u^T - I)     mirrors across the line along u and scales by k (det = -k^2)
 *   project  k u u^T             flattens everything onto the line along u (rank 1; k = 1 is the orthogonal projection)
 * 'free' returns `free`, the matrix the learner dragged into place.
 */
export function presetMatrix(preset: Preset, k: number, thetaDeg: number, free: Mat2 = FREE_START): Mat2 {
  const t = (thetaDeg * Math.PI) / 180, ux = Math.cos(t), uy = Math.sin(t), vx = -uy, vy = ux;
  switch (preset) {
    case 'shear': return cleanMat([1 + k * ux * vx, k * ux * vy, k * uy * vx, 1 + k * uy * vy]);
    case 'stretch': return cleanMat([1 + (k - 1) * ux * ux, (k - 1) * ux * uy, (k - 1) * ux * uy, 1 + (k - 1) * uy * uy]);
    case 'rotate': { const r = rotation2(thetaDeg); return cleanMat([k * r[0], k * r[1], k * r[2], k * r[3]]); }
    case 'reflect': return cleanMat([k * (2 * ux * ux - 1), k * 2 * ux * uy, k * 2 * ux * uy, k * (2 * uy * uy - 1)]);
    case 'project': return cleanMat([k * ux * ux, k * ux * uy, k * ux * uy, k * uy * uy]);
    case 'free': return free;
  }
}
export const isPreset = (s: unknown): s is Preset => typeof s === 'string' && (PRESETS as string[]).includes(s);

export type Orientation = 'kept' | 'mirrored' | 'flat';
export function orientationOf(A: Mat2): Orientation {
  const sv = svd2(A);
  return sv.rank < 2 ? 'flat' : det2(A) > 0 ? 'kept' : 'mirrored';
}
export interface Facts { A: Mat2; det: number; sv: Svd2; orientation: Orientation; squareArea: number }
export function factsOf(A: Mat2): Facts {
  return { A, det: det2(A), sv: svd2(A), orientation: orientationOf(A), squareArea: polygonArea(unitSquareImage(A)) };
}

/** Replace column `which` (0: A e1, 1: A e2) by the point (x, y), clamped to the drawing area and rounded to two decimals. */
export function withTip(A: Mat2, which: 0 | 1, x: number, y: number): Mat2 {
  const cl = (v: number) => Math.round(Math.max(-TIP_LIMIT, Math.min(TIP_LIMIT, v)) * 100) / 100 + 0;
  return which === 0 ? [cl(x), A[1], cl(y), A[3]] : [A[0], cl(x), A[2], cl(y)];
}
/** Which tip is nearest to the plane point (x, y). */
export function nearestTip(A: Mat2, x: number, y: number): 0 | 1 {
  return Math.hypot(A[0] - x, A[2] - y) <= Math.hypot(A[1] - x, A[3] - y) ? 0 : 1;
}

const f2 = (n: number) => fmt(n, 2);
const vec = (x: number, y: number) => `(${f2(x)}, ${f2(y)})`;
export const matrixText = (A: Mat2) => `[[${f2(A[0])}, ${f2(A[1])}], [${f2(A[2])}, ${f2(A[3])}]]`;
export const condText = (sv: Svd2) => (sv.rank < 2 ? '∞' : sv.cond >= 1e4 ? sv.cond.toExponential(1).replace('e+', '×10^') : f2(sv.cond));

/** Plain-language reading of the current matrix (pure, so tests can check it). */
export function describe(lang: Lang, A: Mat2): string {
  const F = factsOf(A), { sv } = F, head = pick(lang,
    `A = ${matrixText(A)} sends e1 to ${vec(A[0], A[2])} and e2 to ${vec(A[1], A[3])}.`,
    `A = ${matrixText(A)} 把 e1 送到 ${vec(A[0], A[2])}，把 e2 送到 ${vec(A[1], A[3])}。`);
  if (sv.rank === 0) return head + pick(lang,
    ' Every entry is zero, so the whole picture lands on the origin: no information is left (rank 0).',
    ' 所有元素都是零，整張影像都落在原點：不剩任何資訊（秩為 0）。');
  if (sv.rank === 1) return head + pick(lang,
    ` The determinant is 0, so the unit square has no area: the whole picture is flattened onto a line, and every point of the picture on a line perpendicular to it lands on the same spot (rank 1, largest singular value σ1 = ${f2(sv.s1)}). Nothing can undo that.`,
    ` 行列式為 0，單位正方形沒有面積：整張影像被壓扁到一條直線上，垂直於這條線的每一條直線上的所有點都落在同一點（秩為 1，最大奇異值 σ1 = ${f2(sv.s1)}）。這是任何矩陣都還原不了的。`);
  const area = f2(Math.abs(F.det)), kappa = condText(sv);
  const side = F.orientation === 'kept'
    ? pick(lang, 'The sign is positive, so the picture keeps its handedness.', '正負號為正，影像的左右手性不變。')
    : pick(lang, 'The sign is negative, so the picture is mirrored: going from Ae1 to Ae2 now turns the other way round.', '正負號為負，所以影像成了鏡像：從 Ae1 轉向 Ae2 的方向反了過來。');
  return head + pick(lang,
    ` The unit square becomes a parallelogram whose area is |det A| = ${area} times as large. ${side} The unit circle becomes an ellipse with semi-axes σ1 = ${f2(sv.s1)} and σ2 = ${f2(sv.s2)}, so the condition number is κ = ${kappa}.`,
    ` 單位正方形變成面積為原來 |det A| = ${area} 倍的平行四邊形。${side}單位圓變成半軸長為 σ1 = ${f2(sv.s1)} 與 σ2 = ${f2(sv.s2)} 的橢圓，所以條件數是 κ = ${kappa}。`);
}
export function metricsOf(lang: Lang, A: Mat2): { label: string; value: string }[] {
  const F = factsOf(A), { sv } = F;
  const rank = sv.rank === 2 ? pick(lang, '2 (full)', '2（滿秩）') : sv.rank === 1 ? pick(lang, '1 (a line)', '1（一條線）') : pick(lang, '0 (a point)', '0（一個點）');
  const orient = F.orientation === 'kept' ? pick(lang, 'kept', '保持') : F.orientation === 'mirrored' ? pick(lang, 'mirrored', '鏡像反轉') : pick(lang, 'none (flat)', '無（已壓扁）');
  return [
    { label: pick(lang, 'det A (signed area scale)', 'det A（帶正負號的面積倍率）'), value: f2(F.det) },
    { label: pick(lang, 'σ1 (largest singular value)', 'σ1（最大奇異值）'), value: f2(sv.s1) },
    { label: pick(lang, 'σ2 (smallest singular value)', 'σ2（最小奇異值）'), value: f2(sv.s2) },
    { label: pick(lang, 'κ = σ1 / σ2 (condition number)', 'κ = σ1 / σ2（條件數）'), value: condText(sv) },
    { label: pick(lang, 'Rank', '秩'), value: rank },
    { label: pick(lang, 'Orientation', '方向'), value: orient },
  ];
}

// ---------------------------------------------------------------------------------------------- drawing (DOM)

const RED = '#ff3b30', YELLOW = '#ffd400', CYAN = '#4fd2ff';
const cx = (x: number) => VIEW / 2 + VIEW_UNIT * x, cy = (y: number) => VIEW / 2 - VIEW_UNIT * y;

function halo(ctx: CanvasRenderingContext2D, color: string, width: number, path: () => void, dash: number[] = []) {
  ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.setLineDash(dash);
  ctx.lineWidth = width + 3; ctx.strokeStyle = 'rgba(0,0,0,0.8)'; ctx.beginPath(); path(); ctx.stroke();
  ctx.lineWidth = width; ctx.strokeStyle = color; ctx.beginPath(); path(); ctx.stroke();
  ctx.setLineDash([]);
}
function arrow(ctx: CanvasRenderingContext2D, x: number, y: number, color: string, ring: boolean, active: boolean) {
  const X = cx(x), Y = cy(y), len = Math.hypot(X - VIEW / 2, Y - VIEW / 2);
  if (len < 1) return;
  halo(ctx, color, 3, () => { ctx.moveTo(VIEW / 2, VIEW / 2); ctx.lineTo(X, Y); });
  const ux = (X - VIEW / 2) / len, uy = (Y - VIEW / 2) / len, hl = Math.min(14, len * 0.6), hw = hl * 0.45;
  const head = () => { ctx.moveTo(X, Y); ctx.lineTo(X - ux * hl - uy * hw, Y - uy * hl + ux * hw); ctx.lineTo(X - ux * hl + uy * hw, Y - uy * hl - ux * hw); ctx.closePath(); };
  halo(ctx, color, 2, head); ctx.fillStyle = color; ctx.beginPath(); head(); ctx.fill();
  if (ring) { halo(ctx, active ? '#ffffff' : 'rgba(255,255,255,0.85)', active ? 3 : 1.6, () => { ctx.moveTo(X + 13, Y); ctx.arc(X, Y, 13, 0, 2 * Math.PI); }, active ? [] : [4, 3]); }
}
function tag(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string) {
  const X = cx(x), Y = cy(y), dx = X - VIEW / 2, dy = Y - VIEW / 2, n = Math.hypot(dx, dy) || 1;
  label(ctx, text, X + (dx / n) * 20, Y + (dy / n) * 20 + 5, { size: 15, bold: true, color, halo: 'rgba(0,0,0,0.85)', align: 'center' });
}
function overlay(ctx: CanvasRenderingContext2D, A: Mat2, o: { names: [string, string]; ring: boolean; active: 0 | 1 | null; svd: boolean }) {
  const sq = unitSquareImage(A);
  ctx.beginPath(); sq.forEach(([x, y], i) => (i ? ctx.lineTo(cx(x), cy(y)) : ctx.moveTo(cx(x), cy(y)))); ctx.closePath();
  ctx.fillStyle = 'rgba(255,255,255,0.14)'; ctx.fill();
  halo(ctx, '#ffffff', 2, () => { sq.forEach(([x, y], i) => (i ? ctx.lineTo(cx(x), cy(y)) : ctx.moveTo(cx(x), cy(y)))); ctx.closePath(); });
  halo(ctx, CYAN, 1.8, () => {
    for (let i = 0; i <= 96; i++) { const [x, y] = ellipsePoint(A, (2 * Math.PI * i) / 96); if (i) ctx.lineTo(cx(x), cy(y)); else ctx.moveTo(cx(x), cy(y)); }
  }, [6, 4]);
  if (o.svd) {
    const sv = svd2(A);
    // When s1 = s2 the image is a circle and its axes are not defined, so they are not drawn.
    if (sv.s1 - sv.s2 > 1e-3 * sv.s1) {
      halo(ctx, CYAN, 1.4, () => { ctx.moveTo(cx(-sv.s1 * sv.u1[0]), cy(-sv.s1 * sv.u1[1])); ctx.lineTo(cx(sv.s1 * sv.u1[0]), cy(sv.s1 * sv.u1[1])); });
      if (sv.s2 > 1e-9) halo(ctx, CYAN, 1.4, () => { ctx.moveTo(cx(-sv.s2 * sv.u2[0]), cy(-sv.s2 * sv.u2[1])); ctx.lineTo(cx(sv.s2 * sv.u2[0]), cy(sv.s2 * sv.u2[1])); });
    }
  }
  arrow(ctx, A[0], A[2], RED, o.ring, o.active === 0);
  arrow(ctx, A[1], A[3], YELLOW, o.ring, o.active === 1);
  tag(ctx, o.names[0], A[0], A[2], RED); tag(ctx, o.names[1], A[1], A[3], YELLOW);
  ctx.beginPath(); ctx.arc(VIEW / 2, VIEW / 2, 3.5, 0, 2 * Math.PI); ctx.fillStyle = '#ffffff'; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = '#000'; ctx.stroke();
}
function putGray(canvas: HTMLCanvasElement, data: ArrayLike<number>, bg?: Warped) {
  const off = document.createElement('canvas');
  off.width = SIZE; off.height = SIZE;
  const octx = off.getContext('2d')!, im = octx.createImageData(SIZE, SIZE), px = im.data;
  for (let i = 0; i < SIZE * SIZE; i++) {
    let v = data[i];
    if (bg) {
      // Light checkerboard where the picture does not reach (alpha 0), so empty space reads as empty and not as black.
      const r = Math.floor(i / SIZE), c = i % SIZE, checker = ((r >> 4) + (c >> 4)) & 1 ? 0.8 : 0.9;
      v = bg.data[i] + (1 - bg.alpha[i]) * checker;
    }
    const b = Math.max(0, Math.min(255, Math.round(v * 255)));
    px[4 * i] = px[4 * i + 1] = px[4 * i + 2] = b; px[4 * i + 3] = 255;
  }
  octx.putImageData(im, 0, 0);
  if (canvas.width !== VIEW) canvas.width = VIEW;
  if (canvas.height !== VIEW) canvas.height = VIEW;
  const ctx = canvas.getContext('2d')!;
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(off, 0, 0, VIEW, VIEW);
}

export const lab: LabDef = {
  kind: 'image', id: 'linear-transform', track: 'foundations',
  title: L('ML 01A · What does a 2 × 2 matrix do to a picture?', 'ML 01A・2 × 2 矩陣會對一張影像做什麼？'),
  description: L(
    'Move the images of the two basis vectors and watch a real photograph follow. The determinant is the signed area scale, the singular values are the stretch along two perpendicular directions, and rank 1 flattens the picture onto a line.',
    '移動兩個基底向量的像，看一張真實照片跟著改變。行列式是帶正負號的面積倍率，奇異值是沿兩個互相垂直方向的伸縮量，秩為 1 時整張影像被壓成一條線。'),
  challenge: L(
    'The default is a shear: det A = 1, so areas are kept, yet σ1 = 1.62 and σ2 = 0.62 (κ = 2.62), and the unit circle becomes an ellipse. Keeping the area does not keep the shape. Now choose Stretch and slide k from 1 down through 0 to −1: at k = 0 the picture collapses onto a line (rank 1, κ = ∞), and for negative k it returns as a mirror image (det A < 0). Last, choose Projection and decide which points of the picture land on the same spot, and whether any matrix could undo that.',
    '預設是一個剪切：det A = 1，所以面積不變，但 σ1 = 1.62、σ2 = 0.62（κ = 2.62），單位圓變成橢圓。面積不變不代表形狀不變。接著選「伸縮」，把 k 從 1 經過 0 滑到 −1：k = 0 時影像塌縮成一條線（秩為 1，κ = ∞），k 為負時它以鏡像的樣子回來（det A < 0）。最後選「投影」，想一想影像中哪些點會落在同一點，以及有沒有任何矩陣能還原。'),
  note: L(
    'Real photograph: skimage.data.camera at 256 × 256 (credit below). The plane origin is the centre of the photo and 64 pixels are one unit, so the photo spans −2 to 2 in x and y, with y pointing up as in a maths figure. The right picture is made by inverse mapping: each output pixel looks up where it came from (the inverse of A applied to its position) with bilinear interpolation, and squeezed regions average several samples to avoid jagged edges; checkerboard marks places the picture does not reach. A rank-1 matrix has no inverse, so it is drawn as a line whose value at each spot is the mean of the photo along the whole line of points that collapsed onto it. Dashed cyan is the image of the unit circle, an ellipse whose semi-axes (solid cyan) have lengths σ1 and σ2; they are left out when σ1 = σ2. Controls: with u = (cos θ, sin θ), Shear is I + k·u·vT (v is u turned by 90°), Stretch is I + (k − 1)·u·uT (k = 1 is the identity), Rotation is k times a rotation by θ, Reflection is k times a mirror across the line along u, Projection is k·u·uT. Because they act along the direction θ, the two sliders move all four entries a, b, c, d. "Your own" has no restriction: drag the red and yellow tips on the right picture, or click it, or focus it and press 1 or 2 to choose an arrow and the arrow keys to move it (Shift for bigger steps). Any matrix with entries between −1.95 and 1.95 in steps of 0.01 is reachable; the sliders are switched off in that mode. Singular values use the closed-form 2 × 2 formula (Blinn), checked against numpy.linalg.svd; a matrix counts as singular when σ2 is below one billionth of σ1. Numbers are rounded to two decimals.',
    '真實照片：skimage.data.camera，256 × 256（來源見下方）。座標原點是照片中心，64 個像素為一個單位，所以照片在 x、y 方向都涵蓋 −2 到 2，y 軸向上，與數學圖形相同。右圖用反向對應產生：每個輸出像素先找出自己來自哪裡（把 A 的反矩陣作用在它的位置上），再以雙線性內插讀值；被壓縮的區域會平均多個取樣點以避免鋸齒；棋盤格表示影像到不了的地方。秩為 1 的矩陣沒有反矩陣，所以畫成一條線，線上每一點的值是照片中所有塌縮到該點的點的平均。青色虛線是單位圓的像，一個橢圓，它的半軸（青色實線）長度是 σ1 與 σ2；當 σ1 = σ2 時不畫半軸。控制項：令 u = (cos θ, sin θ)，剪切是 I + k·u·vT（v 是 u 轉 90°），伸縮是 I + (k − 1)·u·uT（k = 1 為單位矩陣），旋轉是旋轉 θ 再乘以 k，鏡射是對沿 u 方向那條線的鏡射再乘以 k，投影是 k·u·uT。因為它們沿著方向 θ 作用，兩個滑桿會讓 a、b、c、d 四個元素都改變。「自己拖」沒有任何限制：拖曳右圖的紅色與黃色尖端，或直接點一下，或讓圖取得焦點後按 1 或 2 選箭頭、再用方向鍵移動（按住 Shift 步伐較大）。元素介於 −1.95 與 1.95、間隔 0.01 的任何矩陣都能到達；此模式下滑桿停用。奇異值使用 2 × 2 的封閉公式（Blinn），並與 numpy.linalg.svd 比對；當 σ2 小於 σ1 的十億分之一時視為奇異矩陣。數字四捨五入到小數點後兩位。'),
  related: ['ml/ml-01a-linear-algebra'],
  controls: [
    {
      key: 'preset', label: L('Kind of matrix', '矩陣的類型'), kind: 'select', value: DEFAULTS.preset,
      options: [
        { value: 'shear', label: L('Shear along a direction', '沿某方向的剪切') },
        { value: 'stretch', label: L('Stretch along a direction (k = 1 is the identity)', '沿某方向的伸縮（k = 1 為單位矩陣）') },
        { value: 'rotate', label: L('Rotation (times a scale k)', '旋轉（再乘以 k 倍）') },
        { value: 'reflect', label: L('Reflection across a line (times k)', '對一條線的鏡射（再乘以 k 倍）') },
        { value: 'project', label: L('Projection onto a line (rank 1)', '投影到一條線（秩為 1）') },
        { value: 'free', label: L('Your own: drag the two arrow tips', '自己拖：拖曳兩個箭頭尖端') },
      ],
    },
    { key: 'k', label: L('Amount k (scale, stretch or shear)', '量 k（倍率、伸縮或剪切量）'), kind: 'range', value: DEFAULTS.k, min: -2, max: 2, step: 0.1 },
    { key: 'theta', label: L('Angle θ in degrees (rotation angle or direction of the line)', '角度 θ（度；旋轉角或那條線的方向）'), kind: 'range', value: DEFAULTS.theta, min: -180, max: 180, step: 5 },
  ],
  predict: {
    question: L(
      'The matrix A = [[0, 1], [1, 0]] sends e1 to (0, 1) and e2 to (1, 0): it swaps x and y. What does it do to the photograph?',
      '矩陣 A = [[0, 1], [1, 0]] 把 e1 送到 (0, 1)、把 e2 送到 (1, 0)：它把 x 與 y 對調。它對這張照片做了什麼？'),
    options: [
      { value: 'mirror', label: L('It mirrors the photo across the diagonal line from bottom left to top right', '沿著由左下到右上的對角線把照片做鏡射') },
      { value: 'rotate', label: L('It rotates the photo by 90 degrees', '把照片旋轉 90 度') },
      { value: 'line', label: L('It flattens the photo onto a line', '把照片壓扁成一條線') },
      { value: 'same', label: L('Nothing visible: the area is kept, so the picture stays as it is', '看不出變化：面積不變，所以照片維持原樣') },
    ],
    answer: 'mirror',
    explain: L(
      'det A = 0·0 − 1·1 = −1. The absolute value 1 says areas are kept (both singular values are 1, so lengths are kept too), but the minus sign says orientation is reversed, which only a mirror does: going from Ae1 = (0, 1) to Ae2 = (1, 0) turns clockwise, while going from e1 to e2 turns counter-clockwise. A rotation always has det A = +1, and a flattening would have det A = 0. Try it: choose Reflection, k = 1 and θ = 45°.',
      'det A = 0·0 − 1·1 = −1。絕對值 1 表示面積不變（兩個奇異值都是 1，所以長度也不變），但負號表示方向被反轉，只有鏡射會這樣：從 Ae1 = (0, 1) 轉到 Ae2 = (1, 0) 是順時針，而從 e1 轉到 e2 是逆時針。旋轉的 det A 一定是 +1，壓扁的 det A 則是 0。試試看：選「鏡射」，k = 1、θ = 45°。'),
  },
  assets: { 'camera-256': 'gray' },
  paneMin: 280,
  panes: [
    { key: 'original', kind: 'canvas', label: L('Original photograph with the unit square, e1 (red), e2 (yellow) and the unit circle (dashed)', '原始照片，以及單位正方形、e1（紅）、e2（黃）與單位圓（虛線）'), width: VIEW, height: VIEW },
    { key: 'warped', kind: 'canvas', label: L('The photograph after A: the unit square, Ae1 (red), Ae2 (yellow) and the ellipse. Drag the round tips, or click, or focus and use keys 1, 2 and the arrow keys', '經過 A 之後的照片：單位正方形、Ae1（紅）、Ae2（黃）與橢圓。拖曳圓形尖端、點一下，或取得焦點後用 1、2 與方向鍵'), width: VIEW, height: VIEW },
  ],
  sources: [
    { title: 'Gilbert Strang, Introduction to Linear Algebra, 6th ed., Wellesley-Cambridge Press, 2023 (determinant as signed area, SVD, rank)', url: 'https://math.mit.edu/~gs/linearalgebra/ila6/indexila6.html' },
    { title: 'Jim Blinn, Consider the Lowly 2 × 2 Matrix, IEEE Computer Graphics and Applications 16(2), 82–88, 1996 (closed-form 2 × 2 SVD)', url: 'https://doi.org/10.1109/38.486688' },
    { title: 'SciPy documentation, scipy.ndimage.affine_transform (inverse mapping with bilinear interpolation; the reference for the warp)', url: 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html' },
  ],
  create({ lang, images }) {
    const photo = images['camera-256'] as Gray;
    let mem: Mat2 = FREE_START, hold = false, prevPreset: Preset | null = null, lastA: Mat2 = presetMatrix(DEFAULTS.preset, DEFAULTS.k, DEFAULTS.theta);
    let cached: { key: string; warped: Warped } | undefined, active: 0 | 1 = 0, focused = false;
    const warped = (A: Mat2): Warped => {
      const key = A.join(',');
      if (!cached || cached.key !== key) cached = { key, warped: warpMatrix(photo, A, { frame: FRAME, mode: 'grid-constant' }) };
      return cached.warped;
    };
    const paint = (A: Mat2, canvases: Record<string, HTMLCanvasElement>) => {
      putGray(canvases.original, photo.data);
      overlay(canvases.original.getContext('2d')!, [1, 0, 0, 1], { names: ['e1', 'e2'], ring: false, active: null, svd: false });
      putGray(canvases.warped, [], warped(A));
      overlay(canvases.warped.getContext('2d')!, A, { names: ['Ae1', 'Ae2'], ring: true, active: focused ? active : null, svd: true });
      canvases.warped.setAttribute('aria-label', pick(lang, `Warped photograph. A = ${matrixText(A)}. Arrow keys move the selected arrow tip; press 1 or 2 to choose e1 or e2.`, `變換後的照片。A = ${matrixText(A)}。方向鍵移動所選的箭頭尖端；按 1 或 2 選擇 e1 或 e2。`));
    };
    return {
      attach(canvases, setControl) {
        const cv = canvases.warped;
        cv.style.cursor = 'grab'; cv.tabIndex = 0; cv.setAttribute('role', 'application');
        // lastA is updated at once so that several key presses or pointer moves before the next frame build on each other.
        const commit = (A: Mat2) => { mem = A; lastA = A; hold = true; setControl('preset', 'free'); };
        const plane = (e: PointerEvent): [number, number] => {
          const r = cv.getBoundingClientRect();
          return [(((e.clientX - r.left) / r.width) * VIEW - VIEW / 2) / VIEW_UNIT, -(((e.clientY - r.top) / r.height) * VIEW - VIEW / 2) / VIEW_UNIT];
        };
        let dragging = false;
        const move = (e: PointerEvent) => { const [x, y] = plane(e); commit(withTip(lastA, active, x, y)); };
        cv.addEventListener('pointerdown', e => {
          if (e.button > 0) return;
          const [x, y] = plane(e);
          if (e.pointerType === 'touch') return; // a touch is a tap (handled on pointerup) or the start of a page scroll
          active = nearestTip(lastA, x, y); dragging = true; cv.setPointerCapture(e.pointerId); cv.style.cursor = 'grabbing'; move(e);
        });
        cv.addEventListener('pointermove', e => { if (dragging && e.pointerType !== 'touch' && (e.buttons & 1)) move(e); });
        const end = () => { dragging = false; cv.style.cursor = 'grab'; };
        cv.addEventListener('pointerup', e => {
          if (e.pointerType === 'touch') { const [x, y] = plane(e); active = nearestTip(lastA, x, y); move(e); }
          end();
        });
        cv.addEventListener('pointercancel', end);
        cv.addEventListener('keydown', e => {
          const step = e.shiftKey ? 0.25 : 0.05;
          if (e.key === '1' || e.key === '2') { active = e.key === '1' ? 0 : 1; paint(lastA, canvases); e.preventDefault(); return; }
          const d: Record<string, [number, number]> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] };
          if (!d[e.key]) return;
          e.preventDefault();
          const tip: [number, number] = active === 0 ? [lastA[0], lastA[2]] : [lastA[1], lastA[3]];
          commit(withTip(lastA, active, tip[0] + d[e.key][0], tip[1] + d[e.key][1]));
        });
        cv.addEventListener('focus', () => { focused = true; paint(lastA, canvases); });
        cv.addEventListener('blur', () => { focused = false; paint(lastA, canvases); });
      },
      render(state: State, canvases) {
        const preset: Preset = isPreset(state.preset) ? state.preset : DEFAULTS.preset;
        // "Your own" starts from the matrix on screen when it is picked from the list; dragging stores its own result first.
        if (preset === 'free' && prevPreset !== null && prevPreset !== 'free' && !hold) mem = lastA;
        hold = false;
        const A = presetMatrix(preset, num(state, 'k'), num(state, 'theta'), mem);
        prevPreset = preset; lastA = A;
        paint(A, canvases);
        const root = canvases.warped.closest?.('image-lab');
        root?.querySelectorAll<HTMLInputElement>('input[data-key="k"], input[data-key="theta"]').forEach(el => { el.disabled = preset === 'free'; });
        return {
          summary: describe(lang, A) + (preset === 'free' ? pick(lang, ' (Your own matrix: the sliders are off; drag the tips to change it.)', '（自己拖的矩陣：滑桿已停用；拖曳尖端即可修改。）') : ''),
          metrics: metricsOf(lang, A),
        };
      },
    };
  },
};
