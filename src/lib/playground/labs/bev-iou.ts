import type { LabDef } from './types.ts';
import { L, pick, fmt, num, text } from '../types.ts';
import type { Lang } from '../../../i18n/ui';

// ---------------------------------------------------------------------------------------------- geometry (the chapter's code, ported)
// A bird's-eye-view (BEV) box is (centre x, centre y, length l along its heading, width w, yaw). Corners are listed counter-clockwise,
// the polygon of the overlap comes from Sutherland-Hodgman clipping (each edge line of the second box cuts the first polygon in
// turn) and its area from the shoelace formula. IoU = overlap / (area A + area B - overlap).
export interface Box { x: number; y: number; l: number; w: number; yaw: number }
export type Poly = [number, number][];

export function corners(b: Box): Poly {
  const c = Math.cos(b.yaw), s = Math.sin(b.yaw);
  return ([[b.l, b.w], [-b.l, b.w], [-b.l, -b.w], [b.l, -b.w]] as Poly).map(([u, v]) => [b.x + (u * c - v * s) / 2, b.y + (u * s + v * c) / 2] as [number, number]);
}
function crossPoint(p: [number, number], q: [number, number], a: [number, number], b: [number, number]): [number, number] {
  const d1x = q[0] - p[0], d1y = q[1] - p[1], d2x = b[0] - a[0], d2y = b[1] - a[1];
  const t = ((a[0] - p[0]) * d2y - (a[1] - p[1]) * d2x) / (d1x * d2y - d1y * d2x);
  return [p[0] + t * d1x, p[1] + t * d1y];
}
/** The part of the polygon `subject` inside the convex counter-clockwise polygon `clipper`. */
export function clip(subject: Poly, clipper: Poly): Poly {
  let out = subject;
  for (let i = 0; i < clipper.length && out.length; i++) {
    const a = clipper[i], b = clipper[(i + 1) % clipper.length];
    const inside = (p: [number, number]) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0;
    const pts = out; out = [];
    for (let j = 0; j < pts.length; j++) {
      const p = pts[j], q = pts[(j + 1) % pts.length];
      if (inside(q)) { if (!inside(p)) out.push(crossPoint(p, q, a, b)); out.push(q); }
      else if (inside(p)) out.push(crossPoint(p, q, a, b));
    }
  }
  return out;
}
/** Shoelace formula. */
export function area(p: Poly): number {
  if (p.length < 3) return 0;
  let t = 0;
  for (let i = 0; i < p.length; i++) { const [x0, y0] = p[i], [x1, y1] = p[(i + 1) % p.length]; t += x0 * y1 - x1 * y0; }
  return Math.abs(t) / 2;
}
export function overlapPolygon(a: Box, b: Box): Poly { return clip(corners(a), corners(b)); }
export function rotatedIou(a: Box, b: Box): number {
  const inter = area(overlapPolygon(a, b)), union = a.l * a.w + b.l * b.w - inter;
  return union > 0 ? inter / union : 0;
}
/** Axis-aligned enclosing rectangle [x0, x1, y0, y1] of a box. */
export function enclosing(b: Box): [number, number, number, number] {
  const c = corners(b), xs = c.map(p => p[0]), ys = c.map(p => p[1]);
  return [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
}
/** IoU of the two boxes' enclosing axis-aligned rectangles (the shortcut that ignores yaw). */
export function enclosingIou(a: Box, b: Box): number {
  const r = enclosing(a), s = enclosing(b);
  const iw = Math.max(0, Math.min(r[1], s[1]) - Math.max(r[0], s[0])), ih = Math.max(0, Math.min(r[3], s[3]) - Math.max(r[2], s[2])), inter = iw * ih;
  const union = (r[1] - r[0]) * (r[3] - r[2]) + (s[1] - s[0]) * (s[3] - s[2]) - inter;
  return union > 0 ? inter / union : 0;
}

// ---------------------------------------------------------------------------------------------- the scene
/** Illustrative sizes in metres (length, width), the same as the chapter's Figure 4 and Figure 6: not measurements of any dataset. */
export const SHAPES = { pedestrian: { l: 0.7, w: 0.7 }, car: { l: 4.5, w: 1.8 }, bus: { l: 12, w: 2.5 } } as const;
export type Shape = keyof typeof SHAPES;
export const SHAPE_KEYS = Object.keys(SHAPES) as Shape[];
export const DEFAULTS = { yaw: 10, shape: 'bus' as Shape, offset: 0 };
export const THRESHOLD = 0.7, MAX_OFFSET = 2;
const rad = (deg: number) => (deg * Math.PI) / 180;

/** Ground truth: axis-aligned at the origin. Prediction: right size, yawed by `yawDeg` about its own centre, centre moved `offset` metres along the ground truth's length. */
export function makeBoxes(shape: Shape, yawDeg: number, offset: number): { gt: Box; pred: Box } {
  const { l, w } = SHAPES[shape];
  return { gt: { x: 0, y: 0, l, w, yaw: 0 }, pred: { x: offset, y: 0, l, w, yaw: rad(yawDeg) } };
}
export interface Scene { gt: Box; pred: Box; overlap: Poly; iou: number; aabbIou: number; centreDistance: number }
export function evaluate(shape: Shape, yawDeg: number, offset: number): Scene {
  const { gt, pred } = makeBoxes(shape, yawDeg, offset);
  return { gt, pred, overlap: overlapPolygon(gt, pred), iou: rotatedIou(gt, pred), aabbIou: enclosingIou(gt, pred), centreDistance: Math.hypot(pred.x - gt.x, pred.y - gt.y) };
}
export const iouAt = (shape: Shape, yawDeg: number, offset: number) => { const { gt, pred } = makeBoxes(shape, yawDeg, offset); return rotatedIou(gt, pred); };
/** Smallest yaw error (degrees, 0.05 steps up to 90) at which the rotated IoU is below `threshold`; null if it never is. 0 if it already is at no yaw error. */
export function yawBelow(shape: Shape, offset: number, threshold = THRESHOLD): number | null {
  for (let d = 0; d <= 90; d += 0.05) if (iouAt(shape, d, offset) < threshold) return Math.round(d * 100) / 100;
  return null;
}
export function curve(shape: Shape, offset: number, step = 1): [number, number][] {
  const out: [number, number][] = [];
  for (let d = 0; d <= 90; d += step) out.push([d, iouAt(shape, d, offset)]);
  return out;
}

// ---------------------------------------------------------------------------------------------- drawing (SVG)
const SC_W = 360, SC_H = 270, CV_W = 360, CV_H = 290;
const INK = 'fill:var(--plot-ink)', MUTED = 'fill:var(--plot-muted)';
const A = 'var(--plot-a)', B = 'var(--plot-b)', C = 'var(--plot-c)';
const shapeName = (lang: Lang, s: Shape) => pick(lang, s === 'pedestrian' ? 'Pedestrian' : s === 'car' ? 'Car' : 'Bus or truck', s === 'pedestrian' ? '行人' : s === 'car' ? '汽車' : '巴士或卡車');
const shortName = (lang: Lang, s: Shape) => pick(lang, s === 'pedestrian' ? 'Pedestrian' : s === 'car' ? 'Car' : 'Bus', s === 'pedestrian' ? '行人' : s === 'car' ? '汽車' : '巴士');
const shapeSize = (s: Shape) => `${SHAPES[s].l} × ${SHAPES[s].w} m`;
const t = (x: number, y: number, s: string | number, style: string, size = 13, anchor = 'start', weight = 400) =>
  text(x, y, s, `style="${style};font-size:${size}px;font-weight:${weight}" text-anchor="${anchor}"`);
const pathOf = (p: Poly, f: (x: number, y: number) => [number, number]) => p.map(([x, y], i) => { const [sx, sy] = f(x, y); return `${i ? 'L' : 'M'}${sx.toFixed(1)},${sy.toFixed(1)}`; }).join(' ') + ' Z';

function sceneSvg(lang: Lang, shape: Shape, s: Scene): string {
  const { l } = SHAPES[shape], half = l / 2 + MAX_OFFSET + 1, scale = SC_W / (2 * half), cx = SC_W / 2, cy = SC_H / 2 + 14;
  const P = (x: number, y: number): [number, number] => [cx + x * scale, cy - y * scale];
  const arrow = (b: Box, colour: string) => { // heading arrow from the centre to just past the front edge
    const c = Math.cos(b.yaw), sn = Math.sin(b.yaw), len = b.l * 0.5, [x0, y0] = P(b.x, b.y), [x1, y1] = P(b.x + len * c, b.y + len * sn);
    const ang = Math.atan2(y1 - y0, x1 - x0), h = 7;
    return `<path d="M${x0.toFixed(1)},${y0.toFixed(1)} L${x1.toFixed(1)},${y1.toFixed(1)} M${x1.toFixed(1)},${y1.toFixed(1)} l${(-h * Math.cos(ang - 0.5)).toFixed(1)},${(-h * Math.sin(ang - 0.5)).toFixed(1)} M${x1.toFixed(1)},${y1.toFixed(1)} l${(-h * Math.cos(ang + 0.5)).toFixed(1)},${(-h * Math.sin(ang + 0.5)).toFixed(1)}" style="fill:none;stroke:${colour};stroke-width:2"/>`;
  };
  const [ex0, ex1, ey0, ey1] = enclosing(s.pred), [rx0, ry0] = P(ex0, ey1), [rx1, ry1] = P(ex1, ey0);
  // Scale bar: the largest of 0.5, 1, 2, 5, 10 m not longer than a quarter of the picture.
  const bar = [0.5, 1, 2, 5, 10].filter(m => m * scale <= SC_W / 4).pop() ?? 0.5;
  const body = [
    `<rect x="0" y="0" width="${SC_W}" height="${SC_H}" style="fill:var(--plot-bg)"/>`,
    `<path d="${pathOf(s.overlap, P)}" style="fill:${C};fill-opacity:0.45;stroke:none"/>`,
    `<rect x="${rx0.toFixed(1)}" y="${ry0.toFixed(1)}" width="${(rx1 - rx0).toFixed(1)}" height="${(ry1 - ry0).toFixed(1)}" style="fill:none;stroke:${B};stroke-width:1.4;stroke-dasharray:2 4"/>`,
    `<path d="${pathOf(corners(s.gt), P)}" style="fill:${A};fill-opacity:0.10;stroke:${A};stroke-width:2.6"/>`,
    `<path d="${pathOf(corners(s.pred), P)}" style="fill:${B};fill-opacity:0.10;stroke:${B};stroke-width:2.6;stroke-dasharray:7 4"/>`,
    arrow(s.gt, A), arrow(s.pred, B),
    `<circle cx="${P(0, 0)[0]}" cy="${P(0, 0)[1]}" r="3.4" style="fill:${A}"/>`,
    `<circle cx="${P(s.pred.x, s.pred.y)[0].toFixed(1)}" cy="${P(s.pred.x, s.pred.y)[1].toFixed(1)}" r="3.4" style="fill:${B}"/>`,
    // legend
    `<path d="M10,14 h18" style="stroke:${A};stroke-width:3"/>`, t(34, 18, pick(lang, 'ground truth', '真實框'), INK),
    `<path d="M10,34 h18" style="stroke:${B};stroke-width:3;stroke-dasharray:6 3"/>`, t(34, 38, pick(lang, 'prediction', '預測框'), INK),
    `<rect x="190" y="8" width="18" height="10" style="fill:${C};fill-opacity:0.45"/>`, t(214, 18, pick(lang, 'overlap', '重疊'), INK),
    `<path d="M190,34 h18" style="stroke:${B};stroke-width:1.6;stroke-dasharray:2 4"/>`, t(214, 38, pick(lang, 'enclosing rect.', '外接矩形'), INK),
    // scale bar and readout
    `<path d="M10,${SC_H - 40} h${(bar * scale).toFixed(1)} M10,${SC_H - 44} v8 M${(10 + bar * scale).toFixed(1)},${SC_H - 44} v8" style="fill:none;stroke:var(--plot-ink);stroke-width:2"/>`,
    t(10 + bar * scale + 8, SC_H - 35, `${bar} m`, INK),
    t(SC_W / 2, SC_H - 12, pick(lang, `IoU ${fmt(s.iou, 3)} · centre distance ${fmt(s.centreDistance, 2)} m`, `IoU ${fmt(s.iou, 3)}・中心距離 ${fmt(s.centreDistance, 2)} 公尺`), INK, 15, 'middle', 600),
  ].join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${SC_W} ${SC_H}" role="img" aria-hidden="true" style="min-width:0;max-width:460px;margin:0 auto">${body}</svg>`;
}

function curveSvg(lang: Lang, shape: Shape, yaw: number, offset: number): string {
  const x0 = 44, x1 = CV_W - 84, y0 = CV_H - 52, y1 = 36, X = (d: number) => x0 + (d / 90) * (x1 - x0), Y = (v: number) => y0 - v * (y0 - y1);
  const colours: Record<Shape, string> = { pedestrian: C, car: A, bus: B };
  const parts: string[] = [`<rect x="0" y="0" width="${CV_W}" height="${CV_H}" style="fill:var(--plot-bg)"/>`];
  parts.push(t(8, 20, pick(lang, 'BEV IoU against yaw error', 'BEV IoU 與偏航角誤差'), INK, 15, 'start', 600));
  for (const v of [0, 0.25, 0.5, 0.75, 1]) parts.push(`<path d="M${x0},${Y(v).toFixed(1)} H${x1 + 4}" style="stroke:var(--plot-grid);stroke-width:1"/>`, t(x0 - 6, Y(v) + 4, v === 0 || v === 1 ? v : v.toFixed(2), MUTED, 12, 'end'));
  for (let d = 0; d <= 90; d += 15) parts.push(`<path d="M${X(d).toFixed(1)},${y0} v5" style="stroke:var(--plot-muted)"/>`, t(X(d), y0 + 20, `${d}°`, MUTED, 12.5, 'middle'));
  parts.push(t((x0 + x1) / 2, y0 + 40, pick(lang, `yaw error (centre offset ${fmt(offset, 1)} m)`, `偏航角誤差（中心偏移 ${fmt(offset, 1)} 公尺）`), MUTED, 13, 'middle'));
  parts.push(`<path d="M${x0},${Y(THRESHOLD).toFixed(1)} H${x1 + 4}" style="stroke:var(--plot-ink);stroke-width:1.2;stroke-dasharray:3 4"/>`, t(x1, Y(THRESHOLD) + 16, `IoU ${THRESHOLD}`, INK, 12.5, 'end'));
  const labelY: { shape: Shape; y: number }[] = [];
  for (const sh of SHAPE_KEYS) {
    const pts = curve(sh, offset, 1), sel = sh === shape;
    parts.push(`<path d="${pts.map(([d, v], i) => `${i ? 'L' : 'M'}${X(d).toFixed(1)},${Y(v).toFixed(1)}`).join(' ')}" style="fill:none;stroke:${colours[sh]};stroke-width:${sel ? 3.6 : 1.8};stroke-opacity:${sel ? 1 : 0.75}"/>`);
    labelY.push({ shape: sh, y: Y(pts[pts.length - 1][1]) });
  }
  labelY.sort((p, q) => p.y - q.y);
  for (let i = 1; i < labelY.length; i++) if (labelY[i].y - labelY[i - 1].y < 15) labelY[i].y = labelY[i - 1].y + 15;
  for (const { shape: sh, y } of labelY) parts.push(t(x1 + 10, y + 4, shortName(lang, sh), `fill:${colours[sh]}`, 13, 'start', sh === shape ? 700 : 400));
  const v = iouAt(shape, yaw, offset), px = X(yaw), py = Y(v);
  parts.push(`<path d="M${px.toFixed(1)},${y1 - 6} V${y0}" style="stroke:var(--plot-ink);stroke-width:1.4;stroke-dasharray:4 3"/>`, `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="6" style="fill:${colours[shape]};stroke:var(--plot-ink);stroke-width:2"/>`);
  parts.push(t(Math.min(px + 10, x1 - 4), Math.max(py - 10, y1 + 6), `${fmt(v, 3)}`, INK, 14, px + 10 > x1 - 50 ? 'end' : 'start', 700));
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${CV_W} ${CV_H}" role="img" aria-hidden="true" style="min-width:0;max-width:460px;margin:0 auto">${parts.join('')}</svg>`;
}

// ---------------------------------------------------------------------------------------------- the lab
export const lab: LabDef = {
  kind: 'image', id: 'bev-iou', track: 'deep-learning',
  title: L('DLCV 03B · Is a small yaw error harmless?', 'DLCV 03B・小小的偏航角誤差無傷大雅嗎？'),
  description: L('Turn a predicted box a few degrees away from the ground truth in the bird’s-eye view and watch the exact rotated IoU fall while the centre distance stays at zero.', '在鳥瞰視角中，把預測框從真實框轉開幾度，觀察精確的旋轉 IoU 下降，而中心距離維持為零。'),
  challenge: L(
    'At the default setting (bus or truck, 12 × 2.5 m, yaw error 10°, centres identical) the rotated IoU is 0.657 although the centre distance is 0 m; the enclosing rectangles, which ignore the yaw, give 0.539. Lower the yaw error to 5° and the IoU is still only 0.808. Switch to the car (4.5 × 1.8 m) and the same 10° costs less, 0.796, then to the pedestrian (0.7 × 0.7 m): 0.863 at 10°, and never below 0.707 at any yaw error. The last row of the table gives the yaw error at which each shape falls below IoU 0.7: 8.5° for the bus and 16.55° for the car. Which class would a matching rule that uses an IoU threshold of 0.7 punish most for a heading mistake, and what does a rule that only looks at centre distance say about all three?',
    '預設設定（巴士或卡車，12 × 2.5 公尺，偏航角誤差 10°，兩個中心完全重合）下，旋轉 IoU 只有 0.657，儘管中心距離是 0 公尺；不管偏航角的外接矩形則給出 0.539。把偏航角誤差降到 5°，IoU 仍然只有 0.808。換成汽車（4.5 × 1.8 公尺），同樣的 10° 損失較小，為 0.796；再換成行人（0.7 × 0.7 公尺）：10° 時是 0.863，而且在任何偏航角誤差下都不會低於 0.707。表格的最後一列給出各形狀的 IoU 掉到 0.7 以下所需的偏航角誤差：巴士 8.5°，汽車 16.55°。以 0.7 為 IoU 門檻的配對規則，會讓哪一類物體因為朝向錯誤而受到最重的懲罰？而只看中心距離的規則，對這三類又怎麼說？'),
  note: L(
    'The ground-truth box is axis-aligned at the origin with its length along x. The prediction has exactly the right size and is turned by the yaw error about its own centre, then moved along the ground truth’s length by the centre offset. Sizes are illustrative, the ones used in the chapter’s Figures 4 and 6 (pedestrian 0.7 × 0.7 m, car 4.5 × 1.8 m, bus or truck 12 × 2.5 m), not measurements of any dataset. Rotated IoU is exact: the overlap polygon comes from Sutherland–Hodgman clipping and its area from the shoelace formula, and IoU = overlap / (area of the ground truth + area of the prediction − overlap); the code is the chapter’s Python snippet ported to TypeScript and tested against it. The axis-aligned IoU takes the smallest axis-aligned rectangle around each box and computes the ordinary IoU of those two rectangles; it ignores the real shape, so it is lower than the rotated IoU for small yaw errors (the enclosing rectangle of a turned box is larger than the box) and higher for large ones. The centre distance is the Euclidean distance between the two centres on the ground plane. IoU cannot see a 180° flip of the heading, so the yaw error is limited to 0° to 90°. Only the bird’s-eye view is compared here (no height). The nuScenes benchmark matches a detection to an object by the 2D centre distance on the ground plane, to decouple detection from object size and orientation, and scores heading in a separate error term; KITTI’s car threshold of 0.7 is a 3D box overlap.',
    '真實框是軸對齊、位於原點的矩形，長邊沿著 x 軸。預測框的尺寸完全正確，先繞自己的中心轉過偏航角誤差，再沿著真實框的長邊方向平移中心偏移量。尺寸僅為示意，採用章節圖 4 與圖 6 的數值（行人 0.7 × 0.7 公尺、汽車 4.5 × 1.8 公尺、巴士或卡車 12 × 2.5 公尺），並非任何資料集的量測值。旋轉 IoU 是精確的：重疊多邊形由 Sutherland–Hodgman 裁切法求出，面積用鞋帶公式，IoU = 重疊 /（真實框面積 + 預測框面積 − 重疊）；程式碼是把章節的 Python 片段移植到 TypeScript，並與它對照測試。軸對齊 IoU 是取每個框最小的軸對齊外接矩形，再計算這兩個矩形的普通 IoU；它忽略真實形狀，所以在偏航角誤差小時比旋轉 IoU 低（轉過的框的外接矩形比框本身大），在誤差大時則比較高。中心距離是兩個中心在地面上的歐氏距離。IoU 看不出朝向轉了 180°，所以偏航角誤差限制在 0° 到 90°。這裡只比較鳥瞰視角（不含高度）。nuScenes 以地面上的二維中心距離把偵測框配對到物體，目的是讓偵測與物體大小和朝向脫鉤，朝向另以獨立的誤差項評分；KITTI 對汽車的門檻 0.7 是三維框的重疊。'),
  related: ['deep-learning/dl-09-3d-object-detection'],
  controls: [
    { key: 'yaw', label: L('Yaw error of the prediction (degrees)', '預測框的偏航角誤差（度）'), kind: 'range', value: DEFAULTS.yaw, min: 0, max: 90, step: 1 },
    {
      key: 'shape', label: L('Object shape (illustrative sizes)', '物體形狀（示意尺寸）'), kind: 'select', value: DEFAULTS.shape, options: SHAPE_KEYS.map(k => ({
        value: k, label: L(`${shapeName('en', k)} (${shapeSize(k)})`, `${shapeName('zh-tw', k)}（${shapeSize(k).replace(' m', ' 公尺')}）`),
      })),
    },
    { key: 'offset', label: L('Centre offset along the length (m)', '沿長邊方向的中心偏移（公尺）'), kind: 'range', value: DEFAULTS.offset, min: 0, max: MAX_OFFSET, step: 0.1 },
  ],
  predict: {
    question: L(
      'A detector predicts a bus-sized box (12 × 2.5 m) with exactly the right centre and size, but its heading is 10° off. What is the bird’s-eye-view IoU with the ground truth?',
      '偵測器預測了一個巴士大小（12 × 2.5 公尺）的框，中心與尺寸都完全正確，但朝向偏了 10°。它與真實框的鳥瞰視角 IoU 是多少？'),
    options: [
      { value: 'high', label: L('About 0.9: 10° is a tiny error', '約 0.9：10° 是很小的誤差') },
      { value: 'mid', label: L('About 0.65', '約 0.65') },
      { value: 'low', label: L('About 0.3', '約 0.3') },
    ],
    answer: 'mid',
    explain: L(
      'About 0.65. The 12 m box is so long that turning it by 10° swings each end about 1 m sideways, while the box is only 2.5 m wide, so a good part of the overlap is lost: the exact IoU is 0.657 (0.808 at 5°, 0.423 at 20°), although the centre distance is 0 m. This bus-sized box falls below IoU 0.7 at a yaw error of 8.5°, a car-sized one at 16.55°, and a square pedestrian box never does.',
      '約 0.65。這個框長 12 公尺，轉 10° 會讓兩端各往側邊擺動約 1 公尺，但框只有 2.5 公尺寬，所以損失了相當一部分重疊：精確的 IoU 是 0.657（5° 時 0.808，20° 時 0.423），而中心距離卻是 0 公尺。這個巴士大小的框在偏航角誤差 8.5° 時就低於 IoU 0.7，汽車大小的框是 16.55°，方形的行人框則永遠不會。'),
  },
  assets: {},
  paneMin: 330,
  panes: [
    { key: 'scene', kind: 'svg', label: L('Bird’s-eye view: ground-truth box, predicted box and their exact overlap', '鳥瞰視角：真實框、預測框與它們的精確重疊'), width: SC_W, height: SC_H },
    { key: 'curve', kind: 'svg', label: L('Rotated IoU against yaw error for the three shapes, at the chosen centre offset', '三種形狀在所選中心偏移下，旋轉 IoU 與偏航角誤差的關係'), width: CV_W, height: CV_H },
  ],
  sources: [
    { title: 'I. E. Sutherland and G. W. Hodgman, Reentrant Polygon Clipping, Communications of the ACM, 17(1), 32-42, 1974 (the polygon clipping used for the exact overlap)', url: 'https://doi.org/10.1145/360767.360802' },
    { title: 'H. Caesar et al., nuScenes: A Multimodal Dataset for Autonomous Driving, CVPR, 2020 (matching by 2D centre distance on the ground plane instead of IoU)', url: 'https://openaccess.thecvf.com/content_CVPR_2020/html/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.html' },
    { title: 'nuScenes devkit, nuScenes detection task (official evaluation documentation: centre-distance matching, orientation error)', url: 'https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md' },
    { title: 'A. Geiger, P. Lenz, C. Stiller and R. Urtasun, The KITTI Vision Benchmark Suite: 3D object detection evaluation (3D box overlap of 70% for cars)', url: 'https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d' },
  ],
  create({ lang }) {
    return {
      render(state) {
        const yaw = Math.max(0, Math.min(90, num(state, 'yaw'))), offset = Math.max(0, Math.min(MAX_OFFSET, num(state, 'offset')));
        const shape: Shape = (SHAPE_KEYS as string[]).includes(String(state.shape)) ? (state.shape as Shape) : 'bus';
        const s = evaluate(shape, yaw, offset), below = yawBelow(shape, offset);
        const belowText = below === null ? pick(lang, 'never', '從不') : below === 0 ? pick(lang, 'already below at 0°', '在 0° 時就已低於') : `${fmt(below, 2)}°`;
        return {
          summary: pick(lang,
            `${shapeName(lang, shape)}, yaw error ${yaw}°, centre offset ${fmt(offset, 1)} m: the rotated IoU is ${fmt(s.iou, 3)}, the IoU of the enclosing rectangles is ${fmt(s.aabbIou, 3)} and the centre distance is ${fmt(s.centreDistance, 2)} m.`,
            `${shapeName(lang, shape)}，偏航角誤差 ${yaw}°，中心偏移 ${fmt(offset, 1)} 公尺：旋轉 IoU 為 ${fmt(s.iou, 3)}，外接矩形的 IoU 為 ${fmt(s.aabbIou, 3)}，中心距離為 ${fmt(s.centreDistance, 2)} 公尺。`),
          metrics: [
            { label: pick(lang, 'Rotated BEV IoU (exact)', '旋轉 BEV IoU（精確）'), value: fmt(s.iou, 3) },
            { label: pick(lang, 'IoU of the enclosing rectangles', '外接矩形的 IoU'), value: fmt(s.aabbIou, 3) },
            { label: pick(lang, 'Centre distance', '中心距離'), value: `${fmt(s.centreDistance, 2)} m` },
            { label: pick(lang, `Yaw error that drops IoU below ${THRESHOLD} (this shape and offset)`, `使 IoU 低於 ${THRESHOLD} 的偏航角誤差（此形狀與偏移）`), value: belowText },
          ],
          plots: { scene: sceneSvg(lang, shape, s), curve: curveSvg(lang, shape, yaw, offset) },
        };
      },
    };
  },
};
