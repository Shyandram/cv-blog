// Facts for the `bev-iou` lab (DLCV 09): the exact rotated IoU (Sutherland-Hodgman clipping + shoelace) against hand-computed cases,
// an independent point-sampling estimate, the chapter's NumPy snippet (scripts/tests/golden/bev-iou.json, written by
// scripts/figures/playground_goldens_bev.py), and the default state of the lab.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  lab, corners, clip, area, rotatedIou, enclosingIou, overlapPolygon, makeBoxes, evaluate, iouAt, yawBelow, curve, SHAPES, SHAPE_KEYS, DEFAULTS, type Box,
} from '../../src/lib/playground/labs/bev-iou.ts';

const G = JSON.parse(readFileSync(new URL('./golden/bev-iou.json', import.meta.url), 'utf8'));
const near = (a: number, b: number, tol: number, what: string) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);
const box = (x: number, y: number, l: number, w: number, yaw: number): Box => ({ x, y, l, w, yaw });

// ------------------------------------------------------------------------------------------------ known cases
test('hand-computed cases: identical, flipped, turned by 90 degrees, unit square at 45 degrees, shifted, far apart, touching', () => {
  const car = box(0, 0, 4, 2, 0);
  near(rotatedIou(car, car), 1, 1e-12, 'identical');
  near(rotatedIou(car, box(0, 0, 4, 2, Math.PI)), 1, 1e-12, 'a 180 degree flip is invisible to IoU');
  near(rotatedIou(car, box(0, 0, 4, 2, Math.PI / 2)), 1 / 3, 1e-12, '4 x 2 against its 90 degree turn: 2 x 2 overlap, 4 / (8 + 8 - 4)');
  near(rotatedIou(box(0, 0, 1, 1, 0), box(0, 0, 1, 1, Math.PI / 4)), Math.SQRT1_2, 1e-12, 'octagon of area 2(sqrt 2 - 1)');
  near(rotatedIou(car, box(1, 0, 4, 2, 0)), 0.6, 1e-12, 'shifted 1 m along the length: 6 / 10');
  near(rotatedIou(car, box(10, 0, 4, 2, 0.3)), 0, 1e-12, 'far apart');
  near(rotatedIou(car, box(4, 0, 4, 2, 0)), 0, 1e-12, 'sharing one edge');
  near(rotatedIou(box(0, 0, 4, 4, 0), box(0.3, -0.2, 2, 1, 0.7)), 2 / 16, 1e-12, 'a small box fully inside a large one: area ratio');
});

test('the overlap polygon of a 4 x 2 box and its 90 degree turn is the 2 x 2 square, and the 45 degree unit squares give an octagon', () => {
  const sq = overlapPolygon(box(0, 0, 4, 2, 0), box(0, 0, 4, 2, Math.PI / 2));
  assert.equal(sq.length, 4);
  near(area(sq), 4, 1e-12, 'square area');
  const oct = overlapPolygon(box(0, 0, 1, 1, 0), box(0, 0, 1, 1, Math.PI / 4));
  assert.equal(oct.length, 8);
  near(area(oct), 2 * (Math.SQRT2 - 1), 1e-12, 'octagon area');
});

test('IoU is symmetric, unchanged by a flip of either heading, and unchanged by moving and turning both boxes together', () => {
  const a = box(0.4, -0.3, 4.5, 1.8, 0.2), b = box(1.1, 0.5, 3.9, 2.1, -0.6);
  near(rotatedIou(a, b), rotatedIou(b, a), 1e-12, 'symmetry');
  near(rotatedIou(a, { ...b, yaw: b.yaw + Math.PI }), rotatedIou(a, b), 1e-12, 'flip');
  const th = 0.9, c = Math.cos(th), s = Math.sin(th), mv = (p: Box): Box => ({ ...p, x: 7 + p.x * c - p.y * s, y: -3 + p.x * s + p.y * c, yaw: p.yaw + th });
  near(rotatedIou(mv(a), mv(b)), rotatedIou(a, b), 1e-12, 'rigid motion');
});

test('clip keeps the part of a polygon inside a convex polygon: whole, half and none', () => {
  const sq = corners(box(0, 0, 2, 2, 0)), big = corners(box(1, 0, 6, 6, 0));
  near(area(clip(sq, big)), 4, 1e-12, 'a 2 x 2 square inside a 6 x 6 box: kept whole');
  near(area(clip(sq, corners(box(1, 0, 2, 2, 0)))), 2, 1e-12, 'a box shifted by half its length: half is kept');
  near(area(clip(corners(box(3, 0, 2, 2, 0)), corners(box(0, 0, 2, 2, 0)))), 0, 1e-12, 'disjoint');
  assert.equal(clip([], sq).length, 0);
});

// ------------------------------------------------------------------------------------------------ independent estimate
test('the clipped overlap agrees with a point-sampling estimate on random pairs', () => {
  const inside = (b: Box, px: number, py: number) => { const c = Math.cos(b.yaw), s = Math.sin(b.yaw), dx = px - b.x, dy = py - b.y; return Math.abs(dx * c + dy * s) <= b.l / 2 && Math.abs(-dx * s + dy * c) <= b.w / 2; };
  let seed = 11; const rnd = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296;
  for (let k = 0; k < 8; k++) {
    const a = box(0, 0, 1 + 5 * rnd(), 0.5 + 2 * rnd(), 6.28 * rnd()), b = box(3 * rnd() - 1.5, 3 * rnd() - 1.5, 1 + 5 * rnd(), 0.5 + 2 * rnd(), 6.28 * rnd());
    const n = 700, ext = 5; let hit = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const px = -ext + (2 * ext * (i + 0.5)) / n, py = -ext + (2 * ext * (j + 0.5)) / n; if (inside(a, px, py) && inside(b, px, py)) hit++; }
    const sampled = (hit * (2 * ext / n) ** 2), exact = area(overlapPolygon(a, b));
    near(sampled, exact, 0.03 * Math.max(1, exact) + 0.02, `pair ${k}`);
  }
});

// ------------------------------------------------------------------------------------------------ the chapter's NumPy code
test('the chapter snippet’s worked cases and 80 random pairs are reproduced (rotated IoU and enclosing-rectangle IoU)', () => {
  for (const c of G.chapter) near(rotatedIou(c.a, c.b), c.iou, 1e-9, c.name);
  assert.equal(G.random.length, 80);
  for (const r of G.random) { near(rotatedIou(r.a, r.b), r.iou, 1e-9, 'rotated'); near(enclosingIou(r.a, r.b), r.aabb, 1e-9, 'enclosing'); }
});

test('IoU against yaw error for the three shapes, at three offsets, equals the NumPy curves; the yaw at which IoU < 0.7 matches', () => {
  for (const [key, vals] of Object.entries(G.curves) as [string, number[]][]) {
    const [name, off] = key.split('@') as [keyof typeof SHAPES, string];
    vals.forEach((v, i) => near(iouAt(name, i * G.yawStepDeg, Number(off)), v, 1e-9, `${key} at ${i * G.yawStepDeg} deg`));
    assert.equal(yawBelow(name, Number(off)), G.below07[key], `${key} below 0.7`);
  }
});

test('the chapter’s claims: a car falls below 0.7 near 17 degrees, a bus near 8.5, a square pedestrian never (its minimum is sqrt(2)/2)', () => {
  near(yawBelow('car', 0)!, 16.55, 1e-9, 'car'); near(yawBelow('bus', 0)!, 8.5, 1e-9, 'bus');
  assert.equal(yawBelow('pedestrian', 0), null);
  near(Math.min(...curve('pedestrian', 0).map(p => p[1])), Math.SQRT1_2, 1e-9, 'pedestrian minimum at 45 degrees');
  near(iouAt('pedestrian', 90, 0), 1, 1e-9, 'a square turned by 90 degrees is itself');
  assert.equal(yawBelow('car', 2), 0, 'a car shifted by 2 m (IoU 2.5 / 6.5) is already below 0.7 with no yaw error');
});

// ------------------------------------------------------------------------------------------------ the lab
test('default state: a bus with a 10 degree yaw error and identical centres loses about a third of its IoU while the centre distance is 0', () => {
  const s = evaluate(DEFAULTS.shape, DEFAULTS.yaw, DEFAULTS.offset);
  assert.equal(DEFAULTS.shape, 'bus');
  near(s.iou, 0.657, 5e-4, 'rotated IoU'); near(s.aabbIou, 0.539, 5e-4, 'enclosing IoU'); assert.equal(s.centreDistance, 0);
  assert.ok(s.iou < 0.7 && s.iou > 0.6);
  near(iouAt('bus', 5, 0), 0.808, 5e-4, '5 degrees'); near(iouAt('bus', 20, 0), 0.423, 5e-4, '20 degrees');
  near(iouAt('car', 10, 0), 0.796, 5e-4, 'car at 10'); near(iouAt('pedestrian', 10, 0), 0.863, 5e-4, 'pedestrian at 10');
  const { gt, pred } = makeBoxes('bus', 10, 0);
  assert.deepEqual([gt.x, gt.y, pred.x, pred.y], [0, 0, 0, 0]);
});

test('the enclosing-rectangle IoU is below the rotated IoU for small yaw errors and above it for large ones (a bus)', () => {
  assert.ok(enclosingIou(...Object.values(makeBoxes('bus', 5, 0)) as [Box, Box]) < iouAt('bus', 5, 0));
  assert.ok(enclosingIou(...Object.values(makeBoxes('bus', 45, 0)) as [Box, Box]) > iouAt('bus', 45, 0));
});

test('centre distance equals the offset whatever the yaw; IoU falls to 0 when the offset passes the length', () => {
  for (const yaw of [0, 10, 45, 90]) near(evaluate('car', yaw, 1.3).centreDistance, 1.3, 1e-12, 'distance');
  assert.equal(iouAt('pedestrian', 0, 0.7), 0);
  near(iouAt('pedestrian', 0, 0.35), 1 / 3, 1e-12, 'half a pedestrian shifted: 0.35 x 0.7 over 0.7 x 0.7 * 1.5');
});

test('metadata: controls in range, predict answer verified by the numbers, bilingual text, two SVG panes, no NaN in the drawings', () => {
  assert.equal(lab.controls.length, 3);
  for (const c of lab.controls) if (c.kind === 'range') assert.ok(c.min! <= Number(c.value) && Number(c.value) <= c.max!);
  assert.deepEqual(lab.controls[1].options!.map(o => o.value), SHAPE_KEYS);
  assert.ok(lab.predict.options.some(o => o.value === lab.predict.answer));
  const v = iouAt('bus', 10, 0); // the predict question: bus, 10 degrees, exact centre and size
  assert.equal(lab.predict.answer, v > 0.85 ? 'high' : v > 0.5 ? 'mid' : 'low');
  assert.equal(lab.related[0], 'deep-learning/dl-09-3d-object-detection');
  for (const f of [lab.title, lab.description, lab.challenge, lab.note, lab.predict.question, lab.predict.explain]) { assert.ok(f.en.length > 20 && f['zh-tw'].length > 10); }
  const defaults = Object.fromEntries(lab.controls.map(c => [c.key, c.value]));
  for (const lang of ['en', 'zh-tw'] as const) for (const shape of SHAPE_KEYS) for (const [yaw, offset] of [[0, 0], [10, 0], [45, 1], [90, 2]]) {
    const out = lab.create({ lang, images: {} }).render({ ...defaults, shape, yaw, offset }, {});
    assert.equal(out.metrics.length, 4);
    for (const m of out.metrics) assert.ok(m.value && !m.value.includes('NaN'), m.label);
    for (const k of ['scene', 'curve']) { const svgText = out.plots![k]; assert.ok(svgText.startsWith('<svg') && svgText.endsWith('</svg>') && !svgText.includes('NaN') && !svgText.includes('undefined'), `${k} ${lang} ${shape}`); }
  }
});
