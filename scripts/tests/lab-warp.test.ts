// Tests for the `linear-transform` lab (ML 01A: what does a 2 x 2 matrix do to a picture?).
// Goldens come from scripts/figures/playground_goldens_warp.py (numpy.linalg.svd, scipy.ndimage.affine_transform and
// skimage.transform.warp, all independent of the TypeScript). Every number quoted in the lab text is asserted here by running the
// lab's own pure functions on the real asset src/assets/playground/camera-256.png.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { readGray, asset } from './helpers/png.ts';
import { svd2, det2, inv2, mul2, apply2, polygonArea, unitSquareImage, stretchRange, rotation2, SINGULAR_TOL } from '../../src/lib/imagekit/matrix2.ts';
import type { Mat2 } from '../../src/lib/imagekit/matrix2.ts';
import { warpBilinear, warpLine, warpPoint, warpMatrix, centredFrame, bilinearAt, planeToPixel, pixelToPlane, overBackground } from '../../src/lib/imagekit/warp.ts';
import * as lt from '../../src/lib/playground/labs/linear-transform.ts';

const G = JSON.parse(readFileSync(new URL('./golden/warp.json', import.meta.url), 'utf8'));
const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => { assert.equal(a.length, b.length); let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const near = (a: number, b: number, tol: number, msg = '') => assert.ok(Math.abs(a - b) <= tol, `${msg} ${a} vs ${b}`);
const cam = readGray(asset('camera-256.png'));
const mat = (a: number[]): Mat2 => [a[0], a[1], a[2], a[3]];

// ------------------------------------------------------------------------------------------------ closed-form SVD vs NumPy

test('closed-form 2 x 2 SVD matches numpy.linalg.svd, det, matrix_rank and cond on 40 matrices', () => {
  assert.equal(G.svd.length, 40);
  for (const g of G.svd) {
    const A = mat(g.A), sv = svd2(A);
    near(sv.s1, g.s[0], 1e-12 * Math.max(1, g.s[0]), `s1 of ${g.A}`);
    near(sv.s2, g.s[1], 1e-12 * Math.max(1, g.s[0]), `s2 of ${g.A}`);
    near(det2(A), g.det, 1e-12, `det of ${g.A}`);
    assert.equal(sv.rank, g.rank, `rank of ${g.A}`);
    if (g.cond === null) assert.equal(sv.cond, Infinity); else near(sv.cond / g.cond, 1, 1e-9, `cond of ${g.A}`);
    // A v_i = s_i u_i with orthonormal v and u, and s1 * s2 = |det|
    for (const [v, u, s] of [[sv.v1, sv.u1, sv.s1], [sv.v2, sv.u2, sv.s2]] as const) {
      const [x, y] = apply2(A, v[0], v[1]);
      near(x, s * u[0], 1e-12, 'A v = s u (x)'); near(y, s * u[1], 1e-12, 'A v = s u (y)');
      near(Math.hypot(v[0], v[1]), 1, 1e-12); near(Math.hypot(u[0], u[1]), 1, 1e-12);
    }
    near(sv.v1[0] * sv.v2[0] + sv.v1[1] * sv.v2[1], 0, 1e-12); near(sv.u1[0] * sv.u2[0] + sv.u1[1] * sv.u2[1], 0, 1e-12);
    near(sv.s1 * sv.s2, Math.abs(g.det), 1e-12 * Math.max(1, sv.s1 * sv.s1), '|det| = s1 s2');
  }
});

test('the singular values are the longest and shortest stretch of a unit vector (the semi-axes of the image of the unit circle)', () => {
  for (const g of G.svd.slice(0, 25)) {
    const A = mat(g.A), sv = svd2(A), r = stretchRange(A, 7200);
    near(r.max, sv.s1, 1e-5 * Math.max(1, sv.s1), `max stretch ${g.A}`);
    near(r.min, sv.s2, 1e-3 * Math.max(1, sv.s1), `min stretch ${g.A}`); // the sampled minimum misses an exact null direction by up to s1 * pi / 7200
  }
});

test('a matrix with a second singular value below 1e-9 of the first is singular; the zero matrix has rank 0', () => {
  assert.equal(svd2([1, 2, 0.5, 1]).rank, 1);
  assert.equal(svd2([0, 0, 0, 0]).rank, 0);
  assert.equal(svd2([1, 0, 0, 1e-12]).rank, 1);
  assert.equal(svd2([1, 0, 0, 1e-6]).rank, 2);
  assert.equal(SINGULAR_TOL, 1e-9);
});

// ------------------------------------------------------------------------------------------------ warp vs SciPy and scikit-image

test('bilinear inverse-mapping warp equals scipy.ndimage.affine_transform (order 1) in both boundary modes, data and alpha', () => {
  const { w, h } = G.warp, img = { w, h, data: Float32Array.from(G.warp.image) };
  for (const c of G.warp.cases) {
    const A = mat(c.A);
    for (const [mode, key] of [['constant', 'constant'], ['grid-constant', 'grid']] as const) {
      const out = warpBilinear(img, A, { mode });
      const want = (c[key] as number[][]).flat(), wantAlpha = (c['alpha_' + key] as number[][]).flat();
      assert.ok(maxDiff(out.data, want) < 2e-6, `${c.name} ${mode}: data differs by ${maxDiff(out.data, want)}`);
      assert.ok(maxDiff(out.alpha, wantAlpha) < 2e-6, `${c.name} ${mode}: alpha differs by ${maxDiff(out.alpha, wantAlpha)}`);
    }
  }
});

test("boundary convention: 'constant' cuts at the edge pixels, 'grid-constant' fades over one pixel (the convention of skimage.transform.warp)", () => {
  const img = { w: 3, h: 3, data: Float32Array.from([1, 1, 1, 1, 1, 1, 1, 1, 1]) };
  assert.deepEqual(bilinearAt(img, -0.3, 1, 'constant'), { value: 0, alpha: 0 });
  near(bilinearAt(img, -0.3, 1, 'grid-constant').value, 0.7, 1e-12); near(bilinearAt(img, -0.3, 1, 'grid-constant').alpha, 0.7, 1e-12);
  assert.deepEqual(bilinearAt(img, 2, 2, 'constant'), { value: 1, alpha: 1 });
  assert.deepEqual(bilinearAt(img, 2.0001, 1, 'constant'), { value: 0, alpha: 0 });
  assert.equal(bilinearAt(img, -1e-12, 1, 'constant').alpha, 1); // within 1e-9 of the edge counts as on it
  assert.equal(bilinearAt(img, -1.2, 1, 'grid-constant').alpha, 0);
  assert.equal(bilinearAt(img, 3.2, 1, 'grid-constant').alpha, 0);
});

test('a y-up frame with unit u is the same warp as the pixel-space matrix [[a, -b], [-c, d]]', () => {
  const { w, h } = G.warp, img = { w, h, data: Float32Array.from(G.warp.image) };
  for (const c of G.warp.cases) {
    const [a, b, cc, d] = c.A, up = warpBilinear(img, [a, b, cc, d], { frame: centredFrame(w, h, 1, true) }), down = warpBilinear(img, [a, -b, -cc, d]);
    assert.ok(maxDiff(up.data, down.data) < 1e-6, c.name);
  }
  // The unit only changes the length scale: with unit u and matrix A it equals unit 1 and matrix A (plane lengths in pixels are u times larger on both sides).
  const f1 = centredFrame(w, h, 1, true), f5 = centredFrame(w, h, 5, true), A: Mat2 = [0.9, -0.4, 0.3, 1.2];
  assert.ok(maxDiff(warpBilinear(img, A, { frame: f1 }).data, warpBilinear(img, A, { frame: f5 }).data) < 1e-6);
  const [px, py] = planeToPixel(lt.FRAME, 1, 1), [qx, qy] = pixelToPlane(lt.FRAME, px, py);
  near(px, 127.5 + 64, 1e-12); near(py, 127.5 - 64, 1e-12); near(qx, 1, 1e-12); near(qy, 1, 1e-12);
});

// ------------------------------------------------------------------------------------------------ discoveries, verified on the photograph

test('determinant: the image of the unit square has signed area det A, and its sign is the orientation', () => {
  let flips = 0;
  for (const g of G.svd) {
    const A = mat(g.A), area = polygonArea(unitSquareImage(A));
    near(area, det2(A), 1e-12, 'shoelace = det');
    if (det2(A) < 0) { flips++; assert.equal(lt.orientationOf(A), 'mirrored'); }
    if (det2(A) > 0 && svd2(A).rank === 2) assert.equal(lt.orientationOf(A), 'kept');
  }
  assert.ok(flips > 5);
});

test('a negative determinant mirrors the real photograph; a positive one with |det| = 1 never does', () => {
  const run = (A: Mat2) => warpBilinear(cam, A, { frame: lt.FRAME, mode: 'grid-constant' }).data;
  const mirror = run([-1, 0, 0, 1]), rot180 = run([-1, 0, 0, -1]), swap = run([0, 1, 1, 0]), same = run([1, 0, 0, 1]);
  let dm = 0, dr = 0, ds = 0, di = 0;
  for (let r = 0; r < 256; r++) for (let c = 0; c < 256; c++) {
    const i = r * 256 + c;
    dm = Math.max(dm, Math.abs(mirror[i] - cam.data[r * 256 + (255 - c)]));      // left-right mirror
    dr = Math.max(dr, Math.abs(rot180[i] - cam.data[(255 - r) * 256 + (255 - c)])); // rotation by 180 degrees
    ds = Math.max(ds, Math.abs(swap[i] - cam.data[(255 - c) * 256 + (255 - r)]));   // anti-transpose = reflection across the diagonal from bottom left to top right
    di = Math.max(di, Math.abs(same[i] - cam.data[i]));
  }
  for (const d of [dm, dr, ds, di]) assert.ok(d < 1e-5, `max difference ${d}`);
  assert.equal(det2([-1, 0, 0, 1]), -1); assert.equal(det2([-1, 0, 0, -1]), 1); assert.equal(det2([0, 1, 1, 0]), -1);
  // the mirror really differs from a rotation by 180 degrees on this (asymmetric) photograph
  let diff = 0; for (let i = 0; i < mirror.length; i++) diff += Math.abs(mirror[i] - rot180[i]);
  assert.ok(diff / mirror.length > 0.05, `mean difference ${diff / mirror.length}`);
});

test('area scale: the picture shrunk by diag(0.5, 0.9) covers |det| = 0.45 of the canvas (0.4492 counting pixels), also with the antialiased path', () => {
  const A: Mat2 = [0.5, 0, 0, 0.9], plain = warpBilinear(cam, A, { frame: lt.FRAME, mode: 'constant' });
  const fraction = plain.alpha.reduce((s, v) => s + v, 0) / (256 * 256);
  near(fraction, G.asset.coverageHalfNine.fraction, 1e-12); near(fraction, 0.45, 0.01);
  assert.equal(G.asset.coverageHalfNine.fraction, 0.44921875);
  const smooth = warpBilinear(cam, A, { frame: lt.FRAME, antialias: true }).alpha.reduce((s, v) => s + v, 0) / (256 * 256);
  near(smooth, 0.45, 0.01);
  for (const B of [[0.7, 0.2, -0.1, 0.6], [0.6, 0, 0.3, 0.5], [0.8, -0.3, 0.2, 0.7]] as Mat2[]) {
    const f = warpMatrix(cam, B, { frame: lt.FRAME }).alpha.reduce((s, v) => s + v, 0) / (256 * 256);
    near(f, Math.abs(det2(B)), 0.01, `coverage of ${B}`);
  }
});

test('rank 1: a projection flattens the photo onto a line whose value is the mean of the photo along the collapsed line', () => {
  // exact on an odd-sized image whose centre is a pixel: every sample falls on the grid
  const n = 9, data = Float32Array.from({ length: n * n }, (_, i) => ((i * 37) % 11) / 10), img = { w: n, h: n, data };
  const out = warpLine(img, [1, 0, 0, 0], { frame: centredFrame(n, n, 1, false) });
  for (let c = 0; c < n; c++) {
    let mean = 0; for (let r = 0; r < n; r++) mean += data[r * n + c]; mean /= n;
    near(out.data[4 * n + c], mean, 1e-6, `column ${c}`);
    assert.equal(out.alpha[4 * n + c], 1);
    assert.equal(out.alpha[0 * n + c], 0); assert.equal(out.alpha[8 * n + c], 0);
  }
  // on the real photo (even size, so samples fall half-way between rows) it agrees to within about 0.01
  const photo = warpMatrix(cam, lt.presetMatrix('project', 1, 0), { frame: lt.FRAME });
  assert.equal(svd2(lt.presetMatrix('project', 1, 0)).rank, 1);
  let covered = 0; for (const a of photo.alpha) if (a > 0) covered++;
  assert.ok(covered <= 4 * 256, `the line covers ${covered} pixels`);
  for (const c of [20, 127, 200]) {
    let mean = 0; for (let r = 0; r < 256; r++) mean += cam.data[r * 256 + c]; mean /= 256;
    assert.ok(Math.abs(photo.data[127 * 256 + c] / photo.alpha[127 * 256 + c] - mean) < 0.01, `column ${c}`);
  }
  near(G.asset.columnMeanRow127, cam.data.reduce((s, _, i) => (i % 256 === 127 ? s + cam.data[i] : s), 0) / 256, 1e-6);
  // two different points of the picture on the same collapsed line have the same image
  const A = lt.presetMatrix('project', 1, 30), p = apply2(A, 0.3, 0.2), q = apply2(A, 0.3 + 0.5 * -Math.sin(Math.PI / 6), 0.2 + 0.5 * Math.cos(Math.PI / 6));
  near(p[0], q[0], 1e-12); near(p[1], q[1], 1e-12);
  // the zero matrix is a dot
  const dot = warpPoint(cam, { frame: lt.FRAME }); let dots = 0; for (const a of dot.alpha) if (a > 0) dots++;
  assert.ok(dots > 0 && dots < 40);
  assert.equal(warpMatrix(cam, [0, 0, 0, 0], { frame: lt.FRAME }).alpha[127 * 256 + 127] > 0, true);
});

test('the antialiased warp equals the plain one for a stretch, and averages (smooths) a squeeze instead of aliasing it', () => {
  const A: Mat2 = [1.3, 0.2, -0.1, 1.2], a = warpBilinear(cam, A, { frame: lt.FRAME }), b = warpBilinear(cam, A, { frame: lt.FRAME, antialias: true });
  assert.ok(maxDiff(a.data, b.data) < 1e-6);
  // shrink to 1/8: the plain sampler aliases (rough), the antialiased one tracks the local mean of the photo
  const S: Mat2 = [0.125, 0, 0, 0.125], plain = warpBilinear(cam, S, { frame: lt.FRAME }), smooth = warpBilinear(cam, S, { frame: lt.FRAME, antialias: true });
  let ep = 0, es = 0, n = 0;
  for (let r = 0; r < 256; r++) for (let c = 0; c < 256; c++) {
    if (smooth.alpha[r * 256 + c] < 0.999) continue;
    // reference: mean of the 8 x 8 block of the photo this output pixel stands for
    const sc = Math.round(127.5 + (c - 127.5) * 8), sr = Math.round(127.5 + (r - 127.5) * 8);
    if (sc < 4 || sr < 4 || sc > 251 || sr > 251) continue;
    let m = 0; for (let y = -4; y < 4; y++) for (let x = -4; x < 4; x++) m += cam.data[(sr + y) * 256 + sc + x]; m /= 64;
    ep += Math.abs(plain.data[r * 256 + c] - m); es += Math.abs(smooth.data[r * 256 + c] - m); n++;
  }
  assert.ok(n > 100 && es < ep, `antialiased error ${es / n} should be below plain ${ep / n}`);
  assert.ok(Number.isFinite(overBackground(smooth, 0.9)[0]));
});

// ------------------------------------------------------------------------------------------------ the lab's families and the text it quotes

test('every family has the determinant, singular values and rank the note claims, on the whole slider grid', () => {
  const ks = Array.from({ length: 41 }, (_, i) => Math.round((-2 + i * 0.1) * 10) / 10), thetas = Array.from({ length: 73 }, (_, i) => -180 + 5 * i);
  const range = { a: [Infinity, -Infinity], b: [Infinity, -Infinity], c: [Infinity, -Infinity], d: [Infinity, -Infinity] } as Record<string, number[]>;
  for (const preset of lt.PRESETS.filter(p => p !== 'free')) for (const k of ks) for (const t of thetas) {
    const A = lt.presetMatrix(preset, k, t), sv = svd2(A), d = det2(A);
    if (preset === 'shear') {
      near(d, 1, 1e-9); near(sv.s1, (Math.sqrt(k * k + 4) + Math.abs(k)) / 2, 1e-9); near(sv.s2, (Math.sqrt(k * k + 4) - Math.abs(k)) / 2, 1e-9);
    } else if (preset === 'stretch') {
      near(d, k, 1e-9); near(sv.s1, Math.max(1, Math.abs(k)), 1e-9); near(sv.s2, Math.min(1, Math.abs(k)), 1e-9);
      assert.equal(sv.rank, k === 0 ? 1 : 2);
    } else if (preset === 'rotate') {
      near(d, k * k, 1e-9); near(sv.s1, Math.abs(k), 1e-9); near(sv.s2, Math.abs(k), 1e-9);
    } else if (preset === 'reflect') {
      near(d, -k * k, 1e-9); near(sv.s1, Math.abs(k), 1e-9); near(sv.s2, Math.abs(k), 1e-9);
    } else {
      assert.equal(sv.rank, k === 0 ? 0 : 1, `project k=${k} t=${t}`); near(sv.s1, Math.abs(k), 1e-9); near(d, 0, 1e-12);
    }
    (['a', 'b', 'c', 'd'] as const).forEach((key, i) => { range[key][0] = Math.min(range[key][0], A[i]); range[key][1] = Math.max(range[key][1], A[i]); });
  }
  // every one of the four entries sweeps (almost) all of -2..2 under the two sliders
  for (const key of ['a', 'b', 'c', 'd']) assert.ok(range[key][1] - range[key][0] > 3.5, `${key} range ${range[key]}`);
});

test("the default state's numbers quoted in the challenge: shear k = 1 along 30 degrees has det 1, singular values 1.62 and 0.62, kappa 2.62", () => {
  const A = lt.presetMatrix(lt.DEFAULTS.preset, lt.DEFAULTS.k, lt.DEFAULTS.theta), F = lt.factsOf(A), g = G.asset.defaultShear;
  assert.equal(lt.DEFAULTS.preset, 'shear'); assert.equal(lt.DEFAULTS.k, 1); assert.equal(lt.DEFAULTS.theta, 30);
  for (let i = 0; i < 4; i++) near(A[i], g.A[i], 1e-9);
  near(F.det, 1, 1e-9); near(F.sv.s1, g.s[0], 1e-9); near(F.sv.s2, g.s[1], 1e-9); near(F.sv.cond, g.cond, 1e-6);
  assert.equal(lt.metricsOf('en', A).map(m => m.value).join('|'), '1|1.62|0.62|2.62|2 (full)|kept');
  assert.equal(lt.metricsOf('zh-tw', A).map(m => m.value).join('|'), '1|1.62|0.62|2.62|2（滿秩）|保持');
  near((Math.sqrt(5) + 1) / 2, 1.618, 1e-3); near((Math.sqrt(5) - 1) / 2, 0.618, 1e-3);
  assert.ok(/1\.62/.test(lt.lab.challenge.en) && /0\.62/.test(lt.lab.challenge.en) && /2\.62/.test(lt.lab.challenge['zh-tw']));
  // all four entries of the default matrix are nonzero, so the default is not an axis-aligned toy
  assert.ok(A.every(v => Math.abs(v) > 0.2));
});

test('stretch: k = 1 is the identity, k = 0 collapses to a line (rank 1, kappa infinite), k = -1 is a mirror', () => {
  assert.deepEqual(lt.presetMatrix('stretch', 1, 30), [1, 0, 0, 1]);
  const flat = lt.factsOf(lt.presetMatrix('stretch', 0, 30)), mirror = lt.factsOf(lt.presetMatrix('stretch', -1, 30));
  assert.equal(flat.sv.rank, 1); assert.equal(flat.sv.cond, Infinity); assert.equal(flat.orientation, 'flat'); assert.equal(lt.condText(flat.sv), '∞');
  assert.equal(mirror.orientation, 'mirrored'); near(mirror.det, -1, 1e-12);
  assert.equal(lt.metricsOf('en', lt.presetMatrix('stretch', 0, 30))[3].value, '∞');
  for (const lang of ['en', 'zh-tw'] as const) {
    assert.ok(lt.describe(lang, lt.presetMatrix('stretch', 0, 30)).length > 40);
    assert.ok(lt.describe(lang, lt.presetMatrix('project', 0, 30)).length > 20);
  }
  assert.ok(/mirrored/.test(lt.describe('en', lt.presetMatrix('stretch', -1, 30))) && /鏡像/.test(lt.describe('zh-tw', lt.presetMatrix('stretch', -1, 30))));
});

test("predict question: A = [[0, 1], [1, 0]] has det -1 and singular values 1 and 1 (a mirror), reachable as Reflection with k = 1 and theta = 45", () => {
  const swap: Mat2 = [0, 1, 1, 0], R = lt.presetMatrix('reflect', 1, 45);
  assert.deepEqual(R, swap);
  const F = lt.factsOf(swap);
  assert.equal(F.det, -1); near(F.sv.s1, 1, 1e-12); near(F.sv.s2, 1, 1e-12); assert.equal(F.orientation, 'mirrored');
  // a 90 degree rotation has det +1 and is a different matrix; the photo it makes differs from the swap
  const rot = lt.presetMatrix('rotate', 1, 90);
  assert.deepEqual(rot, [0, -1, 1, 0]); assert.equal(det2(rot), 1);
  // e1 -> e2 is counter-clockwise (+), Ae1 -> Ae2 is clockwise (-): the cross product is det
  near(swap[0] * swap[3] - swap[2] * swap[1], -1, 0);
  const q = lt.lab.predict;
  assert.equal(q.answer, 'mirror'); assert.ok(q.options.length >= 2 && q.options.length <= 4 && q.options.some(o => o.value === q.answer));
  assert.ok(/−1/.test(q.explain.en) && /1\.|both singular values are 1/.test(q.explain.en));
});

test('dragging a tip: clamped to the drawing area, rounded to 0.01, nearest tip chosen; any matrix in range is reachable', () => {
  assert.deepEqual(lt.withTip([1, 0, 0, 1], 0, 5, -5), [lt.TIP_LIMIT, 0, -lt.TIP_LIMIT, 1]);
  assert.deepEqual(lt.withTip([1, 0, 0, 1], 1, 0.3333, 0.666), [1, 0.33, 0, 0.67]);
  assert.equal(lt.nearestTip([1, 0, 0, 1], 0.9, 0.1), 0); assert.equal(lt.nearestTip([1, 0, 0, 1], 0.1, 0.8), 1);
  let A: Mat2 = [1, 0, 0, 1];
  A = lt.withTip(A, 0, -1.2, 0.35); A = lt.withTip(A, 1, 0.6, -1.7);
  assert.deepEqual(A, [-1.2, 0.6, 0.35, -1.7]);
  near(det2(A), -1.2 * -1.7 - 0.6 * 0.35, 1e-12);
  const inv = inv2(A)!, I = mul2(A, inv); near(I[0], 1, 1e-12); near(I[3], 1, 1e-12); near(I[1], 0, 1e-12);
  near(rotation2(30)[0], Math.cos(Math.PI / 6), 1e-15);
});

test('lab metadata: three controls, bilingual metric labels, a free preset with its own start matrix, sources with links', () => {
  const l = lt.lab;
  assert.equal(l.id, 'linear-transform'); assert.equal(l.track, 'foundations'); assert.equal(l.controls.length, 3);
  assert.deepEqual(l.controls[0].options!.map(o => o.value), lt.PRESETS);
  for (const o of l.controls[0].options!) assert.ok(o.label.en && o.label['zh-tw']);
  assert.deepEqual(Object.keys(l.assets), ['camera-256']); assert.deepEqual(l.related, ['ml/ml-01a-linear-algebra']);
  const en = lt.metricsOf('en', lt.FREE_START), zh = lt.metricsOf('zh-tw', lt.FREE_START);
  assert.equal(en.length, 6); assert.ok(en.every((m, i) => m.label !== zh[i].label && m.value.length > 0));
  assert.ok(l.sources.every(s => /^https:\/\//.test(s.url)));
  assert.equal(lt.presetMatrix('free', 9, 99, [0.1, 0.2, 0.3, 0.4]).join(), '0.1,0.2,0.3,0.4');
});
