// Facts for the `segmentation-metrics` (DLCV 03B) and `stereo-matching` (DLCV 08A) labs, compared with
// scripts/tests/golden/{segmentation-metrics,stereo-matching}.json (NumPy/SciPy; scripts/figures/playground_goldens_segstereo.py).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as seg from '../../src/lib/playground/labs/segmentation-metrics.ts';
import * as st from '../../src/lib/playground/labs/stereo-matching.ts';
import { labs } from '../../src/lib/playground/labs/registry.ts';
import { labLoaders } from '../../src/lib/playground/labs/loaders.ts';

const SG = JSON.parse(readFileSync(new URL('./golden/segmentation-metrics.json', import.meta.url), 'utf8'));
const TG = JSON.parse(readFileSync(new URL('./golden/stereo-matching.json', import.meta.url), 'utf8'));
const near = (a: number, b: number, tol: number, what: string) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);

// ------------------------------------------------------------------------------------------------ segmentation

/** The chapter's 256 x 256 toy scene (disk, 6-pixel bar, 20 x 20 square), built here with the same arithmetic as its Python snippet. */
function chapterScene(): Uint8Array {
  const g = new Uint8Array(256 * 256);
  for (let y = 0; y < 256; y++) for (let x = 0; x < 256; x++) {
    if ((y - 120) ** 2 + (x - 90) ** 2 < 60 ** 2) g[y * 256 + x] = 1;
    if (y >= 40 && y < 216 && x >= 178 && x < 184) g[y * 256 + x] = 2;
    if (y >= 196 && y < 216 && x >= 60 && x < 80) g[y * 256 + x] = 3;
  }
  return g;
}

test('segmentation metrics reproduce the chapter table (nearest labels and bilinear scores, strides 4 to 32)', () => {
  const g = chapterScene(), table: Record<string, { nearest: number[]; bilinear: number[] }> = {
    // the numbers printed in the chapter (mIoU, then per-class IoU rounded to 2 decimals)
    '4': { nearest: [0.906, 0.99, 0.97, 0.67, 1.0], bilinear: [0.994, 1.0, 1.0, 0.99, 0.99] },
    '8': { nearest: [0.825, 0.98, 0.93, 0.75, 0.64], bilinear: [0.883, 0.99, 0.99, 0.7, 0.86] },
    '16': { nearest: [0.565, 0.95, 0.89, 0.0, 0.41], bilinear: [0.578, 0.97, 0.97, 0.0, 0.37] },
    '32': { nearest: [0.421, 0.92, 0.76, 0.0, 0.0], bilinear: [0.473, 0.96, 0.93, 0.0, 0.0] },
  };
  for (const s of [4, 8, 16, 32]) for (const mode of ['nearest', 'bilinear'] as const) {
    const r = seg.segScores(g, seg.predict(g, 256, 256, 4, s, mode), 4), want = table[String(s)][mode];
    near(r.mIoU, want[0], 0.0006, `mIoU ${mode} s=${s}`);
    r.iou.forEach((v, k) => near(v, want[k + 1], 0.0051, `IoU class ${k} ${mode} s=${s}`));
  }
  for (const c of SG.chapter) for (const mode of ['nearest', 'bilinear'] as const) {
    const r = seg.segScores(g, seg.predict(g, 256, 256, 4, c.stride, mode), 4);
    near(r.pixelAcc, c[mode].pixelAcc, 1e-12, `pixel accuracy ${mode} s=${c.stride}`);
    near(r.mIoU, c[mode].mIoU, 1e-12, `mIoU vs NumPy ${mode} s=${c.stride}`);
    c[mode].iou.forEach((v: number, k: number) => near(r.iou[k], v, 1e-12, `IoU vs NumPy class ${k}`));
  }
});

test('segmentation lab: every stride and both enlarging methods equal the NumPy/SciPy reference', () => {
  const g = seg.makeScene();
  assert.deepEqual(seg.segScores(g, g, seg.K).counts, SG.labCounts);
  for (const c of SG.lab) for (const mode of ['nearest', 'bilinear'] as const) {
    const r = seg.segScores(g, seg.predict(g, seg.W, seg.H, seg.K, c.stride, mode), seg.K);
    near(r.pixelAcc, c[mode].pixelAcc, 1e-12, `pixel accuracy ${mode} s=${c.stride}`);
    near(r.mIoU, c[mode].mIoU, 1e-12, `mIoU ${mode} s=${c.stride}`);
    assert.equal(r.wrong, c[mode].wrong);
    c[mode].iou.forEach((v: number, k: number) => near(r.iou[k], v, 1e-12, `IoU class ${k} ${mode} s=${c.stride}`));
  }
});

test('segmentation lab: stride 1 is exact, and the default state shows high pixel accuracy with a collapsed smallest class', () => {
  const g = seg.makeScene(), one = seg.segScores(g, seg.predict(g, seg.W, seg.H, seg.K, 1, 'nearest'), seg.K);
  assert.equal(one.pixelAcc, 1); assert.equal(one.mIoU, 1);
  const small = seg.smallestClass(one.counts);
  assert.equal(small, 4, 'the five squares are the smallest class');
  assert.ok(one.counts[0] / (seg.W * seg.H) > 0.85, 'the background is a large majority');
  assert.equal(seg.DEFAULT_STRIDE, 8);
  const d = seg.segScores(g, seg.predict(g, seg.W, seg.H, seg.K, seg.DEFAULT_STRIDE, 'nearest'), seg.K);
  assert.ok(d.pixelAcc > 0.94 && d.pixelAcc < 0.95, `pixel accuracy ${d.pixelAcc}`);
  assert.equal(d.iou[small], 0);
  assert.equal(d.iou[3], 0, 'the thin pole is lost at stride 8 as well');
  assert.ok(d.mIoU < 0.52, `mIoU ${d.mIoU}`);
  // the facts quoted in the predict explanation
  assert.equal(seg.segScores(g, g, seg.K).counts[3] + seg.segScores(g, g, seg.K).counts[4], 444);
  near(SG.labMaxFractionClass4At8, 24 / 64, 1e-12, 'no square fills more than 24 of the 64 pixels of a stride-8 cell');
  const lost = seg.predict(g, seg.W, seg.H, seg.K, 8, 'nearest');
  assert.ok(!lost.includes(3) && !lost.includes(4), 'no cell of the stride-8 prediction is labelled pole or square');
});

test('segmentation lab: the pole needs a finer stride than the squares; bilinear raises accuracy but not the small-class IoU at stride 8', () => {
  const g = seg.makeScene(), run = (s: number, mode: 'nearest' | 'bilinear') => seg.segScores(g, seg.predict(g, seg.W, seg.H, seg.K, s, mode), seg.K);
  assert.equal(run(4, 'nearest').iou[3], 0); assert.ok(run(2, 'nearest').iou[3] > 0.6);
  assert.ok(run(4, 'nearest').iou[4] > 0.4);
  assert.ok(run(8, 'bilinear').pixelAcc > run(8, 'nearest').pixelAcc && run(8, 'bilinear').mIoU > run(8, 'nearest').mIoU);
  assert.equal(run(8, 'bilinear').iou[4], 0);
  // at stride 32 only the two large classes keep a non-zero IoU
  assert.equal(run(32, 'nearest').iou.filter(v => v > 0).length, 2);
  // pixel accuracy falls monotonically with the stride (nearest), and so does mIoU
  const accs = seg.STRIDES.map(s => run(s, 'nearest'));
  for (let i = 1; i < accs.length; i++) { assert.ok(accs[i].pixelAcc <= accs[i - 1].pixelAcc); assert.ok(accs[i].mIoU <= accs[i - 1].mIoU); }
});

test('segmentation lab: bilinear upsampling of a constant-fraction field and nearest of a one-hot cell behave as defined', () => {
  const scores = new Float64Array([1, 0, 0, 1, 0, 1, 1, 0]); // 2 classes, 2 x 2 cells: class 0 where [0][0] and [1][1], class 1 elsewhere
  const near2 = seg.upsampleNearest(scores, 2, 2, 2, 2);
  assert.deepEqual(Array.from(near2), [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]);
  assert.throws(() => seg.coarseScores(new Uint8Array(100), 10, 10, 2, 3));
  // a tie goes to the lowest class, like numpy.argmax
  assert.equal(seg.upsampleNearest(new Float64Array([0.5, 0.5]), 1, 1, 2, 1)[0], 0);
});

// ------------------------------------------------------------------------------------------------ stereo

test('stereo scene: the geometry is as described and the occluded count follows from it', () => {
  const sc = st.makeScene();
  let disc = 0, occ = 0, ev = 0, flat = 0, discRows = 0;
  for (let y = 0; y < st.H; y++) {
    let inRow = false;
    for (let x = 0; x < st.W; x++) { const i = y * st.W + x; if (sc.gt[i] === st.D_FG) { disc++; inRow = true; } occ += sc.occluded[i]; ev += sc.evaluated[i]; flat += sc.flat[i]; }
    if (inRow) discRows++;
  }
  // occluded = the D_BG leftmost columns of every row, plus a strip D_FG - D_BG wide just left of the disc in every row that crosses it
  assert.equal(occ, st.D_BG * st.H + (st.D_FG - st.D_BG) * discRows);
  assert.equal(occ, TG.counts.occluded); assert.equal(ev, TG.counts.evaluated); assert.equal(flat, TG.counts.flat);
  assert.ok(disc > 1900 && disc < 2000);
  // a visible pixel has an exact partner in the clean right image (a hidden one does not)
  for (let y = 0; y < st.H; y += 7) for (let x = 0; x < st.W; x++) {
    const i = y * st.W + x, xr = x - sc.gt[i];
    if (xr >= 0 && !sc.occluded[i]) assert.equal(sc.right[y * st.W + xr], sc.left[i]);
  }
});

test('stereo matching equals the NumPy/SciPy pipeline: scores for every window, with and without the left-right check', () => {
  const sc = st.makeScene();
  for (const run of TG.runs) {
    const m = st.matchPair(sc, run.sigma);
    st.WINDOWS.forEach((w, k) => {
      for (const lr of [false, true]) {
        const want = run[lr ? 'on' : 'off'][k], got = st.scoreDisparity(sc, st.finalDisparity(m, k, lr));
        const tag = `sigma ${run.sigma} window ${w} lr ${lr}`;
        for (const key of ['bad1', 'badFlat', 'fgSpill', 'invalid', 'invalidOccluded'] as const) {
          if (want[key] === null) assert.ok(!Number.isFinite(got[key]), `${tag} ${key}`);
          else near(got[key], want[key], 0.15, `${tag} ${key}`);
        }
      }
    });
    for (const [w, ref] of Object.entries<{ left: string }>(run.maps)) {
      const got = m.left[st.WINDOWS.indexOf(Number(w))];
      let differ = 0;
      for (let i = 0; i < got.length; i++) if (String.fromCharCode(48 + got[i]) !== ref.left[i]) differ++;
      assert.ok(differ <= 3, `window ${w}: ${differ} of ${got.length} disparities differ from NumPy`);
    }
  }
});

test('block matching recovers a known integer shift exactly (left and right reference) and the check keeps it', () => {
  const w = 40, h = 24, n = w * h, shift = 5, tex = Float32Array.from({ length: h * 41 + w }, (_, i) => 0.5 + Math.sin(i * 0.37) * 0.3 + Math.sin(i * 1.91) * 0.2 + ((i * 2654435761) % 997) / 9970);
  const left = new Float32Array(n), right = new Float32Array(n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) right[y * w + x] = tex[y * 41 + x];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) left[y * w + x] = x - shift >= 0 ? right[y * w + x - shift] : 0.1; // left(x) = right(x - shift)
  const dl = st.blockMatch(left, right, w, h, 8, [5, 9], 'left'), dr = st.blockMatch(left, right, w, h, 8, [5, 9], 'right');
  for (let k = 0; k < 2; k++) {
    const ok = st.leftRightCheck(dl[k], dr[k], w, h);
    for (let y = 6; y < h - 6; y++) for (let x = 14; x < w - 14; x++) { assert.equal(dl[k][y * w + x], shift); assert.equal(dr[k][y * w + x - shift], shift); assert.equal(ok[y * w + x], shift); }
  }
});

test('left-right check: invalidates on disagreement and outside the picture, keeps a one-pixel difference', () => {
  // every left pixel says d = 3. x = 0..2 would land left of the right image. x = 3 -> xr = 0 holds 3 (equal: kept), x = 4 -> xr = 1 holds 4 (1 apart: kept),
  // x = 5 -> xr = 2 holds 5 (2 apart: invalid).
  const out = st.leftRightCheck(new Uint8Array([3, 3, 3, 3, 3, 3]), new Uint8Array([3, 4, 5, 0, 0, 0]), 6, 1);
  assert.deepEqual(Array.from(out), [st.INVALID, st.INVALID, st.INVALID, 3, 3, st.INVALID]);
});

test('stereo lab: the default state shows the trade-off (noisy small window, fattened large window, a valley in between)', () => {
  const sc = st.makeScene(), m = st.matchPair(sc, st.DEFAULTS.noise), sw = st.sweep(sc, m, false), s = (w: number) => sw.scores[st.WINDOWS.indexOf(w)];
  assert.equal(st.DEFAULTS.win, 3); assert.equal(st.DEFAULTS.lr, 'off');
  assert.ok(s(3).bad1 > 10 && s(3).bad1 < 11, `bad-1 at 3 x 3 is ${s(3).bad1}`);
  assert.ok(s(3).badFlat > 55, 'a 3 x 3 window is mostly wrong inside the textureless patch');
  assert.ok(s(31).badFlat === 0, 'a 31 x 31 window fixes the patch');
  assert.ok([7, 9, 11].includes(st.WINDOWS[sw.best]), `best window ${st.WINDOWS[sw.best]}`);
  assert.ok(sw.scores[sw.best].bad1 > 4 && sw.scores[sw.best].bad1 < 4.2);
  assert.ok(s(31).bad1 > sw.scores[sw.best].bad1 + 1 && s(31).bad1 < s(3).bad1, 'a valley: 31 x 31 is worse than the best window, better than 3 x 3');
  assert.ok(s(31).fgSpill > s(5).fgSpill + 20, 'the disc grows with the window');
  for (let k = 1; k < sw.scores.length; k++) assert.ok(sw.scores[k].badFlat <= sw.scores[k - 1].badFlat + 2, 'the textureless patch improves with the window');
  // the numbers quoted in the predict explanation
  near(s(3).bad1, 10.6, 0.05, 'bad-1 at 3'); near(s(31).bad1, 5.9, 0.05, 'bad-1 at 31'); near(sw.scores[sw.best].bad1, 4.1, 0.05, 'best bad-1');
});

test('stereo lab: the check removes most occluded pixels, few visible ones, and lowers bad-1', () => {
  const sc = st.makeScene(), m = st.matchPair(sc, st.DEFAULTS.noise), k = st.WINDOWS.indexOf(9);
  const off = st.scoreDisparity(sc, st.finalDisparity(m, k, false)), on = st.scoreDisparity(sc, st.finalDisparity(m, k, true));
  assert.equal(off.invalid, 0);
  assert.ok(on.invalidOccluded > 60 && on.invalid < 5, `${on.invalidOccluded} / ${on.invalid}`);
  assert.ok(on.bad1 < off.bad1 / 1.5, `bad-1 ${off.bad1} -> ${on.bad1}`);
  // noise hurts the smallest window most
  const noisy = st.sweep(sc, st.matchPair(sc, 0.06), false), calm = st.sweep(sc, m, false);
  assert.ok(noisy.scores[0].bad1 > 2 * calm.scores[0].bad1);
});

test('stereo helpers: disparity colors span a dark-to-light ramp, the true edge sits on the disc outline', () => {
  const a = st.disparityColor(0), b = st.disparityColor(st.D - 1), sum = (c: number[]) => c[0] + c[1] + c[2];
  assert.ok(sum(b) > sum(a) + 200);
  assert.notDeepEqual(st.disparityColor(st.D_BG), st.INVALID_COLOR);
  const sc = st.makeScene(), e = st.trueEdge(sc);
  let n = 0; for (const v of e) n += v;
  assert.ok(n > 100 && n < 400);
});

// ------------------------------------------------------------------------------------------------ wiring

test('both labs are registered, lazy-loadable and relating to their chapters (the plan itself is checked by scripts/check-playground.mjs), with bilingual text', async () => {
  for (const [id, chapter] of [['segmentation-metrics', 'deep-learning/dl-03b-image-segmentation'], ['stereo-matching', 'deep-learning/dl-08a-depth-optical-flow']]) {
    const lab = labs.find(l => l.id === id)!;
    assert.ok(lab, id); assert.deepEqual(lab.related, [chapter]);
    assert.ok(lab.controls.length >= 1 && lab.controls.length <= 3);
    assert.ok(lab.predict.options.some(o => o.value === lab.predict.answer));
    assert.ok(lab.sources.length >= 1 && lab.sources.every(s => s.url.startsWith('https://')));
    for (const text of [lab.title, lab.description, lab.challenge, lab.note, lab.predict.question, lab.predict.explain, ...lab.controls.map(c => c.label), ...lab.predict.options.map(o => o.label), ...lab.panes.map(p => p.label)]) {
      assert.ok(text.en.length > 3 && text['zh-tw'].length > 3, id);
      assert.ok(!/[数据图网络视频质认样览]/.test(text['zh-tw']), `Simplified characters in ${id}`);
    }
    const loaded = await labLoaders[id]();
    assert.equal(loaded.lab.id, id);
  }
});
