import { test } from 'node:test';
import assert from 'node:assert/strict';
import { projectQuadratic, scalarDual, locationFit, barrierPoint, softThreshold, chapterOptimizationExperiments } from '../../src/lib/playground/chapters-optimization.ts';
import { bernoulliMismatch, typicalSet, waterFill, binaryRateDistortion, testingErrors, chapterInformationExperiments } from '../../src/lib/playground/chapters-information.ts';
const close = (a: number, b: number, t = 1e-9) => assert.ok(Math.abs(a - b) < t, `${a} ≠ ${b}`);
test('constrained quadratic candidates satisfy KKT and improve over clipping', () => {
    for (const a of [-2, -1, 0, 1, 2])
        for (const rho of [-.9, 0, .9]) {
            const r = projectQuadratic(a, 1, rho), [x, y] = r.best, g = [x - a + rho * (y - 1), rho * (x - a) + y - 1];
            assert.ok(x >= 0 && y >= 0);
            assert.ok(g.every(v => v >= -1e-10));
            close(x * g[0] + y * g[1], 0);
            assert.ok(r.f(...r.best) <= r.f(...r.naive) + 1e-10);
        }
    assert.ok(projectQuadratic(-1, 1, .7).f(...projectQuadratic(-1, 1, .7).best) < projectQuadratic(-1, 1, .7).f(0, 1));
});
test('zero apparent primal dual difference can hide infeasibility', () => { const r = scalarDual(1, 0, 0); close(r.gap, 0); assert.equal(r.feasible, false); for (const b of [-1, 0, 2]) {
    const x = Math.max(0, b), s = scalarDual(b, x, x);
    close(s.gap, 0);
    close(s.stationarity, 0);
    close(s.complementarity, 0);
} });
test('Huber influence saturates while mean follows an outlier', () => { close(locationFit(12, 'l2').m, 2); close(locationFit(2, 'huber').m, locationFit(12, 'huber').m); const l1 = locationFit(12, 'l1'); close(l1.loss(0), l1.loss(.5)); close(l1.m, .25); });
test('barrier central path has exact 2/t gap and valid KKT stationarity', () => { for (const t of [.1, 1, 10, 1000]) {
    const r = barrierPoint(t);
    assert.ok(r.x > 0 && r.x < 1);
    close(r.gap, 2 / t);
    close(r.stationarity, 0);
    close(t - 1 / r.x + 1 / (1 - r.x), 0, 1e-8);
} });
test('soft threshold solves the subgradient optimality equation', () => { for (const v of [-3, -.5, 0, 1, 3])
    for (const lambda of [0, .5, 2, 3]) {
        const x = softThreshold(v, lambda);
        if (x === 0)
            assert.ok(Math.abs(v) <= lambda);
        else
            close(x - v + lambda * Math.sign(x), 0);
    } });
test('cross entropy decomposes into entropy plus nonnegative KL', () => { for (const p of [0, .2, .5, .9, 1])
    for (const q of [.01, .2, .5, .99]) {
        const r = bernoulliMismatch(p, q);
        assert.ok(r.kl >= -1e-12);
        close(r.cross, r.entropy + r.kl);
    } close(bernoulliMismatch(.2, .2).kl, 0); });
test('typical-set counts aggregate exact strings and total probability', () => { const r = typicalSet(4, .2, .3); close(r.rows.reduce((s, x) => s + x.probability, 0), 1); assert.equal(r.rows.reduce((s, x) => s + x.count, 0), 16); const fair = typicalSet(32, .5, .02); close(fair.mass, 1); close(fair.fraction, 1); });
test('water filling respects power and equals or improves uniform allocation', () => { for (const p of [0, 1, 4, 12])
    for (const n of [1, 4, 10]) {
        const r = waterFill(p, n);
        close(r.allocation[0] + r.allocation[1], p);
        assert.ok(r.capacity >= r.equal - 1e-12);
        if (r.allocation[1] > 0)
            close(r.allocation[0] + 1, r.allocation[1] + n);
    } close(waterFill(2, 4).allocation[1], 0); });
test('rate distortion reverse channel preserves marginal and Hamming distortion', () => { for (const p of [.05, .3, .5])
    for (const fraction of [0, .2, 1]) {
        const d = p * fraction, r = binaryRateDistortion(p, d);
        close(r.joint.reduce((s, x) => s + x, 0), 1);
        close(r.joint[2] + r.joint[3], p);
        close(r.distortion, d);
        assert.ok(r.joint.every(x => x >= 0));
        const py = [1 - r.q, r.q], px = [1 - p, p];
        let mi = 0;
        r.joint.forEach((v, i) => { if (v > 0)
            mi += v * Math.log2(v / (px[Math.floor(i / 2)] * py[i % 2])); });
        close(mi, r.rate);
    } });
test('finite-sample hypothesis errors have correct extreme decisions', () => { close(testingErrors(10, 0).falseAlarm, 1); close(testingErrors(10, 0).miss, 0); close(testingErrors(10, 11).falseAlarm, 0); close(testingErrors(10, 11).miss, 1); assert.ok(testingErrors(30, 15).falseAlarm < testingErrors(10, 5).falseAlarm); });
test('all chapter math demos render both languages at every individual control endpoint', () => { for (const e of [...chapterOptimizationExperiments, ...chapterInformationExperiments]) {
    const base = Object.fromEntries(e.controls.map(c => [c.key, c.value]));
    for (const lang of ['en', 'zh-tw'] as const) {
        for (const c of e.controls) {
            const values = c.kind === 'range' ? [c.min!, c.max!] : c.options!.map(o => o.value);
            for (const v of values) {
                const r = e.compute({ ...base, [c.key]: v }, lang);
                assert.ok(!/NaN|Infinity|undefined/.test(r.svg), e.id);
                assert.ok(r.summary.trim().length > 0, e.id);
                assert.ok(r.metrics.length > 0);
            }
        }
    }
} });
