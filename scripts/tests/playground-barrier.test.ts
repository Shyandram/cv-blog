import test from 'node:test';
import assert from 'node:assert/strict';
import { polygon, centralPoint, analyticCenter, barrierState, centralPath, optimum, slacks, type Vec } from '../../src/lib/playground/barrier.ts';

const P = polygon([[-1.5, -0.9], [0.6, -1.2], [1.6, 0.0], [0.5, 1.3], [-1.2, 0.9]]);
const dirs = [0, 35, 90, 200, 290].map(d => [Math.cos(d * Math.PI / 180), Math.sin(d * Math.PI / 180)] as Vec);

test('polygon constraints bound the vertices and have unit normals', () => {
  for (const a of P.A) assert.ok(Math.abs(Math.hypot(...a) - 1) < 1e-12);
  for (const v of P.vertices) assert.ok(slacks(P, v).every(s => s > -1e-12));
  assert.ok(slacks(P, [0, 0]).every(s => s > 0));
});

test('the central point satisfies stationarity and stays strictly feasible', () => {
  const center = analyticCenter(P, [0, 0]);
  for (const c of dirs) for (const t of [0.1, 1, 10, 1e3, 1e4]) {
    const r = centralPoint(P, c, t, center);
    assert.ok(slacks(P, r.x).every(s => s > 0));
    assert.ok(r.gradientNorm < 1e-6 * Math.max(1, t), `gradient norm ${r.gradientNorm} at t=${t}`);
  }
});

test('the duality gap of the central point is exactly m/t and bounds the suboptimality', () => {
  const center = analyticCenter(P, [0, 0]);
  for (const c of dirs) for (const t of [0.1, 1, 10, 1e3]) {
    const s = barrierState(P, c, t, center);
    assert.ok(Math.abs(s.gap - P.A.length / t) < 1e-8 * Math.max(1, P.A.length / t), `gap ${s.gap} vs ${P.A.length / t}`);
    assert.ok(s.suboptimality <= s.bound + 1e-9 && s.suboptimality >= -1e-9);
    // dual feasibility: c + Σ λ_i a_i = 0
    const r0 = c[0] + s.lambda.reduce((a, l, i) => a + l * P.A[i][0], 0), r1 = c[1] + s.lambda.reduce((a, l, i) => a + l * P.A[i][1], 0);
    assert.ok(Math.hypot(r0, r1) < 1e-6);
  }
});

test('the central path runs from the analytic center to the optimal vertex', () => {
  const center = analyticCenter(P, [0, 0]);
  for (const c of dirs) {
    const path = centralPath(P, c, -1, 3.5, 40, center), opt = optimum(P, c);
    const objective = path.map(x => c[0] * x[0] + c[1] * x[1]);
    assert.ok(objective.every((v, i) => i === 0 || v <= objective[i - 1] + 1e-9), 'objective decreases along the path');
    assert.ok(Math.hypot(path.at(-1)![0] - opt.vertex[0], path.at(-1)![1] - opt.vertex[1]) < 0.01);
  }
});
