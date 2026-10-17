// diffusion-targets: the lab's own pipeline on the real astronaut-256.png asset, against the NumPy / scikit-image golden
// (scripts/figures/playground_goldens_diffusion.py -> scripts/tests/golden/diffusion.json).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { readRgb, asset } from './helpers/png.ts';
import { normalField } from '../../src/lib/imagekit/rng.ts';
import {
  SEEDS, TARGETS, alphaFromSnr, toSignal, forward, perturbed, targetField, recoverX0, amplification, evaluate, pearson, lab as diffusionLab, type Target,
} from '../../src/lib/playground/labs/diffusion-targets.ts';

const dgold = JSON.parse(readFileSync(new URL('./golden/diffusion.json', import.meta.url), 'utf8'));
const astro = readRgb(asset('astronaut-256.png'));
const x0 = toSignal(astro), eps = normalField(x0.length, SEEDS.eps), delta = normalField(x0.length, SEEDS.err);

test('the seeded noise fields match the NumPy port of the same generator', () => {
  assert.equal(x0.length, 256 * 256 * 3);
  dgold.epsHead.forEach((v: number, i: number) => assert.ok(Math.abs(eps[i] - v) < 1e-5, `eps[${i}]`));
  dgold.deltaHead.forEach((v: number, i: number) => assert.ok(Math.abs(delta[i] - v) < 1e-5, `delta[${i}]`));
  const mean = eps.reduce((s, v) => s + v, 0) / eps.length;
  assert.ok(Math.abs(mean - dgold.epsMean) < 1e-6);
  const mx = x0.reduce((s, v) => s + v, 0) / x0.length;
  assert.ok(Math.abs(mx - dgold.x0Mean) < 1e-6);
});

test('PSNR, error multiplier and correlations equal the golden values at every tested level, error and target', () => {
  for (const row of dgold.rows) {
    const alpha = alphaFromSnr(row.snr), f = forward(x0, eps, alpha);
    assert.ok(Math.abs(alpha - row.alpha) < 1e-12);
    assert.ok(Math.abs(pearson(f.xt, f.eps) - row.corrXtEps) < 1e-4);
    assert.ok(Math.abs(pearson(f.xt, f.x0) - row.corrXtX0) < 1e-4);
    assert.ok(Math.abs(pearson(f.v, f.x0) - row.corrVx0) < 1e-4);
    assert.ok(Math.abs(pearson(f.v, f.eps) - row.corrVEps) < 1e-4);
    for (const r of row.runs) {
      const e = evaluate(f, delta, alpha, r.err, r.kind as Target);
      assert.ok(Math.abs(e.psnr - r.psnr) < 0.02 || (e.psnr > 60 && r.psnr > 60), `snr ${row.snr} ${r.kind} s=${r.err}: ${e.psnr} vs ${r.psnr}`);
      assert.ok(Math.abs(e.amplification - r.theoryAmplification) < 1e-12);
      if (r.err > 0) assert.ok(Math.abs(e.measured - r.measuredAmplification) < 2e-3 * Math.max(1, r.measuredAmplification), `measured amplification ${e.measured} vs ${r.measuredAmplification}`);
    }
  }
});

test('the three targets convert exactly into each other (chapter ML 07 identities); the error multipliers are b/a, 1 and b', () => {
  const alpha = alphaFromSnr(-10), f = forward(x0, eps, alpha), a = Math.sqrt(alpha), b = Math.sqrt(1 - alpha);
  for (const kind of TARGETS) {
    const back = recoverX0(kind, targetField(f, kind), f.xt, alpha);
    let worst = 0;
    for (let i = 0; i < back.length; i += 7) worst = Math.max(worst, Math.abs(back[i] - x0[i]));
    assert.ok(worst < 1e-4, `${kind} round trip ${worst}`);
  }
  // eps = b x_t + a v  and  x0 = a x_t - b v
  for (let i = 0; i < x0.length; i += 997) { assert.ok(Math.abs(b * f.xt[i] + a * f.v[i] - eps[i]) < 1e-5); assert.ok(Math.abs(a * f.xt[i] - b * f.v[i] - x0[i]) < 1e-5); }
  assert.ok(Math.abs(amplification('eps', alpha) - b / a) < 1e-12 && amplification('x0', alpha) === 1 && Math.abs(amplification('v', alpha) - b) < 1e-12);
  // The chapter's code snippet: with abar = 0.2 the clean/noise MSE ratio is b^2 / a^2 = 4.
  assert.ok(Math.abs(amplification('eps', 0.2) ** 2 - 4) < 1e-12);
  const p = perturbed(eps, delta, 0.1);
  assert.ok(Math.abs(p[5] - (eps[5] + 0.1 * delta[5])) < 1e-6);
});

test('facts quoted in the lab text (predict explanation)', () => {
  const at = (snr: number, s: number) => {
    const alpha = alphaFromSnr(snr), f = forward(x0, eps, alpha);
    return { f, runs: Object.fromEntries(TARGETS.map(t => [t, evaluate(f, delta, alpha, s, t)])) as Record<Target, ReturnType<typeof evaluate>> };
  };
  const low = at(-20, 0.05), r1 = (v: number) => v.toFixed(1);
  assert.equal(amplification('eps', alphaFromSnr(-20)).toFixed(1), '10.0');
  assert.equal(amplification('v', alphaFromSnr(-20)).toFixed(3), '0.995');
  assert.deepEqual([r1(low.runs.eps.psnr), r1(low.runs.x0.psnr), r1(low.runs.v.psnr)], ['13.5', '32.4', '32.4']);
  assert.equal(pearson(low.f.xt, low.f.eps).toFixed(3), '0.998');
  assert.equal(pearson(low.f.v, low.f.x0).toFixed(3), '-0.988');
  const high = at(20, 0.05);
  assert.deepEqual([r1(high.runs.eps.psnr), r1(high.runs.x0.psnr), r1(high.runs.v.psnr)], ['52.3', '32.4', '52.3']);
  // The predict answer: at -20 dB the noise route is the worst; at +20 dB the clean-image route is.
  const worst = (r: Record<Target, { psnr: number }>) => TARGETS.reduce((m, t) => (r[t].psnr < r[m].psnr ? t : m), 'eps' as Target);
  assert.equal(worst(low.runs), 'eps');
  assert.equal(worst(high.runs), 'x0');
  // Crossover: at 0 dB (abar = 0.5) the noise route has multiplier 1.
  assert.ok(Math.abs(amplification('eps', alphaFromSnr(0)) - 1) < 1e-12);
});

test('default state shows the idea; zero error recovers x0 almost exactly; the error slider moves the metric', () => {
  const def = Object.fromEntries(diffusionLab.controls.map(c => [c.key, c.value]));
  assert.deepEqual(def, { snr: -20, target: 'eps', err: 0.05 });
  assert.equal(diffusionLab.controls.length, 3);
  assert.ok(diffusionLab.predict.options.some(o => o.value === diffusionLab.predict.answer));
  const alpha = alphaFromSnr(-20), f = forward(x0, eps, alpha);
  assert.ok(evaluate(f, delta, alpha, 0.05, 'x0').psnr - evaluate(f, delta, alpha, 0.05, 'eps').psnr > 15);
  assert.ok(evaluate(f, delta, alpha, 0, 'eps').psnr > 60, 'a perfect eps converts back to x0 up to float32 rounding');
  assert.ok(evaluate(f, delta, alpha, 0.1, 'eps').psnr < evaluate(f, delta, alpha, 0.05, 'eps').psnr - 2, 'clipping to [-1, 1] limits the drop, but doubling the error still lowers PSNR clearly');
});
