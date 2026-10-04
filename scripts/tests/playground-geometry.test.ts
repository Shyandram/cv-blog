import assert from 'node:assert/strict';
import test from 'node:test';
import { transformPoint, geometryExperiments } from '../../src/lib/playground/geometry.ts';

test('rotation, reflection and shear preserve the stated determinant', () => {
  for (const angle of [-180, -35, 0, 90, 180]) for (const stretch of [-2, 0, 0.3, 2]) {
    const a = transformPoint(1,0,angle,stretch,0.7), b = transformPoint(0,1,angle,stretch,0.7);
    assert.ok(Math.abs(a[0]*b[1]-a[1]*b[0]-stretch)<1e-12);
  }
});
test('zero horizontal stretch destroys horizontal information', () => {
  assert.deepEqual(transformPoint(1,2,30,0,0.5),transformPoint(8,2,30,0,0.5));
  const result=geometryExperiments[0].compute({angle:0,stretch:0,shear:0},'en');
  assert.equal(result.metrics[2].value,'1');
  assert.ok(!/NaN|Infinity/.test(result.svg));
});
