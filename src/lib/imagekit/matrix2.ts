// Small 2 x 2 linear algebra for the matrix-warp lab: determinant, inverse, product, a closed-form SVD and polygon area.
// A matrix A = [[a, b], [c, d]] is stored row-major as [a, b, c, d]; its columns A e1 = (a, c) and A e2 = (b, d) are the images of the basis vectors.
export type Mat2 = readonly [number, number, number, number];
export type Vec2 = readonly [number, number];

export const IDENTITY: Mat2 = [1, 0, 0, 1];
export const det2 = (m: Mat2): number => m[0] * m[3] - m[1] * m[2];
export const apply2 = (m: Mat2, x: number, y: number): Vec2 => [m[0] * x + m[1] * y, m[2] * x + m[3] * y];
export const mul2 = (p: Mat2, q: Mat2): Mat2 => [
  p[0] * q[0] + p[1] * q[2], p[0] * q[1] + p[1] * q[3],
  p[2] * q[0] + p[3] * q[2], p[2] * q[1] + p[3] * q[3],
];
/** Rotation by `deg` degrees, counter-clockwise when y points up. */
export function rotation2(deg: number): Mat2 {
  const t = (deg * Math.PI) / 180, c = Math.cos(t), s = Math.sin(t);
  return [c, -s, s, c];
}
/** Inverse by the adjugate formula; returns null when the determinant is exactly zero. */
export function inv2(m: Mat2): Mat2 | null {
  const d = det2(m);
  return d === 0 ? null : [m[3] / d, -m[1] / d, -m[2] / d, m[0] / d];
}
/** Signed area of a polygon by the shoelace formula (positive for a counter-clockwise vertex order when y points up). */
export function polygonArea(points: readonly Vec2[]): number {
  let s = 0;
  for (let i = 0; i < points.length; i++) {
    const [x0, y0] = points[i], [x1, y1] = points[(i + 1) % points.length];
    s += x0 * y1 - x1 * y0;
  }
  return s / 2;
}
/** The unit square 0, e1, e1 + e2, e2 pushed through A (counter-clockwise before the map). */
export const unitSquareImage = (m: Mat2): Vec2[] => [[0, 0], [1, 0], [1, 1], [0, 1]].map(([x, y]) => apply2(m, x, y));

/** Matrices with a smaller second singular value than this (relative to the first) count as singular (rank deficient). */
export const SINGULAR_TOL = 1e-9;
export interface Svd2 {
  /** Singular values, s1 >= s2 >= 0. */
  s1: number; s2: number;
  /** Rank (0, 1 or 2) with the relative tolerance SINGULAR_TOL. */
  rank: 0 | 1 | 2;
  /** Condition number s1 / s2 (Infinity when the matrix is singular). */
  cond: number;
  /** Orthonormal right singular vectors (input directions) and left singular vectors (output directions), A v_i = s_i u_i. */
  v1: Vec2; v2: Vec2; u1: Vec2; u2: Vec2;
}
/**
 * Closed-form SVD of a 2 x 2 matrix (J. Blinn, 1996). With E = (a + d) / 2, F = (a - d) / 2, G = (c + b) / 2, H = (c - b) / 2 and
 * Q = hypot(E, H), R = hypot(F, G): s1 = Q + R and the signed second value is Q - R (its sign is the sign of det A, since det A = Q^2 - R^2).
 * A = Rot(phi) diag(Q + R, Q - R) Rot(theta) with theta = (atan2(H, E) - atan2(G, F)) / 2 and phi = (atan2(H, E) + atan2(G, F)) / 2.
 */
export function svd2(m: Mat2): Svd2 {
  const [a, b, c, d] = m;
  const E = (a + d) / 2, F = (a - d) / 2, G = (c + b) / 2, H = (c - b) / 2;
  const Q = Math.hypot(E, H), R = Math.hypot(F, G);
  const a1 = Math.atan2(G, F), a2 = Math.atan2(H, E), theta = (a2 - a1) / 2, phi = (a2 + a1) / 2;
  const s1 = Q + R, signed2 = Q - R, s2 = Math.abs(signed2), flip = signed2 < 0 ? -1 : 1;
  const cp = Math.cos(phi), sp = Math.sin(phi), ct = Math.cos(theta), st = Math.sin(theta);
  const rank = s1 <= 0 ? 0 : s2 > SINGULAR_TOL * s1 ? 2 : 1;
  return {
    s1, s2, rank, cond: rank === 2 ? s1 / s2 : Infinity,
    v1: [ct, -st], v2: [st, ct], u1: [cp, sp], u2: [-flip * sp, flip * cp],
  };
}
/** The image of the unit circle: an ellipse with semi-axes s1 along u1 and s2 along u2. Returns the point at parameter t (radians). */
export function ellipsePoint(m: Mat2, t: number): Vec2 {
  return apply2(m, Math.cos(t), Math.sin(t));
}
/** Largest and smallest of |A x| over `n` evenly spaced unit vectors x (a brute-force check of the singular values). */
export function stretchRange(m: Mat2, n = 3600): { min: number; max: number } {
  let lo = Infinity, hi = 0;
  for (let i = 0; i < n; i++) {
    const t = (2 * Math.PI * i) / n, [x, y] = apply2(m, Math.cos(t), Math.sin(t)), r = Math.hypot(x, y);
    if (r < lo) lo = r;
    if (r > hi) hi = r;
  }
  return { min: lo, max: hi };
}
