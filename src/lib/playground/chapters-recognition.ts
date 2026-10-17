import { L, pick, num, fmt, svg, text, path } from './types.ts';
import type { Experiment } from './types.ts';

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
const normal = (x: number, mu = 0, sigma = 1) => Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
const sequence = (start: number, end: number, count = 161) => Array.from({ length: count }, (_, i) => start + (end - start) * i / (count - 1));
const axes = () => '<path d="M55,35 V240 H600" stroke="var(--plot-grid)" fill="none"/>';
const dot = (x: number, y: number, color: string, radius = 4) => `<circle cx="${x}" cy="${y}" r="${radius}" fill="var(--plot-${color})"/>`;
const stroke = (color: string) => `stroke="var(--plot-${color})" stroke-width="2.5"`;
const related = (slug: string) => [`pattern-recognition/${slug}`];

export const DENSITY_SAMPLES = [-2.3, -2.1, -1.8, -1.4, -1.1, 0.7, 1.1, 1.4, 1.7, 2.2];
export function kernelDensity(x: number, bandwidth: number, samples = DENSITY_SAMPLES) {
  return mean(samples.map(v => normal(x, v, bandwidth)));
}
export function densityScore(bandwidth: number) {
  return mean(DENSITY_SAMPLES.map((x, i) => Math.log(kernelDensity(x, bandwidth, DENSITY_SAMPLES.filter((_, j) => i !== j)))));
}

export const REGRESSION_X = sequence(-1, 1, 12);
export const REGRESSION_Y = REGRESSION_X.map((x, i) => Math.sin(Math.PI * x) + 0.25 * Math.sin(i * 8.3));
function solve(matrix: number[][], rhs: number[]) {
  const rows = matrix.map((row, i) => [...row, rhs[i]]), n = rhs.length;
  for (let j = 0; j < n; j++) {
    let pivot = j;
    for (let i = j + 1; i < n; i++) if (Math.abs(rows[i][j]) > Math.abs(rows[pivot][j])) pivot = i;
    [rows[j], rows[pivot]] = [rows[pivot], rows[j]];
    const divisor = rows[j][j];
    if (Math.abs(divisor) < 1e-14) throw new Error('Singular teaching system');
    for (let k = j; k <= n; k++) rows[j][k] /= divisor;
    for (let i = 0; i < n; i++) if (i !== j) {
      const factor = rows[i][j];
      for (let k = j; k <= n; k++) rows[i][k] -= factor * rows[j][k];
    }
  }
  return rows.map(row => row[n]);
}
export function ridgeFit(degree: number, lambda: number) {
  const design = REGRESSION_X.map(x => Array.from({ length: degree + 1 }, (_, j) => x ** j));
  const gram = Array.from({ length: degree + 1 }, (_, j) => Array.from({ length: degree + 1 }, (_, k) => design.reduce((s, row) => s + row[j] * row[k], 0) + (j === k && j > 0 ? lambda : 0)));
  const rhs = Array.from({ length: degree + 1 }, (_, j) => design.reduce((s, row, i) => s + row[j] * REGRESSION_Y[i], 0));
  const weights = solve(gram, rhs);
  const predict = (x: number) => weights.reduce((s, w, j) => s + w * x ** j, 0);
  return { weights, predict, trainingMse: mean(REGRESSION_X.map((x, i) => (predict(x) - REGRESSION_Y[i]) ** 2)), functionMse: mean(sequence(-1, 1, 101).map(x => (predict(x) - Math.sin(Math.PI * x)) ** 2)), penalty: weights.slice(1).reduce((s, w) => s + w * w, 0) };
}


export function projectionStats(degrees: number) {
  const angle = degrees * Math.PI / 180, u = [Math.cos(angle), Math.sin(angle)];
  const points = [-3, -2, -1, 1, 2, 3].flatMap(x => [-1, 1].map(y => ({ x, y, label: y > 0 ? 1 : 0, score: x * u[0] + y * u[1] })));
  const variance = mean(points.map(p => p.score ** 2));
  const totalVariance = mean(points.map(p => p.x ** 2 + p.y ** 2));
  const gap = Math.abs(mean(points.filter(p => p.label === 1).map(p => p.score)) - mean(points.filter(p => p.label === 0).map(p => p.score)));
  return { u, points, variance, totalVariance, gap, reconstructionMse: totalVariance - variance };
}

export function alignSequences(warp: number, band: number) {
  const a = sequence(0, 1, 12).map(t => Math.sin(2 * Math.PI * t));
  const b = sequence(0, 1, 12).map(t => Math.sin(2 * Math.PI * t ** warp));
  const n = a.length, accumulated = Array.from({ length: n }, () => Array(n).fill(Infinity) as number[]);
  const costs = a.map(x => b.map(y => (x - y) ** 2));
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    if (Math.abs(i - j) > band) continue;
    if (i === 0 && j === 0) accumulated[i][j] = costs[i][j];
    else accumulated[i][j] = costs[i][j] + Math.min(i && j ? accumulated[i - 1][j - 1] : Infinity, i ? accumulated[i - 1][j] : Infinity, j ? accumulated[i][j - 1] : Infinity);
  }
  let i = n - 1, j = n - 1;
  const route: [number, number][] = [[i, j]];
  while (i || j) {
    const candidates: [number, number, number][] = [];
    if (i && j) candidates.push([accumulated[i - 1][j - 1], i - 1, j - 1]);
    if (i) candidates.push([accumulated[i - 1][j], i - 1, j]);
    if (j) candidates.push([accumulated[i][j - 1], i, j - 1]);
    candidates.sort((a, b) => a[0] - b[0]);
    [, i, j] = candidates[0]; route.push([i, j]);
  }
  return { a, b, costs, accumulated, route: route.reverse(), dtwCost: accumulated[n - 1][n - 1], diagonalCost: costs.reduce((s, row, i) => s + row[i], 0) };
}

export const MIXTURE_SAMPLES = [-2.5, -2.2, -1.9, -1.6, -1.1, 0.8, 1.2, 1.5, 1.9, 2.3];
export function mixtureEM(separation: number, sigma: number, steps: number) {
  let means = [-separation / 2, separation / 2];
  const likelihood = () => MIXTURE_SAMPLES.reduce((s, x) => s + Math.log((normal(x, means[0], sigma) + normal(x, means[1], sigma)) / 2), 0);
  const responsibilities = () => MIXTURE_SAMPLES.map(x => {
    const a = normal(x, means[0], sigma), b = normal(x, means[1], sigma); return b / (a + b);
  });
  const history = [likelihood()];
  for (let step = 0; step < steps; step++) {
    const r = responsibilities();
    means = [0, 1].map(k => {
      const weights = r.map(p => k ? p : 1 - p);
      return weights.reduce((s, w, i) => s + w * MIXTURE_SAMPLES[i], 0) / weights.reduce((a, b) => a + b, 0);
    });
    history.push(likelihood());
  }
  return { means, history, responsibilities: responsibilities() };
}

export function variationalComparison(mu: number, sigma: number, separation: number) {
  const xs = sequence(-12, 12, 1201), dx = xs[1] - xs[0];
  const p = xs.map(x => (normal(x, -separation, 0.65) + normal(x, separation, 0.65)) / 2);
  const q = xs.map(x => normal(x, mu, sigma));
  const pm = p.reduce((a, b) => a + b, 0) * dx, qm = q.reduce((a, b) => a + b, 0) * dx;
  const pn = p.map(v => v / pm), qn = q.map(v => v / qm);
  const kl = (a: number[], b: number[]) => Math.max(0, a.reduce((s, v, i) => s + v * Math.log(Math.max(v, 1e-300) / Math.max(b[i], 1e-300)) * dx, 0));
  return { xs, p: pn, q: qn, forward: kl(pn, qn), reverse: kl(qn, pn), pMass: pm, qMass: qm };
}
export function rejectionUnderShift(threshold: number, shift: number) {
  const xs = sequence(-9, 9, 3601), dx = xs[1] - xs[0];
  let accepted = 0, errors = 0, total = 0;
  for (const x of xs) {
    const mass0 = 0.5 * normal(x, -1 + shift), mass1 = 0.5 * normal(x, 1 + shift), mass = mass0 + mass1;
    const confidence = 1 / (1 + Math.exp(-2 * Math.abs(x)));
    total += mass * dx;
    if (confidence >= threshold) { accepted += mass * dx; errors += (x >= 0 ? mass0 : mass1) * dx; }
  }
  return { coverage: accepted / total, risk: accepted ? errors / accepted : null, acceptedErrorMass: errors / total };
}

export const chapterRecognitionExperiments: Experiment[] = [
  {
    id: 'density-bandwidth', track: 'pattern-recognition',
    title: L('PR 02 · How wide should a density kernel be?', 'PR 02・密度核函數應該多寬？'),
    description: L('Change the bandwidth and inspect individual kernels, the estimated density and leave-one-out prediction.', '調整頻寬，觀察個別核函數、估計密度與留一法預測。'),
    challenge: L('Find a bandwidth that improves the leave-one-out log score. Why can a spiky fit score worse on a held-out point?', '找出能提高留一法對數分數的頻寬。為什麼尖銳的密度曲線，在保留觀測點上的分數反而較低？'),
    note: L('Ten fixed synthetic observations; Gaussian kernels integrate to one. The score averages log density at each point after excluding that point. It is a density score in nats, not classification accuracy. The plot shows individual contributions divided by ten.', '使用十個固定的合成觀測值；Gaussian 核函數的積分為一。分數是在排除每個點後，計算該點的平均對數密度，單位為 nat，不是分類準確率。圖中個別核函數已除以十。'),
    related: related('pr-02-density-estimation'),
    predict: { question: L("Which bandwidth gives the best held-out log-likelihood: h = 0.1, 0.5 or 2?", "哪一個頻寬的留一法對數概似最好：h = 0.1、0.5 或 2？"), options: [{ value: "0.1", label: L("h = 0.1", "h = 0.1") }, { value: "0.5", label: L("h = 0.5", "h = 0.5") }, { value: "2", label: L("h = 2", "h = 2") }], answer: "0.5", explain: L("A small h leaves each held-out point almost alone (score about −5.9), a large h blurs the two clusters together (about −2.2), and a moderate h scores best (about −1.8).", "h 太小會讓每個被留出的點幾乎孤立（分數約 −5.9）；h 太大會把兩群模糊在一起（約 −2.2）；適中的 h 分數最好（約 −1.8）。") },
    controls: [{ key: 'bandwidth', label: L('Kernel bandwidth h', '核函數頻寬 h'), kind: 'range', min: 0.1, max: 2, step: 0.05, value: 0.5 }],
    compute(state, lang) {
      const h = num(state, 'bandwidth'), xs = sequence(-5, 5), values = xs.map(x => kernelDensity(x, h));
      const max = Math.max(...values) * 1.15, px = (x: number) => 55 + (x + 5) * 54.5, py = (y: number) => 240 - y / max * 200;
      let body = axes();
      for (const sample of DENSITY_SAMPLES) body += path(xs.map(x => [px(x), py(normal(x, sample, h) / 10)]), `${stroke('b')} opacity="0.3"`) + dot(px(sample), 240, 'c');
      body += path(xs.map((x, i) => [px(x), py(values[i])]), stroke('a'));
      for (const x of [-4, -2, 0, 2, 4]) body += text(px(x), 263, x, 'text-anchor="middle"');
      body += text(55, 22, pick(lang, 'Density (vertical scale adapts)', '機率密度（縱軸範圍隨之調整）')) + text(590, 291, 'x', 'text-anchor="end"');
      return { svg: svg(body), summary: pick(lang, 'Small bandwidth preserves local bumps; large bandwidth merges them. Leave-one-out evaluation prevents each sample from predicting itself.', '小頻寬保留局部隆起，大頻寬將它們合併。留一法避免每個觀測值用自己預測自己。'), metrics: [{ label: pick(lang, 'Mean leave-one-out log density', '留一法平均對數密度'), value: fmt(densityScore(h)) }, { label: pick(lang, 'Peak density', '密度峰值'), value: fmt(Math.max(...values)) }] };
    },
  },
  {
    id: 'ridge-polynomial', track: 'pattern-recognition',
    title: L('PR 03 · Complexity versus shrinkage', 'PR 03・模型複雜度與收縮'),
    description: L('Fit polynomial basis functions with ridge regression and separate training error from recovery of the underlying function.', '以嶺迴歸擬合多項式基底，區分訓練誤差與原始函數的還原誤差。'),
    challenge: L('At degree seven, increase regularization. Does the lowest training error also give the lowest underlying-function error?', '使用七次多項式並增加正則化。最低訓練誤差，是否也對應最低的原始函數誤差？'),
    note: L('The blue fit solves sum of squared residuals + λ times the squared non-intercept coefficients. Twelve fixed synthetic points have deterministic perturbations. The dashed curve is sin(πx); its grid error is available only because this is a known synthetic problem. Polynomial features are linear in their coefficients.', '藍線最小化殘差平方和加上 λ 乘以非截距係數的平方和。十二個合成點加入固定擾動。虛線為 sin(πx)；只有在這個已知合成問題中，才能直接計算它的網格誤差。多項式特徵對係數仍是線性的。'),
    related: related('pr-03-linear-models'),
    predict: { question: L("Set λ = 0 and raise the degree from 5 to 6. Training error falls (0.0285 to 0.0239). What happens to the error against the true function?", "設 λ = 0，把次數從 5 提高到 6。訓練誤差下降（0.0285 到 0.0239）。與真實函數的誤差會怎麼變？"), options: [{ value: "falls", label: L("It falls too", "也下降") }, { value: "rises", label: L("It rises", "上升") }], answer: "rises", explain: L("It rises from about 0.0005 to about 0.0063. The extra flexibility fits the noise in the 12 training points, which lowers the training error and moves the curve away from the true function.", "它從約 0.0005 升到約 0.0063。多出來的彈性擬合了 12 個訓練點中的雜訊，訓練誤差因此下降，曲線卻離真實函數更遠。") },
    controls: [{ key: 'degree', label: L('Polynomial degree', '多項式次數'), kind: 'range', min: 1, max: 7, step: 1, value: 5 }, { key: 'lambda', label: L('Regularization λ', '正則化 λ'), kind: 'range', min: 0, max: 5, step: 0.05, value: 0.1 }],
    compute(state, lang) {
      const result = ridgeFit(num(state, 'degree'), num(state, 'lambda')), xs = sequence(-1, 1), ys = xs.map(result.predict);
      const extent = Math.max(1.5, ...ys.map(Math.abs)), px = (x: number) => 55 + (x + 1) * 272.5, py = (y: number) => 137.5 - y / extent * 100;
      let body = axes() + path(xs.map(x => [px(x), py(Math.sin(Math.PI * x))]), `${stroke('b')} stroke-dasharray="5 4"`) + path(xs.map((x, i) => [px(x), py(ys[i])]), stroke('a'));
      REGRESSION_X.forEach((x, i) => { body += dot(px(x), py(REGRESSION_Y[i]), 'c'); });
      body += text(55, 22, pick(lang, 'Blue: fitted polynomial · dashed: true function', '藍線：多項式擬合 · 虛線：真實函數'));
      for (const x of [-1, -0.5, 0, 0.5, 1]) body += text(px(x), 265, x, 'text-anchor="middle"');
      return { svg: svg(body), summary: pick(lang, 'Changing λ alters the fitted coefficients, not the observations. Large penalties can remove noise-fitting but also suppress the curvature needed to represent the signal.', '改變 λ 會改變擬合係數，觀測值維持固定。大幅懲罰可抑制對雜訊的擬合，也可能壓低表示訊號所需的曲率。'), metrics: [{ label: pick(lang, 'Training mean squared error', '訓練均方誤差'), value: fmt(result.trainingMse, 4) }, { label: pick(lang, 'True-function grid error', '真實函數網格誤差'), value: fmt(result.functionMse, 4) }, { label: pick(lang, 'Squared coefficient norm', '係數範數平方'), value: fmt(result.penalty) }] };
    },
  },
  {
    id: 'projection-information', track: 'pattern-recognition',
    title: L('PR 05 · Does maximum variance preserve the classes?', 'PR 05・最大變異量能保留類別嗎？'),
    description: L('Rotate a one-dimensional projection and compare reconstruction error with class separation.', '旋轉一維投影方向，比較重建誤差與類別分離程度。'),
    challenge: L('Find the projection that minimizes reconstruction error, then the one that separates the colored classes. Why are they different?', '找出重建誤差最低的投影，再找出能分開兩種類別的投影。為什麼它們不同？'),
    note: L('The centered synthetic observations have horizontal variation within each class and a vertical class difference. Principal component analysis maximizes retained variance without using labels. The centroid gap shown here is descriptive, not a complete classification metric.', '置中的合成觀測值，在每個類別內有水平變化，類別差異則在垂直方向。主成分分析不使用標籤，只最大化保留的變異量。這裡的類別中心距離是描述量，不是完整的分類評估指標。'),
    related: related('pr-05-features-dimensionality'),
    controls: [{ key: 'angle', label: L('Projection angle (degrees)', '投影角度（度）'), kind: 'range', min: 0, max: 180, step: 5, value: 30 }],
    compute(state, lang) {
      const result = projectionStats(num(state, 'angle')), [ux, uy] = result.u;
      let body = path([[80, 145], [560, 145]], 'stroke="var(--plot-grid)"') + path([[320, 30], [320, 260]], 'stroke="var(--plot-grid)"');
      body += path([[320 - ux * 175, 145 + uy * 98], [320 + ux * 175, 145 - uy * 98]], stroke('c'));
      for (const p of result.points) {
        const original: [number, number] = [320 + p.x * 50, 145 - p.y * 28], projected: [number, number] = [320 + p.score * ux * 50, 145 - p.score * uy * 28];
        body += path([original, projected], 'stroke="var(--plot-muted)" stroke-dasharray="3 3"') + dot(...original, p.label ? 'b' : 'a', 5) + dot(...projected, p.label ? 'b' : 'a', 3);
      }
      body += text(55, 22, pick(lang, 'Large: original points · small: reconstructed projections', '大點：原始資料 · 小點：投影後的重建點')) + text(320, 288, pick(lang, 'Horizontal principal component: 0° (x, y use different display scales)', '水平主成分：0°（x、y 使用不同顯示尺度）'), 'text-anchor="middle"');
      return { svg: svg(body), summary: pick(lang, 'At 0°, both classes project to the same six positions. At 90°, within-class horizontal variation disappears and class centers remain distinct.', '在 0° 時，兩種類別投影到相同的六個位置。在 90° 時，類別內的水平變化消失，類別中心則保持分離。'), metrics: [{ label: pick(lang, 'Retained variance fraction', '保留變異量比例'), value: fmt(result.variance / result.totalVariance) }, { label: pick(lang, 'Mean squared reconstruction distance', '平均重建距離平方'), value: fmt(result.reconstructionMse) }, { label: pick(lang, 'Projected class-centroid gap', '投影後的類別中心距離'), value: fmt(result.gap) }] };
    },
  },
  {
    id: 'sequence-alignment', track: 'pattern-recognition',
    title: L('PR 06 · Match shape despite timing changes', 'PR 06・時間不同，如何比對形狀？'),
    description: L('Warp a signal in time and inspect a dynamic time warping cost matrix and its optimal alignment path.', '改變訊號的時間尺度，觀察動態時間校正的成本矩陣與最佳對齊路徑。'),
    challenge: L('Increase the time warp with a zero-width band, then widen the band. What does the lower cost allow the alignment to ignore?', '將帶寬設為零並增加時間扭曲，再放寬帶寬。成本降低時，對齊過程允許忽略了什麼差異？'),
    note: L('Two twelve-sample sine signals: a(t)=sin(2πt), b(t)=sin(2πt^w). Local cost is squared amplitude difference. The path must start and finish at both endpoints; only diagonal, vertical and horizontal forward steps are allowed. Cost is a sum, not normalized by path length.', '兩個各含十二個取樣點的正弦訊號：a(t)=sin(2πt)、b(t)=sin(2πt^w)。局部成本為振幅差的平方。路徑從兩個起點走到兩個終點，只允許向前的對角、垂直與水平步伐。成本為總和，不以路徑長度正規化。'),
    related: related('pr-06-templates-structured-models'),
    controls: [{ key: 'warp', label: L('Time exponent w', '時間指數 w'), kind: 'range', min: 0.5, max: 2.5, step: 0.1, value: 1.7 }, { key: 'band', label: L('Allowed index difference', '允許的索引差'), kind: 'range', min: 0, max: 11, step: 1, value: 3 }],
    compute(state, lang) {
      const band = num(state, 'band'), r = alignSequences(num(state, 'warp'), band);
      let body = text(45, 22, pick(lang, 'Signals (sample index →)', '訊號（取樣索引 →）')) + text(360, 22, pick(lang, 'Local cost matrix', '局部成本矩陣'));
      body += path(r.a.map((v, i) => [45 + i * 22, 130 - v * 65]), stroke('a')) + path(r.b.map((v, i) => [45 + i * 22, 130 - v * 65]), stroke('b'));
      for (let i = 0; i < 12; i++) for (let j = 0; j < 12; j++) body += `<rect x="${360 + j * 18}" y="${35 + i * 18}" width="17" height="17" fill="var(--plot-${Math.abs(i - j) > band ? 'grid' : 'a'})" opacity="${Math.abs(i - j) > band ? 0.35 : 0.08 + r.costs[i][j] / 4 * 0.85}"/>`;
      body += path(r.route.map(([i, j]) => [369 + j * 18, 44 + i * 18]), stroke('c'));
      body += text(45, 250, pick(lang, 'Blue: a · orange: warped b', '藍色：a · 橘色：扭曲後的 b')) + text(360, 275, pick(lang, 'Rows: a · columns: b', '列：a · 欄：b'));
      return { svg: svg(body), summary: pick(lang, 'Dark cells have larger amplitude mismatch; gray cells are forbidden by the band. The path minimizes accumulated allowed cost. A wider band cannot raise the optimum but can permit implausible repetitions.', '深色格代表較大的振幅差；灰色格被帶寬限制排除。路徑最小化允許區域內的累積成本。放寬帶寬不會提高最佳成本，卻可能允許不合理的重複對齊。'), metrics: [{ label: pick(lang, 'Aligned total cost', '對齊後總成本'), value: fmt(r.dtwCost) }, { label: pick(lang, 'Same-index total cost', '相同索引總成本'), value: fmt(r.diagonalCost) }, { label: pick(lang, 'Path cells', '路徑格數'), value: String(r.route.length) }] };
    },
  },
  {
    id: 'mixture-em', track: 'pattern-recognition',
    title: L('PR 07 · Watch soft assignments move the means', 'PR 07・軟分配如何移動平均值？'),
    description: L('Step through Gaussian-mixture EM and observe responsibilities, likelihood and the symmetric-initialization trap.', '逐步執行 Gaussian 混合模型的 EM，觀察責任值、概似值與對稱初始化的陷阱。'),
    challenge: L('Set initial separation to zero and add iterations. Why do the two means remain identical? Restart with a small separation.', '將初始間距設為零並增加迭代次數。為什麼兩個平均值保持相同？再以小間距重新開始。'),
    note: L('A constrained two-component mixture: both weights stay 1/2 and both standard deviations stay at the selected σ. EM updates only the means. Each point is colored by its current responsibility for component 1; these are component memberships, not supervised labels.', '受限制的兩成分混合模型：權重固定為 1/2，兩個標準差固定為選定的 σ。EM 只更新平均值。每個點依成分 1 的目前責任值著色；這是成分歸屬程度，不是監督式類別標籤。'),
    related: related('pr-07-clustering-mixtures'),
    controls: [{ key: 'separation', label: L('Initial mean separation', '初始平均值間距'), kind: 'range', min: 0, max: 4, step: 0.2, value: 1 }, { key: 'sigma', label: L('Fixed standard deviation σ', '固定標準差 σ'), kind: 'range', min: 0.4, max: 2, step: 0.1, value: 0.8 }, { key: 'steps', label: L('EM iterations', 'EM 迭代次數'), kind: 'range', min: 0, max: 20, step: 1, value: 3 }],
    compute(state, lang) {
      const sigma = num(state, 'sigma'), r = mixtureEM(num(state, 'separation'), sigma, num(state, 'steps')), xs = sequence(-4, 4), px = (x: number) => 55 + (x + 4) * 68.125, py = (y: number) => 240 - y * 350;
      let body = axes();
      r.means.forEach((mu, k) => { body += path(xs.map(x => [px(x), py(0.5 * normal(x, mu, sigma))]), stroke(k ? 'b' : 'a')); });
      MIXTURE_SAMPLES.forEach((x, i) => { body += `<circle cx="${px(x)}" cy="240" r="5" fill="var(--plot-a)"/><circle cx="${px(x)}" cy="240" r="5" fill="var(--plot-b)" opacity="${r.responsibilities[i]}"/>`; });
      body += text(55, 22, pick(lang, 'Weighted component densities (each has mass 1/2)', '加權成分密度（各自的積分為 1/2）'));
      for (const x of [-4, -2, 0, 2, 4]) body += text(px(x), 265, x, 'text-anchor="middle"');
      return { svg: svg(body), summary: pick(lang, 'The E step recomputes soft memberships; the M step takes responsibility-weighted means. With identical initial means every responsibility is 1/2, so symmetry cannot break on its own.', 'E 步驟重新計算軟分配；M 步驟計算責任值加權平均。初始平均值相同時，所有責任值都是 1/2，對稱狀態不會自行打破。'), metrics: [{ label: pick(lang, 'Component means', '成分平均值'), value: r.means.map(v => fmt(v)).join(' / ') }, { label: pick(lang, 'Log likelihood (nats)', '對數概似值（nat）'), value: fmt(r.history.at(-1)!) }, { label: pick(lang, 'Gain since initialization', '相對初始化的增加量'), value: fmt(r.history.at(-1)! - r.history[0]) }] };
    },
  },
  {
    id: 'variational-kl', track: 'pattern-recognition',
    title: L('PR 08 · Which direction of KL are you minimizing?', 'PR 08・你最小化的是哪個方向的 KL？'),
    description: L('Move and widen a Gaussian approximation to a two-mode target; compare penalties for missing mass and filling the valley.', '移動並放寬雙峰目標的 Gaussian 近似，比較漏掉機率質量與填滿峰間谷地的懲罰。'),
    challenge: L('With separated modes, compare a narrow approximation on one mode against a wide one covering both. Which KL prefers each?', '當兩個峰分開時，比較集中在單一峰的窄近似與涵蓋兩個峰的寬近似。哪個方向的 KL 偏好哪一種？'),
    note: L('Target p is an equal mixture N(−a,0.65²) and N(a,0.65²). Approximation q is N(μ,σ²). Both are numerically normalized on [−12,12] with spacing 0.02; KL values are grid approximations in nats. This is an adjustable distribution family, not an optimizer.', '目標 p 為 N(−a,0.65²) 與 N(a,0.65²) 的等權混合，近似 q 為 N(μ,σ²)。兩者在 [−12,12] 上以 0.02 間距數值正規化；KL 為網格近似，單位是 nat。這是可調整的分布族，不是最佳化器。'),
    related: related('pr-08-approximate-inference'),
    predict: { question: L("The target has two separated bumps and q is a single Gaussian. Which divergence is smallest for a q that sits on just one bump?", "目標分布有兩個分開的峰，q 是單一高斯。只落在其中一個峰上的 q，哪一種散度比較小？"), options: [{ value: "forward", label: L("Forward KL(p‖q)", "前向 KL(p‖q)") }, { value: "reverse", label: L("Reverse KL(q‖p)", "反向 KL(q‖p)") }], answer: "reverse", explain: L("Reverse KL punishes q for putting mass where p has none, so a narrow q settles on one bump (mode seeking). Forward KL punishes q for missing any of p's mass, so its best q is wide and covers both bumps. In this setup the best forward fit has mean 0 and σ about 2.1; the best reverse fit sits on a bump at ±2 with σ about 0.7.", "反向 KL 會懲罰 q 把機率放在 p 沒有質量的地方，因此窄的 q 會落在單一峰上（偏向單一模式）。前向 KL 則懲罰 q 漏掉 p 的任何質量，所以最佳的 q 又寬又涵蓋兩個峰。在這個設定下，前向最佳解的平均為 0、σ 約 2.1；反向最佳解落在 ±2 的峰上，σ 約 0.7。") },
    controls: [{ key: 'separation', label: L('Target half-separation a', '目標峰間距的一半 a'), kind: 'range', min: 0, max: 3, step: 0.2, value: 2 }, { key: 'mu', label: L('Approximation mean μ', '近似平均值 μ'), kind: 'range', min: -3, max: 3, step: 0.1, value: -2 }, { key: 'sigma', label: L('Approximation standard deviation σ', '近似標準差 σ'), kind: 'range', min: 0.3, max: 3, step: 0.05, value: 0.65 }],
    compute(state, lang) {
      const r = variationalComparison(num(state, 'mu'), num(state, 'sigma'), num(state, 'separation'));
      const px = (x: number) => 55 + (x + 6) / 12 * 545, py = (v: number) => 240 - v / Math.max(...r.p, ...r.q) * 195;
      let body = axes();
      for (const [values, color] of [[r.p, 'a'], [r.q, 'b']] as const) body += path(r.xs.flatMap((x, i) => Math.abs(x) <= 6 ? [[px(x), py(values[i])] as [number, number]] : []), stroke(color));
      body += text(55, 22, pick(lang, 'Blue: target p · orange: approximation q', '藍色：目標 p · 橘色：近似 q'));
      for (const x of [-6, -3, 0, 3, 6]) body += text(px(x), 265, x, 'text-anchor="middle"');
      return { svg: svg(body), summary: pick(lang, 'KL(p‖q) weights mismatch where the target has mass; KL(q‖p) weights mismatch where the approximation puts mass. A single Gaussian may cover both modes or settle on one, depending on the objective.', 'KL(p‖q) 依目標的機率質量加權差異；KL(q‖p) 依近似分布配置的質量加權差異。單一 Gaussian 可能涵蓋兩個峰，也可能集中在一個峰，取決於目標函數。'), metrics: [{ label: 'KL(p ‖ q)', value: fmt(r.forward) }, { label: 'KL(q ‖ p)', value: fmt(r.reverse) }] };
    },
  },
  {
    id: 'rejection-shift', track: 'pattern-recognition',
    title: L('PR 10 · Does confidence survive a shifted population?', 'PR 10・母體偏移後，信心還可靠嗎？'),
    description: L('Set a rejection threshold, shift both class distributions and compare accepted coverage with selective error.', '設定拒絕門檻並平移兩種類別分布，比較接受比例與選擇性錯誤率。'),
    challenge: L('Hold the rejection threshold fixed while shifting the population. Can high reported confidence coexist with more accepted mistakes?', '固定拒絕門檻並平移母體。模型回報高信心時，接受的錯誤是否仍可能增加？'),
    note: L('The fixed classifier uses training posterior sigmoid(2x), learned for equal-prior N(−1,1) and N(1,1). Deployment shifts both means by δ without updating the classifier. Coverage and accepted error are numerical integrals over the synthetic deployment population, not estimates from a benchmark.', '固定分類器使用訓練後驗機率 sigmoid(2x)，對應等先驗的 N(−1,1) 與 N(1,1)。部署時將兩個平均值都平移 δ，分類器保持固定。接受比例與接受後錯誤率是合成部署母體上的數值積分，不是基準資料集的估計值。'),
    related: related('pr-10-recognition-capstone'),
    controls: [{ key: 'threshold', label: L('Required model confidence', '所需模型信心'), kind: 'range', min: 0.5, max: 1, step: 0.05, value: 0.8 }, { key: 'shift', label: L('Deployment mean shift δ', '部署平均值偏移 δ'), kind: 'range', min: -2, max: 2, step: 0.1, value: 0 }],
    compute(state, lang) {
      const threshold = num(state, 'threshold'), shift = num(state, 'shift'), r = rejectionUnderShift(threshold, shift), xs = sequence(-6, 6), px = (v: number) => 55 + (v + 6) / 12 * 545, py = (v: number) => 240 - v * 440;
      const edge = threshold === 1 ? 6 : Math.min(6, Math.log(threshold / (1 - threshold)) / 2);
      let body = `<rect x="${px(-edge)}" y="35" width="${px(edge) - px(-edge)}" height="205" fill="var(--plot-muted)" opacity="0.13"/>` + axes();
      for (const [mu, color] of [[-1 + shift, 'a'], [1 + shift, 'b']] as const) body += path(xs.map(x => [px(x), py(normal(x, mu))]), stroke(color));
      body += path([[px(0), 35], [px(0), 240]], 'stroke="var(--plot-c)" stroke-dasharray="4 3"') + text(55, 22, pick(lang, 'Deployment densities · gray: model rejects', '部署密度 · 灰色：模型拒絕判斷'));
      for (const x of [-6, -3, 0, 3, 6]) body += text(px(x), 265, x, 'text-anchor="middle"');
      return { svg: svg(body), summary: threshold === 1 ? pick(lang, 'At confidence 1, every finite observation is rejected. Coverage is zero and selective error is undefined, not zero.', '信心門檻為 1 時，所有有限觀測值都被拒絕。接受比例為零，選擇性錯誤率未定義，而不是零。') : pick(lang, 'The classifier still places its decision boundary at zero, while the deployment-optimal boundary moves with δ. Rejecting uncertain predictions does not repair confidence under shift.', '分類器的決策邊界仍在零，部署母體的最佳邊界卻隨 δ 移動。拒絕不確定的預測，並不能修復分布偏移後的信心。'), metrics: [{ label: pick(lang, 'Accepted coverage', '接受比例'), value: fmt(r.coverage) }, { label: pick(lang, 'Error among accepted predictions', '接受預測中的錯誤率'), value: r.risk === null ? pick(lang, 'Undefined: no accepted mass', '未定義：沒有接受的機率質量') : fmt(r.risk) }, { label: pick(lang, 'Deployment-optimal boundary', '部署母體最佳邊界'), value: fmt(shift) }] };
    },
  },
];
