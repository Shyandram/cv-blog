import { L, pick, num, fmt, svg, text, path, type Experiment, type Control } from './types.ts';
import { polygon, analyticCenter, centralPath, barrierState, optimum, type Vec } from './barrier.ts';
const POLY = polygon([[-1.5, -0.9], [0.6, -1.2], [1.6, 0.0], [0.5, 1.3], [-1.2, 0.9]]);
const range = (key: string, en: string, zh: string, min: number, max: number, step: number, value: number): Control => ({ key, label: L(en, zh), kind: 'range', min, max, step, value });
const curve = (f: (x: number) => number, lo: number, hi: number, ymin: number, ymax: number, color = 'var(--plot-a)') => path(Array.from({ length: 121 }, (_, i) => { const x = lo + (hi - lo) * i / 120; return [50 + 540 * i / 120, 260 - 220 * (f(x) - ymin) / (ymax - ymin)] as [
    number,
    number
]; }), `stroke="${color}" stroke-width="2.5"`);
const axes = (left: string, right: string) => path([[50, 35], [50, 260], [590, 260]], 'stroke="var(--plot-grid)"') + text(50, 290, left) + text(590, 290, right, 'text-anchor="end"');
export function projectQuadratic(a: number, b: number, rho: number) {
    const f = (x: number, y: number) => 0.5 * ((x - a) ** 2 + 2 * rho * (x - a) * (y - b) + (y - b) ** 2);
    const candidates: [
        [
            number,
            number
        ],
        ...[
            number,
            number
        ][]
    ] = [[0, 0], [0, Math.max(0, b + rho * a)], [Math.max(0, a + rho * b), 0]];
    if (a >= 0 && b >= 0)
        candidates.push([a, b]);
    const best = candidates.reduce((p, q) => f(...p) < f(...q) ? p : q);
    return { best, naive: [Math.max(0, a), Math.max(0, b)] as [
            number,
            number
        ], f };
}
export function scalarDual(b: number, x: number, lambda: number) {
    const primal = .5 * x * x, dual = lambda * b - .5 * lambda * lambda;
    return { primal, dual, gap: primal - dual, feasible: x >= b, stationarity: x - lambda, complementarity: lambda * (x - b), optimum: .5 * Math.max(0, b) ** 2 };
}
export const huber = (r: number, delta: number) => Math.abs(r) <= delta ? .5 * r * r : delta * (Math.abs(r) - .5 * delta);
export function locationFit(outlier: number, kind: string) {
    const data = [-1, -.5, 0, .5, 1, outlier];
    const loss = (m: number) => data.reduce((s, x) => s + (kind === 'l2' ? .5 * (x - m) ** 2 : kind === 'l1' ? Math.abs(x - m) : huber(x - m, 1)), 0);
    let lo = -2, hi = 12;
    if (kind === 'l1')
        return { data, m: .25, loss }; // Median interval [0,.5]; choose its midpoint.
    for (let i = 0; i < 80; i++) {
        const m = (lo + hi) / 2;
        const derivative = data.reduce((s, x) => s + (kind === 'l2' ? m - x : Math.max(-1, Math.min(1, m - x))), 0);
        if (derivative > 0)
            hi = m;
        else
            lo = m;
    }
    return { data, m: (lo + hi) / 2, loss };
}
export const softThreshold = (v: number, lambda: number) => Math.sign(v) * Math.max(Math.abs(v) - lambda, 0);
export const chapterOptimizationExperiments: Experiment[] = [
    {
        id: 'convex-combinations', track: 'optimization', title: L('Does the whole segment stay feasible?', '整條線段都可行嗎？'),
        description: L('Move a convex combination between two feasible points and compare a disk with an annulus.', '在兩個可行點之間移動凸組合，比較圓盤與圓環。'),
        related: ['optimization/opt-01-modeling-convex-sets'],
        challenge: L('Choose the annulus. Find a mixing weight that makes the point infeasible even though both endpoints are feasible.', '選圓環，找出一個混合權重，使中間點不可行，但兩個端點仍可行。'),
        note: L('Endpoints are (−1,0) and (1,0). The disk has radius 1; the annulus also requires radius ≥ 0.5. One counterexample disproves convexity; sampled examples alone do not prove it.', '端點為 (−1,0) 與 (1,0)。圓盤半徑為 1；圓環另要求半徑至少為 0.5。一個反例就能否定凸性，有限取樣本身不能證明凸性。'),
        controls: [range('mix', 'Mixing weight λ', '混合權重 λ', 0, 1, .01, .5), { key: 'set', label: L('Feasible set', '可行集合'), kind: 'select', value: 'annulus', options: [{ value: 'disk', label: L('Disk', '圓盤') }, { value: 'annulus', label: L('Annulus', '圓環') }] }],
        compute(s, lang) {
            const x = 2 * num(s, 'mix') - 1, hole = s.set === 'annulus', feasible = !hole || Math.abs(x) >= .5;
            return { svg: svg(`<circle cx="320" cy="150" r="112" fill="var(--plot-a)" fill-opacity=".15" stroke="var(--plot-a)"/>${hole ? '<circle cx="320" cy="150" r="56" fill="var(--plot-bg)" stroke="var(--plot-b)"/>' : ''}` + path([[208, 150], [432, 150]], 'stroke="var(--plot-ink)" stroke-dasharray="4 3"') + [-1, 1].map(v => `<circle cx="${320 + 112 * v}" cy="150" r="5" fill="var(--plot-a)"/>`).join('') + `<circle cx="${320 + 112 * x}" cy="150" r="8" fill="var(--plot-${feasible ? 'c' : 'b'})"/>` + text(208, 285, '(−1, 0)') + text(432, 285, '(1, 0)', 'text-anchor="end"')),
                summary: pick(lang, feasible ? 'The selected combination is feasible. Check the rest of the segment too.' : 'This point lies in the hole: the annulus is not convex.', feasible ? '目前的組合可行，也請檢查線段其他位置。' : '此點落在中間的孔洞內，因此圓環不是凸集合。'), metrics: [{ label: pick(lang, 'Point x', '點的 x 座標'), value: fmt(x) }, { label: pick(lang, 'Feasible', '可行'), value: pick(lang, feasible ? 'Yes' : 'No', feasible ? '是' : '否') }] };
        },
    },
    {
        id: 'convex-chords', track: 'optimization', title: L('Compare a function with its chord', '比較函數與弦線'),
        description: L('Measure the Jensen gap while moving between two inputs. A negative gap supplies a concrete counterexample.', '在兩個輸入之間移動並計算 Jensen 差距，負值就是具體反例。'), related: ['optimization/opt-02-convex-functions'],
        challenge: L('Compare x² with −x² at weight 0.5. Why do both gaps become zero at the endpoints?', '在權重 0.5 時比較 x² 與 −x²。為什麼端點的差距都變成零？'),
        note: L('For endpoints a=−1 and b=2, gap = (1−λ)f(a)+λf(b)−f((1−λ)a+λb). Nonnegative gap on this single segment is not a proof on every domain.', '端點 a=−1、b=2，差距為 (1−λ)f(a)+λf(b)−f((1−λ)a+λb)。只檢查這一段不能證明整個定義域的凸性。'),
        controls: [range('mix', 'Mixing weight λ', '混合權重 λ', 0, 1, .01, .5), { key: 'kind', label: L('Function', '函數'), kind: 'select', value: 'convex', options: [{ value: 'convex', label: L('x²', 'x²') }, { value: 'concave', label: L('−x²', '−x²') }] }],
        compute(s, lang) {
            const sign = s.kind === 'convex' ? 1 : -1, f = (x: number) => sign * x * x, l = num(s, 'mix'), x = -1 + 3 * l, chord = sign * (1 + 3 * l), gap = chord - f(x), px = 50 + 540 * l, py = (v: number) => 260 - 220 * (v + 4) / 8;
            return { svg: svg(axes('x = −1', 'x = 2') + curve(f, -1, 2, -4, 4) + path([[50, py(f(-1))], [590, py(f(2))]], 'stroke="var(--plot-b)" stroke-width="2"') + path([[px, py(chord)], [px, py(f(x))]], 'stroke="var(--plot-c)" stroke-width="4"') + text(320, 25, pick(lang, 'Blue: function · Orange: chord', '藍：函數 · 橙：弦線'), 'text-anchor="middle"')),
                summary: pick(lang, gap < -.000001 ? 'The function lies above its chord here, violating convexity.' : 'The function is no higher than this chord at the selected point.', gap < -.000001 ? '此處函數高於弦線，違反凸性。' : '此處函數不高於弦線。'), metrics: [{ label: pick(lang, 'Input', '輸入'), value: fmt(x) }, { label: pick(lang, 'Chord − function', '弦線 − 函數'), value: fmt(gap) }] };
        },
    },
    {
        id: 'constrained-quadratic', track: 'optimization', title: L('Why clipping a solution can be wrong', '為什麼直接截斷解可能出錯'),
        description: L('Compare coordinate clipping with the exact solution of a coupled quadratic under nonnegative constraints.', '在非負限制下，比較逐座標截斷與耦合二次問題的精確解。'), related: ['optimization/opt-03-convex-programs'],
        challenge: L('Set coupling to zero, then increase it. When does clipping cease to minimize the objective?', '先把耦合設為零，再增加它。逐座標截斷何時不再使目標最小？'),
        note: L('Minimize ½[(x−a)²+2ρ(x−a)(y−1)+(y−1)²], x,y≥0. |ρ|<1 ensures a positive definite Hessian. The solver checks the interior and both boundary minima; it does not use a grid approximation.', '最小化 ½[(x−a)²+2ρ(x−a)(y−1)+(y−1)²]，且 x,y≥0。|ρ|<1 保證 Hessian 正定。求解器比較內部與兩條邊界的極小值，不是網格近似。'),
        controls: [range('a', 'Unconstrained center a', '無限制中心 a', -2, 2, .1, -1), range('rho', 'Coupling ρ', '耦合 ρ', -.9, .9, .05, .7)],
        compute(s, lang) {
            const a = num(s, 'a'), rho = num(s, 'rho'), r = projectQuadratic(a, 1, rho), X = (x: number) => 260 + 80 * x, Y = (y: number) => 265 - 70 * y;
            let b = `<rect x="260" y="35" width="330" height="230" fill="var(--plot-c)" fill-opacity=".08"/>` + path([[50, Y(0)], [590, Y(0)]], 'stroke="var(--plot-grid)"') + path([[X(0), 35], [X(0), 280]], 'stroke="var(--plot-grid)"');
            for (const [i, [pt, label, color]] of ([[[a, 1], pick(lang, 'Unconstrained', '無限制'), 'a'], [r.naive, pick(lang, 'Clipped', '截斷'), 'b'], [r.best, pick(lang, 'Optimal', '最佳'), 'c']] as const).entries())
                b += `<circle cx="${X(pt[0])}" cy="${Y(pt[1])}" r="6" fill="var(--plot-${color})"/>` + text(50 + 180 * i, 290, label, `fill="var(--plot-${color})"`);
            return { svg: svg(b + text(320, 25, pick(lang, 'Shaded: x ≥ 0, y ≥ 0', '底色：x ≥ 0、y ≥ 0'), 'text-anchor="middle"')), summary: pick(lang, `Optimal (x,y)=(${fmt(r.best[0])}, ${fmt(r.best[1])}). Coupling changes the best boundary position.`, `最佳 (x,y)=(${fmt(r.best[0])}, ${fmt(r.best[1])})。耦合會改變邊界上的最佳位置。`), metrics: [{ label: pick(lang, 'Optimal objective', '最佳目標值'), value: fmt(r.f(...r.best)) }, { label: pick(lang, 'Clipped objective', '截斷後目標值'), value: fmt(r.f(...r.naive)) }, { label: pick(lang, 'Excess objective', '多出的目標值'), value: fmt(r.f(...r.naive) - r.f(...r.best)) }] };
        },
    },
    {
        id: 'duality-certificate', track: 'optimization', title: L('Build a primal–dual certificate', '建立原始與對偶證書'), description: L('Change a candidate point and multiplier. A small gap is meaningful only with feasibility.', '調整候選點與乘數。只有滿足可行性，小差距才有意義。'), related: ['optimization/opt-04-duality-kkt'],
        challenge: L('Use b=1, x=0, λ=0. The displayed difference is zero: why is this not an optimality certificate?', '設 b=1、x=0、λ=0。顯示的差為零，為何這不是最佳性證書？'),
        note: L('Primal: minimize x²/2 subject to x≥b. L=x²/2+λ(b−x), λ≥0; dual g(λ)=λb−λ²/2. Stationarity is x−λ=0 and complementarity is λ(x−b)=0.', '原始問題：在 x≥b 下最小化 x²/2。L=x²/2+λ(b−x)，λ≥0；對偶 g(λ)=λb−λ²/2。定態條件為 x−λ=0，互補條件為 λ(x−b)=0。'),
        controls: [range('b', 'Lower bound b', '下界 b', -1, 2, .1, 1), range('x', 'Candidate x', '候選 x', -1, 3, .1, 1.5), range('lambda', 'Multiplier λ', '乘數 λ', 0, 3, .1, 1)],
        compute(s, lang) {
            const b = num(s, 'b'), x = num(s, 'x'), l = num(s, 'lambda'), r = scalarDual(b, x, l), lo = Math.min(r.primal, r.dual, r.optimum) - .5, hi = Math.max(r.primal, r.dual, r.optimum) + .5, py = (v: number) => 235 - 180 * (v - lo) / (hi - lo);
            const lines = [[r.primal, pick(lang, 'Candidate objective', '候選目標'), 'a'], [r.dual, pick(lang, 'Dual lower bound', '對偶下界'), 'b'], [r.optimum, pick(lang, 'True optimum', '真正最佳值'), 'c']] as const;
            return { svg: svg(lines.map(([v, label, col], i) => path([[60 + 170 * i, py(v)], [190 + 170 * i, py(v)]], `stroke="var(--plot-${col})" stroke-width="4"`) + text(125 + 170 * i, 280, label, 'text-anchor="middle"') + text(125 + 170 * i, py(v) - 12, fmt(v), 'text-anchor="middle"')).join('')), summary: pick(lang, r.feasible ? 'The candidate is feasible. Its objective minus the dual bound is an upper bound on suboptimality.' : 'The candidate violates x ≥ b. Its objective is not a valid upper bound; the difference is not a certificate.', r.feasible ? '候選點可行，目標值減去對偶下界可界定次佳程度。' : '候選點違反 x ≥ b，其目標值不是有效上界，這個差不是證書。'), metrics: [{ label: pick(lang, 'Objective − dual', '目標 − 對偶'), value: fmt(r.gap) }, { label: pick(lang, 'Constraint violation', '限制違反量'), value: fmt(Math.max(0, b - x)) }, { label: pick(lang, 'Stationarity residual', '定態殘差'), value: fmt(r.stationarity) }, { label: pick(lang, 'Complementarity', '互補殘差'), value: fmt(r.complementarity) }] };
        },
    },
    {
        id: 'robust-location', track: 'optimization', title: L('One outlier, three meanings of best fit', '一個離群值，三種最佳擬合'), description: L('Move one observation and compare squared, absolute and Huber losses for fitting a constant.', '移動一筆觀測，比較平方、絕對值與 Huber 損失擬合常數的結果。'), related: ['optimization/opt-05-fitting-estimation'],
        challenge: L('Increase the outlier from 2 to 12. Compare the movement of the squared-loss and Huber estimates.', '把離群值從 2 增加到 12，比較平方損失與 Huber 估計移動多少。'),
        note: L('Five fixed observations are −1, −0.5, 0, 0.5, 1. The sixth is adjustable. Squared loss includes ½; Huber uses δ=1. For absolute loss, any point in [0,0.5] minimizes the loss; the plot chooses 0.25.', '固定五筆觀測為 −1、−0.5、0、0.5、1，第六筆可調。平方損失含 ½，Huber 使用 δ=1。絕對損失在 [0,0.5] 內皆為極小值，圖中取 0.25。'),
        predict: { question: L("The sixth observation moves from 2 up to 12 while the other five stay put. Which fit is dragged furthest?", "第六筆觀測從 2 移到 12，其餘五筆不動。哪一種擬合被拉得最遠？"), options: [{ value: "l2", label: L("Squared loss (the mean)", "平方損失（平均數）") }, { value: "l1", label: L("Absolute loss (the median)", "絕對值損失（中位數）") }, { value: "huber", label: L("Huber loss", "Huber 損失") }], answer: "l2", explain: L("The squared loss pulls with a force that grows with distance, so its fit moves from 0.33 to 2. The absolute and Huber losses cap each point's pull, so their fits stay at 0.25.", "平方損失的拉力隨距離增加，擬合值從 0.33 移到 2。絕對值與 Huber 損失限制每個點的拉力，擬合值維持在 0.25。") },
        controls: [range('outlier', 'Sixth observation', '第六筆觀測', 2, 12, .1, 8), { key: 'loss', label: L('Loss', '損失'), kind: 'select', value: 'l2', options: [{ value: 'l2', label: L('Squared', '平方') }, { value: 'l1', label: L('Absolute', '絕對值') }, { value: 'huber', label: L('Huber (δ=1)', 'Huber（δ=1）') }] }],
        compute(s, lang) {
            const r = locationFit(num(s, 'outlier'), String(s.loss)), ymax = Math.max(r.loss(-2), r.loss(12)), px = (x: number) => 50 + 540 * (x + 2) / 14;
            return { svg: svg(axes('m = −2', 'm = 12') + curve(r.loss, -2, 12, 0, ymax) + path([[px(r.m), 40], [px(r.m), 260]], 'stroke="var(--plot-b)" stroke-dasharray="4 3"') + r.data.map(x => `<circle cx="${px(x)}" cy="257" r="4" fill="var(--plot-c)"/>`).join('') + text(320, 25, pick(lang, 'Curve: total loss · Dots: observations', '曲線：總損失 · 點：觀測值'), 'text-anchor="middle"')), summary: pick(lang, 'The loss defines what an error costs. Robust residual loss and parameter regularization solve different problems.', '損失定義每種誤差的代價。穩健殘差損失與參數正則化處理不同問題。'), metrics: [{ label: pick(lang, 'Fitted constant', '擬合常數'), value: fmt(r.m) }, { label: pick(lang, 'Total loss', '總損失'), value: fmt(r.loss(r.m)) }] };
        },
    },
    {
        id: 'barrier-central-path', track: 'optimization', title: L('Follow the central path to the optimal vertex', '沿著中心路徑走向最佳頂點'),
        description: L('Raise the barrier weight t on a two-variable linear program and watch the central point, its dual bound and the gap m/t.', '在兩變數線性規劃上提高障礙權重 t，觀察中心點、對偶界與差距 m/t。'),
        related: ['optimization/opt-07-constrained-interior-point'],
        challenge: L('Multiply t by ten. How do the distance to the boundary, the true suboptimality and the bound m/t change? Then rotate the objective: does the path still end at a vertex?', '把 t 乘以十。與邊界的距離、真實的次佳程度與界限 m/t 如何改變？再旋轉目標方向：路徑仍然會停在頂點嗎？'),
        note: L('Minimize c·x over a convex pentagon with m = 5 edge constraints a_i·x ≤ b_i. The central point minimizes t·c·x − Σ log(b_i − a_i·x) (damped Newton). With λ_i = 1/(t s_i), where s_i is the slack, the dual objective is −b·λ and the duality gap equals Σ λ_i s_i = m/t exactly. The true suboptimality c·x − p* is at most m/t.', '在有 m = 5 條邊限制 a_i·x ≤ b_i 的凸五邊形上最小化 c·x。中心點最小化 t·c·x − Σ log(b_i − a_i·x)（阻尼牛頓法）。令 λ_i = 1/(t s_i)（s_i 為鬆弛量），對偶目標為 −b·λ，對偶差距恰為 Σ λ_i s_i = m/t。真正的次佳程度 c·x − p* 不會超過 m/t。'),
        predict: { question: L('You multiply the barrier weight t by 10. What happens to the duality gap m/t?', '你把障礙權重 t 乘以 10。對偶差距 m/t 會怎麼變？'),
            options: [{ value: 'tenth', label: L('It shrinks by a factor of 10', '縮小為十分之一') }, { value: 'sqrt', label: L('It shrinks by a factor of about 3', '縮小為約三分之一') }, { value: 'same', label: L('It stays the same', '不變') }],
            answer: 'tenth',
            explain: L('The gap is exactly m/t, so it is inversely proportional to t. With m = 5 constraints it is 5/t: 0.5 at t = 10 and 0.05 at t = 100. The central point moves toward the optimal vertex, and its true suboptimality never exceeds that bound.', '差距恰為 m/t，所以與 t 成反比。有 m = 5 條限制時為 5/t：t = 10 時是 0.5，t = 100 時是 0.05。中心點朝最佳頂點移動，其真正的次佳程度絕不超過這個界限。') },
        controls: [range('logt', 'log₁₀ t', 'log₁₀ t', -1, 3.5, .1, 1), range('angle', 'Objective direction (degrees)', '目標方向（度）', 0, 360, 5, 200)],
        compute(s, lang) {
            const c: Vec = [Math.cos(num(s, 'angle') * Math.PI / 180), Math.sin(num(s, 'angle') * Math.PI / 180)], t = 10 ** num(s, 'logt');
            const center = analyticCenter(POLY, [0, 0]), path0 = centralPath(POLY, c, -1, 3.5, 70, center), at = barrierState(POLY, c, t, center), opt = optimum(POLY, c);
            const px = (v: Vec) => [60 + (v[0] + 1.9) / 3.8 * 520, 160 - v[1] / 1.7 * 120] as [number, number];
            const poly = POLY.vertices.map(v => px(v).join(',')).join(' ');
            const lineAt = (k: number): [number, number][] => { const o = px(at.x), dir: Vec = [-c[1], c[0]]; return [[o[0] - 400 * dir[0] * k, o[1] + 400 * dir[1] * k], [o[0] + 400 * dir[0] * k, o[1] - 400 * dir[1] * k]]; };
            let body = `<polygon points="${poly}" fill="var(--plot-a)" fill-opacity="0.08" stroke="var(--plot-a)" stroke-width="2"/>` + path(lineAt(1), 'stroke="var(--plot-muted)" stroke-dasharray="5 4"')
                + path(path0.map(px), 'stroke="var(--plot-c)" stroke-width="3"');
            const o = px(opt.vertex), a = px(center), q = px(at.x);
            body += `<circle cx="${o[0]}" cy="${o[1]}" r="7" fill="var(--plot-d)"/>` + `<circle cx="${a[0]}" cy="${a[1]}" r="5" fill="var(--plot-muted)"/>` + `<circle cx="${q[0]}" cy="${q[1]}" r="7" fill="var(--plot-b)"/>`
                + text(a[0] + 8, a[1] + 16, pick(lang, 'analytic center (t → 0)', '解析中心（t → 0）'), 'fill="var(--plot-muted)"') + text(Math.min(o[0] + 10, 470), o[1] - 10, pick(lang, 'optimal vertex', '最佳頂點'), 'fill="var(--plot-d)"')
                + text(30, 24, pick(lang, 'Green: central path · orange: x*(t) · dashed: level line of c·x through x*(t)', '綠線：中心路徑 · 橘點：x*(t) · 虛線：通過 x*(t) 的 c·x 等高線'));
            return {
                svg: svg(body, 330),
                summary: pick(lang, `At t = ${fmt(t, 3)} the central point is ${fmt(at.minSlack, 3)} from the nearest edge. Its suboptimality ${fmt(at.suboptimality, 4)} is at most the duality gap m/t = ${fmt(at.bound, 4)}, and the gap computed from the dual point λ = 1/(t·slack) is ${fmt(at.gap, 4)}.`, `t = ${fmt(t, 3)} 時，中心點離最近的邊 ${fmt(at.minSlack, 3)}。它的次佳程度 ${fmt(at.suboptimality, 4)} 不超過對偶差距 m/t = ${fmt(at.bound, 4)}；由對偶點 λ = 1/(t·鬆弛量) 算出的差距為 ${fmt(at.gap, 4)}。`),
                metrics: [{ label: pick(lang, 'Gap m/t', '差距 m/t'), value: fmt(at.bound, 4) }, { label: pick(lang, 'Suboptimality c·x − p*', '次佳程度 c·x − p*'), value: fmt(at.suboptimality, 4) }, { label: pick(lang, 'Distance to nearest edge', '到最近邊的距離'), value: fmt(at.minSlack, 3) }],
            };
        },
    },
    {
        id: 'proximal-threshold', track: 'optimization', title: L('Why a proximal step creates exact zeros', '為什麼近端步驟會產生精確零值'), description: L('Compare soft thresholding with ordinary shrinkage and inspect its nonsmooth objective.', '比較軟閾值與一般縮小，並觀察不可微的目標函數。'), related: ['optimization/opt-08-applications-modern-methods'],
        challenge: L('Keep v=1 and increase λ through 1. Why does the optimum stay at exactly zero afterward?', '固定 v=1，讓 λ 增加並超過 1。為何最佳解之後會保持精確的零？'),
        note: L('Solve minₓ ½(x−v)²+λ|x|. The exact proximal solution is sign(v) max(|v|−λ,0). This scalar identity extends coordinatewise to an L1 penalty; it is not a complete total-variation denoiser.', '求解 minₓ ½(x−v)²+λ|x|。精確近端解為 sign(v) max(|v|−λ,0)。此純量公式可逐座標用於 L1 懲罰，不代表完整的全變異去雜訊器。'),
        controls: [range('v', 'Input v', '輸入 v', -3, 3, .1, 1.5), range('lambda', 'Penalty λ', '懲罰 λ', 0, 3, .1, 1)],
        compute(s, lang) {
            const v = num(s, 'v'), l = num(s, 'lambda'), x = softThreshold(v, l), f = (z: number) => .5 * (z - v) ** 2 + l * Math.abs(z), max = Math.max(f(-4), f(4));
            return { svg: svg(axes('x = −4', 'x = 4') + curve(f, -4, 4, 0, max) + `<circle cx="${50 + 540 * (x + 4) / 8}" cy="${260 - 220 * f(x) / max}" r="6" fill="var(--plot-b)"/>`), summary: pick(lang, Math.abs(v) <= l ? 'The kink contains a zero subgradient, so the optimum is exactly zero.' : 'Outside the threshold, the magnitude is reduced by λ.', Math.abs(v) <= l ? '轉折點的次梯度包含零，因此最佳解恰好為零。' : '超過閾值時，幅度減少 λ。'), metrics: [{ label: pick(lang, 'Proximal solution', '近端解'), value: fmt(x) }, { label: pick(lang, 'Objective', '目標值'), value: fmt(f(x)) }, { label: pick(lang, 'Residual v − x', '殘差 v − x'), value: fmt(v - x) }] };
        },
    }
];
