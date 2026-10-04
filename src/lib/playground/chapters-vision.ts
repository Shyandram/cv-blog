import type { Experiment, Control, State } from "./types.ts";
import { L, pick, fmt, svg, text, path } from "./types.ts";
const clamp = (v: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));
const n = (s: State, k: string) => Number(s[k]);
const range = (
  key: string,
  en: string,
  zh: string,
  value: number,
  min: number,
  max: number,
  step: number,
): Control => ({ key, label: L(en, zh), kind: "range", value, min, max, step });
const select = (
  key: string,
  en: string,
  zh: string,
  value: string,
  choices: [string, string, string][],
): Control => ({
  key,
  label: L(en, zh),
  kind: "select",
  value,
  options: choices.map(([value, a, b]) => ({ value, label: L(a, b) })),
});
const metric = (
  lang: "en" | "zh-tw",
  en: string,
  zh: string,
  value: number | string,
) => ({
  label: pick(lang, en, zh),
  value: typeof value === "number" ? fmt(value, 5) : value,
});
const mse = (a: number[], b: number[]) =>
  a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0) / a.length;
const grid = (a: number[], side: number, x: number, y: number, size = 216) =>
  a
    .map((v, i) => {
      const c = Math.round(255 * clamp(v));
      return `<rect x="${x + ((i % side) * size) / side}" y="${y + (Math.floor(i / side) * size) / side}" width="${size / side + 0.1}" height="${size / side + 0.1}" fill="rgb(${c},${c},${c})"/>`;
    })
    .join("");
const lines = (
  sets: number[][],
  labels: string[],
  lo?: number,
  hi?: number,
) => {
  const low = lo ?? Math.min(...sets.flat()),
    high = hi ?? Math.max(...sets.flat()),
    span = Math.max(high - low, 1e-9);
  return (
    text(34, 42, fmt(high)) +
    text(34, 244, fmt(low)) +
    sets
      .map(
        (a, k) =>
          path(
            a.map((v, i) => [
              70 + (530 * i) / Math.max(a.length - 1, 1),
              240 - (190 * (v - low)) / span,
            ]),
            `stroke="var(--plot-${"abcd"[k]})" stroke-width="2.5"${k === 0 ? ' stroke-dasharray="5 3"' : ""}`,
          ) +
          text(70 + k * 180, 280, labels[k], `fill="var(--plot-${"abcd"[k]})"`),
      )
      .join("")
  );
};
const IP = (slug: string) => `image-processing/${slug}`;
const DL = (slug: string) => `deep-learning/${slug}`;

/** A real diagonal inverse problem in a cosine basis. DC is left unchanged. */
export function regularizedInverse(
  blur: number,
  noise: number,
  lambda: number,
) {
  const clean = Array.from(
    { length: 32 },
    (_, i) =>
      0.5 +
      0.27 * Math.cos((Math.PI * (i + 0.5) * 2) / 32) +
      0.14 * Math.cos((Math.PI * (i + 0.5) * 11) / 32),
  );
  const basis = (i: number, k: number) =>
    Math.cos((Math.PI * (i + 0.5) * k) / 32) *
    (k ? Math.sqrt(2 / 32) : Math.sqrt(1 / 32));
  const coeff = Array.from({ length: 32 }, (_, k) =>
    clean.reduce((s, x, i) => s + x * basis(i, k), 0),
  );
  const h = coeff.map((_, k) => Math.exp(-blur * (k / 31) ** 2));
  const observed = coeff.map(
    (c, k) => h[k] * c + (k ? noise * Math.sin(k * 7.13) : 0),
  );
  const inverse = observed.map((c, k) =>
    k ? (h[k] * c) / (h[k] ** 2 + lambda) : c,
  );
  const decode = (c: number[]) =>
    clean.map((_, i) => c.reduce((s, v, k) => s + v * basis(i, k), 0));
  return {
    clean,
    observed: decode(observed),
    restored: decode(inverse),
    gain: Math.max(...h.slice(1).map((v) => v / (v * v + lambda))),
  };
}
export function colorGain(rgb: number[], gain: number) {
  const raw = rgb.map((v) => v * gain),
    clipped = raw.map((v) => clamp(v));
  const sum = rgb.reduce((a, b) => a + b, 0),
    out = clipped.reduce((a, b) => a + b, 0);
  return {
    raw,
    clipped,
    count: raw.filter((v) => v > 1).length,
    shift: Math.sqrt(
      rgb.reduce((a, v, i) => a + (v / sum - clipped[i] / out) ** 2, 0),
    ),
  };
}
export function haarForward(a: number[]): number[] {
  if (a.length === 1) return a.slice();
  const low = [],
    high = [];
  for (let i = 0; i < a.length; i += 2) {
    low.push((a[i] + a[i + 1]) / Math.SQRT2);
    high.push((a[i] - a[i + 1]) / Math.SQRT2);
  }
  return [...haarForward(low), ...high];
}
export function haarInverse(c: number[]): number[] {
  if (c.length === 1) return c.slice();
  const low = haarInverse(c.slice(0, c.length / 2)),
    high = c.slice(c.length / 2);
  return low.flatMap((v, i) => [
    (v + high[i]) / Math.SQRT2,
    (v - high[i]) / Math.SQRT2,
  ]);
}
export function dctQuantize(step: number) {
  const side = 8;
  const input = Array.from(
    { length: 64 },
    (_, i) =>
      0.15 +
      (0.5 * (i % 8)) / 7 +
      (i % 8 > 3 && Math.floor(i / 8) > 2 ? 0.22 : 0),
  );
  const b = (x: number, u: number) =>
    (u ? Math.sqrt(2 / 8) : Math.sqrt(1 / 8)) *
    Math.cos(((2 * x + 1) * u * Math.PI) / 16);
  const coeff = input.map((_, k) =>
    input.reduce(
      (s, v, i) =>
        s + v * b(i % 8, k % 8) * b(Math.floor(i / 8), Math.floor(k / 8)),
      0,
    ),
  );
  const quant = coeff.map((c) =>
    step === 0 ? c : Math.round(c / step) * step,
  );
  const output = input.map((_, i) =>
    quant.reduce(
      (s, c, k) =>
        s + c * b(i % 8, k % 8) * b(Math.floor(i / 8), Math.floor(k / 8)),
      0,
    ),
  );
  return { side, input, output, coeff, quant, error: mse(input, output) };
}
export function binaryMorph(
  a: number[],
  side: number,
  radius: number,
  shape: string,
  op: string,
): number[] {
  const offsets: [number, number][] = [];
  for (let y = -radius; y <= radius; y++)
    for (let x = -radius; x <= radius; x++)
      if (shape === "square" || Math.abs(x) + Math.abs(y) <= radius)
        offsets.push([x, y]);
  const primitive = (input: number[], dilate: boolean) =>
    input.map((_, i) => {
      const vals = offsets.map(([dx, dy]) => {
        const x = (i % side) + dx,
          y = Math.floor(i / side) + dy;
        return x >= 0 && y >= 0 && x < side && y < side
          ? input[y * side + x]
          : 0;
      });
      return Number(dilate ? vals.some(Boolean) : vals.every(Boolean));
    });
  return op === "open"
    ? primitive(primitive(a, false), true)
    : op === "close"
      ? primitive(primitive(a, true), false)
      : primitive(a, op === "dilate");
}
export function thresholdScene(
  illumination: number,
  threshold: number,
  automatic: boolean,
) {
  const side = 20,
    truth = Array.from({ length: 400 }, (_, i) =>
      Number(((i % 20) - 10) ** 2 + (Math.floor(i / 20) - 10) ** 2 < 36),
    );
  const image = truth.map((v, i) =>
    clamp(0.2 + 0.42 * v + illumination * ((i % 20) / 19 - 0.5)),
  );
  let best = -1,
    chosen = 0;
  for (let k = 0; k < 256; k++) {
    const t = k / 255,
      low = image.filter((v) => v <= t),
      high = image.filter((v) => v > t);
    if (!low.length || !high.length) continue;
    const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
    const score =
      ((low.length * high.length) / 400 ** 2) * (mean(low) - mean(high)) ** 2;
    if (score > best + 1e-12) {
      best = score;
      chosen = t;
    }
  }
  const t = automatic ? chosen : threshold;
  const mask = image.map((v) => Number(v > t));
  const tp = mask.reduce((s, v, i) => s + v * truth[i], 0),
    union = mask.reduce((s, v, i) => s + Number(v || truth[i]), 0);
  return {
    side,
    truth,
    image,
    mask,
    threshold: t,
    iou: union ? tp / union : 1,
  };
}
/** A deliberately restricted circular snake: radius is the only degree of freedom. */
export function radialSnake(initial: number, tension: number, steps: number) {
  const energy = (r: number) =>
    tension * r * r - Math.exp(-(((r - 1) / 0.16) ** 2));
  const derivative = (r: number) =>
    2 * tension * r +
    ((2 * (r - 1)) / 0.16 ** 2) * Math.exp(-(((r - 1) / 0.16) ** 2));
  let r = initial;
  const history = [r];
  for (let i = 0; i < steps; i++) {
    const g = derivative(r);
    let rate = 0.025;
    for (
      let j = 0;
      j < 20 && energy(clamp(r - rate * g, 0, 1.8)) > energy(r);
      j++
    )
      rate *= 0.5;
    r = clamp(r - rate * g, 0, 1.8);
    history.push(r);
  }
  return {
    radius: r,
    history,
    energy,
    initialEnergy: energy(initial),
    finalEnergy: energy(r),
  };
}
export function structureTensor(kind: string, angle: number, contrast: number) {
  const theta = (angle * Math.PI) / 180,
    c = Math.cos(theta),
    s = Math.sin(theta);
  const gradients =
    kind === "flat"
      ? [
          [0, 0],
          [0, 0],
        ]
      : kind === "edge"
        ? [
            [1, 0],
            [1, 0],
          ]
        : [
            [1, 0],
            [0, 1],
          ];
  const g = gradients.map(([x, y]) => [
    contrast * (c * x - s * y),
    contrast * (s * x + c * y),
  ]);
  const xx = g.reduce((a, v) => a + v[0] ** 2, 0),
    yy = g.reduce((a, v) => a + v[1] ** 2, 0),
    xy = g.reduce((a, v) => a + v[0] * v[1], 0);
  const trace = xx + yy,
    delta = Math.hypot(xx - yy, 2 * xy);
  return {
    g,
    xx,
    yy,
    xy,
    eigen: [(trace + delta) / 2, (trace - delta) / 2],
    response: xx * yy - xy ** 2 - 0.05 * trace ** 2,
  };
}
export function prototypeDistances(x: number, y: number, scale: number) {
  return [
    [0.25, 0.7],
    [0.75, 0.3],
  ].map(([a, b]) => (x - a) ** 2 + scale ** 2 * (y - b) ** 2);
}
export function receptiveField(
  depth: number,
  stride: number,
  dilation: number,
) {
  let field = 1,
    jump = 1;
  const layers = [{ field, jump }];
  for (let i = 0; i < depth; i++) {
    field += 2 * dilation * jump;
    jump *= stride;
    layers.push({ field, jump });
  }
  return { field, jump, layers };
}
export function classificationThreshold(
  threshold: number,
  falsePositiveCost: number,
) {
  const scores = [0.95, 0.9, 0.82, 0.73, 0.66, 0.58, 0.47, 0.35, 0.21, 0.08];
  const truth = [1, 1, 0, 1, 1, 0, 1, 0, 0, 0];
  const prediction = scores.map((score) => Number(score >= threshold));
  let tp = 0,
    fp = 0,
    fn = 0,
    tn = 0;
  prediction.forEach((p, i) => {
    if (p) {
      if (truth[i]) tp++;
      else fp++;
    } else {
      if (truth[i]) fn++;
      else tn++;
    }
  });
  return {
    scores,
    truth,
    prediction,
    tp,
    fp,
    fn,
    tn,
    precision: tp + fp ? tp / (tp + fp) : null,
    recall: tp / (tp + fn),
    f1: (2 * tp) / (2 * tp + fp + fn),
    accuracy: (tp + tn) / 10,
    cost: (falsePositiveCost * fp + fn) / 10,
  };
}
export function ambiguousRestoration(probability: number, mix: number) {
  const a = [0, 0, 1, 1, 0, 0, 1, 1],
    b = [1, 1, 0, 0, 1, 1, 0, 0];
  const mean = a.map((v, i) => probability * v + (1 - probability) * b[i]);
  const output = mean.map((v, i) => (1 - mix) * v + mix * a[i]);
  return {
    a,
    b,
    mean,
    output,
    risk: probability * mse(a, output) + (1 - probability) * mse(b, output),
    riskMean: probability * mse(a, mean) + (1 - probability) * mse(b, mean),
  };
}
export function photonNoise(exposure: number, gain: number, read: number) {
  const signal = exposure * 0.25,
    variance = signal + read ** 2;
  return {
    mean: gain * signal,
    std: gain * Math.sqrt(variance),
    snr: signal / Math.sqrt(variance),
    relative: Math.sqrt(variance) / signal,
  };
}
export function trackingAssignment(
  motionWeight: number,
  appearanceError: number,
  crossing: number,
) {
  const predictions = [0.3, 0.7],
    detections = [0.3 + 0.4 * crossing, 0.7 - 0.4 * crossing];
  const cost = predictions.map((x, i) =>
    detections.map(
      (y, j) =>
        motionWeight * Math.abs(x - y) +
        (1 - motionWeight) * (i === j ? appearanceError : 1 - appearanceError),
    ),
  );
  const direct = cost[0][0] + cost[1][1],
    swapped = cost[0][1] + cost[1][0];
  return {
    predictions,
    detections,
    cost,
    direct,
    swapped,
    assignment: direct <= swapped ? [0, 1] : [1, 0],
    ties: Math.abs(direct - swapped) < 1e-10,
  };
}
export function worldRollout(
  coefficient: number,
  steps: number,
  action: number,
  refresh: number,
) {
  const truth = [1],
    open = [1],
    corrected = [1],
    one = [];
  for (let t = 0; t < steps; t++) {
    const u = action * Math.sin(t * 0.45);
    truth.push(0.96 * truth[t] + u);
    open.push(coefficient * open[t] + u);
    one.push(coefficient * truth[t] + u);
    corrected.push(
      coefficient *
        (refresh > 0 && t % refresh === 0 ? truth[t] : corrected[t]) +
        u,
    );
  }
  return {
    truth,
    open,
    corrected,
    oneError: mse(one, truth.slice(1)),
    openError: mse(open.slice(1), truth.slice(1)),
    correctedError: mse(corrected.slice(1), truth.slice(1)),
  };
}

export const chapterVisionExperiments: Experiment[] = [
  {
    id: "inverse-regularization",
    track: "image-processing",
    title: L(
      "IP 05 · Undo blur without amplifying every error",
      "IP 05・反模糊與雜訊放大",
    ),
    description: L(
      "Trade inverse-filter amplification against regularization bias in a real cosine-basis reconstruction.",
      "在餘弦基底重建中，權衡反濾波的雜訊放大與正則化偏差。",
    ),
    challenge: L(
      "With noise set to zero, try λ = 0. Then add noise without changing the blur. Find a λ that reduces reconstruction error.",
      "先令雜訊與 λ 都為 0，再固定模糊程度並加入雜訊。找出能降低重建誤差的 λ。",
    ),
    note: L(
      "A 32-sample synthetic signal. H(k) = exp(−b(k/31)²); reconstruction gain is H/(H²+λ), with DC preserved. Fixed deterministic perturbations isolate sensitivity; λ is not an estimated noise spectrum. The vertical axis rescales to include all values.",
      "使用 32 點合成訊號，H(k) = exp(−b(k/31)²)，重建增益為 H/(H²+λ)，直流項保持不變。固定擾動用於觀察敏感度；λ 並非估計出的雜訊頻譜。縱軸會調整範圍以包含所有數值。",
    ),
    related: [IP("dip-05-restoration-reconstruction")],
    controls: [
      range("blur", "Blur severity b", "模糊程度 b", 3, 0, 6, 0.2),
      range(
        "noise",
        "Coefficient perturbation",
        "係數擾動幅度",
        0.03,
        0,
        0.1,
        0.005,
      ),
      range("lambda", "Regularization λ", "正則化 λ", 0.02, 0, 0.2, 0.005),
    ],
    compute(s, l) {
      const r = regularizedInverse(n(s, "blur"), n(s, "noise"), n(s, "lambda"));
      return {
        svg: svg(
          lines(
            [r.clean, r.observed, r.restored],
            [
              pick(l, "Clean", "乾淨訊號"),
              pick(l, "Observed", "觀測訊號"),
              pick(l, "Restored", "重建訊號"),
            ],
          ),
        ),
        summary: pick(
          l,
          "Regularization caps unreliable frequency gains, but also suppresses true detail. Compare error against the known synthetic clean signal.",
          "正則化限制不可靠頻率的增益，也會抑制真實細節。此處以已知的合成乾淨訊號計算誤差。",
        ),
        metrics: [
          metric(
            l,
            "Reconstruction MSE",
            "重建均方誤差",
            mse(r.clean, r.restored),
          ),
          metric(
            l,
            "Observation MSE",
            "觀測均方誤差",
            mse(r.clean, r.observed),
          ),
          metric(l, "Largest AC gain", "最大交流增益", r.gain),
        ],
      };
    },
  },
  {
    id: "color-clipping",
    track: "image-processing",
    title: L(
      "IP 06 · Brightening can change color",
      "IP 06・提亮也可能改變色彩",
    ),
    description: L(
      "Multiply linear RGB channels, inspect clipping, and track normalized channel proportions.",
      "乘上線性 RGB 增益，觀察截斷與正規化通道比例。",
    ),
    challenge: L(
      "Find a gain that clips just the red channel. Why does a common gain stop preserving the channel ratios?",
      "找出只讓紅色通道截斷的增益。為什麼共同增益不再維持通道比例？",
    ),
    note: L(
      "Synthetic linear RGB, displayed through the sRGB transfer curve. Normalized RGB proportions are a teaching coordinate, not CIE chromaticity or a perceptual color-difference metric. Clipping applies independently to each channel.",
      "使用合成線性 RGB，經 sRGB 轉換曲線顯示。正規化 RGB 比例是教學座標，並非 CIE 色度或感知色差指標。各通道分別截斷。",
    ),
    related: [IP("dip-06-color")],
    controls: [
      range("red", "Linear red", "線性紅色", 0.7, 0.05, 1, 0.05),
      range("green", "Linear green", "線性綠色", 0.3, 0.05, 1, 0.05),
      range("blue", "Linear blue", "線性藍色", 0.1, 0.05, 1, 0.05),
      range("gain", "Common gain", "共同增益", 1.5, 0.2, 4, 0.1),
    ],
    compute(s, l) {
      const input = [n(s, "red"), n(s, "green"), n(s, "blue")],
        r = colorGain(input, n(s, "gain"));
      const color = (a: number[]) =>
        `rgb(${a.map((v) => Math.round(255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055))).join(",")})`;
      let body =
        text(45, 30, pick(l, "Original", "原始色彩")) +
        text(345, 30, pick(l, "After gain + clipping", "增益與截斷後"));
      body += `<rect x="45" y="45" width="250" height="100" fill="${color(input)}"/><rect x="345" y="45" width="250" height="100" fill="${color(r.clipped)}"/>`;
      body += input
        .map((v, i) =>
          text(
            45,
            185 + i * 29,
            `${"RGB"[i]}: ${fmt(v)} → ${fmt(r.raw[i])} → ${fmt(r.clipped[i])}`,
          ),
        )
        .join("");
      return {
        svg: svg(body),
        summary: pick(
          l,
          "A common gain preserves linear RGB ratios only before clipping. The swatches use display encoding; the numbers remain linear-light values.",
          "共同增益只在截斷前保持線性 RGB 比例。色塊使用顯示編碼，數字仍為線性光數值。",
        ),
        metrics: [
          metric(l, "Clipped channels", "截斷通道數", r.count),
          metric(l, "Normalized RGB shift", "正規化 RGB 位移", r.shift),
        ],
      };
    },
  },
  {
    id: "haar-details",
    track: "image-processing",
    title: L(
      "IP 07 · Which wavelet details can disappear?",
      "IP 07・哪些小波細節可以移除？",
    ),
    description: L(
      "Threshold Haar detail coefficients, invert the transform, and measure exactly what was lost.",
      "對 Haar 細節係數設定門檻，再反轉換並量測遺失內容。",
    ),
    challenge: L(
      "Increase the threshold until the fine alternating pattern disappears. Does the average brightness change?",
      "提高門檻，直到細密的交替模式消失。平均亮度有改變嗎？",
    ),
    note: L(
      "Orthonormal 16-point Haar transform with hard thresholding of detail coefficients only. The first coefficient carries the mean and is never removed. Coefficient count is not a compressed file size.",
      "使用正交正規的 16 點 Haar 轉換，只對細節係數進行硬門檻處理。第一個係數承載平均值，永不移除。係數數量不等於壓縮檔案大小。",
    ),
    related: [IP("dip-07-wavelets-transforms")],
    controls: [
      range("threshold", "Detail threshold", "細節門檻", 0.2, 0, 1, 0.05),
      range("texture", "Fine texture amplitude", "細紋幅度", 0.1, 0, 0.2, 0.02),
    ],
    compute(s, l) {
      const input = Array.from(
        { length: 16 },
        (_, i) => (i < 8 ? 0.25 : 0.75) + n(s, "texture") * (i % 2 ? 1 : -1),
      );
      const c = haarForward(input),
        kept = c.map((v, i) => (i && Math.abs(v) < n(s, "threshold") ? 0 : v)),
        output = haarInverse(kept);
      return {
        svg: svg(
          lines(
            [input, output],
            [pick(l, "Input", "輸入"), pick(l, "Reconstruction", "重建")],
            0,
            1,
          ),
        ),
        summary: pick(
          l,
          `Coefficients in coarse-to-fine order: ${kept.map((v) => fmt(v, 2)).join(", ")}. Energy removed in coefficient space equals squared reconstruction error.`,
          `由粗至細的係數：${kept.map((v) => fmt(v, 2)).join("、")}。係數空間移除的能量等於重建平方誤差總和。`,
        ),
        metrics: [
          metric(
            l,
            "Nonzero coefficients",
            "非零係數",
            kept.filter((v) => Math.abs(v) > 1e-10).length,
          ),
          metric(l, "Reconstruction MSE", "重建均方誤差", mse(input, output)),
          metric(
            l,
            "Mean shift",
            "平均值位移",
            output.reduce((a, b) => a + b, 0) / 16 - 0.5,
          ),
        ],
      };
    },
  },
  {
    id: "dct-quantization",
    track: "image-processing",
    title: L(
      "IP 08 · Transform coefficients are not yet compression",
      "IP 08・轉換係數還不等於壓縮",
    ),
    description: L(
      "Quantize a synthetic 8 × 8 block in the DCT domain and inspect reconstruction error.",
      "在 DCT 領域量化合成的 8 × 8 區塊，觀察重建誤差。",
    ),
    challenge: L(
      "Start with zero quantization, then increase the step. Which happens first: many zeros or visible brightness distortion?",
      "先不量化，再增加量化步長。大量零值與可見的亮度失真，哪個先出現？",
    ),
    note: L(
      "An orthonormal 2-D DCT followed by uniform scalar quantization, including DC. Zero step bypasses quantization. This isolates one coding stage: no JPEG bitstream or entropy coder is implemented. Display clips to [0,1]; MSE uses unclipped values.",
      "正交正規二維 DCT 後，包含直流係數在內皆採均勻純量量化。步長為 0 時略過量化。此處只探討一個編碼階段，未實作 JPEG 位元串流或熵編碼。顯示截斷至 [0,1]，均方誤差使用未截斷數值。",
    ),
    related: [IP("dip-08-compression-watermarking")],
    controls: [
      range("step", "Quantization step", "量化步長", 0.15, 0, 1, 0.025),
    ],
    compute(s, l) {
      const r = dctQuantize(n(s, "step"));
      return {
        svg: svg(
          text(45, 28, pick(l, "Original block", "原始區塊")) +
            text(365, 28, pick(l, "Decoded block", "解碼區塊")) +
            grid(r.input, 8, 45, 45) +
            grid(r.output, 8, 365, 45),
        ),
        summary: pick(
          l,
          "Larger quantization steps create more zero coefficients that an entropy coder could exploit. The zero count alone does not determine the number of bits.",
          "更大的量化步長產生更多零係數，讓熵編碼有機會節省位元；僅知道零值數量，仍無法決定實際位元數。",
        ),
        metrics: [
          metric(
            l,
            "Zero coefficients",
            "零係數數量",
            r.quant.filter((v) => Math.abs(v) < 1e-10).length,
          ),
          metric(l, "Unclipped MSE", "未截斷均方誤差", r.error),
          metric(
            l,
            "Squared coefficient error / 64",
            "係數平方誤差／64",
            mse(r.coeff, r.quant),
          ),
        ],
      };
    },
  },
  {
    id: "morphology-shapes",
    track: "image-processing",
    title: L(
      "IP 09 · A structuring element defines what survives",
      "IP 09・結構元素決定哪些形狀留下",
    ),
    description: L(
      "Compare erosion, dilation, opening and closing on the same binary shape with holes and specks.",
      "在含孔洞與孤立點的同一張二值影像上，比較侵蝕、膨脹、開運算與閉運算。",
    ),
    challenge: L(
      "Remove the isolated speck without filling the central hole; then reverse the goal. Does changing the structuring-element shape matter?",
      "先移除孤立點而保留中央孔洞，再反過來操作。改變結構元素的形狀有差嗎？",
    ),
    note: L(
      "A 20 × 20 binary image with a zero-valued exterior. The diamond uses Manhattan distance ≤ radius; the square uses a full (2r+1)² neighborhood. Opening is erosion then dilation; closing reverses that order.",
      "使用 20 × 20 二值影像，邊界外視為 0。菱形元素採曼哈頓距離 ≤ 半徑；方形元素使用完整 (2r+1)² 鄰域。開運算先侵蝕再膨脹，閉運算則相反。",
    ),
    related: [IP("dip-09-morphology")],
    controls: [
      select("operation", "Operation", "運算", "open", [
        ["erode", "Erosion", "侵蝕"],
        ["dilate", "Dilation", "膨脹"],
        ["open", "Opening", "開運算"],
        ["close", "Closing", "閉運算"],
      ]),
      select("shape", "Structuring element", "結構元素", "diamond", [
        ["diamond", "Diamond", "菱形"],
        ["square", "Square", "方形"],
      ]),
      range("radius", "Radius", "半徑", 1, 1, 3, 1),
    ],
    compute(s, l) {
      const input = Array.from({ length: 400 }, (_, i) => {
        const x = i % 20,
          y = Math.floor(i / 20);
        return Number(
          (x >= 5 &&
            x <= 14 &&
            y >= 5 &&
            y <= 14 &&
            !(x >= 9 && x <= 10 && y >= 9 && y <= 10)) ||
            (x === 3 && y === 3) ||
            (x >= 14 && x <= 17 && y === 10),
        );
      });
      const output = binaryMorph(
        input,
        20,
        n(s, "radius"),
        String(s.shape),
        String(s.operation),
      );
      return {
        svg: svg(
          text(45, 28, pick(l, "Input", "輸入")) +
            text(365, 28, pick(l, "Output", "輸出")) +
            grid(input, 20, 45, 45) +
            grid(output, 20, 365, 45),
        ),
        summary: pick(
          l,
          "The operation asks whether a translated neighborhood fits inside the foreground or touches it. Shape and radius are part of the model, not cosmetic controls.",
          "運算判斷平移後的鄰域是否完全位於前景，或是否碰到前景。形狀與半徑是模型的一部分，並非外觀設定。",
        ),
        metrics: [
          metric(
            l,
            "Foreground before",
            "原始前景像素",
            input.reduce((a, b) => a + b, 0),
          ),
          metric(
            l,
            "Foreground after",
            "輸出前景像素",
            output.reduce((a, b) => a + b, 0),
          ),
          metric(
            l,
            "Changed pixels",
            "改變的像素",
            output.filter((v, i) => v !== input[i]).length,
          ),
        ],
      };
    },
  },
  {
    id: "segmentation-threshold",
    track: "image-processing",
    title: L(
      "IP 10 · A histogram can hide spatial illumination",
      "IP 10・直方圖可能隱藏空間照明變化",
    ),
    description: L(
      "Segment a disk under an illumination ramp using a manual or Otsu threshold and compare against the known mask.",
      "用手動或 Otsu 門檻分割照明漸變下的圓盤，並與已知遮罩比較。",
    ),
    challenge: L(
      "Increase the illumination ramp. Can the automatic global threshold still separate the whole disk from the background?",
      "增加照明漸變幅度。自動全域門檻還能把整個圓盤與背景分開嗎？",
    ),
    note: L(
      "Synthetic intensities: background 0.2, disk +0.42, horizontal illumination ramp, then clipping. Otsu searches 256 candidate thresholds and takes the first maximum. This differs from implementations that average ties. IoU uses the known synthetic disk.",
      "合成亮度包含背景 0.2、圓盤加上 0.42，以及水平照明漸變，最後截斷。Otsu 搜尋 256 個候選門檻，取第一個最大值，與平均並列解的實作不同。IoU 使用已知的合成圓盤作為基準。",
    ),
    related: [IP("dip-10-segmentation")],
    controls: [
      range(
        "illumination",
        "Illumination ramp",
        "照明漸變幅度",
        0.3,
        0,
        1,
        0.05,
      ),
      select("method", "Threshold selection", "門檻選擇", "otsu", [
        ["otsu", "Otsu", "Otsu"],
        ["manual", "Manual", "手動"],
      ]),
      range("threshold", "Manual threshold", "手動門檻", 0.4, 0, 1, 0.02),
    ],
    compute(s, l) {
      const r = thresholdScene(
        n(s, "illumination"),
        n(s, "threshold"),
        s.method === "otsu",
      );
      return {
        svg: svg(
          text(45, 28, pick(l, "Observed intensities", "觀測亮度")) +
            text(365, 28, pick(l, "Predicted foreground", "預測前景")) +
            grid(r.image, 20, 45, 45) +
            grid(r.mask, 20, 365, 45),
        ),
        summary: pick(
          l,
          "Otsu maximizes between-class intensity separation; it does not know which pixels belong to the disk. A spatial lighting gradient can violate the global two-class intensity assumption.",
          "Otsu 最大化兩類亮度的分離程度，並不知道哪些像素屬於圓盤。空間照明漸變可能破壞全域亮度可分成兩類的假設。",
        ),
        metrics: [
          metric(l, "Applied threshold", "實際門檻", r.threshold),
          metric(l, "Mask IoU", "遮罩 IoU", r.iou),
          metric(
            l,
            "Foreground pixels",
            "前景像素",
            r.mask.reduce((a, b) => a + b, 0),
          ),
        ],
      };
    },
  },
  {
    id: "contour-energy",
    track: "image-processing",
    title: L(
      "IP 11 · Why a contour can miss the boundary",
      "IP 11・輪廓為何可能錯過邊界",
    ),
    description: L(
      "Move a circular contour through a narrow edge-attraction energy and observe initialization and tension.",
      "讓圓形輪廓沿狹窄的邊緣吸引能量移動，觀察初始化與張力的影響。",
    ),
    challenge: L(
      "Use the same tension but start at radius 0.5 and 1.2. Why can decreasing energy lead to different final contours?",
      "使用相同張力，分別從半徑 0.5 與 1.2 開始。為何能量都下降，最後輪廓卻不同？",
    ),
    note: L(
      "A one-parameter circular restriction, not a full snake or level-set solver: E(r)=αr²−exp(−((r−1)/0.16)²). Radius lies in [0,1.8]. Projected gradient steps use backtracking to avoid increasing energy. The dashed circle is the target boundary.",
      "這是只有一個參數的圓形限制模型，並非完整 snake 或水平集求解器：E(r)=αr²−exp(−((r−1)/0.16)²)，半徑限制在 [0,1.8]。投影梯度步驟使用回溯，避免能量增加。虛線圓為目標邊界。",
    ),
    related: [IP("dip-11-active-contours")],
    controls: [
      range("radius", "Initial radius", "初始半徑", 1.3, 0.1, 1.8, 0.05),
      range("tension", "Tension α", "張力 α", 0.15, 0, 1, 0.05),
      range("steps", "Descent steps", "下降步數", 30, 0, 80, 1),
    ],
    compute(s, l) {
      const r = radialSnake(n(s, "radius"), n(s, "tension"), n(s, "steps"));
      const energies = Array.from({ length: 81 }, (_, i) =>
          r.energy((i * 1.8) / 80),
        ),
        lo = Math.min(...energies),
        hi = Math.max(...energies),
        py = (v: number) => 240 - (180 * (v - lo)) / (hi - lo);
      const body =
        `<circle cx="165" cy="155" r="65" fill="none" stroke="var(--plot-a)" stroke-dasharray="5 3"/><circle cx="165" cy="155" r="${65 * n(s, "radius")}" fill="none" stroke="var(--plot-muted)"/><circle cx="165" cy="155" r="${65 * r.radius}" fill="none" stroke="var(--plot-b)" stroke-width="3"/>` +
        path(
          energies.map((e, i) => [350 + i * 3.1, py(e)]),
          `stroke="var(--plot-a)" stroke-width="2"`,
        ) +
        `<circle cx="${350 + (r.radius / 1.8) * 248}" cy="${py(r.finalEnergy)}" r="5" fill="var(--plot-b)"/>` +
        text(
          60,
          285,
          pick(l, "Gray: initial; color: final", "灰色：起點；彩色：終點"),
        ) +
        text(365, 285, pick(l, "Radius 0 → 1.8", "半徑 0 → 1.8"));
      return {
        svg: svg(body),
        summary: pick(
          l,
          "Local descent balances shrinking tension against edge attraction. Outside the edge’s capture range, the contour can collapse even though every step lowers energy.",
          "局部下降平衡收縮張力與邊緣吸引力。在邊緣的捕捉範圍之外，即使每一步都降低能量，輪廓仍可能塌縮。",
        ),
        metrics: [
          metric(l, "Final radius", "最終半徑", r.radius),
          metric(l, "Initial energy", "初始能量", r.initialEnergy),
          metric(l, "Final energy", "最終能量", r.finalEnergy),
        ],
      };
    },
  },
  {
    id: "corner-tensor",
    track: "image-processing",
    title: L(
      "IP 12 · Flat patch, edge, or corner?",
      "IP 12・平坦區域、邊緣，還是角點？",
    ),
    description: L(
      "Build a structure tensor from local gradients and separate orientation from corner strength.",
      "由局部梯度建立結構張量，分辨方向與角點強度。",
    ),
    challenge: L(
      "Rotate an edge through 90 degrees. Which tensor entries change, and which eigenvalues remain constant? Then double contrast.",
      "將邊緣旋轉 90 度。哪些張量元素改變，哪些特徵值保持不變？接著加倍對比。",
    ),
    note: L(
      "Two synthetic gradient observations, summed with unit weights: edge=(1,0),(1,0); corner=(1,0),(0,1); flat=(0,0),(0,0). Rotation and contrast apply to both gradients. Harris response is det(M)−0.05 tr(M)². This isolates the descriptor, not image-wide detection.",
      "使用兩個等權重合成梯度：邊緣為 (1,0)、(1,0)；角點為 (1,0)、(0,1)；平坦區域皆為 (0,0)。旋轉與對比作用於兩個梯度。Harris 響應為 det(M)−0.05 tr(M)²。此處探討描述子的原理，未進行整張影像的偵測。",
    ),
    related: [IP("dip-12-feature-extraction")],
    controls: [
      select("kind", "Local structure", "局部結構", "corner", [
        ["flat", "Flat", "平坦"],
        ["edge", "Edge", "邊緣"],
        ["corner", "Corner", "角點"],
      ]),
      range("angle", "Rotation (degrees)", "旋轉角度", 30, 0, 180, 5),
      range("contrast", "Contrast multiplier", "對比倍率", 1, 0, 2, 0.1),
    ],
    compute(s, l) {
      const r = structureTensor(
        String(s.kind),
        n(s, "angle"),
        n(s, "contrast"),
      );
      const body =
        r.g
          .map((g, i) =>
            path(
              [
                [170, 155],
                [170 + g[0] * 50, 155 - g[1] * 50],
              ],
              `stroke="var(--plot-${"ab"[i]})" stroke-width="${i ? 3 : 7}"`,
            ),
          )
          .join("") +
        text(55, 285, pick(l, "Observed gradient vectors", "觀測梯度向量")) +
        text(350, 70, "M = Σ g gᵀ") +
        text(350, 115, `[ ${fmt(r.xx)}   ${fmt(r.xy)} ]`) +
        text(350, 145, `[ ${fmt(r.xy)}   ${fmt(r.yy)} ]`) +
        text(350, 205, `λ₁ = ${fmt(r.eigen[0])}`) +
        text(350, 235, `λ₂ = ${fmt(r.eigen[1])}`);
      return {
        svg: svg(body),
        summary: pick(
          l,
          "An edge has one weak direction: motion along it changes little. A corner has two strong directions. Rotation preserves eigenvalues; contrast scales them quadratically and the response quartically.",
          "邊緣具有一個弱方向：沿邊緣移動時變化很小；角點具有兩個強方向。旋轉保持特徵值不變，對比讓特徵值按平方、響應按四次方縮放。",
        ),
        metrics: [
          metric(l, "Largest eigenvalue", "最大特徵值", r.eigen[0]),
          metric(l, "Smallest eigenvalue", "最小特徵值", r.eigen[1]),
          metric(l, "Harris response", "Harris 響應", r.response),
        ],
      };
    },
  },
  {
    id: "prototype-scaling",
    track: "image-processing",
    title: L(
      "IP 13 · Feature units move the decision boundary",
      "IP 13・特徵單位會移動決策邊界",
    ),
    description: L(
      "Change one feature’s scale in a nearest-prototype classifier and watch the classification regions move.",
      "改變最近原型分類器中一個特徵的尺度，觀察分類區域如何移動。",
    ),
    challenge: L(
      "Place the query near the diagonal boundary and multiply the second feature by four. Did new evidence arrive, or did the distance definition change?",
      "將查詢點放在斜向邊界附近，再把第二個特徵乘以四。這是新增證據，還是改變了距離定義？",
    ),
    note: L(
      "Two fixed synthetic class means: A=(0.25,0.7), B=(0.75,0.3). Distance is (x−μx)²+s²(y−μy)²; both query and prototype use the same scaling. No training or claim of statistical optimality. Ties choose A.",
      "兩個固定合成類別平均值：A=(0.25,0.7)、B=(0.75,0.3)。距離為 (x−μx)²+s²(y−μy)²；查詢點與原型使用相同縮放。未進行訓練，也不宣稱統計最優性。並列時選 A。",
    ),
    related: [IP("dip-13-pattern-classification")],
    controls: [
      range("x", "Query feature x", "查詢特徵 x", 0.6, 0, 1, 0.025),
      range("y", "Query feature y", "查詢特徵 y", 0.65, 0, 1, 0.025),
      range("scale", "Feature y multiplier", "特徵 y 倍率", 1, 0.25, 4, 0.25),
    ],
    compute(s, l) {
      const d = prototypeDistances(n(s, "x"), n(s, "y"), n(s, "scale"));
      let body = "";
      for (let y = 0; y < 20; y++)
        for (let x = 0; x < 32; x++) {
          const q = prototypeDistances(
            (x + 0.5) / 32,
            1 - (y + 0.5) / 20,
            n(s, "scale"),
          );
          body += `<rect x="${60 + x * 16}" y="${40 + y * 10}" width="16.1" height="10.1" fill="var(--plot-${q[0] <= q[1] ? "a" : "b"})" opacity=".22"/>`;
        }
      body +=
        text(60 + 0.25 * 512, 240 - 0.7 * 200, "A") +
        text(60 + 0.75 * 512, 240 - 0.3 * 200, "B") +
        `<circle cx="${60 + n(s, "x") * 512}" cy="${240 - n(s, "y") * 200}" r="6" fill="var(--plot-ink)"/>` +
        text(
          60,
          275,
          pick(
            l,
            "Horizontal: x; vertical: y (original units)",
            "水平：x；垂直：y（原始單位）",
          ),
        );
      return {
        svg: svg(body),
        summary: pick(
          l,
          "Distance-based decisions depend on the units of every feature. Standardization is a modeling choice that changes which differences count as large.",
          "以距離為基礎的決策取決於每個特徵的單位。標準化是一種模型選擇，會改變哪些差異被視為很大。",
        ),
        metrics: [
          metric(l, "Squared distance to A", "到 A 的平方距離", d[0]),
          metric(l, "Squared distance to B", "到 B 的平方距離", d[1]),
          metric(l, "Prediction", "預測", d[0] <= d[1] ? "A" : "B"),
        ],
      };
    },
  },
  {
    id: "receptive-field",
    track: "deep-learning",
    title: L(
      "DLCV 01 · How much input can one feature see?",
      "DLCV 01・一個特徵能看到多少輸入？",
    ),
    description: L(
      "Stack convolutional layers and trace receptive-field size and the spacing between neighboring outputs.",
      "堆疊卷積層，追蹤感受野大小與相鄰輸出間距。",
    ),
    challenge: L(
      "Compare three stride-1 layers with three stride-2 layers. Why does the later kernel enlarge the receptive field faster?",
      "比較三層步幅 1 與三層步幅 2。為什麼後面的卷積核能更快擴大感受野？",
    ),
    note: L(
      "One-dimensional, single-path stack with kernel size 3 and constant stride/dilation. r←r+2dj and j←sj, starting at r=j=1. The result is theoretical span, not learned effective influence; dilation can leave holes. Padding and finite input borders are excluded.",
      "一維單一路徑堆疊，卷積核大小為 3，步幅與膨脹率固定。從 r=j=1 開始，更新 r←r+2dj、j←sj。結果是理論跨度，不是學得的有效影響；膨脹卷積可能留下空隙。未考慮填補與有限輸入邊界。",
    ),
    related: [DL("dl-01-deep-learning-for-image-processing")],
    controls: [
      range("depth", "Number of layers", "層數", 3, 1, 5, 1),
      range("stride", "Stride per layer", "每層步幅", 1, 1, 2, 1),
      range("dilation", "Dilation per layer", "每層膨脹率", 1, 1, 3, 1),
    ],
    compute(s, l) {
      const r = receptiveField(n(s, "depth"), n(s, "stride"), n(s, "dilation"));
      const body = r.layers
        .map(
          (v, i) =>
            text(
              40,
              40 + i * 40,
              `${i === 0 ? pick(l, "Input", "輸入") : `L${i}`}   r=${v.field}, j=${v.jump}`,
            ) +
            `<rect x="230" y="${25 + i * 40}" width="${(340 * v.field) / r.field}" height="20" fill="var(--plot-a)" opacity="${0.3 + (0.7 * i) / (r.layers.length - 1)}"/>`,
        )
        .join("");
      return {
        svg: svg(body, 310),
        summary: pick(
          l,
          "A deeper layer operates on features already spaced apart in input coordinates. Stride multiplies that spacing, so every later kernel spans a larger input region.",
          "深層處理的特徵在輸入座標中已彼此分隔。步幅乘上這個間距，使後續每個卷積核涵蓋更大的輸入區域。",
        ),
        metrics: [
          metric(l, "Theoretical receptive span", "理論感受野跨度", r.field),
          metric(
            l,
            "Output spacing in input pixels",
            "輸出對應的輸入像素間距",
            r.jump,
          ),
        ],
      };
    },
  },
  {
    id: "classification-threshold",
    track: "deep-learning",
    title: L(
      "DLCV 02 · A decision threshold changes which mistakes you make",
      "DLCV 02・決策門檻改變錯誤的種類",
    ),
    description: L(
      "Move the decision threshold on fixed classifier scores and inspect the confusion matrix, precision, recall and error cost.",
      "移動固定分類分數的決策門檻，觀察混淆矩陣、精確率、召回率與錯誤成本。",
    ),
    challenge: L(
      "Find a threshold with no false positives. How many positives are missed? Increase the false-positive cost and compare the preferred operating point.",
      "找出沒有偽陽性的門檻。漏掉了多少正例？提高偽陽性成本，比較較合適的操作點。",
    ),
    note: L(
      "Ten fixed synthetic scores with five positive labels. Predict class 1 when score ≥ threshold; no model is trained. Precision is undefined if nothing is predicted positive. Mean error cost = (c×FP+FN)/10, with false-negative cost fixed at 1. Adjusting c changes evaluation, not scores or decisions.",
      "十筆固定合成分數包含五個正例。分數 ≥ 門檻時預測類別 1；未進行模型訓練。若未預測任何正例，精確率無定義。平均錯誤成本 = (c×FP+FN)/10，偽陰性成本固定為 1。調整 c 改變評估成本，不會改變分數或決策。",
    ),
    related: [DL("dl-02-image-classification")],
    controls: [
      range("threshold", "Decision threshold", "決策門檻", 0.5, 0, 1, 0.01),
      range("cost", "False-positive cost c", "偽陽性成本 c", 1, 0.25, 5, 0.25),
    ],
    compute(s, l) {
      const r = classificationThreshold(n(s, "threshold"), n(s, "cost"));
      let body =
        text(160, 30, pick(l, "Predict 1", "預測 1")) +
        text(380, 30, pick(l, "Predict 0", "預測 0")) +
        text(20, 100, pick(l, "True 1", "真實 1")) +
        text(20, 185, pick(l, "True 0", "真實 0"));
      [r.tp, r.fn, r.fp, r.tn].forEach((v, i) => {
        const x = 130 + (i % 2) * 220,
          y = 50 + Math.floor(i / 2) * 85;
        body +=
          `<rect x="${x}" y="${y}" width="195" height="70" rx="5" fill="var(--plot-${i === 0 || i === 3 ? "a" : "b"})" opacity=".2"/>` +
          text(x + 65, y + 42, `${["TP", "FN", "FP", "TN"][i]} = ${v}`);
      });
      body +=
        text(
          40,
          245,
          pick(
            l,
            "Scores, descending (● = true positive class):",
            "分數由高至低（●＝真實正類）：",
          ),
        ) +
        text(
          40,
          275,
          r.scores
            .map((v, i) => `${r.truth[i] ? "●" : "○"}${fmt(v, 2)}`)
            .join("  "),
        );
      return {
        svg: svg(body),
        summary: pick(
          l,
          `True positives (TP): ${r.tp}; false positives (FP): ${r.fp}; false negatives (FN): ${r.fn}; true negatives (TN): ${r.tn}. Lowering the threshold can recover missed positives while admitting false positives. Precision and recall use different denominators; accuracy alone cannot express unequal error costs.`,
          `真陽性（TP）：${r.tp}；偽陽性（FP）：${r.fp}；偽陰性（FN）：${r.fn}；真陰性（TN）：${r.tn}。降低門檻可能找回漏掉的正例，也可能增加偽陽性。精確率與召回率使用不同分母；單靠正確率無法表達不對稱的錯誤成本。`,
        ),
        metrics: [
          metric(
            l,
            "Precision",
            "精確率",
            r.precision === null ? pick(l, "Undefined", "無定義") : r.precision,
          ),
          metric(l, "Recall", "召回率", r.recall),
          metric(l, "F1", "F1", r.f1),
          metric(l, "Accuracy", "正確率", r.accuracy),
          metric(l, "Mean error cost", "平均錯誤成本", r.cost),
        ],
      };
    },
  },
  {
    id: "restoration-ambiguity",
    track: "deep-learning",
    title: L(
      "DLCV 04 · A sharp guess can be less faithful",
      "DLCV 04・清晰的猜測可能較不忠實",
    ),
    description: L(
      "Two sharp signals produce the same observation. Compare their conditional mean with a sharper, committed reconstruction.",
      "兩個清晰訊號產生相同觀測。比較條件平均值與較清晰、但選定單一假設的重建。",
    ),
    challenge: L(
      "At equal probabilities, compare the mean with a complete guess of A. Which minimizes expected squared error? Does that make it a possible clean signal?",
      "兩種情況等機率時，比較平均值與完全猜測 A。哪個最小化期望平方誤差？它就一定是可能的乾淨訊號嗎？",
    ),
    note: L(
      "Two fixed binary patterns A and B share the same global mean observation. P(A|y)=p is supplied, not learned. The slider blends the conditional mean toward A. Expected MSE averages over both possible truths; sharpness here is adjacent-sample total variation, not a perceptual quality score.",
      "兩個固定二值模式 A、B 具有相同的全域平均觀測。P(A|y)=p 由設定提供，並非學得。滑桿將條件平均值逐漸混合至 A。期望均方誤差涵蓋兩種可能真相；此處以相鄰取樣值的總變差表示清晰程度，並非感知品質分數。",
    ),
    related: [DL("dl-04-image-restoration")],
    controls: [
      range(
        "probability",
        "Probability of clean pattern A",
        "乾淨模式 A 的機率",
        0.5,
        0.1,
        0.9,
        0.05,
      ),
      range(
        "mix",
        "Commitment toward pattern A",
        "偏向模式 A 的程度",
        0.5,
        0,
        1,
        0.05,
      ),
    ],
    compute(s, l) {
      const r = ambiguousRestoration(n(s, "probability"), n(s, "mix"));
      const variation = r.output
        .slice(1)
        .reduce((a, v, i) => a + Math.abs(v - r.output[i]), 0);
      return {
        svg: svg(
          lines(
            [r.a, r.b, r.output],
            ["A", "B", pick(l, "Reconstruction", "重建")],
            0,
            1,
          ),
        ),
        summary: pick(
          l,
          "Averaging incompatible possibilities minimizes expected squared error, but can remove structures present in every valid clean outcome. Choosing one restores structure while risking the wrong structure.",
          "平均互不相容的可能性可最小化期望平方誤差，卻可能移除所有合法乾淨結果都具有的結構。選定一種結果能恢復結構，也可能恢復錯誤的結構。",
        ),
        metrics: [
          metric(l, "Expected reconstruction MSE", "重建期望均方誤差", r.risk),
          metric(
            l,
            "Conditional-mean minimum MSE",
            "條件平均最小均方誤差",
            r.riskMean,
          ),
          metric(l, "Adjacent total variation", "相鄰取樣總變差", variation),
        ],
      };
    },
  },
  {
    id: "lowlight-photons",
    track: "deep-learning",
    title: L(
      "DLCV 05 · Gain cannot create missing photons",
      "DLCV 05・增益無法補回缺少的光子",
    ),
    description: L(
      "Separate exposure, read noise and digital gain using the mean and variance of a photon-counting model.",
      "用光子計數模型的平均值與變異數，分辨曝光、讀出雜訊與數位增益。",
    ),
    challenge: L(
      "Double digital gain, then double exposure. Which changes the signal-to-noise ratio, and why?",
      "先加倍數位增益，再加倍曝光。哪個會改變訊雜比？為什麼？",
    ),
    note: L(
      "Flat scene x=0.25, electrons N~Poisson(exposure×x), independent zero-mean read noise with standard deviation σr electrons. Output is gain×(N+noise). Curves show exact mean ±1 standard deviation, not simulated images or confidence intervals. Clipping and quantization are excluded.",
      "平坦場景 x=0.25，電子數 N~Poisson(曝光×x)，並加入獨立、零均值、標準差為 σr 電子的讀出雜訊。輸出為增益×(N+雜訊)。曲線顯示精確平均值 ±1 標準差，並非模擬影像或信賴區間。未考慮截斷與量化。",
    ),
    related: [DL("dl-05-low-light-enhancement")],
    controls: [
      range(
        "exposure",
        "Exposure scale (electrons at x=1)",
        "曝光尺度（x=1 時的電子數）",
        40,
        4,
        200,
        4,
      ),
      range("gain", "Digital gain", "數位增益", 2, 1, 8, 0.5),
      range(
        "read",
        "Read-noise standard deviation",
        "讀出雜訊標準差",
        2,
        0,
        8,
        0.5,
      ),
    ],
    compute(s, l) {
      const r = photonNoise(n(s, "exposure"), n(s, "gain"), n(s, "read"));
      const values = Array.from({ length: 32 }, (_, i) =>
        photonNoise(4 + (i * 196) / 31, n(s, "gain"), n(s, "read")),
      );
      const body = lines(
        [
          values.map((v) => v.mean),
          values.map((v) => v.mean + v.std),
          values.map((v) => v.mean - v.std),
        ],
        [
          pick(l, "Mean", "平均"),
          pick(l, "Mean + σ", "平均 + σ"),
          pick(l, "Mean − σ", "平均 − σ"),
        ],
      );
      const selectedX = 70 + (530 * (n(s, "exposure") - 4)) / 196;
      const selectedMarker = `<line x1="${selectedX}" y1="45" x2="${selectedX}" y2="245" stroke="var(--plot-ink)" stroke-dasharray="3 3"/>`;
      return {
        svg: svg(
          text(
            70,
            20,
            pick(l, "Exposure sweep: 4 → 200", "曝光掃描：4 → 200"),
          ) +
            body +
            selectedMarker,
        ),
        summary: pick(
          l,
          "Digital gain multiplies signal and noise equally, leaving their ratio unchanged. More collected photons improve relative precision; read noise dominates most when the signal is small.",
          "數位增益同時乘上訊號與雜訊，因此不改變兩者比例。收集更多光子可改善相對精度；訊號微弱時，讀出雜訊的影響尤其明顯。",
        ),
        metrics: [
          metric(l, "Selected output mean", "所選輸出平均值", r.mean),
          metric(
            l,
            "Selected output standard deviation",
            "所選輸出標準差",
            r.std,
          ),
          metric(l, "Signal/noise ratio (linear)", "訊雜比（線性）", r.snr),
        ],
      };
    },
  },
  {
    id: "tracking-association",
    track: "deep-learning",
    title: L(
      "DLCV 06 · Crossing objects challenge identity",
      "DLCV 06・交錯物體考驗身分關聯",
    ),
    description: L(
      "Balance positional and appearance costs and solve the complete two-object one-to-one assignment.",
      "權衡位置與外觀成本，求解兩物體的一對一完整指派。",
    ),
    challenge: L(
      "Make the objects cross, then increase motion weight. Under what appearance error does adding appearance stop helping?",
      "讓兩個物體交錯，再提高運動權重。外觀誤差多大時，加入外觀不再有幫助？",
    ),
    note: L(
      "Two predicted positions are fixed at 0.3 and 0.7. Detection identities are known only for evaluation. Cost = w×absolute position error+(1−w)×appearance cost. Both permutations are enumerated exactly; no detector, learned embedding, birth/death logic or Hungarian implementation. Ties choose identity order.",
      "兩個預測位置固定為 0.3 與 0.7。偵測身分僅供評估。成本 = w×位置絕對誤差+(1−w)×外觀成本。精確列舉兩個排列；未實作偵測器、學得的嵌入、軌跡新增刪除或 Hungarian 演算法。並列時選身分順序。",
    ),
    related: [DL("dl-06-object-tracking")],
    controls: [
      range("crossing", "Crossing progress", "交錯進度", 1, 0, 1, 0.05),
      range("motion", "Motion weight w", "運動權重 w", 0.7, 0, 1, 0.05),
      range(
        "appearance",
        "Same-identity appearance cost",
        "同身分外觀成本",
        0.1,
        0,
        1,
        0.05,
      ),
    ],
    compute(s, l) {
      const r = trackingAssignment(
        n(s, "motion"),
        n(s, "appearance"),
        n(s, "crossing"),
      );
      let body = text(
        40,
        30,
        pick(l, "Predicted tracks → current detections", "預測軌跡 → 當前偵測"),
      );
      r.predictions.forEach((x, i) => {
        body +=
          `<circle cx="${70 + x * 500}" cy="75" r="10" fill="var(--plot-${"ab"[i]})"/><circle cx="${70 + r.detections[i] * 500}" cy="195" r="10" fill="var(--plot-${"ab"[i]})"/>` +
          path(
            [
              [70 + x * 500, 90],
              [70 + r.detections[r.assignment[i]] * 500, 180],
            ],
            `stroke="var(--plot-${"ab"[i]})" stroke-width="2.5"`,
          );
      });
      body +=
        text(
          40,
          250,
          `C = [${r.cost[0].map((v) => fmt(v)).join(", ")}; ${r.cost[1].map((v) => fmt(v)).join(", ")}]`,
        ) +
        text(
          40,
          280,
          pick(
            l,
            "Colors show true identity; lines show assignment.",
            "顏色表示真實身分；連線表示指派。",
          ),
        );
      return {
        svg: svg(body),
        summary: pick(
          l,
          r.ties
            ? "Both assignments have equal cost. The deterministic tie break is not evidence of identity."
            : "Matching nearest positions can switch identities after a crossing. Appearance helps only when its cost is informative.",
          r.ties
            ? "兩種指派成本相同。固定的並列處理規則並非身分證據。"
            : "交錯後，匹配最近位置可能交換身分。只有在外觀成本提供可靠訊息時，外觀才有幫助。",
        ),
        metrics: [
          metric(l, "Identity-preserving cost", "保持身分的成本", r.direct),
          metric(l, "Swapped-identity cost", "交換身分的成本", r.swapped),
          metric(
            l,
            "Identity switches",
            "身分交換數",
            r.assignment[0] === 0 ? 0 : 2,
          ),
        ],
      };
    },
  },
  {
    id: "world-model-rollout",
    track: "deep-learning",
    title: L(
      "DLCV 07 · Small dynamics errors compound in rollouts",
      "DLCV 07・小型動態誤差會在展開中累積",
    ),
    description: L(
      "Compare one-step predictions, open-loop rollout and periodic observation refresh under the same actions.",
      "在相同行動下，比較單步預測、開迴路展開與定期更新觀測。",
    ),
    challenge: L(
      "Set the model coefficient to the true value 0.96. Then change it to 1.02 and increase the horizon. How much can observation refresh correct?",
      "先把模型係數設為真值 0.96，再改成 1.02 並增加展開長度。更新觀測能修正多少誤差？",
    ),
    note: L(
      "True scalar state: x(t+1)=0.96x(t)+u(t), x(0)=1, u(t)=a sin(0.45t). The model replaces 0.96 with a chosen coefficient. Refresh feeds the true state every k steps; k=0 disables it. Fully observed deterministic dynamics, no learned video model or replanning.",
      "真實純量狀態為 x(t+1)=0.96x(t)+u(t)，x(0)=1，u(t)=a sin(0.45t)。模型以選定係數取代 0.96。更新每 k 步輸入真實狀態；k=0 表示停用。此處為完全可觀測的確定性動態，未使用學得的影片模型或重新規劃。",
    ),
    related: [DL("dl-07-world-models")],
    controls: [
      range(
        "coefficient",
        "Model transition coefficient",
        "模型轉移係數",
        1.02,
        0.85,
        1.08,
        0.01,
      ),
      range("steps", "Rollout horizon", "展開長度", 30, 5, 60, 1),
      range("action", "Action amplitude", "行動幅度", 0.08, 0, 0.2, 0.01),
      range(
        "refresh",
        "Observation refresh interval (0 = off)",
        "觀測更新間隔（0＝關閉）",
        5,
        0,
        15,
        1,
      ),
    ],
    compute(s, l) {
      const r = worldRollout(
        n(s, "coefficient"),
        n(s, "steps"),
        n(s, "action"),
        n(s, "refresh"),
      );
      return {
        svg: svg(
          lines(
            [r.truth, r.open, r.corrected],
            [
              pick(l, "True state", "真實狀態"),
              pick(l, "Open loop", "開迴路"),
              pick(l, "With observations", "加入觀測"),
            ],
          ),
        ),
        summary: pick(
          l,
          "One-step evaluation repeatedly supplies the true state. An open-loop rollout must consume its own imperfect predictions. Periodic observations limit drift without improving the transition model itself.",
          "單步評估反覆提供真實狀態；開迴路展開必須使用自身不完美的預測。定期觀測可限制漂移，但不會改善轉移模型本身。",
        ),
        metrics: [
          metric(l, "One-step MSE", "單步均方誤差", r.oneError),
          metric(l, "Open-loop MSE", "開迴路均方誤差", r.openError),
          metric(
            l,
            "Refreshed-rollout MSE",
            "更新觀測後均方誤差",
            r.correctedError,
          ),
        ],
      };
    },
  },
];
