import type { LabDef } from './types.ts';
import { L, pick, fmt, svg, text, path, num } from '../types.ts';
import type { Gray } from '../../imagekit/types.ts';
import { fft2d, radialLowpass, logSpectrum, type RadialKind } from '../../imagekit/fft.ts';
import { drawGray, drawOutOfRange } from '../../imagekit/draw.ts';
import { psnr } from '../../imagekit/metrics.ts';

const TOL = 0.02;

export const lab: LabDef = {
  kind: 'image', id: 'frequency-filter', track: 'image-processing',
  title: L('IP 04 · Does the sharpest low-pass filter give the best image?', 'IP 04・最銳利的低通濾波器能得到最好的影像嗎？'),
  description: L('Filter a real photo in the frequency domain and compare an ideal, a Butterworth and a Gaussian low-pass at the same cutoff.', '在頻率域濾波一張真實照片，並在相同截止頻率下比較理想、Butterworth 與高斯低通濾波器。'),
  challenge: L('Choose the ideal filter with cutoff 12 and look at the red and blue pixels around the edges. Switch to Gaussian at the same cutoff. Which filter rings, and which has the higher PSNR?', '選擇理想濾波器、截止頻率 12，觀察邊緣周圍的紅色與藍色像素。再切換成相同截止頻率的高斯濾波器。哪一種會產生振鈴？哪一種的 PSNR 較高？'),
  note: L('A real 128 × 128 crop of the camera test image. The transfer function multiplies its 2-D FFT (no padding, so the image is treated as periodic) and the inverse transform is not clipped before ripples are counted. Cutoff D0 is in cycles per image; Gaussian H = exp(−D²/2D0²); Butterworth H = 1/(1 + (D/D0)^2n). Overshoot counts pixels more than 0.02 outside the original image’s own gray range. PSNR compares the filtered image, clipped to [0, 1], with the original.', '使用相機測試影像中真實的 128 × 128 裁切區域。轉移函數乘在其二維 FFT 上（未補零，因此影像視為週期性），並在計算振鈴像素前不對反變換結果做截斷。截止頻率 D0 的單位是每張影像的週期數；高斯 H = exp(−D²/2D0²)；Butterworth H = 1/(1 + (D/D0)^2n)。過衝是指超出原影像灰階範圍 0.02 以上的像素。PSNR 是將濾波後影像截斷到 [0, 1] 後與原影像比較。'),
  related: ['image-processing/dip-04-frequency-domain'],
  controls: [
    { key: 'filter', label: L('Low-pass filter', '低通濾波器'), kind: 'select', value: 'ideal', options: [
      { value: 'ideal', label: L('Ideal', '理想') }, { value: 'butterworth', label: L('Butterworth', 'Butterworth') }, { value: 'gaussian', label: L('Gaussian', '高斯') }] },
    { key: 'cutoff', label: L('Cutoff D0 (cycles per image)', '截止頻率 D0（每張影像週期數）'), kind: 'range', value: 12, min: 4, max: 48, step: 1 },
    { key: 'order', label: L('Butterworth order n (Butterworth only)', 'Butterworth 階數 n（僅 Butterworth 使用）'), kind: 'range', value: 2, min: 1, max: 8, step: 1 },
  ],
  predict: {
    question: L('At the same cutoff, which of the three filters pushes the most pixels outside the original image’s brightness range (ringing)?', '在相同的截止頻率下，三種濾波器中哪一種會讓最多像素超出原影像的亮度範圍（振鈴）？'),
    options: [{ value: 'ideal', label: L('Ideal', '理想') }, { value: 'butterworth', label: L('Butterworth, n = 2', 'Butterworth，n = 2') }, { value: 'gaussian', label: L('Gaussian', '高斯') }],
    answer: 'ideal',
    explain: L('A sharp cut in frequency is a sinc in space, and a sinc oscillates around every edge, so the ideal filter overshoots. Butterworth softens the cut (its ringing grows with the order n) and the Gaussian is Gaussian in both domains, so it never overshoots. At cutoff 12 the ideal filter pushes about 1.6% of the pixels out of range and the Gaussian none, and the Gaussian also has the higher PSNR.', '頻率上的銳利截斷在空間中是 sinc 函數，而 sinc 會在每個邊緣附近振盪，所以理想濾波器會過衝。Butterworth 讓截斷變得平緩（階數 n 越高，振鈴越明顯）；高斯在兩個域中都是高斯形狀，因此不會過衝。截止頻率 12 時，理想濾波器約使 1.6% 的像素超出範圍，高斯則為零，而且高斯的 PSNR 也比較高。'),
  },
  assets: { 'camera-crop-128': 'gray' },
  panes: [
    { key: 'original', kind: 'canvas', label: L('Original crop', '原始裁切影像'), width: 128, height: 128 },
    { key: 'spectrum', kind: 'canvas', label: L('Spectrum kept by the filter (log scale)', '濾波器保留的頻譜（對數尺度）'), width: 128, height: 128 },
    { key: 'filtered', kind: 'canvas', label: L('Filtered image', '濾波後影像'), width: 128, height: 128 },
    { key: 'ripples', kind: 'canvas', label: L('Pixels outside the original range: brighter (red), darker (blue)', '超出原影像範圍的像素：較亮（紅）、較暗（藍）'), width: 128, height: 128 },
    { key: 'profile', kind: 'svg', label: L('One image row through a strong edge', '穿過強邊緣的一列像素'), width: 640, height: 250, wide: true },
  ],
  sources: [
    { title: 'Rafael C. Gonzalez & Richard E. Woods, Digital Image Processing, 4th ed., Pearson, 2018 (frequency-domain filtering)', url: 'https://www.imageprocessingplace.com/DIP-4E/dip4e_main_page.htm' },
  ],
  create({ lang, images }) {
    const img = images['camera-crop-128'] as Gray, { w, h } = img;
    const lo = img.data.reduce((a, b) => Math.min(a, b)), hi = img.data.reduce((a, b) => Math.max(a, b));
    const re0 = Float64Array.from(img.data), im0 = new Float64Array(w * h);
    fft2d(re0, im0, w, h);
    let total = 0;
    for (let i = 0; i < re0.length; i++) total += re0[i] ** 2 + im0[i] ** 2;
    const spec = logSpectrum(img.data, w, h);
    // The row with the largest single horizontal jump shows ringing best.
    let row = 0, best = -1;
    for (let y = 0; y < h; y++) for (let x = 8; x < w - 9; x++) { const d = Math.abs(img.data[y * w + x + 1] - img.data[y * w + x]); if (d > best) { best = d; row = y; } }
    const re = new Float64Array(w * h), im = new Float64Array(w * h);
    return {
      render(state, canvases) {
        const kind = (['ideal', 'butterworth', 'gaussian'].includes(String(state.filter)) ? String(state.filter) : 'ideal') as RadialKind;
        const d0 = num(state, 'cutoff'), order = num(state, 'order');
        const H = radialLowpass(w, h, kind, d0, order);
        let kept = 0;
        for (let i = 0; i < re.length; i++) { re[i] = re0[i] * H[i]; im[i] = im0[i] * H[i]; kept += (re0[i] ** 2 + im0[i] ** 2) * H[i] * H[i]; }
        fft2d(re, im, w, h, true);
        const out = Float32Array.from(re);
        const clipped = out.map(v => Math.max(0, Math.min(1, v)));
        let outside = 0;
        for (const v of out) if (v > hi + TOL || v < lo - TOL) outside++;
        // Spectrum, dimmed outside the pass band, with the cutoff circle.
        const shown = new Float32Array(w * h);
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) shown[y * w + x] = spec[y * w + x] * H[((y + h / 2) % h) * w + ((x + w / 2) % w)];
        drawGray(canvases.original, img);
        drawGray(canvases.spectrum, { w, h, data: shown });
        const ctx = canvases.spectrum.getContext('2d')!;
        ctx.strokeStyle = '#17a398'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(w / 2, h / 2, d0, 0, 2 * Math.PI); ctx.stroke();
        drawGray(canvases.filtered, { w, h, data: clipped });
        drawOutOfRange(canvases.ripples, { w, h, data: out }, lo, hi, TOL);
        // 1-D profile along the chosen row.
        const px = (x: number) => 50 + (x / (w - 1)) * 570, py = (v: number) => 215 - ((v + 0.15) / 1.35) * 185;
        const series = (get: (x: number) => number) => path(Array.from({ length: w }, (_, x) => [px(x), py(get(x))] as [number, number]), 'stroke-width="2"');
        let plot = path([[50, 30], [50, 215], [620, 215]], 'stroke="var(--plot-grid)"')
          + path([[50, py(hi)], [620, py(hi)]], 'stroke="var(--plot-muted)" stroke-dasharray="4 4"') + path([[50, py(lo)], [620, py(lo)]], 'stroke="var(--plot-muted)" stroke-dasharray="4 4"')
          + series(x => img.data[row * w + x]).replace('fill="none"', 'fill="none" stroke="var(--plot-a)"') + series(x => out[row * w + x]).replace('fill="none"', 'fill="none" stroke="var(--plot-b)"')
          + text(55, 22, pick(lang, 'Blue: original · Orange: filtered (not clipped) · Dashed: original min and max', '藍色：原始 · 橘色：濾波後（未截斷）· 虛線：原影像最小與最大值'))
          + text(50, 238, pick(lang, `column 0 → ${w - 1} of row ${row}`, `第 ${row} 列，欄 0 → ${w - 1}`)) + text(620, 238, pick(lang, 'brightness (0 to 1)', '亮度（0 到 1）'), 'text-anchor="end"');
        const fraction = outside / out.length, p = psnr(img.data, clipped);
        const name = pick(lang, kind === 'ideal' ? 'The ideal filter' : kind === 'gaussian' ? 'The Gaussian filter' : `The order-${order} Butterworth filter`, kind === 'ideal' ? '理想濾波器' : kind === 'gaussian' ? '高斯濾波器' : `${order} 階 Butterworth 濾波器`);
        const verdict = fraction > 0.002
          ? pick(lang, `${name} rings: ${fmt(100 * fraction, 2)}% of the pixels fall outside the original range.`, `${name}出現振鈴：${fmt(100 * fraction, 2)}% 的像素超出原影像範圍。`)
          : pick(lang, `${name} stays inside the original range: no visible ringing.`, `${name}沒有超出原影像範圍：看不到振鈴。`);
        return {
          summary: `${verdict} ${pick(lang, `Keeping the frequencies up to ${d0} preserves ${fmt(100 * kept / total, 1)}% of the image energy.`, `保留到頻率 ${d0} 為止，可保存 ${fmt(100 * kept / total, 1)}% 的影像能量。`)}`,
          metrics: [
            { label: pick(lang, 'PSNR vs original', '與原影像的 PSNR'), value: `${fmt(p, 1)} dB` },
            { label: pick(lang, 'Pixels out of range', '超出範圍的像素'), value: `${fmt(100 * fraction, 2)}%` },
            { label: pick(lang, 'Energy kept', '保留的能量'), value: `${fmt(100 * kept / total, 1)}%` },
          ],
          plots: { profile: svg(plot, 250) },
        };
      },
    };
  },
};
