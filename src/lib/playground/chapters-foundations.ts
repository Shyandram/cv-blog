import { L, pick, num, svg, text, fmt, type Experiment } from './types.ts';
export const chapterFoundationsExperiments: Experiment[] = [{
        id: 'array-dtype', track: 'foundations', title: L('When brightening makes a pixel darker', '為什麼增亮反而讓像素變暗'),
        description: L('Compare 8-bit wraparound, clipping and floating-point arithmetic on the same RGB pixel.', '對同一個 RGB 像素比較 8 位元回繞、截斷與浮點運算。'),
        related: ['foundations/images-as-arrays'],
        challenge: L('Set red to 240 and add 40. Compare the wrapped byte with the clipped value.', '把紅色設成 240 再加 40，比較回繞位元組與截斷值。'),
        note: L('The RGB channels are (R,100,40); add the offset to each. Wrap uses modulo 256. Clip keeps values in [0,255]. Float preserves the arithmetic result; its preview still needs display clipping. This illustrates arithmetic semantics, not a physical exposure model.', 'RGB 通道為 (R,100,40)，各自加上偏移量。回繞使用模 256，截斷保留在 [0,255]，浮點保留運算結果，但預覽仍須截斷至顯示範圍。此處示範運算語義，不是物理曝光模型。'),
        controls: [{ key: 'red', kind: 'range', label: L('Red byte R', '紅色位元組 R'), min: 0, max: 255, step: 1, value: 240 }, { key: 'offset', kind: 'range', label: L('Added offset', '增加的偏移量'), min: -100, max: 100, step: 1, value: 40 }],
        compute(s, lang) {
            const input = [num(s, 'red'), 100, 40], raw = input.map(x => x + num(s, 'offset')), wrapped = raw.map(x => (x % 256 + 256) % 256), clipped = raw.map(x => Math.max(0, Math.min(255, x)));
            let b = '';
            [[input, pick(lang, 'Original', '原始')], [wrapped, pick(lang, 'Uint8 wrap', 'Uint8 回繞')], [clipped, pick(lang, 'Clip', '截斷')], [raw, pick(lang, 'Float', '浮點')]].forEach(([values, label], i) => { const v = values as number[], x = 20 + i * 155; b += text(x + 65, 40, String(label), 'text-anchor="middle"') + `<rect x="${x}" y="65" width="130" height="130" rx="8" fill="rgb(${v.map(n => Math.max(0, Math.min(255, n))).join(',')})"/>` + text(x + 65, 235, v.map(n => fmt(n)).join(', '), 'text-anchor="middle"'); });
            return { svg: svg(b, 275), summary: pick(lang, 'Choose the arithmetic dtype before applying an operation. Casting after overflow cannot recover the lost value.', '應在運算之前選擇資料型別。溢位之後才轉型，無法恢復遺失的值。'), metrics: [{ label: pick(lang, 'Raw red', '紅色原始計算'), value: fmt(raw[0]) }, { label: pick(lang, 'Wrapped red', '紅色回繞值'), value: fmt(wrapped[0]) }, { label: pick(lang, 'Clipped red', '紅色截斷值'), value: fmt(clipped[0]) }] };
        },
    }];
