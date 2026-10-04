import { L, pick, num, fmt, svg, text, path, type Experiment } from './types.ts';

export function transformPoint(x: number, y: number, angle: number, stretch: number, shear: number): [number, number] {
  const a = angle * Math.PI / 180, u = stretch * x + shear * y;
  return [Math.cos(a) * u - Math.sin(a) * y, Math.sin(a) * u + Math.cos(a) * y];
}

export const geometryExperiments: Experiment[] = [{
  id: 'linear-transform', track: 'foundations',
  title: L('A matrix moves a coordinate system', '矩陣如何改變座標系'),
  description: L('Rotate, stretch and shear a unit square. Watch its basis vectors and area change together.', '旋轉、伸縮與剪切單位正方形，觀察基底向量與面積如何一起改變。'),
  challenge: L('Set the horizontal stretch to zero. Can you recover both original coordinates from the result?', '把水平伸縮設成零。你還能由結果還原兩個原始座標嗎？'),
  note: L('The map is rotation × shear/stretch, with the origin fixed. Its determinant is the horizontal stretch; absolute determinant is the area multiplier.', '此映射先剪切與伸縮，再旋轉，原點保持固定。行列式等於水平伸縮值，其絕對值就是面積倍率。'),
  related: ['foundations/images-as-arrays', 'ml/ml-01a-linear-algebra'],
  controls: [
    { key: 'angle', label: L('Rotation (degrees)', '旋轉角度（度）'), kind: 'range', min: -180, max: 180, step: 5, value: 30 },
    { key: 'stretch', label: L('Horizontal stretch', '水平伸縮'), kind: 'range', min: -2, max: 2, step: 0.1, value: 1.5 },
    { key: 'shear', label: L('Shear', '剪切'), kind: 'range', min: -1, max: 1, step: 0.1, value: 0.3 },
  ],
  compute(state, lang) {
    const angle = num(state, 'angle'), stretch = num(state, 'stretch'), shear = num(state, 'shear');
    const map = (x: number, y: number) => transformPoint(x, y, angle, stretch, shear);
    let body = '';
    for (let panel = 0; panel < 2; panel++) {
      const ox = 160 + panel * 320, oy = 195, scale = 39;
      const screen = ([x,y]: [number, number]): [number, number] => [ox+x*scale, oy-y*scale];
      for (let i = -3; i <= 3; i++) {
        body += path([[ox-3*scale, oy+i*scale],[ox+3*scale, oy+i*scale]], 'stroke="var(--plot-grid)"');
        body += path([[ox+i*scale, oy-3*scale],[ox+i*scale, oy+3*scale]], 'stroke="var(--plot-grid)"');
      }
      body += text(ox, 30, panel ? pick(lang,'Transformed','映射後') : pick(lang,'Original','原始'), 'text-anchor="middle"');
      const vertices: [number, number][] = [[0,0],[1,0],[1,1],[0,1],[0,0]];
      const points = vertices.map(([x,y]) => screen(panel ? map(x,y) : [x,y]));
      body += `<polygon points="${points.map(p=>p.join(',')).join(' ')}" fill="var(--plot-a)" fill-opacity="0.16" stroke="var(--plot-a)" stroke-width="2"/>`;
      for (const [x,y,label,color] of [[1,0,'e₁','var(--plot-a)'],[0,1,'e₂','var(--plot-b)']] as const) {
        const end = screen(panel ? map(x,y) : [x,y]);
        body += path([[ox,oy],end], `stroke="${color}" stroke-width="3"`);
        body += `<circle cx="${end[0]}" cy="${end[1]}" r="4" fill="${color}"/>`;
        body += text(end[0]+8,end[1]-8,label);
      }
      body += `<circle cx="${ox}" cy="${oy}" r="3" fill="var(--plot-ink)"/>`;
    }
    const rank = Math.abs(stretch) < 1e-10 ? 1 : 2;
    return {
      svg: svg(body, 340),
      summary: rank === 1 ? pick(lang,'The square collapses to a line: distinct inputs now share an output.', '正方形塌縮成一條線：不同輸入現在可能得到相同輸出。') : pick(lang, `The area is multiplied by ${fmt(Math.abs(stretch))}; orientation ${stretch < 0 ? 'reverses' : 'is preserved'}.`, `面積乘上 ${fmt(Math.abs(stretch))}；方向${stretch < 0 ? '翻轉' : '保持不變'}。`),
      metrics: [
        { label: pick(lang,'Determinant','行列式'), value: fmt(stretch) },
        { label: pick(lang,'Area multiplier','面積倍率'), value: fmt(Math.abs(stretch)) },
        { label: pick(lang,'Rank','秩'), value: String(rank) },
      ],
    };
  },
}];
