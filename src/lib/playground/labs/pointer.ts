/**
 * Click-or-drag placement of a probe on one or more canvases, for labs with x and y sliders. A tap or click sets the position;
 * a mouse or pen can also drag. Touch keeps its normal page scrolling. The sliders stay the source of truth (setControl writes them).
 */
export function attachProbe(
  canvases: Record<string, HTMLCanvasElement>,
  panes: string[],
  setControl: (key: string, value: string | number) => void,
  keys: { x: string; y: string } = { x: 'x', y: 'y' },
) {
  for (const pane of panes) {
    const cv = canvases[pane];
    if (!cv) continue;
    cv.style.cursor = 'crosshair';
    const move = (e: PointerEvent) => {
      const r = cv.getBoundingClientRect();
      const x = Math.floor(((e.clientX - r.left) / r.width) * cv.width), y = Math.floor(((e.clientY - r.top) / r.height) * cv.height);
      setControl(keys.x, Math.max(0, Math.min(cv.width - 1, x)));
      setControl(keys.y, Math.max(0, Math.min(cv.height - 1, y)));
    };
    cv.addEventListener('pointerdown', e => { move(e); if (e.pointerType !== 'touch') cv.setPointerCapture(e.pointerId); });
    cv.addEventListener('pointermove', e => { if (e.pointerType !== 'touch' && (e.buttons & 1)) move(e); });
  }
}
