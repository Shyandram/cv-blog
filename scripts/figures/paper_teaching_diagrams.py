"""Purpose-specific original diagrams for paper guides whose source figure is insufficient."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from requested_paper_guides import draw

ROOT = Path(__file__).resolve().parents[2]

def morphology(item):
    foreground = {2, 3, 4, 6}
    footprint = {-1, 0, 1}
    universe = set(range(-3, 12))
    def dilate(values):
        return {a+b for a in values for b in footprint}
    def erode(values):
        return {x for x in universe if {x+b for b in footprint} <= values}
    opened = dilate(erode(foreground)); closed = erode(dilate(foreground))
    assert erode(foreground) == {3}
    assert opened == {2, 3, 4} and closed == {2, 3, 4, 5, 6}
    assert dilate(erode(opened)) == opened and erode(dilate(closed)) == closed
    rows = [('Input A', foreground), ('Erosion', erode(foreground)),
            ('Dilation', dilate(foreground)), ('Opening', opened), ('Closing', closed)]
    fig, ax = plt.subplots(figsize=(9.6, 4.8), dpi=150, facecolor='white')
    ax.set(xlim=(-3, 9), ylim=(-.7, 5.4)); ax.axis('off')
    for row, (label, values) in enumerate(rows):
        y = 4-row
        ax.text(-1.1, y, label, ha='right', va='center', fontsize=11, color='#203746')
        for x in range(9):
            ax.add_patch(Rectangle((x-.35, y-.28), .7, .56,
                                  facecolor='#3775a8' if x in values else '#f1f4f7',
                                  edgecolor='#94a7b7', linewidth=.8))
            if row == 0:
                ax.text(x, y+.4, str(x), ha='center', fontsize=10, color='#526574')
    fig.suptitle('Morphology: what a three-point footprint keeps or fills', x=.05,
                 ha='left', fontsize=15, color='#173553', fontweight='semibold')
    fig.text(.05, .035, 'B = {-1, 0, 1}. Active cells are blue; the operations use an unbounded integer line.',
             fontsize=10, color='#526574')
    dest = ROOT/item['output']; dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, facecolor='white'); plt.close(fig)

if __name__ == '__main__':
    items = json.loads((ROOT/'docs/paper-guide-teaching-diagrams.json').read_text())
    for item in items:
        if item['kind'] == 'teaching-flow':
            draw(item['key'], (item['slug'], item['title'], item['note'], item['nodes'], item['edges']))
        elif item['kind'] == 'worked-set-example':
            morphology(item)
        else:
            continue
        path = ROOT/item['output']
        if path.stat().st_size >= 1_000_000:
            raise ValueError(f'Figure exceeds size limit: {path}')
    print('Generated six purpose-specific teaching diagrams.')
