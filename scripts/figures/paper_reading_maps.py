"""Original cross-paper reading maps; no source figure covers these comparisons."""
from pathlib import Path
import json
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]

def draw(item):
    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=150, facecolor='white')
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    colors = ['#edf4fc', '#edf6ef', '#f4effa', '#fff3e2']
    for i, card in enumerate(item['cards']):
        x = .035 + (i % 2) * .49
        y = .46 if i < 2 else .09
        ax.add_patch(FancyBboxPatch((x, y), .44, .29, boxstyle='round,pad=.012',
                                   facecolor=colors[i], edgecolor='#8d9ead', linewidth=1))
        ax.text(x+.018, y+.225, f"{i+1}. {card['title']}", fontsize=12,
                color='#173553', fontweight='semibold', va='center')
        ax.text(x+.018, y+.145, textwrap.fill(card['question'], 34), fontsize=11,
                color='#203746', va='top', linespacing=1.45)
    fig.suptitle('\n'.join(textwrap.wrap(item['title'], 62)), fontsize=15,
                 color='#173553', fontweight='semibold', x=.055, ha='left', y=.96)
    fig.text(.055, .035, 'Complementary reading questions; no performance ranking is implied.',
             fontsize=10, color='#526574')
    dest = ROOT / item['output']; dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, facecolor='white'); plt.close(fig)
    if dest.stat().st_size >= 1_000_000:
        raise ValueError(f'Figure exceeds size limit: {dest}')

if __name__ == '__main__':
    items = json.loads((ROOT/'docs/paper-guide-reading-maps.json').read_text())
    for item in items:
        draw(item)
    print(f"Generated {len(items)} comparative reading maps.")
