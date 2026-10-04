"""Original schematic of BERT input packing and alternative task heads."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / 'src/assets/nlp/nlp-05-language-models'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.facecolor': 'white',
                     'savefig.facecolor': 'white'})
fig, ax = plt.subplots(figsize=(10.2, 5.4))
ax.set(xlim=(-0.6, 7.6), ylim=(-0.2, 5.5))
ax.axis('off')

def box(x, y, width, height, text, color):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                 boxstyle='round,pad=0.05', facecolor=color,
                 edgecolor='#597082', linewidth=1))
    ax.text(x + width/2, y + height/2, text, ha='center', va='center')

def arrow(start, end):
    ax.annotate('', xy=end, xytext=start,
                arrowprops={'arrowstyle': '->', 'color': '#597082', 'lw': 1.4})

ax.text(3.5, 5.25, 'One packed text pair; three alternative output interfaces',
        ha='center', va='center', fontsize=13)
tokens = ['[CLS]', 'pump', 'failed', '[SEP]', 'pressure', 'dropped', '[SEP]']
for i, token in enumerate(tokens):
    segment = 0 if i <= 3 else 1
    box(i, 4.35, .9, .5, token, '#e5eef6' if segment == 0 else '#e9e2f3')
    ax.text(i+.45, 4.02, f'pos {i}\nsegment {segment}', ha='center', va='center', fontsize=10)
    arrow((i+.45, 3.66), (i+.45, 3.46))
box(0, 2.85, 6.9, .58, 'Sum token + position + segment embeddings', '#eef3f5')
arrow((3.45, 2.8), (3.45, 2.53))
box(0, 1.82, 6.9, .65, 'Bidirectional Transformer encoder\nEach real token can read both text parts', '#d9e9e8')
centers = [.75, 3.45, 6.15]
for x in centers:
    arrow((x, 1.77), (x, 1.19))
box(-.35, .32, 2.2, .8, '[CLS] state + pooler\nSequence classifier', '#e5eef6')
box(2.35, .32, 2.2, .8, 'Each token state\nToken-label classifier', '#e9e2f3')
box(5.05, .32, 2.2, .8, 'Passage token states\nStart / end scores', '#f6e8de')
fig.tight_layout()
fig.savefig(OUT / 'bert-task-heads.png', dpi=150)
plt.close(fig)
