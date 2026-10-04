"""Generate an original diagram of the two Word2Vec prediction directions."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / 'src/assets/nlp/nlp-02-classification-embeddings'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), dpi=150)

def box(ax, x, y, text, color, width=0.27, height=0.16):
    ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
                              boxstyle='round,pad=0.015', facecolor=color,
                              edgecolor='#334155', linewidth=1))
    ax.text(x, y, text, ha='center', va='center')

def arrow(ax, start, end):
    ax.annotate('', xy=end, xytext=start,
                arrowprops={'arrowstyle': '->', 'color': '#475569', 'lw': 1.5})

for ax in axes:
    ax.set(xlim=(0,1), ylim=(-.11,1))
    ax.axis('off')
ax = axes[0]
ax.set_title('CBOW: context → center', fontsize=13, pad=12)
box(ax, .20,.78,'red\ninput row', '#dbeafe')
box(ax, .80,.78,'move\ninput row', '#dbeafe')
box(ax, .50,.46,'mean vector', '#e0e7ff', width=.31)
box(ax, .50,.14,'robots\ntarget', '#dcfce7')
arrow(ax, (.20,.68),(.44,.56)); arrow(ax, (.80,.68),(.56,.56))
arrow(ax, (.50,.36),(.50,.24))
ax.text(.76,.30,'score with\noutput table', ha='center',va='center',color='#334155')
ax = axes[1]
ax.set_title('Skip-gram: center → context', fontsize=13, pad=12)
box(ax,.50,.78,'robots\ninput row','#dbeafe')
box(ax,.50,.46,'center vector','#e0e7ff',width=.31)
box(ax,.20,.14,'red\ntarget','#dcfce7'); box(ax,.80,.14,'move\ntarget','#dcfce7')
arrow(ax,(.50,.68),(.50,.56))
arrow(ax,(.44,.36),(.20,.24)); arrow(ax,(.56,.36),(.80,.24))
ax.text(.50,-.04,'Separate losses; shared output table',ha='center',va='center',fontsize=10)
fig.suptitle('Window: small red [robots] move slowly · radius 1',fontsize=13,y=.97)
fig.subplots_adjust(top=.81,bottom=.08,left=.04,right=.96,wspace=.12)
fig.savefig(OUT / 'word2vec-objectives.png',dpi=150,facecolor='white')
plt.close(fig)
