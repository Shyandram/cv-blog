"""Original symbolic illustrations of attachment and quantifier scope."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / 'src/assets/nlp/nlp-00-language-structure-meaning'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=150)
for ax, parent, title in zip(axes, ['inspected', 'panel'], ['Camera is the instrument', 'Camera describes the panel']):
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    ax.set_title(title, fontsize=12)
    positions = {'inspected': (.40,.78), 'technician': (.16,.43), 'panel': (.65,.43), 'with the camera': (.65,.10)}
    for word, (x,y) in positions.items():
        ax.text(x, y, word, ha='center', va='center', bbox={'boxstyle':'round,pad=.4', 'fc':'#dbeafe' if word!='with the camera' else '#dcfce7', 'ec':'#64748b'})
    for source, target in [('inspected','technician'), ('inspected','panel'), (parent,'with the camera')]:
        x,y = positions[source]; xx,yy = positions[target]
        ax.annotate('', xy=(xx,yy+.06), xytext=(x,y-.06), arrowprops={'arrowstyle':'->', 'lw':1.7, 'color':'#475569'})
    ax.text(.50,.96,'Simplified dependency relations',ha='center',fontsize=10,color='#475569')
fig.subplots_adjust(top=.82,bottom=.06,left=.04,right=.98,wspace=.10)
fig.savefig(OUT/'attachment.png',dpi=150,facecolor='white'); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=150)
worlds = [np.array([[1,0],[0,1],[1,0]]), np.array([[1,0],[1,0],[1,0]])]
for ax, matrix, title in zip(axes, worlds, ['Different cameras can satisfy each person', 'One camera satisfies every person']):
    ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    ax.set_xticks([0,1],['Camera A','Camera B']); ax.set_yticks([0,1,2],['Ada','Bo','Cy'])
    ax.set_title(title,fontsize=11,pad=10)
    for i in range(3):
        for j in range(2):
            ax.text(j,i,'uses' if matrix[i,j] else '—',ha='center',va='center',color='white' if matrix[i,j] else '#334155')
    ax.text(.5,-.30,f'Each has some: {bool(matrix.any(axis=1).all())}\nSome shared by all: {bool(matrix.all(axis=0).any())}',transform=ax.transAxes,ha='center',va='top',fontsize=11)
fig.subplots_adjust(top=.83,bottom=.33,left=.09,right=.98,wspace=.40)
fig.savefig(OUT/'quantifier-scope.png',dpi=150,facecolor='white'); plt.close(fig)
