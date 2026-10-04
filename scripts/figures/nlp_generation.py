"""Original synthetic figures for NLP08; no downloaded data or weights.
Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/nlp_generation.py
Writes only src/assets/nlp/nlp-08-generation-deployment/.
"""
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[2]/'src/assets/nlp/nlp-08-generation-deployment'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'figure.facecolor':'white','axes.facecolor':'white'})
sentences=["Pump A temperature reached 80 C during the test.",
           "The test recorded Pump A temperature at 80 C.",
           "Pump A vibration triggered a shutdown warning.",
           "The technician replaced a label on the cabinet."]
query='pump temperature vibration shutdown'
stop={'a','the','at','on','during'}
def tokens(text):
    return set(re.findall(r'[a-z0-9]+',text.lower()))-stop
vocab=sorted(set().union(*(tokens(s) for s in sentences),tokens(query)))
def vector(text):
    v=np.array([w in tokens(text) for w in vocab],dtype=float)
    return v/max(np.linalg.norm(v),1e-12)
V=np.stack([vector(s) for s in sentences]);q=vector(query)
sim=V@V.T;relevance=V@q
selected=[]
for _ in range(2):
    red=sim[:,selected].max(axis=1) if selected else np.zeros(4)
    score=.55*relevance-.45*red;score[selected]=-np.inf
    selected.append(int(np.argmax(score)))
fig,axs=plt.subplots(1,2,figsize=(10.2,4.1),layout='constrained')
im=axs[0].imshow(sim,vmin=0,vmax=1,cmap='Blues')
axs[0].set(xticks=range(4),yticks=range(4),xlabel='Sentence index',ylabel='Sentence index',title='Candidate similarity')
for i in range(4):
    for j in range(4):
        axs[0].text(j,i,f'{sim[i,j]:.2f}',ha='center',va='center',color='white' if sim[i,j]>.6 else 'black')
fig.colorbar(im,ax=axs[0],label='Cosine similarity',shrink=.7)
axs[1].bar(range(4),relevance,color=['#b65332' if i in selected else '#769db7' for i in range(4)])
axs[1].set(xticks=range(4),xlabel='Sentence index',ylabel='Query relevance',title=f'MMR selected: {selected}',ylim=(0,.8))
fig.savefig(OUT/'selection.png',dpi=150);plt.close(fig)
lengths=np.array([8,64,12,60,10,56,14,52])
fig,axs=plt.subplots(1,2,figsize=(10.2,4.1),layout='constrained')
for offset,values,label in ((0,lengths,'Arrival'),(10,np.sort(lengths),'Bucketed')):
    padded=np.repeat(values.reshape(2,4).max(axis=1),4)
    pos=np.arange(8)+offset
    axs[0].barh(pos,padded,color='#e6ebf1',label='Padding' if offset==0 else None)
    axs[0].barh(pos,values,color='#337d9f',label='Real tokens' if offset==0 else None)
axs[0].set(yticks=[3.5,13.5],yticklabels=['Arrival','Bucketed'],xlabel='Token slots per request',title='Same tokens; different padding')
axs[0].legend(loc='lower right',fontsize=10)
T=np.arange(1,257);memory=2*4*2*16*T*4/1024
axs[1].plot(T,memory,color='#8553a8',lw=2)
axs[1].scatter([128],[128],color='#b65332',zorder=3)
axs[1].annotate('128 tokens: 128 KiB',(128,128),xytext=(22,215),arrowprops={'arrowstyle':'->'})
axs[1].set(xlabel='Cached tokens',ylabel='Toy cache memory (KiB)',title='Cache grows with context')
fig.savefig(OUT/'batching.png',dpi=150);plt.close(fig)
print('Generated NLP08 figures; selected:',selected)
