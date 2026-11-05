"""Synthetic measurement ambiguity and observed/unobserved regularization trade-off."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[2]/'src/assets/deep-learning/dl-04e-one-step-diffusion-super-resolution'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'figure.facecolor':'white','axes.facecolor':'white','savefig.dpi':150})
# BEGIN TUTORIAL TOY
row, col = np.mgrid[:32, :32]
lr = 0.5 + 0.12*np.sin(2*np.pi*col/16) + 0.08*np.cos(2*np.pi*row/13)
base = lr.repeat(2, 0).repeat(2, 1)
i, j = np.indices(base.shape)
texture = 0.12*(-1.)**(i+j)  # zero mean in every 2x2 block
truth = base + texture
candidates = {'conditional mean': base, 'matching mode': base+texture, 'opposite mode': base-texture}
def measure(x):
    return x.reshape(32,2,32,2).mean((1,3))
for name, x in candidates.items():
    mse = np.mean((x-truth)**2)
    discrepancy = np.max(np.abs(measure(x)-lr))
    print(f'{name:16s} MSE {mse:.4f} | LR discrepancy {discrepancy:.1e}')
# END TUTORIAL TOY
fig,axes=plt.subplots(1,4,figsize=(10.5,3))
for ax,(name,x) in zip(axes,[('Observed LR',lr),*candidates.items()]):
 ax.imshow(x,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_title(name);ax.axis('off')
fig.tight_layout();fig.savefig(OUT/'measurement-ambiguity.png');plt.close(fig)

# Separate modeled coordinate for measured content and one for unmeasured texture.
gamma=np.linspace(0,3,121)
# min (a-0)^2 + lambda [(a-1)^2+(b-1)^2] vs data-preserving null-space-only change.
a=gamma/(1+gamma);b=np.ones_like(a)
fig,axes=plt.subplots(1,2,figsize=(9.4,3.5))
axes[0].plot(a,b,label='unconstrained prior pull');axes[0].scatter([0],[1],label='preserve measured coordinate',s=45)
axes[0].set(xlabel='measured coordinate error',ylabel='unmeasured texture coordinate',title='Plausibility is not a measurement constraint',xlim=(-.05,.85),ylim=(-.1,1.15));axes[0].legend(loc='lower right',fontsize=10)
axes[1].plot(gamma,a*a,label='measurement squared error');axes[1].plot(gamma,(a-1)**2,label='prior-coordinate squared error')
axes[1].set(xlabel='prior weight',ylabel='squared coordinate error',title='Illustrative quadratic objectives');axes[1].legend(fontsize=10);axes[1].grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'prior-measurement-tradeoff.png');plt.close(fig)
