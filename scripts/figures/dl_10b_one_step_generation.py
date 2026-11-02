"""Oracle Gaussian-mixture flow: numerical error, not a trained image generator."""
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[2]/'src/assets/deep-learning/dl-10b-one-step-generation'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'figure.facecolor':'white','axes.facecolor':'white','savefig.dpi':150})

# BEGIN TUTORIAL TOY
centers = np.array([[-1.7, 0.], [1.7, 0.], [0., -1.7], [0., 1.7]])
base_var = 0.12**2
sigma_start, sigma_end = 3.0, 0.04

def score(x, sigma):
    """Exact score of the equally weighted Gaussian mixture after Gaussian noising."""
    var = base_var + sigma**2
    delta = x[:, None, :] - centers[None, :, :]
    logits = -(delta**2).sum(-1) / (2 * var)
    weights = np.exp(logits - logits.max(1, keepdims=True))
    weights /= weights.sum(1, keepdims=True)
    return -(weights[:, :, None] * delta).sum(1) / var

def euler(initial, steps):
    x = initial.copy()
    grid = np.linspace(sigma_start, sigma_end, steps + 1)
    for s0, s1 in zip(grid[:-1], grid[1:]):
        x += (s1 - s0) * (-s0 * score(x, s0))
    return x

rng = np.random.default_rng(7)
initial = centers[rng.integers(4, size=512)] + np.sqrt(base_var + sigma_start**2) * rng.normal(size=(512, 2))
shape = initial.shape
solution = solve_ivp(lambda sigma, flat: (-sigma * score(flat.reshape(shape), sigma)).ravel(),
                     (sigma_start, sigma_end), initial.ravel(), rtol=1e-9, atol=1e-10, dense_output=True)
reference = solution.y[:, -1].reshape(shape)
for steps in [1, 2, 4, 8, 16, 32, 64]:
    estimate = euler(initial, steps)
    rmse = np.sqrt(np.mean((estimate - reference)**2))
    print(f'{steps:2d} evaluations: endpoint RMSE {rmse:.4f}')
# END TUTORIAL TOY

fig,axes=plt.subplots(1,4,figsize=(10.5,2.8),sharex=True,sharey=True)
for ax,steps in zip(axes,[1,4,16,None]):
 x=reference if steps is None else euler(initial,steps)
 ax.scatter(x[:,0],x[:,1],s=4,alpha=.45)
 ax.scatter(centers[:,0],centers[:,1],marker='+',c='black',s=60)
 ax.set(title='Reference ODE' if steps is None else f'Euler: {steps} evaluation'+('s' if steps>1 else ''),xlim=(-2.6,2.6),ylim=(-2.6,2.6),xlabel='coordinate 1',aspect='equal')
axes[0].set_ylabel('coordinate 2')
fig.tight_layout();fig.savefig(OUT/'oracle-endpoints.png');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(9.6,3.6))
grid=np.linspace(sigma_start,sigma_end,100)
paths=solution.sol(grid).T.reshape(100,*shape)
for i in np.arange(0,512,32):axes[0].plot(paths[:,i,0],paths[:,i,1],lw=1)
axes[0].scatter(centers[:,0],centers[:,1],c='black',marker='+',s=45)
axes[0].set(title='Oracle flow paths bend',xlabel='coordinate 1',ylabel='coordinate 2',xlim=(-4,4),ylim=(-4,4),aspect='equal')
counts=np.array([1,2,4,8,16,32,64]);errors=[np.sqrt(np.mean((euler(initial,int(n))-reference)**2)) for n in counts]
axes[1].loglog(counts,errors,'o-');axes[1].set(title='Same starts, numerical endpoint error',xlabel='score evaluations',ylabel='RMSE against accurate ODE')
axes[1].grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'oracle-paths-error.png');plt.close(fig)
