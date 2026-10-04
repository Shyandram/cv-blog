"""Original synthetic figures for ML07 additions; no downloads.
Writes only its own ML07 chapter folder.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parents[2] / 'src/assets/ml/ml-07-diffusion-flow-matching'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
y, x = np.mgrid[:64, :64]
image = .6*x/63 + .4*((x+y) % 2)
f = 4
latent = image.reshape(64//f, f, 64//f, f).mean(axis=(1, 3))
recon = np.repeat(np.repeat(latent, f, axis=0), f, axis=1)
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), layout='constrained')
for ax, values, title in zip(axes, [image, recon, np.abs(image-recon)],
                            ['Synthetic input', 'Block-average reconstruction', 'Absolute error']):
    im = ax.imshow(values, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes, shrink=.75, label='Intensity / absolute error')
fig.savefig(OUT/'latent-bottleneck.png', dpi=150); plt.close(fig)
abar = np.logspace(-4, np.log10(.999), 250)
t = np.linspace(.01, .99, 250)
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), layout='constrained')
axes[0].loglog(abar, (1-abar)/abar, lw=2, color='#337d9f')
axes[0].set(xlabel='VP signal variance fraction', ylabel='Clean MSE / noise MSE',
            title='Noise error converted to clean error')
axes[1].semilogy(t, 1/(1-t)**2, lw=2, color='#8553a8')
axes[1].set(xlabel='Flow time (noise 0 to clean 1)', ylabel='Velocity MSE / clean MSE',
            title='Clean error converted to flow error')
fig.savefig(OUT/'prediction-targets.png', dpi=150); plt.close(fig)
print('Generated original ML07 representation figures; toy MSE:', np.mean((image-recon)**2))
