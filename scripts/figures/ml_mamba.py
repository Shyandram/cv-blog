"""Original selective-state and image-scan illustrations; no external data."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / 'src/assets/ml'
plt.rcParams.update({'font.size': 11, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})

def save(fig, slug, name):
    folder = ROOT / slug
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def scan_image(image, four=False):
    ids = np.arange(image.size).reshape(image.shape)
    row, col = ids.ravel(), ids.T.ravel()
    orders = [row, row[::-1]]
    if four:
        orders += [col, col[::-1]]
    outputs = []
    for order in orders:
        h = 0.0
        out = np.zeros(image.size)
        for i in order:
            h = .8 * h + .2 * image.ravel()[i]
            out[i] = h
        outputs.append(out.reshape(image.shape))
    return np.mean(outputs, axis=0)

def main():
    steps = np.arange(31)
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)
    for delta in [.05, .5, 2.]:
        axs[0].plot(steps, np.exp(-.4 * delta * steps), label=f'constant delta = {delta}')
    axs[0].set(xlabel='Steps since stored impulse', ylabel='Retained fraction', title='Step size changes memory')
    axs[0].legend(fontsize=10)
    constant = np.full(30, .05)
    reset = constant.copy()
    reset[14] = 4.
    for ds, label in [(constant, 'small steps throughout'), (reset, 'large step at position 15')]:
        retained = np.r_[1., np.exp(-.4 * np.cumsum(ds))]
        axs[1].plot(steps, retained, label=label)
    axs[1].axvline(15, color='gray', linestyle=':', linewidth=1)
    axs[1].set(xlabel='Steps since stored impulse', ylabel='Retained fraction', title='A selective step reduces history')
    axs[1].legend(fontsize=10)
    for ax in axs:
        ax.grid(alpha=.2)
        ax.set_ylim(0, 1.04)
    save(fig, 'ml-03a-recurrent-recursive-networks', 'selective-state-memory.png')

    image = np.zeros((6, 6))
    image[1, 4] = 1.
    two, four = scan_image(image), scan_image(image, four=True)
    two_rotated = np.rot90(scan_image(np.rot90(image)), -1)
    four_rotated = np.rot90(scan_image(np.rot90(image), four=True), -1)
    assert np.max(np.abs(two-two_rotated)) > .01
    assert 0 < np.max(np.abs(four-four_rotated)) < np.max(np.abs(two-two_rotated))
    fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.2), constrained_layout=True)
    for ax, data, title in zip(axs, [image, two, four], ['Input impulse', 'Two raster routes', 'Four image routes']):
        im = ax.imshow(data, cmap='viridis', vmin=0, vmax=1 if title == 'Input impulse' else .2)
        ax.set(title=title, xlabel='Column', ylabel='Row')
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.plot(4, 1, 'rs', fillstyle='none', markersize=13, markeredgewidth=1.5)
        fig.colorbar(im, ax=ax, shrink=.7)
    save(fig, 'ml-04-vision-transformers', 'scan-order-influence.png')

if __name__ == '__main__':
    main()
