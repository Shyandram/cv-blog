"""Supplementary DPS computation diagram; the original probability graph is shown separately."""
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
# Nodes: identifier, original explanatory label, column, row, role.
# Each graph separates observations, learned operations, and constraints.
SPECS = {'dps': ('dl-04c-generative-restoration',
         'DPS: guide a prior through a denoised estimate',
         'Likelihood gradients pass through the denoiser; the forward operator is assumed known',
         [('xt', 'Noisy state x_t', 0, 0, 'data'),
          ('den', 'Score and clean-image estimate', 1.5, 0, 'learned'),
          ('pr', 'Reverse-diffusion proposal', 3, 0, 'op'),
          ('y', 'Measurement y and known operator', 0, 2, 'data'),
          ('r', 'Forward prediction and residual', 2.5, 2, 'op'),
          ('g', 'Backpropagate likelihood correction', 4, 2, 'constraint'),
          ('o', 'Corrected next noisy state', 5, 0, 'data')],
         [('xt', 'den'),
          ('den', 'pr'),
          ('den', 'r'),
          ('y', 'r'),
          ('r', 'g'),
          ('g', 'o'),
          ('pr', 'o')])}


def draw(key, spec):
    slug, title, note, nodes, edges = spec
    edges = [tuple(edge) for edge in edges]
    fig, ax = plt.subplots(figsize=(10.4, 5.3), dpi=150, facecolor="white")
    ax.set(xlim=(-0.65, max(5.7, max(n[2] for n in nodes)+.65)), ylim=(-0.7, 2.9))
    ax.axis("off")
    colors = {"data": ("#edf4fc", "#45719b"), "learned": ("#e8f3ed", "#367654"),
              "op": ("#f1edf8", "#7a5c9e"), "constraint": ("#fff1df", "#ad7438")}
    positions = {name: (col, 2-row) for name, _, col, row, _ in nodes}
    patches = {}
    for name, label, col, row, role in nodes:
        x, y = positions[name]; fill, edge = colors[role]
        box = FancyBboxPatch((x-.41, y-.32), .82, .64, boxstyle="round,pad=0.025,rounding_size=0.05",
                             facecolor=fill, edgecolor=edge, linewidth=1.25, zorder=2)
        ax.add_patch(box); patches[name] = box
        ax.text(x, y, textwrap.fill(label, 17), ha="center", va="center",
                fontsize=10, color="#172a3a", zorder=3)
    for source, target in dict.fromkeys(edges):
        x1, y1 = positions[source]; x2, y2 = positions[target]
        radius = 0.15 if abs(y1-y2) > 0.5 and abs(x1-x2) > 2 else 0
        if abs(x1-x2) > 2 and abs(y1-y2) < .1: radius = -.32
        if (target, source) in edges: radius = 0.2
        arrow = FancyArrowPatch((x1, y1), (x2, y2), patchA=patches[source], patchB=patches[target],
                                arrowstyle="-|>", mutation_scale=12, shrinkA=3, shrinkB=4,
                                color="#6c7886", linewidth=1.15,
                                connectionstyle=f"arc3,rad={radius}", zorder=1)
        ax.add_patch(arrow)
    fig.suptitle(title, fontsize=15, fontweight="semibold", x=.04, ha="left", y=.975, color="#173553")
    fig.text(.04, .075, textwrap.fill(note, 115), fontsize=10, color="#394a5b")
    fig.text(.04, .025, "Blue: observations / states     Green: learned modules     Purple: operations     Orange: constraints / control",
             fontsize=10, color="#526170")
    fig.subplots_adjust(left=.025, right=.99, top=.88, bottom=.17)
    folder = ROOT / "src/assets/deep-learning" / slug
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"paper-{key}.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    # DPS Figure 2 is a probabilistic graph; this extra schematic explains the computation.
    # Other requested methods use their attributed original architecture figures.
    draw("dps", SPECS["dps"])
