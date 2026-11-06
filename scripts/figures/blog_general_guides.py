"""Original teaching figures for the general blog guides (agents, models, evaluation).

Writes only to src/assets/blog/general-guides/. Figures 1 and 2 are conceptual diagrams with no
measured data. Figure 3 is computed from the closed-form pass@k / pass^k expressions for
independent trials; it is not benchmark data.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "src/assets/blog/general-guides"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.dpi": 150})
INK, LINE, SOFT, ACCENT = "#24352d", "#82918a", "#f6f6f3", "#eaf0ed"


def box(ax, x, y, w, h, title, detail="", accent=False, size=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                facecolor=ACCENT if accent else SOFT, edgecolor=LINE, linewidth=1))
    if detail:
        ax.text(x + w / 2, y + h * .66, title, ha="center", va="center", fontsize=size, weight="bold", color=INK)
        ax.text(x + w / 2, y + h * .28, detail, ha="center", va="center", fontsize=10, color=INK)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=size, weight="bold", color=INK)


def arrow(ax, a, b, text=None, offset=(0, .18)):
    ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": "#465b50", "lw": 1.5})
    if text:
        ax.text((a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1], text,
                ha="center", va="bottom", fontsize=10, color="#465b50")


# Figure 1: one tool-calling round trip and what it adds to the next request.
fig, ax = plt.subplots(figsize=(10, 6.4))
ax.set(xlim=(0, 10), ylim=(0, 10)); ax.axis("off")
ax.set_title("One tool call is two model requests", fontsize=16, loc="left", pad=14)
box(ax, .3, 7.4, 2.9, 1.7, "Application", "sends tool schemas\n+ conversation")
box(ax, 3.9, 7.4, 2.3, 1.7, "Model", "returns a call:\nname, args, id")
box(ax, 6.9, 7.4, 2.8, 1.7, "Harness", "validate, authorize,\nexecute", True)
box(ax, 6.9, 4.6, 2.8, 1.7, "Tool", "file, API, search,\nshell, browser")
box(ax, 3.9, 4.6, 2.3, 1.7, "Model", "reads result,\nanswers or calls again")
arrow(ax, (3.25, 8.25), (3.85, 8.25)); arrow(ax, (6.25, 8.25), (6.85, 8.25))
arrow(ax, (8.3, 7.35), (8.3, 6.35))
ax.text(8.45, 6.85, "run if allowed", fontsize=10, va="center", color="#465b50")
arrow(ax, (6.85, 5.45), (6.25, 5.45))
ax.text(6.55, 4.25, "result returned with the call's id", ha="center", fontsize=10, color="#465b50")
ax.text(.3, 5.95, "The model never runs the tool.\nCode decides whether the\nproposed call may execute.", fontsize=10.5,
        va="center", color=INK)
# context stack for the second request
ax.text(.3, 3.55, "What the second request carries (all of it counts toward the context window):",
        fontsize=11, weight="bold", color=INK)
parts = [("system\ninstructions", 1.3), ("tool\ndefinitions", 1.5), ("conversation\nso far", 2.0),
         ("model's\ntool call", 1.3), ("tool\nresult", 2.0), ("room for\noutput", 1.3)]
x = .3
for i, (label, w) in enumerate(parts):
    ax.add_patch(FancyBboxPatch((x, 1.4), w - .08, 1.6, boxstyle="square,pad=0",
                                facecolor=ACCENT if label.startswith("tool\nresult") else SOFT,
                                edgecolor=LINE, linestyle="--" if "output" in label else "-", linewidth=1))
    ax.text(x + (w - .08) / 2, 2.2, label, ha="center", va="center", fontsize=10, color=INK)
    x += w
ax.text(.3, .75, "Widths are schematic, not measured. A tool result can be large unless the tool bounds it.",
        fontsize=10, color="#465b50")
fig.tight_layout(); fig.savefig(OUT / "tool-call-context.png"); plt.close(fig)


# Figure 2: the layers you license (or rely on) separately.
fig, ax = plt.subplots(figsize=(10, 6.2))
ax.set(xlim=(0, 10), ylim=(0, 10)); ax.axis("off")
ax.set_title("\"Open\" is a property of each layer, not of a model name", fontsize=15, loc="left", pad=14)
rows = [("Hosted service", "Terms of service, data retention, rate limits, regions",
         "Who receives my inputs, and may they keep or train on them?"),
        ("Model weights", "Weight license: permissive, custom/community, or non-commercial",
         "May I download, run, fine-tune, and ship the weights for my use?"),
        ("Code", "Inference/training code and client tools: OSI-approved or other licenses",
         "May I read, modify, and redistribute the software?"),
        ("Training data", "Data information, data cards, provenance notes (often partial)",
         "Can I learn what the model was trained on and rebuild it?")]
for i, (name, what, question) in enumerate(rows):
    y = 7.3 - i * 2.15
    box(ax, .3, y, 2.4, 1.6, name, accent=(i == 1), size=12)
    ax.text(2.95, y + 1.1, what, fontsize=10.5, va="center", color=INK)
    ax.text(2.95, y + .45, "Ask: " + question, fontsize=10.5, va="center", color="#465b50", style="italic")
ax.text(.3, .25, "Each layer can carry different terms. Example: an open-source client calling a closed hosted model.",
        fontsize=10, color="#465b50")
fig.tight_layout(); fig.savefig(OUT / "licensing-layers.png"); plt.close(fig)


# Figure 3: "at least one success in k" versus "success on all k", independent trials.
ks = list(range(1, 11))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for p, style in [(0.5, ":"), (0.7, "--"), (0.9, "-")]:
    axes[0].plot(ks, [1 - (1 - p) ** k for k in ks], style, color=INK, marker="o", ms=4, label=f"p = {p}")
    axes[1].plot(ks, [p ** k for k in ks], style, color=INK, marker="o", ms=4, label=f"p = {p}")
axes[0].set_title("pass@k: at least one of k attempts passes", fontsize=12)
axes[1].set_title("pass^k: all k attempts pass", fontsize=12)
for a in axes:
    a.set_xlabel("number of attempts k"); a.set_xticks(ks); a.grid(alpha=.3); a.set_ylim(0, 1.02)
axes[0].set_ylabel("probability"); axes[0].legend(loc="lower right", frameon=False)
fig.suptitle("Same per-attempt success rate p, opposite trends (independent attempts assumed)", fontsize=12, x=.02, ha="left")
fig.tight_layout(); fig.savefig(OUT / "pass-at-k.png"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
