"""Original teaching maps for the agent blog series; no benchmark data."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "src/assets/blog/agent-engineering"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.dpi": 150})


def panel(title):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.axis("off")
    ax.set_title(title, fontsize=16, loc="left", pad=16)
    return fig, ax


def box(ax, x, y, width, height, title, detail, accent=False):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1",
                              facecolor="#eaf0ed" if accent else "#f6f6f3",
                              edgecolor="#82918a", linewidth=1))
    ax.text(x + width / 2, y + height * .68, title, ha="center", va="center",
            fontsize=12, weight="bold", color="#24352d")
    ax.text(x + width / 2, y + height * .29, detail, ha="center", va="center", fontsize=10)


def arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": "#465b50", "lw": 1.5})


fig, ax = panel("An agent needs more than a model")
box(ax, .6, 7.8, 8.8, 1.4, "Goal and acceptance checks", "What should change? How will we know?")
box(ax, .6, 5.5, 3.9, 1.4, "Context + skills", "Evidence, instructions, procedures")
box(ax, 5.5, 5.5, 3.9, 1.4, "Model", "Propose the next action")
box(ax, .6, 3.1, 8.8, 1.4, "Harness", "Validate, authorize, execute, observe, checkpoint", True)
box(ax, .6, .7, 3.9, 1.4, "Tools + environment", "Files, APIs, browser, sandbox")
box(ax, 5.5, .7, 3.9, 1.4, "Evidence + artifacts", "Tests, reports, durable state")
arrow(ax, (5, 7.7), (5, 7.1)); arrow(ax, (4.6, 6.2), (5.4, 6.2))
arrow(ax, (7.5, 5.4), (7.5, 4.6)); arrow(ax, (2.5, 3), (2.5, 2.2))
arrow(ax, (4.6, 1.4), (5.4, 1.4)); arrow(ax, (7.5, 2.2), (7.5, 3))
ax.text(5, .1, "A teaching map: components overlap in real implementations.", ha="center", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "agent-system.png"); plt.close(fig)

fig, ax = panel("A recoverable work cycle")
for y, title, detail in [(7.8, "1  Load state", "Goal, verified progress, unresolved issues"),
                         (5.5, "2  Propose and gate", "Schema, authority, remaining budget"),
                         (3.2, "3  Execute and check", "Observe the artifact, not only the final message"),
                         (.9, "4  Save a checkpoint", "Evidence + next action + replay identity")]:
    box(ax, 1.3, y, 7.4, 1.4, title, detail, y == 3.2)
for y in [7.8, 5.5, 3.2]: arrow(ax, (5, y - .1), (5, y - .8))
ax.plot([8.85, 9.5, 9.5, 8.85], [1.6, 1.6, 8.5, 8.5], color="#465b50", lw=1.2)
arrow(ax, (9.1, 8.5), (8.8, 8.5))
ax.text(.9, 4.8, "Stop / escalate on denied, exhausted, or stuck work", rotation=90,
        ha="center", va="center", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "recoverable-loop.png"); plt.close(fig)

fig, ax = panel("OpenClaw: follow one request")
box(ax, .6, 7.8, 8.8, 1.4, "Chat channel or Control UI", "An authenticated request enters the system")
box(ax, .6, 5.5, 8.8, 1.4, "Gateway", "Route the request to an agent and session", True)
box(ax, .6, 3.1, 3.9, 1.4, "Agent runtime", "Model + eligible skills + tools")
box(ax, 5.5, 3.1, 3.9, 1.4, "Workspace + session", "Files, history, memory, artifacts")
box(ax, .6, .7, 3.9, 1.4, "Execution boundary", "Host, sandbox, or paired node")
box(ax, 5.5, .7, 3.9, 1.4, "Result and delivery", "Inspect first; send when authorized")
arrow(ax, (5, 7.7), (5, 7)); arrow(ax, (2.5, 5.4), (2.5, 4.6))
arrow(ax, (4.6, 3.8), (5.4, 3.8)); arrow(ax, (2.5, 3), (2.5, 2.2))
arrow(ax, (4.6, 1.4), (5.4, 1.4))
ax.text(5, .1, "Simplified teaching map; not an official protocol diagram.", ha="center", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "openclaw-request.png"); plt.close(fig)
