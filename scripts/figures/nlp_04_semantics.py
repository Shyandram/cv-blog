"""Original synthetic semantics diagrams; rerun from the project root."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "src/assets/nlp/nlp-04-semantics-extraction"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.set(xlim=(0, 10), ylim=(0, 4.2))
ax.axis("off")
for y, text in [(3.2, "Mina repaired the sensor."), (1.1, "The sensor was repaired by Mina.")]:
    ax.add_patch(FancyBboxPatch((.2, y-.35), 4.3, .8, boxstyle="round,pad=.1", fc="#eef4ff", ec="#3875cb"))
    ax.text(2.35, y+.04, text, ha="center", va="center")
    ax.annotate("", xy=(6, 2.1), xytext=(4.65, y), arrowprops={"arrowstyle": "->", "lw": 2})
ax.add_patch(FancyBboxPatch((6, 1.25), 3.4, 1.75, boxstyle="round,pad=.15", fc="#f0f9ee", ec="#4b8741"))
ax.text(7.7, 2.12, "Event: repair\nRepairer: Mina\nRepaired object: sensor", ha="center", va="center", linespacing=1.8)
ax.set_title("Grammatical subject changes; event roles can stay the same", pad=15)
fig.tight_layout()
fig.savefig(OUT / "roles.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3.7))
words = ["Mina", "works", "at", "Orion", "Labs"]
for x, word in enumerate(words):
    ax.text(x, 2.5, word, ha="center", fontsize=13)
for y, spans in [(1.7, [(-.35,.35,"PER","#3875cb"),(2.65,4.35,"ORG","#3875cb")]),
                 (.7, [(-.35,.35,"ORG","#cf5543"),(2.65,3.35,"ORG","#cf5543")])]:
    for left, right, label, color in spans:
        ax.plot([left,right],[y,y],lw=7,color=color,solid_capstyle="butt")
        ax.text((left+right)/2,y+.2,label,ha="center",color=color)
ax.text(-1.35,1.7,"Gold",va="center")
ax.text(-1.35,.7,"Predicted",va="center")
ax.text(2,0,"Wrong type + truncated boundary = no exact entity matches",ha="center")
ax.set(xlim=(-1.55,4.65),ylim=(-.25,3))
ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "spans.png",dpi=150)
plt.close(fig)
