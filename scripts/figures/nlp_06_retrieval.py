"""Original evidence-contract diagrams from synthetic equipment examples."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "src/assets/nlp/nlp-06-retrieval-question-answering"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})

fig, ax = plt.subplots(figsize=(10, 3.3))
ax.set(xlim=(0,10),ylim=(0,3.3))
ax.axis("off")
labels = ["User query\n+ permissions", "Authorized\ncandidate\nretrieval", "Rank + select\napplicable\nevidence", "Check entity,\nrelation, units\nand scope", "Answer\n+ version\n+ source span"]
for i, label in enumerate(labels):
    x = .15 + i*2
    ax.add_patch(FancyBboxPatch((x,.8),1.65,1.55,boxstyle="round,pad=.06",fc="#eef4ff",ec="#3875cb"))
    ax.text(x+.825,1.575,label,ha="center",va="center",linespacing=1.5)
    if i < 4:
        ax.annotate("",xy=(x+1.93,1.575),xytext=(x+1.73,1.575),arrowprops={"arrowstyle":"->","lw":1.8})
ax.set_title("A retrieved passage is a candidate, not a certified answer",pad=10)
fig.tight_layout()
fig.savefig(OUT / "pipeline.png",dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10,4.2))
ax.set(xlim=(0,10),ylim=(0,4.2))
ax.axis("off")
rows = [
    ("Pump A has temperature monitoring.", "Topic only\nNo limit value", "#e9eef8"),
    ("Pump B temperature limit is 80 C.", "Wrong entity\nDo not transfer value", "#fff0e8"),
    ("Pump A temperature limit is 80 C.", "Matching evidence\nAnswer with source", "#eaf7e9"),
]
for i, (text, verdict, color) in enumerate(rows):
    y=3.15-i*1.12
    ax.add_patch(FancyBboxPatch((.15,y-.4),9.55,.8,boxstyle="round,pad=.08",fc=color,ec="#8791a0"))
    ax.text(.4,y,text,va="center")
    ax.text(7.15,y,verdict,va="center",linespacing=1.4)
ax.set_title("Question: What is the Pump A temperature limit?",pad=10)
fig.tight_layout()
fig.savefig(OUT / "support.png",dpi=150)
plt.close(fig)
