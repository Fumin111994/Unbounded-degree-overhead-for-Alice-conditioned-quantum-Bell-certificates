"""Plot a numerical illustration and a separately labeled exact lower bound."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / "artifacts/prl/randomness/curve.json").read_text())["records"]
x = np.array([r["chsh"] for r in rows])
y = np.array([1000*np.log2(r["os2"]/r["quantum_guess"]) for r in rows])
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "font.family": "serif", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.4, 2.15))
ax.plot(x, y, "--", color="#255f9e", linewidth=1.5, label="Conditioned level 2 (numerical)")
ax.plot([2.3], [1], "_", color="#a34021", markersize=8, markeredgewidth=1.5)
ax.annotate("", xy=(2.3, 1.31), xytext=(2.3, 1.01),
            arrowprops={"arrowstyle": "->", "color": "#a34021", "lw": 1.2})
ax.annotate("Exact lower bound", xy=(2.3, 1), xytext=(2.312, 1.52),
            color="#a34021", fontsize=7,
            arrowprops={"arrowstyle": "-", "color": "#a34021", "lw": .7})
ax.text(2.312, 1.32, "$>10^{-3}$ bits at $S=2.3$", fontsize=7, color="#a34021")
ax.axhline(0, color="#777777", linewidth=.65)
ax.set(xlim=(x[0]-.001, x[-1]+.001), ylim=(0, 3.2),
       xlabel="CHSH expectation $s$", ylabel="Entropy deficit ($10^{-3}$ bits)")
ax.text(.97, .90, "Strict loss on the full interval\n(exact interval theorem)", transform=ax.transAxes,
        ha="right", va="top", fontsize=7)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper right", bbox_to_anchor=(1, 1.1), frameon=False, fontsize=6.7)
fig.tight_layout(pad=.7)
fig.savefig(ROOT / "figures/prl_randomness_cost.pdf", bbox_inches="tight")
fig.savefig(ROOT / "artifacts/prl/randomness/figure.png", dpi=180, bbox_inches="tight")
