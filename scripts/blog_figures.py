"""Generate blog data figures from the M3/M5/M5b results.

Values are transcribed from the progress writeups (STATUS.md tables):
- decodability: per-anchor LR LOO peak, reported as x-chance multiples.
- steering: flip-to-source rate by position (rollout CAA add).
- SAE: dense vs sparse LR LOO at end_ready (6-class balanced, chance 0.167).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt

OUT = Path("docs/blog_figures")
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1a1a1a"
C12 = "#9aa7b1"   # 12B muted
C27 = "#d1495b"   # 27B accent
plt.rcParams.update({
    "font.size": 12, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})

pos = ["Ready", "Q1", "Q2", "Q3", "Q4", "pre-reveal"]
x = range(len(pos))

# ---- Fig 1: decodability (x chance) by position, 12B vs 27B ----
# 27B chance 0.143; 12B chance 0.25. x-chance normalizes both.
xc_27 = [3.55, 4.02, 3.73, 3.50, 4.70, 5.74]
xc_12 = [1.20, 1.90, 1.60, 1.50, 3.15, None]   # 12B has no pre-reveal probe
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.axhline(1.0, ls="--", lw=1, color=INK, alpha=0.5)
ax.text(4.5, 0.74, "chance", fontsize=10, alpha=0.6)
ax.plot(x, xc_27, "-o", color=C27, lw=2.4, ms=7, label="Gemma-3-27B")
x12 = [i for i, v in zip(x, xc_12) if v is not None]
y12 = [v for v in xc_12 if v is not None]
ax.plot(x12, y12, "-o", color=C12, lw=2.4, ms=7, label="Gemma-3-12B")
ax.set_xticks(list(x)); ax.set_xticklabels(pos)
ax.set_ylabel("probe accuracy (x chance)")
ax.set_ylim(0, 6.3)
ax.set_title("How readable is the chosen animal, and when?", loc="left", fontsize=13)
ax.annotate("27B reads it already at \"Ready\"", xy=(0, 3.55), xytext=(0.55, 4.55),
            fontsize=10, color=C27,
            arrowprops=dict(arrowstyle="->", color=C27, lw=1.2))
ax.annotate("12B is at chance until\nthe game is underway", xy=(0, 1.20),
            xytext=(1.1, 0.35), fontsize=10, color="#5a6b76",
            arrowprops=dict(arrowstyle="->", color="#5a6b76", lw=1.2))
ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.55))
fig.tight_layout(); fig.savefig(OUT / "fig1_decodability.png", bbox_inches="tight")

# ---- Fig 2: steering flip-to-source by position (the causality figure) ----
flip = [1.0, 35.4, 61.5, 56.8, 70.3, 80.2]   # % flipped to steered target
taut = [False, False, False, False, False, True]
fig, ax = plt.subplots(figsize=(7, 4.2))
bars = ax.bar(x, flip, color=[("#b8b8b8" if t else C27) for t in taut],
              width=0.62)
for i, (v, t) in enumerate(zip(flip, taut)):
    ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10,
            color=(INK if not t else "#777"))
ax.set_xticks(list(x)); ax.set_xticklabels(pos)
ax.set_xlim(-0.75, 5.6)
ax.set_ylabel("reveal flipped to steered animal (%)")
ax.set_ylim(0, 92)
ax.set_title("Does the same steering change the model's mind?", loc="left",
             fontsize=13)
ax.annotate("steering at \"Ready\"\ndoes nothing", xy=(0, 2.0), xytext=(0, 34),
            ha="center", fontsize=10, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
ax.text(5, 86, "positive control\n(near readout)", ha="center", fontsize=8.5,
        color="#777")
fig.tight_layout(); fig.savefig(OUT / "fig2_steering.png", bbox_inches="tight")

print("wrote:", *(p.name for p in sorted(OUT.glob("*.png"))))
