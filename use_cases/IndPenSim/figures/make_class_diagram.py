"""Regenerate fig_class_diagram.png (class-diagram figure of the companion SoftwareX paper).

Produces the layered class diagram of drift_detectors_pack:
- DriftDetector (ABC) -> ScoreDriftResult / StreamingDriftResult /
  PointwiseDriftResult
- univariate / multivariate / model_based families
- one leaf per detector with its public name.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT = Path(__file__).resolve().parent / "fig_class_diagram.png"

FIG_W, FIG_H = 11, 7

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis("off")


def box(x, y, w, h, text, *, bg="#f5f5f5", edge="#444", fontsize=9, bold=False):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=0.7",
                                        facecolor=bg, edgecolor=edge, linewidth=1.0))
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight)


def conn(x1, y1, x2, y2, *, c="#666"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-", color=c, lw=0.9))


# Top: public API
box(36, 60, 28, 6, "from drift_detectors import ...", bg="#fff4e0", bold=True)

# Abstraction layer
box(20, 50, 28, 6, "DriftDetector (ABC)", bg="#e8f1ff", bold=True)
box(52, 50, 28, 6, "ScoreDriftResult / StreamingDriftResult / PointwiseDriftResult",
    bg="#e8f1ff", fontsize=8)
conn(50, 60, 34, 56)
conn(50, 60, 66, 56)

# Three families
fams = [("univariate/",      6, 38, 24, 6),
        ("multivariate/",   38, 38, 24, 6),
        ("model_based/",    70, 38, 24, 6)]
for name, x, y, w, h in fams:
    box(x, y, w, h, name, bg="#e1f5e1", bold=True)
    conn(34, 50, x + w / 2, y + h)

# Univariate leaves
uni = ["PSI", "KSDetector", "Adwin", "PageHinkley", "HDDM_A", "EDDM"]
for i, n in enumerate(uni):
    box(2 + (i // 3) * 14, 24 - (i % 3) * 5, 12, 4, n, bg="#fafafa", fontsize=8)
conn(18, 38, 14, 28)

# Multivariate leaves
mv = ["MMDDetector", "PCA_CD", "KDQTree"]
for i, n in enumerate(mv):
    box(36, 24 - i * 5, 14, 4, n, bg="#fafafa", fontsize=8)
conn(50, 38, 43, 28)

# Model-based leaves
mb = ["ModelDisagreementMetric", "DisagreementMetric (ABC)"]
for i, n in enumerate(mb):
    box(70, 24 - i * 5, 24, 4, n, bg="#fafafa", fontsize=8)

# Disagreement metric leaves
dm = ["MSEDisagreement", "PearsonDisagreement", "SpearmanDisagreement"]
for i, n in enumerate(dm):
    box(70, 12 - i * 4, 24, 3.5, n, bg="#fff", fontsize=7.5)
conn(82, 19, 82, 14)
conn(82, 38, 82, 28)

# Per-detector folder annotation
ax.text(50, 4, "each leaf folder ships:  detector.py  +  metadata.yaml  +  usage.md",
        ha="center", va="center", fontsize=8.5, style="italic", color="#555")

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
