"""Regenerate fig_class_diagram.png (class-diagram figure of the companion SoftwareX paper).

Shows the layered architecture of drift_detectors_pack: the abstract
DriftDetector, the result dataclasses, and the three detector families
(univariate, multivariate, model-based) with every detector grouped by
sub-family. Each leaf folder ships detector.py + metadata.yaml + usage.md.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "fig_class_diagram.png"

fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(14, 100)
ax.axis("off")


def box(x, y, w, h, text, *, bg="#f5f5f5", edge="#444", fs=9, bold=False, va="center", pad=0.5):
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=0.7",
        facecolor=bg, edgecolor=edge, linewidth=1.0))
    ty = y + h / 2 if va == "center" else y + h - 1.5
    ax.text(x + w / 2 if va == "center" else x + 1.5, ty, text,
            ha="center" if va == "center" else "left", va=va,
            fontsize=fs, fontweight="bold" if bold else "normal", family="DejaVu Sans")


def conn(x1, y1, x2, y2, c="#777"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-", color=c, lw=0.9))


def card(x, y, w, h, title, sections, *, bg, title_bg):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.0",
                                        facecolor=bg, edgecolor="#3a6b3a", linewidth=1.3))
    box(x + 1.5, y + h - 6.5, w - 3, 5, title, bg=title_bg, edge="#3a6b3a", fs=10, bold=True)
    cy = y + h - 9
    for sub, items in sections:
        ax.text(x + 2.5, cy, sub, fontsize=8.2, style="italic", color="#333", va="top", fontweight="bold")
        cy -= 2.6
        ax.text(x + 3.5, cy, "\n".join(items), fontsize=8.0, va="top", color="#111", linespacing=1.35)
        cy -= 1.9 * len(items) + 1.6


# ---- top: public API + abstraction layer
box(36, 92, 28, 5.5, "from drift_detectors import ...", bg="#fff4e0", bold=True)
box(14, 82, 30, 5.5, "DriftDetector  (abstract base class)", bg="#e8f1ff", bold=True)
box(50, 82, 40, 5.5, "ScoreDriftResult / StreamingDriftResult / PointwiseDriftResult",
    bg="#e8f1ff", fs=8)
conn(50, 92, 29, 87.5)
conn(50, 92, 70, 87.5)

# ---- three family cards
card(3, 22, 40, 56, "univariate/   (19 detectors)",
     [("distributional (9, batch)",
       ["PSI", "KS", "JensenShannon", "Hellinger", "TotalVariation",
        "ChiSquare", "Bhattacharyya", "Wasserstein", "CramerVonMises"]),
      ("sequential / error-based (10, streaming)",
       ["ADWIN", "PageHinkley", "HDDM_A", "EDDM", "DDM",
        "RDDM", "CUSUM", "ECDD", "KSWIN", "STEPD"])],
     bg="#eef8ee", title_bg="#d7efd7")

card(45, 22, 27, 56, "multivariate/   (10, batch)",
     [("",
       ["MMD", "MMDLinear", "PCA_CD", "PCAReconstruction",
        "KDQTree", "EnergyDistance", "C2ST", "HDDDM",
        "MultivariateKS", "NNTwoSample"])],
     bg="#eef8ee", title_bg="#d7efd7")

card(74, 22, 23, 56, "model_based/   (1)",
     [("ModelDisagreementMetric (MDM)", ["+ pluggable DisagreementMetric:"]),
      ("error / correlation", ["MSE", "Pearson", "Spearman"]),
      ("temporal (time-series)", ["DTW", "LaggedCorrelation", "ACFDivergence"])],
     bg="#eef8ee", title_bg="#d7efd7")

for cx in (23, 58.5, 85.5):
    conn(29, 82, cx, 78.2)

ax.text(50, 18, "each detector folder ships:  detector.py  +  metadata.yaml  +  usage.md   —   NumPy-only runtime",
        ha="center", va="center", fontsize=9, style="italic", color="#555")

plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
