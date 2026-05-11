"""Regenerate fig_fault_timelines.png (Fig. 3 of the AI4D 2026 paper).

Plots, for each of the 10 test batches (91-100), the four diagnostic
process variables of IndPenSim in real units:
  - aeration rate (L/h)
  - sugar feed rate Fs (L/h)
  - dissolved oxygen DO2 (mg/L)
  - penicillin concentration (g/L, right axis)

It also shades the documented fault windows from the IndPenSim docs.

Requires:
  - the full IndPenSim CSV "100_Batches_IndPenSim_V3.1.csv" (downloadable
    via use_cases/IndPenSim/data/download_indpensim.py)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "data" / "100_Batches_IndPenSim_V3.1.csv"
OUT = HERE / "fig_fault_timelines.png"

PV = {
    "Aer":     "Aeration rate(Fg:L/h)",
    "Fs":      "Sugar feed rate(Fs:L/h)",
    "DO2":     "Dissolved oxygen concentration(DO2:mg/L)",
    "Pen":     "Penicillin concentration(P:g/L)",
}
TIME_COL = "Time (h)"
BATCH_COL = "Batch ID"

# Fault windows from IndPenSim Goldrick et al. (2019)
FAULT_WINDOWS = {
    91:  [("Fs",  90, 110)],
    92:  [],
    93:  [],
    94:  [("Aer", 90, 110)],
    95:  [("Fs",  90, 110)],
    96:  [],
    97:  [("Fs", 100, 110)],
    98:  [],
    99:  [("Aer", 90, 110)],
    100: [("Aer", 90, 110)],
}
BATCH_LABEL = {91: "Fs fault", 92: "in-control reference",
               93: "in-control reference", 94: "aeration fault",
               95: "Fs fault", 96: "near in-control",
               97: "Fs fault, mild", 98: "near in-control",
               99: "aeration fault", 100: "aeration fault"}

PHASE_BOUNDS = [30, 120, 200]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to IndPenSim CSV.")
    p.add_argument("--out", default=str(OUT), help="Path to write PNG.")
    args = p.parse_args()

    csv = Path(args.csv)
    if not csv.exists():
        raise SystemExit(
            f"IndPenSim CSV not found at {csv}.\n"
            "Run use_cases/IndPenSim/data/download_indpensim.py first, or pass --csv."
        )

    df = pd.read_csv(csv, usecols=list(PV.values()) + [TIME_COL, BATCH_COL])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    batches = [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    fig, axes = plt.subplots(5, 2, figsize=(11, 13), sharex=True)
    fig.suptitle("Per-batch input trajectories and fault windows on IndPenSim test batches 91-100",
                 y=0.995, fontsize=11)
    colours = {"Aer": "tab:orange", "Fs": "tab:blue", "DO2": "tab:green"}

    for ax, b in zip(axes.flat, batches):
        sub = df[df[BATCH_COL] == b]
        t = sub[TIME_COL].to_numpy()
        for k in ("Aer", "Fs", "DO2"):
            ax.plot(t, sub[PV[k]], color=colours[k], lw=0.9, alpha=0.85, label=k)
        ax2 = ax.twinx()
        ax2.plot(t, sub[PV["Pen"]], color="black", lw=0.9, label="Pen")
        for phase_t in PHASE_BOUNDS:
            ax.axvline(phase_t, color="#888", lw=0.5, ls=":", alpha=0.7)
        for var, t0, t1 in FAULT_WINDOWS.get(b, []):
            ax.axvspan(t0, t1, color="tab:red" if var == "Aer" else "tab:pink",
                       alpha=0.18)
        title = f"batch {b} ({BATCH_LABEL[b]})"
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="both", labelsize=8); ax2.tick_params(axis="both", labelsize=8)
        ax.set_ylabel("Aer / Fs (L/h)\nDO2 (mg/L)", fontsize=7)
        ax2.set_ylabel("Pen (g/L)", fontsize=7)

    for ax in axes[-1]:
        ax.set_xlabel("time in batch (h)", fontsize=8)

    handles = [plt.Line2D([0], [0], color=colours["Aer"], lw=1.5, label="Aeration (L/h)"),
               plt.Line2D([0], [0], color=colours["Fs"],  lw=1.5, label="Sugar feed Fs (L/h)"),
               plt.Line2D([0], [0], color=colours["DO2"], lw=1.5, label="Dissolved O2 (mg/L)"),
               plt.Line2D([0], [0], color="black",        lw=1.5, label="Penicillin (g/L)"),
               plt.matplotlib.patches.Patch(color="tab:red",  alpha=0.18, label="Aeration fault window"),
               plt.matplotlib.patches.Patch(color="tab:pink", alpha=0.18, label="Substrate-feed fault window"),
               plt.Line2D([0], [0], color="#888", lw=0.7, ls=":", label="Phase boundary (30, 120, 200 h)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=4, fontsize=8, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
