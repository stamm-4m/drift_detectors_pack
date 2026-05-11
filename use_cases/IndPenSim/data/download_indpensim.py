"""Download the IndPenSim 100-batch dataset from Mendeley Data.

The reference set used in the AI4D 2026 paper experiments is batches 1-60
of this dataset. The dataset is published under Creative Commons Attribution
4.0 by Goldrick et al. (2019).

Usage:
    python download_indpensim.py            # default destination is alongside this file
    python download_indpensim.py --out /path/to/save/

Dataset DOI: 10.17632/pdnjz7zz5x.1
Reference:   Goldrick, S. et al. (2019). Modern day monitoring and control
             challenges outlined on an industrial-scale benchmark fermentation
             process. Comp. Chem. Eng. 130, 106471.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

DOI = "10.17632/pdnjz7zz5x.1"
FNAME = "100_Batches_IndPenSim_V3.1.csv"
# Mendeley direct-file URL (may change; if this fails, browse to the DOI and
# download manually)
URL = (
    "https://data.mendeley.com/public-files/datasets/pdnjz7zz5x/files/"
    "9eaff7fb-4866-4998-9776-7e90fffe66f7/file_downloaded"
)
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description="Fetch the full IndPenSim 100-batch CSV.")
    p.add_argument("--out", default=str(HERE / FNAME),
                   help=f"Destination path (default: {HERE / FNAME}).")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"already present: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"downloading IndPenSim 100-batch dataset (DOI {DOI})")
    print(f"  from: {URL}")
    print(f"  to:   {out}")
    try:
        urlretrieve(URL, str(out))
    except Exception as e:
        print(f"\nautomatic download failed: {e}")
        print("Please download manually:")
        print(f"  https://data.mendeley.com/datasets/pdnjz7zz5x/1")
        print(f"  then save 100_Batches_IndPenSim_V3.1.csv to {out}")
        sys.exit(1)
    print(f"done. {out.stat().st_size / 1e6:.1f} MB")
    print("Reference batches for the AI4D 2026 paper experiments are IDs 1-60.")


if __name__ == "__main__":
    main()
