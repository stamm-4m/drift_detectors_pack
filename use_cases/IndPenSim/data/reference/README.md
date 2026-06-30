# Reference set: IndPenSim batches 1–60

This folder is the destination of the reference set used in the companion SoftwareX
paper experiments — the LSTM training set of Metcalfe *et al.* (2025),
batches 1–60 of the IndPenSim dataset.

The 60 reference batches are too large to ship in-tree (~12 MB compressed,
~50 MB unpacked). The experiment scripts (`run_experiment_1.py`,
`run_experiment_2.py`) read directly from the full 100-batch CSV. To
download it:

```bash
python use_cases/IndPenSim/data/download_indpensim.py
```

This writes `100_Batches_IndPenSim_V3.1.csv` (~21 MB) to
`use_cases/IndPenSim/data/`. From there the experiment scripts filter on
the `Batch ID` column.
