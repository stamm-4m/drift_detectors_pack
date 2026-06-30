# Use case: IndPenSim — industrial penicillin fermentation

This use case applies `drift_detectors_pack` to the
[IndPenSim](https://data.mendeley.com/datasets/pdnjz7zz5x/1) dataset, an
industrial-scale fed-batch fermentation simulator and dataset published by
Goldrick *et al.* (2015, 2019). It is the case study reported in the companion SoftwareX paper, an
Original Software Publication describing `drift_detectors_pack`.

The paper analyses the ten faulty test batches (91–100) of the LSTM soft
sensor of Metcalfe *et al.* (2025) against the same training set
(batches 1–60), with one twist: every detector is run *independently* in
each of the four canonical microbial growth phases (lag, log/exponential,
stationary, death) because fed-batch fermentation is non-stationary.

## Layout

```
use_cases/IndPenSim/
├── README.md                       (this file)
├── data/
│   ├── download_indpensim.py       fetcher for the full Goldrick CSV (~21 MB)
│   ├── test/                       ten test batches 91–100 shipped in-tree
│   │   ├── batch_91.csv  ...  batch_100.csv
│   │   └── README.md               fault windows and in-control labels per batch
│   └── reference/                  destination of the reference set (batches 1–60)
├── experiments/
│   ├── run_experiment_1.py         drift detectors over inputs, per phase
│   ├── run_experiment_2.py         MDM across CART / M5 / CUBIST / RF, per phase
│   ├── soft_sensors.py             pure-NumPy CART / M5 / CUBIST / RF
│   └── README.md
├── figures/
│   ├── fig_stamm_integration.png   Fig. 1 of the paper
│   ├── fig_class_diagram.png       Fig. 2 of the paper
│   ├── fig_fault_timelines.png     Fig. 3 of the paper
│   ├── make_class_diagram.py
│   └── make_fault_timelines.py
└── results/
    ├── expA_per_phase.json         per-variable PSI / KS per phase
    ├── expA_multivar_per_phase.json   MMD / PCA-CD / KDQ-Tree per phase
    ├── expB_per_phase.json         CART / M5 / CUBIST / RF perf + MDM per phase
    └── README.md                   per-batch tables S0–S3 (Markdown)
```

## Reproducing the paper

1. **Fetch the dataset.** The test batches are in `data/test/`. The 60-batch
   reference set is too large to ship in-tree (~12 MB compressed); fetch it
   with:

   ```bash
   python use_cases/IndPenSim/data/download_indpensim.py
   ```

2. **Install dependencies.**

   ```bash
   pip install -e ".[benchmark]"
   ```

3. **Run the experiments.**

   ```bash
   python -m use_cases.IndPenSim.experiments.run_experiment_1
   python -m use_cases.IndPenSim.experiments.run_experiment_2
   ```

   Results land in `use_cases/IndPenSim/results/`. The expected per-phase
   tables are reproduced in `results/README.md`.

4. **(Optional) regenerate the paper figures.**

   ```bash
   python use_cases/IndPenSim/figures/make_class_diagram.py
   python use_cases/IndPenSim/figures/make_fault_timelines.py
   ```

## Headline findings

* In the lag and log phases, per-variable PSI cleanly fingerprints the
  documented faults: PSI on the aerator reaches ≈2.39 on aeration-fault
  batches (94, 99, 100) and stays below 0.1 elsewhere; PSI on the sugar
  feed Fs reaches ≈3.44 on substrate-feed-fault batches (91, 95, 97) and
  stays below 0.01 elsewhere.
* The in-control batches 92 and 93 stay close to zero on both signals
  through lag and log, confirming the protocol does not generate false
  alarms.
* In the stationary and death phases the upstream actuator signals have
  largely returned to their reference distribution, but the downstream
  variables (DO2, off-gas CO2) accumulate the fault effect — PSI on
  off-gas CO2 reaches 21.2 on batch 91 in the stationary phase.
* Model Disagreement (MDM) across CART, M5, CUBIST and Random Forest soft
  sensors cleanly separates near-in-control batches (MDM ≈ 0.137–0.147)
  from faulty batches (MDM ≈ 0.181–0.427) in the log phase.

See `results/README.md` for the full per-batch tables.

## Citing

If you use this use case, please cite the companion SoftwareX paper:

```
Galindez, E., Crowther, M., Metcalfe, B., Koehorst, J. J.,
Aristizabal Morales, S., Suarez, C., Daboussi, F., Corrales, D. C.:
drift_detectors_pack: A unified drift detection toolkit for soft sensor
monitoring in industrial bioprocesses. SoftwareX (under review).
```

and the upstream IndPenSim dataset:

```
Goldrick, S., Stefan, A., Lovett, D., Montague, G., Lennox, B. (2015).
The development of an industrial-scale fed-batch fermentation simulation.
Journal of Biotechnology 193, 70-82.

Goldrick, S., Duran-Villalobos, C. A., Jankauskas, K., Lovett, D.,
Farid, S. S., Lennox, B. (2019). Modern day monitoring and control
challenges outlined on an industrial-scale benchmark fermentation
process. Computers & Chemical Engineering 130, 106471.
```
