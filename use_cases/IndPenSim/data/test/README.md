# IndPenSim test batches 91–100

The ten test batches used in the companion paper experiments. Each CSV
contains the full per-minute trace of one batch (~1100 rows, 31 columns).
The schema matches the upstream IndPenSim 100-batch dataset
(`100_Batches_IndPenSim_V3.1.csv`).

## Fault metadata

| Batch | Label                  | Documented fault                       | Phase active |
| ----- | ---------------------- | -------------------------------------- | ------------ |
| 91    | substrate-feed fault   | Fs disturbance ≈90–110 h               | log          |
| 92    | in-control reference   | none                                   | —            |
| 93    | in-control reference   | none                                   | —            |
| 94    | aeration fault         | Aerator disturbance ≈90–110 h          | log          |
| 95    | substrate-feed fault   | Fs disturbance ≈90–110 h               | log          |
| 96    | near in-control        | minor process variation only           | —            |
| 97    | substrate-feed fault   | Fs disturbance ≈100–110 h (mild)       | log          |
| 98    | near in-control        | minor process variation only           | —            |
| 99    | aeration fault         | Aerator disturbance ≈90–110 h          | log          |
| 100   | aeration fault         | Aerator disturbance ≈90–110 h          | log          |

The reference set used in the paper experiments is the LSTM training set of
Metcalfe *et al.* (2025), namely batches 1–60 of the same dataset. The 60
reference batches are not shipped in-tree because of size; fetch them with
`python use_cases/IndPenSim/data/download_indpensim.py`.

## Schema

Each CSV has the following columns:

```
time, aeration_rate, agitator, sugar_feed_rate,
acid_flow_rate, base_flow_rate, heating/cooling_water_flow_rate,
heating_water_flow_rate, water_for_injection/dilution,
air_head_pressure, dumped_broth_flow, substrate_concentration,
dissolved_oxygen_concentration, penicillin_concentration,
vessel_volume, vessel_weight, pH, temperature, generated_heat,
CO2_percent_in_off_gas, PAA_flow, PAA_concentration, oil_flow,
NH3_concentration, oxygen_uptake_rate,
oxygen_in_percent_in_off_gas, offline_penicillin_concentration,
offline_biomass_concentration, carbon_evolution_rate, ammonia_shots,
experiment_ID
```

`time` is in hours from the start of the batch. The four canonical
fermentation phases are: lag (0–30 h), log (30–120 h), stationary
(120–200 h), death (>200 h).

## License

IndPenSim is published under Creative Commons Attribution 4.0
(DOI 10.17632/pdnjz7zz5x.1).
