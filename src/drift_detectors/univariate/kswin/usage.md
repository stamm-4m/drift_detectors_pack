# Kolmogorov-Smirnov Windowing (KSWIN)
The **Kolmogorov-Smirnov Windowing (KSWIN)** detector (Raab et al., 2020) is a univariate, sequential (streaming) drift detector. It consumes any numeric value stream and reports whether a change point occurred.

A sliding window of the most recent values is kept; a two-sample KS test is run between the newest `stat_size` values and a random sample of the older part of the window, and drift fires when the KS p-value drops below `alpha`.

Unlike the distributional detectors, streaming detectors do not take a reference/test pair. They are used in two modes:

* **Batch (offline)** — pass a whole array to `calculate()`. The internal state is reset on each call and the array is scanned start to finish.
* **Online** — construct with `online=True` and feed one value (or a small batch) per call. State persists across calls, so the detector behaves as a true single-pass stream monitor.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `alpha` | `0.00002` | significance level for the KS test (smaller = stricter) |
| `window_size` | `100` | total sliding-window length |
| `stat_size` | `30` | size of the recent block compared against the window |
| `seed` | `0` | RNG seed for the random subsample (reproducibility) |
| `online` | `False` | persist state across `calculate()` calls |

## Batch (Offline) Usage
```python
import numpy as np
from drift_detectors import KSWIN

# A stationary segment followed by an abrupt change at t = 500
stream = np.concatenate([np.zeros(500), np.ones(500)])

detector = KSWIN()
result = detector.calculate(stream)

print("Drift Detected :", result.drift)
print("Change Index   :", result.last_index)   # -1 if no drift
print("Details        :", result.details)       # includes the 'warning' flag
```

## Online (Streaming) Usage
```python
import numpy as np
from drift_detectors import KSWIN

stream = np.concatenate([np.zeros(500), np.ones(500)])

detector = KSWIN(online=True)
for t, x in enumerate(stream):
    result = detector.calculate([x])   # feed one observation at a time
    if result.details["warning"]:
        pass                            # optional early-warning zone
    if result.drift:
        print(f"drift detected at t = {t}")
        break
```

## Result

`calculate()` returns a `StreamingDriftResult` with:

* `drift` — `True` if a change point fired within this call;
* `last_index` — index (within the values passed to this call) of the last drift point, or `-1`;
* `details` — a dict with a `warning` flag, the `metric` name and the `mode` (`"online"`/`"offline"`).
