# Statistical Test of Equal Proportions (STEPD)
The **Statistical Test of Equal Proportions (STEPD)** detector (Nishida & Yamauchi, 2007) is a univariate, sequential (streaming) drift detector. It consumes a **binary error stream** (values `> 0.5` = error; accuracy = `1 - error`) and reports whether a change point occurred.

The accuracy over a recent window is compared against the accuracy over all older observations with a two-proportion z-test; a *warning* fires when the p-value drops below `alpha_warning` and *drift* when it drops below `alpha_drift`.

Unlike the distributional detectors, streaming detectors do not take a reference/test pair. They are used in two modes:

* **Batch (offline)** — pass a whole array to `calculate()`. The internal state is reset on each call and the array is scanned start to finish.
* **Online** — construct with `online=True` and feed one value (or a small batch) per call. State persists across calls, so the detector behaves as a true single-pass stream monitor.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `window_size` | `30` | length of the recent-accuracy window |
| `alpha_warning` | `0.01` | p-value threshold for the warning zone |
| `alpha_drift` | `0.00001` | p-value threshold for the drift decision |
| `online` | `False` | persist state across `calculate()` calls |

## Batch (Offline) Usage
```python
import numpy as np
from drift_detectors import STEPD

# A stationary segment followed by an abrupt change at t = 500
stream = np.concatenate([np.zeros(500), np.ones(500)])

detector = STEPD()
result = detector.calculate(stream)

print("Drift Detected :", result.drift)
print("Change Index   :", result.last_index)   # -1 if no drift
print("Details        :", result.details)       # includes the 'warning' flag
```

## Online (Streaming) Usage
```python
import numpy as np
from drift_detectors import STEPD

stream = np.concatenate([np.zeros(500), np.ones(500)])

detector = STEPD(online=True)
for t, x in enumerate(stream):
    result = detector.calculate([x])   # feed one observation at a time
    if result.details["warning"]:
        pass                            # optional early-warning zone
    if result.drift:
        print(f"drift detected at t = {t}")
        break
```

## Feeding residuals instead of a 0/1 stream

This detector expects a **binary error stream**. To monitor a continuous model error, threshold it first:

```python
errors = (np.abs(residuals) > tolerance).astype(int)
STEPD().calculate(errors)
```

## Result

`calculate()` returns a `StreamingDriftResult` with:

* `drift` — `True` if a change point fired within this call;
* `last_index` — index (within the values passed to this call) of the last drift point, or `-1`;
* `details` — a dict with a `warning` flag, the `metric` name and the `mode` (`"online"`/`"offline"`).
