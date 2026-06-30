# Model Disagreement Metric (MDM) -- usage

## Default configuration: predictions in, drift score out

```python
import numpy as np
from drift_detectors import ModelDisagreementMetric

# `y1`, `y2`, `y3`, `y4` are 1-D NumPy arrays of predictions exported by
# four co-deployed soft sensors on the same set of inputs.
res = ModelDisagreementMetric().calculate(predictions=[y1, y2, y3, y4])
print(res.score, res.drift)
print(res.details["metric_means"])   # {'mse': ..., 'pearson': ..., 'spearman': ...}
```

## Plug in a custom metric set

```python
from drift_detectors.model_based.disagreement_metrics import (
    DisagreementMetric, MSEDisagreement, PearsonDisagreement,
)

class MaxAbsDiff(DisagreementMetric):
    name = "max_abs"
    def pair(self, y_i, y_j, *, scale=1.0):
        return float(min(max(abs(y_i - y_j)) / max(scale, 1e-12), 1.0))

mdm = ModelDisagreementMetric(
    metrics=[MSEDisagreement(), PearsonDisagreement(), MaxAbsDiff()],
    threshold=0.20,
)
mdm.calculate(predictions=[y1, y2, y3])
```

## Why predictions, not models?

The STAMM model registry already exports per-batch simulations as plain
arrays. Having MDM consume those arrays (rather than callable model
objects) avoids coupling the monitoring layer to the model registry's
serialization format and lets MDM run wherever the predictions land --
in a notebook, in a dashboard backend, or inside an Airflow DAG.
