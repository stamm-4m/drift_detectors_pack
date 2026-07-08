"""Load the IndPenSim soft sensors from the STAMM model registry and run
inference, driven by each model's metadata (feature order, scalers, shape).

Panel used by Experiment 2 (Model Disagreement Mechanism):

    RF    0001  Acosta-Pavas et al. (2024)  sklearn RandomForestRegressor   (no scaling)
    CART  0005  Acosta-Pavas et al. (2024)  sklearn DecisionTreeRegressor   (no scaling)
    SVM   0007  STAMM (2026)                sklearn SVR                     (Z-score scalers)
    LSTM  0009  STAMM (2026)                keras LSTM, input (n, 1, 8)      (Min-Max scalers)

Two interpretable models (RF, CART) and two black-box models (SVM, LSTM):
the structural diversity of the panel is deliberate, so that model
disagreement reflects a genuine divergence in the input--output mapping
rather than the shared inductive bias of a single learner family.

The model binaries, their scalers and their registry config files live in
``use_cases/IndPenSim/models/`` (vendored from the STAMM model registry,
github.com/stamm-4m). Each config is the model's self-describing
``metadata.yaml`` and is the single source of truth for the feature order,
the scaler filenames and the output transform.

Requirements (not needed by the core drift_detectors package, only by this
use case): scikit-learn>=1.5 (RF/CART/SVM and the scalers) and
tensorflow>=2.18 (the LSTM). Install with::

    pip install "scikit-learn>=1.5" "tensorflow>=2.18" pyyaml
"""
from __future__ import annotations

import joblib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE.parent / "models"
CONFIGS_DIR = MODELS_DIR / "configs"

# label -> registry config (metadata.yaml). Model and scaler filenames are
# read FROM the config, so the registry stays the single source of truth.
REGISTRY = {
    "RF":   "0001_[python]_penicillin_RF.yaml",
    "CART": "0005_[python]_penicillin_CART.yaml",
    "SVM":  "0007_[python]_penicillin_SVM.yaml",
    "LSTM": "0009_[Python]_penicillin_LSTM.yaml",
}

# Order in which Experiment 2 builds the feature matrix. Verified at load time
# against every model's config (all four share this order).
FEATURES = [
    "temperature",
    "pH",
    "dissolved_oxygen_concentration",
    "agitator",
    "CO2_percent_in_off_gas",
    "oxygen_in_percent_in_off_gas",
    "vessel_volume",
    "sugar_feed_rate",
]


def _unpickle(path: Path):
    # sklearn models and scalers in the STAMM registry are saved with
    # joblib.dump, so we load them exactly like the model-registry /predict
    # endpoint does -- joblib.load, not pickle.load. joblib ships with
    # scikit-learn, so no extra dependency is required.
    return joblib.load(path)


def _read_config(label: str) -> dict:
    cfg = yaml.safe_load((CONFIGS_DIR / REGISTRY[label]).read_text(encoding="utf-8"))
    return cfg["ml_model_configuration"]


@dataclass
class SoftSensor:
    """A registry soft sensor: scale -> predict -> inverse-scale the target."""

    label: str
    feature_order: list[str]
    model: object
    feature_scaler: object | None
    target_scaler: object | None
    is_sequence: bool   # keras LSTM expects (n, 1, n_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict penicillin concentration (original units) for X = (n, 8),
        columns in :data:`FEATURES` order."""
        X = np.asarray(X, dtype=float)
        Xs = self.feature_scaler.transform(X) if self.feature_scaler is not None else X
        if self.is_sequence:
            Xs = Xs.reshape((Xs.shape[0], 1, Xs.shape[1]))
            pred = np.asarray(self.model.predict(Xs, verbose=0)).reshape(-1, 1)
        else:
            pred = np.asarray(self.model.predict(Xs)).reshape(-1, 1)
        if self.target_scaler is not None:
            pred = self.target_scaler.inverse_transform(pred)
        return pred.ravel()


def load_model(label: str) -> SoftSensor:
    cfg = _read_config(label)
    feature_order = [f["name"] for f in cfg["inputs"]["features"]]
    model_file = cfg["model_description"]["config_files"]["model_file"]
    fscaler_file = cfg["inputs"].get("scaler")
    tscaler_file = cfg["outputs"].get("scaler")

    model_path = MODELS_DIR / model_file
    is_sequence = model_path.suffix == ".keras"
    if is_sequence:
        from tensorflow import keras  # imported lazily; only the LSTM needs it
        model = keras.models.load_model(model_path)
    else:
        model = _unpickle(model_path)   # needs scikit-learn importable

    fscaler = _unpickle(MODELS_DIR / fscaler_file) if fscaler_file else None
    tscaler = _unpickle(MODELS_DIR / tscaler_file) if tscaler_file else None
    return SoftSensor(label, feature_order, model, fscaler, tscaler, is_sequence)


def load_all() -> dict[str, SoftSensor]:
    """Load the four-model panel and verify they share the FEATURES order."""
    models = {label: load_model(label) for label in REGISTRY}
    for label, m in models.items():
        if m.feature_order != FEATURES:
            raise ValueError(
                f"{label} feature order {m.feature_order} != expected {FEATURES}"
            )
    return models


if __name__ == "__main__":
    # Smoke test: load every model and predict on a tiny dummy batch.
    ms = load_all()
    Xd = np.tile([300.0, 6.5, 8.0, 100.0, 3.0, 18.0, 60000.0, 0.05], (4, 1))
    for name, m in ms.items():
        print(f"{name:5s} ->", np.round(m.predict(Xd), 4))
