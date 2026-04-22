from __future__ import annotations
import os
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import yaml

curr_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = Path(os.path.join(curr_dir))

def load_metadata_for_class(detector_cls) -> Dict[str, Any]:
    """
    Load `meta.yaml` or `metadata.yaml` next to the detector’s Python file.
    Returns {} if no metadata is found.
    """
    mod_path = Path(inspect.getfile(detector_cls)).resolve().parent
    for name in ("meta.yaml", "metadata.yaml"):
        meta_file = mod_path / name
        if meta_file.is_file():
            with meta_file.open(encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
    return {}


def get_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Recursively scan for 'metadata.yaml' files and return all detector metadata.

    Each detector lives in its own subfolder
    and this function returns a dictionary 
    keyed by relative detector path.

    Returns:
        Dict[str, Dict[str, Any]]: e.g. {
            'univariate/ks': {...},
            'multivariate/mmd': {...},
        }
    """
    metadata_dict = {}

    for file in root_dir.rglob("metadata.yaml"):
        try:
            with file.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            metadata_dict[file.parent.name  ] = data

        except Exception as e:
            pass

    return metadata_dict

class DriftDetector(ABC):
    """
    Abstract base class for all drift detectors.

    Supports both online and offline detectors, with optional reference
    data management for stateful/stateless configurations.
    """

    def __init__(
        self,
        reference_data: Optional[np.ndarray] = None,
        online: Optional[bool] = False,
    ) -> None:
        """
        Initialize the drift detector.

        Parameters:
            reference_data (Optional[np.ndarray], optional): An optional array of
                reference data against which future test data will be compared.
                If not provided, it can be set later using `set_reference_data()`
                or passed directly to the `calculate()` method.

            online (bool, optional): Whether the detector should operate in
                online mode. In online mode, the detector is expected to consume
                data incrementally (e.g., one sample at a time) and maintain internal
                state (e.g., sliding windows or accumulators). Defaults to False.
        """

        self._reference_data = reference_data
        self._online = online
        self.metadata = load_metadata_for_class(self.__class__)

    def set_reference_data(self, reference_data: np.ndarray) -> None:
        """
        Store or update reference data used for drift comparison.
        """
        self._reference_data = reference_data

    def is_online(self) -> bool:
        """
        Indicates whether this detector operates in online mode.
        """
        return self._online

    @abstractmethod
    def calculate(
        self,
        test_data: np.ndarray,
        reference_data: Optional[np.ndarray] = None,
        **kwargs: Any,
    ):
        """
        Perform drift detection.

        Parameters:
            test_data (np.ndarray): New or streaming data to evaluate.
            reference_data (Optional[np.ndarray]): If provided, overrides internal state.

        Returns:
            DriftDetectionResult: Unified result object with score, flag, and metadata.
        """
        ...
