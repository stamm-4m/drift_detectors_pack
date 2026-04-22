import numpy as np
from river.drift import ADWIN as AW_ALG

from drift_detectors.drift_detector import DriftDetector
from drift_detectors.utility.drift_detection_output import StreamingDriftResult


class Adwin(DriftDetector):
    """
    ADWIN Drift Detector (univariate, streaming).

    ADWIN (ADaptive WINdowing) detects changes in the data distribution
    by maintaining a variable-length window and applying Hoeffding's inequality
    to identify statistically significant shifts in the mean.
    """

    def __init__(self, delta: float = 0.002, online: bool = False):
        """
        Initialize the ADWIN detector.

        Parameters:
            delta (float): Confidence parameter. Smaller values are more sensitive.
            online (bool): Whether to accumulate data over multiple calls.
        """
        super().__init__(reference_data=None, online=online)
        self._delta = delta
        self._adwin = AW_ALG(delta=self._delta)

    def calculate(
        self, test_data: np.ndarray, delta: float = None
    ) -> StreamingDriftResult:
        """
        Perform drift detection on the incoming data using ADWIN.

        Parameters:
            test_data (np.ndarray): New data point(s). Single float or 1D array.
            delta (float): Optional override of the delta parameter.

        Returns:
            StreamingDriftResult: Drift decision and index.
        """
        # Update delta if overridden
        if delta is not None and delta != self._delta:
            self._delta = delta
            if self.is_online():
                self._adwin = AW_ALG(delta=self._delta)

        # Choose detector instance depending on online/offline mode
        if self.is_online():
            detector = self._adwin
        else:
            detector = AW_ALG(delta=self._delta)

        test_data = np.asarray(test_data).ravel()
        drift_flag = False
        last_index = -1

        for i, value in enumerate(test_data):
            detector.update(value)
            if detector.drift_detected:
                drift_flag = True
                last_index = i

        # Only update internal ADWIN in online mode
        if self.is_online():
            self._adwin = detector

        return StreamingDriftResult(
            last_index=last_index,
            drift=drift_flag,
            details={
                "delta": self._delta,
                "mode": "online" if self.is_online() else "offline",
            },
        )
