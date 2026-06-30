from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoreDriftResult:
    """
    Single numeric score summarising drift plus a Boolean flag.
    """
    score: float
    drift: bool
    details: dict[str, Any] = field(default_factory=dict)

@dataclass
class PointwiseDriftResult:
    """
    List of dataset indices (or timestamps) where drift/change points occur.
    """
    indices: list[int]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingDriftResult:
    """
    Drift signal for the most-recent observation in a data stream.
    """
    last_index: int
    drift: bool
    details: dict[str, Any] = field(default_factory=dict)
