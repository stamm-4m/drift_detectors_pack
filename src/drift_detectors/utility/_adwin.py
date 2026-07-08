"""Pure-Python ADWIN (ADaptive WINdowing) drift detector.

An independent implementation of ADWIN2 (Bifet & Gavalda, "Learning from
Time-Changing Data with Adaptive Windowing", SDM 2007), using the
exponential-histogram bucket structure. It replaces the previous dependency
on ``river.drift.ADWIN`` so that the package needs only NumPy at runtime.

The public surface mirrors what the detector used from river: construct with
``delta``, call ``update(value)`` per observation, and read
``drift_detected`` after each update.
"""
from __future__ import annotations

import math

MAX_BUCKETS = 5


class _Row:
    """A doubly-linked node holding up to ``MAX_BUCKETS+1`` buckets, all of the
    same capacity (2**row_index)."""

    __slots__ = ("total", "variance", "count", "next", "previous")

    def __init__(self, next_row=None, prev_row=None):
        self.total = [0.0] * (MAX_BUCKETS + 1)
        self.variance = [0.0] * (MAX_BUCKETS + 1)
        self.count = 0
        self.next = next_row
        self.previous = prev_row
        if next_row is not None:
            next_row.previous = self
        if prev_row is not None:
            prev_row.next = self

    def insert(self, value, variance):
        self.total[self.count] = value
        self.variance[self.count] = variance
        self.count += 1

    def drop(self, n=1):
        for k in range(n, MAX_BUCKETS + 1):
            self.total[k - n] = self.total[k]
            self.variance[k - n] = self.variance[k]
        for k in range(1, n + 1):
            self.total[MAX_BUCKETS - k + 1] = 0.0
            self.variance[MAX_BUCKETS - k + 1] = 0.0
        self.count -= n


class ADWIN:
    """Adaptive windowing change detector."""

    def __init__(self, delta: float = 0.002):
        self.delta = float(delta)
        self.min_window_length = 5
        self.min_clock = 32
        self.reset()

    def reset(self):
        self.width = 0
        self.total = 0.0
        self.variance = 0.0
        self.bucket_number = 0
        self.time = 0
        self.last_row_index = 0
        self.drift_detected = False
        # linked list of rows; head is the most-recent row
        self.head = _Row()
        self.tail = self.head

    # ------------------------------------------------------------------ API
    def update(self, value: float) -> bool:
        """Add one observation; set and return ``drift_detected``."""
        value = float(value)
        self.width += 1
        self._insert_element(value)
        self._compress_buckets()
        self.drift_detected = self._detect_change()
        return self.drift_detected

    # --------------------------------------------------------------- helpers
    def _insert_element(self, value):
        self.head.insert(value, 0.0)
        self.bucket_number += 1
        if self.width > 1:
            self.variance += (
                (self.width - 1.0)
                * (value - self.total / (self.width - 1.0)) ** 2
                / self.width
            )
        self.total += value

    @staticmethod
    def _bucket_size(row_index: int) -> int:
        return 1 << row_index  # 2**row_index

    def _delete_element(self):
        """Drop the oldest bucket (tail row, first bucket)."""
        row = self.tail
        n = self._bucket_size(self.last_row_index)
        self.width -= n
        self.total -= row.total[0]
        mean = row.total[0] / n
        inc = (
            row.variance[0]
            + n * self.width * (mean - self.total / self.width) ** 2 / (n + self.width)
            if self.width > 0
            else row.variance[0]
        )
        self.variance -= inc
        self.variance = max(self.variance, 0.0)
        row.drop(1)
        self.bucket_number -= 1
        if row.count == 0 and self.tail.previous is not None:
            self.tail = self.tail.previous
            self.tail.next = None
            self.last_row_index -= 1

    def _compress_buckets(self):
        row = self.head
        i = 0
        while row is not None:
            if row.count == MAX_BUCKETS + 1:
                nxt = row.next
                if nxt is None:
                    nxt = _Row(prev_row=row)
                    self.tail = nxt
                    self.last_row_index += 1
                n1 = self._bucket_size(i)
                mean1 = row.total[0] / n1
                mean2 = row.total[1] / n1
                total12 = row.total[0] + row.total[1]
                temp = n1 * n1 * (mean1 - mean2) ** 2 / (n1 + n1)
                var12 = row.variance[0] + row.variance[1] + temp
                nxt.insert(total12, var12)
                self.bucket_number -= 1
                row.drop(2)
                if nxt.count <= MAX_BUCKETS:
                    break
            else:
                break
            row = row.next
            i += 1

    def _detect_change(self) -> bool:
        changed = False
        exit_flag = False
        while not exit_flag:
            exit_flag = True
            n0 = 0.0
            n1 = float(self.width)
            u0 = 0.0
            u1 = float(self.total)
            v0 = 0.0
            v1 = float(self.variance)
            # walk buckets from the tail (oldest) forward, growing W0
            row = self.tail
            i = self.last_row_index
            while row is not None:
                for b in range(row.count):
                    n = self._bucket_size(i)
                    mean = row.total[b] / n
                    n0 += n
                    n1 -= n
                    u0 += row.total[b]
                    u1 -= row.total[b]
                    if n0 > 0 and n1 > 0:
                        # move the bucket's variance contribution across the cut
                        v0 += row.variance[b] + n * (n0 - n) * (mean - (u0 - row.total[b]) / max(n0 - n, 1)) ** 2 / n0
                        v1 -= row.variance[b]
                    # skip the very last bucket (can't cut past the newest data)
                    is_last = (row.previous is None) and (b == row.count - 1)
                    if (
                        (not is_last)
                        and n0 >= self.min_window_length
                        and n1 >= self.min_window_length
                        and self._cut_expression(n0, n1, u0, u1, v0, v1)
                    ):
                        changed = True
                        exit_flag = False
                        if self.width > 0:
                            self._delete_element()
                        break
                if not exit_flag:
                    break
                row = row.previous
                i -= 1
        return changed

    def _cut_expression(self, n0, n1, u0, u1, v0, v1) -> bool:
        mean0 = u0 / n0
        mean1 = u1 / n1
        # harmonic window length
        m = 1.0 / (n0 - self.min_window_length + 1) + 1.0 / (n1 - self.min_window_length + 1)
        d = math.log(2.0 * math.log(self.width) / self.delta) if self.width > 1 else math.log(2.0 / self.delta)
        variance = (v0 + v1) / self.width
        epsilon = math.sqrt(2.0 * m * variance * d) + 2.0 / 3.0 * d * m
        return abs(mean0 - mean1) > epsilon
