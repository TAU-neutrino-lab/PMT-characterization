"""PMT waveform preprocessing helpers."""

from __future__ import annotations

import numpy as np


def subtract_baseline(time_ns, voltage_mV, baseline_window_ns=None, baseline_window=None):
    """baseline is the mean wavefrom in baseline_window_ns."""

    if baseline_window_ns is None:
        baseline_window_ns = baseline_window
    if baseline_window_ns is None:
        baseline_window_ns = (0.0, 20.0)

    mask = (time_ns >= baseline_window_ns[0]) & (time_ns <= baseline_window_ns[1])
    if not np.any(mask):
        raise ValueError(f"Baseline window {baseline_window_ns} contains no samples")

    baseline_mV = np.mean(voltage_mV[:, mask], axis=1)
    voltage_bs_mV = voltage_mV - baseline_mV[:, None]
    return voltage_bs_mV, baseline_mV


__all__ = ["subtract_baseline"]
