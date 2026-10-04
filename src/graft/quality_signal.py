from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
from dataclasses import dataclass, asdict, replace
import json, math, hashlib
import numpy as np
from scipy import signal
from .waveform_header import adjacent_equal_fraction,parse_header,signal_role


class GraftSignalQualityError(RuntimeError):
    """Fail-closed error for the M1 credibility study."""

@dataclass(frozen=True)
class PeakTrain:
    times_seconds: tuple[float, ...]
    prominences: tuple[float, ...]
    invalid_fraction: float
    adjacent_equal_fraction: float
    role: str

    @property
    def count(self) -> int:
        return len(self.times_seconds)

    @property
    def rate_bpm(self) -> float:
        if self.count < 2:
            return 0.0
        duration = self.times_seconds[-1] - self.times_seconds[0]
        return 60.0 * (self.count - 1) / duration if duration > 0.0 else 0.0

def stable_seed(base_seed: int, *parts: str) -> int:
    payload = "\x1f".join([str(base_seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)

def robust_standardize(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(values))
    if not math.isfinite(scale) or scale <= 1e-8:
        return np.zeros_like(values, dtype=np.float64)
    return (values - median) / scale

def interpolate_physical(
    digital: np.ndarray, baseline: float, gain: float, invalid_value: int
) -> tuple[np.ndarray, np.ndarray]:
    valid = digital != invalid_value
    if int(valid.sum()) < 2:
        raise GraftSignalQualityError("Signal has fewer than two valid samples")
    physical = (digital.astype(np.float64) - baseline) / gain
    indices = np.arange(len(physical), dtype=np.float64)
    interpolated = np.interp(indices, indices[valid], physical[valid])
    return interpolated, valid

def bandpass(values: np.ndarray, fs: float, band: Sequence[float], order: int) -> np.ndarray:
    sos = signal.butter(order, [float(band[0]), float(band[1])], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, values)

def select_channel_indices(
    rows: Sequence[Mapping[str, str]], config: Mapping[str, Any]
) -> dict[str, list[int]]:
    usable = [row for row in rows if row["provisional_usable"] == "True"]
    by_role: dict[str, list[int]] = {"ecg": [], "ppg": [], "abp": []}
    for row in sorted(usable, key=lambda value: int(value["channel_index"])):
        role = row["signal_role"]
        if role in by_role:
            by_role[role].append(int(row["channel_index"]))
    required = int(config["channel_policy"]["required_ecg_channels"])
    by_role["ecg"] = by_role["ecg"][:required]
    by_role["ppg"] = by_role["ppg"][:1]
    by_role["abp"] = by_role["abp"][:1]
    return by_role

def impulse_cross_correlation(
    anchor_times: np.ndarray,
    target_times: np.ndarray,
    fs: float,
    duration: float,
    minimum_lag: float,
    maximum_lag: float,
) -> float:
    length = int(round(duration * fs))
    anchor = np.zeros(length, dtype=np.float64)
    target = np.zeros(length, dtype=np.float64)
    for value in anchor_times:
        index = int(round(value * fs))
        if 0 <= index < length:
            anchor[index] = 1.0
    for value in target_times:
        index = int(round(value * fs))
        if 0 <= index < length:
            target[index] = 1.0
    kernel_x = np.arange(-8, 9, dtype=np.float64)
    kernel = np.exp(-0.5 * (kernel_x / 2.0) ** 2)
    anchor = np.convolve(anchor, kernel, mode="same")
    target = np.convolve(target, kernel, mode="same")
    anchor_norm = float(np.linalg.norm(anchor))
    target_norm = float(np.linalg.norm(target))
    if anchor_norm <= 0.0 or target_norm <= 0.0:
        return 0.0
    values: list[float] = []
    for lag in range(int(math.floor(minimum_lag * fs)), int(math.ceil(maximum_lag * fs)) + 1):
        if lag >= 0:
            left, right = anchor[: length - lag or None], target[lag:]
        else:
            left, right = anchor[-lag:], target[: length + lag]
        if len(left):
            values.append(float(np.dot(left, right) / (anchor_norm * target_norm)))
    return max(values, default=0.0)
