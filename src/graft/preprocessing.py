"""Frozen common preprocessing for GRAFT and matched comparators.

Long invalid gaps are never interpolated across.  Filtering is performed
independently on contiguous finite segments with explicit odd padding; segments
too short for zero-phase filtering remain invalid and become zero only after
normalization.  A 60 Hz Q=30 notch is applied before each modality bandpass,
following the machine-readable Candidate 003 configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import signal


NOTCH_QUALITY_FACTOR = 30.0
NOTCH_PADLEN = 9


class GraftPreprocessingError(RuntimeError):
    """Raised when a waveform cannot be handled under the frozen policy."""


@dataclass(frozen=True)
class PreprocessedChannel:
    tensor: np.ndarray
    original_invalid_mask: np.ndarray
    model_valid_mask: np.ndarray
    diagnostics: dict[str, Any]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def interpolate_short_internal_runs(
    physical: Sequence[float],
    invalid_mask: Sequence[bool],
    maximum_run_samples: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    values = np.asarray(physical, dtype=np.float64)
    invalid = np.asarray(invalid_mask, dtype=bool)
    if values.ndim != 1 or invalid.shape != values.shape or not len(values):
        raise GraftPreprocessingError("physical and invalid_mask must be aligned nonempty vectors")
    if maximum_run_samples < 0:
        raise GraftPreprocessingError("maximum_run_samples must be nonnegative")
    invalid = invalid | ~np.isfinite(values)
    output = values.copy()
    output[invalid] = np.nan
    model_valid = ~invalid
    interpolated_samples = 0
    for start, stop in _runs(invalid):
        length = stop - start
        internal = start > 0 and stop < len(output)
        if not internal or length > maximum_run_samples:
            continue
        left = output[start - 1]
        right = output[stop]
        if not math.isfinite(float(left)) or not math.isfinite(float(right)):
            continue
        output[start:stop] = np.linspace(left, right, length + 2, dtype=np.float64)[1:-1]
        model_valid[start:stop] = True
        interpolated_samples += length
    return output, model_valid, interpolated_samples


def design_filters(
    role: str,
    sampling_frequency_hz: float,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if role == "ecg":
        filter_spec = config["ecg_filters_hz"]
        band = (float(filter_spec["highpass"]), float(filter_spec["lowpass"]))
    elif role == "ppg":
        filter_spec = config["ppg_filters_hz"]
        band = tuple(float(value) for value in filter_spec["bandpass"])
    elif role == "abp":
        filter_spec = config["abp_filters_hz"]
        band = tuple(float(value) for value in filter_spec["bandpass"])
    else:
        raise GraftPreprocessingError(f"Unsupported signal role: {role!r}")
    fs = float(sampling_frequency_hz)
    nyquist = fs / 2.0
    if not (0.0 < band[0] < band[1] < nyquist):
        raise GraftPreprocessingError(f"Invalid {role} band {band!r} at fs={fs}")
    order = int(config["filter_order"])
    if order < 1:
        raise GraftPreprocessingError("filter_order must be positive")
    sos = signal.butter(order, band, btype="bandpass", fs=fs, output="sos")
    notch_hz = float(filter_spec["notch"])
    if not (0.0 < notch_hz < nyquist):
        raise GraftPreprocessingError(f"Notch {notch_hz} is not meaningful at fs={fs}")
    notch_b, notch_a = signal.iirnotch(notch_hz, NOTCH_QUALITY_FACTOR, fs=fs)
    bandpass_padlen = 3 * (2 * len(sos) + 1)
    return sos, notch_b, notch_a, bandpass_padlen


def preprocess_channel(
    digital: Sequence[float],
    *,
    baseline: float,
    gain: float,
    role: str,
    sampling_frequency_hz: float,
    config: Mapping[str, Any],
) -> PreprocessedChannel:
    """Convert a digital WFDB channel to physical units, then preprocess it.

    Stress generation must not call this entry point because doing so would
    reinterpret already-physical values as digital samples.  It uses
    :func:`preprocess_physical_channel` instead.
    """

    digital_array = np.asarray(digital, dtype=np.float64)
    if digital_array.ndim != 1 or not len(digital_array):
        raise GraftPreprocessingError("digital waveform must be a nonempty vector")
    if not math.isfinite(float(baseline)):
        raise GraftPreprocessingError("baseline must be finite")
    if not math.isfinite(float(gain)) or gain <= 0.0:
        raise GraftPreprocessingError("gain must be positive and finite")
    expected_samples = int(round(float(sampling_frequency_hz) * 10.0))
    if len(digital_array) != expected_samples:
        raise GraftPreprocessingError(
            f"Expected {expected_samples} samples for the frozen window, got {len(digital_array)}"
        )
    sentinel = float(config["invalid_digital_value"])
    original_invalid = (digital_array == sentinel) | ~np.isfinite(digital_array)
    physical = (digital_array - float(baseline)) / float(gain)
    processed = _preprocess_physical(
        physical,
        original_invalid,
        role=role,
        sampling_frequency_hz=sampling_frequency_hz,
        config=config,
        input_domain="digital_converted_to_physical",
    )
    return processed


def preprocess_physical_channel(
    physical: Sequence[float],
    valid_mask: Sequence[bool],
    *,
    role: str,
    sampling_frequency_hz: float,
    config: Mapping[str, Any],
) -> PreprocessedChannel:
    """Preprocess physical values without re-quantisation or sentinel inference."""

    physical_array = np.asarray(physical, dtype=np.float64)
    valid = np.asarray(valid_mask)
    if (
        physical_array.ndim != 1
        or not len(physical_array)
        or valid.shape != physical_array.shape
        or valid.dtype != np.bool_
    ):
        raise GraftPreprocessingError(
            "physical waveform and literal Boolean valid_mask must be aligned nonempty vectors"
        )
    expected_samples = int(round(float(sampling_frequency_hz) * 10.0))
    if len(physical_array) != expected_samples:
        raise GraftPreprocessingError(
            f"Expected {expected_samples} samples for the frozen window, got {len(physical_array)}"
        )
    original_invalid = ~valid | ~np.isfinite(physical_array)
    return _preprocess_physical(
        physical_array,
        original_invalid,
        role=role,
        sampling_frequency_hz=sampling_frequency_hz,
        config=config,
        input_domain="physical_with_explicit_valid_mask",
    )


def _preprocess_physical(
    physical: np.ndarray,
    original_invalid: np.ndarray,
    *,
    role: str,
    sampling_frequency_hz: float,
    config: Mapping[str, Any],
    input_domain: str,
) -> PreprocessedChannel:
    """Shared filtering and normalization after physical-domain validation."""

    if not math.isfinite(float(sampling_frequency_hz)) or sampling_frequency_hz <= 0.0:
        raise GraftPreprocessingError("sampling_frequency_hz must be positive and finite")
    maximum_run = int(
        round(
            float(config["interpolate_internal_invalid_runs_max_seconds"])
            * float(sampling_frequency_hz)
        )
    )
    interpolated, model_valid, interpolated_samples = interpolate_short_internal_runs(
        physical, original_invalid, maximum_run
    )
    sos, notch_b, notch_a, bandpass_padlen = design_filters(
        role, sampling_frequency_hz, config
    )
    filtered = np.full_like(interpolated, np.nan)
    short_segment_samples = 0
    minimum_segment_length = max(NOTCH_PADLEN, bandpass_padlen) + 1
    for start, stop in _runs(model_valid):
        segment = interpolated[start:stop]
        if len(segment) < minimum_segment_length:
            model_valid[start:stop] = False
            short_segment_samples += len(segment)
            continue
        notched = signal.filtfilt(
            notch_b,
            notch_a,
            segment,
            padtype="odd",
            padlen=NOTCH_PADLEN,
        )
        filtered[start:stop] = signal.sosfiltfilt(
            sos,
            notched,
            padtype="odd",
            padlen=bandpass_padlen,
        )
    model_valid &= np.isfinite(filtered)
    valid_values = filtered[model_valid]
    degenerate = len(valid_values) == 0
    if degenerate:
        median = 0.0
        mad_scale = float(config["mad_floor"])
        normalized = np.zeros_like(filtered)
    else:
        median = float(np.median(valid_values))
        mad_scale = 1.4826 * float(np.median(np.abs(valid_values - median)))
        floor = float(config["mad_floor"])
        if not math.isfinite(mad_scale) or mad_scale < floor:
            mad_scale = floor
            degenerate = True
        normalized = (filtered - median) / mad_scale
        normalized[~model_valid] = float(config["longer_invalid_runs_filled_after_normalization_with"])
    clip_low, clip_high = (float(value) for value in config["amplitude_clip_after_normalization"])
    if not clip_low < clip_high:
        raise GraftPreprocessingError("Invalid amplitude clipping interval")
    normalized = np.clip(normalized, clip_low, clip_high)
    normalized[~np.isfinite(normalized)] = float(
        config["longer_invalid_runs_filled_after_normalization_with"]
    )
    tensor = normalized.astype(np.float32)
    if not np.all(np.isfinite(tensor)):
        raise GraftPreprocessingError("Preprocessing produced a non-finite tensor")
    return PreprocessedChannel(
        tensor=tensor,
        original_invalid_mask=original_invalid.astype(bool),
        model_valid_mask=model_valid.astype(bool),
        diagnostics={
            "input_domain": input_domain,
            "role": role,
            "samples": len(tensor),
            "original_invalid_samples": int(np.sum(original_invalid)),
            "short_gap_interpolated_samples": interpolated_samples,
            "short_filter_segment_discarded_samples": short_segment_samples,
            "model_valid_samples": int(np.sum(model_valid)),
            "normalization_median": median,
            "normalization_mad_scale": mad_scale,
            "normalization_degenerate": degenerate,
            "notch_hz": float(
                config[f"{role}_filters_hz"]["notch"]
            ),
            "notch_quality_factor": NOTCH_QUALITY_FACTOR,
            "notch_padtype": "odd",
            "notch_padlen": NOTCH_PADLEN,
            "bandpass_sos": sos.tolist(),
            "bandpass_padtype": "odd",
            "bandpass_padlen": bandpass_padlen,
            "clip": [clip_low, clip_high],
        },
    )


def select_channels(
    channel_rows: Sequence[Mapping[str, str]], ecg_anchor_channels: int = 2
) -> dict[str, tuple[int, ...]]:
    """Apply the exact Candidate 003 usable-first channel policy."""

    if not channel_rows:
        raise GraftPreprocessingError("Event has no channel metadata")
    indices = [int(row["channel_index"]) for row in channel_rows]
    if len(indices) != len(set(indices)):
        raise GraftPreprocessingError("Duplicate channel index")
    ordered = sorted(channel_rows, key=lambda row: int(row["channel_index"]))
    all_ecg = [int(row["channel_index"]) for row in ordered if row["signal_role"] == "ecg"]
    usable_ecg = [
        int(row["channel_index"])
        for row in ordered
        if row["signal_role"] == "ecg" and row["provisional_usable"] == "True"
    ]
    selected_ecg = usable_ecg[:ecg_anchor_channels]
    if len(selected_ecg) < ecg_anchor_channels:
        selected_ecg.extend(
            index
            for index in all_ecg
            if index not in selected_ecg
        )
        selected_ecg = selected_ecg[:ecg_anchor_channels]
    if not selected_ecg:
        raise GraftPreprocessingError(
            "Event has no ECG channel in its header"
        )
    # Some released VTaC events contain one ECG channel although the modeling
    # tensor has two ECG slots.  Return the one observed lead exactly once; the
    # cache builder leaves the unfilled second slot at zero with an all-false
    # validity mask.  Duplication would fabricate a second observation.

    def first_usable(role: str) -> tuple[int, ...]:
        matches = [
            int(row["channel_index"])
            for row in ordered
            if row["signal_role"] == role and row["provisional_usable"] == "True"
        ]
        return tuple(matches[:1])

    return {
        "ecg": tuple(selected_ecg),
        "ppg": first_usable("ppg"),
        "abp": first_usable("abp"),
    }
