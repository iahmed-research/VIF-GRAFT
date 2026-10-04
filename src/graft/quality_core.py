from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
from dataclasses import dataclass, asdict, replace
import json, math, hashlib
import numpy as np
from scipy import signal
from sklearn.metrics import roc_auc_score
from .quality_signal import GraftSignalQualityError,PeakTrain,bandpass,impulse_cross_correlation,robust_standardize
from .waveform_header import adjacent_equal_fraction


@dataclass(frozen=True)
class ModalityRepresentation:
    role: str
    peak_train: PeakTrain
    quality: dict[str, float]
    waveform: tuple[float, ...] = ()
    waveform_fs: float = 0.0

def deterministic_limit_peaks(
    indices: np.ndarray, prominences: np.ndarray, maximum: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(indices) <= maximum:
        return indices, prominences
    ranked = sorted(
        range(len(indices)),
        key=lambda position: (-float(prominences[position]), int(indices[position])),
    )[:maximum]
    keep = np.asarray(
        sorted(ranked, key=lambda position: int(indices[position])), dtype=int
    )
    return indices[keep], prominences[keep]

def detect_ecg_peak_train(
    channels: Sequence[tuple[np.ndarray, np.ndarray]],
    fs: float,
    config: Mapping[str, Any],
) -> PeakTrain:
    if len(channels) != int(config["channel_policy"]["required_ecg_channels"]):
        raise GraftSignalQualityError("ECG detector requires exactly the frozen number of channels")
    filtered: list[np.ndarray] = []
    invalid_fractions: list[float] = []
    equal_fractions: list[float] = []
    for physical, valid in channels:
        filtered.append(
            robust_standardize(
                bandpass(
                    physical,
                    fs,
                    config["filters"]["ecg_band_hz"],
                    int(config["filters"]["butterworth_order"]),
                )
            )
        )
        invalid_fractions.append(float(1.0 - valid.mean()))
        equal = adjacent_equal_fraction(physical, valid)
        equal_fractions.append(float(equal) if math.isfinite(equal) else 1.0)
    energy = np.sqrt(np.mean(np.square(np.stack(filtered)), axis=0))
    width = max(
        1,
        int(round(float(config["event_detection"]["ecg_envelope_seconds"]) * fs)),
    )
    envelope = np.convolve(energy, np.ones(width) / width, mode="same")
    baseline = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - baseline)))
    prominence = max(
        1e-6,
        float(config["event_detection"]["prominence_mad_multiplier"])
        * 1.4826
        * mad,
    )
    indices, properties = signal.find_peaks(
        envelope,
        distance=max(
            1,
            int(
                float(config["event_detection"]["ecg_minimum_peak_distance_seconds"])
                * fs
            ),
        ),
        prominence=prominence,
        height=baseline + 0.5 * prominence,
    )
    indices, prominences = deterministic_limit_peaks(
        indices,
        properties["prominences"],
        int(config["event_detection"]["maximum_peaks"]),
    )
    return PeakTrain(
        times_seconds=tuple(float(value / fs) for value in indices),
        prominences=tuple(float(value) for value in prominences),
        invalid_fraction=float(np.mean(invalid_fractions)),
        adjacent_equal_fraction=float(np.mean(equal_fractions)),
        role="ecg",
    )

def coefficient_of_variation(values: np.ndarray, default: float = 2.0) -> float:
    if len(values) < 2:
        return default
    centre = float(np.mean(np.abs(values)))
    if centre <= 1e-8:
        return default
    return float(np.std(values) / centre)

def constrained_dtw_distance(left: np.ndarray, right: np.ndarray, band: int) -> float:
    if left.shape != right.shape or left.ndim != 1:
        raise GraftSignalQualityError("DTW inputs must be equal-length vectors")
    length = len(left)
    previous = np.full(length + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for i in range(1, length + 1):
        current = np.full(length + 1, np.inf, dtype=np.float64)
        start = max(1, i - band)
        stop = min(length, i + band)
        for j in range(start, stop + 1):
            cost = float((left[i - 1] - right[j - 1]) ** 2)
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    value = previous[length]
    return float(math.sqrt(value / length)) if math.isfinite(value) else 4.0

def beat_templates(
    waveform: np.ndarray,
    peak_times: Sequence[float],
    fs: float,
    output_length: int,
) -> list[np.ndarray]:
    peaks = np.asarray([int(round(value * fs)) for value in peak_times], dtype=int)
    peaks = peaks[(peaks >= 0) & (peaks < len(waveform))]
    if len(peaks) < 3:
        return []
    midpoints = ((peaks[:-1] + peaks[1:]) // 2).astype(int)
    templates: list[np.ndarray] = []
    for index in range(1, len(peaks) - 1):
        start = int(midpoints[index - 1])
        end = int(midpoints[index])
        if end - start < 5:
            continue
        segment = waveform[start:end]
        source = np.linspace(0.0, 1.0, len(segment))
        target = np.linspace(0.0, 1.0, output_length)
        templates.append(robust_standardize(np.interp(target, source, segment)))
    return templates

def representation_quality(
    waveform: np.ndarray | None,
    train: PeakTrain,
    fs: float,
    config: Mapping[str, Any],
) -> dict[str, float]:
    intervals = np.diff(np.asarray(train.times_seconds, dtype=np.float64))
    prominences = np.asarray(train.prominences, dtype=np.float64)
    quality = {
        "template_correlation_median": 0.0,
        "template_correlation_p10": 0.0,
        "template_dtw_distance_median": 2.0,
        "plateau_fraction": float(train.adjacent_equal_fraction),
        "invalid_fraction": float(train.invalid_fraction),
        "adjacent_equal_fraction": float(train.adjacent_equal_fraction),
        "interval_coefficient_of_variation": coefficient_of_variation(intervals),
        "prominence_coefficient_of_variation": coefficient_of_variation(prominences),
        "spectral_entropy": 1.0,
        "spectral_peak_fraction": 0.0,
        "robust_amplitude_range": 0.0,
    }
    if waveform is None or not len(waveform):
        return quality
    values = np.asarray(waveform, dtype=np.float64)
    tolerance = float(config["morphology"]["plateau_difference_tolerance"])
    quality["plateau_fraction"] = float(np.mean(np.abs(np.diff(values)) <= tolerance))
    quality["robust_amplitude_range"] = float(np.quantile(values, 0.99) - np.quantile(values, 0.01))
    spectrum = np.abs(np.fft.rfft(values - np.mean(values))) ** 2
    if len(spectrum) > 1:
        spectrum = spectrum[1:]
    total = float(np.sum(spectrum))
    if total > 1e-12:
        probabilities = spectrum / total
        entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-15)))
        quality["spectral_entropy"] = entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
        quality["spectral_peak_fraction"] = float(np.max(probabilities))
    templates = beat_templates(
        values,
        train.times_seconds,
        fs,
        int(config["morphology"]["template_length_samples"]),
    )
    if len(templates) >= int(config["morphology"]["minimum_complete_beats"]):
        stack = np.stack(templates)
        template = np.median(stack, axis=0)
        template = robust_standardize(template)
        correlations: list[float] = []
        distances: list[float] = []
        for beat in stack:
            if np.std(beat) <= 1e-8 or np.std(template) <= 1e-8:
                correlations.append(0.0)
            else:
                correlations.append(float(np.corrcoef(beat, template)[0, 1]))
            distances.append(
                constrained_dtw_distance(
                    beat,
                    template,
                    int(config["morphology"]["dtw_band_samples"]),
                )
            )
        quality["template_correlation_median"] = float(np.median(correlations))
        quality["template_correlation_p10"] = float(np.quantile(correlations, 0.10))
        quality["template_dtw_distance_median"] = float(np.median(distances))
    if any(not math.isfinite(value) for value in quality.values()):
        raise GraftSignalQualityError("Non-finite modality quality feature")
    return quality

def detect_pulse_representation(
    waveform: np.ndarray,
    role: str,
    fs: float,
    invalid_fraction: float,
    adjacent_equal_fraction: float,
    config: Mapping[str, Any],
) -> ModalityRepresentation:
    values = np.asarray(waveform, dtype=np.float64)
    baseline = float(np.median(values))
    mad = float(np.median(np.abs(values - baseline)))
    prominence = max(
        0.05,
        float(config["event_detection"]["prominence_mad_multiplier"]) * 1.4826 * mad,
    )
    indices, properties = signal.find_peaks(
        values,
        distance=max(
            1,
            int(round(float(config["event_detection"]["pulse_minimum_peak_distance_seconds"]) * fs)),
        ),
        prominence=prominence,
    )
    indices, prominences = deterministic_limit_peaks(
        indices,
        properties["prominences"],
        int(config["event_detection"]["maximum_peaks"]),
    )
    train = PeakTrain(
        times_seconds=tuple(float(index / fs) for index in indices),
        prominences=tuple(float(value) for value in prominences),
        invalid_fraction=float(invalid_fraction),
        adjacent_equal_fraction=float(adjacent_equal_fraction),
        role=role,
    )
    quality = representation_quality(values, train, fs, config)
    return ModalityRepresentation(
        role=role,
        peak_train=train,
        quality=quality,
        waveform=tuple(float(value) for value in values),
        waveform_fs=float(fs),
    )

def basic_representation(train: PeakTrain, config: Mapping[str, Any]) -> ModalityRepresentation:
    return ModalityRepresentation(
        role=train.role,
        peak_train=train,
        quality=representation_quality(None, train, 0.0, config),
    )

def serialize_representations(value: Mapping[str, ModalityRepresentation]) -> str:
    return json.dumps({role: asdict(rep) for role, rep in value.items()}, sort_keys=True, allow_nan=False)

def deserialize_representations(value: str) -> dict[str, ModalityRepresentation]:
    payload = json.loads(value)
    output: dict[str, ModalityRepresentation] = {}
    for role, rep in payload.items():
        train = rep["peak_train"]
        output[role] = ModalityRepresentation(
            role=str(rep["role"]),
            peak_train=PeakTrain(
                times_seconds=tuple(float(value) for value in train["times_seconds"]),
                prominences=tuple(float(value) for value in train["prominences"]),
                invalid_fraction=float(train["invalid_fraction"]),
                adjacent_equal_fraction=float(train["adjacent_equal_fraction"]),
                role=str(train["role"]),
            ),
            quality={key: float(value) for key, value in rep["quality"].items()},
            waveform=tuple(float(value) for value in rep["waveform"]),
            waveform_fs=float(rep["waveform_fs"]),
        )
    return output

def greedy_constant_lag_match(
    anchor: np.ndarray,
    target: np.ndarray,
    centre: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower, upper = centre - tolerance, centre + tolerance
    i = j = 0
    matched_anchor: list[float] = []
    matched_target: list[float] = []
    while i < len(anchor) and j < len(target):
        lag = float(target[j] - anchor[i])
        if lag < lower:
            j += 1
        elif lag > upper:
            i += 1
        else:
            matched_anchor.append(float(anchor[i]))
            matched_target.append(float(target[j]))
            i += 1
            j += 1
    left = np.asarray(matched_anchor, dtype=np.float64)
    right = np.asarray(matched_target, dtype=np.float64)
    return left, right, right - left

def consensus_lag_match(
    anchor: np.ndarray,
    target: np.ndarray,
    minimum: float,
    maximum: float,
    grid_step: float,
    tolerance: float,
    candidates_to_evaluate: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if not len(anchor) or not len(target):
        empty = np.asarray([], dtype=np.float64)
        return (minimum + maximum) / 2.0, empty, empty, empty
    differences = (target[:, None] - anchor[None, :]).ravel()
    differences = differences[(differences >= minimum) & (differences <= maximum)]
    if not len(differences):
        empty = np.asarray([], dtype=np.float64)
        return (minimum + maximum) / 2.0, empty, empty, empty
    edges = np.arange(minimum - grid_step / 2.0, maximum + 1.5 * grid_step, grid_step)
    histogram, _ = np.histogram(differences, bins=edges)
    radius = max(0, int(round(tolerance / grid_step)))
    support = np.convolve(histogram, np.ones(2 * radius + 1, dtype=int), mode="same")
    centres = minimum + np.arange(len(histogram), dtype=np.float64) * grid_step
    candidate_indices = sorted(
        range(len(centres)),
        key=lambda index: (-int(support[index]), abs(float(centres[index])), float(centres[index])),
    )[: max(1, candidates_to_evaluate)]
    best: tuple[Any, ...] | None = None
    best_value: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for index in candidate_indices:
        centre = float(centres[index])
        left, right, lags = greedy_constant_lag_match(anchor, target, centre, tolerance)
        residual = float(np.median(np.abs(lags - centre))) if len(lags) else math.inf
        score = (len(lags), -residual, -abs(centre), -centre)
        if best is None or score > best:
            best = score
            best_value = (centre, left, right, lags)
    assert best_value is not None
    centre, left, right, lags = best_value
    if len(lags):
        centre = float(np.median(lags))
        left, right, lags = greedy_constant_lag_match(anchor, target, centre, tolerance)
    return centre, left, right, lags

def phase_features(anchor: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    if len(anchor) < 2 or not len(target):
        return 0.0, 1.0
    phases: list[float] = []
    for value in target:
        index = int(np.searchsorted(anchor, value) - 1)
        if 0 <= index < len(anchor) - 1 and anchor[index + 1] > anchor[index]:
            phases.append(float((value - anchor[index]) / (anchor[index + 1] - anchor[index]) % 1.0))
    if not phases:
        return 0.0, 1.0
    resultant = abs(np.mean(np.exp(2j * np.pi * np.asarray(phases))))
    return float(resultant), float(1.0 - resultant)

def pair_feature_blocks(
    anchor: ModalityRepresentation,
    target: ModalityRepresentation,
    task_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    anchor_times = np.asarray(anchor.peak_train.times_seconds, dtype=np.float64)
    target_times = np.asarray(target.peak_train.times_seconds, dtype=np.float64)
    minimum = float(task_config["minimum_lag_seconds"])
    maximum = float(task_config["maximum_lag_seconds"])
    centre, matched_anchor, matched_target, lags = consensus_lag_match(
        anchor_times,
        target_times,
        minimum,
        maximum,
        float(config["alignment"]["lag_grid_seconds"]),
        float(config["alignment"]["matching_tolerance_seconds"]),
        int(config["alignment"]["candidate_centres_to_evaluate"]),
    )
    residuals = lags - centre
    residual_median = float(np.median(np.abs(residuals))) if len(residuals) else maximum - minimum
    residual_mad = (
        float(np.median(np.abs(residuals - np.median(residuals))))
        if len(residuals)
        else maximum - minimum
    )
    residual_iqr = (
        float(np.quantile(residuals, 0.75) - np.quantile(residuals, 0.25))
        if len(residuals)
        else maximum - minimum
    )
    lag_slope = (
        float(np.polyfit(matched_anchor, lags, 1)[0])
        if len(lags) >= 3 and float(np.ptp(matched_anchor)) > 0.0
        else 0.0
    )
    if len(matched_anchor) >= 3:
        anchor_intervals = np.diff(matched_anchor)
        target_intervals = np.diff(matched_target)
        interval_mae = float(np.mean(np.abs(anchor_intervals - target_intervals)))
        if np.std(anchor_intervals) > 1e-8 and np.std(target_intervals) > 1e-8:
            interval_corr = float(np.corrcoef(anchor_intervals, target_intervals)[0, 1])
        else:
            interval_corr = 0.0
    else:
        interval_mae = 2.0
        interval_corr = 0.0
    phase_locking, phase_variance = phase_features(anchor_times, target_times)
    duration = (
        int(config["window"]["end_sample_exclusive"])
        - int(config["window"]["start_sample"])
    ) / float(config["window"]["sampling_frequency_hz"])
    coupling = {
        "anchor_peak_count": float(anchor.peak_train.count),
        "target_peak_count": float(target.peak_train.count),
        "anchor_rate_bpm": float(anchor.peak_train.rate_bpm),
        "target_rate_bpm": float(target.peak_train.rate_bpm),
        "absolute_rate_difference_bpm": abs(anchor.peak_train.rate_bpm - target.peak_train.rate_bpm),
        "consensus_lag_seconds": centre,
        "anchor_match_fraction": float(len(lags) / anchor.peak_train.count) if anchor.peak_train.count else 0.0,
        "target_match_fraction": float(len(lags) / target.peak_train.count) if target.peak_train.count else 0.0,
        "lag_residual_median_absolute_seconds": residual_median,
        "lag_residual_mad_seconds": residual_mad,
        "lag_residual_iqr_seconds": residual_iqr,
        "lag_slope_seconds_per_second": lag_slope,
        "paired_interval_mae_seconds": interval_mae,
        "paired_interval_correlation": interval_corr,
        "phase_locking_value": phase_locking,
        "phase_circular_variance": phase_variance,
        "impulse_cross_correlation": impulse_cross_correlation(
            anchor_times,
            target_times,
            float(config["window"]["sampling_frequency_hz"]),
            duration,
            minimum,
            maximum,
        ),
    }
    quality: dict[str, float] = {}
    for prefix, representation in (("anchor", anchor), ("target", target)):
        for name in config["quality_features"]:
            quality[f"{prefix}_{name}"] = float(representation.quality[name])
    if any(not math.isfinite(value) for value in [*coupling.values(), *quality.values()]):
        raise GraftSignalQualityError("Non-finite quality feature")
    return coupling, quality

def timing_corruption(
    representation: ModalityRepresentation,
    family: str,
    severity: float,
    duration: float,
    seed: int,
    config: Mapping[str, Any],
) -> ModalityRepresentation:
    generator = np.random.default_rng(seed)
    original = representation.peak_train
    times = np.asarray(original.times_seconds, dtype=np.float64)
    prominences = np.asarray(original.prominences, dtype=np.float64)
    if family == "jitter":
        times = np.clip(times + generator.normal(0.0, severity, len(times)), 0.0, duration - 1e-6)
    elif family == "circular_shift":
        times = np.mod(times + severity, duration)
    elif family == "lag_drift":
        displacement = severity * (times / duration - 0.5)
        times = np.clip(times + displacement, 0.0, duration - 1e-6)
    elif family == "beat_edit":
        original_count = len(times)
        if original_count:
            delete_count = min(
                original_count - 1,
                max(1, int(round(original_count * float(config["corruptions"]["beat_delete_fraction"])))),
            )
            insert_count = max(
                1,
                int(round(original_count * float(config["corruptions"]["beat_insert_fraction"]))),
            )
            keep = np.ones(original_count, dtype=bool)
            keep[generator.choice(original_count, size=delete_count, replace=False)] = False
            times = times[keep]
            prominences = prominences[keep]
            inserted = generator.uniform(0.0, duration, size=insert_count)
            inserted_prominence = np.full(
                insert_count,
                float(np.median(prominences)) if len(prominences) else 1.0,
            )
            times = np.concatenate([times, inserted])
            prominences = np.concatenate([prominences, inserted_prominence])
    else:
        raise GraftSignalQualityError(f"Unknown timing corruption: {family}")
    order = np.argsort(times, kind="mergesort")
    times, prominences = times[order], prominences[order]
    train = replace(
        original,
        times_seconds=tuple(float(value) for value in times),
        prominences=tuple(float(value) for value in prominences),
    )
    quality = dict(representation.quality)
    if family == "beat_edit":
        intervals = np.diff(times)
        quality["interval_coefficient_of_variation"] = coefficient_of_variation(intervals)
        quality["prominence_coefficient_of_variation"] = coefficient_of_variation(prominences)
    return replace(representation, peak_train=train, quality=quality)

def waveform_corruption(
    representation: ModalityRepresentation,
    family: str,
    seed: int,
    config: Mapping[str, Any],
) -> ModalityRepresentation:
    if not representation.waveform or representation.waveform_fs <= 0.0:
        raise GraftSignalQualityError("Waveform corruption requires a pulse waveform")
    generator = np.random.default_rng(seed)
    waveform = np.asarray(representation.waveform, dtype=np.float64).copy()
    fs = representation.waveform_fs
    if family == "flatline":
        width = min(len(waveform) - 1, int(round(float(config["corruptions"]["flatline_duration_seconds"]) * fs)))
        start = int(generator.integers(0, max(1, len(waveform) - width)))
        waveform[start : start + width] = waveform[start]
    elif family == "clipping":
        lower_q, upper_q = config["corruptions"]["clipping_quantiles"]
        lower, upper = np.quantile(waveform, [float(lower_q), float(upper_q)])
        waveform = np.clip(waveform, lower, upper)
    elif family == "band_limited_noise":
        noise = generator.normal(size=len(waveform))
        sos = signal.butter(3, [0.5, 8.0], btype="bandpass", fs=fs, output="sos")
        noise = signal.sosfiltfilt(sos, noise)
        signal_rms = float(np.sqrt(np.mean(np.square(waveform))))
        noise_rms = float(np.sqrt(np.mean(np.square(noise))))
        snr = float(config["corruptions"]["band_limited_noise_snr_db"])
        scale = signal_rms / (10.0 ** (snr / 20.0) * max(noise_rms, 1e-12))
        waveform = waveform + scale * noise
    elif family == "morphology_reverse":
        peaks = np.asarray(
            [int(round(value * fs)) for value in representation.peak_train.times_seconds],
            dtype=int,
        )
        if len(peaks) >= 3:
            midpoints = ((peaks[:-1] + peaks[1:]) // 2).astype(int)
            segments = [(int(midpoints[index - 1]), int(midpoints[index])) for index in range(1, len(peaks) - 1)]
            count = max(1, int(round(len(segments) * float(config["corruptions"]["morphology_reverse_fraction"]))))
            for index in generator.choice(len(segments), size=min(count, len(segments)), replace=False):
                start, end = segments[int(index)]
                waveform[start:end] = waveform[start:end][::-1]
    else:
        raise GraftSignalQualityError(f"Unknown waveform corruption: {family}")
    equal_fraction = float(np.mean(np.abs(np.diff(waveform)) <= float(config["morphology"]["plateau_difference_tolerance"])))
    return detect_pulse_representation(
        waveform,
        representation.role,
        fs,
        representation.peak_train.invalid_fraction,
        equal_fraction,
        config,
    )

def family_record_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    weights = np.zeros(len(rows), dtype=np.float64)
    by_record: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_record.setdefault(str(row["record"]), []).append(index)
    for indices in by_record.values():
        real = [index for index in indices if int(rows[index]["is_real_pair"]) == 1]
        corrupt_by_family: dict[str, list[int]] = {}
        for index in indices:
            if int(rows[index]["is_real_pair"]) == 0:
                corrupt_by_family.setdefault(str(rows[index]["corruption_family"]), []).append(index)
        if not real or not corrupt_by_family:
            continue
        for index in real:
            weights[index] = 0.5 / len(real)
        family_mass = 0.5 / len(corrupt_by_family)
        for family_indices in corrupt_by_family.values():
            for index in family_indices:
                weights[index] = family_mass / len(family_indices)
    if not np.any(weights > 0.0):
        raise GraftSignalQualityError("No evaluable record-balanced rows")
    return weights

def record_family_weighted_auc(rows: Sequence[Mapping[str, Any]], prediction_key: str) -> float:
    labels = np.asarray([int(row["is_real_pair"]) for row in rows], dtype=int)
    predictions = np.asarray([float(row[prediction_key]) for row in rows], dtype=np.float64)
    if set(labels) != {0, 1}:
        raise GraftSignalQualityError("AUROC subset does not contain both classes")
    return float(roc_auc_score(labels, predictions, sample_weight=family_record_weights(rows)))
