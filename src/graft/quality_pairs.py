from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
from dataclasses import dataclass, asdict, replace
import json, math, hashlib
import numpy as np
from scipy import signal
from .quality_core import *
from .quality_signal import *
from .waveform_header import adjacent_equal_fraction,parse_header,signal_role


SCHEMA_VERSION = "graft-credibility-1.0.0"

CORRUPTION_FAMILIES = (
    "jitter",
    "circular_shift",
    "lag_drift",
    "beat_edit",
    "cross_record",
    "flatline",
    "clipping",
    "band_limited_noise",
    "morphology_reverse",
)

TIMING_FAMILIES = (
    "jitter",
    "circular_shift",
    "lag_drift",
    "beat_edit",
    "cross_record",
)

def extract_event_representations(
    dataset_root: Path,
    event_row: Mapping[str, str],
    channel_rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> dict[str, ModalityRepresentation]:
    stem = dataset_root / "waveforms" / event_row["record"] / event_row["event"]
    header = parse_header(stem.with_suffix(".hea"))
    source_fs = float(config["window"]["sampling_frequency_hz"])
    pulse_fs = float(config["window"]["pulse_representation_hz"])
    if header.sampling_frequency_hz != source_fs:
        raise GraftSignalQualityError(f"Unexpected sample rate for {event_row['event']}")
    files = {spec.file_name for spec in header.signals}
    if len(files) != 1:
        raise GraftSignalQualityError(f"Multiple waveform files for {event_row['event']}")
    memory = np.memmap(
        stem.parent / next(iter(files)),
        dtype="<i2",
        mode="r",
        shape=(header.sample_count, header.signal_count),
    )
    start = int(config["window"]["start_sample"])
    end = int(config["window"]["end_sample_exclusive"])
    if end > header.sample_count or start < 0 or start >= end:
        raise GraftSignalQualityError(f"Frozen window is invalid for {event_row['event']}")
    invalid = int(config["window"]["invalid_digital_value"])
    selected = select_channel_indices(channel_rows, config)
    output: dict[str, ModalityRepresentation] = {}
    required_ecg = int(config["channel_policy"]["required_ecg_channels"])
    if len(selected["ecg"]) == required_ecg:
        channels: list[tuple[np.ndarray, np.ndarray]] = []
        for index in selected["ecg"]:
            spec = header.signals[index]
            physical, valid = interpolate_physical(
                np.asarray(memory[start:end, index]), spec.baseline, spec.gain, invalid
            )
            channels.append((physical, valid))
        output["ecg"] = basic_representation(
            detect_ecg_peak_train(channels, source_fs, config), config
        )
    down = int(round(source_fs / pulse_fs))
    if not math.isclose(source_fs / pulse_fs, down, abs_tol=1e-12):
        raise GraftSignalQualityError("Pulse representation rate must be an integer divisor")
    for role in ("ppg", "abp"):
        if not selected[role]:
            continue
        index = selected[role][0]
        spec = header.signals[index]
        expected = signal_role(
            spec.name,
            {"signal_roles": {"ecg": [], "ppg": ["PLETH"], "abp": ["ABP"]}},
        )
        if expected != role:
            raise GraftSignalQualityError(f"Role disagreement for {event_row['event']} channel {index}")
        physical, valid = interpolate_physical(
            np.asarray(memory[start:end, index]), spec.baseline, spec.gain, invalid
        )
        filtered = bandpass(
            physical,
            source_fs,
            config["filters"]["pulse_band_hz"],
            int(config["filters"]["butterworth_order"]),
        )
        normalized = robust_standardize(filtered)
        resampled = signal.resample_poly(normalized, up=1, down=down)
        equal = adjacent_equal_fraction(physical, valid)
        output[role] = detect_pulse_representation(
            resampled,
            role,
            pulse_fs,
            float(1.0 - valid.mean()),
            float(equal) if math.isfinite(equal) else 1.0,
            config,
        )
    del memory
    return output

QUALITY_ORIENTATION = {
    "template_correlation_median": 1.0,
    "template_correlation_p10": 1.0,
    "template_dtw_distance_median": -1.0,
    "plateau_fraction": -1.0,
    "invalid_fraction": -1.0,
    "adjacent_equal_fraction": -1.0,
    "interval_coefficient_of_variation": -1.0,
    "prominence_coefficient_of_variation": -1.0,
    "spectral_entropy": -1.0,
    "spectral_peak_fraction": 1.0,
}

def fit_quality_transformer(
    extracted: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {"schema_version": "graft-quality-transformer-1.0.0", "roles": {}}
    bins = int(config["corruptions"]["cross_record_quality_bins"])
    for role in ("ppg", "abp"):
        representations = [
            row["representations"][role]
            for row in extracted.values()
            if role in row["representations"]
        ]
        if not representations:
            raise GraftSignalQualityError(f"No development representations for quality role {role}")
        parameters: dict[str, dict[str, float]] = {}
        components: list[np.ndarray] = []
        for name, orientation in QUALITY_ORIENTATION.items():
            values = np.asarray([rep.quality[name] for rep in representations], dtype=np.float64)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median))) * 1.4826
            scale = mad if mad > 1e-8 else max(float(np.std(values)), 1e-8)
            parameters[name] = {"median": median, "scale": scale, "orientation": orientation}
            components.append(orientation * (values - median) / scale)
        scores = np.mean(np.stack(components), axis=0)
        cutpoints = [float(np.quantile(scores, index / bins)) for index in range(1, bins)]
        output["roles"][role] = {"parameters": parameters, "cutpoints": cutpoints}
    return output

def quality_score_and_bin(
    representation: ModalityRepresentation,
    transformer: Mapping[str, Any],
) -> tuple[float, int]:
    role = transformer["roles"][representation.role]
    components = []
    for name, parameters in role["parameters"].items():
        components.append(
            float(parameters["orientation"])
            * (representation.quality[name] - float(parameters["median"]))
            / float(parameters["scale"])
        )
    score = float(np.mean(components))
    quality_bin = int(np.searchsorted(np.asarray(role["cutpoints"]), score, side="right"))
    return score, quality_bin

def annotate_quality(
    extracted: Mapping[str, dict[str, Any]], transformer: Mapping[str, Any]
) -> None:
    for row in extracted.values():
        row["quality"] = {}
        for role in ("ppg", "abp"):
            if role in row["representations"]:
                score, quality_bin = quality_score_and_bin(row["representations"][role], transformer)
                row["quality"][role] = {"score": score, "bin": quality_bin}

def choose_donor(
    anchor: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    role: str,
    maximum_rate_difference: float,
    seed: int,
) -> Mapping[str, Any] | None:
    target_rate = anchor["representations"][role].peak_train.rate_bpm
    target_bin = anchor["quality"][role]["bin"]
    eligible = [
        row
        for row in candidates
        if row["record"] != anchor["record"]
        and row["label"] == anchor["label"]
        and row["partition"] == anchor["partition"]
        and row["fold"] == anchor["fold"]
        and role in row["representations"]
        and row["quality"][role]["bin"] == target_bin
        and abs(row["representations"][role].peak_train.rate_bpm - target_rate)
        <= maximum_rate_difference
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda row: (
            abs(row["representations"][role].peak_train.rate_bpm - target_rate),
            abs(row["quality"][role]["score"] - anchor["quality"][role]["score"]),
            stable_seed(seed, anchor["event"], row["event"], role),
        )
    )
    return eligible[0]

def build_pair_rows(
    extracted: Mapping[str, dict[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    events = list(extracted.values())
    duration = (
        int(config["window"]["end_sample_exclusive"])
        - int(config["window"]["start_sample"])
    ) / float(config["window"]["sampling_frequency_hz"])
    minimum_peaks = int(config["event_detection"]["minimum_peaks"])
    base_seed = int(config["seed"])
    rows: list[dict[str, Any]] = []
    for task, task_config in config["pair_tasks"].items():
        anchor_role = task_config["anchor"]
        target_role = task_config["target"]
        eligible = [
            row
            for row in events
            if anchor_role in row["representations"]
            and target_role in row["representations"]
            and row["representations"][anchor_role].peak_train.count >= minimum_peaks
            and row["representations"][target_role].peak_train.count >= minimum_peaks
        ]
        for event in eligible:
            anchor_rep = event["representations"][anchor_role]
            target_rep = event["representations"][target_role]

            def append_row(
                family: str,
                severity: float,
                target: ModalityRepresentation,
                donor_event: str = "",
            ) -> None:
                coupling, quality = pair_feature_blocks(anchor_rep, target, task_config, config)
                row = {
                    "task": task,
                    "event": event["event"],
                    "record": event["record"],
                    "partition": event["partition"],
                    "fold": int(event["fold"]),
                    "corruption_family": family,
                    "severity": float(severity),
                    "donor_event": donor_event,
                    "is_real_pair": 1 if family == "real" else 0,
                    **coupling,
                    **quality,
                }
                rows.append(row)

            append_row("real", 0.0, target_rep)
            for severity in config["corruptions"]["jitter_seconds"]:
                append_row(
                    "jitter",
                    float(severity),
                    timing_corruption(
                        target_rep,
                        "jitter",
                        float(severity),
                        duration,
                        stable_seed(base_seed, task, event["event"], "jitter", str(severity)),
                        config,
                    ),
                )
            shifts = [float(value) for value in config["corruptions"]["circular_shift_seconds"]]
            shift = shifts[stable_seed(base_seed, task, event["event"], "shift") % len(shifts)]
            append_row(
                "circular_shift",
                shift,
                timing_corruption(
                    target_rep,
                    "circular_shift",
                    shift,
                    duration,
                    stable_seed(base_seed, task, event["event"], "circular_shift"),
                    config,
                ),
            )
            for severity in config["corruptions"]["lag_drift_end_to_end_seconds"]:
                append_row(
                    "lag_drift",
                    float(severity),
                    timing_corruption(
                        target_rep,
                        "lag_drift",
                        float(severity),
                        duration,
                        stable_seed(base_seed, task, event["event"], "lag_drift", str(severity)),
                        config,
                    ),
                )
            append_row(
                "beat_edit",
                float(config["corruptions"]["beat_delete_fraction"]),
                timing_corruption(
                    target_rep,
                    "beat_edit",
                    float(config["corruptions"]["beat_delete_fraction"]),
                    duration,
                    stable_seed(base_seed, task, event["event"], "beat_edit"),
                    config,
                ),
            )
            donor = choose_donor(
                event,
                eligible,
                target_role,
                float(config["corruptions"]["cross_record_max_rate_difference_bpm"]),
                base_seed,
            )
            if donor is not None:
                append_row(
                    "cross_record",
                    1.0,
                    donor["representations"][target_role],
                    donor["event"],
                )
            for family in ("flatline", "clipping", "band_limited_noise", "morphology_reverse"):
                append_row(
                    family,
                    1.0,
                    waveform_corruption(
                        target_rep,
                        family,
                        stable_seed(base_seed, task, event["event"], family),
                        config,
                    ),
                )
    if not rows:
        raise GraftSignalQualityError("No quality pair rows were constructed")
    return rows

def feature_sets(config: Mapping[str, Any]) -> dict[str, list[str]]:
    coupling = list(config["coupling_features"])
    quality = [
        f"{prefix}_{name}"
        for prefix in ("anchor", "target")
        for name in config["quality_features"]
    ]
    marginal_count_rate = list(config["marginal_count_rate_features"])
    if not marginal_count_rate or not set(marginal_count_rate).issubset(coupling):
        raise GraftSignalQualityError("Marginal count/rate features must be a non-empty coupling-feature subset")
    marginal = marginal_count_rate + quality
    return {
        "full": coupling + quality,
        "marginal": marginal,
        "quality": quality,
        "coupling": coupling,
    }
