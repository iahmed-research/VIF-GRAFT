from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
from dataclasses import dataclass, asdict, replace
import json, math, hashlib
import numpy as np
from scipy import signal
from .quality_pairs import annotate_quality,build_pair_rows,feature_sets,fit_quality_transformer
from .quality_fusion import supported_credibility_tasks
@dataclass(frozen=True)
class PairMaterialization:
    partition_role: str
    allowed_folds: tuple[int,...]
    feature_names: Mapping[str,tuple[str,...]]
    rows: tuple[Mapping[str,Any],...]
    audit: Mapping[str,Any]


SCHEMA_VERSION = "graft-pair-materialization-1.0.0"

TASKS = ("EP", "EA", "PA")

TASK_FEATURE_KIND = {"EP": "marginal", "EA": "marginal", "PA": "full"}

PAIR_FIELDS = (
    "task",
    "record",
    "event",
    "partition_role",
    "source_fold",
    "is_real_pair",
    "corruption_family",
    "severity",
    "donor_event",
)

OUTCOME_FIELDS = {
    "label",
    "labels",
    "outcome",
    "outcomes",
    "target",
    "targets",
    "y",
    "y_true",
    "development_label",
    "validation_label",
    "test_label",
}

class GraftPairMaterializationError(RuntimeError):
    """Raised when pair generation or persistence violates an information boundary."""

def m2_fold_roles(outer_fold: int) -> dict[str, tuple[int, ...]]:
    if not isinstance(outer_fold, int) or isinstance(outer_fold, bool) or not 0 <= outer_fold < 5:
        raise GraftPairMaterializationError("outer_fold must be an integer in {0,1,2,3,4}")
    calibration = (outer_fold + 1) % 5
    fit = tuple(sorted(set(range(5)) - {outer_fold, calibration}))
    return {"fit": fit, "calibration": (calibration,), "evaluation": (outer_fold,)}

def _feature_names(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    sets = feature_sets(config)
    return {
        task: tuple(str(value) for value in sets[TASK_FEATURE_KIND[task]]) for task in TASKS
    }

def _required_non_cross_counts(config: Mapping[str, Any]) -> dict[str, int]:
    return {
        "real": 1,
        "jitter": len(config["corruptions"]["jitter_seconds"]),
        "circular_shift": 1,
        "lag_drift": len(config["corruptions"]["lag_drift_end_to_end_seconds"]),
        "beat_edit": 1,
        "flatline": 1,
        "clipping": 1,
        "band_limited_noise": 1,
        "morphology_reverse": 1,
    }

def _eligible_real_grid(
    extracted: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]
) -> set[tuple[str, str]]:
    grid: set[tuple[str, str]] = set()
    for event, row in extracted.items():
        representations = row["representations"]
        has_ppg = "ppg" in representations
        has_abp = "abp" in representations
        availability = (
            "both" if has_ppg and has_abp
            else "ppg_only" if has_ppg
            else "abp_only" if has_abp
            else "no_usable_pulse"
        )
        for task in supported_credibility_tasks(availability, representations, config):
            grid.add((str(event), task))
    return grid

def project_and_audit_pair_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    extracted: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    partition_role: str,
    allowed_folds: Sequence[int],
) -> PairMaterialization:
    """Remove clinical labels and verify all donor and corruption contracts."""

    if partition_role not in {"fit", "calibration"}:
        raise GraftPairMaterializationError("Pair projection role must be fit or calibration")
    folds = tuple(sorted(set(int(value) for value in allowed_folds)))
    if not folds or any(fold not in range(5) for fold in folds):
        raise GraftPairMaterializationError("Allowed pair folds are invalid")
    if int(config.get("seed", -1)) != 20260828:
        raise GraftPairMaterializationError("GRAFT credibility must retain frozen M1-R seed 20260828")
    features = _feature_names(config)
    event_metadata: dict[str, Mapping[str, Any]] = {}
    record_fold: dict[str, int] = {}
    for event, row in extracted.items():
        if str(row.get("event")) != str(event) or not str(row.get("record", "")):
            raise GraftPairMaterializationError("Extracted event identity is inconsistent")
        if str(row.get("partition")) != partition_role or int(row.get("fold", -1)) not in folds:
            raise GraftPairMaterializationError("Extracted event crossed its permitted fold role")
        if int(row.get("label", -1)) not in {0, 1}:
            raise GraftPairMaterializationError("Donor matching requires a binary development label")
        record = str(row["record"])
        fold = int(row["fold"])
        if record in record_fold and record_fold[record] != fold:
            raise GraftPairMaterializationError("One record group crosses GRAFT folds")
        record_fold[record] = fold
        event_metadata[str(event)] = row

    expected_real = _eligible_real_grid(extracted, config)
    allowed_families = set(_required_non_cross_counts(config)) | {"cross_record"}
    counts: dict[tuple[str, str, str], int] = {}
    projected: list[dict[str, Any]] = []
    cross_record_rows = 0
    for raw in raw_rows:
        fields = {str(field).lower() for field in raw}
        if fields & OUTCOME_FIELDS or any(field.endswith("_label") for field in fields):
            raise GraftPairMaterializationError("Raw pair row unexpectedly contains a clinical outcome")
        task = str(raw.get("task", ""))
        event = str(raw.get("event", ""))
        record = str(raw.get("record", ""))
        family = str(raw.get("corruption_family", ""))
        if task not in TASKS or event not in event_metadata:
            raise GraftPairMaterializationError("Raw pair row has an unknown task or event")
        if (event, task) not in expected_real:
            raise GraftPairMaterializationError("Raw pair row was generated for an ineligible task")
        metadata = event_metadata[event]
        if record != str(metadata["record"]) or int(raw.get("fold", -1)) != int(metadata["fold"]):
            raise GraftPairMaterializationError("Raw pair row provenance differs from extracted event")
        if str(raw.get("partition")) != partition_role:
            raise GraftPairMaterializationError("Raw pair row crossed its information role")
        pair_label = int(raw.get("is_real_pair", -1))
        if family not in allowed_families:
            raise GraftPairMaterializationError("Raw pair row has an undeclared corruption family")
        if pair_label not in {0, 1} or (pair_label == 1) != (family == "real"):
            raise GraftPairMaterializationError("Raw pair family and artificial label disagree")
        donor_event = str(raw.get("donor_event", ""))
        if family == "cross_record":
            cross_record_rows += 1
            if donor_event not in event_metadata:
                raise GraftPairMaterializationError("Cross-record donor is outside the permitted role")
            donor = event_metadata[donor_event]
            target_role = str(config["pair_tasks"][task]["target"])
            minimum_peaks = int(config["event_detection"]["minimum_peaks"])
            donor_representation = donor["representations"].get(target_role)
            donor_quality = donor.get("quality", {}).get(target_role)
            recipient_quality = metadata.get("quality", {}).get(target_role)
            if (
                str(donor["record"]) == record
                or int(donor["fold"]) != int(metadata["fold"])
                or int(donor["label"]) != int(metadata["label"])
                or str(donor["partition"]) != partition_role
                or donor_representation is None
                or donor_representation.peak_train.count < minimum_peaks
                or donor_quality is None
                or recipient_quality is None
                or int(donor_quality["bin"]) != int(recipient_quality["bin"])
                or abs(
                    float(donor_representation.peak_train.rate_bpm)
                    - float(metadata["representations"][target_role].peak_train.rate_bpm)
                )
                > float(config["corruptions"]["cross_record_max_rate_difference_bpm"])
            ):
                raise GraftPairMaterializationError(
                    "Cross-record donor violates record, fold, role, label, quality, or rate matching"
                )
        elif donor_event:
            raise GraftPairMaterializationError("Only cross-record corruptions may name a donor")
        severity = float(raw.get("severity", 0.0))
        if not math.isfinite(severity):
            raise GraftPairMaterializationError("Raw pair severity must be finite")
        row = {
            "task": task,
            "record": record,
            "event": event,
            "partition_role": partition_role,
            "source_fold": int(metadata["fold"]),
            "is_real_pair": pair_label,
            "corruption_family": family,
            "severity": severity,
            "donor_event": donor_event,
        }
        for name in features[task]:
            try:
                row[name] = float(raw[name])
            except (KeyError, TypeError, ValueError) as error:
                raise GraftPairMaterializationError(
                    f"Raw {task} pair lacks numeric feature {name!r}"
                ) from error
            if not math.isfinite(row[name]):
                raise GraftPairMaterializationError(f"Raw {task} feature {name!r} is non-finite")
        projected.append(row)
        key = (event, task, family)
        counts[key] = counts.get(key, 0) + 1

    observed_real = {
        (event, task) for (event, task, family), count in counts.items() if family == "real" and count == 1
    }
    if observed_real != expected_real:
        raise GraftPairMaterializationError("Real-pair task grid differs from eligible extracted events")
    required_counts = _required_non_cross_counts(config)
    for event, task in expected_real:
        for family, expected in required_counts.items():
            if counts.get((event, task, family), 0) != expected:
                raise GraftPairMaterializationError(
                    f"{event}/{task} has wrong {family} corruption multiplicity"
                )
        if counts.get((event, task, "cross_record"), 0) > 1:
            raise GraftPairMaterializationError(
                f"{event}/{task} has duplicate cross-record null rows"
            )
    if not projected:
        raise GraftPairMaterializationError("No credibility pair rows were projected")
    projected.sort(
        key=lambda row: (
            row["task"], row["record"], row["event"], row["corruption_family"],
            row["severity"], row["donor_event"],
        )
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "partition_role": partition_role,
        "allowed_folds": list(folds),
        "input_events": len(extracted),
        "input_record_groups": len(record_fold),
        "eligible_real_event_task_pairs": len(expected_real),
        "projected_pair_rows": len(projected),
        "cross_record_pair_rows": cross_record_rows,
        "clinical_labels_used_only_for_cross_record_donor_matching": True,
        "clinical_outcome_fields_written": False,
        "donor_same_record_count": 0,
        "donor_cross_fold_count": 0,
        "donor_cross_role_count": 0,
        "donor_label_mismatch_count": 0,
    }
    return PairMaterialization(partition_role, folds, features, tuple(projected), audit)

def materialize_fold_role_pairs(
    extracted: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    partition_role: str,
    allowed_folds: Sequence[int],
) -> PairMaterialization:
    """Generate pair rows within one role; no evaluation rows are accepted."""

    local: dict[str, dict[str, Any]] = {}
    for event, source in extracted.items():
        if str(source.get("partition")) != partition_role:
            raise GraftPairMaterializationError(
                "Input event has the wrong information role; relabelling is prohibited"
            )
        if int(source.get("fold", -1)) not in {int(value) for value in allowed_folds}:
            raise GraftPairMaterializationError("Input event is outside the allowed folds")
        row = dict(source)
        row["event"] = str(event)
        row["quality"] = {}
        local[str(event)] = row
    transformer = fit_quality_transformer(local, config)
    annotate_quality(local, transformer)
    raw_rows = build_pair_rows(local, config)
    return project_and_audit_pair_rows(
        raw_rows,
        local,
        config,
        partition_role=partition_role,
        allowed_folds=allowed_folds,
    )
