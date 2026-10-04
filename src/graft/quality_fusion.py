from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
from dataclasses import dataclass, asdict, replace
import json, math, hashlib
import numpy as np
from scipy import signal
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .quality_core import family_record_weights
from .io import write as atomic_json,canonical as canonical_sha256
from .calibration import NonnegativeLogisticCalibrator,fit_nonnegative_logistic_calibrator
from .quality_null import GraftCredibilityError,RecordBalancedNullCDF


SCHEMA_VERSION = "graft-fold-local-credibility-1.0.0"

TASKS = ("EP", "EA", "PA")

TASK_FEATURE_KIND = {"EP": "marginal", "EA": "marginal", "PA": "full"}

AVAILABILITY_TASKS = {
    "both": ("EP", "EA", "PA"),
    "ppg_only": ("EP",),
    "abp_only": ("EA",),
    "no_usable_pulse": (),
}

UNSUPPORTED_EXPECTED_TASK_POLICY = (
    "maximum_entropy_neutral_marginal_0_5_and_zero_dependence"
)

NEUTRAL_MARGINAL_CREDIBILITY = 0.5

CLINICAL_OUTCOME_FIELDS = {
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
    "is_true_alarm",
    "true_alarm",
    "alarm_truth",
    "ground_truth",
    "annotation",
}

class GraftCredibilityPipelineError(RuntimeError):
    """Raised before a fold-local information or reproducibility rule is violated."""

def supported_credibility_tasks(
    availability_state: str,
    representations: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return structurally computable tasks without inspecting any outcome.

    Sensor availability and credibility estimability are deliberately separate:
    a pulse remains available to every outcome model when its frozen
    ECG-anchored credibility task is unsupported. A task is supported only when
    both frozen roles exist and both peak trains meet the frozen minimum.
    """

    expected = AVAILABILITY_TASKS.get(str(availability_state))
    if expected is None:
        raise GraftCredibilityPipelineError("Unknown modality availability state")
    try:
        task_config = config["pair_tasks"]
        minimum_peaks = int(config["event_detection"]["minimum_peaks"])
    except (KeyError, TypeError, ValueError) as error:
        raise GraftCredibilityPipelineError("Credibility support configuration is invalid") from error
    if minimum_peaks <= 0 or set(task_config) != set(TASKS):
        raise GraftCredibilityPipelineError("Credibility support configuration differs from schema")
    supported: list[str] = []
    for task in expected:
        anchor = str(task_config[task]["anchor"])
        target = str(task_config[task]["target"])
        if anchor not in representations or target not in representations:
            continue
        try:
            counts = (
                int(representations[anchor].peak_train.count),
                int(representations[target].peak_train.count),
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise GraftCredibilityPipelineError(
                "Credibility representation peak-train contract differs"
            ) from error
        if min(counts) >= minimum_peaks:
            supported.append(task)
    return tuple(supported)

def _finite_tuple(name: str, values: Sequence[float], length: int | None = None) -> tuple[float, ...]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise GraftCredibilityPipelineError(f"{name} must be a finite numeric vector") from error
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise GraftCredibilityPipelineError(f"{name} must be a nonempty finite vector")
    if length is not None and len(array) != length:
        raise GraftCredibilityPipelineError(f"{name} has the wrong length")
    return tuple(float(value) for value in array)

def _reject_clinical_outcomes(row: Mapping[str, Any]) -> None:
    fields = {str(field).lower() for field in row}
    forbidden = fields & CLINICAL_OUTCOME_FIELDS
    forbidden.update(
        field for field in fields if field.endswith("_label") and field != "is_real_pair"
    )
    forbidden.update(
        field
        for field in fields
        if any(
            token in field
            for token in ("ground_truth", "true_alarm", "alarm_truth", "annotation")
        )
    )
    if forbidden:
        raise GraftCredibilityPipelineError(
            f"Credibility row contains prohibited clinical outcome field(s): {sorted(forbidden)}"
        )

def _validate_features_by_task(
    features_by_task: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    if set(features_by_task) != set(TASKS):
        raise GraftCredibilityPipelineError("features_by_task must define exactly EP, EA, and PA")
    normalized: dict[str, tuple[str, ...]] = {}
    for task in TASKS:
        names = tuple(str(name) for name in features_by_task[task])
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise GraftCredibilityPipelineError(f"{task} feature names must be unique and nonempty")
        if any(
            name.lower() in CLINICAL_OUTCOME_FIELDS
            or name.lower().endswith("_label")
            or any(
                token in name.lower()
                for token in ("ground_truth", "true_alarm", "alarm_truth", "annotation")
            )
            for name in names
        ):
            raise GraftCredibilityPipelineError(f"{task} feature set contains an outcome")
        normalized[task] = names
    return normalized

def _pair_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row["task"]),
        str(row["record"]),
        str(row["event"]),
        str(row["source_fold"]),
        str(row["corruption_family"]),
        format(float(row.get("severity", 0.0)), ".17g"),
        str(row.get("donor_event", "")),
    )

def _validate_training_pair_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_role: str,
    features_by_task: Mapping[str, tuple[str, ...]],
) -> dict[str, list[Mapping[str, Any]]]:
    if expected_role not in {"fit", "calibration"}:
        raise GraftCredibilityPipelineError("Unknown pair-row information role")
    if not rows:
        raise GraftCredibilityPipelineError(f"{expected_role} pair rows cannot be empty")
    by_task: dict[str, list[Mapping[str, Any]]] = {task: [] for task in TASKS}
    identities: set[tuple[str, ...]] = set()
    event_records: dict[str, str] = {}
    for row in rows:
        _reject_clinical_outcomes(row)
        required = {
            "task",
            "record",
            "event",
            "partition_role",
            "source_fold",
            "is_real_pair",
            "corruption_family",
            "severity",
            "donor_event",
        }
        task_value = str(row.get("task", ""))
        expected_fields = required | set(features_by_task.get(task_value, ()))
        if set(row) != expected_fields:
            raise GraftCredibilityPipelineError(
                "Pair row must have the exact task-specific feature and provenance fields"
            )
        task = str(row["task"])
        record = str(row["record"])
        event = str(row["event"])
        if task not in TASKS or not record or not event:
            raise GraftCredibilityPipelineError("Pair task, record, and event must be valid")
        if str(row["partition_role"]) != expected_role:
            raise GraftCredibilityPipelineError(
                f"{expected_role} pair table contains another information role"
            )
        try:
            source_fold = int(row["source_fold"])
        except (TypeError, ValueError) as error:
            raise GraftCredibilityPipelineError("Pair source_fold must be an integer") from error
        if isinstance(row["source_fold"], bool) or not 0 <= source_fold < 5:
            raise GraftCredibilityPipelineError("Pair source_fold must lie in {0,1,2,3,4}")
        previous_record = event_records.setdefault(event, record)
        if previous_record != record:
            raise GraftCredibilityPipelineError("One event maps to more than one record group")
        try:
            pair_label = int(row["is_real_pair"])
        except (TypeError, ValueError) as error:
            raise GraftCredibilityPipelineError("is_real_pair must be binary") from error
        family = str(row["corruption_family"])
        if pair_label not in {0, 1} or (pair_label == 1) != (family == "real"):
            raise GraftCredibilityPipelineError("Pair label and corruption family are inconsistent")
        identity = _pair_identity(row)
        if identity in identities:
            raise GraftCredibilityPipelineError("Duplicate pair identity would change experiment weights")
        identities.add(identity)
        for feature in features_by_task[task]:
            try:
                value = float(row[feature])
            except (KeyError, TypeError, ValueError) as error:
                raise GraftCredibilityPipelineError(
                    f"{task} row lacks finite feature {feature!r}"
                ) from error
            if not math.isfinite(value):
                raise GraftCredibilityPipelineError(f"{task} feature {feature!r} is non-finite")
        by_task[task].append(row)
    for task, task_rows in by_task.items():
        if not task_rows:
            raise GraftCredibilityPipelineError(f"{expected_role} contains no {task} pair rows")
        labels = {int(row["is_real_pair"]) for row in task_rows}
        if labels != {0, 1}:
            raise GraftCredibilityPipelineError(f"{expected_role} {task} task lacks a pair class")
        per_record: dict[str, set[int]] = {}
        for row in task_rows:
            per_record.setdefault(str(row["record"]), set()).add(int(row["is_real_pair"]))
        incomplete = sorted(record for record, observed in per_record.items() if observed != {0, 1})
        if incomplete:
            raise GraftCredibilityPipelineError(
                f"{expected_role} {task} record(s) lack real or corrupt rows: {incomplete[:5]}"
            )
    return by_task

def _rows_digest(
    rows: Sequence[Mapping[str, Any]], features_by_task: Mapping[str, tuple[str, ...]]
) -> str:
    normalized = []
    for row in rows:
        task = str(row["task"])
        normalized.append(
            {
                "identity": _pair_identity(row),
                "partition_role": str(row["partition_role"]),
                "is_real_pair": int(row["is_real_pair"]),
                "features": [format(float(row[name]), ".17g") for name in features_by_task[task]],
            }
        )
    normalized.sort(key=lambda item: item["identity"])
    return canonical_sha256(normalized)

@dataclass(frozen=True)
class FrozenPairLogisticModel:
    task: str
    feature_kind: str
    feature_names: tuple[str, ...]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    logistic_c: float
    maximum_iterations: int
    random_seed: int

    def __post_init__(self) -> None:
        if self.task not in TASKS or self.feature_kind != TASK_FEATURE_KIND[self.task]:
            raise GraftCredibilityPipelineError("Pair model task or feature-kind binding is invalid")
        dimension = len(self.feature_names)
        if not dimension or len(set(self.feature_names)) != dimension:
            raise GraftCredibilityPipelineError("Pair model feature names are invalid")
        _finite_tuple("scaler_mean", self.scaler_mean, dimension)
        scale = _finite_tuple("scaler_scale", self.scaler_scale, dimension)
        _finite_tuple("coefficients", self.coefficients, dimension)
        if any(value <= 0.0 for value in scale):
            raise GraftCredibilityPipelineError("Pair model scaler scales must be positive")
        if not math.isfinite(self.intercept) or not math.isfinite(self.logistic_c):
            raise GraftCredibilityPipelineError("Pair model parameters must be finite")
        if self.logistic_c <= 0.0 or self.maximum_iterations <= 0 or self.random_seed < 0:
            raise GraftCredibilityPipelineError("Pair model fit controls are invalid")

    def logits(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        matrix = np.asarray(
            [[float(row[name]) for name in self.feature_names] for row in rows],
            dtype=np.float64,
        )
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise GraftCredibilityPipelineError("Pair-model application matrix has wrong shape")
        if not np.all(np.isfinite(matrix)):
            raise GraftCredibilityPipelineError("Pair-model application features must be finite")
        standardized = (matrix - np.asarray(self.scaler_mean)) / np.asarray(self.scaler_scale)
        return standardized @ np.asarray(self.coefficients) + self.intercept

    def to_artifact(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "feature_kind": self.feature_kind,
            "feature_names": list(self.feature_names),
            "scaler_mean": list(self.scaler_mean),
            "scaler_scale": list(self.scaler_scale),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "logistic_c": self.logistic_c,
            "maximum_iterations": self.maximum_iterations,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> "FrozenPairLogisticModel":
        fields = {
            "task",
            "feature_kind",
            "feature_names",
            "scaler_mean",
            "scaler_scale",
            "coefficients",
            "intercept",
            "logistic_c",
            "maximum_iterations",
            "random_seed",
        }
        if set(artifact) != fields:
            raise GraftCredibilityPipelineError("Pair model artifact fields differ from schema")
        return cls(
            task=str(artifact["task"]),
            feature_kind=str(artifact["feature_kind"]),
            feature_names=tuple(str(value) for value in artifact["feature_names"]),
            scaler_mean=_finite_tuple("scaler_mean", artifact["scaler_mean"]),
            scaler_scale=_finite_tuple("scaler_scale", artifact["scaler_scale"]),
            coefficients=_finite_tuple("coefficients", artifact["coefficients"]),
            intercept=float(artifact["intercept"]),
            logistic_c=float(artifact["logistic_c"]),
            maximum_iterations=int(artifact["maximum_iterations"]),
            random_seed=int(artifact["random_seed"]),
        )

def _fit_pair_model(
    task: str,
    rows: Sequence[Mapping[str, Any]],
    features: tuple[str, ...],
    *,
    logistic_c: float,
    maximum_iterations: int,
    random_seed: int,
) -> tuple[FrozenPairLogisticModel, np.ndarray]:
    matrix = np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float64)
    labels = np.asarray([int(row["is_real_pair"]) for row in rows], dtype=np.int8)
    try:
        weights = np.asarray(family_record_weights(rows), dtype=np.float64)
    except Exception as error:
        raise GraftCredibilityPipelineError(f"{task} record-family weights failed: {error}") from error
    if not np.all(weights > 0.0):
        raise GraftCredibilityPipelineError(f"{task} contains a zero-weight pair row")
    scaler = StandardScaler()
    standardized = scaler.fit_transform(matrix)
    estimator = LogisticRegression(
        C=logistic_c,
        max_iter=maximum_iterations,
        random_state=random_seed,
        solver="lbfgs",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            estimator.fit(standardized, labels, sample_weight=weights)
        except ConvergenceWarning as error:
            raise GraftCredibilityPipelineError(f"{task} logistic model did not converge") from error
    model = FrozenPairLogisticModel(
        task=task,
        feature_kind=TASK_FEATURE_KIND[task],
        feature_names=features,
        scaler_mean=tuple(float(value) for value in scaler.mean_),
        scaler_scale=tuple(float(value) for value in scaler.scale_),
        coefficients=tuple(float(value) for value in estimator.coef_[0]),
        intercept=float(estimator.intercept_[0]),
        logistic_c=float(logistic_c),
        maximum_iterations=int(maximum_iterations),
        random_seed=int(random_seed),
    )
    return model, weights

def _calibrator_artifact(calibrator: NonnegativeLogisticCalibrator) -> dict[str, Any]:
    return {
        "intercept": calibrator.intercept,
        "slope": calibrator.slope,
        "input_kind": calibrator.input_kind,
        "probability_clip": calibrator.probability_clip,
    }

def _calibrator_from_artifact(artifact: Mapping[str, Any]) -> NonnegativeLogisticCalibrator:
    if set(artifact) != {"intercept", "slope", "input_kind", "probability_clip"}:
        raise GraftCredibilityPipelineError("Credibility calibrator artifact fields differ from schema")
    calibrator = NonnegativeLogisticCalibrator(
        intercept=float(artifact["intercept"]),
        slope=float(artifact["slope"]),
        input_kind=str(artifact["input_kind"]),
        probability_clip=float(artifact["probability_clip"]),
    )
    if (
        calibrator.input_kind != "logit"
        or not math.isfinite(calibrator.intercept)
        or not math.isfinite(calibrator.slope)
        or calibrator.slope < 0.0
        or not 0.0 < calibrator.probability_clip < 0.5
    ):
        raise GraftCredibilityPipelineError("Credibility calibrator artifact is invalid")
    return calibrator

@dataclass(frozen=True)
class FoldLocalCredibilityBundle:
    immutable_context: Mapping[str, Any]
    models: Mapping[str, FrozenPairLogisticModel]
    calibrators: Mapping[str, NonnegativeLogisticCalibrator]
    pa_null: RecordBalancedNullCDF
    fit_pair_rows_sha256: str
    calibration_pair_rows_sha256: str
    fit_record_set_sha256: str
    calibration_record_set_sha256: str
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.immutable_context or "schema_version" in self.immutable_context:
            raise GraftCredibilityPipelineError(
                "Credibility immutable context cannot be empty or override schema_version"
            )
        if set(self.models) != set(TASKS) or set(self.calibrators) != set(TASKS):
            raise GraftCredibilityPipelineError("Credibility bundle must contain EP, EA, and PA")
        for task in TASKS:
            if self.models[task].task != task or self.calibrators[task].input_kind != "logit":
                raise GraftCredibilityPipelineError("Credibility bundle task binding is invalid")
        for digest in (
            self.fit_pair_rows_sha256,
            self.calibration_pair_rows_sha256,
            self.fit_record_set_sha256,
            self.calibration_record_set_sha256,
        ):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise GraftCredibilityPipelineError("Credibility bundle hashes must be lowercase SHA-256")

    @property
    def immutable_context_sha256(self) -> str:
        return canonical_sha256({"schema_version": SCHEMA_VERSION, **dict(self.immutable_context)})

    def transform(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        expected_event_records: Mapping[str, str],
        availability_by_event: Mapping[str, str],
        supported_tasks_by_event: Mapping[str, Sequence[str]] | None = None,
    ) -> list[dict[str, Any]]:
        if set(availability_by_event) != set(expected_event_records):
            raise GraftCredibilityPipelineError("Availability map must exactly cover expected events")
        expected_tasks: dict[str, tuple[str, ...]] = {}
        for event, availability in availability_by_event.items():
            tasks = AVAILABILITY_TASKS.get(availability)
            if tasks is None:
                raise GraftCredibilityPipelineError("Unknown modality availability state")
            expected_tasks[event] = tasks
        if supported_tasks_by_event is None:
            supported = dict(expected_tasks)
        else:
            if set(supported_tasks_by_event) != set(expected_event_records):
                raise GraftCredibilityPipelineError(
                    "Credibility support map must exactly cover expected events"
                )
            supported = {}
            for event in expected_event_records:
                raw_tasks = tuple(str(task) for task in supported_tasks_by_event[event])
                if len(raw_tasks) != len(set(raw_tasks)):
                    raise GraftCredibilityPipelineError("Credibility support map contains duplicates")
                if any(task not in expected_tasks[event] for task in raw_tasks):
                    raise GraftCredibilityPipelineError(
                        "Credibility support exceeds modality availability"
                    )
                supported[event] = tuple(
                    task for task in expected_tasks[event] if task in set(raw_tasks)
                )
        expected_grid = {
            (event, task) for event, tasks in supported.items() for task in tasks
        }
        observed: dict[tuple[str, str], Mapping[str, Any]] = {}
        for row in rows:
            _reject_clinical_outcomes(row)
            if "is_real_pair" in row or "corruption_family" in row:
                raise GraftCredibilityPipelineError(
                    "Application rows cannot carry pair-task labels or corruption identities"
                )
            required = {"task", "record", "event", "partition_role"}
            task_value = str(row.get("task", ""))
            expected_fields = required | set(
                self.models[task_value].feature_names if task_value in self.models else ()
            )
            if set(row) != expected_fields or str(row.get("partition_role", "")) != "application":
                raise GraftCredibilityPipelineError("Application row provenance is incomplete")
            event = str(row["event"])
            task = str(row["task"])
            record = str(row["record"])
            if event not in expected_event_records or record != expected_event_records[event]:
                raise GraftCredibilityPipelineError("Application event-to-record mapping differs")
            if task not in TASKS:
                raise GraftCredibilityPipelineError("Unknown application credibility task")
            key = (event, task)
            if key in observed:
                raise GraftCredibilityPipelineError("Duplicate application event-task row")
            for feature in self.models[task].feature_names:
                try:
                    value = float(row[feature])
                except (KeyError, TypeError, ValueError) as error:
                    raise GraftCredibilityPipelineError(
                        f"Application {task} row lacks finite feature {feature!r}"
                    ) from error
                if not math.isfinite(value):
                    raise GraftCredibilityPipelineError("Application credibility feature is non-finite")
            observed[key] = row
        if set(observed) != expected_grid:
            raise GraftCredibilityPipelineError("Application event-task grid is incomplete or unexpected")

        outputs: list[dict[str, Any]] = []
        for event in sorted(expected_event_records):
            availability = availability_by_event[event]
            r_ppg = 0.0
            r_abp = 0.0
            eta: float | None = None
            s_pa: float | None = None
            if "EP" in AVAILABILITY_TASKS[availability]:
                if "EP" in supported[event]:
                    row = observed[(event, "EP")]
                    logit = self.models["EP"].logits([row])
                    r_ppg = float(self.calibrators["EP"].predict_from_logits(logit)[0])
                else:
                    r_ppg = NEUTRAL_MARGINAL_CREDIBILITY
            if "EA" in AVAILABILITY_TASKS[availability]:
                if "EA" in supported[event]:
                    row = observed[(event, "EA")]
                    logit = self.models["EA"].logits([row])
                    r_abp = float(self.calibrators["EA"].predict_from_logits(logit)[0])
                else:
                    r_abp = NEUTRAL_MARGINAL_CREDIBILITY
            if "PA" in AVAILABILITY_TASKS[availability]:
                if "PA" in supported[event]:
                    row = observed[(event, "PA")]
                    logit = self.models["PA"].logits([row])
                    s_pa = float(self.calibrators["PA"].predict_from_logits(logit)[0])
                    eta = float(self.pa_null.eta(s_pa))
                else:
                    eta = 0.0
            outputs.append(
                {
                    "record": expected_event_records[event],
                    "event": event,
                    "availability_state": availability,
                    "r_ppg": r_ppg,
                    "r_abp": r_abp,
                    "s_pa": s_pa,
                    "eta": eta,
                }
            )
        return outputs

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "immutable_context": dict(self.immutable_context),
            "immutable_context_sha256": self.immutable_context_sha256,
            "models": {task: self.models[task].to_artifact() for task in TASKS},
            "calibrators": {
                task: _calibrator_artifact(self.calibrators[task]) for task in TASKS
            },
            "pa_null": self.pa_null.to_artifact(),
            "fit_pair_rows_sha256": self.fit_pair_rows_sha256,
            "calibration_pair_rows_sha256": self.calibration_pair_rows_sha256,
            "fit_record_set_sha256": self.fit_record_set_sha256,
            "calibration_record_set_sha256": self.calibration_record_set_sha256,
            "audit": dict(self.audit),
        }

    def save(self, path: Path) -> dict[str, Any]:
        payload = self.payload()
        wrapper = {
            "schema_version": "graft-fold-local-credibility-artifact-1.0.0",
            "payload_sha256": canonical_sha256(payload),
            "payload": payload,
        }
        path = path.resolve()
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise GraftCredibilityPipelineError(f"Existing credibility artifact unreadable: {error}")
            if existing != wrapper:
                raise GraftCredibilityPipelineError("Refusing to replace a different credibility artifact")
            return wrapper
        atomic_json(path, wrapper)
        return wrapper

    @classmethod
    def load(
        cls, path: Path, *, expected_immutable_context: Mapping[str, Any]
    ) -> "FoldLocalCredibilityBundle":
        try:
            wrapper = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise GraftCredibilityPipelineError(f"Credibility artifact unreadable: {error}") from error
        if set(wrapper) != {"schema_version", "payload_sha256", "payload"}:
            raise GraftCredibilityPipelineError("Credibility wrapper fields differ from schema")
        if wrapper["schema_version"] != "graft-fold-local-credibility-artifact-1.0.0":
            raise GraftCredibilityPipelineError("Unknown credibility wrapper schema")
        payload = wrapper["payload"]
        if canonical_sha256(payload) != wrapper["payload_sha256"]:
            raise GraftCredibilityPipelineError("Credibility payload hash mismatch")
        fields = {
            "schema_version",
            "immutable_context",
            "immutable_context_sha256",
            "models",
            "calibrators",
            "pa_null",
            "fit_pair_rows_sha256",
            "calibration_pair_rows_sha256",
            "fit_record_set_sha256",
            "calibration_record_set_sha256",
            "audit",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise GraftCredibilityPipelineError("Credibility payload fields differ from schema")
        expected_context_hash = canonical_sha256(
            {"schema_version": SCHEMA_VERSION, **dict(expected_immutable_context)}
        )
        if payload["immutable_context"] != dict(expected_immutable_context):
            raise GraftCredibilityPipelineError("Credibility immutable context differs")
        if payload["immutable_context_sha256"] != expected_context_hash:
            raise GraftCredibilityPipelineError("Credibility immutable context hash differs")
        if payload["schema_version"] != SCHEMA_VERSION:
            raise GraftCredibilityPipelineError("Unknown credibility payload schema")
        if set(payload["models"]) != set(TASKS) or set(payload["calibrators"]) != set(TASKS):
            raise GraftCredibilityPipelineError("Credibility task artifact keys differ from schema")
        bundle = cls(
            immutable_context=payload["immutable_context"],
            models={
                task: FrozenPairLogisticModel.from_artifact(payload["models"][task])
                for task in TASKS
            },
            calibrators={
                task: _calibrator_from_artifact(payload["calibrators"][task])
                for task in TASKS
            },
            pa_null=RecordBalancedNullCDF.from_artifact(payload["pa_null"]),
            fit_pair_rows_sha256=str(payload["fit_pair_rows_sha256"]),
            calibration_pair_rows_sha256=str(payload["calibration_pair_rows_sha256"]),
            fit_record_set_sha256=str(payload["fit_record_set_sha256"]),
            calibration_record_set_sha256=str(payload["calibration_record_set_sha256"]),
            audit=payload["audit"],
        )
        expected_audit_fields = {
            "schema_version",
            "status",
            "clinical_outcomes_used_as_credibility_inputs",
            "outer_evaluation_rows_received_during_fit",
            "fit_calibration_record_overlap",
            "pair_weighting",
            "task_models",
            "pa_null",
        }
        if not isinstance(bundle.audit, dict) or set(bundle.audit) != expected_audit_fields:
            raise GraftCredibilityPipelineError("Credibility audit fields differ from schema")
        if (
            bundle.audit["schema_version"] != SCHEMA_VERSION
            or bundle.audit["status"] != "pass"
            or bundle.audit["clinical_outcomes_used_as_credibility_inputs"] is not False
            or bundle.audit["outer_evaluation_rows_received_during_fit"] is not False
            or bundle.audit["fit_calibration_record_overlap"] != 0
            or bundle.audit["pair_weighting"]
            != "equal_record_then_equal_real_vs_corrupt_then_equal_corruption_family"
            or set(bundle.audit["task_models"]) != set(TASKS)
            or bundle.audit["pa_null"] != bundle.pa_null.audit()
        ):
            raise GraftCredibilityPipelineError("Credibility audit content fails reconstruction checks")
        for task in TASKS:
            task_audit = bundle.audit["task_models"][task]
            if (
                task_audit.get("feature_kind") != TASK_FEATURE_KIND[task]
                or task_audit.get("feature_names") != list(bundle.models[task].feature_names)
                or task_audit.get("calibrator_slope") != bundle.calibrators[task].slope
            ):
                raise GraftCredibilityPipelineError(
                    f"Credibility {task} audit disagrees with reconstructed objects"
                )
        return bundle

def fit_fold_local_credibility(
    fit_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    *,
    features_by_task: Mapping[str, Sequence[str]],
    immutable_context: Mapping[str, Any],
    logistic_c: float = 1.0,
    maximum_iterations: int = 3000,
    random_seed: int = 20260828,
    minimum_pa_null_rows: int = 100,
    minimum_pa_null_groups: int = 50,
    maximum_pa_null_standard_error: float = 0.05,
    precision_probabilities: Sequence[float] = tuple(np.arange(0.1, 1.0, 0.1)),
) -> FoldLocalCredibilityBundle:
    """Fit all credibility objects without receiving any outer-evaluation rows."""

    if not immutable_context or "schema_version" in immutable_context:
        raise GraftCredibilityPipelineError(
            "immutable_context must be nonempty and cannot override schema_version"
        )
    if not math.isfinite(logistic_c) or logistic_c <= 0.0:
        raise GraftCredibilityPipelineError("logistic_c must be positive and finite")
    if maximum_iterations <= 0 or random_seed < 0:
        raise GraftCredibilityPipelineError("Logistic fit controls are invalid")
    features = _validate_features_by_task(features_by_task)
    fit_by_task = _validate_training_pair_rows(
        fit_rows, expected_role="fit", features_by_task=features
    )
    calibration_by_task = _validate_training_pair_rows(
        calibration_rows, expected_role="calibration", features_by_task=features
    )
    fit_records = {str(row["record"]) for row in fit_rows}
    calibration_records = {str(row["record"]) for row in calibration_rows}
    overlap = fit_records & calibration_records
    if overlap:
        raise GraftCredibilityPipelineError(
            f"Fit and calibration record groups overlap: {sorted(overlap)[:5]}"
        )

    models: dict[str, FrozenPairLogisticModel] = {}
    calibrators: dict[str, NonnegativeLogisticCalibrator] = {}
    task_audit: dict[str, Any] = {}
    calibrated_scores: dict[str, np.ndarray] = {}
    for task in TASKS:
        model, fit_weights = _fit_pair_model(
            task,
            fit_by_task[task],
            features[task],
            logistic_c=logistic_c,
            maximum_iterations=maximum_iterations,
            random_seed=random_seed,
        )
        calibration_task_rows = calibration_by_task[task]
        try:
            calibration_weights = np.asarray(
                family_record_weights(calibration_task_rows), dtype=np.float64
            )
        except Exception as error:
            raise GraftCredibilityPipelineError(
                f"{task} calibration record-family weights failed: {error}"
            ) from error
        if not np.all(calibration_weights > 0.0):
            raise GraftCredibilityPipelineError(f"{task} contains a zero-weight calibration pair row")
        logits = model.logits(calibration_task_rows)
        labels = [int(row["is_real_pair"]) for row in calibration_task_rows]
        groups = [str(row["record"]) for row in calibration_task_rows]
        calibrator = fit_nonnegative_logistic_calibrator(
            logits,
            labels,
            groups,
            input_kind="logit",
            sample_weights=calibration_weights,
        )
        scores = calibrator.predict_from_logits(logits)
        models[task] = model
        calibrators[task] = calibrator
        calibrated_scores[task] = scores
        task_audit[task] = {
            "feature_kind": TASK_FEATURE_KIND[task],
            "feature_names": list(features[task]),
            "fit_pair_rows": len(fit_by_task[task]),
            "fit_record_groups": len({str(row["record"]) for row in fit_by_task[task]}),
            "fit_real_weight_mass": float(
                np.sum(fit_weights[[int(row["is_real_pair"]) == 1 for row in fit_by_task[task]]])
            ),
            "fit_corrupt_weight_mass": float(
                np.sum(fit_weights[[int(row["is_real_pair"]) == 0 for row in fit_by_task[task]]])
            ),
            "calibration_pair_rows": len(calibration_task_rows),
            "calibration_record_groups": len(set(groups)),
            "calibration_real_weight_mass": float(
                np.sum(calibration_weights[np.asarray(labels) == 1])
            ),
            "calibration_corrupt_weight_mass": float(
                np.sum(calibration_weights[np.asarray(labels) == 0])
            ),
            "calibrator_slope": calibrator.slope,
        }

    pa_rows = calibration_by_task["PA"]
    cross_record_indices = [
        index for index, row in enumerate(pa_rows) if str(row["corruption_family"]) == "cross_record"
    ]
    if not cross_record_indices:
        raise GraftCredibilityPipelineError("Calibration PA task contains no cross-record null rows")
    try:
        pa_null = RecordBalancedNullCDF(
            calibrated_scores["PA"][cross_record_indices],
            [str(pa_rows[index]["record"]) for index in cross_record_indices],
            minimum_rows=minimum_pa_null_rows,
            minimum_groups=minimum_pa_null_groups,
            maximum_standard_error=maximum_pa_null_standard_error,
            precision_probabilities=precision_probabilities,
        )
    except GraftCredibilityError as error:
        raise GraftCredibilityPipelineError(f"PA null gate failed: {error}") from error

    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "clinical_outcomes_used_as_credibility_inputs": False,
        "outer_evaluation_rows_received_during_fit": False,
        "fit_calibration_record_overlap": 0,
        "pair_weighting": "equal_record_then_equal_real_vs_corrupt_then_equal_corruption_family",
        "task_models": task_audit,
        "pa_null": pa_null.audit(),
    }
    return FoldLocalCredibilityBundle(
        immutable_context=dict(immutable_context),
        models=models,
        calibrators=calibrators,
        pa_null=pa_null,
        fit_pair_rows_sha256=_rows_digest(fit_rows, features),
        calibration_pair_rows_sha256=_rows_digest(calibration_rows, features),
        fit_record_set_sha256=canonical_sha256(sorted(fit_records)),
        calibration_record_set_sha256=canonical_sha256(sorted(calibration_records)),
        audit=audit,
    )
