"""Frozen nonnegative-slope calibration and safety-constrained thresholding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


class GraftCalibrationError(RuntimeError):
    """Raised when calibration or threshold selection is not identifiable."""


def _finite_vector(name: str, values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise GraftCalibrationError(f"{name} must be a nonempty finite vector")
    return array


def record_weights(record_groups: Sequence[str]) -> np.ndarray:
    groups = np.asarray(record_groups, dtype=str)
    if groups.ndim != 1 or not len(groups) or any(not group for group in groups):
        raise GraftCalibrationError("record_groups must be nonempty identifiers")
    unique, counts = np.unique(groups, return_counts=True)
    lookup = dict(zip(unique, counts, strict=True))
    return np.asarray([1.0 / lookup[group] for group in groups], dtype=np.float64)


def probability_to_logit(probabilities: Sequence[float], clip: float = 1e-6) -> np.ndarray:
    values = _finite_vector("probabilities", probabilities)
    if np.any((values < 0.0) | (values > 1.0)):
        raise GraftCalibrationError("probabilities must lie in [0,1]")
    if not 0.0 < clip < 0.5:
        raise GraftCalibrationError("probability clip must lie in (0,0.5)")
    values = np.clip(values, clip, 1.0 - clip)
    return np.log(values) - np.log1p(-values)


@dataclass(frozen=True)
class NonnegativeLogisticCalibrator:
    intercept: float
    slope: float
    input_kind: str
    probability_clip: float = 1e-6

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.intercept)
            or not math.isfinite(self.slope)
            or self.slope < 0.0
            or self.input_kind not in {"probability", "logit"}
            or not math.isfinite(self.probability_clip)
            or not 0.0 < self.probability_clip < 0.5
        ):
            raise GraftCalibrationError(
                "Calibrator requires finite intercept, nonnegative finite slope, "
                "declared input kind, and probability clip in (0,0.5)"
            )

    def predict_from_logits(self, logits: Sequence[float]) -> np.ndarray:
        values = _finite_vector("logits", logits)
        transformed = self.intercept + self.slope * values
        return np.exp(-np.logaddexp(0.0, -transformed))

    def predict_from_probabilities(self, probabilities: Sequence[float]) -> np.ndarray:
        return self.predict_from_logits(probability_to_logit(probabilities, self.probability_clip))


def fit_nonnegative_logistic_calibrator(
    values: Sequence[float],
    outcomes: Sequence[int],
    record_groups: Sequence[str],
    *,
    input_kind: str,
    probability_clip: float = 1e-6,
    sample_weights: Sequence[float] | None = None,
) -> NonnegativeLogisticCalibrator:
    raw = _finite_vector("values", values)
    labels = np.asarray(outcomes, dtype=np.int8)
    if labels.shape != raw.shape or not np.all((labels == 0) | (labels == 1)):
        raise GraftCalibrationError("outcomes must be aligned binary values")
    if len(set(labels.tolist())) != 2:
        raise GraftCalibrationError("Calibration partition must contain both outcome classes")
    if len(record_groups) != len(raw):
        raise GraftCalibrationError("record_groups must align with calibration rows")
    if input_kind == "probability":
        predictors = probability_to_logit(raw, probability_clip)
    elif input_kind == "logit":
        predictors = raw
    else:
        raise GraftCalibrationError("input_kind must be 'probability' or 'logit'")
    if sample_weights is None:
        weights = record_weights(record_groups)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if (
            weights.shape != raw.shape
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.any(weights > 0.0)
        ):
            raise GraftCalibrationError(
                "sample_weights must be aligned, finite, nonnegative, and have positive mass"
            )
        groups = np.asarray(record_groups, dtype=str)
        if any(not group for group in groups):
            raise GraftCalibrationError("record_groups must be nonempty identifiers")
        if any(float(np.sum(weights[groups == group])) <= 0.0 for group in np.unique(groups)):
            raise GraftCalibrationError("Every calibration record group must have positive weight")
    if float(np.sum(weights[labels == 0])) <= 0.0 or float(np.sum(weights[labels == 1])) <= 0.0:
        raise GraftCalibrationError("Both calibration classes must have positive weight")
    prevalence = float(np.sum(weights * labels) / np.sum(weights))
    initial_intercept = math.log(prevalence) - math.log1p(-prevalence)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, slope = parameters
        linear = intercept + slope * predictors
        losses = np.logaddexp(0.0, linear) - labels * linear
        residual = np.exp(-np.logaddexp(0.0, -linear)) - labels
        denominator = np.sum(weights)
        value = float(np.sum(weights * losses) / denominator)
        gradient = np.asarray(
            [
                np.sum(weights * residual) / denominator,
                np.sum(weights * residual * predictors) / denominator,
            ]
        )
        return value, gradient

    fit = minimize(
        objective,
        np.asarray([initial_intercept, 1.0]),
        method="L-BFGS-B",
        jac=True,
        bounds=((None, None), (0.0, None)),
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 10000, "maxls": 50},
    )
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise GraftCalibrationError(f"Constrained logistic calibration failed: {fit.message}")
    intercept, slope = (float(value) for value in fit.x)
    if slope < -1e-12:
        raise GraftCalibrationError("Optimizer violated the nonnegative-slope constraint")
    return NonnegativeLogisticCalibrator(
        intercept=intercept,
        slope=max(0.0, slope),
        input_kind=input_kind,
        probability_clip=probability_clip,
    )


def weighted_confusion(
    outcomes: np.ndarray, predictions: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    return {
        "TP": float(np.sum(weights[(outcomes == 1) & (predictions == 1)])),
        "TN": float(np.sum(weights[(outcomes == 0) & (predictions == 0)])),
        "FP": float(np.sum(weights[(outcomes == 0) & (predictions == 1)])),
        "FN": float(np.sum(weights[(outcomes == 1) & (predictions == 0)])),
    }


def physionet_challenge_score(confusion: dict[str, float]) -> float:
    numerator = confusion["TP"] + confusion["TN"]
    denominator = numerator + confusion["FP"] + 5.0 * confusion["FN"]
    return numerator / denominator if denominator > 0.0 else math.nan


def select_safety_threshold(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    record_groups: Sequence[str],
    *,
    minimum_sensitivity: float = 0.90,
) -> dict[str, float]:
    values = _finite_vector("probabilities", probabilities)
    labels = np.asarray(outcomes, dtype=np.int8)
    if labels.shape != values.shape or not np.all((labels == 0) | (labels == 1)):
        raise GraftCalibrationError("outcomes must be aligned binary values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise GraftCalibrationError("probabilities must lie in [0,1]")
    if not np.any(labels == 1) or not np.any(labels == 0):
        raise GraftCalibrationError("Threshold partition must contain both outcome classes")
    if len(record_groups) != len(values):
        raise GraftCalibrationError("record_groups must align with threshold rows")
    if not 0.0 <= minimum_sensitivity <= 1.0:
        raise GraftCalibrationError("minimum_sensitivity must lie in [0,1]")
    weights = record_weights(record_groups)
    thresholds = np.unique(np.concatenate(([0.0], values, [1.0])))
    eligible: list[dict[str, float]] = []
    for threshold in thresholds:
        predicted = (values >= threshold).astype(np.int8)
        confusion = weighted_confusion(labels, predicted, weights)
        positive_weight = confusion["TP"] + confusion["FN"]
        sensitivity = confusion["TP"] / positive_weight
        score = physionet_challenge_score(confusion)
        if sensitivity + 1e-15 >= minimum_sensitivity:
            eligible.append(
                {
                    "threshold": float(threshold),
                    "sensitivity": sensitivity,
                    "physionet_challenge_score": score,
                    **confusion,
                }
            )
    if not eligible:
        raise GraftCalibrationError("No threshold satisfies the sensitivity constraint")
    # Stated tie-break: maximum score, then sensitivity, then lower threshold.
    selected = eligible[0]
    for candidate in eligible[1:]:
        better_score = candidate["physionet_challenge_score"] > selected["physionet_challenge_score"] + 1e-15
        tied_score = math.isclose(
            candidate["physionet_challenge_score"],
            selected["physionet_challenge_score"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        better_sensitivity = candidate["sensitivity"] > selected["sensitivity"] + 1e-15
        tied_sensitivity = math.isclose(
            candidate["sensitivity"], selected["sensitivity"], rel_tol=0.0, abs_tol=1e-15
        )
        if better_score or (
            tied_score
            and (
                better_sensitivity
                or (tied_sensitivity and candidate["threshold"] < selected["threshold"])
            )
        ):
            selected = candidate
    return selected
