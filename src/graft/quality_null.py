from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
from dataclasses import dataclass, asdict, replace
import json, math, hashlib
import numpy as np
from scipy import signal


class GraftCredibilityError(RuntimeError):
    """Raised when a credibility object violates a frozen information gate."""

def _finite_vector(name: str, values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise GraftCredibilityError(f"{name} must be a nonempty finite vector")
    return array

def _stable_hash(seed: int, *parts: object) -> str:
    payload = "\x1f".join(str(value) for value in (seed, *parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

@dataclass(frozen=True)
class NullPrecisionPoint:
    probability: float
    score: float
    standard_error: float

class RecordBalancedNullCDF:
    """Record-group-balanced empirical null with midranks and smoothing."""

    def __init__(
        self,
        scores: Sequence[float],
        record_groups: Sequence[str],
        *,
        minimum_rows: int = 100,
        minimum_groups: int = 50,
        maximum_standard_error: float = 0.05,
        precision_probabilities: Sequence[float] = tuple(np.arange(0.1, 1.0, 0.1)),
    ) -> None:
        score_array = _finite_vector("scores", scores)
        group_array = np.asarray(record_groups, dtype=str)
        if (
            isinstance(minimum_rows, bool)
            or isinstance(minimum_groups, bool)
            or int(minimum_rows) != minimum_rows
            or int(minimum_groups) != minimum_groups
            or minimum_rows <= 0
            or minimum_groups <= 1
        ):
            raise GraftCredibilityError("PA null minimum row/group gates must be positive integers")
        if group_array.ndim != 1 or len(group_array) != len(score_array):
            raise GraftCredibilityError("record_groups must align one-to-one with scores")
        if any(not group for group in group_array):
            raise GraftCredibilityError("record group identifiers must be nonempty")
        if len(score_array) < minimum_rows:
            raise GraftCredibilityError(
                f"PA null has {len(score_array)} rows; at least {minimum_rows} required"
            )
        unique_groups = np.unique(group_array)
        if len(unique_groups) < minimum_groups:
            raise GraftCredibilityError(
                f"PA null has {len(unique_groups)} record groups; at least {minimum_groups} required"
            )
        probabilities = _finite_vector("precision_probabilities", precision_probabilities)
        if np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
            raise GraftCredibilityError("precision probabilities must lie strictly in (0, 1)")
        if np.any(np.diff(probabilities) <= 0.0):
            raise GraftCredibilityError("precision probabilities must be strictly increasing")
        if not math.isfinite(maximum_standard_error) or maximum_standard_error <= 0.0:
            raise GraftCredibilityError("maximum_standard_error must be positive and finite")

        self._scores = score_array
        self._groups = group_array
        self.minimum_rows = int(minimum_rows)
        self.minimum_groups = int(minimum_groups)
        self.precision_probabilities = tuple(float(value) for value in probabilities)
        self._unique_groups = tuple(sorted(str(group) for group in unique_groups))
        self._group_indices = {
            group: np.flatnonzero(group_array == group) for group in self._unique_groups
        }
        row_weights = np.asarray(
            [1.0 / len(self._group_indices[str(group)]) for group in group_array],
            dtype=np.float64,
        )
        # Sum of row weights is exactly G; normalization below gives each group
        # total mass 1/G irrespective of how many matched controls it contains.
        self._row_weights = row_weights / len(self._unique_groups)
        self._knots = np.unique(score_array)
        self._knot_values = np.asarray(
            [self._smoothed_midrank_at(float(value)) for value in self._knots],
            dtype=np.float64,
        )
        self._precision = tuple(
            self._precision_at(float(probability)) for probability in probabilities
        )
        worst = max(point.standard_error for point in self._precision)
        if worst > maximum_standard_error + 1e-15:
            raise GraftCredibilityError(
                f"PA null maximum cluster standard error {worst:.8f} exceeds "
                f"{maximum_standard_error:.8f}"
            )
        self.maximum_standard_error = float(maximum_standard_error)

    @property
    def row_count(self) -> int:
        return len(self._scores)

    @property
    def group_count(self) -> int:
        return len(self._unique_groups)

    @property
    def precision(self) -> tuple[NullPrecisionPoint, ...]:
        return self._precision

    def _group_contributions(self, x: float) -> np.ndarray:
        return np.asarray(
            [
                float(
                    np.mean(
                        (self._scores[indices] < x).astype(np.float64)
                        + 0.5 * (self._scores[indices] == x).astype(np.float64)
                    )
                )
                for indices in self._group_indices.values()
            ],
            dtype=np.float64,
        )

    def _smoothed_midrank_at(self, x: float) -> float:
        contributions = self._group_contributions(x)
        return float((0.5 + np.sum(contributions)) / (self.group_count + 1.0))

    def _weighted_quantile(self, probability: float) -> float:
        order = np.argsort(self._scores, kind="stable")
        scores = self._scores[order]
        weights = self._row_weights[order]
        cumulative = np.cumsum(weights)
        index = int(np.searchsorted(cumulative, probability, side="left"))
        return float(scores[min(index, len(scores) - 1)])

    def _precision_at(self, probability: float) -> NullPrecisionPoint:
        score = self._weighted_quantile(probability)
        contributions = self._group_contributions(score)
        standard_error = float(np.std(contributions, ddof=1) / math.sqrt(self.group_count))
        return NullPrecisionPoint(probability, score, standard_error)

    def cdf(self, score: float | Sequence[float]) -> float | np.ndarray:
        values = np.asarray(score, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise GraftCredibilityError("Query scores must be finite")
        # Constant tails and linear interpolation between distinct null-score knots.
        result = np.interp(
            values,
            self._knots,
            self._knot_values,
            left=0.5 / (self.group_count + 1.0),
            right=(self.group_count + 0.5) / (self.group_count + 1.0),
        )
        if values.ndim == 0:
            return float(result)
        return result

    def eta(self, score: float | Sequence[float]) -> float | np.ndarray:
        result = 2.0 * np.asarray(self.cdf(score), dtype=np.float64) - 1.0
        if result.ndim == 0:
            return float(result)
        return result

    def audit(self) -> dict[str, object]:
        return {
            "rows": self.row_count,
            "record_groups": self.group_count,
            "weighting": "equal_record_group_then_equal_row_within_group",
            "ties": "midrank",
            "smoothing": "(0.5 + sum_group_contributions) / (G + 1)",
            "interpolation": "linear_between_distinct_score_knots_constant_outside",
            "maximum_observed_standard_error": max(
                point.standard_error for point in self.precision
            ),
            "maximum_permitted_standard_error": self.maximum_standard_error,
            "precision": [
                {
                    "probability": point.probability,
                    "score": point.score,
                    "standard_error": point.standard_error,
                }
                for point in self.precision
            ],
        }

    def to_artifact(self) -> dict[str, object]:
        """Return sufficient state to reconstruct and re-run every null gate."""

        return {
            "schema_version": "graft-record-balanced-null-cdf-1.0.0",
            "scores": self._scores.tolist(),
            "record_groups": self._groups.tolist(),
            "minimum_rows": self.minimum_rows,
            "minimum_groups": self.minimum_groups,
            "maximum_standard_error": self.maximum_standard_error,
            "precision_probabilities": list(self.precision_probabilities),
            "audit": self.audit(),
        }

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, object]) -> "RecordBalancedNullCDF":
        expected = {
            "schema_version",
            "scores",
            "record_groups",
            "minimum_rows",
            "minimum_groups",
            "maximum_standard_error",
            "precision_probabilities",
            "audit",
        }
        if set(artifact) != expected:
            raise GraftCredibilityError("PA null artifact does not have the exact declared fields")
        if artifact["schema_version"] != "graft-record-balanced-null-cdf-1.0.0":
            raise GraftCredibilityError("Unknown PA null artifact schema")
        rebuilt = cls(
            artifact["scores"],  # type: ignore[arg-type]
            artifact["record_groups"],  # type: ignore[arg-type]
            minimum_rows=int(artifact["minimum_rows"]),
            minimum_groups=int(artifact["minimum_groups"]),
            maximum_standard_error=float(artifact["maximum_standard_error"]),
            precision_probabilities=artifact["precision_probabilities"],  # type: ignore[arg-type]
        )
        if rebuilt.audit() != artifact["audit"]:
            raise GraftCredibilityError("Reconstructed PA null audit differs from frozen artifact")
        return rebuilt

def percentile_midranks(
    values: Sequence[float], reference: Sequence[float]
) -> np.ndarray:
    """Map values to the unsmoothed empirical midrank CDF of a reference sample."""

    query = _finite_vector("values", values)
    reference_array = _finite_vector("reference", reference)
    ordered = np.sort(reference_array, kind="stable")
    lower = np.searchsorted(ordered, query, side="left")
    upper = np.searchsorted(ordered, query, side="right")
    return (lower + 0.5 * (upper - lower)) / len(ordered)

def fixed_quality_stratum(q: float) -> int:
    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise GraftCredibilityError(f"Joint quality q must be finite in [0,1], got {q!r}")
    if q < 1.0 / 3.0:
        return 0
    if q < 2.0 / 3.0:
        return 1
    return 2

def eta_cross_record_derangement(
    eta: Sequence[float],
    record_groups: Sequence[str],
    q: Sequence[float],
    event_ids: Sequence[str],
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Exactly permute eta within fixed q strata with no same-record donor.

    Events are hash-ordered within hash-ordered contiguous record blocks.  A
    circular shift by the largest block size is a valid cross-record derangement
    exactly when that size is no greater than half the stratum size.  The mapping
    is bijective, so each stratum's eta multiset is preserved exactly.
    """

    eta_array = _finite_vector("eta", eta)
    q_array = _finite_vector("q", q)
    group_array = np.asarray(record_groups, dtype=str)
    event_array = np.asarray(event_ids, dtype=str)
    count = len(eta_array)
    if len(q_array) != count or len(group_array) != count or len(event_array) != count:
        raise GraftCredibilityError("eta, q, record_groups and event_ids must have equal length")
    if len(set(event_array)) != count or any(not value for value in event_array):
        raise GraftCredibilityError("event_ids must be unique and nonempty")
    if any(not value for value in group_array):
        raise GraftCredibilityError("record_groups must be nonempty")
    strata = np.asarray([fixed_quality_stratum(float(value)) for value in q_array], dtype=int)
    result = np.empty_like(eta_array)
    donor_indices = np.full(count, -1, dtype=int)
    reports: list[dict[str, object]] = []

    for stratum in range(3):
        indices = np.flatnonzero(strata == stratum)
        if not len(indices):
            reports.append({"stratum": stratum, "event_count": 0, "status": "empty"})
            continue
        groups = sorted(
            set(group_array[indices]),
            key=lambda group: (_stable_hash(seed, "group", stratum, group), group),
        )
        if len(groups) < 2:
            raise GraftCredibilityError(
                f"Quality stratum {stratum} contains fewer than two record groups"
            )
        ordered_indices: list[int] = []
        sizes: list[int] = []
        for group in groups:
            members = [int(index) for index in indices if group_array[index] == group]
            members.sort(
                key=lambda index: (
                    _stable_hash(seed, "event", stratum, event_array[index]),
                    event_array[index],
                )
            )
            ordered_indices.extend(members)
            sizes.append(len(members))
        maximum_size = max(sizes)
        if 2 * maximum_size > len(indices):
            raise GraftCredibilityError(
                f"Quality stratum {stratum} has no cross-record event-level derangement: "
                f"largest group {maximum_size} of {len(indices)} events"
            )
        for position, recipient in enumerate(ordered_indices):
            donor = ordered_indices[(position + maximum_size) % len(ordered_indices)]
            donor_indices[recipient] = donor
            result[recipient] = eta_array[donor]
        same_group = int(
            sum(group_array[index] == group_array[donor_indices[index]] for index in indices)
        )
        fixed_points = int(sum(index == donor_indices[index] for index in indices))
        if same_group or fixed_points:
            raise GraftCredibilityError("Constructed eta mapping is not a cross-record derangement")
        if sorted(result[indices].tolist()) != sorted(eta_array[indices].tolist()):
            raise GraftCredibilityError("Eta derangement failed to preserve the stratum multiset")
        reports.append(
            {
                "stratum": stratum,
                "status": "pass",
                "event_count": len(indices),
                "distinct_record_group_count": len(groups),
                "q_minimum": float(np.min(q_array[indices])),
                "q_median": float(np.median(q_array[indices])),
                "q_maximum": float(np.max(q_array[indices])),
                "eta_minimum": float(np.min(eta_array[indices])),
                "eta_median": float(np.median(eta_array[indices])),
                "eta_maximum": float(np.max(eta_array[indices])),
                "largest_record_group_events": maximum_size,
                "same_group_donor_recipient_count": same_group,
                "fixed_point_count": fixed_points,
                "eta_multiset_preserved": True,
            }
        )
    if np.any(donor_indices < 0):
        raise GraftCredibilityError("Not every event received an eta donor")
    audit = {
        "status": "pass",
        "seed": seed,
        "cutpoints": [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        "algorithm": "hash_ordered_contiguous_record_blocks_circular_shift_by_largest_block",
        "outcomes_used": False,
        "global_same_group_donor_recipient_count": int(
            np.sum(group_array == group_array[donor_indices])
        ),
        "global_fixed_point_count": int(np.sum(np.arange(count) == donor_indices)),
        "eta_multiset_preserved_within_each_stratum": True,
        "strata": reports,
    }
    return result, audit
