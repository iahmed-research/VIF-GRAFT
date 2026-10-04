"""Record-balanced training losses for the frozen M2 model panel."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


class GraftLossError(RuntimeError):
    """Raised when a loss would violate denominator or weighting rules."""


def record_balanced_weights(record_groups: Sequence[str], *, device=None, dtype=None) -> Tensor:
    if not record_groups or any(not str(group) for group in record_groups):
        raise GraftLossError("record_groups must be nonempty identifiers")
    counts = Counter(str(group) for group in record_groups)
    raw = torch.tensor(
        [1.0 / counts[str(group)] for group in record_groups], device=device, dtype=dtype or torch.float32
    )
    # Sum is the number of groups; normalization makes weighted_mean simply sum/N_groups.
    return raw


def record_balanced_mean(
    values: Tensor, weights: Tensor, *, batch_unbiased: bool = False
) -> Tensor:
    if values.ndim != 1 or weights.shape != values.shape:
        raise GraftLossError("values and weights must be aligned vectors")
    if not torch.all(torch.isfinite(values)) or not torch.all(torch.isfinite(weights)):
        raise GraftLossError("Loss values and record weights must be finite")
    if not torch.all(weights > 0):
        raise GraftLossError("Record weights must be positive")
    denominator = values.numel() if batch_unbiased else torch.sum(weights)
    return torch.sum(values * weights) / denominator


def binary_probability_loss(
    probability: Tensor,
    target: Tensor,
    record_weights: Tensor,
    *,
    batch_unbiased: bool = False,
) -> Tensor:
    if probability.shape != target.shape or probability.ndim != 1:
        raise GraftLossError("probability and target must be aligned vectors")
    if not torch.all(torch.isfinite(probability)) or not torch.all((probability >= 0) & (probability <= 1)):
        raise GraftLossError("Predicted probabilities must be finite in [0,1]")
    losses = F.binary_cross_entropy(
        probability.clamp(1e-6, 1.0 - 1e-6), target.to(probability.dtype), reduction="none"
    )
    return record_balanced_mean(
        losses, record_weights.to(losses), batch_unbiased=batch_unbiased
    )


def coalition_joint_loss(
    coalition_logits: Tensor,
    coalition_mask: Tensor,
    target: Tensor,
    record_weights: Tensor,
    *,
    batch_unbiased: bool = False,
) -> Tensor:
    if coalition_logits.ndim != 2 or coalition_logits.shape != coalition_mask.shape:
        raise GraftLossError("coalition logits and mask must be aligned matrices")
    if target.shape != (coalition_logits.shape[0],):
        raise GraftLossError("target must align with coalition rows")
    if not torch.all(coalition_mask[:, 0]):
        raise GraftLossError("Every event must have its ECG coalition")
    finite_where_used = torch.isfinite(coalition_logits) | ~coalition_mask
    if not torch.all(finite_where_used):
        raise GraftLossError("An available coalition has a non-finite logit")
    expanded_target = target[:, None].expand_as(coalition_logits).to(coalition_logits.dtype)
    elementwise = F.binary_cross_entropy_with_logits(
        torch.nan_to_num(coalition_logits, nan=0.0), expanded_target, reduction="none"
    )
    # Equal weight per event regardless of its number of operational coalitions.
    per_event = torch.sum(elementwise * coalition_mask, dim=1) / torch.sum(coalition_mask, dim=1)
    return record_balanced_mean(
        per_event, record_weights.to(per_event), batch_unbiased=batch_unbiased
    )


def qmf_training_loss(
    output: dict[str, Tensor],
    target: Tensor,
    record_weights: Tensor,
    ranking_penalty: Tensor | None = None,
    ranking_coefficient: float = 0.1,
    batch_unbiased: bool = False,
) -> Tensor:
    view_logits = output["view_logits"]
    view_mask = output["view_mask"]
    if view_logits.ndim != 3 or view_logits.shape[2] != 2 or view_mask.shape != view_logits.shape[:2]:
        raise GraftLossError("Malformed QMF view outputs")
    class_target = target.to(torch.long)
    fused_logits = output["fused_logits"]
    fused = F.cross_entropy(fused_logits, class_target, reduction="none")
    finite = torch.nan_to_num(view_logits, nan=0.0).reshape(-1, 2)
    repeated_target = class_target[:, None].expand_as(view_mask).reshape(-1)
    view_losses = F.cross_entropy(finite, repeated_target, reduction="none").reshape_as(view_mask)
    # Faithful variable-view adaptation of QMF Eq. 18: fused CE plus the sum of
    # all available modality CEs. Missing views contribute exactly zero.
    per_event = fused + torch.sum(view_losses * view_mask, dim=1)
    loss = record_balanced_mean(
        per_event, record_weights.to(per_event), batch_unbiased=batch_unbiased
    )
    if ranking_penalty is not None:
        if ranking_penalty.ndim != 0 or not torch.isfinite(ranking_penalty):
            raise GraftLossError("QMF ranking penalty must be a finite scalar")
        loss = loss + float(ranking_coefficient) * ranking_penalty
    return loss


@dataclass
class QMFHistory:
    """Per-event, per-view running loss used in QMF trajectory regularization."""

    cumulative_loss: np.ndarray
    observations: np.ndarray

    @classmethod
    def create(cls, event_count: int, views: int = 3) -> "QMFHistory":
        if event_count < 1 or views < 1:
            raise GraftLossError("QMF history dimensions must be positive")
        return cls(
            cumulative_loss=np.zeros((event_count, views), dtype=np.float64),
            observations=np.zeros((event_count, views), dtype=np.int64),
        )

    def update(self, event_indices: Tensor, losses: Tensor, view_mask: Tensor) -> None:
        indices = event_indices.detach().cpu().numpy().astype(int)
        observed_losses = losses.detach().cpu().numpy()
        observed_mask = view_mask.detach().cpu().numpy().astype(bool)
        if observed_losses.shape != observed_mask.shape or observed_losses.shape[0] != len(indices):
            raise GraftLossError("QMF history update arrays do not align")
        for row, event_index in enumerate(indices):
            if not 0 <= event_index < len(self.cumulative_loss):
                raise GraftLossError("QMF event index is outside history")
            self.cumulative_loss[event_index, observed_mask[row]] += observed_losses[row, observed_mask[row]]
            self.observations[event_index, observed_mask[row]] += 1

    def normalized_running_mean(self, view: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= view < self.cumulative_loss.shape[1]:
            raise GraftLossError("QMF view index is outside history")
        seen = self.observations[:, view] > 0
        values = np.zeros(len(seen), dtype=np.float64)
        values[seen] = self.cumulative_loss[seen, view] / self.observations[seen, view]
        if np.sum(seen) >= 2:
            low = float(np.min(values[seen]))
            high = float(np.max(values[seen]))
            if high > low:
                values[seen] = (values[seen] - low) / (high - low)
            else:
                values[seen] = 0.0
        return values, seen


def qmf_equation16_ranking_penalty(
    confidence: Tensor,
    view_mask: Tensor,
    event_indices: Tensor,
    history: QMFHistory,
) -> Tensor:
    """Implement QMF Eq. 16 with fixed cyclic within-batch pairs.

    Only pairs for which both events have prior history and the view is present
    contribute. This removes the zero-range first-pass
    division without inventing pseudo-history.
    """

    if confidence.shape != view_mask.shape or confidence.ndim != 2:
        raise GraftLossError("QMF confidence and view mask must be aligned matrices")
    if event_indices.shape != (confidence.shape[0],):
        raise GraftLossError("QMF event indices must align with the batch")
    if confidence.shape[0] < 2:
        return torch.zeros((), dtype=confidence.dtype, device=confidence.device)
    indices_numpy = event_indices.detach().cpu().numpy().astype(int)
    if np.any(indices_numpy < 0) or np.any(indices_numpy >= len(history.cumulative_loss)):
        raise GraftLossError("QMF event index is outside history")
    penalties: list[Tensor] = []
    for view in range(confidence.shape[1]):
        kappa, seen = history.normalized_running_mean(view)
        paired_indices = np.roll(indices_numpy, -1)
        eligible_numpy = (
            seen[indices_numpy]
            & seen[paired_indices]
            & view_mask[:, view].detach().cpu().numpy().astype(bool)
            & np.roll(view_mask[:, view].detach().cpu().numpy().astype(bool), -1)
        )
        eligible = torch.as_tensor(eligible_numpy, device=confidence.device)
        if not torch.any(eligible):
            continue
        kappa_difference = torch.as_tensor(
            kappa[indices_numpy] - kappa[paired_indices],
            dtype=confidence.dtype,
            device=confidence.device,
        )
        weight_difference = confidence[:, view] - torch.roll(confidence[:, view], -1)
        equation_16 = torch.relu(
            torch.sign(weight_difference) * kappa_difference + torch.abs(weight_difference)
        )
        penalties.append(torch.mean(equation_16[eligible]))
    if not penalties:
        return torch.zeros((), dtype=confidence.dtype, device=confidence.device)
    return torch.mean(torch.stack(penalties))


def dirichlet_kl_to_uniform(alpha: Tensor) -> Tensor:
    if alpha.ndim != 2 or not torch.all(alpha > 0):
        raise GraftLossError("Dirichlet alpha must be a positive matrix")
    classes = alpha.shape[1]
    beta = torch.ones((1, classes), dtype=alpha.dtype, device=alpha.device)
    strength_alpha = torch.sum(alpha, dim=1, keepdim=True)
    strength_beta = torch.sum(beta, dim=1, keepdim=True)
    log_beta_alpha = torch.lgamma(strength_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    log_beta_uniform = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(strength_beta)
    return (
        torch.sum(
            (alpha - beta) * (torch.digamma(alpha) - torch.digamma(strength_alpha)),
            dim=1,
            keepdim=True,
        )
        + log_beta_alpha
        + log_beta_uniform
    ).squeeze(1)


def tmc_evidential_ce(alpha: Tensor, target: Tensor, epoch: int, annealing_epochs: int = 50) -> Tensor:
    if alpha.ndim != 2 or alpha.shape[1] != 2 or target.shape != (alpha.shape[0],):
        raise GraftLossError("TMC alpha and targets do not align for binary classification")
    if epoch < 1 or annealing_epochs < 1:
        raise GraftLossError("TMC epochs must be positive")
    strength = torch.sum(alpha, dim=1, keepdim=True)
    label = F.one_hot(target.to(torch.long), num_classes=2).to(alpha.dtype)
    expected_ce = torch.sum(label * (torch.digamma(strength) - torch.digamma(alpha)), dim=1)
    evidence = alpha - 1.0
    adjusted = evidence * (1.0 - label) + 1.0
    coefficient = min(1.0, epoch / annealing_epochs)
    return expected_ce + coefficient * dirichlet_kl_to_uniform(adjusted)


def tmc_training_loss(
    output: dict[str, Tensor],
    target: Tensor,
    record_weights: Tensor,
    *,
    epoch: int,
    annealing_epochs: int = 50,
    batch_unbiased: bool = False,
) -> Tensor:
    view_alpha = output["view_alpha"]
    view_mask = output["view_mask"]
    combined = output["combined_alpha"]
    per_event = tmc_evidential_ce(combined, target, epoch, annealing_epochs)
    for view in range(view_alpha.shape[1]):
        indices = torch.nonzero(view_mask[:, view], as_tuple=False).flatten()
        if len(indices):
            contribution = tmc_evidential_ce(
                view_alpha.index_select(0, indices)[:, view, :],
                target.index_select(0, indices),
                epoch,
                annealing_epochs,
            )
            per_event = per_event.index_add(0, indices, contribution)
    return record_balanced_mean(
        per_event, record_weights.to(per_event), batch_unbiased=batch_unbiased
    )
