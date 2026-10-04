"""Scientific training operations for QMF and early concatenation."""
from __future__ import annotations
import math
import torch
from torch import Tensor
import torch.nn.functional as F
from .losses import (QMFHistory,binary_probability_loss,coalition_joint_loss,qmf_equation16_ranking_penalty,qmf_training_loss,tmc_training_loss)

class GraftTrainingError(RuntimeError):
    """Raised before training when a frozen or safety invariant fails."""


def augment_batch(signals: Tensor, availability: Tensor, valid: Tensor, seed: int) -> Tensor:
    """Apply the frozen alignment-preserving augmentations deterministically."""

    if signals.ndim != 3 or availability.shape != (signals.shape[0], 3) or valid.shape != signals.shape:
        raise GraftTrainingError("Malformed augmentation batch")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    output = signals.clone()
    maximum_shift = int(round(0.20 * 250))
    shifts = torch.randint(-maximum_shift, maximum_shift + 1, (len(output),), generator=generator)
    scales = 0.90 + 0.20 * torch.rand((len(output), 4, 1), generator=generator)
    snr_db = 20.0 + 10.0 * torch.rand((len(output), 4, 1), generator=generator)
    for row, shift in enumerate(shifts.tolist()):
        output[row] = torch.roll(output[row], shifts=shift, dims=-1)
        valid[row] = torch.roll(valid[row], shifts=shift, dims=-1)
    output = output * scales
    valid_float = valid.to(output.dtype)
    denom = torch.clamp(valid_float.sum(dim=-1, keepdim=True), min=1.0)
    power = torch.sum(output.square() * valid_float, dim=-1, keepdim=True) / denom
    noise_sd = torch.sqrt(power / torch.pow(10.0, snr_db / 10.0))
    noise = torch.randn(output.shape, generator=generator, dtype=output.dtype) * noise_sd
    output = (output + noise) * valid_float
    # Slots 2 and 3 correspond to optional PPG and ABP; absent slots remain exact zero.
    channel_present = torch.stack(
        (availability[:, 0], availability[:, 0], availability[:, 1], availability[:, 2]), dim=1
    )
    return output * channel_present[:, :, None].to(output.dtype)


def learning_rate_multiplier(epoch: int, maximum_epochs: int = 100) -> float:
    if not 1 <= epoch <= maximum_epochs:
        raise GraftTrainingError("Epoch is outside the frozen schedule")
    if epoch <= 5:
        return epoch / 5.0
    progress = (epoch - 5) / (maximum_epochs - 5)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def training_loss(
    family: str,
    output: dict[str, Tensor],
    target: Tensor,
    weights: Tensor,
    *,
    epoch: int,
    event_indices: Tensor,
    qmf_history: QMFHistory | None,
) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
    history_update = None
    if family == "FCMF":
        loss = coalition_joint_loss(
            output["coalition_logits"], output["coalition_mask"], target, weights,
            batch_unbiased=True,
        )
    elif family == "learned_simplex_gate":
        coalition = coalition_joint_loss(
            output["coalition_logits"], output["coalition_mask"], target, weights,
            batch_unbiased=True,
        )
        fused = binary_probability_loss(
            output["probability"], target, weights, batch_unbiased=True
        )
        loss = 0.5 * (coalition + fused)
    elif family in {"early_concatenation", "official_VTaC_FCN"}:
        loss = binary_probability_loss(
            output["probability"], target, weights, batch_unbiased=True
        )
    elif family == "QMF":
        if qmf_history is None:
            raise GraftTrainingError("QMF history is required")
        view_logits = output["view_logits"]
        class_target = target.to(torch.long)
        flat = torch.nan_to_num(view_logits, nan=0.0).reshape(-1, 2)
        repeated = class_target[:, None].expand_as(output["view_mask"]).reshape(-1)
        view_losses = F.cross_entropy(flat, repeated, reduction="none").reshape_as(output["view_mask"])
        ranking = qmf_equation16_ranking_penalty(
            output["energy_confidence"], output["view_mask"], event_indices, qmf_history
        )
        loss = qmf_training_loss(
            output, target, weights, ranking_penalty=ranking, batch_unbiased=True
        )
        history_update = (view_losses.detach(), output["view_mask"].detach())
    elif family == "TMC":
        loss = tmc_training_loss(
            output, target, weights, epoch=epoch, annealing_epochs=50,
            batch_unbiased=True,
        )
    else:
        raise GraftTrainingError(f"Unsupported family loss: {family}")
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise GraftTrainingError("Training produced a non-finite scalar loss")
    return loss, history_update
