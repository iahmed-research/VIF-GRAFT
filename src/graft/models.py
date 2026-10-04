"""Same-encoder QMF adaptation and matched early-concatenation model."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F
EMBEDDING_WIDTH=192

class GraftModelError(RuntimeError):
    """Raised when a model receives an impossible availability pattern."""


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class FirstLayer(nn.Module):
    def __init__(self, in_channels: int, width: int = 32, kernel: int = 15) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, width, kernel_size=kernel, stride=2, padding=kernel // 2, bias=False
        )
        self.norm = nn.GroupNorm(_group_count(width), width)
        self.activation = nn.GELU()

    def forward(self, values: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(values)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel, padding=padding, bias=False
        )
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation = nn.GELU()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, values: Tensor) -> Tensor:
        residual = self.skip(values)
        hidden = self.activation(self.norm1(self.conv1(values)))
        hidden = self.norm2(self.conv2(hidden))
        return self.activation(hidden + residual)


class ResidualTrunk(nn.Module):
    """Two blocks at each width; transitions realize strides 1, 2 and 2."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlock(32, 32, 15),
            ResidualBlock(32, 32, 15),
            ResidualBlock(32, 64, 9, stride=2),
            ResidualBlock(64, 64, 9),
            ResidualBlock(64, 96, 5, stride=2),
            ResidualBlock(96, 96, 5),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.blocks(values)


class StatsPool(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        mean = torch.mean(values, dim=-1)
        # Population SD is defined for every temporal length and deterministic.
        standard_deviation = torch.std(values, dim=-1, correction=0)
        return torch.cat((mean, standard_deviation), dim=1)


@dataclass(frozen=True)
class EncodedModalities:
    ecg: Tensor
    ppg: Tensor
    abp: Tensor
    availability: Tensor


class CommonBackbone(nn.Module):
    """ECG encoder plus modality-specific pulse first layers and shared trunk."""

    def __init__(self) -> None:
        super().__init__()
        self.ecg_first = FirstLayer(2)
        self.ecg_trunk = ResidualTrunk()
        self.ppg_first = FirstLayer(1)
        self.abp_first = FirstLayer(1)
        self.pulse_trunk = ResidualTrunk()
        self.pool = StatsPool()

    def _optional_embedding(
        self, values: Tensor, present: Tensor, first: nn.Module
    ) -> Tensor:
        batch = values.shape[0]
        output = torch.full(
            (batch, EMBEDDING_WIDTH),
            float("nan"),
            dtype=values.dtype,
            device=values.device,
        )
        indices = torch.nonzero(present, as_tuple=False).flatten()
        if len(indices):
            encoded = self.pool(self.pulse_trunk(first(values.index_select(0, indices))))
            output.index_copy_(0, indices, encoded)
        return output

    def forward(self, signals: Tensor, availability: Tensor) -> EncodedModalities:
        if signals.ndim != 3 or signals.shape[1] != 4:
            raise GraftModelError("signals must have shape [batch, 4, time]")
        if availability.shape != (signals.shape[0], 3) or availability.dtype != torch.bool:
            raise GraftModelError("availability must be Boolean with shape [batch, 3]")
        if not torch.all(availability[:, 0]):
            raise GraftModelError("ECG anchor must be available for every cached event")
        ecg = self.pool(self.ecg_trunk(self.ecg_first(signals[:, 0:2, :])))
        ppg = self._optional_embedding(signals[:, 2:3, :], availability[:, 1], self.ppg_first)
        abp = self._optional_embedding(signals[:, 3:4, :], availability[:, 2], self.abp_first)
        return EncodedModalities(ecg, ppg, abp, availability)


class TwoLayerHead(nn.Module):
    def __init__(self, input_width: int, output_width: int, hidden_width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, output_width),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


class ModalityDecisionHeads(nn.Module):
    def __init__(self, hidden_width: int) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {role: TwoLayerHead(EMBEDDING_WIDTH, 2, hidden_width) for role in ("ecg", "ppg", "abp")}
        )

    def forward(self, encoded: EncodedModalities) -> tuple[Tensor, Tensor]:
        embeddings = (encoded.ecg, encoded.ppg, encoded.abp)
        batch = encoded.ecg.shape[0]
        logits = torch.full(
            (batch, 3, 2), float("nan"), dtype=encoded.ecg.dtype, device=encoded.ecg.device
        )
        for view, (role, embedding) in enumerate(zip(("ecg", "ppg", "abp"), embeddings, strict=True)):
            indices = torch.nonzero(encoded.availability[:, view], as_tuple=False).flatten()
            if len(indices):
                prediction = self.heads[role](embedding.index_select(0, indices))
                logits[indices, view, :] = prediction.to(dtype=logits.dtype)
        return logits, encoded.availability


def qmf_energy_confidence(logits: Tensor, temperature: float = 1.0, scale: float = 0.1) -> Tensor:
    if temperature <= 0 or scale <= 0:
        raise GraftModelError("QMF temperature and scale must be positive")
    return scale * temperature * torch.logsumexp(logits / temperature, dim=-1)


class QMFModel(nn.Module):
    """Same-encoder adaptation of QMF's energy-weighted decision fusion."""

    def __init__(self, hidden_width: int = 337, temperature: float = 1.0) -> None:
        super().__init__()
        self.backbone = CommonBackbone()
        self.decisions = ModalityDecisionHeads(hidden_width)
        self.temperature = float(temperature)

    def forward(self, signals: Tensor, availability: Tensor) -> dict[str, Tensor]:
        encoded = self.backbone(signals, availability)
        logits, view_mask = self.decisions(encoded)
        finite_logits = torch.nan_to_num(logits, nan=0.0)
        confidence = qmf_energy_confidence(finite_logits, self.temperature)
        confidence = confidence.masked_fill(~view_mask, 0.0)
        fused_logits = torch.sum(confidence[:, :, None] * finite_logits, dim=1)
        return {
            "probability": torch.softmax(fused_logits, dim=1)[:, 1],
            "fused_logits": fused_logits,
            "view_logits": logits,
            "view_mask": view_mask,
            "energy_confidence": confidence,
            "negative_confidence_count": torch.sum((confidence < 0) & view_mask),
        }


class EarlyConcatenationModel(nn.Module):
    def __init__(self, hidden_width: int = 337) -> None:
        super().__init__()
        self.backbone = CommonBackbone()
        self.ppg_missing_token = nn.Parameter(torch.zeros(EMBEDDING_WIDTH))
        self.abp_missing_token = nn.Parameter(torch.zeros(EMBEDDING_WIDTH))
        # Candidate 008 gives the strongest generic concatenation baseline the
        # same credibility and availability coordinates as the coalition models.
        self.classifier = TwoLayerHead(3 * EMBEDDING_WIDTH + 5, 1, hidden_width)

    def forward(
        self,
        signals: Tensor,
        availability: Tensor,
        r_ppg: Tensor,
        r_abp: Tensor,
        eta: Tensor,
    ) -> dict[str, Tensor]:
        encoded = self.backbone(signals, availability)
        ppg = torch.where(
            availability[:, 1:2],
            torch.nan_to_num(encoded.ppg, nan=0.0),
            self.ppg_missing_token[None, :],
        )
        abp = torch.where(
            availability[:, 2:3],
            torch.nan_to_num(encoded.abp, nan=0.0),
            self.abp_missing_token[None, :],
        )
        quality = torch.stack(
            (
                torch.where(availability[:, 1], r_ppg, torch.zeros_like(r_ppg)),
                torch.where(availability[:, 2], r_abp, torch.zeros_like(r_abp)),
                torch.where(availability[:, 1] & availability[:, 2], eta, torch.zeros_like(eta)),
                availability[:, 1].to(signals.dtype),
                availability[:, 2].to(signals.dtype),
            ),
            dim=1,
        )
        inputs = torch.cat((encoded.ecg, ppg, abp, quality), dim=1)
        logits = self.classifier(inputs).squeeze(1)
        return {"probability": torch.sigmoid(logits), "logit": logits}
