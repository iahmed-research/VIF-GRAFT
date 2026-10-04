from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
from dataclasses import dataclass, asdict, replace
import json, math, hashlib
import numpy as np
from scipy import signal
import re
GAIN_RE = re.compile(r"^(?P<gain>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\((?P<baseline>[+-]?\d+)\))?/(?P<units>.+)$")


@dataclass(frozen=True)
class SignalSpec:
    file_name: str
    wfdb_format: int
    gain: float
    baseline: int
    units: str
    adc_resolution_bits: int
    adc_zero: int
    initial_value: int
    checksum: int
    block_size: int
    name: str

@dataclass(frozen=True)
class RecordHeader:
    record_name: str
    signal_count: int
    sampling_frequency_hz: float
    sample_count: int
    signals: tuple[SignalSpec, ...]

class AuditError(RuntimeError):
    """Raised when a binding data-integrity invariant fails."""

def parse_signal_spec(line: str) -> SignalSpec:
    fields = line.split()
    if len(fields) < 9:
        raise AuditError(f"Malformed WFDB signal line: {line!r}")
    format_text = fields[1].split("x", 1)[0].split(":", 1)[0]
    match = GAIN_RE.fullmatch(fields[2])
    if match is None:
        raise AuditError(f"Unsupported gain/baseline/unit field: {fields[2]!r}")
    adc_zero = int(fields[4])
    baseline = int(match.group("baseline")) if match.group("baseline") is not None else adc_zero
    gain = float(match.group("gain"))
    if not math.isfinite(gain) or gain <= 0.0:
        raise AuditError(f"Non-positive or non-finite ADC gain: {fields[2]!r}")
    return SignalSpec(
        file_name=fields[0],
        wfdb_format=int(format_text),
        gain=gain,
        baseline=baseline,
        units=match.group("units"),
        adc_resolution_bits=int(fields[3]),
        adc_zero=adc_zero,
        initial_value=int(fields[5]),
        checksum=int(fields[6]),
        block_size=int(fields[7]),
        name=fields[-1].upper(),
    )

def parse_header(path: Path) -> RecordHeader:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not lines:
        raise AuditError(f"Empty WFDB header: {path}")
    first = lines[0].split()
    if len(first) < 4:
        raise AuditError(f"Malformed WFDB record line: {lines[0]!r}")
    signal_count = int(first[1])
    frequency = float(first[2].split("/", 1)[0])
    sample_count = int(first[3])
    signal_lines = lines[1 : 1 + signal_count]
    if len(signal_lines) != signal_count:
        raise AuditError(
            f"Header declares {signal_count} channels but contains {len(signal_lines)}: {path}"
        )
    return RecordHeader(
        record_name=first[0],
        signal_count=signal_count,
        sampling_frequency_hz=frequency,
        sample_count=sample_count,
        signals=tuple(parse_signal_spec(line) for line in signal_lines),
    )

def signal_role(name: str, config: Mapping[str, Any]) -> str:
    roles = config["signal_roles"]
    for role in ("ecg", "ppg", "abp"):
        if name in roles[role]:
            return role
    return "other"

def adjacent_equal_fraction(values: np.ndarray, valid_mask: np.ndarray) -> float:
    if len(values) < 2:
        return math.nan
    pair_valid = valid_mask[:-1] & valid_mask[1:]
    denominator = int(pair_valid.sum())
    if denominator == 0:
        return math.nan
    return float(np.count_nonzero((np.diff(values) == 0) & pair_valid) / denominator)
