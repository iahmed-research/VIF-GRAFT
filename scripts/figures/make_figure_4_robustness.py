"""Generate the GRAFT robustness figure from saved robustness results.

Input:  robustness/RESULTS.json from a completed GRAFT reproduction run.
Output: PDF/PNG/SVG figure files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_conditions(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "conditions" not in payload:
        raise ValueError("Expected a robustness RESULTS.json file with a 'conditions' object")
    return payload["conditions"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 4: GRAFT robustness")
    parser.add_argument("--results", type=Path, required=True, help="Path to robustness/RESULTS.json")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--prefix", default="Graft-robustness", help="Output filename prefix")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    conditions = load_conditions(args.results)
    preferred_order = ["clean", "missing", "noise_20dB", "noise_10dB", "noise_0dB", "gap_0.1s", "gap_0.5s", "gap_2s"]
    names = [name for name in preferred_order if name in conditions]
    names += [name for name in conditions if name not in names]

    means = []
    lows = []
    highs = []
    for name in names:
        entry = conditions[name]
        if "graft_minus_qmf_anchor" in entry:
            contrast = entry["graft_minus_qmf_anchor"]
        elif "B_minus_QMF" in entry:
            contrast = entry["B_minus_QMF"]
        else:
            raise KeyError(f"No GRAFT-vs-QMF contrast found for condition {name!r}")
        mean = float(contrast.get("loss_difference", contrast.get("candidate_minus_reference")))
        ci = contrast.get("ci95", contrast.get("difference_ci95"))
        means.append(mean)
        lows.append(float(ci[0]))
        highs.append(float(ci[1]))

    means = np.asarray(means)
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    y = np.arange(len(names))
    labels = [name.replace("noise_", "noise ").replace("gap_", "gap ").replace("dB", " dB") for name in names]

    fig, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
    xerr = np.vstack([np.maximum(0, means - lows), np.maximum(0, highs - means)])
    ax.errorbar(means, y, xerr=xerr, fmt="o", capsize=3, color="#1f4e79", ecolor="#1f4e79")
    ax.axvline(0, color="0.45", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("GRAFT minus QMF record-balanced log loss (paired 95% interval)")
    ax.set_title("Robustness to supplementary-ECG corruption")
    ax.grid(axis="x", color="0.9", linewidth=0.8)

    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(args.out / f"{args.prefix}.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
