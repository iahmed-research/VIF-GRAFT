"""Generate the GRAFT audit/XAI figure from saved explanation results.

Input:  explanations/RESULTS.json from a completed GRAFT reproduction run.
Output: PDF/PNG/SVG figure files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_groups(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    groups = payload.get("group_contributions")
    if not groups:
        raise ValueError("Expected an explanations RESULTS.json file with group_contributions")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 5: GRAFT residual audit")
    parser.add_argument("--results", type=Path, required=True, help="Path to explanations/RESULTS.json")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--prefix", default="Graft-audit", help="Output filename prefix")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = load_groups(args.results)
    names = [row["group"] for row in groups]
    values = np.asarray([float(row["mean_absolute_score"]) for row in groups])
    order = np.argsort(values)

    fig, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
    ax.barh(np.arange(len(order)), values[order], color="#4f81bd")
    ax.set_yticks(np.arange(len(order)), [names[i] for i in order])
    ax.set_xlabel("Mean absolute contribution to pre-tanh residual score")
    ax.set_title("Audit of the GRAFT residual correction")
    ax.grid(axis="x", color="0.9", linewidth=0.8)

    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(args.out / f"{args.prefix}.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
