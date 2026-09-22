"""Render the Julia-generated coordinate slices with NumPy and Matplotlib.

Usage: python continuous_slices.py INPUT.csv OUTPUT.svg
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(source, destination):
    data = np.genfromtxt(source, delimiter=",", names=True)
    plt.rcParams.update({
        "font.size": 11,
        "svg.fonttype": "none",
        "svg.hashsalt": "lineartrees-continuous",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, layout="constrained")
    colors = ("#399779", "#7a59a4", "#cf7430")
    for output in (1, 2):
        for feature in (1, 2):
            ax = axes[output - 1, feature - 1]
            ax.axvline(0, color="#b8bec6", linewidth=1, linestyle=":")
            for fixed, color in zip((-0.7, 0.0, 0.7), colors):
                rows = data[(data["output"] == output) & (data["feature"] == feature)
                            & (data["fixed"] == fixed)]
                ax.plot(rows["x"], rows["mean"], color=color, linewidth=2,
                        label=f"{fixed:+.1f}")
            ax.grid(axis="y", color="#eceef1", linewidth=0.7)
            ax.set_xlim(-1.2, 1.2)
            ax.set_xticks((-1, -0.5, 0, 0.5, 1))
            ax.set_xlabel(f"Predictor {feature}")
            ax.set_ylabel(f"Output {output}")
            if output == 1:
                ax.set_title(f"Vary predictor {feature}", loc="left", fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Fixed value of the other predictor",
               loc="outside upper center", ncol=3, frameon=False)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, metadata={"Date": None})
    fig.savefig(destination.with_suffix(".png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    render(*sys.argv[1:])
