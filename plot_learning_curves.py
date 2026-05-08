"""
Plot learning curve figures (PNG) from results produced by q2q3_run_all_stats.py.

Usage:
    # 1) Generate results
    python3 q2q3_run_all_stats.py -i 5 -o results.json

    # 2) Make plots
    python3 plot_learning_curves.py results.json

This script writes four PNGs in the current directory:
    digits_error.png   - mean test error (with std error bars) vs training fraction
    digits_time.png    - mean training time vs training fraction
    faces_error.png    - same for face dataset
    faces_time.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


TRAINING_FRACTIONS = list(range(10, 101, 10))


def get_xy(stats_list, ykey, yerrkey=None):
    """Extract x, y (and optional yerr) arrays from a list of stats dicts."""
    # Our stats dicts (see q1a_perceptron_digits.main) use "training_percent".
    x = [d.get("training_percent") for d in stats_list]

    # If for some reason the percent is missing, fall back to fixed 10..100.
    if any(v is None for v in x):
        x = TRAINING_FRACTIONS

    y = [d[ykey] for d in stats_list]
    yerr = [d[yerrkey] for d in stats_list] if yerrkey else None
    return x, y, yerr


def plot_dataset(results, models, title_prefix, out_prefix):
    """Make error/time plots for a single dataset (digits or faces)."""
    # Error curve with std error bars
    plt.figure()
    for m in models:
        stats = results[m]
        x, y, yerr = get_xy(stats, "mean_error", "std_error")
        # Convert error from fraction to percentage for plotting
        y_pct = [val * 100.0 for val in y]
        yerr_pct = [val * 100.0 for val in yerr] if yerr is not None else None
        plt.errorbar(
            x,
            y_pct,
            yerr=yerr_pct,
            marker="o",
            capsize=3,
            label=m.replace("_", " "),
        )
    plt.xlabel("Training fraction (%)")
    plt.ylabel("Mean test error (%)")
    plt.title(f"{title_prefix}: Test error vs training fraction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{out_prefix}_error.png", dpi=200, bbox_inches="tight")

    # Training time curve
    plt.figure()
    for m in models:
        stats = results[m]
        x, y, _ = get_xy(stats, "mean_train_time")
        plt.plot(
            x,
            y,
            marker="o",
            label=m.replace("_", " "),
        )
    plt.xlabel("Training fraction (%)")
    plt.ylabel("Mean training time (s)")
    plt.title(f"{title_prefix}: Training time vs training fraction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{out_prefix}_time.png", dpi=200, bbox_inches="tight")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_learning_curves.py results.json")
        sys.exit(1)

    results_path = Path(sys.argv[1])
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        sys.exit(1)

    with results_path.open("r") as f:
        results = json.load(f)

    # Experiments names as defined in q2q3_run_all_stats.EXPERIMENTS
    digits_models = ["perceptron_digits", "scratch_digits", "pytorch_digits"]
    faces_models = ["perceptron_faces", "scratch_faces", "pytorch_faces"]

    # Basic sanity check: ensure expected keys exist
    for name in digits_models + faces_models:
        if name not in results:
            raise KeyError(f"Missing experiment '{name}' in {results_path}")

    plot_dataset(results, digits_models, "Digits", "digits")
    plot_dataset(results, faces_models, "Faces", "faces")

    print("Wrote: digits_error.png, digits_time.png, faces_error.png, faces_time.png")


if __name__ == "__main__":
    main()

