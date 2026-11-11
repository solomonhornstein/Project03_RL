"""
compare_runs.py -- plots learning curves from several experiments on one chart.

Usage:
  python compare_runs.py                         plots every scores_*.csv in this folder
  python compare_runs.py scores_baseline.csv scores_alpha03.csv   plots just these

Saves the chart as learning_curves.png and prints a summary table.
Requires matplotlib:  pip install matplotlib

AI USE (required by the DA Honor Code):
This file was written by Claude (Anthropic's AI) at my request. I used it
to compare my experiment runs and made the charts in the README with it. See AI_USAGE.md for details.
"""

import csv
import glob
import sys

import matplotlib.pyplot as plt

WINDOW = 100  # moving-average window; raw scores are too noisy to read


def load_scores(path):
    with open(path) as f:
        return [int(row["score"]) for row in csv.DictReader(f)]


def moving_average(values, window):
    out, total = [], 0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def main():
    paths = sys.argv[1:] or sorted(glob.glob("scores_*.csv"))
    if not paths:
        print("No scores_*.csv files found.")
        return

    print(f"{'run':<22}{'episodes':>9}{'final avg':>11}{'best':>6}{'1st ep avg>=50':>16}")
    for path in paths:
        scores = load_scores(path)
        name = path.removeprefix("scores_").removesuffix(".csv")
        avg = moving_average(scores, WINDOW)
        plt.plot(range(1, len(avg) + 1), avg, label=name)

        # "Speed of learning": first episode where the moving average reaches 50
        reached = next((i + 1 for i, a in enumerate(avg) if a >= 50), "never")
        print(f"{name:<22}{len(scores):>9}{avg[-1]:>11.1f}{max(scores):>6}{reached!s:>16}")

    plt.xlabel("Episode")
    plt.ylabel(f"Score ({WINDOW}-episode moving average)")
    plt.title("Flappy Bird Q-learning: experiment comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150)
    print("\nSaved learning_curves.png")
    plt.show()


if __name__ == "__main__":
    main()