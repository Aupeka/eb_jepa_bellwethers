"""Cross-dataset stability summary.

Reads each dataset's ``metrics.json`` (written by ``eval_baselines.py``) and produces a
single ``stability_summary.png`` + ``stability_summary.csv`` under ``outputs/baselines/``.
The headline metric is the per-model *stability horizon* = number of leading rollout steps
the model stays at/below VRMSE 1 (i.e. at least as good as the mean predictor).

  PYTHONPATH=. python -m examples.gray_scott.baselines.summarize
  PYTHONPATH=. python -m examples.gray_scott.baselines.summarize --datasets gray_scott_reaction_diffusion turbulent_radiative_layer_2D
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.gray_scott.baselines.eval_baselines import resolve_out_dir
from examples.gray_scott.baselines.viz import model_color

DEFAULT_DATASETS = [
    "gray_scott_reaction_diffusion",
    "turbulent_radiative_layer_2D",
    "rayleigh_benard",
]
BASELINES = ["FNO", "TFNO", "U-Net", "CNextU-Net"]


def load_metrics(dataset):
    path = os.path.join(resolve_out_dir(dataset), "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Cross-dataset stability summary")
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    args = ap.parse_args()

    summary_dir = os.path.dirname(resolve_out_dir(args.datasets[0]))
    os.makedirs(summary_dir, exist_ok=True)

    found = {}
    for ds in args.datasets:
        m = load_metrics(ds)
        if m is None:
            print(f"[skip] no metrics.json for {ds} (run the eval first)", flush=True)
            continue
        found[ds] = m
    if not found:
        raise RuntimeError("No metrics.json found for any requested dataset.")

    # CSV: one row per (dataset, model)
    csv_path = os.path.join(summary_dir, "stability_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "n_params_M", "H", "stability_horizon",
                    "one_step_vrmse", "rollout_6:12", "rollout_13:30"])
        for ds, m in found.items():
            H = m["protocol"]["H"]
            for name, entry in m["models"].items():
                wins = entry.get("rollout_window_vrmse_official", {})
                w.writerow([
                    ds, name, entry.get("n_params_millions", ""), H,
                    entry.get("stability_horizon", ""),
                    entry.get("one_step_vrmse_field_avg", ""),
                    wins.get("6:12", ""), wins.get("13:30", ""),
                ])

    # Grouped bar chart: stability horizon per model per dataset (baselines only)
    datasets = list(found.keys())
    n_ds = len(datasets)
    width = 0.8 / max(len(BASELINES), 1)
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(2.6 * n_ds + 3, 5.5))
    x = np.arange(n_ds)
    for i, model in enumerate(BASELINES):
        vals = []
        for ds in datasets:
            entry = found[ds]["models"].get(model, {})
            vals.append(entry.get("stability_horizon", 0))
        ax.bar(x + i * width, vals, width, label=model, color=model_color(model, i))
    # reference: full horizon line per dataset
    for j, ds in enumerate(datasets):
        H = found[ds]["protocol"]["H"]
        ax.hlines(H, x[j] - 0.1, x[j] + 0.8, color="gray", ls=":", lw=1.2)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels([d.replace("_", "\n") for d in datasets], fontsize=10)
    ax.set_ylabel("stability horizon\n(steps with VRMSE <= 1)")
    ax.set_title("Long-horizon stability of field-space surrogates\n(higher = stable longer; dotted = full rollout H)")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    png_path = os.path.join(summary_dir, "stability_summary.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    plt.rcParams.update({"font.size": 10})
    print(f"[done] wrote {csv_path} and {png_path}", flush=True)


if __name__ == "__main__":
    main()
