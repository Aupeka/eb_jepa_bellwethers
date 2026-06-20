"""Headline figure: best ablation JEPA vs The Well field-space baselines.

Answers the track's research question on one plot — does the latent JEPA predictor
give more *stable* long-horizon autoregressive rollouts (field-space VRMSE) than the
neural-operator surrogates (FNO / TFNO / U-Net / CNext-U-Net)?

Reads:
  * baseline per-horizon VRMSE from <baselines-dir>/per_model_rollout_vrmse_aggregated.csv
    (the variance-pooled metric the JEPA eval uses). Falls back to the official per-sample
    CSV (clearly labeled) if the aggregated one hasn't been regenerated yet.
  * baseline param counts + stability horizons from <baselines-dir>/metrics.json.
  * JEPA per-horizon curves + param counts from the ablation summary.json.

Writes <out-dir>/comparison.png, comparison.csv, comparison.json.

Run:
  python -m examples.gray_scott.baselines.compare_jepa \
      --baselines-dir outputs/baselines/gray_scott_reaction_diffusion \
      --ablation-summary outputs/ablations/summary.json \
      --out-dir outputs/comparison/gray_scott
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE_MODELS = ["FNO", "TFNO", "U-Net", "CNextU-Net"]


def _read_curve_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    horizons = [int(float(r["horizon"])) for r in rows]
    cols = {k: np.array([float(r[k]) for r in rows], dtype=float)
            for k in rows[0] if k != "horizon"}
    return horizons, cols


def load_baselines(bdir):
    agg = os.path.join(bdir, "per_model_rollout_vrmse_aggregated.csv")
    off = os.path.join(bdir, "per_model_rollout_vrmse.csv")
    if os.path.exists(agg):
        path, metric = agg, "aggregated (variance-pooled)"
    elif os.path.exists(off):
        path, metric = off, "official per-sample (re-run baselines for aggregated)"
    else:
        raise FileNotFoundError(f"no baseline rollout CSV found in {bdir}")
    horizons, cols = _read_curve_csv(path)
    params, stab = {}, {}
    mpath = os.path.join(bdir, "metrics.json")
    if os.path.exists(mpath):
        models = json.load(open(mpath)).get("models", {})
        for name, d in models.items():
            if "n_params_millions" in d:
                params[name] = d["n_params_millions"]
            if "stability_horizon" in d:
                stab[name] = d["stability_horizon"]
    return horizons, cols, params, stab, metric


def _stability_horizon(curve, horizons):
    """First horizon whose VRMSE exceeds 1.0 (else '>H')."""
    for h, v in zip(horizons, curve):
        if not np.isfinite(v) or v > 1.0:
            return h
    return f">{horizons[-1]}"


def select_best(combos, criterion):
    def score(m):
        c = np.asarray(m["curves"]["jepa"], dtype=float)
        finite = c[np.isfinite(c)]
        if finite.size == 0:
            return np.inf
        if criterion in ("h1", "h10", "h30"):
            return m["vrmse"]["jepa"][criterion]
        return float(np.mean(finite))  # "mean": overall stability across horizons
    return min(combos.values(), key=score)


def main():
    p = argparse.ArgumentParser(description="JEPA vs Well baselines comparison figure")
    p.add_argument("--baselines-dir", required=True)
    p.add_argument("--ablation-summary", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--select", default="mean", choices=["mean", "h1", "h10", "h30"],
                   help="criterion to pick the headline JEPA combo (default: mean VRMSE over horizons)")
    p.add_argument("--dataset", default="gray_scott_reaction_diffusion")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    b_h, b_cols, b_params, b_stab, b_metric = load_baselines(args.baselines_dir)
    summary = json.load(open(args.ablation_summary))
    combos = summary["combos"]
    best = select_best(combos, args.select)
    j_h = best["horizons"]
    j_curve = np.asarray(best["curves"]["jepa"], dtype=float)
    pers = b_cols.get("persistence")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"FNO": "#1f77b4", "TFNO": "#ff7f0e", "U-Net": "#2ca02c", "CNextU-Net": "#9467bd"}
    for name in BASELINE_MODELS:
        if name in b_cols:
            ax.plot(b_h, b_cols[name], color=colors[name], lw=2, label=name)
    if pers is not None:
        ax.plot(b_h, pers, color="black", lw=1.5, ls="--", label="persistence")

    # other JEPA combos, faint (shows the K / regularizer sweep without clutter)
    for name, m in combos.items():
        if name == best["combo"]:
            continue
        ax.plot(m["horizons"], np.asarray(m["curves"]["jepa"], dtype=float),
                color="0.7", lw=1.0, alpha=0.6, zorder=1)
    ax.plot([], [], color="0.7", lw=1.0, label="other JEPA combos")
    ax.plot(j_h, j_curve, color="#d62728", lw=3.0, zorder=5,
            label=f"JEPA (best: {best['combo']}, {best['param_count_m']:.2f}M)")

    ax.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("autoregressive horizon", fontsize=13)
    ax.set_ylabel("field-space VRMSE (log)", fontsize=13)
    ax.set_title(f"{args.dataset}: JEPA vs Well baselines\n"
                 f"VRMSE = {b_metric}", fontsize=13)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    png = os.path.join(args.out_dir, "comparison.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)

    # ---- table (csv + json) ----
    def at(curve, horizons, h):
        c = np.asarray(curve, dtype=float)
        return float(c[min(h, len(c)) - 1]) if h <= len(c) else float("nan")

    table = []
    for name in BASELINE_MODELS:
        if name not in b_cols:
            continue
        table.append({"model": name, "type": "baseline",
                      "params_M": b_params.get(name),
                      "vrmse_h1": at(b_cols[name], b_h, 1),
                      "vrmse_h10": at(b_cols[name], b_h, 10),
                      "vrmse_h30": at(b_cols[name], b_h, 30),
                      "stability_horizon": b_stab.get(name, _stability_horizon(b_cols[name], b_h))})
    if pers is not None:
        table.append({"model": "persistence", "type": "baseline", "params_M": 0.0,
                      "vrmse_h1": at(pers, b_h, 1), "vrmse_h10": at(pers, b_h, 10),
                      "vrmse_h30": at(pers, b_h, 30),
                      "stability_horizon": _stability_horizon(pers, b_h)})
    for name, m in combos.items():
        c = m["curves"]["jepa"]
        table.append({"model": f"JEPA/{name}", "type": "jepa",
                      "params_M": m["param_count_m"],
                      "vrmse_h1": m["vrmse"]["jepa"]["h1"],
                      "vrmse_h10": m["vrmse"]["jepa"]["h10"],
                      "vrmse_h30": m["vrmse"]["jepa"]["h30"],
                      "stability_horizon": _stability_horizon(c, m["horizons"]),
                      "best": name == best["combo"]})

    with open(os.path.join(args.out_dir, "comparison.json"), "w") as f:
        json.dump({"dataset": args.dataset, "baseline_metric": b_metric,
                   "best_jepa": best["combo"], "select_criterion": args.select,
                   "table": table}, f, indent=2)
    with open(os.path.join(args.out_dir, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "type", "params_M", "vrmse_h1", "vrmse_h10", "vrmse_h30", "stability_horizon"])
        for r in table:
            w.writerow([r["model"], r["type"], r.get("params_M"),
                        f"{r['vrmse_h1']:.4f}", f"{r['vrmse_h10']:.4f}",
                        f"{r['vrmse_h30']:.4f}", r["stability_horizon"]])

    print(f"[compare] best JEPA = {best['combo']} ({best['param_count_m']:.2f}M)", flush=True)
    print(f"[compare] baseline metric: {b_metric}", flush=True)
    print(f"[compare] wrote {png}, comparison.csv, comparison.json", flush=True)


if __name__ == "__main__":
    main()
