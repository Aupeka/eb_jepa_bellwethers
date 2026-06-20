"""Final presentation figure: Well field-space baselines + Poseidon (+ optional best JEPA)
on one VRMSE-vs-horizon plot.

Poseidon was evaluated by a *separate* harness (scOT checkpoint, resize-to-model, time
rescaling), so its curve is not in the baseline ``rollout`` dict and never auto-appears in
``vrmse_vs_horizon.png``. This script merges Poseidon's per-horizon curve (same CSV format
as the baselines: ``horizon, <model>, persistence, mean``) onto the baseline curves and
re-plots a single slide-ready figure.

Reads:
  * baselines:  <baselines-dir>/per_model_rollout_vrmse.csv (official) or
                <baselines-dir>/per_model_rollout_vrmse_aggregated.csv (variance-pooled),
                plus metrics.json for param counts.
  * Poseidon:   <poseidon-dir>/per_model_rollout_vrmse[_aggregated].csv (column "Poseidon*"),
                or pass --poseidon-csv directly; optional --poseidon-metrics for params.
  * JEPA (opt): ablation summary.json (best combo overlaid).

Writes <out-dir>/final_vrmse_vs_horizon.png, final_comparison.csv, final_comparison.json.

Run:
  python -m examples.gray_scott.baselines.final_plot \
      --baselines-dir outputs/baselines/gray_scott_reaction_diffusion \
      --poseidon-dir  outputs/baselines/gray_scott_reaction_diffusion_poseidon \
      --out-dir       outputs/comparison/gray_scott_final
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.gray_scott.baselines.viz import (
    comparison_ymax,
    model_color,
    plot_comparison_curve,
    plot_reference_curves,
    stability_horizon,
    truncate_curve_for_display,
)

BASELINE_MODELS = ["FNO", "TFNO", "U-Net", "CNextU-Net"]
_CSV = {"official": "per_model_rollout_vrmse.csv",
        "aggregated": "per_model_rollout_vrmse_aggregated.csv"}


def _read_curve_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    horizons = [int(float(r["horizon"])) for r in rows]
    cols = {k: np.array([float(r[k]) for r in rows], dtype=float)
            for k in rows[0] if k != "horizon"}
    return horizons, cols


def _pick_csv(d, metric):
    """Prefer the requested metric CSV; fall back to the other one (labeled)."""
    want = os.path.join(d, _CSV[metric])
    if os.path.exists(want):
        return want, metric
    other = "official" if metric == "aggregated" else "aggregated"
    alt = os.path.join(d, _CSV[other])
    if os.path.exists(alt):
        return alt, other
    raise FileNotFoundError(f"no rollout CSV ({_CSV['official']} / {_CSV['aggregated']}) in {d}")


def _params_from_metrics(path):
    out = {}
    if path and os.path.exists(path):
        for name, d in json.load(open(path)).get("models", {}).items():
            if isinstance(d, dict) and "n_params_millions" in d:
                out[name] = d["n_params_millions"]
    return out


def _color(name, idx):
    n = name.lower()
    if "poseidon" in n:
        return "#9467bd"
    if "jepa" in n:
        return "#8c564b"
    return model_color(name, idx)


def _at(curve, horizons, h):
    c = np.asarray(curve, dtype=float)
    return float(c[h - 1]) if h <= len(c) else float("nan")


def main():
    p = argparse.ArgumentParser(description="Final baselines + Poseidon (+JEPA) VRMSE figure")
    p.add_argument("--baselines-dir", required=True)
    p.add_argument("--poseidon-dir", default=None,
                   help="Poseidon output dir (same CSV layout as the baselines)")
    p.add_argument("--poseidon-csv", default=None,
                   help="explicit Poseidon per-horizon CSV (overrides --poseidon-dir)")
    p.add_argument("--poseidon-metrics", default=None,
                   help="optional Poseidon metrics.json for param count")
    p.add_argument("--ablation-summary", default=None, help="optional JEPA ablation summary.json")
    p.add_argument("--select", default="mean", choices=["mean", "h1", "h10", "h30"],
                   help="criterion to pick the headline JEPA combo")
    p.add_argument("--metric", default="official", choices=["official", "aggregated"],
                   help="VRMSE definition to plot (default official, the gray_scott headline)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset", default="gray_scott_reaction_diffusion")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    b_path, b_metric = _pick_csv(args.baselines_dir, args.metric)
    horizons, b_cols = _read_curve_csv(b_path)
    h = np.asarray(horizons, dtype=float)
    b_params = _params_from_metrics(os.path.join(args.baselines_dir, "metrics.json"))

    # ---- Poseidon (separate harness, same CSV format) ----
    pose_name, pose_curve = None, None
    pose_csv = args.poseidon_csv
    if pose_csv is None and args.poseidon_dir:
        pose_csv, _ = _pick_csv(args.poseidon_dir, args.metric)
    if pose_csv:
        _, p_cols = _read_curve_csv(pose_csv)
        learned = [k for k in p_cols if k not in ("persistence", "mean")]
        if learned:
            pose_name = next((k for k in learned if "poseidon" in k.lower()), learned[0])
            pose_curve = p_cols[pose_name]
    pose_params = _params_from_metrics(args.poseidon_metrics).get(pose_name) if pose_name else None

    # ---- optional best JEPA ----
    jepa_name, jepa_curve, jepa_h, jepa_params, jepa_combos = None, None, None, None, {}
    if args.ablation_summary and os.path.exists(args.ablation_summary):
        jepa_combos = json.load(open(args.ablation_summary)).get("combos", {})
        if jepa_combos:
            from examples.gray_scott.baselines.compare_jepa import select_best
            best = select_best(jepa_combos, args.select)
            jepa_name = f"JEPA ({best['combo']})"
            jepa_curve = np.asarray(best["curves"]["jepa"], dtype=float)
            jepa_h = np.asarray(best["horizons"], dtype=float)
            jepa_params = best.get("param_count_m")

    # ---- readable y-ceiling (FNO/TFNO excluded; early horizons only) ----
    all_curves = dict(b_cols)
    if pose_curve is not None:
        all_curves[pose_name] = pose_curve
    if jepa_curve is not None:
        all_curves[jepa_name] = jepa_curve
    ymax = comparison_ymax(h, all_curves)

    # ---- figure ----
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(10, 6.2))

    plot_models = [(n, b_cols[n]) for n in BASELINE_MODELS if n in b_cols]
    if pose_curve is not None:
        plot_models.append((pose_name, pose_curve))
    if jepa_curve is not None:
        plot_models.append((jepa_name, jepa_curve))

    for idx, (name, curve) in enumerate(plot_models):
        xh = jepa_h if (jepa_curve is not None and name == jepa_name) else h
        is_jepa = name == jepa_name
        plot_comparison_curve(
            ax, xh, curve, name=name, color=_color(name, idx), ymax=ymax,
            lw=3.0 if is_jepa else 2.0, zorder=5 if is_jepa else 3)

    # faint sweep of the other JEPA combos for context
    for cname, m in jepa_combos.items():
        if jepa_name and cname in jepa_name:
            continue
        jc = np.asarray(m["curves"]["jepa"], dtype=float)
        jx = np.asarray(m["horizons"], dtype=float)
        h_cut, _ = truncate_curve_for_display(jc, ymax)
        if h_cut > 0:
            ax.plot(jx[:h_cut], jc[:h_cut], color="0.75", lw=1.0, alpha=0.6, zorder=1)
    if jepa_combos and len(jepa_combos) > 1:
        ax.plot([], [], color="0.75", lw=1.0, label="other JEPA combos")

    plot_reference_curves(ax, h, b_cols, ymax)

    ax.set_yscale("log")
    ax.set_ylim(top=ymax)
    ax.axhline(1.0, color="gray", ls=":", lw=1.2)
    ax.axhspan(1.0, ymax, color="gray", alpha=0.06)
    ax.text(h[-1], 1.0, " worse than mean predictor", color="gray",
            va="bottom", ha="right", fontsize=11)
    ax.set_xlabel("autoregressive rollout horizon (steps)")
    ax.set_ylabel("VRMSE (field-averaged, physical space)")
    ax.set_title(f"{args.dataset}: field-space baselines vs Poseidon"
                 + (" vs JEPA" if jepa_curve is not None else "")
                 + f"\nVRMSE = {b_metric}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=11, loc="best")
    fig.tight_layout()
    png = os.path.join(args.out_dir, "final_vrmse_vs_horizon.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)

    # ---- table (csv + json) ----
    table = []

    def add_row(name, curve, params, xh):
        table.append({
            "model": name,
            "params_M": params,
            "vrmse_h1": _at(curve, xh, 1),
            "vrmse_h10": _at(curve, xh, 10),
            "vrmse_h30": _at(curve, xh, 30),
            "stability_horizon": stability_horizon(curve, len(curve)),
        })

    for name in BASELINE_MODELS:
        if name in b_cols:
            add_row(name, b_cols[name], b_params.get(name), horizons)
    if pose_curve is not None:
        add_row(pose_name, pose_curve, pose_params, horizons)
    if jepa_curve is not None:
        add_row(jepa_name, jepa_curve, jepa_params, list(jepa_h.astype(int)))
    for ref in ("persistence", "mean"):
        if ref in b_cols:
            add_row(ref, b_cols[ref], 0.0, horizons)

    with open(os.path.join(args.out_dir, "final_comparison.json"), "w") as f:
        json.dump({"dataset": args.dataset, "metric": b_metric,
                   "poseidon_model": pose_name, "table": table}, f, indent=2)
    with open(os.path.join(args.out_dir, "final_comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "params_M", "vrmse_h1", "vrmse_h10", "vrmse_h30", "stability_horizon"])
        for r in table:
            w.writerow([r["model"], r["params_M"],
                        f"{r['vrmse_h1']:.4f}", f"{r['vrmse_h10']:.4f}",
                        f"{r['vrmse_h30']:.4f}", r["stability_horizon"]])

    print(f"[final] baseline metric: {b_metric}", flush=True)
    if pose_name:
        print(f"[final] Poseidon column: {pose_name}"
              + (f" ({pose_params:.2f}M)" if pose_params else " (no param count)"), flush=True)
    if jepa_name:
        print(f"[final] JEPA overlay: {jepa_name}", flush=True)
    print(f"[final] wrote {png}, final_comparison.csv, final_comparison.json", flush=True)


if __name__ == "__main__":
    main()
