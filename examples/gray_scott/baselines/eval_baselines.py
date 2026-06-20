"""Dataset-agnostic evaluation of The Well's official surrogate baselines.

Loads the published Hugging Face checkpoints (FNO, TFNO, UNetClassic, UNetConvNext)
for a given 2D Well dataset and scores them under The Well's official protocol
(``n_steps_input=4``, ZScore normalization, stride 1), reusing the_well's own
``WellDataModule``, normalization, metadata and VRMSE metric. We report:

  * one-step VRMSE (sliding windows from ground truth, like paper Table 2),
  * per-horizon VRMSE for the autoregressive rollout from the start of each
    trajectory (1..H),
  * window-averaged VRMSE over 6:12 and 13:30 (like paper Table 3),
  * cheap persistence and spatial-mean reference baselines,

and write metrics.json / metrics.csv / per_model_rollout_vrmse.csv plus two plots.

This computes MEASURED checkpoint numbers under the official protocol; it is NOT a
claim to exactly reproduce the paper tables (checkpoint/version/normalization drift,
see the_well issue #49). Run:

  python -m examples.gray_scott.baselines.eval_baselines --dataset turbulent_radiative_layer_2D --smoke
  python -m examples.gray_scott.baselines.eval_baselines --dataset gray_scott_reaction_diffusion --H 30
"""
import argparse
import csv
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from the_well.benchmark.models import FNO, TFNO, UNetClassic, UNetConvNext
from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization

from examples.gray_scott.baselines.well_rollout import (
    ROLLOUT_WINDOWS,
    WellEvaluator,
    field_average,
    field_names,
    window_means,
)
from examples.gray_scott.baselines.viz import (
    make_rollout_gif,
    plot_per_field_vrmse,
    plot_spectral_diagnostic,
    plot_vrmse_with_bands,
    stability_horizon,
)

MODEL_KINDS = {
    "FNO": FNO,
    "TFNO": TFNO,
    "UNetClassic": UNetClassic,
    "UNetConvNext": UNetConvNext,
}
HERE = os.path.dirname(os.path.abspath(__file__))


def resolve_well_base_path() -> str:
    """Where The Well data lives. Prefer a staged local copy on /lustre/work, else stream from HF."""
    for env in ("THE_WELL_BASE_PATH", "EBJEPA_DSETS"):
        p = os.environ.get(env)
        if p:
            return p
    return "hf://datasets/polymathic-ai/"


def resolve_out_dir(dataset: str) -> str:
    base = os.environ.get("EBJEPA_BASELINE_OUT")
    if not base:
        work = os.environ.get("EBJEPA_WORK")
        base = os.path.join(work, "outputs") if work else "outputs"
    return os.path.join(base, "baselines", dataset)


def load_config(args) -> OmegaConf:
    cfg_path = args.cfg or os.path.join(HERE, "cfgs", f"{args.dataset}.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No config for dataset '{args.dataset}' at {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    if args.smoke:
        cfg = OmegaConf.merge(cfg, cfg.get("smoke", {}))
    if args.H is not None:
        cfg.H = args.H
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    return cfg


def build_datasets(cfg, well_base_path: str):
    """Build the one-step (test) and rollout (full-trajectory) datasets directly.

    NOTE: we build ``WellDataset`` directly instead of via ``WellDataModule`` because in
    the_well 1.2.0 the datamodule does NOT forward ``use_normalization`` /
    ``normalization_type`` to its test/rollout splits (fixed on main after 1.2.0), which
    silently feeds RAW (un-normalized) inputs to the checkpoints. ``WellDataset`` honors
    these args, so normalization (the protocol the checkpoints expect) is guaranteed here.
    """
    common = dict(
        well_base_path=well_base_path,
        well_dataset_name=cfg.dataset,
        well_split_name="test",
        use_normalization=True,
        normalization_type=ZScoreNormalization,
        n_steps_input=int(cfg.n_steps_input),
        n_steps_output=1,
        min_dt_stride=1,
        max_dt_stride=1,
    )
    test_dataset = WellDataset(**common)
    rollout_dataset = WellDataset(full_trajectory_mode=True, max_rollout_steps=int(cfg.H), **common)
    return test_dataset, rollout_dataset


def make_dataloader(dataset, batch_size: int, workers: int, drop_last: bool = True) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last,
        num_workers=workers, pin_memory=True,
    )


def load_models(cfg, meta, device):
    expected_in = int(cfg.n_steps_input) * meta.n_fields
    expected_out = meta.n_fields
    models = {}
    param_counts = {}
    for entry in cfg.models:
        kind = MODEL_KINDS[entry.kind]
        model = kind.from_pretrained(entry.id).to(device).eval().float()
        # Verify the checkpoint matches this dataset (field count / input history).
        dim_in = getattr(model, "dim_in", None)
        dim_out = getattr(model, "dim_out", None)
        if dim_in is not None and dim_in != expected_in:
            raise ValueError(
                f"{entry.name} ({entry.id}) has dim_in={dim_in}, expected "
                f"{expected_in} = n_steps_input({cfg.n_steps_input}) * n_fields({meta.n_fields}). "
                "Checkpoint does not match this dataset."
            )
        if dim_out is not None and dim_out != expected_out:
            raise ValueError(
                f"{entry.name} ({entry.id}) has dim_out={dim_out}, expected {expected_out} fields."
            )
        n_params = sum(p.numel() for p in model.parameters())
        models[entry.name] = model
        param_counts[entry.name] = n_params
        print(f"[load] {entry.name:11s} {entry.id}  dim_in={dim_in} dim_out={dim_out} "
              f"params={n_params/1e6:.2f}M", flush=True)
    return models, param_counts


def run_one_step(ev: WellEvaluator, models, test_loader, max_batches=None):
    """One-step VRMSE on sliding ground-truth windows (test split). Returns {model: [C]}."""
    per_field = {name: [] for name in models}
    n = 0
    for bi, batch in enumerate(test_loader):
        if max_batches is not None and bi >= max_batches:
            break
        for name, model in models.items():
            y_pred, y_ref = ev.rollout_model(model, batch, max_rollout_steps=1)
            assert y_pred.shape == y_ref.shape, f"{name}: {y_pred.shape} vs {y_ref.shape}"
            assert y_pred.shape[-1] == ev.meta.n_fields, "unexpected field count"
            per_field[name].append(ev.vrmse_per_step(y_pred, y_ref)[0])  # [C] (single step)
        n += 1
        if bi % 20 == 0:
            cap = f"/{max_batches}" if max_batches else ""
            print(f"   [one-step] batch {bi}{cap}", flush=True)
    if n == 0:
        raise RuntimeError("No test batches consumed for one-step eval.")
    return {name: torch.stack(v).mean(0).numpy() for name, v in per_field.items()}


def _bands_from_per_traj(per_traj_list):
    """Stack per-batch ``[B, T]`` VRMSE into ``[N_traj, T]`` and derive mean + percentile bands.

    Non-finite (diverged) trajectories are kept in the mean (so it goes inf where the model
    blows up) but mapped to nan for the percentile band so the band ends at divergence."""
    arr = torch.cat(per_traj_list, dim=0).numpy()  # [N_traj, T]
    finite = np.where(np.isfinite(arr), arr, np.nan)
    with warnings.catch_warnings():  # a fully-diverged horizon is an all-nan slice -> nan band
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanpercentile(finite, 50, axis=0)
        p10 = np.nanpercentile(finite, 10, axis=0)
        p90 = np.nanpercentile(finite, 90, axis=0)
    return {
        "official": np.mean(arr, axis=0),  # inf-aware mean curve (matches prior behavior)
        "median": median,
        "p10": p10,
        "p90": p90,
        "per_traj": arr,
    }


def run_rollout(ev: WellEvaluator, models, rollout_loader, H, max_trajectories=None):
    """Autoregressive rollout from trajectory start. Returns per-step VRMSE arrays + viz cache.

    The rollout loader may be batched (B>1); ``max_trajectories`` caps the number of
    TRAJECTORIES (samples) consumed, not batches."""
    series = list(models) + ["persistence", "mean"]
    per_traj = {name: [] for name in series}           # list of [B, T] field-averaged per sample
    agg_num = {name: None for name in series}           # running sum [T, C]
    agg_den = {name: None for name in series}
    viz = None
    seen = 0

    def _accumulate(name, y_pred, y_ref):
        per_traj[name].append(ev.vrmse_per_sample(y_pred, y_ref))  # [B, T]
        num, den = ev.vrmse_terms_per_step(y_pred, y_ref)
        agg_num[name] = num if agg_num[name] is None else agg_num[name] + num
        agg_den[name] = den if agg_den[name] is None else agg_den[name] + den

    for bi, batch in enumerate(rollout_loader):
        first_batch = bi == 0
        viz_models = {}
        for name, model in models.items():
            y_pred, y_ref = ev.rollout_model(model, batch, max_rollout_steps=H)
            assert y_pred.shape == y_ref.shape, f"{name}: {y_pred.shape} vs {y_ref.shape}"
            _accumulate(name, y_pred, y_ref)
            if first_batch:
                viz_models[name] = y_pred[0].float().cpu().numpy()  # [T, *spatial, C]
        persistence, mean_pred, y_ref = ev.persistence_and_mean(batch, max_rollout_steps=H)
        _accumulate("persistence", persistence, y_ref)
        _accumulate("mean", mean_pred, y_ref)
        if first_batch:
            viz = {"truth": y_ref[0].float().cpu().numpy(), "models": viz_models}
        seen += y_ref.shape[0]
        if bi % 10 == 0:
            cap = f"/{max_trajectories}" if max_trajectories else ""
            print(f"   [rollout] trajectory {seen}{cap}", flush=True)
        if max_trajectories is not None and seen >= max_trajectories:
            break
    if seen == 0:
        raise RuntimeError("No rollout trajectories consumed.")

    out = {}
    for name in series:
        bands = _bands_from_per_traj(per_traj[name])
        agg = torch.sqrt(agg_num[name] / (agg_den[name] + 1e-7))  # [T, C]
        bands["aggregated"] = field_average(agg)                  # [T]
        bands["aggregated_per_field"] = agg.numpy()               # [T, C]
        out[name] = bands
    return out, viz


def write_outputs(out_dir, cfg, meta, one_step, rollout, viz, param_counts):
    os.makedirs(out_dir, exist_ok=True)
    fields = field_names(meta)
    model_names = [e.name for e in cfg.models]
    H = int(cfg.H)
    horizons = list(range(1, H + 1))

    # metrics.json (rich) + metrics.csv (model x summary)
    metrics = {
        "dataset": cfg.dataset,
        "protocol": {
            "n_steps_input": int(cfg.n_steps_input),
            "n_steps_output": 1,
            "stride": 1,
            "normalization": "ZScore (denormalized/physical-space VRMSE)",
            "H": H,
            "windows": {k: list(v) for k, v in ROLLOUT_WINDOWS.items()},
            "vrmse": "official per-sample the_well.benchmark.metrics.VRMSE; 'aggregated' = sqrt(sum MSE / sum var)",
            "bands": "per-trajectory spread (median + 10-90th percentile) from a single full-test pass",
        },
        "fields": fields,
        "models": {},
    }
    for name in model_names:
        r = rollout[name]
        metrics["models"][name] = {
            "n_params": int(param_counts[name]),
            "n_params_millions": round(param_counts[name] / 1e6, 3),
            "one_step_vrmse_field_avg": float(np.mean(one_step[name])),
            "one_step_vrmse_per_field": {f: float(v) for f, v in zip(fields, one_step[name])},
            "rollout_window_vrmse_official": window_means(r["official"]),
            "rollout_window_vrmse_aggregated": window_means(r["aggregated"]),
            "rollout_window_vrmse_aggregated_per_field": {
                f: window_means(np.asarray(r["aggregated_per_field"], dtype=float)[:, c])
                for c, f in enumerate(fields)
            },
            "stability_horizon": stability_horizon(r["official"], H),
        }
    for name in ("persistence", "mean"):
        metrics["models"][name] = {
            "rollout_window_vrmse_official": window_means(rollout[name]["official"]),
            "stability_horizon": stability_horizon(rollout[name]["official"], H),
        }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_params_M", "one_step_vrmse"]
                   + [f"rollout_{k}" for k in ROLLOUT_WINDOWS] + ["stability_horizon"])
        for name in model_names:
            m = metrics["models"][name]
            wins = m["rollout_window_vrmse_official"]
            w.writerow([name, f"{m['n_params_millions']:.3f}", f"{np.mean(one_step[name]):.5f}"]
                       + [f"{wins[k]:.5f}" for k in ROLLOUT_WINDOWS] + [m["stability_horizon"]])
        for name in ("persistence", "mean"):
            m = metrics["models"][name]
            wins = m["rollout_window_vrmse_official"]
            w.writerow([name, "", ""] + [f"{wins[k]:.5f}" for k in ROLLOUT_WINDOWS] + [m["stability_horizon"]])

    with open(os.path.join(out_dir, "per_model_rollout_vrmse.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["horizon"] + model_names + ["persistence", "mean"]
        w.writerow(cols)
        for i, h in enumerate(horizons):
            row = [h] + [f"{rollout[n]['official'][i]:.6f}" for n in model_names]
            row += [f"{rollout['persistence']['official'][i]:.6f}", f"{rollout['mean']['official'][i]:.6f}"]
            w.writerow(row)

    # Variance-pooled (aggregated) per-horizon curve — same VRMSE definition the JEPA
    # eval uses (sqrt(sum MSE / sum var)), so JEPA can be overlaid apples-to-apples.
    with open(os.path.join(out_dir, "per_model_rollout_vrmse_aggregated.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["horizon"] + model_names + ["persistence", "mean"]
        w.writerow(cols)
        for i, h in enumerate(horizons):
            row = [h] + [f"{rollout[n]['aggregated'][i]:.6f}" for n in model_names]
            row += [f"{rollout['persistence']['aggregated'][i]:.6f}", f"{rollout['mean']['aggregated'][i]:.6f}"]
            w.writerow(row)

    # per_model_rollout_vrmse_band.csv: per-horizon median / p10 / p90 for every model + baselines
    band_names = model_names + ["persistence", "mean"]
    with open(os.path.join(out_dir, "per_model_rollout_vrmse_band.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["horizon"]
        for n in band_names:
            cols += [f"{n}_median", f"{n}_p10", f"{n}_p90"]
        w.writerow(cols)
        for i, h in enumerate(horizons):
            row = [h]
            for n in band_names:
                r = rollout[n]
                row += [f"{r['median'][i]:.6f}", f"{r['p10'][i]:.6f}", f"{r['p90'][i]:.6f}"]
            w.writerow(row)

    plot_vrmse_with_bands(out_dir, cfg, rollout, horizons)
    plot_per_field_vrmse(out_dir, cfg, meta, rollout, horizons)
    if viz is not None:
        _plot_rollout_comparison(out_dir, cfg, meta, viz, H)
        make_rollout_gif(out_dir, cfg, meta, viz, H)
        plot_spectral_diagnostic(out_dir, cfg, meta, viz)
    print(f"[done] wrote outputs -> {out_dir}", flush=True)


def _plot_rollout_comparison(out_dir, cfg, meta, viz, H):
    fld = int(cfg.get("viz_field", meta.n_fields - 1))
    steps = sorted(set([1, max(1, H // 3), max(1, 2 * H // 3), H]))
    rows = [("ground truth", viz["truth"])] + [(e.name, viz["models"][e.name]) for e in cfg.models]
    fig, axes = plt.subplots(len(rows), len(steps), figsize=(2.4 * len(steps), 2.2 * len(rows)), squeeze=False)
    vmin, vmax = np.percentile(viz["truth"][..., fld], [1, 99])
    for ri, (label, arr) in enumerate(rows):
        for ci, h in enumerate(steps):
            ax = axes[ri][ci]
            ax.imshow(arr[h - 1, ..., fld], vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"h={h}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle(f"{cfg.dataset}: rollout vs ground truth (field '{field_names(meta)[fld]}')", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rollout_comparison.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="The Well surrogate-baseline evaluator")
    ap.add_argument("--dataset", default="gray_scott_reaction_diffusion")
    ap.add_argument("--cfg", default=None, help="explicit config path (overrides --dataset)")
    ap.add_argument("--smoke", action="store_true", help="fast debug run (few trajectories, small H)")
    ap.add_argument("--H", type=int, default=None, help="rollout horizon (overrides config)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--rollout-batch-size", type=int, default=None,
                    help="batch size for the autoregressive rollout loader (overrides config; default 16)")
    ap.add_argument("--workers", type=int, default=None, help="data loader workers (overrides config)")
    ap.add_argument("--max-one-step-batches", type=int, default=None,
                    help="cap one-step eval to N batches (subsample test set; overrides config)")
    ap.add_argument("--max-trajectories", type=int, default=None,
                    help="cap rollout eval to N trajectories (subsample test set; overrides config)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args)
    if args.workers is not None:
        cfg.data_workers = args.workers
    if args.rollout_batch_size is not None:
        cfg.rollout_batch_size = args.rollout_batch_size
    if args.max_one_step_batches is not None:
        cfg.max_one_step_batches = args.max_one_step_batches
    if args.max_trajectories is not None:
        cfg.max_trajectories = args.max_trajectories
    device = torch.device(args.device)
    well_base_path = resolve_well_base_path()
    out_dir = resolve_out_dir(cfg.dataset)
    print(f"[cfg] dataset={cfg.dataset} H={cfg.H} batch_size={cfg.batch_size} smoke={args.smoke}")
    print(f"[cfg] well_base_path={well_base_path}")
    print(f"[cfg] out_dir={out_dir}")

    test_dataset, rollout_dataset = build_datasets(cfg, well_base_path)
    meta = test_dataset.metadata
    dset_norm = test_dataset.norm
    assert dset_norm is not None, "normalization not enabled — checkpoints expect normalized inputs"
    workers = int(cfg.get("data_workers", 4))
    rollout_bs = int(cfg.get("rollout_batch_size", 16))
    test_loader = make_dataloader(test_dataset, int(cfg.batch_size), workers)
    rollout_loader = make_dataloader(rollout_dataset, rollout_bs, workers, drop_last=False)
    print(f"[data] {meta.dataset_name}: n_fields={meta.n_fields} fields={field_names(meta)} "
          f"resolution={tuple(meta.spatial_resolution)} rollout_batch_size={rollout_bs}", flush=True)

    ev = WellEvaluator(meta, dset_norm, device, is_delta=False)
    models, param_counts = load_models(cfg, meta, device)

    max_one_step = int(cfg.get("max_one_step_batches", 0)) or None
    max_traj = int(cfg.get("max_trajectories", 0)) or None

    print("[eval] one-step VRMSE ...", flush=True)
    one_step = run_one_step(ev, models, test_loader, max_batches=max_one_step)
    print("[eval] autoregressive rollout ...", flush=True)
    rollout, viz = run_rollout(ev, models, rollout_loader, int(cfg.H), max_trajectories=max_traj)

    for name in models:
        v = rollout[name]["official"]
        # Autoregressive surrogates (esp. FNO/TFNO) can genuinely diverge to inf/nan at
        # long horizons. That is a reportable result, not a crash: warn and keep going so
        # outputs are still written for every model (the diverging step is recorded as inf/nan).
        if not np.isfinite(v).all():
            first_bad = int(np.argmax(~np.isfinite(v))) + 1
            print(f"   [warn] {name}: non-finite VRMSE from horizon {first_bad} "
                  f"(rollout diverged) - recorded as inf/nan", flush=True)
        hH_str = f"{v[-1]:.4f}" if np.isfinite(v[-1]) else "inf"
        print(f"   {name:11s} one-step={np.mean(one_step[name]):.4f}  "
              f"h1={v[0]:.4f} hH={hH_str}  windows={window_means(v)}", flush=True)

    write_outputs(out_dir, cfg, meta, one_step, rollout, viz, param_counts)


if __name__ == "__main__":
    main()
