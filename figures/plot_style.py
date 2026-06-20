"""Shared helpers for the Gray-Scott presentation notebooks.

This module is intentionally light (numpy + matplotlib only at import time) so the
two notebooks under ``figures/`` run anywhere — torch / the_well are imported lazily
and only when actually used. It provides:

  * ``setup_style()``          — talk-quality matplotlib defaults.
  * ``FIELD_CMAPS``            — consistent colormaps for the two chemical fields.
  * ``savefig(fig, name)``     — write ``out/<name>.png`` (dpi 200) and ``out/<name>.svg``.
  * data access (``load_demo_trajectory`` / ``list_regimes``) that transparently
    uses local HDF5 -> the_well HF streaming -> a synthetic Gray-Scott fallback,
    so a notebook never crashes just because the 15 TB dataset is not staged.

The dataset conventions mirror ``eb_jepa/datasets/gray_scott/dataset.py``:
a trajectory is ``[2, T, 128, 128]`` (chemical fields A, B), 1001 steps per traj,
one (F, k) reaction-diffusion regime per HDF5 file.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------------- #
# Dataset constants (kept in sync with eb_jepa/datasets/gray_scott/dataset.py)
# ----------------------------------------------------------------------------- #
MEAN = np.array([0.729227819893941, 0.09658732411527585], dtype=np.float32)  # A, B
STD = np.array([0.23988766176449572, 0.12366442840472558], dtype=np.float32)  # A, B
NT = 1001  # timesteps per trajectory
FIELD_NAMES = ("A", "B")
FIELD_CMAPS = {"A": "viridis", "B": "magma"}

# Output directory for generated figures (sibling of this file).
OUT_DIR = Path(__file__).resolve().parent / "out"

# Default local store for downloaded Well data. The repo's .gitignore ignores
# /datasets/, so this stays out of version control.
DSET_BASE = Path(__file__).resolve().parents[1] / "datasets" / "the_well"
# HuggingFace dataset repo template for The Well datasets.
WELL_REPO_FMT = "polymathic-ai/{dataset}"


# ----------------------------------------------------------------------------- #
# Matplotlib styling
# ----------------------------------------------------------------------------- #
def setup_style() -> None:
    """Apply presentation-grade matplotlib defaults (large fonts, clean axes)."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "image.cmap": "viridis",
            "lines.linewidth": 2.2,
        }
    )


def savefig(fig, name: str, formats: Tuple[str, ...] = ("png", "svg")) -> List[Path]:
    """Save ``fig`` to ``out/<name>.<fmt>`` for each format and return the paths."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = OUT_DIR / f"{name}.{fmt}"
        fig.savefig(p, format=fmt)
        paths.append(p)
    return paths


def style_field_ax(ax, title: Optional[str] = None) -> None:
    """Strip ticks / grid from an image axis and optionally set a title."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_title(title)


# ----------------------------------------------------------------------------- #
# Normalization helpers
# ----------------------------------------------------------------------------- #
def zscore(x: np.ndarray) -> np.ndarray:
    """Z-score a ``[2, ...]`` field with the dataset per-channel stats."""
    return (x - MEAN[:, *([None] * (x.ndim - 1))]) / STD[:, *([None] * (x.ndim - 1))]


def unzscore(x: np.ndarray) -> np.ndarray:
    """Invert :func:`zscore`."""
    return x * STD[:, *([None] * (x.ndim - 1))] + MEAN[:, *([None] * (x.ndim - 1))]


# ----------------------------------------------------------------------------- #
# Data access
# ----------------------------------------------------------------------------- #
@dataclass
class TrajectoryBundle:
    """A single Gray-Scott trajectory plus provenance metadata.

    ``fields`` is ``[2, T, H, W]`` in *physical* (un-normalized) units.
    """

    fields: np.ndarray            # [2, T, H, W] float32, physical space
    source: str                   # "hdf5" | "well" | "synthetic"
    label: str                    # human-readable regime / source label
    t_index: np.ndarray           # actual timestep indices selected


def find_hdf5_files(data_root: str, split: str) -> List[str]:
    """Return sorted HDF5 files under ``<data_root>/data/<split>`` (may be empty)."""
    if not data_root:
        return []
    return sorted(glob.glob(os.path.join(data_root, "data", split, "*.hdf5")))


_REGIME_RE = re.compile(r"_([a-zA-Z]+)_F_([0-9.]+)_k_([0-9.]+)\.hdf5$")


def parse_regime(filename: str) -> Tuple[str, Optional[float], Optional[float]]:
    """Parse ``..._<regime>_F_<F>_k_<k>.hdf5`` -> ``(regime, F, k)``.

    Falls back to ``(stem, None, None)`` when the name does not match.
    """
    m = _REGIME_RE.search(filename)
    if not m:
        return (Path(filename).stem, None, None)
    regime, F, k = m.group(1), float(m.group(2)), float(m.group(3))
    return (regime, F, k)


def _regime_label_from_path(path: str) -> str:
    """Best-effort short label for a regime file (e.g. ``maze (F=0.029, k=0.057)``)."""
    regime, F, k = parse_regime(os.path.basename(path))
    if F is None:
        return Path(path).stem
    return f"{regime} (F={F}, k={k})"


# ----------------------------------------------------------------------------- #
# Remote (HuggingFace) data download — per-file, by id
# ----------------------------------------------------------------------------- #
def list_remote_files(
    dataset: str = "gray_scott_reaction_diffusion",
    split: str = "valid",
    repo_id: Optional[str] = None,
) -> List[dict]:
    """List the HDF5 files of ``<dataset>/data/<split>`` in the HF repo, with ids.

    Returns dicts ``{id, filename, rfilename, regime, F, k, size_mb}`` sorted by
    filename, so ``id`` matches the on-disk order used by :func:`find_hdf5_files`
    (and the loader's ``sorted(glob(...))``). Requires ``huggingface_hub``.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for remote listing/download "
            "(pip install huggingface_hub)"
        ) from e

    repo_id = repo_id or WELL_REPO_FMT.format(dataset=dataset)
    info = HfApi().repo_info(repo_id, repo_type="dataset", files_metadata=True)
    prefix = f"data/{split}/"
    sibs = [s for s in info.siblings
            if s.rfilename.startswith(prefix) and s.rfilename.endswith(".hdf5")]
    sibs.sort(key=lambda s: s.rfilename)
    out = []
    for i, s in enumerate(sibs):
        fname = os.path.basename(s.rfilename)
        regime, F, k = parse_regime(fname)
        size_mb = (s.size or 0) / 1e6
        out.append({"id": i, "filename": fname, "rfilename": s.rfilename,
                    "regime": regime, "F": F, "k": k, "size_mb": size_mb})
    return out


def format_remote_listing(files: List[dict]) -> str:
    """Pretty-print a :func:`list_remote_files` result as an id table."""
    if not files:
        return "(no files found)"
    lines = [f"{'id':>3}  {'regime':<10} {'F':>7} {'k':>7} {'size':>9}  filename"]
    for f in files:
        F = "-" if f["F"] is None else f"{f['F']:.3f}"
        k = "-" if f["k"] is None else f"{f['k']:.3f}"
        lines.append(f"{f['id']:>3}  {f['regime']:<10} {F:>7} {k:>7} "
                     f"{f['size_mb']:>7.0f}MB  {f['filename']}")
    return "\n".join(lines)


def download_well_files(
    dataset: str = "gray_scott_reaction_diffusion",
    split: str = "valid",
    ids=(0,),
    base_dir=None,
    with_stats: bool = True,
    repo_id: Optional[str] = None,
) -> str:
    """Download selected ``<dataset>/data/<split>`` HDF5 files by id; return DATA_ROOT.

    Files land at ``<base_dir>/<dataset>/data/<split>/<file>.hdf5`` (the layout
    :func:`find_hdf5_files` expects), so the returned path is a ready-to-use
    ``DATA_ROOT``. Already-present files are skipped. Requires ``huggingface_hub``.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for remote listing/download "
            "(pip install huggingface_hub)"
        ) from e

    repo_id = repo_id or WELL_REPO_FMT.format(dataset=dataset)
    base_dir = Path(base_dir) if base_dir is not None else DSET_BASE
    data_root = base_dir / dataset
    data_root.mkdir(parents=True, exist_ok=True)

    available = list_remote_files(dataset, split, repo_id=repo_id)
    by_id = {f["id"]: f for f in available}
    if isinstance(ids, int):
        ids = [ids]
    selected = []
    for i in ids:
        if i not in by_id:
            raise ValueError(f"id {i} not in available ids {sorted(by_id)} "
                             f"for {dataset}/{split}")
        selected.append(by_id[i])

    total_mb = sum(f["size_mb"] for f in selected)
    print(f"[download] {repo_id} :: {split} :: ids={list(ids)} "
          f"-> {data_root}  (~{total_mb:.0f} MB)", flush=True)

    for f in selected:
        dest = data_root / f["rfilename"]
        if dest.exists():
            print(f"   skip (exists): {f['rfilename']}", flush=True)
            continue
        print(f"   fetching: {f['rfilename']} (~{f['size_mb']:.0f} MB)", flush=True)
        hf_hub_download(repo_id, filename=f["rfilename"], repo_type="dataset",
                        local_dir=str(data_root))
    if with_stats:
        try:
            hf_hub_download(repo_id, filename="data/stats.yaml",
                            repo_type="dataset", local_dir=str(data_root))
        except Exception as e:  # stats are optional for our loader
            print(f"   (stats.yaml not fetched: {e})", flush=True)
    return str(data_root)


def _load_hdf5_trajectory(
    path: str, traj_idx: int, t0: int, n_frames: int, time_stride: int
) -> np.ndarray:
    import h5py

    span = (n_frames - 1) * time_stride + 1
    sl = slice(t0, t0 + span, time_stride)
    with h5py.File(path, "r") as f:
        A = f["t0_fields/A"][traj_idx, sl]   # [T, H, W]
        B = f["t0_fields/B"][traj_idx, sl]
    return np.stack([A, B], axis=0).astype(np.float32)   # [2, T, H, W]


# --- synthetic fallback: a compact Gray-Scott reaction-diffusion simulator ----- #
def simulate_gray_scott(
    F: float = 0.04,
    k: float = 0.06,
    size: int = 128,
    n_frames: int = 16,
    sim_steps_per_frame: int = 60,
    burn_in: int = 600,
    Du: float = 0.16,
    Dv: float = 0.08,
    seed: int = 0,
) -> np.ndarray:
    """A tiny periodic Gray-Scott solver used only as an offline fallback.

    Returns ``[2, n_frames, size, size]`` (channels A, B) so the notebooks can
    render real reaction-diffusion patterns even when the dataset is not staged.
    Different ``(F, k)`` give visually distinct regimes (spots / worms / maze).
    """
    rng = np.random.default_rng(seed)
    A = np.ones((size, size), dtype=np.float32)
    B = np.zeros((size, size), dtype=np.float32)
    r = size // 10
    c = size // 2
    B[c - r : c + r, c - r : c + r] = 0.25
    A[c - r : c + r, c - r : c + r] = 0.5
    A += 0.02 * rng.standard_normal((size, size)).astype(np.float32)
    B += 0.02 * rng.standard_normal((size, size)).astype(np.float32)

    def laplacian(Z):
        return (
            -4.0 * Z
            + np.roll(Z, 1, 0)
            + np.roll(Z, -1, 0)
            + np.roll(Z, 1, 1)
            + np.roll(Z, -1, 1)
        )

    def step():
        nonlocal A, B
        rab = A * B * B
        A += Du * laplacian(A) - rab + F * (1.0 - A)
        B += Dv * laplacian(B) + rab - (F + k) * B
        np.clip(A, 0.0, 1.0, out=A)
        np.clip(B, 0.0, 1.0, out=B)

    for _ in range(burn_in):
        step()
    frames = []
    for _ in range(n_frames):
        for _ in range(sim_steps_per_frame):
            step()
        frames.append(np.stack([A.copy(), B.copy()], axis=0))
    return np.stack(frames, axis=1).astype(np.float32)   # [2, T, H, W]


# A few illustrative (F, k) regimes for the synthetic gallery.
SYNTHETIC_REGIMES = [
    ("spots", 0.030, 0.062),
    ("worms", 0.040, 0.060),
    ("maze", 0.029, 0.057),
    ("holes", 0.039, 0.058),
    ("waves", 0.014, 0.045),
    ("chaos", 0.026, 0.051),
]


def _stream_well_trajectory(n_frames: int, time_stride: int) -> Optional[np.ndarray]:
    """Best-effort single trajectory from the_well HF streaming; ``None`` on failure."""
    try:
        from the_well.data import WellDataset

        span = (n_frames - 1) * time_stride + 1
        ds = WellDataset(
            well_base_path="hf://datasets/polymathic-ai/",
            well_dataset_name="gray_scott_reaction_diffusion",
            well_split_name="valid",
            n_steps_input=span,
            n_steps_output=0,
        )
        sample = ds[0]
        # WellDataset emits [T, H, W, n_fields]; take input window, fields A,B.
        arr = np.asarray(sample["input_fields"])
        arr = arr[::time_stride][:n_frames]                 # [T, H, W, C]
        fields = np.transpose(arr, (3, 0, 1, 2)).astype(np.float32)  # [C, T, H, W]
        return fields[:2]
    except Exception:
        return None


def load_demo_trajectory(
    data_root: str = "",
    split: str = "train",
    n_frames: int = 16,
    time_stride: int = 4,
    traj_idx: int = 0,
    t0: int = 0,
    prefer: str = "auto",
) -> TrajectoryBundle:
    """Load one trajectory, trying local HDF5 -> the_well streaming -> synthetic.

    ``prefer`` may be ``"auto"``, ``"hdf5"``, ``"well"`` or ``"synthetic"`` to force
    a particular source.
    """
    files = find_hdf5_files(data_root, split)
    if prefer in ("auto", "hdf5") and files:
        fields = _load_hdf5_trajectory(files[0], traj_idx, t0, n_frames, time_stride)
        span = (n_frames - 1) * time_stride + 1
        return TrajectoryBundle(
            fields=fields,
            source="hdf5",
            label=_regime_label_from_path(files[0]),
            t_index=np.arange(t0, t0 + span, time_stride),
        )
    if prefer in ("auto", "well"):
        fields = _stream_well_trajectory(n_frames, time_stride)
        if fields is not None:
            return TrajectoryBundle(
                fields=fields,
                source="well",
                label="gray_scott_reaction_diffusion (HF stream)",
                t_index=np.arange(n_frames) * time_stride,
            )
    name, F, k = SYNTHETIC_REGIMES[1]
    fields = simulate_gray_scott(F=F, k=k, n_frames=n_frames)
    return TrajectoryBundle(
        fields=fields,
        source="synthetic",
        label=f"synthetic Gray-Scott (F={F}, k={k}, regime '{name}')",
        t_index=np.arange(n_frames),
    )


def list_regimes(
    data_root: str = "",
    split: str = "train",
    max_regimes: int = 6,
    frame: int = 500,
) -> List[Tuple[str, np.ndarray, str]]:
    """Return ``(label, field[2,H,W], source)`` for several regimes for the gallery.

    Uses one frame from each local HDF5 regime file when available, else falls
    back to distinct synthetic (F, k) regimes.
    """
    files = find_hdf5_files(data_root, split)
    out: List[Tuple[str, np.ndarray, str]] = []
    if files:
        import h5py

        for path in files[:max_regimes]:
            with h5py.File(path, "r") as f:
                fr = min(frame, f["t0_fields/A"].shape[1] - 1)
                A = f["t0_fields/A"][0, fr]
                B = f["t0_fields/B"][0, fr]
            out.append(
                (_regime_label_from_path(path), np.stack([A, B], 0).astype(np.float32), "hdf5")
            )
        return out
    for name, F, k in SYNTHETIC_REGIMES[:max_regimes]:
        fld = simulate_gray_scott(F=F, k=k, n_frames=1)[:, 0]   # [2, H, W]
        out.append((f"{name}\n(F={F}, k={k})", fld, "synthetic"))
    return out


# ----------------------------------------------------------------------------- #
# Small analysis utilities (used by both notebooks)
# ----------------------------------------------------------------------------- #
def make_demo_results(base_dir) -> Tuple[Path, Path, Path]:
    """Write a small set of *synthetic* result artifacts mirroring the real layout.

    Used only by ``02_results.ipynb`` when no real ``RUN_DIR`` / ``BASELINE_DIR`` is
    configured, so the notebook runs end-to-end and previews every figure. The files
    match exactly what ``eb_jepa/experiment_logger.py`` and the baseline evaluator
    write. Returns ``(run_dir, baseline_dir, jepa_rollout_csv)``.
    """
    import csv
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = Path(base_dir)
    run_dir = base / "run_demo"
    baseline_dir = base / "baselines_demo"
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # losses_step.csv
    n_epochs, steps_per_epoch = 20, 100
    with open(run_dir / "losses_step.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "total_loss", "vc_loss", "pred_loss", "lr", "wall_time"])
        for e in range(n_epochs):
            for s in range(steps_per_epoch):
                gstep = e * steps_per_epoch + s
                prog = gstep / (n_epochs * steps_per_epoch)
                pred = 1.2 * np.exp(-3 * prog) + 0.05 + 0.03 * rng.standard_normal()
                vc = 0.6 * np.exp(-2 * prog) + 0.1 + 0.02 * rng.standard_normal()
                w.writerow([gstep, e, f"{pred + vc:.4f}", f"{vc:.4f}", f"{pred:.4f}", "1e-3", gstep])

    # losses_epoch.csv
    epochs_rows = []
    with open(run_dir / "losses_epoch.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "epoch_time_s", "wall_time"])
        for e in range(n_epochs):
            prog = e / n_epochs
            tr = 1.6 * np.exp(-3 * prog) + 0.15
            va = tr + 0.05 + 0.02 * rng.standard_normal()
            w.writerow([e, f"{tr:.4f}", f"{va:.4f}", "42", e])
            epochs_rows.append({"epoch": e, "train_loss": tr, "val_loss": va})

    # predictions/latent_metrics.json (a few epochs, MSE grows with horizon)
    H = 10
    latent = {}
    for e in (0, 5, 10, 19):
        base_mse = 0.4 * np.exp(-0.12 * e)
        mse = [float(base_mse * (1 + 0.35 * h) + 0.01) for h in range(1, H + 1)]
        latent[str(e)] = {"epoch": e, "horizon": list(range(1, H + 1)), "latent_mse": mse}
    with open(run_dir / "predictions" / "latent_metrics.json", "w") as f:
        json.dump(latent, f, indent=2)

    # predictions/field_metrics.json (optional comparison)
    field = {}
    for e in (0, 19):
        base_mse = 0.25 * np.exp(-0.1 * e)
        mse = [float(base_mse * (1 + 0.3 * h) + 0.02) for h in range(1, H + 1)]
        field[str(e)] = {"epoch": e, "horizon": list(range(1, H + 1)), "field_mse": mse}
    with open(run_dir / "predictions" / "field_metrics.json", "w") as f:
        json.dump(field, f, indent=2)

    # predictions/latent_epoch19.png (target / pred / |error| panel)
    cols = 4
    fig, axes = plt.subplots(3, cols, figsize=(2.2 * cols, 6.6), squeeze=False)
    demo = simulate_gray_scott(n_frames=cols + 1)[0]  # field A frames
    for j in range(cols):
        tgt = demo[j]
        prd = tgt + 0.05 * rng.standard_normal(tgt.shape)
        err = np.abs(tgt - prd)
        for i, img in enumerate((tgt, prd, err)):
            ax = axes[i][j]
            ax.imshow(img, cmap="viridis")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"h={j + 1}", fontsize=9)
        axes[0][0].set_ylabel("target z", fontsize=9)
    for i, lab in enumerate(("target z", "pred z_hat", "|error|")):
        axes[i][0].set_ylabel(lab, fontsize=9)
    fig.suptitle("Latent prediction vs target — epoch 19 (demo)", fontsize=11)
    fig.tight_layout()
    fig.savefig(run_dir / "predictions" / "latent_epoch19.png", dpi=120)
    plt.close(fig)

    with open(run_dir / "experiment.json", "w") as f:
        json.dump({"run_dir": str(run_dir), "meta": {"git_sha": "demo"},
                   "epochs": epochs_rows, "predictions": {"latent": latent, "field": field}},
                  f, indent=2, default=str)

    # baselines: per_model_rollout_vrmse.csv + metrics.csv
    horizons = list(range(1, 31))
    curves = {
        "FNO": [0.08 * h ** 0.55 for h in horizons],
        "UNetClassic": [0.10 * h ** 0.6 for h in horizons],
        "persistence": [0.12 * h ** 0.75 for h in horizons],
        "mean": [1.0 for _ in horizons],
    }
    jepa = [0.07 * h ** 0.45 for h in horizons]   # the (synthetic) JEPA decoded rollout
    with open(baseline_dir / "per_model_rollout_vrmse.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon", "FNO", "UNetClassic", "persistence", "mean"])
        for i, h in enumerate(horizons):
            w.writerow([h] + [f"{curves[k][i]:.6f}" for k in ("FNO", "UNetClassic", "persistence", "mean")])

    def win(arr, a, b):
        return float(np.mean(arr[a:b]))

    with open(baseline_dir / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "one_step_vrmse", "rollout_6:12", "rollout_13:30"])
        for name in ("FNO", "UNetClassic"):
            arr = curves[name]
            w.writerow([name, f"{arr[0]:.5f}", f"{win(arr, 6, 12):.5f}", f"{win(arr, 13, 30):.5f}"])
        for name in ("persistence", "mean"):
            arr = curves[name]
            w.writerow([name, "", f"{win(arr, 6, 12):.5f}", f"{win(arr, 13, 30):.5f}"])

    # JEPA decoded-rollout csv (horizon,vrmse) — the overlay for the headline figure
    jepa_csv = baseline_dir / "jepa_rollout_vrmse.csv"
    with open(jepa_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon", "vrmse"])
        for h, v in zip(horizons, jepa):
            w.writerow([h, f"{v:.6f}"])

    return run_dir, baseline_dir, jepa_csv


def radial_power_spectrum(img: np.ndarray, n_bins: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Azimuthally-averaged power spectrum of a 2D field.

    Returns ``(k, power)`` where ``k`` is radial wavenumber (bin centers).
    """
    f = np.fft.fftshift(np.fft.fft2(img - img.mean()))
    power = np.abs(f) ** 2
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = min(cy, cx)
    bins = np.linspace(0, r_max, n_bins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    radial = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    pflat = power.ravel()
    for i in range(n_bins):
        m = idx == i
        if m.any():
            radial[i] = pflat[m].mean()
            counts[i] = m.sum()
    k = 0.5 * (bins[:-1] + bins[1:])
    valid = counts > 0
    return k[valid], radial[valid]


# ----------------------------------------------------------------------------- #
# CLI: list / download Well files by id without a notebook
# ----------------------------------------------------------------------------- #
def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="List or download The Well HDF5 files by id (figures helper).")
    ap.add_argument("--dataset", default="gray_scott_reaction_diffusion")
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--list", action="store_true", help="list available files + ids")
    ap.add_argument("--download", type=int, nargs="+", metavar="ID",
                    help="download these file ids")
    ap.add_argument("--base-dir", default=None, help=f"store under here (default {DSET_BASE})")
    args = ap.parse_args()

    if args.list or not args.download:
        print(format_remote_listing(list_remote_files(args.dataset, args.split)))
    if args.download:
        root = download_well_files(args.dataset, args.split, ids=args.download,
                                   base_dir=args.base_dir)
        print(f"DATA_ROOT={root}")


if __name__ == "__main__":
    _main()
