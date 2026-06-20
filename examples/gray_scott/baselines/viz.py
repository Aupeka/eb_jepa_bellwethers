"""Presentation-grade plots for the surrogate-baseline rollout evaluation.

All functions are fed from data already computed by ``eval_baselines.py``:
  * ``rollout`` dict: per model {"official","median","p10","p90","per_traj",...},
  * ``viz`` cache: {"truth": [T,*spatial,C], "models": {name: [T,*spatial,C]}}.

Figures are styled for slides (large fonts, log-y where relevant). Diverged models
(FNO/TFNO going inf/nan at long horizons) are handled gracefully: non-finite points are
masked out of line/band plots and clipped for image/spectral plots.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Consistent, colorblind-friendly palette (matplotlib tab10) keyed by the model names
# we use across all datasets; unknown names fall back to the cycle.
_KNOWN_COLORS = {
    "FNO": "#1f77b4",
    "TFNO": "#ff7f0e",
    "U-Net": "#2ca02c",
    "CNextU-Net": "#d62728",
    "Poseidon": "#9467bd",
    "JEPA": "#8c564b",
}
_FALLBACK_CYCLE = ["#17becf", "#bcbd22", "#7f7f7f", "#e377c2"]


def model_color(name: str, idx: int) -> str:
    return _KNOWN_COLORS.get(name, _FALLBACK_CYCLE[idx % len(_FALLBACK_CYCLE)])


def stability_horizon(v, H: int) -> int:
    """Number of leading rollout steps a model stays at/below VRMSE 1 (>=mean-predictor).

    A clean scalar for ranking long-horizon stability: 0 means worse than the mean
    predictor already at step 1; H means stable through the full rollout."""
    v = np.asarray(v, dtype=float)
    k = 0
    for val in v:
        if np.isfinite(val) and val <= 1.0:
            k += 1
        else:
            break
    return int(k)


def _viz_field(cfg, meta) -> int:
    return int(cfg.get("viz_field", meta.n_fields - 1))


def _readable_ymax(cfg, rollout) -> float:
    """y-axis ceiling from finite reference curves only (persistence, mean, and the
    medians of NON-diverged models), with headroom. Never derived from diverged spikes,
    so the towering FNO/TFNO blow-ups don't crush the readable curves. Persistence/mean
    are always finite, so this stays well-defined even if every model diverges."""
    refs = []
    for ref in ("persistence", "mean"):
        if ref in rollout:
            refs.append(np.asarray(rollout[ref]["median"], dtype=float))
    for e in cfg.models:
        if np.isfinite(np.asarray(rollout[e.name]["official"], dtype=float)).all():
            refs.append(np.asarray(rollout[e.name]["median"], dtype=float))
    ref_max = 1.0
    for v in refs:
        fin = v[np.isfinite(v)]
        if fin.size:
            ref_max = max(ref_max, float(fin.max()))
    return ref_max * 4.0


def plot_vrmse_with_bands(out_dir, cfg, rollout, horizons):
    """VRMSE-vs-horizon, slide-styled.

    Solid line + shaded band = per-sample 'official' VRMSE (median, 10-90th percentile
    over test trajectories). Thin dashed line = variance-pooled 'aggregated' VRMSE, which
    is robust to low-spatial-variance snapshots (e.g. RB velocity at convection onset).

    Diverged models (FNO/TFNO blowing up) are truncated at the first off-scale horizon and
    marked with an X. This keeps the y-axis readable and avoids drawing the misleading
    survivorship 'recovery' tail (the per-trajectory median descends once enough
    trajectories overflow to inf->nan, which only reflects a shrinking surviving pool)."""
    from matplotlib.lines import Line2D

    h = np.asarray(horizons)
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ymax = _readable_ymax(cfg, rollout)

    for idx, e in enumerate(cfg.models):
        r = rollout[e.name]
        color = model_color(e.name, idx)
        median = np.asarray(r["median"], dtype=float)
        p10 = np.asarray(r["p10"], dtype=float)
        p90 = np.asarray(r["p90"], dtype=float)
        official = np.asarray(r["official"], dtype=float)
        agg = np.asarray(r["aggregated"], dtype=float)

        # Leading on-scale region: finite median at or below the readable ceiling.
        on = np.isfinite(median) & (median <= ymax)
        h_cut = len(on) if on.all() else int(np.argmin(on))  # first off-scale horizon index
        diverged = (not np.isfinite(official).all()) or (h_cut < len(on))
        sl = slice(0, h_cut)
        label = e.name + (" (diverged)" if diverged else "")
        ax.plot(h[sl], median[sl], color=color, marker="o", ms=3, lw=2, label=label)
        band_ok = np.isfinite(p10[sl]) & np.isfinite(p90[sl])
        ax.fill_between(h[sl][band_ok], p10[sl][band_ok], p90[sl][band_ok],
                        color=color, alpha=0.18, linewidth=0)
        # variance-pooled aggregated curve (robust), thin dashed, clipped to scale
        agg_on = np.isfinite(agg) & (agg <= ymax)
        ax.plot(h[agg_on], agg[agg_on], color=color, lw=1.2, ls="--", alpha=0.9)
        if diverged and h_cut < len(h):  # mark where the model leaves the readable scale
            ax.plot([h[h_cut]], [ymax], color=color, marker="X", ms=11, mec="k", mew=0.6, clip_on=False)

    pers = np.asarray(rollout["persistence"]["median"], dtype=float)
    mp = np.isfinite(pers) & (pers <= ymax)
    ax.plot(h[mp], pers[mp], "k--", lw=1.6, label="persistence")

    ax.set_yscale("log")
    ax.set_ylim(top=ymax)
    ax.axhline(1.0, color="gray", ls=":", lw=1.2)
    ax.axhspan(1.0, ymax, color="gray", alpha=0.06)
    ax.text(h[-1], 1.0, " worse than mean predictor", color="gray", va="bottom", ha="right", fontsize=10)

    ax.set_xlabel("rollout horizon (steps)")
    ax.set_ylabel("VRMSE (field-averaged, physical space)")
    ax.set_title(f"{cfg.dataset}: autoregressive rollout VRMSE\n"
                 f"(solid+band: per-sample median + 10-90th pct; dashed: variance-pooled)",
                 fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="gray", lw=1.2, ls="--"))
    labels.append("aggregated (variance-pooled)")
    ax.legend(handles, labels, fontsize=10, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "vrmse_vs_horizon.png"), dpi=150)
    plt.close(fig)
    plt.rcParams.update({"font.size": 10})


def plot_per_field_vrmse(out_dir, cfg, meta, rollout, horizons):
    """Per-field aggregated VRMSE-vs-horizon (small multiples, one subplot per field).

    Uses the variance-pooled 'aggregated' metric (robust to low-variance snapshots), so it
    cleanly separates which physical fields a model handles well. For Rayleigh-Benard this
    exposes the story behind the inflated headline VRMSE: buoyancy/pressure are predicted
    well while the velocity fields (near-zero spatial variance at convection onset) dominate
    the error."""
    fields = field_names_safe(meta)
    nC = len(fields)
    h = np.asarray(horizons)
    ncols = min(nC, 2)
    nrows = int(np.ceil(nC / ncols))
    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.6 * nrows), squeeze=False)
    axes = axes.ravel()

    pers = rollout.get("persistence", {}).get("aggregated_per_field")
    for c in range(nC):
        ax = axes[c]
        for idx, e in enumerate(cfg.models):
            agg = np.asarray(rollout[e.name]["aggregated_per_field"], dtype=float)[:, c]
            m = np.isfinite(agg)
            ax.plot(h[m], agg[m], color=model_color(e.name, idx), marker="o", ms=2.5, lw=1.8,
                    label=e.name)
        if pers is not None:
            pv = np.asarray(pers, dtype=float)[:, c]
            mp = np.isfinite(pv)
            ax.plot(h[mp], pv[mp], "k--", lw=1.4, label="persistence")
        ax.axhline(1.0, color="gray", ls=":", lw=1.0)
        ax.set_yscale("log")
        ax.set_title(fields[c], fontsize=12)
        ax.set_xlabel("horizon")
        if c % ncols == 0:
            ax.set_ylabel("aggregated VRMSE")
        ax.grid(alpha=0.3, which="both")
    for j in range(nC, len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=9, ncol=2)
    fig.suptitle(f"{cfg.dataset}: per-field aggregated rollout VRMSE", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "vrmse_per_field.png"), dpi=150)
    plt.close(fig)
    plt.rcParams.update({"font.size": 10})


def _clip_for_display(arr2d, vmin, vmax):
    """Replace inf/nan and clip so diverged frames don't break imshow / wash out colors."""
    a = np.nan_to_num(arr2d, nan=vmin, posinf=vmax, neginf=vmin)
    return np.clip(a, vmin, vmax)


def make_rollout_gif(out_dir, cfg, meta, viz, H, fps: int = 5):
    """Animated rollout.gif: ground truth vs each model across horizons 1..H (one field)."""
    if meta.n_spatial_dims != 2:
        return  # GIF only makes sense for 2D fields
    fld = _viz_field(cfg, meta)
    rows = [("ground truth", viz["truth"])] + [(e.name, viz["models"][e.name]) for e in cfg.models]
    T = min(H, viz["truth"].shape[0])
    vmin, vmax = np.percentile(viz["truth"][..., fld], [1, 99])
    if not np.isfinite([vmin, vmax]).all() or vmin == vmax:
        vmin, vmax = float(np.nanmin(viz["truth"][..., fld])), float(np.nanmax(viz["truth"][..., fld]) + 1e-6)

    fig, axes = plt.subplots(1, len(rows), figsize=(2.5 * len(rows), 3.0), squeeze=False)
    axes = axes[0]
    ims = []
    for ax, (label, arr) in zip(axes, rows):
        im = ax.imshow(_clip_for_display(arr[0, ..., fld], vmin, vmax),
                       vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ims.append(im)
    suptitle = fig.suptitle(f"{cfg.dataset} — field '{field_names_safe(meta)[fld]}' — h=1", fontsize=12)
    fig.tight_layout()

    def update(t):
        for im, (_, arr) in zip(ims, rows):
            im.set_data(_clip_for_display(arr[t, ..., fld], vmin, vmax))
        suptitle.set_text(f"{cfg.dataset} — field '{field_names_safe(meta)[fld]}' — h={t + 1}")
        return ims

    anim = FuncAnimation(fig, update, frames=T, interval=1000 // max(fps, 1), blit=False)
    anim.save(os.path.join(out_dir, "rollout.gif"), writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


def field_names_safe(meta):
    from the_well.data.utils import flatten_field_names

    return flatten_field_names(meta, include_constants=False)


def _radial_psd(field2d):
    """Radially-averaged power spectral density of a 2D field -> (k_bins, psd)."""
    f = np.nan_to_num(field2d.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    f = f - f.mean()
    fft = np.fft.fftshift(np.fft.fft2(f))
    power = np.abs(fft) ** 2
    ny, nx = f.shape
    ky = np.fft.fftshift(np.fft.fftfreq(ny)) * ny
    kx = np.fft.fftshift(np.fft.fftfreq(nx)) * nx
    kxx, kyy = np.meshgrid(kx, ky)
    kr = np.sqrt(kxx ** 2 + kyy ** 2)
    kbin = kr.astype(int)
    nbins = min(ny, nx) // 2
    psd = np.zeros(nbins)
    for k in range(nbins):
        sel = kbin == k
        if sel.any():
            psd[k] = power[sel].mean()
    return np.arange(nbins), psd


def plot_spectral_diagnostic(out_dir, cfg, meta, viz):
    """Power spectrum of predicted vs ground-truth field at a mid horizon.

    Shows FNO/TFNO injecting energy at high wavenumbers (spectral aliasing) — the
    quantitative 'why' behind their divergence."""
    if meta.n_spatial_dims != 2:
        return
    fld = _viz_field(cfg, meta)
    T = viz["truth"].shape[0]
    h = min(10, T)  # mid horizon (1-indexed h -> index h-1)
    idx = h - 1

    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    k, psd_true = _radial_psd(viz["truth"][idx, ..., fld])
    ax.loglog(k[1:], psd_true[1:], "k-", lw=2.5, label="ground truth")
    for i, e in enumerate(cfg.models):
        arr = viz["models"][e.name][idx, ..., fld]
        finite = np.isfinite(arr).all()
        kk, psd = _radial_psd(arr)
        label = e.name + ("" if finite else " (diverged)")
        ax.loglog(kk[1:], psd[1:], color=model_color(e.name, i), lw=1.8,
                  ls="-" if finite else "--", label=label)
    ax.set_xlabel("wavenumber k (radial)")
    ax.set_ylabel("power spectral density")
    ax.set_title(f"{cfg.dataset}: spatial power spectrum at h={h}\n(field '{field_names_safe(meta)[fld]}')")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spectral_diagnostic.png"), dpi=150)
    plt.close(fig)
    plt.rcParams.update({"font.size": 10})
