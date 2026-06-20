# Why the field-space surrogates diverge (and FNO/TFNO are the worst)

This note explains the headline result of the surrogate-baseline evaluation: under the
official The Well protocol (4-frame input history, stride 1, ZScore normalization,
physical-space VRMSE), the spectral neural operators **FNO** and **TFNO** diverge under
long autoregressive rollout, while the local conv nets **U-Net** and **CNextU-Net** stay
bounded. The same harness, normalization, and metric are used for every model, so the
difference is genuine model behavior, not an evaluation artifact (U-Net / CNextU-Net come
out textbook-correct and paper-consistent through that same code).

## What the artifacts show

Per dataset (`outputs/baselines/<dataset>/`):
- `vrmse_vs_horizon.png` - median VRMSE per horizon with a shaded 10-90th percentile band
  over test trajectories. FNO/TFNO climb as straight lines on the log axis (exponential
  growth) and are marked `(diverged)`; U-Net/CNextU-Net degrade gracefully.
- `rollout.gif` - ground truth vs each model across horizons. FNO collapses into diagonal
  spectral-aliasing stripes; TFNO saturates to a near-constant field; CNextU-Net tracks the
  truth the longest.
- `spectral_diagnostic.png` - radial power spectrum at a mid horizon. FNO/TFNO pile energy
  at the wrong (high) wavenumbers relative to the ground truth.
- `rollout_comparison.png` - static small-multiples of the same story.
- Cross-dataset: `outputs/baselines/stability_summary.png` ranks models by *stability
  horizon* (leading steps with VRMSE <= 1).

## Mechanisms

1. **Autoregressive feedback (the core driver).** The checkpoints are trained as one-step
   predictors on ground-truth inputs (teacher forcing). At rollout, step `t` consumes the
   model's own step `t-1` output, which carries error and is out-of-distribution versus
   training. Error compounds roughly as `e_t ~ g * e_{t-1}`; if the effective per-step gain
   `g > 1`, error grows exponentially and overflows float32 (-> `inf`/`nan`).

2. **Spectral truncation / aliasing (why FNO/TFNO specifically).** FNO and TFNO apply a
   learned multiplication on only the lowest K Fourier modes. Gray-Scott spots and turbulent
   fronts are sharp / high-frequency; truncating to low modes produces Gibbs ringing and
   aliasing - exactly the diagonal-stripe artifacts in the GIF and the high-wavenumber energy
   in `spectral_diagnostic.png`. A learned linear spectral operator with any mode gain above
   1 amplifies that mode geometrically each step, so there is no built-in bound and it blows
   up to infinity.

3. **Local conv nets saturate instead of exploding.** U-Net / CNextU-Net are local
   convolutional models with normalization layers; their errors grow but tend to saturate
   (large-but-finite VRMSE) rather than diverge. CNextU-Net is consistently the strongest.

4. **TFNO is more constrained than FNO.** Tucker-factorized spectral weights have less
   capacity, so on these stiff PDEs TFNO often diverges even faster (e.g. one-step VRMSE
   already >> 1 on gray_scott).

5. **Chaotic dynamics add intrinsic error growth.** `turbulent_radiative_layer_2D` and
   `rayleigh_benard` are turbulent/chaotic (positive Lyapunov exponents), so trajectories
   separate naturally; the model instability in (1)-(2) sits on top of this. `gray_scott` is
   smooth reaction-diffusion, which is why the local models stay bounded there.

## VRMSE reports this faithfully

VRMSE = sqrt( <(pred-true)^2>_space / (Var_space(true) + 1e-7) ), computed per field and
per snapshot in physical space (`the_well.benchmark.metrics.spatial.VRMSE`). When the
prediction diverges, the numerator explodes while the denominator is fixed by the ground
truth, so VRMSE follows. VRMSE = 1 means "no better than predicting the spatial mean", so
VRMSE > 1 is worse than the mean predictor.

## Takeaway for the JEPA comparison

The divergence is the result to *report*, not suppress: it is the long-horizon instability
of field-space neural-operator surrogates that a video-JEPA latent predictor is meant to
improve on. Headline framing:
- One-step VRMSE: all models look reasonable.
- Long-horizon rollout: FNO (and often TFNO) diverge; U-Net/CNextU-Net degrade but stay
  bounded - and these curves are what JEPA's denoised latent rollout is compared against,
  at a fraction of the parameters (see `n_params` in each `metrics.json`).

Note on `rayleigh_benard`: its tiny per-step change makes persistence a very strong baseline
and the learned models can sit above it; read the banded curve and the persistence reference
together rather than the windowed numbers alone.
