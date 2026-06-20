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
  over test trajectories (solid+band = per-sample "official"; thin dashed = variance-pooled
  "aggregated"). FNO/TFNO climb as straight lines on the log axis (exponential growth), are
  marked `(diverged)`, and are truncated with an `X` at the horizon where they leave the
  readable scale (the y-axis is capped from the finite reference curves so the bounded models
  stay legible); U-Net/CNextU-Net degrade gracefully.
- `vrmse_per_field.png` - per-field aggregated VRMSE per horizon (one panel per physical
  field). This is where `rayleigh_benard` becomes interpretable: buoyancy/pressure are
  predicted well while the velocity fields dominate the headline error (see RB note below).
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

## Reading `rayleigh_benard`: a VRMSE normalization pathology, not a second bug

RB's headline (per-sample "official") VRMSE looks alarming and even *decreases* with horizon
for FNO/TFNO. This is a property of the metric on this dataset, not model recovery:

- VRMSE divides by the **per-snapshot spatial variance of the ground-truth field**. In
  Rayleigh-Benard the **velocity fields are near-uniform at the start of each trajectory**
  (convection has not developed yet), so `Var_space(velocity) -> ~0` and the per-sample VRMSE
  explodes for those early snapshots even when the absolute error is small.
- The per-field breakdown in `metrics.json` / `vrmse_per_field.png` confirms this: one-step
  VRMSE is tiny for buoyancy/pressure but large for velocity (e.g. FNO velocity_x 11.5,
  velocity_y 31.5; even CNextU-Net velocity ~0.45 vs buoyancy ~0.06).
- The FNO/TFNO rollout curves *decrease* (FNO h1=75 -> h30=11) because as convection develops
  the velocity variance (the denominator) grows, not because the error shrinks. A genuinely
  diverging curve increases.

The **aggregated** VRMSE (`sqrt(sum MSE / sum var)`, pooled over snapshots so low-variance
frames cannot blow up) tells the trustworthy story. At the 13:30 window, official vs
aggregated:

| model      | official | aggregated |
|------------|----------|------------|
| FNO        | 24.0     | 2.5        |
| TFNO       | 35.0     | 3.3        |
| U-Net      | 83.0     | 6.4        |
| CNextU-Net | 16.8     | 1.7        |

Guidance: for RB read the **aggregated** curve and the **per-field** plot, not the per-sample
windowed numbers. Also note RB's tiny per-step change makes persistence a very strong baseline
(VRMSE ~0.01-0.5), so the learned models sit above it; compare against the persistence
reference. ("official" remains the headline for gray_scott / TRL2D, whose fields stay
well-excited and do not trigger this pathology.)
