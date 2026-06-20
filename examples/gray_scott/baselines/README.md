# The Well surrogate-baseline evaluation

Measure the official The Well neural-operator surrogates (FNO, TFNO, U-Net = `UNetClassic`,
CNext-U-Net = `UNetConvNext`) on 2D Well datasets, under The Well's own protocol, so the
field-space VRMSE numbers are directly comparable to our later JEPA decoded rollouts.

This computes **measured checkpoint VRMSE under the official protocol** — it is *not* a
claim to exactly reproduce the paper tables. Small differences are expected (checkpoint /
library version / normalization drift; see `the_well` issue #49).

## Install (runtime dependency, not in pyproject)

```bash
uv pip install "the_well[benchmark]"
```

`the_well[benchmark]` pins `neuraloperator==0.3.0`; if it conflicts with the repo's pinned
torch, install it into a separate/uv environment. We deliberately do not edit
`pyproject.toml`.

## Protocol (fixed, identical for all datasets)

- `n_steps_input=4`, `n_steps_output=1`, `ZScoreNormalization`, stride 1 (`min=max=1`).
- VRMSE = official `the_well.benchmark.metrics.VRMSE`,
  `sqrt( <(pred-true)^2>_space / (<(true-<true>)^2>_space + 1e-7) )`, per field then
  averaged over fields, in **physical (denormalized)** space.
- **one-step**: rollout of length 1 on sliding ground-truth windows (test split) — paper Table 2.
- **rollout**: autoregressive from the start of each test trajectory, horizons `1..H` — paper Table 3.
- **windows**: `6:12` and `13:30` = `mean(per_step_vrmse[6:12])` / `[13:30]` (python slices,
  matching `Trainer.temporal_split_losses`; i.e. 1-indexed horizons 7..12 and 14..30).
- Reference baselines: **persistence** (repeat last input frame) and **mean** (spatial mean of
  last input frame).

## Run

Smoke (fast debug; streams a few trajectories from HF, `H=8`):

```bash
# P1 — smoke on the smallest dataset first
python -m examples.gray_scott.baselines.eval_baselines --dataset turbulent_radiative_layer_2D --smoke
```

Full evaluations:

```bash
# P2 — main deliverable
python -m examples.gray_scott.baselines.eval_baselines --dataset gray_scott_reaction_diffusion --H 30

# P3 — second dataset
python -m examples.gray_scott.baselines.eval_baselines --dataset turbulent_radiative_layer_2D --H 30

# P4 — optional, heavy (342 GB); smoke always, full only if staged/fast
python -m examples.gray_scott.baselines.eval_baselines --dataset rayleigh_benard --smoke
python -m examples.gray_scott.baselines.eval_baselines --dataset rayleigh_benard --H 30
```

Useful flags: `--H`, `--batch-size`, `--device cpu`, `--cfg <path>`.

## Where things are stored (env-driven, on /lustre/work — never HOME)

- Hugging Face cache: `HF_HOME` (set by `env.sh` to `$EBJEPA_WORK/.cache/huggingface`); checkpoints
  download there.
- Data: `THE_WELL_BASE_PATH` (or `EBJEPA_DSETS`) points at the folder containing
  `{dataset}/data/{split}`; if unset, streams from `hf://datasets/polymathic-ai/`.
- Outputs: `${EBJEPA_BASELINE_OUT:-$EBJEPA_WORK/outputs}/baselines/{dataset}/`.

## Outputs (per dataset)

- `metrics.json` — protocol, fields, per-model one-step + per-field + window VRMSE (official and
  aggregated-num/den cross-check).
- `metrics.csv` — model x {one-step, 6:12, 13:30}.
- `per_model_rollout_vrmse.csv` — per-horizon VRMSE (1..H) for every model + persistence + mean.
- `vrmse_vs_horizon.png` — VRMSE vs horizon (log y), one curve per model + persistence + VRMSE=1 line.
- `rollout_comparison.png` — ground truth vs each model at several horizons (one field).

## Hugging Face model IDs

`polymathic-ai/{FNO,TFNO,UNetClassic,UNetConvNext}-{dataset}` for
`gray_scott_reaction_diffusion`, `turbulent_radiative_layer_2D`, `rayleigh_benard`.

## How this plugs into the JEPA comparison

The JEPA will decode its latent autoregressive rollout to field space and be scored by the
**same** `well_rollout.py` + official VRMSE, on the same test split, same horizons and 6:12/13:30
windows — so the JEPA-vs-surrogate plot is apples-to-apples. The JEPA must use the same protocol:
stride 1, 4-frame context, denormalized VRMSE.

`per_model_rollout_vrmse_aggregated.csv` holds the variance-pooled VRMSE (the same definition the
JEPA `eval.py` uses), so `compare_jepa.py` overlays the best ablation JEPA against the baselines:

```bash
python -m examples.gray_scott.baselines.compare_jepa \
    --baselines-dir outputs/baselines/gray_scott_reaction_diffusion \
    --ablation-summary outputs/ablations/summary.json \
    --out-dir outputs/comparison/gray_scott --select mean
```
Writes `comparison.png` (the headline figure), `comparison.csv`, `comparison.json`. If the
aggregated CSV is absent (baselines not re-run yet), it falls back to the official per-sample CSV
and labels the plot accordingly.
