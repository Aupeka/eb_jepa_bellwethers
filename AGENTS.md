# Repo wiki — eb_jepa_bellwethers

A map of this repository for fast orientation. EB-JEPA is a lightweight library for
**Energy-Based Joint-Embedding Predictive Architectures** (JEPAs): models that predict
the *latent* representation of future/masked inputs instead of raw pixels. The library
core lives in `eb_jepa/`; each application lives in `examples/`.

- Python `3.12` (see `.python-version`, `pyproject.toml`).
- Package management with `uv` (or conda + uv). Configs are OmegaConf YAML.
- Default configs are tuned for H100; reduce batch size on smaller GPUs.

## Core library — `eb_jepa/`

| File | What it holds |
|------|----------------|
| `architectures.py` | Encoders/predictors: `ResNet5`, `ImpalaEncoder` (2D encoders), `ResUNet` (latent→latent predictor backbone), `StateOnlyPredictor` (rolls latents forward), `Projector`, `DetHead`, RNN/IDM heads. 2D encoders fold time via `TemporalBatchMixin`. |
| `jepa.py` | `JEPAbase`/`JEPA`/`JEPAProbe`. Built as `JEPA(encoder, aencoder, predictor, regularizer, predcost)`. Driven by `.unroll(x, actions, nsteps, unroll_mode={"parallel","autoregressive"}, compute_loss, return_all_steps)`. Keeps an EMA target encoder internally. |
| `losses.py` | `VCLoss` (variance+covariance anti-collapse), `SquareLossSeq` (prediction loss on projected latents), `HingeStdLoss`, `CovarianceLoss`, IDM/sim regularizers. |
| `image_decoder.py`, `state_decoder.py` | Eval-only decoders / probe heads (`ImageDecoder`, `MLPXYHead`, `GoalValueHead`). |
| `planning.py`, `hierarchical.py` | MPC/planning utilities and hierarchical JEPA. |
| `schedulers.py`, `nn_utils.py` | LR schedules; `TemporalBatchMixin`, weight init. |
| `training_utils.py` | Config loading, device/seed/wandb setup, checkpoint I/O, experiment-dir naming, logging helpers. |
| `logging.py`, `vis_utils.py` | Logger factory and visualization helpers. |
| `datasets/` | Per-domain loaders: `gray_scott/`, `moving_mnist.py`, `two_rooms/`, `maze/`, `audio/`, `eeg/`, `ltsf/`, `fintime/`, `pointcloud/`, plus `traj_dset.py`, `precomputed.py`, `utils.py`. |

## Applications — `examples/`

Each example has `main.py` (train), usually `eval.py`, and `cfgs/*.yaml`. Run with
`python -m examples.<name>.main --fname examples/<name>/cfgs/<cfg>.yaml`.

`image_jepa`, `video_jepa`, `ac_video_jepa` (world modeling + planning), `gray_scott`,
`audio`, `eeg`, `ltsf`, `fintime`, `pointcloud`, `intuitive_physics`,
`factors_of_variation`. `launch_sbatch.py` is the SLURM sweep launcher.

### Current task home — `examples/gray_scott/`

The Well (Ohana et al. 2024) reaction-diffusion track: temporal/predictive JEPA vs
field-space neural-operator surrogates (FNO / U-Net), scored by field-space VRMSE over
multi-step rollouts.

- Data loader: `eb_jepa/datasets/gray_scott/dataset.py` — emits clips `[B, 2, T, 128, 128]` (chemical fields A, B), z-scored. `GrayScottConfig` + `make_loader`.
- `main.py` — temporal-JEPA pretraining (encoder + EMA target + `StateOnlyPredictor(ResUNet)` + `VCLoss` + `SquareLossSeq`). `build_encoder`/`build_jepa` are implemented.
- `eval.py` — autoregressive **latent** rollout + decode to field + VRMSE. Open `# TODO`s: `build_decoder` (latent→field decoder) and `vrmse_per_horizon` (VRMSE vs persistence / FNO / U-Net).
- No pixel loss in pretraining — the decoder is added only at eval time to score VRMSE.

## Supporting dirs

| Path | Purpose |
|------|---------|
| `tests/` | pytest suite (`*_test.py` / `test_*.py`). New library code must add tests. |
| `references/paper/` | JEPA literature; many entries have a `SUMMARY.md`. |
| `docs/` | Architecture figures, code of conduct, contributing guide. |
| `cluster/`, `hackathon_guide/` | HTW/Dalia cluster docs and hackathon materials. |
| `setup.md`, `setup.sh`, `env.sh` | Cluster-only setup (skip for local work). |
| `pyproject.toml`, `Makefile` | Deps + tooling (black, isort, autoflake, pytest). |

## Key conventions (see `.cursor/rules/`)

- Tensors are `[B, C, T, H, W]` with the feature/channel dim at index 1.
- Reuse core components (`ResNet5`, `ResUNet`, `StateOnlyPredictor`, `Projector`, `VCLoss`, `SquareLossSeq`, `JEPA`) — do not reimplement them.
- JEPA pretraining predicts latents (no pixel loss); decoders are eval-only.
- Add and run tests for new code; format before finishing.

## Common commands

```bash
uv sync                                   # install deps
source .venv/bin/activate                 # or: uv run <cmd>
uv run pytest tests/                       # run tests (conda+uv: pytest tests/)
python -m examples.gray_scott.main --fname examples/gray_scott/cfgs/train.yaml
python -m examples.gray_scott.eval --ckpt <.../latest.pth.tar> --H 10

# format (run before finishing)
autoflake --remove-all-unused-imports -r --in-place .
python -m isort eb_jepa examples tests
python -m black eb_jepa examples tests
```

Env vars: `EBJEPA_DSETS` (datasets root), `EBJEPA_CKPTS` (checkpoints/logs).
