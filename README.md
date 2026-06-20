# EB-JEPA for Gray-Scott Reaction-Diffusion

**Question:** Can a Joint-Embedding Predictive Architecture (JEPA) learn the *dynamics* of a Partial Differential Equation (PDE) by predicting the *latent representation* of the future (instead of predicting raw pixels), and how does latent-space prediction compare, in The Well's field-space VRMSE, to neural-operator surrogates (FNO / U-Net)?

This repository adapts the **Energy-Based Joint-Embedding Predictive Architectures (EB-JEPA)** to model the Gray-Scott reaction-diffusion system from Polymathic AI's *The Well* dataset.

## The Model: Temporal JEPA vs Neural Operator Surrogates

```text
context  z[:, :context_length=2]  --predictor(ResUNet)-->  z_hat (future latent)
target   z_target = target_encoder(future frames)        (EMA, no grad)
loss     = || z_hat - z_target ||  (SquareLossSeq) + VCLoss(std, cov)  (anti-collapse)
```
There is **no pixel loss in pretraining** — the model predicts a *representation* of the future. A latent-to-field decoder is added only at evaluation time to score VRMSE.

## Baseline Outputs (The Well Surrogates)
To provide a comparison point, we measure the official The Well neural-operator surrogates (FNO, TFNO, U-Net, CNext-U-Net) under the same evaluation protocol that the JEPA will use.

| Model | Parameters (M) | One-Step VRMSE | Rollout (6:12) VRMSE | Rollout (13:30) VRMSE | Stability Horizon |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FNO | 19.009 | 38.03166 | 266127.34 | inf | 0 |
| TFNO | 19.274 | 1212.57666 | ~1.44e16 | nan | 0 |
| U-Net | 17.464 | 0.04070 | 0.62610 | 15.47198 | 11 |
| CNextU-Net | 18.572 | 0.03208 | 0.31217 | 2.14047 | 18 |
| persistence | N/A | - | 2.58721 | 25.41451 | 3 |
| mean | N/A | - | 1.69018 | 19.58668 | 0 |

*(Note: Evaluated on `gray_scott_reaction_diffusion` under The Well's official protocol with stride 1, 4-frame context, and denormalized VRMSE.)*

## Codebase Layout
The primary code specific to the Gray-Scott task is located in:
- `eb_jepa/datasets/gray_scott/` - The HDF5 data loader and dataset configurations.
- `examples/gray_scott/` - The main entrypoints for training, evaluation, and baseline comparisons.

## How to Run Experiments

### 1. Pretraining (Temporal-JEPA)
Use the `main.py` entrypoint for temporal-JEPA pretraining. This implements the `JEPA` with a shared encoder and EMA target, a `StateOnlyPredictor(ResUNet)` that rolls latents forward, and `VCLoss` for anti-collapse representation learning.

```bash
python -m examples.gray_scott.main --fname examples/gray_scott/cfgs/train.yaml
```

### 2. Evaluation (Field-space VRMSE)
Use the `eval.py` entrypoint for multi-step VRMSE rollout. This builds a frozen-JEPA latent-to-field decoder and evaluates multi-step VRMSE (variance-scaled RMSE) for JEPA vs persistence across different horizons `1..H`.

```bash
python -m examples.gray_scott.eval --ckpt <.../latest.pth.tar> --H 10
```

### 3. Hyperparameter Ablation
You can sweep across different configurations (like varying the number of rollout steps `K` and the regularizer type) using the ablation script. This automates the tuning and saves summary metrics.

```bash
# Smoke test (tiny):
python -m examples.gray_scott.ablation \
    --combos resnet5_vicreg_K2 \
    --n_trials 2 --short_run_epochs 1 --final_epochs 1 --H 6
```
