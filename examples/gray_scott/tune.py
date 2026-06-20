import os
import argparse
from omegaconf import OmegaConf

try:
    import optuna
except ImportError:
    print("Optuna is not installed. Please install it using: uv run pip install optuna")
    print("Or if using a standard environment: pip install optuna")
    exit(1)

from examples.gray_scott.main import run, build_encoder, build_jepa

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

import io
import sys
import re

class CaptureStdoutAndPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._stringio = io.StringIO()
        sys.stdout = Tee(self._original_stdout, self._stringio)
        return self._stringio
    def __exit__(self, *args):
        sys.stdout = self._original_stdout

def _count_params(cfg):
    """Build encoder+JEPA on CPU and count parameters (for the budget cap)."""
    import torch  # local import keeps the module importable without torch present
    enc = build_encoder(cfg.model)
    model = build_jepa(enc, cfg)
    return sum(p.numel() for p in model.parameters())


def objective(trial, base_cfg_path, short_run_epochs, combo=None, param_cap_m=None,
              out_root=None, wandb_trials=False, wandb_project=None):
    """One Optuna trial. ``combo`` (optional) pins the ablation axes
    ``{"K": int, "regularizer": "vicreg"|"sigreg", "encoder": "resnet5"}``; the
    regularizer-specific coefficients are swept accordingly. Trials whose model exceeds
    ``param_cap_m`` million parameters are pruned before any training."""
    cfg = OmegaConf.load(base_cfg_path)
    cfg.optim.epochs = short_run_epochs

    # --- ablation axes (pinned by the combo, not searched) ---
    regularizer = (combo or {}).get("regularizer", cfg.loss.get("regularizer", "vicreg"))
    cfg.loss.regularizer = regularizer
    if combo and "K" in combo:
        cfg.model.steps = int(combo["K"])

    # --- shared search space ---
    cfg.optim.lr = trial.suggest_float("optim.lr", 1e-4, 5e-3, log=True)
    cfg.model.dstc = trial.suggest_categorical("model.dstc", [8, 16, 32])
    cfg.model.henc = trial.suggest_categorical("model.henc", [16, 32, 64])
    cfg.model.hpre = trial.suggest_categorical("model.hpre", [16, 32, 64])

    # --- regularizer-specific search space ---
    if regularizer == "vicreg":
        cfg.loss.std_coeff = trial.suggest_float("loss.std_coeff", 1.0, 100.0, log=True)
        cfg.loss.cov_coeff = trial.suggest_float("loss.cov_coeff", 1.0, 500.0, log=True)
        hp = f"std={cfg.loss.std_coeff:.2f} cov={cfg.loss.cov_coeff:.2f}"
    elif regularizer == "sigreg":
        cfg.loss.sigreg_coeff = trial.suggest_float("loss.sigreg_coeff", 1.0, 100.0, log=True)
        cfg.loss.num_slices = trial.suggest_categorical("loss.num_slices", [64, 128, 256])
        hp = f"sigreg={cfg.loss.sigreg_coeff:.2f} slices={cfg.loss.num_slices}"
    else:
        raise ValueError(f"unknown regularizer {regularizer!r}")

    # --- parameter budget cap: prune oversized models before training ---
    if param_cap_m is not None:
        n_params = _count_params(cfg)
        if n_params > param_cap_m * 1e6:
            print(f"[trial {trial.number}] pruned: {n_params/1e6:.2f}M > {param_cap_m}M cap", flush=True)
            raise optuna.TrialPruned()
        trial.set_user_attr("n_params", int(n_params))

    base_ckpt_dir = out_root or cfg.meta.get("ckpt_dir", "checkpoints/gray_scott/dev")
    trial_folder = os.path.join(base_ckpt_dir, f"tune_trial_{trial.number}")
    cfg.meta.ckpt_dir = trial_folder

    # Trials log to W&B only when explicitly requested (otherwise 20xN runs are noisy).
    if wandb_trials:
        cfg.logging.wandb = True
        if wandb_project:
            cfg.logging.wandb_project = wandb_project
        cfg.logging.wandb_group = (combo or {}).get("name", "tune") + "-tune"
        cfg.logging.wandb_run = f"{cfg.logging.wandb_group}-t{trial.number}"
    else:
        cfg.logging.wandb = False

    print("\n" + "=" * 40)
    print(f"--- Trial {trial.number} [{(combo or {}).get('name', regularizer)}] ---")
    print(f"Hyperparameters: lr={cfg.optim.lr:.1e}, dstc={cfg.model.dstc}, "
          f"henc={cfg.model.henc}, hpre={cfg.model.hpre}, {hp}")
    print("=" * 40)

    with CaptureStdoutAndPrint() as captured:
        best_val_loss = run(cfg=cfg, folder=trial_folder)

    if best_val_loss is not None:
        return best_val_loss

    out_str = captured.getvalue()
    losses = re.findall(r"val_pred=([0-9.]+)", out_str) or re.findall(r"val_loss=([0-9.]+)", out_str)
    if losses:
        fallback_loss = min(float(l) for l in losses)
        print(f"[fallback] Parsed best validation loss from stdout: {fallback_loss}")
        return fallback_loss

    raise ValueError("run() returned None and no validation losses were printed.")


def run_study(base_cfg_path, combo, n_trials, short_run_epochs, study_name,
              param_cap_m=None, out_root=None, wandb_trials=False, wandb_project=None):
    """Run one Optuna study for a single ablation combo. Returns
    ``(best_params: dict, best_value: float)``; if every trial was pruned (e.g. cap too
    low), returns ``({}, inf)`` instead of raising."""
    study = optuna.create_study(direction="minimize", study_name=study_name)
    study.optimize(
        lambda trial: objective(trial, base_cfg_path, short_run_epochs, combo=combo,
                                param_cap_m=param_cap_m, out_root=out_root,
                                wandb_trials=wandb_trials, wandb_project=wandb_project),
        n_trials=n_trials)
    try:
        return dict(study.best_trial.params), float(study.best_trial.value)
    except ValueError:
        print(f"[run_study] {study_name}: all trials pruned/failed", flush=True)
        return {}, float("inf")

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Gray-Scott temporal-JEPA")
    parser.add_argument("--fname", type=str, default="examples/gray_scott/cfgs/train.yaml", help="Path to base config file")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of tuning trials to run")
    parser.add_argument("--short_run_epochs", type=int, default=3, help="Number of epochs for each short tuning run")
    parser.add_argument("--study_name", type=str, default="gray_scott_tuning", help="Name of the optuna study")
    
    args = parser.parse_args()
    
    # Create Optuna study optimizing for minimum validation loss
    study = optuna.create_study(direction="minimize", study_name=args.study_name)
    
    # Run optimization
    study.optimize(lambda trial: objective(trial, args.fname, args.short_run_epochs), n_trials=args.n_trials)
    
    print("\n" + "="*50)
    print("Tuning Completed!")
    print("="*50)
    print(f"Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    print(f"\nBest trial: #{best_trial.number}")
    print(f"  Best Validation Loss: {best_trial.value}")
    print("  Best Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
