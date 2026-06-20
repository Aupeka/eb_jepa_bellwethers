import os
import argparse
from omegaconf import OmegaConf

try:
    import optuna
except ImportError:
    print("Optuna is not installed. Please install it using: uv run pip install optuna")
    print("Or if using a standard environment: pip install optuna")
    exit(1)

from examples.gray_scott.main import run

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

def objective(trial, base_cfg_path, short_run_epochs):
    # Suggest hyperparameters
    # Most importantly: VICReg parameters (VCLoss variance and covariance terms)
    std_coeff = trial.suggest_float("loss.std_coeff", 1.0, 100.0, log=True)
    cov_coeff = trial.suggest_float("loss.cov_coeff", 1.0, 500.0, log=True)
    
    # Other important parameters present in cfgs/train.yaml
    lr = trial.suggest_float("optim.lr", 1e-4, 5e-3, log=True)
    dstc = trial.suggest_categorical("model.dstc", [8, 16, 32])
    henc = trial.suggest_categorical("model.henc", [16, 32, 64])
    hpre = trial.suggest_categorical("model.hpre", [16, 32, 64])

    # Load base config to modify
    cfg = OmegaConf.load(base_cfg_path)
    
    # Override for short runs to detect optimal values quickly
    cfg.optim.epochs = short_run_epochs
    
    # Set the suggested hyperparameters
    cfg.loss.std_coeff = std_coeff
    cfg.loss.cov_coeff = cov_coeff
    cfg.optim.lr = lr
    cfg.model.dstc = dstc
    cfg.model.henc = henc
    cfg.model.hpre = hpre
    
    # Create a unique checkpoint directory for this trial to avoid collisions
    base_ckpt_dir = cfg.meta.get("ckpt_dir", "checkpoints/gray_scott/dev")
    trial_folder = os.path.join(base_ckpt_dir, f"tune_trial_{trial.number}")
    cfg.meta.ckpt_dir = trial_folder
    
    print(f"\n" + "="*40)
    print(f"--- Starting Trial {trial.number} ---")
    print(f"Hyperparameters: std_coeff={std_coeff:.2f}, cov_coeff={cov_coeff:.2f}, lr={lr:.1e}, dstc={dstc}, henc={henc}, hpre={hpre}")
    print("="*40)
    
    # Run the training pipeline and capture output
    with CaptureStdoutAndPrint() as captured:
        best_val_loss = run(cfg=cfg, folder=trial_folder)
    
    # If run() successfully returned a float (because main.py was updated), use it
    if best_val_loss is not None:
        return best_val_loss
        
    # If run() returned None, fallback to parsing the standard output
    out_str = captured.getvalue()
    losses = re.findall(r"val_loss=([0-9.]+)", out_str)
    if losses:
        fallback_loss = min(float(l) for l in losses)
        print(f"[fallback] Parsed best validation loss from stdout: {fallback_loss}")
        return fallback_loss
        
    raise ValueError("run() returned None and no validation losses were printed.")

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
