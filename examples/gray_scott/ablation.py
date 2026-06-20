"""Gray-Scott temporal-JEPA ablation matrix.

For each combo on the matrix below we (1) run an independent Optuna study to pick
hyperparameters under a parameter-budget cap, (2) train ONE final model on the best
config, (3) score it with the H=30 field-space VRMSE protocol from ``eval.py``, and
(4) aggregate everything into ``<out_root>/summary.json``.

Matrix axes:
  * K (prediction horizon ``model.steps``) in {1, 2, 4, 8}
  * regularizer (anti-collapse) in {vicreg, sigreg}
  * encoder fixed to ResNet5 (stride-1, full-res latent so the decoder can map back)
  * predictor fixed to StateOnlyPredictor(ResUNet)

=> 8 combos named ``resnet5_<regularizer>_K<k>``.

Everything is additive: it reuses ``main.run`` (training), ``tune.run_study`` (Optuna),
and ``eval.evaluate_checkpoint`` (scoring) unchanged.

Smoke:  python -m examples.gray_scott.ablation --n_trials 2 --short_run_epochs 1 \
            --final_epochs 1 --H 6 --combos resnet5_vicreg_K2
Full:   python -m examples.gray_scott.ablation --n_trials 20 --short_run_epochs 3 \
            --final_epochs 20 --H 30 --param_cap_m 10 --wandb
"""
import argparse
import json
import os

import numpy as np
from omegaconf import OmegaConf

from examples.gray_scott.main import run
from examples.gray_scott.tune import run_study
from examples.gray_scott.eval import evaluate_checkpoint

KS = [1, 2, 4, 8]
REGULARIZERS = ["vicreg", "sigreg"]
ENCODER = "resnet5"


def all_combos():
    return [{"name": f"{ENCODER}_{reg}_K{k}", "K": k, "regularizer": reg, "encoder": ENCODER}
            for reg in REGULARIZERS for k in KS]


def _hidx(arr, h):
    """1-based horizon h -> array value, clamped to the available range."""
    arr = np.asarray(arr, dtype=float)
    return float(arr[min(h, len(arr)) - 1])


def _build_final_cfg(base_cfg_path, combo, best_params, final_epochs, ckpt_dir,
                     use_wandb, wandb_project, batch_size=None):
    cfg = OmegaConf.load(base_cfg_path)
    cfg.optim.epochs = final_epochs
    cfg.model.steps = int(combo["K"])
    cfg.loss.regularizer = combo["regularizer"]
    if batch_size is not None:
        cfg.data.batch_size = batch_size
    if best_params:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(
            [f"{k}={v}" for k, v in best_params.items()]))
    cfg.meta.ckpt_dir = ckpt_dir
    cfg.logging.wandb = bool(use_wandb)
    if use_wandb:
        cfg.logging.wandb_project = wandb_project
        cfg.logging.wandb_group = combo["name"]
        cfg.logging.wandb_run = combo["name"]
    return cfg


def run_combo(combo, args):
    out = os.path.join(args.out_root, combo["name"])
    ckpt_dir = os.path.join(out, "checkpoint")
    os.makedirs(out, exist_ok=True)
    print("\n" + "#" * 70)
    print(f"# COMBO {combo['name']}  (K={combo['K']}, reg={combo['regularizer']})")
    print("#" * 70, flush=True)

    # 1) Optuna study (short runs) -> best config under the param cap
    best_params, best_value = run_study(
        args.fname, combo, n_trials=args.n_trials, short_run_epochs=args.short_run_epochs,
        study_name=combo["name"], param_cap_m=args.param_cap_m,
        out_root=os.path.join(out, "tune"),
        wandb_trials=args.wandb_trials, wandb_project=args.wandb_project,
        batch_size=args.batch_size)
    print(f"[{combo['name']}] best optuna val_pred={best_value:.4f} params={best_params}", flush=True)

    # 2) Final training on the best config
    final_cfg = _build_final_cfg(args.fname, combo, best_params, args.final_epochs,
                                 ckpt_dir, args.wandb, args.wandb_project, args.batch_size)
    OmegaConf.save(final_cfg, os.path.join(out, "best_config.yaml"))
    final_val = run(cfg=final_cfg, folder=ckpt_dir)

    # 3) H=30 field-space VRMSE on the best checkpoint
    best_ckpt = os.path.join(ckpt_dir, "best.pth.tar")
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(ckpt_dir, "latest.pth.tar")
    result = evaluate_checkpoint(best_ckpt, args.H)
    scores = result["scores"]
    horizons = list(range(1, args.H + 1))

    metrics = {
        "combo": combo["name"],
        "K": combo["K"],
        "regularizer": combo["regularizer"],
        "encoder": combo["encoder"],
        "param_count": result["param_count"],
        "param_count_m": round(result["param_count"] / 1e6, 4),
        "best_optuna_val_pred_loss": best_value,
        "final_val_pred_loss": final_val,
        "best_params": best_params,
        "horizons": horizons,
        "vrmse": {
            "jepa": {"h1": _hidx(scores["jepa"], 1), "h10": _hidx(scores["jepa"], 10),
                     "h30": _hidx(scores["jepa"], 30)},
            "persistence": {"h1": _hidx(scores["persistence"], 1),
                            "h10": _hidx(scores["persistence"], 10),
                            "h30": _hidx(scores["persistence"], 30)},
            "decoder_floor": {"h1": _hidx(scores["decoder_floor"], 1),
                              "h10": _hidx(scores["decoder_floor"], 10),
                              "h30": _hidx(scores["decoder_floor"], 30)},
        },
        "curves": {k: np.asarray(v, dtype=float).tolist() for k, v in scores.items()},
    }
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{combo['name']}] jepa VRMSE h1={metrics['vrmse']['jepa']['h1']:.3f} "
          f"h10={metrics['vrmse']['jepa']['h10']:.3f} h30={metrics['vrmse']['jepa']['h30']:.3f}",
          flush=True)
    return metrics


def _log_summary_wandb(all_metrics, project):
    if os.environ.get("WANDB_DISABLED", "false").lower() == "true":
        return
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; skipping summary table", flush=True)
        return
    run_ = wandb.init(project=project, name="ablation-summary", reinit=True,
                      dir=os.environ.get("WANDB_DIR"))
    cols = ["combo", "K", "regularizer", "params_M", "best_optuna_val",
            "vrmse_h1", "vrmse_h10", "vrmse_h30", "decoder_floor_h30", "persistence_h30"]
    table = wandb.Table(columns=cols)
    for m in all_metrics:
        v = m["vrmse"]
        table.add_data(m["combo"], m["K"], m["regularizer"], m["param_count_m"],
                       m["best_optuna_val_pred_loss"], v["jepa"]["h1"], v["jepa"]["h10"],
                       v["jepa"]["h30"], v["decoder_floor"]["h30"], v["persistence"]["h30"])
    log = {"ablation/summary": table}
    if all_metrics:
        xs = all_metrics[0]["horizons"]
        ys = [m["curves"]["jepa"] for m in all_metrics]
        keys = [m["combo"] for m in all_metrics]
        log["ablation/jepa_vrmse_vs_horizon"] = wandb.plot.line_series(
            xs=xs, ys=ys, keys=keys, title="JEPA VRMSE vs horizon", xname="horizon")
    run_.log(log)
    run_.finish()


def main():
    p = argparse.ArgumentParser(description="Gray-Scott temporal-JEPA ablation matrix")
    p.add_argument("--fname", default="examples/gray_scott/cfgs/train.yaml")
    p.add_argument("--out_root", default="outputs/ablations")
    p.add_argument("--n_trials", type=int, default=20, help="Optuna trials per combo")
    p.add_argument("--short_run_epochs", type=int, default=3, help="epochs per Optuna trial")
    p.add_argument("--final_epochs", type=int, default=20, help="epochs for the final model")
    p.add_argument("--H", type=int, default=30, help="evaluation rollout horizon")
    p.add_argument("--param_cap_m", type=float, default=10.0, help="param budget cap (millions)")
    p.add_argument("--batch_size", type=int, default=None,
                   help="override data.batch_size (lower to fit GPU memory; default uses train.yaml)")
    p.add_argument("--combos", nargs="*", default=None,
                   help="subset of combo names to run (default: all 8)")
    p.add_argument("--wandb", action="store_true", help="log final training runs + summary to W&B")
    p.add_argument("--wandb_trials", action="store_true", help="also log every Optuna trial to W&B")
    p.add_argument("--wandb_project", default="gray-scott-jepa-ablation")
    args = p.parse_args()

    combos = all_combos()
    if args.combos:
        wanted = set(args.combos)
        combos = [c for c in combos if c["name"] in wanted]
        if not combos:
            raise SystemExit(f"no combos matched {args.combos}; available: "
                             f"{[c['name'] for c in all_combos()]}")

    os.makedirs(args.out_root, exist_ok=True)
    all_metrics = []
    for combo in combos:
        all_metrics.append(run_combo(combo, args))

    summary = {
        "param_cap_m": args.param_cap_m,
        "H": args.H,
        "final_epochs": args.final_epochs,
        "n_trials": args.n_trials,
        "combos": {m["combo"]: m for m in all_metrics},
    }
    summary_path = os.path.join(args.out_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[ablation] wrote {summary_path} ({len(all_metrics)} combos)", flush=True)

    if args.wandb or args.wandb_trials:
        _log_summary_wandb(all_metrics, args.wandb_project)


if __name__ == "__main__":
    main()
