"""Gray-Scott — temporal-JEPA pretraining entrypoint (PDE reaction-diffusion).

Research question: can a JEPA learn the *dynamics* of a PDE by predicting the
*latent* of the future (not the pixels)? Each simulation is a 2D physical video
``[2, T, 128, 128]`` (chemical fields A, B). This is a PREDICTIVE / temporal
JEPA (video-style), NOT a two-view objective:

  context  z[:, :context_length]  --predictor-->  z_hat  (future latent)
  target   z_target = target_encoder(future frames)      (EMA, no grad)
  loss     = || z_hat - z_target ||  (latent prediction) + VC(z) (anti-collapse)

The DATA + TRAINING LOOP are provided. The two modelling pieces you implement are
marked ``# TODO`` below — that is the whole point of the track:
  1. the 2D encoder over a frame  ``[B, 2, H, W] -> [B, D, h, w]``
  2. the temporal-JEPA assembly (encoder + EMA target + predictor + VCLoss)

Run:  python -m examples.gray_scott.main --fname examples/gray_scott/cfgs/train.yaml
"""
import os
import sys
import time

import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.gray_scott.dataset import GrayScottConfig, make_loader
from eb_jepa import architectures, losses, jepa
from eb_jepa.training_utils import setup_wandb

# Reuse the eb_jepa core — DO NOT reimplement these:
#   eb_jepa.architectures: ResNet5 / ImpalaEncoder (2D encoders), ResUNet
#                          (latent->latent predictor backbone),
#                          StateOnlyPredictor (rolls latents forward), Projector
#   eb_jepa.losses:        VCLoss (variance+covariance anti-collapse), SquareLossSeq
#   eb_jepa.jepa:          JEPA (online+target encoder, predictor, .unroll(...))


# --------------------------------------------------------------------------- #
# 1) ENCODER  — # TODO
# --------------------------------------------------------------------------- #
def build_encoder(cfg):
    """TODO: return a 2D frame encoder mapping a frame ``[B, 2, H, W]`` (the two
    chemical fields) to a latent ``[B, D, h, w]``. It must also accept the 5D clip
    ``[B, 2, T, H, W]`` and return ``[B, D, T, h, w]`` (the eb_jepa 2D encoders do
    this via ``TemporalBatchMixin`` — they fold T into the batch automatically).

    Hints: ``eb_jepa.architectures.ResNet5(in_d=2, h_d=henc, out_d=dstc)`` is the
    drop-in choice — stride-1 / no avg-pool keeps the latent at full ``h=w=128``
    resolution (so a decoder can later map it back to a field). ``ImpalaEncoder``
    is the heavier alternative. Expose ``out_d`` (= D = dstc) for downstream use.

    Set ``cfg.encoder: fno`` to use the Fourier Neural Operator encoder
    (``FNOEncoder``) instead — a resolution-preserving, periodic-domain operator
    well suited to PDE fields. ``cfg.fno_modes`` / ``cfg.fno_layers`` tune it."""
    in_d = cfg.get("dobs", 2)
    if cfg.get("encoder", "resnet5") == "fno":
        return architectures.FNOEncoder(
            in_d=in_d,
            h_d=cfg.henc,
            out_d=cfg.dstc,
            modes=cfg.get("fno_modes", 16),
            n_layers=cfg.get("fno_layers", 4),
        )
    return architectures.ResNet5(in_d=in_d, h_d=cfg.henc, out_d=cfg.dstc)


# --------------------------------------------------------------------------- #
# 2) TEMPORAL-JEPA ASSEMBLY  — # TODO
# --------------------------------------------------------------------------- #
def build_jepa(encoder, cfg):
    """TODO: assemble and return an ``eb_jepa.jepa.JEPA`` (predictive/temporal,
    NOT two-view). The pieces, all reused from eb_jepa:
      * online + target encoder: pass ``encoder`` as both — JEPA keeps an EMA copy
        of the target internally (no-grad target of the future latent).
      * predictor that ROLLS LATENTS FORWARD: wrap a ``ResUNet(2*D, hpre, D)`` in
        ``StateOnlyPredictor(..., context_length=2)`` — it predicts the next latent
        from the previous two (state-only, no actions).
      * anti-collapse: ``VCLoss(std_coeff, cov_coeff, proj=Projector("D-4D-4D"))``.
      * prediction loss: ``SquareLossSeq(projector)`` on the projected latents.
    Build via ``JEPA(encoder, encoder, predictor, regularizer, predcost)``; the
    training loop below drives it with ``jepa.unroll(x, actions=None, ...)``.
    Keep the VC anti-collapse term — it is what stops the latent from collapsing.

    Takes the full ``cfg`` so it can read architecture dims from ``cfg.model`` and
    the VC anti-collapse coefficients from ``cfg.loss``."""
    mcfg, lcfg = cfg.model, cfg.loss

    predictor = architectures.StateOnlyPredictor(
        architectures.ResUNet(2 * mcfg.dstc, mcfg.hpre, mcfg.dstc),
        context_length=2)

    proj = architectures.Projector(f"{mcfg.dstc}-{mcfg.dstc * 4}-{mcfg.dstc * 4}")

    # Anti-collapse regularizer: vicreg (variance+covariance) or sigreg (Epps-Pulley
    # Gaussianity over random slices). Both share the same call signature and `.proj`.
    reg_kind = lcfg.get("regularizer", "vicreg")
    if reg_kind == "vicreg":
        anti_collapse = losses.VCLoss(lcfg.std_coeff, lcfg.cov_coeff, proj=proj)
    elif reg_kind == "sigreg":
        anti_collapse = losses.SIGReg(
            lcfg.sigreg_coeff, num_slices=int(lcfg.get("num_slices", 256)), proj=proj)
    else:
        raise ValueError(f"unknown loss.regularizer={reg_kind!r} (expected vicreg|sigreg)")

    pred_cost = losses.SquareLossSeq(anti_collapse.proj)

    return jepa.JEPA(encoder, encoder, predictor, anti_collapse, pred_cost)


# --------------------------------------------------------------------------- #
# TRAINING LOOP  — provided
# --------------------------------------------------------------------------- #
def run(fname="examples/gray_scott/cfgs/train.yaml", cfg=None, folder=None, **overrides):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    dcfg = GrayScottConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    train_loader = make_loader(dcfg)
    val_loader = make_loader(GrayScottConfig(**{**dcfg.__dict__, "split": "valid",
                                                "epoch_size": dcfg.batch_size * 10}), shuffle=False)
    print(f"[gs] {len(train_loader.dataset.files)} train hdf5 | "
          f"clip=[{dcfg.channels},{dcfg.n_frames},{dcfg.img_size},{dcfg.img_size}] "
          f"stride={dcfg.time_stride} | {len(train_loader)} steps/epoch", flush=True)
    if len(train_loader) == 0:
        raise ValueError(
            f"train_loader has 0 steps/epoch: data.epoch_size ({dcfg.epoch_size}) "
            f"< data.batch_size ({dcfg.batch_size}) with drop_last=True, so no "
            f"training step runs and the model never updates. Increase "
            f"data.epoch_size (>= batch_size) or decrease data.batch_size.")

    encoder = build_encoder(cfg.model).to(device)
    jepa = build_jepa(encoder, cfg).to(device)
    n_params = sum(p.numel() for p in jepa.parameters())
    print(f"[gs] params: {n_params / 1e6:.2f}M", flush=True)

    opt = torch.optim.Adam(jepa.parameters(), lr=cfg.optim.lr)
    use_amp = bool(cfg.training.use_amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.training.get("dtype", "bfloat16") == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and amp_dtype == torch.float16)

    ckpt_dir = folder or cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    gstep = 0

    # region agent log (debug session 5a0aa9)
    def _dbg(hyp, message, data):
        import json as _json
        import time as _time
        rec = {"sessionId": "5a0aa9", "runId": "post-fix", "hypothesisId": hyp,
               "location": "examples/gray_scott/main.py", "message": message,
               "data": data, "timestamp": int(_time.time() * 1000)}
        line = _json.dumps(rec, default=str)
        print("[DBG] " + line, flush=True)
        try:
            with open("/home/sardi/eb_jepa_bellwethers/.cursor/debug-5a0aa9.log", "a") as _f:
                _f.write(line + "\n")
        except Exception:
            pass

    _dbg("H0", "instrumented_main_active", {
        "encoder": str(cfg.model.get("encoder", "resnet5")),
        "encoder_class": type(jepa.encoder).__name__,
        "has_spectral_attr": hasattr(jepa.encoder, "spectral"),
        "lr": float(cfg.optim.lr),
        "epochs": int(cfg.optim.epochs),
    })
    # endregion

    # -- W&B logging
    use_wandb = cfg.logging.get("log_wandb", False)
    wandb_run = setup_wandb(
        project=cfg.logging.get("wandb_project", "eb_jepa"),
        config={
            "example": "gray_scott",
            "n_params": int(n_params),
            **OmegaConf.to_container(cfg, resolve=True),
        },
        run_dir=ckpt_dir,
        run_name=cfg.logging.get("wandb_run") or f"gs_seed{cfg.meta.seed}",
        tags=[f"seed_{cfg.meta.seed}", "gray_scott"],
        group=cfg.logging.get("wandb_group"),
        enabled=use_wandb,
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )
    if use_wandb:
        import wandb

    best_val_pred_loss = float('inf')
    epochs_without_improvement = 0
    patience = 15

    for epoch in range(cfg.optim.epochs):
        jepa.train()
        t0 = time.time()
        for batch in train_loader:
            x = batch["video"].to(device, non_blocking=True)        # [B,2,T,H,W]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                _, (jepa_loss, regl, _, _, pl) = jepa.unroll(
                    x, actions=None, nsteps=cfg.model.steps,
                    unroll_mode="parallel", compute_loss=True, return_all_steps=False)
            # backward (works for both AMP branches)
            if scaler.is_enabled():
                scaler.scale(jepa_loss).backward()
            else:
                jepa_loss.backward()

            # region agent log (debug session 5a0aa9)
            _do_dbg = gstep < 3
            if _do_dbg:
                spec_sq = rest_sq = 0.0
                spec_none = rest_none = 0
                for _n, _p in jepa.named_parameters():
                    if _p.grad is None:
                        if "spectral" in _n:
                            spec_none += 1
                        else:
                            rest_none += 1
                        continue
                    _gn = float((_p.grad.detach().abs() ** 2).sum().item())
                    if "spectral" in _n:
                        spec_sq += _gn
                    else:
                        rest_sq += _gn
                _enc = jepa.encoder
                _has_spec = hasattr(_enc, "spectral") and hasattr(_enc, "head")
                _w1_before = (
                    float(_enc.spectral[0].w1.detach().abs().sum().item())
                    if _has_spec else None)
                _hw_before = (
                    float(_enc.head[-1].weight.detach().abs().sum().item())
                    if _has_spec else None)
                with torch.no_grad():
                    _z = _enc(x)
                _dbg("H1H2H3H4H5", "pre_step", {
                    "gstep": gstep,
                    "enc_class": type(_enc).__name__,
                    "scaler_enabled": bool(scaler.is_enabled()),
                    "use_amp": bool(use_amp),
                    "amp_dtype": str(amp_dtype),
                    "loss": float(jepa_loss.item()),
                    "loss_requires_grad": bool(jepa_loss.requires_grad),
                    "loss_has_grad_fn": jepa_loss.grad_fn is not None,
                    "loss_isnan": bool(torch.isnan(jepa_loss).item()),
                    "loss_isinf": bool(torch.isinf(jepa_loss).item()),
                    "spec_grad_norm": spec_sq ** 0.5,
                    "rest_grad_norm": rest_sq ** 0.5,
                    "spec_grad_none": spec_none,
                    "rest_grad_none": rest_none,
                    "enc_out_std": float(_z.float().std().item()),
                    "enc_out_mean": float(_z.float().mean().item()),
                })
            # endregion

            # optimizer step (works for both AMP branches)
            if scaler.is_enabled():
                scaler.step(opt); scaler.update()
            else:
                opt.step()

            # region agent log (debug session 5a0aa9)
            if _do_dbg:
                _enc = jepa.encoder
                _w1_after = (
                    float(_enc.spectral[0].w1.detach().abs().sum().item())
                    if _w1_before is not None else None)
                _hw_after = (
                    float(_enc.head[-1].weight.detach().abs().sum().item())
                    if _hw_before is not None else None)
                _dbg("H1", "post_step", {
                    "gstep": gstep,
                    "w1_abs_sum_delta": (None if _w1_before is None else _w1_after - _w1_before),
                    "head_w_abs_sum_delta": (None if _hw_before is None else _hw_after - _hw_before),
                })
            # endregion
            gstep += 1
            if gstep % cfg.logging.log_every == 0:
                print(f"e{epoch} s{gstep} loss={jepa_loss.item():.4f} "
                      f"vc={regl.item():.4f} pred={pl.item():.4f}", flush=True)
                if use_wandb:
                    wandb.log({
                        "train/total_loss": jepa_loss.item(),
                        "train/vc_loss": regl.item(),
                        "train/pred_loss": pl.item(),
                        "train/lr": opt.param_groups[0]["lr"],
                        "epoch": epoch,
                    }, step=gstep)

        # val
        jepa.eval(); vl = 0.0; vp = 0.0; nb = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["video"].to(device)
                with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                    _, (jl, _, _, _, pl) = jepa.unroll(x, actions=None, nsteps=cfg.model.steps,
                                                      unroll_mode="parallel", compute_loss=True)
                vl += jl.item(); vp += pl.item(); nb += 1
        val_loss = vl / max(nb, 1)
        val_pred_loss = vp / max(nb, 1)
        epoch_time = time.time() - t0
        print(f"[epoch {epoch}] {epoch_time:.0f}s | val_loss={val_loss:.4f} val_pred={val_pred_loss:.4f}", flush=True)
        if use_wandb:
            wandb.log({
                "val/total_loss": val_loss,
                "val/pred_loss": val_pred_loss,
                "epoch": epoch,
                "epoch_time": epoch_time,
            }, step=gstep)
        
        # Save latest checkpoint
        torch.save({"epoch": epoch,
                    "encoder": encoder.state_dict(),
                    "jepa": jepa.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, "latest.pth.tar"))
                   
        # Save periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({"epoch": epoch,
                        "encoder": encoder.state_dict(),
                        "jepa": jepa.state_dict(),
                        "cfg": OmegaConf.to_container(cfg, resolve=True)},
                       os.path.join(ckpt_dir, f"ckpt_epoch_{epoch + 1}.pth.tar"))
                       
        # Early stopping logic
        if val_pred_loss < best_val_pred_loss:
            best_val_pred_loss = val_pred_loss
            epochs_without_improvement = 0
            # Save best checkpoint
            checkpoint_data = {
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "jepa": jepa.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True)
            }
            torch.save(checkpoint_data, os.path.join(ckpt_dir, "best.pth.tar"))
            torch.save(checkpoint_data, os.path.join(ckpt_dir, f"best_epoch_{epoch}.pth.tar"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[gs] Early stopping triggered! No improvement in pred-loss for {patience} epochs.", flush=True)
                if use_wandb:
                    wandb.log({"early_stopped": True, "early_stop_epoch": epoch}, step=gstep)
                break

    if use_wandb:
        wandb.log({"best_val_pred_loss": best_val_pred_loss}, step=gstep)
        wandb.finish()
    print(f"[gs] done -> {ckpt_dir}/latest.pth.tar and best.pth.tar", flush=True)
    return best_val_pred_loss


if __name__ == "__main__":
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/gray_scott/cfgs/train.yaml"
    run(fname=fname)
