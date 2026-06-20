"""Gray-Scott — downstream evaluation (The Well's open question, in field space).

The Well asks: does latent prediction give more *stable* long-horizon rollouts
than the field-space neural-operator surrogates (FNO / U-Net)? To answer it we
roll the frozen JEPA predictor forward in LATENT space, DECODE each latent back
to a 2-channel field, and score multi-step VRMSE against ground truth and a
PERSISTENCE baseline (optionally vs FNO / U-Net surrogates).

The rollout-extraction harness is provided. What you implement (``# TODO``) is the
latent->field DECODER and the VRMSE metric that makes the comparison meaningful.

Run:  python -m examples.gray_scott.eval --ckpt <.../latest.pth.tar> --H 10
"""
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.gray_scott.dataset import GrayScottConfig, make_loader
from examples.gray_scott.main import build_encoder, build_jepa

C = 2            # context_length (StateOnlyPredictor predicts from the previous 2 frames)


# region agent log (debug session 870c4c)
def _dbg(hyp, message, data):
    import json as _json
    import time as _time
    rec = {"sessionId": "870c4c", "runId": "initial", "hypothesisId": hyp,
           "location": "examples/gray_scott/eval.py", "message": message,
           "data": data, "timestamp": int(_time.time() * 1000)}
    line = _json.dumps(rec, default=str)
    print("[DBG] " + line, flush=True)
    try:
        with open("/home/sardi/eb_jepa_bellwethers/.cursor/debug-870c4c.log", "a") as _f:
            _f.write(line + "\n")
    except Exception:
        pass
# endregion


def load_jepa(ckpt, device):
    """Provided: rebuild encoder + JEPA from a training checkpoint and freeze."""
    cfg = OmegaConf.create(ckpt["cfg"])
    encoder = build_encoder(cfg.model).to(device)
    jepa = build_jepa(encoder, cfg).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    jepa.load_state_dict(ckpt["jepa"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    return jepa, encoder


@torch.no_grad()
def rollout_latents(jepa, x, H, device):
    """Provided: autoregressive latent rollout from C context frames.

    Feeds the first C frames of the clip and rolls the predictor forward H steps
    in latent space (``ctxt_window_time=C`` — the StateOnlyPredictor needs 2
    context frames, else the autoregressive loop yields an empty time axis).
    Returns the predicted latent sequence ``[B, D, C+H, h, w]``."""
    pred, _ = jepa.unroll(x[:, :, :C], actions=None, nsteps=H,
                          unroll_mode="autoregressive", ctxt_window_time=C,
                          compute_loss=False, return_all_steps=False)
    return pred


# --------------------------------------------------------------------------- #
# LATENT -> FIELD DECODER  — # TODO
# --------------------------------------------------------------------------- #
def build_decoder(encoder, dstc, device, ckpt_path, cfg):
    """Return a trained latent->field decoder mapping ``[B, D, T, H, W]`` ->
    ``[B, 2, T, H, W]``. Trains it on-the-fly if weights are not found."""
    class Decoder(torch.nn.Module):
        def __init__(self, latent_dim, hidden_dim=64):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(latent_dim, hidden_dim, kernel_size=3, padding=1)
            self.gelu1 = torch.nn.GELU()
            self.conv2 = torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            self.gelu2 = torch.nn.GELU()
            self.conv3 = torch.nn.Conv2d(hidden_dim, 2, kernel_size=1)

        def forward(self, x):
            B, D, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, D, H, W)
            x = self.conv1(x)
            x = self.gelu1(x)
            x = self.conv2(x)
            x = self.gelu2(x)
            x = self.conv3(x)
            x = x.reshape(B, T, 2, H, W).permute(0, 2, 1, 3, 4)
            return x

    decoder = Decoder(dstc).to(device)
    decoder_path = os.path.join(os.path.dirname(ckpt_path), "decoder.pth")

    # region agent log (debug session 870c4c)
    _exists = os.path.exists(decoder_path)
    _init_pnorm = float(sum(p.detach().abs().sum().item() for p in decoder.parameters()))
    _dbg("H-A", "build_decoder_entry", {
        "decoder_path": decoder_path,
        "exists": bool(_exists),
        "branch": "load" if _exists else "train",
        "fresh_init_param_abs_sum": _init_pnorm,
    })
    # endregion

    if os.path.exists(decoder_path):
        print(f"[decoder] Loading weights from {decoder_path}", flush=True)
        decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
        # region agent log (debug session 870c4c)
        _loaded_pnorm = float(sum(p.detach().abs().sum().item() for p in decoder.parameters()))
        _dbg("H-B", "decoder_loaded", {
            "loaded_param_abs_sum": _loaded_pnorm,
            "changed_from_fresh_init": abs(_loaded_pnorm - _init_pnorm) > 1e-6,
        })
        # endregion
    else:
        print(f"[decoder] Weights not found. Training decoder from scratch...", flush=True)
        dcfg_dict = OmegaConf.to_container(cfg.data, resolve=True)
        dcfg_dict["split"] = "train"
        dcfg_dict["n_frames"] = 4
        dcfg_dict["epoch_size"] = 1000
        train_loader = make_loader(GrayScottConfig(**dcfg_dict), shuffle=True)

        opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
        decoder.train()
        encoder.eval()

        for epoch in range(100):
            try:
                total_loss = 0
                for batch in train_loader:
                    x = batch["video"].to(device, non_blocking=True)
                    with torch.no_grad():
                        z = encoder(x)
                    pred = decoder(z)
                    loss = torch.nn.functional.mse_loss(pred, x)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                print(f"[decoder] Epoch {epoch} loss: {total_loss / len(train_loader):.4f}", flush=True)
            except Exception as e:
                print(f"Epoch {epoch} out of range")
                continue


        torch.save(decoder.state_dict(), decoder_path)
        print(f"[decoder] Saved weights to {decoder_path}", flush=True)

    decoder.eval()
    return decoder

    

# --------------------------------------------------------------------------- #
# METRIC  — # TODO
# --------------------------------------------------------------------------- #
def vrmse_per_horizon(jepa, encoder, decoder, loader, device, H):
    """Per-horizon field-space VRMSE for JEPA vs a persistence baseline and decoder floor."""
    num_jepa = np.zeros(H)
    num_pers = np.zeros(H)
    num_flor = np.zeros(H)
    den_total = np.zeros(H)

    _first_batch = True
    for batch in loader:
        x = batch["video"].to(device)  # [B, 2, C+H, height, width]
        B = x.shape[0]

        true_future = x[:, :, C : C + H]  # [B, 2, H, height, width]

        # 1. JEPA Rollout
        latents_pred = rollout_latents(jepa, x, H, device)  # [B, D, C+H, h, w]
        latents_future = latents_pred[:, :, C : C + H]
        field_pred = decoder(latents_future)  # [B, 2, H, height, width]

        # 2. Persistence Baseline
        last_context = x[:, :, C - 1 : C]  # [B, 2, 1, height, width]
        field_pers = last_context.repeat(1, 1, H, 1, 1)

        # 3. Decoder Floor
        with torch.no_grad():
            latents_true = encoder(true_future)
            field_floor = decoder(latents_true)

        # region agent log (debug session 870c4c)
        if _first_batch:
            _first_batch = False
            with torch.no_grad():
                _dbg("H-C", "encoder_latent_stats", {
                    "latents_true_std": float(latents_true.float().std().item()),
                    "latents_true_mean": float(latents_true.float().mean().item()),
                    "latents_pred_std": float(latents_pred.float().std().item()),
                    "latents_true_shape": list(latents_true.shape),
                })
                _dbg("H-D", "field_scale_stats", {
                    "true_future_std": float(true_future.float().std().item()),
                    "true_future_mean": float(true_future.float().mean().item()),
                    "field_floor_std": float(field_floor.float().std().item()),
                    "field_floor_mean": float(field_floor.float().mean().item()),
                    "field_pred_std": float(field_pred.float().std().item()),
                })
                _dbg("H-B", "floor_recon_quality", {
                    "floor_mse_vs_true": float(((field_floor - true_future) ** 2).mean().item()),
                    "mean_baseline_mse": float(
                        ((true_future - true_future.mean(dim=(-2, -1), keepdim=True)) ** 2).mean().item()),
                })
        # endregion

        # Accumulate metrics per horizon step
        for h in range(H):
            t = true_future[:, :, h]  # [B, 2, height, width]
            t_mean = t.mean(dim=(-2, -1), keepdim=True)

            # Sum over spatial and channel dims, then sum over batch
            den = ((t - t_mean) ** 2).sum().item()
            
            n_j = ((field_pred[:, :, h] - t) ** 2).sum().item()
            n_p = ((field_pers[:, :, h] - t) ** 2).sum().item()
            n_f = ((field_floor[:, :, h] - t) ** 2).sum().item()

            den_total[h] += den
            num_jepa[h] += n_j
            num_pers[h] += n_p
            num_flor[h] += n_f

    # Add epsilon to prevent division by zero
    den_total = np.maximum(den_total, 1e-8)
    
    return {
        "jepa": np.sqrt(num_jepa / den_total),
        "persistence": np.sqrt(num_pers / den_total),
        "decoder_floor": np.sqrt(num_flor / den_total),
    }


def evaluate_checkpoint(ckpt_path, H, device=None, epoch_size=400, batch_size=8):
    """Load a JEPA checkpoint, (re)build its decoder, and score per-horizon VRMSE.

    Returns ``{"scores": {jepa/persistence/decoder_floor: np.ndarray[H]},
    "param_count": int, "epoch": int}``. Reused by ``main()`` (CLI) and by the
    ablation driver so every final model is scored under the identical H=30 protocol."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    jepa, encoder = load_jepa(ckpt, device)
    cfg = OmegaConf.create(ckpt["cfg"])
    dstc = int(cfg.model.dstc)
    decoder = build_decoder(encoder, dstc, device, ckpt_path, cfg)
    print(f"[gs-eval] loaded {ckpt_path} (epoch {ckpt.get('epoch')}), H={H}", flush=True)

    dcfg = GrayScottConfig(split="valid", n_frames=C + H, time_stride=4,
                           epoch_size=epoch_size, batch_size=batch_size, num_workers=8)
    loader = make_loader(dcfg, shuffle=False)
    scores = vrmse_per_horizon(jepa, encoder, decoder, loader, device, H)
    param_count = int(sum(p.numel() for p in jepa.parameters()))
    return {"scores": scores, "param_count": param_count, "epoch": ckpt.get("epoch")}


def main():
    ckpt_path = sys.argv[sys.argv.index("--ckpt") + 1]
    H = int(sys.argv[sys.argv.index("--H") + 1]) if "--H" in sys.argv else 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = evaluate_checkpoint(ckpt_path, H, device)
    print(f"[gs-eval] params={result['param_count'] / 1e6:.2f}M", flush=True)
    for name, arr in result["scores"].items():
        print(f"   {name:14s} h1={arr[0]:.3f} h{H}={arr[-1]:.3f} | {np.round(arr, 3).tolist()}", flush=True)


if __name__ == "__main__":
    main()
