"""Per-run experiment logging for eb_jepa training scripts.

Research/engineering question this answers: *after the fact, can I fully explain
why a run behaved the way it did?* Each run gets its own timestamped folder with
a JSON of all run metadata + metrics, per-step / per-epoch loss CSVs, and
prediction-vs-actual artifacts. The JEPA predicts *latents*, so the always-on
comparison is latent (``z_hat`` vs the encoder target ``z``); a field-space
comparison is logged only when a latent->field decoder is supplied.

The logger is deliberately framework-light (std ``csv``/``json`` + matplotlib for
the optional heatmap) so any example under ``examples/`` can reuse it.
"""
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf

from eb_jepa.logging import get_logger

logger = get_logger(__name__)

_STEP_FIELDS = ["step", "epoch", "total_loss", "vc_loss", "pred_loss", "lr", "wall_time"]
_EPOCH_FIELDS = ["epoch", "train_loss", "val_loss", "epoch_time_s", "wall_time"]


def _git_sha() -> Optional[str]:
    """Best-effort current git commit; ``None`` if not a repo / git unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


class ExperimentLogger:
    """Create + own a single timestamped run directory and write all run artifacts.

    Layout under ``base_dir``::

        run_YYYY-MM-DD_HH-MM-SS/
          config.yaml           resolved config snapshot
          experiment.json       metadata + per-epoch metrics (rewritten each epoch)
          losses_step.csv       step,epoch,total_loss,vc_loss,pred_loss,lr,wall_time
          losses_epoch.csv      epoch,train_loss,val_loss,epoch_time_s,wall_time
          checkpoints/          training checkpoints
          predictions/          latent_metrics.json + latent_epoch{E}.png (+ field_*)
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        config: Optional[Union[Dict, DictConfig]] = None,
        name: Optional[str] = None,
    ):
        stamp = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        run_name = f"{name}_{stamp}" if name else stamp
        self.path = (Path(base_dir) / run_name).absolute()
        self.checkpoint_dir = self.path / "checkpoints"
        self.predictions_dir = self.path / "predictions"
        for d in (self.path, self.checkpoint_dir, self.predictions_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.created = datetime.now().isoformat(timespec="seconds")
        self.meta: Dict[str, Any] = {"git_sha": _git_sha()}
        self.epochs: list = []
        self.predictions: Dict[str, Any] = {}

        # Resolve + snapshot the config (so the run is fully reproducible from disk).
        self.config: Optional[dict] = None
        if config is not None:
            self.config = (
                OmegaConf.to_container(config, resolve=True)
                if isinstance(config, DictConfig)
                else dict(config)
            )
            OmegaConf.save(OmegaConf.create(self.config), self.path / "config.yaml")

        self._step_csv = self.path / "losses_step.csv"
        self._epoch_csv = self.path / "losses_epoch.csv"
        self._write_header(self._step_csv, _STEP_FIELDS)
        self._write_header(self._epoch_csv, _EPOCH_FIELDS)

        logger.info(f"Experiment logging to {self.path}")
        self.save_json()

    @staticmethod
    def _write_header(path: Path, fields) -> None:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    @staticmethod
    def _append_row(path: Path, fields, values: Dict[str, Any]) -> None:
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow([values.get(k, "") for k in fields])

    # ----------------------------- metadata ------------------------------- #
    def set_meta(self, key: str, value: Any) -> None:
        self.meta[key] = value

    def update_meta(self, values: Dict[str, Any]) -> None:
        self.meta.update(values)

    # ----------------------------- metrics -------------------------------- #
    def log_step(self, step: int, epoch: int, metrics: Dict[str, float]) -> None:
        """Append one row of per-step training losses to ``losses_step.csv``."""
        row = {"step": step, "epoch": epoch, "wall_time": time.time(), **metrics}
        self._append_row(self._step_csv, _STEP_FIELDS, row)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Append per-epoch summary, retain it in JSON, and persist the JSON."""
        row = {"epoch": epoch, "wall_time": time.time(), **metrics}
        self._append_row(self._epoch_csv, _EPOCH_FIELDS, row)
        self.epochs.append({"epoch": epoch, **metrics})
        self.save_json()

    # -------------------- prediction vs actual (latent) ------------------- #
    @torch.no_grad()
    def log_latent_prediction(
        self,
        jepa,
        encoder,
        batch,
        device,
        context_length: int,
        horizon: int,
        epoch: int,
        max_horizons_plotted: int = 4,
    ) -> Dict[str, Any]:
        """Compare rolled-out latents ``z_hat`` to the encoder target ``z``.

        Autoregressive rollout (same call as ``examples/gray_scott/eval.py``'s
        ``rollout_latents``) gives ``z_hat`` ``[B, D, C+H, h, w]``; the target is
        ``encoder(x)`` over the same frames. Logs per-horizon mean MSE to
        ``predictions/latent_metrics.json`` and a target/pred/abs-error heatmap.
        """
        was_training = jepa.training
        jepa.eval()
        x = batch["video"].to(device)  # [B, 2, T, h, w]
        C, H = context_length, horizon

        z_target = encoder(x)  # [B, D, T, h, w]
        z_hat, _ = jepa.unroll(
            x[:, :, :C], actions=None, nsteps=H, unroll_mode="autoregressive",
            ctxt_window_time=C, compute_loss=False, return_all_steps=False,
        )  # [B, D, C+H, h, w]

        # Horizon h (1..H) lives at time index C-1+h in both tensors.
        mse_per_h = []
        for h in range(1, H + 1):
            idx = C - 1 + h
            if idx >= z_hat.shape[2] or idx >= z_target.shape[2]:
                break
            mse = ((z_hat[:, :, idx] - z_target[:, :, idx]) ** 2).mean().item()
            mse_per_h.append(mse)

        record = {
            "epoch": epoch,
            "horizon": list(range(1, len(mse_per_h) + 1)),
            "latent_mse": mse_per_h,
        }
        self.predictions.setdefault("latent", {})[str(epoch)] = record
        with open(self.predictions_dir / "latent_metrics.json", "w") as f:
            json.dump(self.predictions["latent"], f, indent=2)

        self._plot_latent_panel(
            z_target, z_hat, C, len(mse_per_h), epoch, max_horizons_plotted
        )
        self.save_json()
        if was_training:
            jepa.train()
        return record

    def _plot_latent_panel(self, z_target, z_hat, C, n_h, epoch, max_cols) -> None:
        """3-row (target / pred / |error|) heatmap of latent channel 0, sample 0."""
        if n_h == 0:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:  # plotting is best-effort, never fail a run for it
            logger.warning(f"Skipping latent panel (matplotlib unavailable): {e}")
            return

        import numpy as np

        cols = min(max_cols, n_h)
        horizons = np.linspace(1, n_h, cols, dtype=int).tolist()
        rows = ["target z", "pred z_hat", "|error|"]
        fig, axes = plt.subplots(3, cols, figsize=(2.2 * cols, 6.6), squeeze=False)
        for j, h in enumerate(horizons):
            idx = C - 1 + h
            tgt = z_target[0, 0, idx].float().cpu().numpy()
            prd = z_hat[0, 0, idx].float().cpu().numpy()
            err = np.abs(tgt - prd)
            for i, img in enumerate((tgt, prd, err)):
                ax = axes[i][j]
                ax.imshow(img, cmap="viridis")
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"h={h}", fontsize=9)
                if j == 0:
                    ax.set_ylabel(rows[i], fontsize=9)
        fig.suptitle(f"Latent prediction vs target — epoch {epoch}", fontsize=11)
        fig.tight_layout()
        fig.savefig(self.predictions_dir / f"latent_epoch{epoch}.png", dpi=120)
        plt.close(fig)

    # --------------------- prediction vs actual (field) ------------------- #
    @torch.no_grad()
    def log_field_prediction(
        self,
        jepa,
        encoder,
        decoder,
        batch,
        device,
        context_length: int,
        horizon: int,
        epoch: int,
    ) -> Optional[Dict[str, Any]]:
        """Optional field-space comparison; runs only if ``decoder`` is provided.

        Decodes the latent rollout back to the 2-channel field and scores
        per-horizon field MSE vs ground truth. Returns ``None`` (and records the
        reason) when no decoder is available.
        """
        if decoder is None:
            self.predictions["field"] = "skipped (no decoder)"
            self.save_json()
            return None

        was_training = jepa.training
        jepa.eval()
        x = batch["video"].to(device)  # [B, 2, T, h, w]
        C, H = context_length, horizon
        z_hat, _ = jepa.unroll(
            x[:, :, :C], actions=None, nsteps=H, unroll_mode="autoregressive",
            ctxt_window_time=C, compute_loss=False, return_all_steps=False,
        )
        field_hat = decoder(z_hat)  # [B, 2, C+H, h, w]

        mse_per_h = []
        for h in range(1, H + 1):
            idx = C - 1 + h
            if idx >= field_hat.shape[2] or idx >= x.shape[2]:
                break
            mse = ((field_hat[:, :, idx] - x[:, :, idx]) ** 2).mean().item()
            mse_per_h.append(mse)

        record = {
            "epoch": epoch,
            "horizon": list(range(1, len(mse_per_h) + 1)),
            "field_mse": mse_per_h,
        }
        field = self.predictions.get("field")
        if not isinstance(field, dict):
            field = {}
        field[str(epoch)] = record
        self.predictions["field"] = field
        with open(self.predictions_dir / "field_metrics.json", "w") as f:
            json.dump(field, f, indent=2)
        self.save_json()
        if was_training:
            jepa.train()
        return record

    # ------------------------------- json --------------------------------- #
    def save_json(self) -> None:
        payload = {
            "run_dir": str(self.path),
            "created": self.created,
            "meta": self.meta,
            "config": self.config,
            "epochs": self.epochs,
            "predictions": self.predictions,
        }
        with open(self.path / "experiment.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)
