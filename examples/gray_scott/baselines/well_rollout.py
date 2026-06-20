"""Minimal, faithful port of The Well's evaluation rollout + VRMSE windowing.

We reuse The Well's own pieces wherever possible:
  * ``DefaultChannelsFirstFormatter`` for input/output tensor layout,
  * the ``ZScoreNormalization`` object carried by ``WellDataset`` (denormalize),
  * the official ``VRMSE`` metric (``the_well.benchmark.metrics.spatial.VRMSE``).

The only thing copied (not imported) is the autoregressive ``rollout_model`` loop
and ``denormalize``/``normalize`` from ``the_well.benchmark.trainer.training.Trainer``,
because that logic is bolted onto a hydra-instantiated ``Trainer`` (optimizer, folders,
loggers, ...) that we do not want to build just to score pretrained checkpoints. The
loop below is the ``train=False`` path of ``Trainer.rollout_model`` for the non-delta
(``WellDataset``) case, kept line-for-line so the protocol matches.

All metrics are computed in PHYSICAL (denormalized) space, per field, exactly like the
benchmark. VRMSE = sqrt( <(pred-true)^2>_space / (<(true-<true>_space)^2>_space + eps) ).
"""

from typing import Dict, List, Tuple

import numpy as np
import torch

from the_well.benchmark.metrics.spatial import VRMSE
from the_well.data.data_formatter import DefaultChannelsFirstFormatter

# Rollout windows reported by The Well (paper Table 3 / benchmarks page). These are
# python slices on the 0-indexed per-step rollout VRMSE array, matching
# ``Trainer.temporal_split_losses`` which labels windows as f"{start}:{end}" and
# averages ``loss_values[start:end]``. So "6:12" -> predicted horizons 7..12 (1-indexed),
# "13:30" -> predicted horizons 14..30.
ROLLOUT_WINDOWS: Dict[str, Tuple[int, int]] = {"6:12": (6, 12), "13:30": (13, 30)}


class WellEvaluator:
    """Holds the formatter + normalization needed to roll a model out and score it."""

    def __init__(self, metadata, dset_norm, device: torch.device, is_delta: bool = False):
        self.meta = metadata
        self.dset_norm = dset_norm
        self.use_normalization = dset_norm is not None
        self.is_delta = is_delta
        self.device = device
        self.formatter = DefaultChannelsFirstFormatter(metadata)
        self.vrmse = VRMSE()
        self._spatial_dims = tuple(range(-metadata.n_spatial_dims - 1, -1))  # excludes field axis

    # --- normalization (ported from Trainer.normalize/denormalize, non-delta path) --- #
    def _normalize_inputs(self, batch_dict):
        if self.use_normalization:
            batch_dict["input_fields"] = self.dset_norm.normalize_flattened(
                batch_dict["input_fields"], "variable"
            )
            if "constant_fields" in batch_dict:
                batch_dict["constant_fields"] = self.dset_norm.normalize_flattened(
                    batch_dict["constant_fields"], "constant"
                )
        return batch_dict

    def _denormalize(self, batch_dict=None, direct_tensor=None):
        if self.use_normalization:
            if batch_dict is not None:
                batch_dict["input_fields"] = self.dset_norm.denormalize_flattened(
                    batch_dict["input_fields"], "variable"
                )
                if "constant_fields" in batch_dict:
                    batch_dict["constant_fields"] = self.dset_norm.denormalize_flattened(
                        batch_dict["constant_fields"], "constant"
                    )
            if direct_tensor is not None:
                # non-delta WellDataset -> full denormalization
                direct_tensor = self.dset_norm.denormalize_flattened(direct_tensor, "variable")
        return batch_dict, direct_tensor

    @torch.inference_mode()
    def rollout_model(self, model, batch, max_rollout_steps: int):
        """Autoregressive rollout, denormalized. Returns (y_pred, y_ref).

        Both are channels-last ``[B, T, *spatial, C]`` in physical space. Faithful port
        of ``Trainer.rollout_model(train=False)`` for the non-delta case.
        """
        _, y_ref = self.formatter.process_input(batch)
        rollout_steps = min(y_ref.shape[1], max_rollout_steps)
        y_ref = y_ref[:, :rollout_steps]
        _, y_ref = self._denormalize(None, y_ref)

        moving_batch = dict(batch)
        moving_batch["input_fields"] = moving_batch["input_fields"].to(self.device)
        if "constant_fields" in moving_batch:
            moving_batch["constant_fields"] = moving_batch["constant_fields"].to(self.device)

        y_preds = []
        for i in range(rollout_steps):
            if self.use_normalization and i > 0:
                moving_batch = self._normalize_inputs(moving_batch)
            inputs, _ = self.formatter.process_input(moving_batch)
            inputs = [x.to(self.device) for x in inputs]
            y_pred = model(*inputs)
            y_pred = self.formatter.process_output_channel_last(y_pred)
            moving_batch, y_pred = self._denormalize(moving_batch, y_pred)
            y_pred = self.formatter.process_output_expand_time(y_pred)
            if i != rollout_steps - 1:
                moving_batch["input_fields"] = torch.cat(
                    [moving_batch["input_fields"][:, 1:], y_pred], dim=1
                )
            y_preds.append(y_pred)
        y_pred_out = torch.cat(y_preds, dim=1)
        return y_pred_out, y_ref.to(self.device)

    # --- cheap reference baselines (physical space) --- #
    @torch.inference_mode()
    def persistence_and_mean(self, batch, max_rollout_steps: int):
        """Persistence (repeat last input frame) and spatial-mean baselines.

        Returns (persistence, mean_pred, y_ref), all channels-last physical-space
        ``[B, T, *spatial, C]``.
        """
        _, y_ref = self.formatter.process_input(batch)
        rollout_steps = min(y_ref.shape[1], max_rollout_steps)
        y_ref = y_ref[:, :rollout_steps]
        _, y_ref = self._denormalize(None, y_ref)
        y_ref = y_ref.to(self.device)

        inp = batch["input_fields"].to(self.device)  # [B, T_in, *spatial, C] (normalized)
        if self.use_normalization:
            inp = self.dset_norm.denormalize_flattened(inp, "variable")
        last = inp[:, -1:]  # [B, 1, *spatial, C]
        persistence = last.expand(-1, rollout_steps, *([-1] * (last.ndim - 2)))
        space_axes = tuple(range(2, 2 + self.meta.n_spatial_dims))
        spatial_mean = last.mean(dim=space_axes, keepdim=True)  # [B,1,1,1,C]
        mean_pred = spatial_mean.expand_as(persistence)
        return persistence.contiguous(), mean_pred.contiguous(), y_ref

    # --- metrics --- #
    def vrmse_per_step(self, y_pred, y_ref) -> torch.Tensor:
        """Official per-sample VRMSE, mean over batch -> ``[T, C]`` (per step, per field)."""
        v = self.vrmse(y_pred, y_ref, self.meta)  # [B, T, C]
        return v.mean(0).float().cpu()

    def vrmse_per_sample(self, y_pred, y_ref) -> torch.Tensor:
        """Official per-sample, field-averaged VRMSE -> ``[B, T]`` (one curve per trajectory).

        Used to form per-trajectory confidence bands; we deliberately do NOT average over
        the batch here so the spread across initial conditions is preserved."""
        v = self.vrmse(y_pred, y_ref, self.meta)  # [B, T, C]
        return v.mean(-1).float().cpu()  # [B, T]

    def vrmse_terms_per_step(self, y_pred, y_ref) -> Tuple[torch.Tensor, torch.Tensor]:
        """Numerator (MSE) and denominator (variance) summed over batch+space -> ``[T, C]`` each.

        Lets the caller form an aggregated VRMSE = sqrt(sum_num / sum_den), which is more
        robust than per-sample ratios on near-uniform fields (e.g. Gray-Scott B)."""
        num = ((y_pred - y_ref) ** 2).mean(dim=self._spatial_dims)  # [B, T, C]
        mean_ref = y_ref.mean(dim=self._spatial_dims, keepdim=True)
        den = ((y_ref - mean_ref) ** 2).mean(dim=self._spatial_dims)  # [B, T, C]
        return num.sum(0).float().cpu(), den.sum(0).float().cpu()


def field_average(per_step_field: torch.Tensor) -> np.ndarray:
    """``[T, C]`` -> ``[T]`` averaged over fields (matches benchmark 'full' aggregation)."""
    return per_step_field.mean(dim=-1).numpy()


def window_means(per_step: np.ndarray) -> Dict[str, float]:
    """Mean VRMSE over each reported rollout window (python-slice convention)."""
    out: Dict[str, float] = {}
    for name, (start, end) in ROLLOUT_WINDOWS.items():
        seg = per_step[start:end]
        out[name] = float(np.mean(seg)) if seg.size > 0 else float("nan")
    return out


def field_names(metadata) -> List[str]:
    from the_well.data.utils import flatten_field_names

    return flatten_field_names(metadata, include_constants=False)
