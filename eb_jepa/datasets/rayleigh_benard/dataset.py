"""Rayleigh-Benard convection (The Well) as a physical video for JEPA.

4 physical fields (buoyancy, pressure, velocity_x, velocity_y) on a 512 x 128 grid,
200 steps/trajectory (Chebyshev-sampled in y but stored on a uniform pixel grid, so
loading is unchanged). A training item is a clip of ``n_frames`` frames with all fields
stacked as channels -> ``[4, T, 512, 128]``, z-scored with the dataset's own
``stats.yaml``.

Like the turbulent_radiative_layer_2D loader, field names and per-field z-score
statistics are read from The Well metadata (we wrap ``the_well.data.WellDataset`` with
``use_normalization=True`` + ``ZScoreNormalization``) rather than hardcoded; we only
reshape channels-last ``input_fields`` (``[T, H, W, C]``) to the repo's video format
``[C, T, H, W]``.

This is the heaviest target dataset (342 GB); prefer a staged copy on /lustre/work.
"""
import os
from dataclasses import dataclass

import torch

from the_well.data import WellDataset
from the_well.data.normalization import ZScoreNormalization

DATASET_NAME = "rayleigh_benard"


def resolve_well_base_path() -> str:
    for env in ("THE_WELL_BASE_PATH", "EBJEPA_DSETS"):
        p = os.environ.get(env)
        if p:
            return p
    return "hf://datasets/polymathic-ai/"


@dataclass
class RayleighBenardConfig:
    well_base_path: str = ""           # parent dir holding {dataset}/data/{split}; "" -> env/HF
    well_dataset_name: str = DATASET_NAME
    split: str = "train"               # train | valid | test
    channels: int = 4                  # buoyancy, pressure, velocity_x, velocity_y
    img_size: tuple = (512, 128)
    n_frames: int = 16                 # frames per clip
    time_stride: int = 1               # stride 1 to match the official protocol
    epoch_size: int = 8000             # random clips drawn per epoch (0 -> use full dataset)
    batch_size: int = 4               # larger grid -> smaller batch
    num_workers: int = 8


class RayleighBenardDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: RayleighBenardConfig):
        self.cfg = cfg
        split = {"valid": "valid", "validation": "valid"}.get(cfg.split, cfg.split)
        self.wd = WellDataset(
            well_base_path=cfg.well_base_path or resolve_well_base_path(),
            well_dataset_name=cfg.well_dataset_name,
            well_split_name=split,
            n_steps_input=cfg.n_frames,
            n_steps_output=1,  # required >= 1; we only use input_fields as the clip
            min_dt_stride=cfg.time_stride,
            max_dt_stride=cfg.time_stride,
            use_normalization=True,
            normalization_type=ZScoreNormalization,
        )

    def __len__(self):
        return len(self.wd)

    def __getitem__(self, idx):
        x = self.wd[idx]["input_fields"]      # [T, H, W, C], z-scored per field
        video = x.permute(3, 0, 1, 2).contiguous()  # [C, T, H, W]
        return {"video": video}


def make_loader(cfg: RayleighBenardConfig, shuffle=True):
    dataset = RayleighBenardDataset(cfg)
    sampler = None
    if shuffle and cfg.epoch_size:
        sampler = torch.utils.data.RandomSampler(
            dataset, replacement=True, num_samples=cfg.epoch_size
        )
        shuffle = False
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.num_workers > 0)
