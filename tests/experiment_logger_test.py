"""Tests for eb_jepa.experiment_logger.ExperimentLogger (run-dir + JSON/CSV)."""
import csv
import json

from omegaconf import OmegaConf

from eb_jepa.experiment_logger import ExperimentLogger


def test_creates_run_dir_and_artifacts(tmp_path):
    cfg = OmegaConf.create({"model": {"dstc": 16}, "optim": {"lr": 1e-3}})
    logger = ExperimentLogger(tmp_path, config=cfg)

    assert logger.path.exists() and logger.path.name.startswith("run_")
    assert logger.checkpoint_dir.exists()
    assert logger.predictions_dir.exists()
    assert (logger.path / "config.yaml").exists()
    assert (logger.path / "experiment.json").exists()
    assert (logger.path / "losses_step.csv").exists()
    assert (logger.path / "losses_epoch.csv").exists()


def test_step_and_epoch_logging(tmp_path):
    logger = ExperimentLogger(tmp_path, config={"a": 1})
    logger.update_meta({"device": "cpu", "params_M": 1.23})

    for step in range(1, 4):
        logger.log_step(step, epoch=0, metrics={
            "total_loss": 0.1 * step, "vc_loss": 0.01, "pred_loss": 0.02, "lr": 1e-3,
        })
    logger.log_epoch(0, {"train_loss": 0.3, "val_loss": 0.25, "epoch_time_s": 12.0})
    logger.log_epoch(1, {"train_loss": 0.2, "val_loss": 0.18, "epoch_time_s": 11.0})

    with open(logger.path / "losses_step.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[0]["total_loss"] == "0.1"
    assert set(rows[0].keys()) == {
        "step", "epoch", "total_loss", "vc_loss", "pred_loss", "lr", "wall_time",
    }

    with open(logger.path / "losses_epoch.csv") as f:
        epoch_rows = list(csv.DictReader(f))
    assert len(epoch_rows) == 2
    assert epoch_rows[1]["val_loss"] == "0.18"

    with open(logger.path / "experiment.json") as f:
        payload = json.load(f)
    assert payload["meta"]["device"] == "cpu"
    assert len(payload["epochs"]) == 2
    assert payload["epochs"][0]["train_loss"] == 0.3
    assert payload["config"] == {"a": 1}


def test_field_prediction_skipped_without_decoder(tmp_path):
    logger = ExperimentLogger(tmp_path)
    out = logger.log_field_prediction(
        jepa=None, encoder=None, decoder=None, batch=None, device="cpu",
        context_length=2, horizon=4, epoch=0,
    )
    assert out is None
    with open(logger.path / "experiment.json") as f:
        payload = json.load(f)
    assert payload["predictions"]["field"] == "skipped (no decoder)"
