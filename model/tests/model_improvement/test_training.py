import pytest
from pathlib import Path
from unittest.mock import patch
from model_improvement.training import detect_accelerator_config, find_checkpoint


class TestDetectAcceleratorConfig:
    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_config(self, _mock):
        cfg = detect_accelerator_config()
        assert cfg["precision"] == "bf16-mixed"
        assert cfg["accelerator"] == "auto"
        assert cfg["deterministic"] is True

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_mps_config(self, _mock_mps, _mock_cuda):
        cfg = detect_accelerator_config()
        assert cfg["precision"] == "32-true"
        assert cfg["accelerator"] == "auto"
        assert cfg["deterministic"] is False

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_cpu_config(self, _mock_mps, _mock_cuda):
        cfg = detect_accelerator_config()
        assert cfg["precision"] == "32-true"
        assert cfg["accelerator"] == "cpu"
        assert cfg["deterministic"] is False


class TestFindCheckpoint:
    def test_finds_existing_checkpoint(self, tmp_path):
        ckpt_dir = tmp_path / "A1" / "fold_0"
        ckpt_dir.mkdir(parents=True)
        ckpt_file = ckpt_dir / "epoch=5-val_loss=0.1234.ckpt"
        ckpt_file.touch()
        result = find_checkpoint(tmp_path, "A1", 0)
        assert result == ckpt_file

    def test_returns_none_when_no_checkpoint(self, tmp_path):
        result = find_checkpoint(tmp_path, "A1", 0)
        assert result is None

    def test_returns_none_for_empty_dir(self, tmp_path):
        ckpt_dir = tmp_path / "A1" / "fold_0"
        ckpt_dir.mkdir(parents=True)
        result = find_checkpoint(tmp_path, "A1", 0)
        assert result is None
