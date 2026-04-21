"""Tests for config.startup_diagnostics."""

import logging
from unittest.mock import patch

from config.startup_diagnostics import log_startup_diagnostics, _missing_cuda_deps


class TestLogStartupDiagnostics:
    def test_logs_onnx_providers(self, caplog):
        with caplog.at_level(logging.INFO):
            log_startup_diagnostics()
        assert any(
            "ONNX Runtime providers available" in r.message
            for r in caplog.records
        )

    def test_warns_when_cuda_deps_missing(self, caplog):
        # Pretend CUDA provider is listed but the libs are missing.
        with patch("onnxruntime.get_available_providers",
                   return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]), \
             patch("config.startup_diagnostics._missing_cuda_deps",
                   return_value=["libcudnn.so.9"]):
            with caplog.at_level(logging.WARNING):
                log_startup_diagnostics()

        assert any(
            "libcudnn.so.9" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )

    def test_no_warning_when_all_deps_present(self, caplog):
        with patch("onnxruntime.get_available_providers",
                   return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]), \
             patch("config.startup_diagnostics._missing_cuda_deps",
                   return_value=[]):
            with caplog.at_level(logging.WARNING):
                log_startup_diagnostics()

        assert not any(
            "libcudnn" in r.message or "libcublas" in r.message
            for r in caplog.records
        )


class TestMissingCudaDeps:
    def test_missing_fake_lib_is_reported(self):
        with patch("ctypes.CDLL", side_effect=OSError("not found")):
            missing = _missing_cuda_deps()
        # All three expected libs should be missing
        assert "libcudnn.so.9" in missing
        assert "libcublas.so.12" in missing
        assert "libcublasLt.so.12" in missing

    def test_returns_empty_when_all_loadable(self):
        with patch("ctypes.CDLL"):  # default MagicMock returns a mock
            missing = _missing_cuda_deps()
        assert missing == []
