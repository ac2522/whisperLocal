"""Tests for config.startup_diagnostics."""

import logging
from unittest.mock import patch

from config.startup_diagnostics import (
    _missing_cuda_deps,
    _onnxruntime_cuda_provider_present,
    log_startup_diagnostics,
)


class TestLogStartupDiagnostics:
    def test_logs_onnxruntime_presence(self, caplog):
        with caplog.at_level(logging.INFO):
            log_startup_diagnostics()
        assert any(
            "onnxruntime installed" in r.message
            or "onnxruntime not installed" in r.message
            for r in caplog.records
        )

    def test_warns_when_cuda_deps_missing(self, caplog):
        # Pretend onnxruntime's CUDA provider lib is present but the runtime
        # deps are not. Never actually import onnxruntime.
        with patch(
            "config.startup_diagnostics._onnxruntime_cuda_provider_present",
            return_value=True,
        ), patch(
            "config.startup_diagnostics._missing_cuda_deps",
            return_value=["libcudnn.so.9"],
        ):
            with caplog.at_level(logging.WARNING):
                log_startup_diagnostics()

        assert any(
            "libcudnn.so.9" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )

    def test_no_warning_when_all_deps_present(self, caplog):
        with patch(
            "config.startup_diagnostics._onnxruntime_cuda_provider_present",
            return_value=True,
        ), patch(
            "config.startup_diagnostics._missing_cuda_deps", return_value=[]
        ):
            with caplog.at_level(logging.WARNING):
                log_startup_diagnostics()

        assert not any(
            "libcudnn" in r.message or "libcublas" in r.message
            for r in caplog.records
        )

    def test_does_not_import_onnxruntime(self):
        """Critical: startup diagnostics must NOT import onnxruntime because
        that triggers CUDA runtime init before ggml loads its own — causing
        a hard abort in ggml_cuda_init.
        """
        import sys
        sys.modules.pop("onnxruntime", None)
        log_startup_diagnostics()
        assert "onnxruntime" not in sys.modules


class TestMissingCudaDeps:
    def test_missing_fake_lib_is_reported(self):
        with patch("ctypes.CDLL", side_effect=OSError("not found")):
            missing = _missing_cuda_deps()
        assert "libcudnn.so.9" in missing
        assert "libcublas.so.12" in missing
        assert "libcublasLt.so.12" in missing

    def test_returns_empty_when_all_loadable(self):
        with patch("ctypes.CDLL"):
            missing = _missing_cuda_deps()
        assert missing == []


class TestOnnxruntimeCudaProviderPresent:
    def test_false_when_onnxruntime_not_installed(self):
        with patch("importlib.util.find_spec", return_value=None):
            assert _onnxruntime_cuda_provider_present() is False

    def test_does_not_actually_import_onnxruntime(self):
        import sys
        sys.modules.pop("onnxruntime", None)
        _onnxruntime_cuda_provider_present()
        assert "onnxruntime" not in sys.modules
