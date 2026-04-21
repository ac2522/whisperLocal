"""Log availability of optional runtime dependencies at app startup.

Surfaces missing GPU libraries and optional packages so users can see in
the logs what acceleration is (or isn't) available and what they can
install to improve things. Never raises — all checks are best-effort.
"""

import ctypes
import logging

logger = logging.getLogger(__name__)


def log_startup_diagnostics() -> None:
    """Emit one-time INFO/WARNING log lines describing runtime capability."""
    _log_onnxruntime_providers()
    _log_hf_xet()


def _log_onnxruntime_providers() -> None:
    try:
        import onnxruntime
    except ImportError:
        logger.warning(
            "onnxruntime not installed — Parakeet engine will be unavailable. "
            "Install with: pip install onnxruntime-gpu",
        )
        return

    providers = onnxruntime.get_available_providers()
    logger.info("ONNX Runtime providers available: %s", providers)

    if "CUDAExecutionProvider" in providers:
        missing = _missing_cuda_deps()
        if missing:
            logger.warning(
                "ONNX Runtime advertises CUDAExecutionProvider, but required "
                "shared libraries are missing: %s. Parakeet CUDA inference "
                "will fall back to CPU. On Ubuntu you can install cuDNN via "
                "NVIDIA's CUDA apt repo — see "
                "https://developer.nvidia.com/cudnn-downloads "
                "(standard 'apt install libcudnn9-cuda-12' only works once the "
                "NVIDIA CUDA repo is configured).",
                ", ".join(missing),
            )


def _missing_cuda_deps() -> list[str]:
    """Return names of CUDA runtime libs that ONNX Runtime's CUDA provider needs
    but cannot currently be dlopened on this machine.

    Keep the list focused on the libs that actually block Parakeet inference.
    """
    wanted = ("libcudnn.so.9", "libcublas.so.12", "libcublasLt.so.12")
    missing: list[str] = []
    for libname in wanted:
        try:
            ctypes.CDLL(libname)
        except OSError:
            missing.append(libname)
    return missing


def _log_hf_xet() -> None:
    try:
        import hf_xet  # noqa: F401
    except ImportError:
        logger.info(
            "hf_xet not installed — Parakeet model downloads will use regular "
            "HTTP. For faster downloads: pip install hf_xet",
        )
        return
    logger.info("hf_xet available — Parakeet model downloads will use Xet")
