"""Log availability of optional runtime dependencies at app startup.

Surfaces missing GPU libraries and optional packages so users can see in
the logs what acceleration is (or isn't) available and what they can
install to improve things. Never raises — all checks are best-effort.

Importantly: this module must NOT import heavy CUDA-using packages like
``onnxruntime`` at call time. Doing so pollutes the CUDA runtime state
before ``pywhispercpp`` / ``ggml`` gets to initialise its own CUDA
backend, causing a hard abort inside ``ggml_cuda_init()``. Every
capability check here is done via ``importlib.util.find_spec`` or a
``ctypes.CDLL`` probe so we never actually load the target module.
"""

import ctypes
import importlib.util
import logging

logger = logging.getLogger(__name__)


def log_startup_diagnostics() -> None:
    """Emit one-time INFO/WARNING log lines describing runtime capability."""
    _log_onnxruntime_providers()
    _log_hf_xet()


def _log_onnxruntime_providers() -> None:
    if importlib.util.find_spec("onnxruntime") is None:
        logger.warning(
            "onnxruntime not installed — Parakeet engine will be unavailable. "
            "Install with: pip install onnxruntime-gpu",
        )
        return

    # Detect CUDA provider availability without importing onnxruntime.
    # onnxruntime-gpu installs libonnxruntime_providers_cuda.so in its
    # capi/ directory; the module's availability is a strong indicator.
    cuda_provider_available = _onnxruntime_cuda_provider_present()
    logger.info(
        "onnxruntime installed (CUDA provider lib present: %s)",
        cuda_provider_available,
    )

    if cuda_provider_available:
        missing = _missing_cuda_deps()
        if missing:
            logger.warning(
                "onnxruntime CUDA provider is installed but required shared "
                "libraries are missing: %s. Parakeet CUDA inference will "
                "fall back to CPU. On Ubuntu you can install cuDNN via "
                "NVIDIA's CUDA apt repo — see "
                "https://developer.nvidia.com/cudnn-downloads "
                "(standard 'apt install libcudnn9-cuda-12' only works once "
                "the NVIDIA CUDA repo is configured).",
                ", ".join(missing),
            )


def _onnxruntime_cuda_provider_present() -> bool:
    """True if onnxruntime's CUDA provider shared library is on disk.

    This reads the file path, not the Python module, so it does not
    trigger any CUDA runtime initialisation.
    """
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None or not spec.origin:
        return False
    import os
    ort_dir = os.path.dirname(spec.origin)
    # onnxruntime/__init__.py lives alongside the capi/ directory.
    capi_dir = os.path.join(ort_dir, "capi")
    lib = os.path.join(capi_dir, "libonnxruntime_providers_cuda.so")
    return os.path.isfile(lib)


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
    if importlib.util.find_spec("hf_xet") is None:
        logger.info(
            "hf_xet not installed — Parakeet model downloads will use regular "
            "HTTP. For faster downloads: pip install hf_xet",
        )
        return
    logger.info("hf_xet available — Parakeet model downloads will use Xet")
