"""Log availability of optional runtime dependencies at app startup.

Surfaces missing GPU libraries and optional packages so users can see in
the logs what acceleration is (or isn't) available and what they can
install to improve things. Never raises — all checks are best-effort.

Importantly: this module must NOT actually *load* any CUDA-adjacent
shared library. Doing so pollutes the CUDA runtime state before
``pywhispercpp`` / ``ggml`` gets to initialise its own CUDA backend,
causing a hard abort inside ``ggml_cuda_init()``. Every capability
check here is filesystem-only (``importlib.util.find_spec`` or
``ldconfig -p`` parsing) — we never call ``dlopen`` / ``ctypes.CDLL``
on anything with ``cuda``, ``cublas``, or ``cudnn`` in its name.
"""

import importlib.util
import logging
import subprocess

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
    """Return names of CUDA runtime libs that ONNX Runtime's CUDA provider
    needs but are not currently registered with the dynamic linker on this
    machine, according to ``ldconfig -p``.

    Filesystem-only check — we never dlopen any CUDA library here, because
    loading the system libcublas pre-empts the bundled one that ggml is
    about to use and hard-aborts the whole process. cuDNN is the one the
    user normally has to install separately, so it's the most useful signal.
    """
    wanted = ("libcudnn.so.9",)
    try:
        result = subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True, timeout=5,
        )
        registered = result.stdout
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return []  # can't tell — stay silent

    return [libname for libname in wanted if libname not in registered]


def _log_hf_xet() -> None:
    if importlib.util.find_spec("hf_xet") is None:
        logger.info(
            "hf_xet not installed — Parakeet model downloads will use regular "
            "HTTP. For faster downloads: pip install hf_xet",
        )
        return
    logger.info("hf_xet available — Parakeet model downloads will use Xet")
