"""Model manager for downloading and managing Whisper GGML model files."""

import glob
import os
import shutil
import urllib.request

from huggingface_hub import snapshot_download

AVAILABLE_MODELS = [
    # ── Standard Whisper (ggerganov/whisper.cpp) ──────────────────────
    {"name": "ggml-base.bin", "type": "whisper", "size_mb": 142,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
     "description": "Base (~142MB) — good starting point"},
    {"name": "ggml-base.en.bin", "type": "whisper", "size_mb": 142,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
     "description": "Base English-only (~142MB)"},
    {"name": "ggml-small.bin", "type": "whisper", "size_mb": 466,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
     "description": "Small (~466MB) — better accuracy"},
    {"name": "ggml-small.en.bin", "type": "whisper", "size_mb": 466,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
     "description": "Small English-only (~466MB)"},
    {"name": "ggml-medium.bin", "type": "whisper", "size_mb": 1533,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
     "description": "Medium (~1.5GB) — high accuracy"},
    {"name": "ggml-medium-q5_0.bin", "type": "whisper", "size_mb": 539,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin",
     "description": "Medium quantized Q5 (~539MB)"},
    {"name": "ggml-large-v3.bin", "type": "whisper", "size_mb": 3095,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
     "description": "Large V3 (~3GB) — best accuracy"},
    {"name": "ggml-large-v3-q5_0.bin", "type": "whisper", "size_mb": 1080,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_0.bin",
     "description": "Large V3 quantized Q5 (~1GB)"},
    {"name": "ggml-large-v3-turbo.bin", "type": "whisper", "size_mb": 1624,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
     "description": "Large V3 Turbo (~1.6GB) — fast + accurate"},
    {"name": "ggml-large-v3-turbo-q5_0.bin", "type": "whisper", "size_mb": 574,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
     "description": "Large V3 Turbo Q5 (~574MB)"},
    {"name": "ggml-large-v3-turbo-q8_0.bin", "type": "whisper", "size_mb": 874,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin",
     "description": "Large V3 Turbo Q8 (~874MB)"},

    # ── Distil-Whisper (distilled, 5-6x faster) ──────────────────────
    {"name": "ggml-distil-small.en.bin", "type": "whisper", "size_mb": 334,
     "url": "https://huggingface.co/distil-whisper/distil-small.en/resolve/main/ggml-distil-small.en.bin",
     "description": "Distil Small EN (~334MB) — 4x faster than large-v2"},
    {"name": "ggml-distil-medium.en.bin", "type": "whisper", "size_mb": 789,
     "url": "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin",
     "description": "Distil Medium EN (~789MB) — 4x faster, <1% WER loss"},
    {"name": "ggml-distil-large-v3.bin", "type": "whisper", "size_mb": 756,
     "url": "https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin",
     "description": "Distil Large V3 (~756MB) — 5x faster than large-v3"},
    {"name": "ggml-distil-large-v3.5.bin", "type": "whisper", "size_mb": 756,
     "url": "https://huggingface.co/distil-whisper/distil-large-v3.5-ggml/resolve/main/ggml-model.bin",
     "description": "Distil Large V3.5 (~756MB) — latest distilled, best speed/accuracy"},

    # ── NVIDIA Parakeet (TDT 0.6B, ONNX INT8) ────────────────────────
    {"name": "parakeet-tdt-0.6b-v2-int8", "type": "parakeet", "size_mb": 600,
     "hf_repo": "istupakov/parakeet-tdt-0.6b-v2-onnx",
     "hf_revision": "main",
     "description": "Parakeet TDT v2 INT8 (~600MB) — English-only, very fast on GPU"},
    {"name": "parakeet-tdt-0.6b-v3-int8", "type": "parakeet", "size_mb": 670,
     "hf_repo": "istupakov/parakeet-tdt-0.6b-v3-onnx",
     "hf_revision": "main",
     "description": "Parakeet TDT v3 INT8 (~670MB) — multilingual (25 EU langs), fast on GPU"},
]

# Build a lookup dict for quick access by model name.
_MODELS_BY_NAME = {m["name"]: m for m in AVAILABLE_MODELS}


class ModelManager:
    """Manage Whisper GGML model files on disk."""

    def __init__(self, models_dir: str) -> None:
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def list_downloaded(self) -> list[dict]:
        """Return a list of dicts for every downloaded model.

        Recognizes two on-disk shapes:
          * Whisper: any single ``.bin`` file in ``models_dir``.
          * Parakeet: any subdirectory containing ``encoder-model*.onnx``.

        Each dict has keys: name, path, size_mb, description, type.
        """
        results = []
        for filename in sorted(os.listdir(self.models_dir)):
            full_path = os.path.join(self.models_dir, filename)

            if os.path.isfile(full_path) and filename.endswith(".bin"):
                size_bytes = os.path.getsize(full_path)
                model_type = "whisper"
            elif os.path.isdir(full_path) and glob.glob(
                os.path.join(full_path, "encoder-model*.onnx")
            ):
                size_bytes = sum(
                    os.path.getsize(os.path.join(dirpath, f))
                    for dirpath, _, files in os.walk(full_path)
                    for f in files
                )
                model_type = "parakeet"
            else:
                continue

            size_mb = round(size_bytes / (1024 * 1024), 1)
            known = _MODELS_BY_NAME.get(filename)
            description = known["description"] if known else filename

            results.append({
                "name": filename,
                "path": full_path,
                "size_mb": size_mb,
                "description": description,
                "type": model_type,
            })
        return results

    @staticmethod
    def list_available() -> list[dict]:
        """Return the static list of models available for download.

        Each dict has keys: name, size_mb, description.
        """
        return [dict(m) for m in AVAILABLE_MODELS]

    def is_downloaded(self, model_name: str) -> bool:
        """Return True if ``model_name`` exists as a file or directory."""
        path = os.path.join(self.models_dir, model_name)
        return os.path.isfile(path) or os.path.isdir(path)

    def get_model_path(self, model_name: str) -> str:
        """Return the full path for ``model_name`` (file or directory).

        Raises ``FileNotFoundError`` if the model has not been downloaded.
        """
        path = os.path.join(self.models_dir, model_name)
        if not (os.path.isfile(path) or os.path.isdir(path)):
            raise FileNotFoundError(
                f"Model '{model_name}' not found in {self.models_dir}"
            )
        return path

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    def delete_model(self, model_name: str) -> None:
        """Delete the model file or directory from ``models_dir``."""
        path = os.path.join(self.models_dir, model_name)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    def download_model(self, model_name: str, progress_callback=None) -> str:
        """Download ``model_name`` into ``models_dir``.

        Whisper models stream a single ``.bin`` file from the URL in the
        catalog. Parakeet models snapshot a HuggingFace repo into a
        subdirectory. Both flows stage into ``<name>.partial`` and rename
        atomically on success so incomplete downloads are never visible.
        """
        known = _MODELS_BY_NAME.get(model_name)
        model_type = (known or {}).get("type", "whisper")

        if model_type == "parakeet":
            return self._download_parakeet(known, model_name, progress_callback)
        return self._download_whisper(known, model_name, progress_callback)

    def _download_whisper(self, known, model_name, progress_callback) -> str:
        if known and "url" in known:
            url = known["url"]
        else:
            url = (
                "https://huggingface.co/ggerganov/whisper.cpp/"
                f"resolve/main/{model_name}"
            )
        dest_path = os.path.join(self.models_dir, model_name)
        partial_path = dest_path + ".partial"

        def _reporthook(block_num: int, block_size: int, total_size: int):
            if progress_callback is None:
                return
            downloaded = block_num * block_size
            percent = (
                min(downloaded / total_size * 100, 100.0)
                if total_size > 0
                else 0.0
            )
            progress_callback(percent, downloaded, total_size)

        try:
            urllib.request.urlretrieve(url, partial_path, reporthook=_reporthook)
            os.rename(partial_path, dest_path)
        except Exception:
            if os.path.exists(partial_path):
                os.remove(partial_path)
            raise
        return dest_path

    def _download_parakeet(self, known, model_name, progress_callback) -> str:
        if not known or "hf_repo" not in known:
            raise ValueError(
                f"Parakeet model '{model_name}' has no hf_repo entry in the catalog"
            )

        dest_path = os.path.join(self.models_dir, model_name)
        partial_path = dest_path + ".partial"

        if progress_callback is not None:
            progress_callback(0, 0, 0)

        try:
            snapshot_download(
                repo_id=known["hf_repo"],
                revision=known.get("hf_revision", "main"),
                local_dir=partial_path,
            )
            os.rename(partial_path, dest_path)
        except Exception:
            if os.path.isdir(partial_path):
                shutil.rmtree(partial_path, ignore_errors=True)
            raise

        if progress_callback is not None:
            progress_callback(100, 0, 0)
        return dest_path
