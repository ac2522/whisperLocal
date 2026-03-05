"""Model manager for downloading and managing Whisper GGML model files."""

import os
import urllib.request

AVAILABLE_MODELS = [
    # ── Standard Whisper (ggerganov/whisper.cpp) ──────────────────────
    {"name": "ggml-base.bin", "size_mb": 142,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
     "description": "Base (~142MB) — good starting point"},
    {"name": "ggml-base.en.bin", "size_mb": 142,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
     "description": "Base English-only (~142MB)"},
    {"name": "ggml-small.bin", "size_mb": 466,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
     "description": "Small (~466MB) — better accuracy"},
    {"name": "ggml-small.en.bin", "size_mb": 466,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
     "description": "Small English-only (~466MB)"},
    {"name": "ggml-medium.bin", "size_mb": 1533,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
     "description": "Medium (~1.5GB) — high accuracy"},
    {"name": "ggml-medium-q5_0.bin", "size_mb": 539,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin",
     "description": "Medium quantized Q5 (~539MB)"},
    {"name": "ggml-large-v3.bin", "size_mb": 3095,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
     "description": "Large V3 (~3GB) — best accuracy"},
    {"name": "ggml-large-v3-q5_0.bin", "size_mb": 1080,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_0.bin",
     "description": "Large V3 quantized Q5 (~1GB)"},
    {"name": "ggml-large-v3-turbo.bin", "size_mb": 1624,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
     "description": "Large V3 Turbo (~1.6GB) — fast + accurate"},
    {"name": "ggml-large-v3-turbo-q5_0.bin", "size_mb": 574,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
     "description": "Large V3 Turbo Q5 (~574MB)"},
    {"name": "ggml-large-v3-turbo-q8_0.bin", "size_mb": 874,
     "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin",
     "description": "Large V3 Turbo Q8 (~874MB)"},

    # ── Distil-Whisper (distilled, 5-6x faster) ──────────────────────
    {"name": "ggml-distil-small.en.bin", "size_mb": 334,
     "url": "https://huggingface.co/distil-whisper/distil-small.en/resolve/main/ggml-distil-small.en.bin",
     "description": "Distil Small EN (~334MB) — 4x faster than large-v2"},
    {"name": "ggml-distil-medium.en.bin", "size_mb": 789,
     "url": "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin",
     "description": "Distil Medium EN (~789MB) — 4x faster, <1% WER loss"},
    {"name": "ggml-distil-large-v3.bin", "size_mb": 756,
     "url": "https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin",
     "description": "Distil Large V3 (~756MB) — 5x faster than large-v3"},
    {"name": "ggml-distil-large-v3.5.bin", "size_mb": 756,
     "url": "https://huggingface.co/distil-whisper/distil-large-v3.5-ggml/resolve/main/ggml-model.bin",
     "description": "Distil Large V3.5 (~756MB) — latest distilled, best speed/accuracy"},
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
        """Return a list of dicts for every .bin file present in *models_dir*.

        Each dict has keys: name, path, size_mb, description.
        """
        results = []
        for filename in sorted(os.listdir(self.models_dir)):
            if not filename.endswith(".bin"):
                continue
            filepath = os.path.join(self.models_dir, filename)
            if not os.path.isfile(filepath):
                continue
            size_bytes = os.path.getsize(filepath)
            size_mb = round(size_bytes / (1024 * 1024), 1)

            # Try to get description from known models, otherwise generic.
            known = _MODELS_BY_NAME.get(filename)
            description = known["description"] if known else filename

            results.append({
                "name": filename,
                "path": filepath,
                "size_mb": size_mb,
                "description": description,
            })
        return results

    @staticmethod
    def list_available() -> list[dict]:
        """Return the static list of models available for download.

        Each dict has keys: name, size_mb, description.
        """
        return [dict(m) for m in AVAILABLE_MODELS]

    def is_downloaded(self, model_name: str) -> bool:
        """Return *True* if *model_name* exists in *models_dir*."""
        return os.path.isfile(os.path.join(self.models_dir, model_name))

    def get_model_path(self, model_name: str) -> str:
        """Return the full path for *model_name*.

        Raises ``FileNotFoundError`` if the model has not been downloaded.
        """
        path = os.path.join(self.models_dir, model_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Model '{model_name}' not found in {self.models_dir}"
            )
        return path

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    def delete_model(self, model_name: str) -> None:
        """Delete the model file from *models_dir*."""
        path = os.path.join(self.models_dir, model_name)
        if os.path.isfile(path):
            os.remove(path)

    def download_model(self, model_name: str, progress_callback=None) -> str:
        """Download *model_name* from HuggingFace into *models_dir*.

        The file is first downloaded to a ``.partial`` file and then renamed
        on success so that incomplete downloads are never mistaken for valid
        models.

        Parameters
        ----------
        model_name:
            Filename of the model (e.g. ``ggml-base.bin``).
        progress_callback:
            Optional callable ``(percent, downloaded, total_size)`` invoked
            during the download to report progress.

        Returns
        -------
        str
            Full path of the downloaded model file.
        """
        known = _MODELS_BY_NAME.get(model_name)
        if known and "url" in known:
            url = known["url"]
        else:
            url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{model_name}"
        dest_path = os.path.join(self.models_dir, model_name)
        partial_path = dest_path + ".partial"

        def _reporthook(block_num: int, block_size: int, total_size: int):
            if progress_callback is None:
                return
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(downloaded / total_size * 100, 100.0)
            else:
                percent = 0.0
            progress_callback(percent, downloaded, total_size)

        try:
            urllib.request.urlretrieve(url, partial_path, reporthook=_reporthook)
            os.rename(partial_path, dest_path)
        except Exception:
            # Clean up partial file on failure.
            if os.path.exists(partial_path):
                os.remove(partial_path)
            raise

        return dest_path
