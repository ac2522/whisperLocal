"""Model manager for downloading and managing Whisper GGML model files."""

import os
import urllib.request

HUGGINGFACE_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

AVAILABLE_MODELS = [
    {"name": "ggml-base.bin", "size_mb": 142, "description": "Base model (~142MB)"},
    {"name": "ggml-base.en.bin", "size_mb": 142, "description": "Base English-only (~142MB)"},
    {"name": "ggml-small.bin", "size_mb": 466, "description": "Small model (~466MB)"},
    {"name": "ggml-small.en.bin", "size_mb": 466, "description": "Small English-only (~466MB)"},
    {"name": "ggml-small-q5_1.bin", "size_mb": 181, "description": "Small quantized Q5 (~181MB)"},
    {"name": "ggml-medium.bin", "size_mb": 1533, "description": "Medium model (~1.5GB)"},
    {"name": "ggml-medium-q5_0.bin", "size_mb": 539, "description": "Medium quantized Q5 (~539MB)"},
    {"name": "ggml-large-v3.bin", "size_mb": 3095, "description": "Large V3 (~3GB)"},
    {"name": "ggml-large-v3-q5_0.bin", "size_mb": 1080, "description": "Large V3 quantized Q5 (~1GB)"},
    {"name": "ggml-large-v3-turbo.bin", "size_mb": 1624, "description": "Large V3 Turbo (~1.6GB)"},
    {"name": "ggml-large-v3-turbo-q5_0.bin", "size_mb": 574, "description": "Large V3 Turbo Q5 (~574MB)"},
    {"name": "ggml-large-v3-turbo-q8_0.bin", "size_mb": 874, "description": "Large V3 Turbo Q8 (~874MB)"},
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
        url = f"{HUGGINGFACE_BASE}/{model_name}"
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
