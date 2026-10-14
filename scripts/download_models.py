from __future__ import annotations

import argparse
import hashlib
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/refs/heads/master/"
    "src/silero_vad/data/silero_vad.onnx"
)
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
MODEL_FILENAME = "silero_vad.onnx"
logger = logging.getLogger("chintu.models")


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest without reading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, expected_sha256: str) -> Path:
    """Resume a public model download and publish only a checksum-verified file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    existing_size = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {"User-Agent": "Chintu-AI/1.0"}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            append = existing_size > 0 and response.status == 206
            if existing_size and not append:
                logger.warning("Server did not resume the partial model; restarting download")
            mode = "ab" if append else "wb"
            with partial_path.open(mode) as model_file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    model_file.write(chunk)
    except (OSError, urllib.error.URLError) as exc:
        logger.error("Model download failed; partial data is retained for resume: %s", exc)
        raise RuntimeError(f"Could not download {destination.name}: {exc}") from exc

    actual_sha256 = sha256_file(partial_path)
    if actual_sha256.lower() != expected_sha256.lower():
        partial_path.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch for {destination.name}: expected {expected_sha256}, "
            f"received {actual_sha256}; the incomplete download was removed"
        )

    os.replace(partial_path, destination)
    logger.info("Downloaded and verified model: %s", destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the pinned local Silero VAD model.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models",
        help="Directory for local model files (default: ./models)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    destination = args.models_dir / MODEL_FILENAME
    if destination.is_file():
        actual_sha256 = sha256_file(destination)
        if actual_sha256.lower() == MODEL_SHA256:
            logger.info("Verified model already exists: %s", destination)
            return
        logger.warning("Existing model checksum is incorrect; replacing it")
        destination.unlink()

    download_file(MODEL_URL, destination, MODEL_SHA256)


if __name__ == "__main__":
    main()
