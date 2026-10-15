from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from pathlib import PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
VOSK_MODEL_NAME = "vosk-model-small-en-in-0.4"
MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/refs/heads/master/"
    "src/silero_vad/data/silero_vad.onnx"
)
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
MODEL_FILENAME = "silero_vad.onnx"
VOSK_MODEL_URL = f"https://alphacephei.com/vosk/models/{VOSK_MODEL_NAME}.zip"
VOSK_MODEL_SHA256 = "20663dcac4d5cb783a579c54d98339344a688e4ec6e1b4a4b059fd1235454cc7"
VOSK_ARCHIVE_FILENAME = f"{VOSK_MODEL_NAME}.zip"
WHISPER_MODEL_NAME = "base"
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


def extract_vosk_model(archive_path: Path, destination: Path) -> Path:
    """Safely extract the expected Vosk model and atomically install its directory."""

    if destination.exists():
        raise FileExistsError(
            f"Refusing to replace existing Vosk model directory: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{VOSK_MODEL_NAME}-", dir=destination.parent)
    )
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_path = PurePosixPath(member.filename)
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or "\\" in member.filename
                    or "\x00" in member.filename
                    or (member_path.parts and ":" in member_path.parts[0])
                ):
                    raise ValueError(f"Unsafe path in Vosk model archive: {member.filename}")
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError(f"Symbolic link in Vosk model archive: {member.filename}")
            archive.extractall(staging)

        extracted_model = staging / VOSK_MODEL_NAME
        if not (extracted_model / "am" / "final.mdl").is_file():
            raise ValueError(
                f"Vosk model archive is missing {VOSK_MODEL_NAME}/am/final.mdl"
            )
        os.replace(extracted_model, destination)
        logger.info("Installed Vosk model: %s", destination)
        return destination
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def download_vosk_model(models_dir: Path) -> Path:
    """Download and install the checksum-pinned Indian English small Vosk model."""

    destination = models_dir / VOSK_MODEL_NAME
    if (destination / "am" / "final.mdl").is_file():
        logger.info("Vosk model already exists: %s", destination)
        return destination
    if destination.exists():
        raise ValueError(f"Vosk model directory is incomplete: {destination}")

    archive_path = models_dir / VOSK_ARCHIVE_FILENAME
    if archive_path.is_file() and sha256_file(archive_path).lower() != VOSK_MODEL_SHA256:
        logger.warning("Existing Vosk archive checksum is incorrect; replacing it")
        archive_path.unlink()
    if not archive_path.is_file():
        download_file(VOSK_MODEL_URL, archive_path, VOSK_MODEL_SHA256)
    else:
        logger.info("Verified Vosk model archive already exists: %s", archive_path)

    installed = extract_vosk_model(archive_path, destination)
    archive_path.unlink()
    return installed


def download_whisper_model() -> None:
    """Ensure the local faster-whisper confirmation model is present in its cache."""

    from faster_whisper import WhisperModel

    logger.info("Checking local faster-whisper %s confirmation model", WHISPER_MODEL_NAME)
    try:
        model = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cpu",
            compute_type="int8",
            local_files_only=True,
        )
        del model
    except Exception:
        logger.info("faster-whisper %s is not cached locally; downloading it", WHISPER_MODEL_NAME)
        try:
            model = WhisperModel(
                WHISPER_MODEL_NAME,
                device="cpu",
                compute_type="int8",
                local_files_only=False,
            )
            del model
        except Exception as exc:
            logger.error(
                "Could not prepare faster-whisper %s model: %s",
                WHISPER_MODEL_NAME,
                exc,
            )
            raise RuntimeError(
                f"Could not prepare faster-whisper {WHISPER_MODEL_NAME}; "
                "check model download access and available disk space"
            ) from exc
    logger.info("faster-whisper %s model is ready", WHISPER_MODEL_NAME)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the local audio models required by Chintu."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models",
        help="Directory for local model files (default: ./models)",
    )
    parser.add_argument(
        "--wake",
        action="store_true",
        help="Download the Indian English Vosk model and prepare faster-whisper base",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.models_dir.mkdir(parents=True, exist_ok=True)
    if args.wake:
        download_vosk_model(args.models_dir)
        download_whisper_model()
        (args.models_dir / "wake_models.ready").write_text(
            "Vosk Indian English small + faster-whisper base\n",
            encoding="utf-8",
        )
        return

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
