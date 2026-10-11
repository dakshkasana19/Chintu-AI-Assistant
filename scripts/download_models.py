from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True, parents=True)
    print(f"[models] Prepared model directory: {models_dir}")
    print("[models] No remote model downloads are required for Phase 0; install future model packages as needed.")


if __name__ == "__main__":
    main()
