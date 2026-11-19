import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from chintu.single_instance import acquire_single_instance


@pytest.mark.skipif(sys.platform != "win32", reason="Chintu targets Windows.")
def test_single_instance_guard_rejects_duplicate_and_releases_mutex() -> None:
    mutex_name = f"Local\\ChintuAI.Tests.{uuid4().hex}"
    guard = acquire_single_instance(mutex_name)

    assert guard is not None
    try:
        assert acquire_single_instance(mutex_name) is None
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from chintu.single_instance import acquire_single_instance; "
                    f"assert acquire_single_instance({mutex_name!r}) is None"
                ),
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
            text=True,
            timeout=10,
        )
        assert child.returncode == 0, child.stderr
    finally:
        guard.close()

    replacement_guard = acquire_single_instance(mutex_name)
    assert replacement_guard is not None
    replacement_guard.close()
