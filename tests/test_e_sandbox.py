"""Real macOS Seatbelt checks; these tests must run outside another sandbox."""

import sys
import subprocess
from pathlib import Path

import pytest

from packages.analysis_runtime import sandbox


def test_sandbox_launcher_proves_access_controls(tmp_path: Path) -> None:
    if sys.platform != "darwin":
        pytest.skip("macOS Seatbelt test")
    data = tmp_path / "input.nc"
    data.write_bytes(b"netcdf test")
    run_root = tmp_path / "run"
    run_root.mkdir()

    # build_sandbox_command runs a real worker probe before returning argv. It
    # asserts allowed data read, task write, project/tool imports, and denied
    # unrelated file read, network connection, code write, and child process.
    worker_code = """
import errno, os
for name in ('code', 'logs'):
    try:
        os.rename(name, name + '_moved')
    except OSError as error:
        assert error.errno in (errno.EPERM, errno.EACCES)
    else:
        raise AssertionError(name + ' directory was renamed')
print('worker started')
"""
    command = sandbox.build_sandbox_command(
        sys.executable, ["-c", worker_code], run_root, [data]
    )
    assert command[:2] == ["/usr/bin/sandbox-exec", "-p"]
    assert command[-2:] == ["-c", worker_code]
    launched = subprocess.run(command, cwd=run_root, env={"PATH": "/usr/bin:/bin"},
                              capture_output=True, text=True, timeout=10, check=False)
    assert launched.returncode == 0, launched.stderr
    assert launched.stdout.strip() == "worker started"


def test_broad_data_or_workspace_root_is_rejected(tmp_path: Path) -> None:
    if sys.platform != "darwin":
        pytest.skip("macOS Seatbelt test")
    with pytest.raises(ValueError, match="Data read roots"):
        sandbox.build_sandbox_command(sys.executable, ["-c", "pass"], tmp_path, [Path.home()])
    with pytest.raises(ValueError, match="Writable root"):
        sandbox.build_sandbox_command(
            sys.executable, ["-c", "pass"], sandbox._PROJECT_ROOT, []
        )


def test_missing_sandbox_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_SANDBOX_EXEC", tmp_path / "missing")
    with pytest.raises(sandbox.SandboxUnavailableError, match="unavailable"):
        sandbox.build_sandbox_command(sys.executable, ["-c", "pass"], tmp_path, [])
