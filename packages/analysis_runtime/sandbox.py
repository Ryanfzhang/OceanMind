"""Verified host isolation for analysis scripts.

The child may read its task directory, explicitly supplied data paths, the
OceanMind Python source, and the Python/system runtime. It may write only its
task directory. Network access is denied. The probe runs under the same policy
before a worker command is returned; Windows uses a verified AppContainer
session instead of a command wrapper. Unsupported hosts fail closed.

This is filesystem and network isolation, not a CPU/memory quota. The parent
executor must still impose timeouts, log limits, and a clean environment.
The script shares the worker's writable records and artifact index, so the
parent must validate metadata and artifact paths before trusting them.
"""

from __future__ import annotations

import errno
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4


class SandboxUnavailableError(RuntimeError):
    """A verified process sandbox cannot be established on this host."""


_SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
_BWRAP_EXEC = Path("/usr/bin/bwrap")
_PRLIMIT_EXEC = Path("/usr/bin/prlimit")
_LINUX_PROCESS_LIMIT = 64
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SYSTEM_READ_ROOTS = (Path("/System"), Path("/usr/lib"), Path("/private/var/db/dyld"))


def _homebrew_python_roots(base_prefix: Path) -> set[Path]:
    """Include Homebrew's bin symlink chain as well as its Framework tree."""
    parts = base_prefix.resolve(strict=True).parts
    if "Cellar" not in parts:
        return set()
    index = parts.index("Cellar")
    if index + 1 >= len(parts):
        return set()
    package = parts[index + 1]
    cellar = Path(*parts[:index + 2])
    opt = Path(*parts[:index]) / "opt" / package
    return {cellar, opt}


def _path(value: str | os.PathLike[str], *, directory: bool | None = None) -> Path:
    candidate = Path(value).expanduser().resolve(strict=True)
    if directory is True and not candidate.is_dir():
        raise ValueError(f"Expected a directory: {candidate}")
    if directory is False and not candidate.is_file():
        raise ValueError(f"Expected a file: {candidate}")
    return candidate


def _quote(path: Path) -> str:
    # JSON quoted strings are valid Seatbelt string literals. Reject control
    # characters rather than letting a path affect the policy grammar.
    value = str(path)
    if any(ord(char) < 32 for char in value):
        raise ValueError("Sandbox path contains a control character")
    return json.dumps(value, ensure_ascii=False)


def _rule(path: Path) -> str:
    return f'({"subpath" if path.is_dir() else "literal"} {_quote(path)})'


def _validate_data_roots(roots: tuple[Path, ...]) -> None:
    # A broad user/home/temp root would also expose unrelated files. Callers
    # should pass the specific dataset file or a narrowly scoped data folder.
    broad = {Path("/"), Path.home().resolve(), Path("/Users"), Path("/home"),
             Path("/private"), Path("/private/tmp"), Path("/var"), Path("/tmp"),
             Path("/import"), Path("/etc"), Path("/usr")}
    if os.name == "nt":
        broad.update(Path(root.anchor) for root in roots)
    if any(root in broad or _PROJECT_ROOT.is_relative_to(root) for root in roots):
        raise ValueError("Data read roots must be specific files or narrow directories")


def validate_data_roots(roots: tuple[Path, ...]) -> None:
    """Apply the same source-scope check before parent inspection and child launch."""
    _validate_data_roots(roots)


def _validate_writable_root(root: Path) -> None:
    if root in {Path("/"), Path.home().resolve(), Path("/private/tmp"), Path("/tmp")} or \
            _PROJECT_ROOT.is_relative_to(root):
        raise ValueError("Writable root must be a task directory, not a broad workspace root")


def _policy(python: Path, writable_root: Path, read_roots: tuple[Path, ...]) -> str:
    # The venv and base interpreter trees include extension modules, dylibs,
    # and site packages. They are trusted installation paths, never task data.
    runtime_roots = {Path(sys.prefix), Path(sys.base_prefix),
                     _path(sys.prefix, directory=True), _path(sys.base_prefix, directory=True),
                     python, python.resolve(strict=True)}
    runtime_roots.update(_homebrew_python_roots(Path(sys.base_prefix)))
    code_roots = {_path(_PROJECT_ROOT / "packages", directory=True),
                  _path(_PROJECT_ROOT / "domain", directory=True),
                  _path(_PROJECT_ROOT / "configs" / "dataset_config.yaml", directory=False)}
    reads = set(_SYSTEM_READ_ROOTS) | runtime_roots | code_roots | set(read_roots) | {writable_root}
    read_rules = " ".join(_rule(path) for path in sorted(reads, key=str) if path.exists())
    ancestors = {parent for path in reads for parent in path.parents}
    ancestor_rules = " ".join(f"(literal {_quote(path)})" for path in sorted(ancestors, key=str))
    return "\n".join((
        "(version 1)",
        "(deny default)",
        "(allow process-exec)",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        f"(allow file-read* {read_rules})",
        f"(allow file-read* {ancestor_rules})",
        f"(allow file-write* {_rule(writable_root)} (literal \"/dev/null\"))",
        f"(deny file-write* (subpath {_quote(writable_root / 'code')}))",
        f"(deny file-write* (subpath {_quote(writable_root / 'logs')}))",
        "(deny network*)",
    ))


def _configuration(
    python: str | os.PathLike[str], writable_root: str | os.PathLike[str],
    allowed_read_roots: tuple[str | os.PathLike[str], ...] | list[str | os.PathLike[str]],
) -> tuple[Path, Path, tuple[Path, ...], str]:
    if platform.system() != "Darwin" or not _SANDBOX_EXEC.is_file():
        raise SandboxUnavailableError("macOS sandbox-exec is unavailable")
    # Keep the venv symlink for execution: resolving it to the base binary
    # makes Python lose pyvenv.cfg and its installed site-packages.
    executable = Path(python).expanduser().absolute()
    if not executable.is_file():
        raise ValueError(f"Expected a Python executable: {executable}")
    root = _path(writable_root, directory=True)
    _validate_writable_root(root)
    data = tuple(_path(path) for path in allowed_read_roots)
    _validate_data_roots(data)
    return executable, root, data, _policy(executable, root, data)


def _linux_configuration(
    python: str | os.PathLike[str], writable_root: str | os.PathLike[str],
    allowed_read_roots: tuple[str | os.PathLike[str], ...] | list[str | os.PathLike[str]],
) -> tuple[Path, Path, tuple[Path, ...]]:
    if (platform.system() != "Linux" or not _BWRAP_EXEC.is_file()
            or not _PRLIMIT_EXEC.is_file()):
        raise SandboxUnavailableError("Linux Bubblewrap and prlimit are unavailable")
    executable = Path(python).expanduser().absolute()
    if not executable.is_file():
        raise ValueError(f"Expected a Python executable: {executable}")
    root = _path(writable_root, directory=True)
    _validate_writable_root(root)
    data = tuple(_path(path) for path in allowed_read_roots)
    _validate_data_roots(data)
    if any(root.is_relative_to(path) for path in data):
        raise ValueError("Data read root cannot contain the writable task directory")
    return executable, root, data


def _linux_command(
    executable: Path, root: Path, data: tuple[Path, ...], worker_args: list[str],
) -> list[str]:
    command = [str(_BWRAP_EXEC), "--unshare-user", "--unshare-pid", "--unshare-net",
               "--unshare-ipc", "--unshare-uts", "--die-with-parent", "--new-session",
               "--tmpfs", "/tmp"]
    mounts = {Path("/usr"), _path(sys.prefix, directory=True),
              _path(sys.base_prefix, directory=True),
              _path(_PROJECT_ROOT / "packages", directory=True),
              _path(_PROJECT_ROOT / "domain", directory=True),
              _path(_PROJECT_ROOT / "configs" / "dataset_config.yaml", directory=False),
              *data}
    resolved_executable = executable.resolve(strict=True)
    if not any(resolved_executable.is_relative_to(path) for path in mounts if path.is_dir()):
        mounts.add(resolved_executable)
    for path in sorted(mounts, key=lambda item: (len(item.parts), str(item))):
        command.extend(("--ro-bind", str(path), str(path)))
    for path in (Path("/bin"), Path("/lib"), Path("/lib64"), Path("/sbin")):
        if path.is_symlink():
            command.extend(("--symlink", os.readlink(path), str(path)))
        elif path.exists():
            command.extend(("--ro-bind", str(path), str(path)))
    for path in (Path("/etc/ld.so.cache"), Path("/etc/localtime")):
        if path.exists():
            command.extend(("--ro-bind", str(path), str(path)))
    aliases = {parent for path in (executable, Path(sys.prefix), Path(sys.base_prefix))
               for parent in path.parents if parent.is_symlink()}
    for path in sorted(aliases, key=lambda item: (len(item.parts), str(item))):
        command.extend(("--symlink", os.readlink(path), str(path)))
    command.extend(("--dev", "/dev", "--proc", "/proc",
                    "--bind", str(root), str(root)))
    for name in ("code", "logs"):
        protected = _path(root / name, directory=True)
        command.extend(("--ro-bind", str(protected), str(protected)))
    command.extend(("--chdir", str(root), str(_PRLIMIT_EXEC),
                    f"--nproc={_LINUX_PROCESS_LIMIT}:{_LINUX_PROCESS_LIMIT}",
                    "--", str(executable), *worker_args))
    return command


_PROBE = r'''
import errno, json, os, resource, socket, subprocess, sys
from pathlib import Path
import numpy, xarray
import packages.analysis_runtime.records
import packages.analysis_runtime.worker
import packages.analysis_runtime.session_worker
from packages.tool_loader.introspect import get_tools_cached
tool_count = len(get_tools_cached())
data, secret, output, code_target, log_target, escape_link = map(Path, sys.argv[1:7])
try:
    if data.is_dir():
        with os.scandir(data) as entries:
            next(entries, None)
    else:
        with data.open("rb") as file:
            file.read(1)
    data_read = True
except OSError:
    data_read = False
output.write_text("sandbox write works", encoding="utf-8")
try:
    secret.read_bytes()
    secret_errno = None
except OSError as error:
    secret_errno = error.errno
try:
    code_target.write_text("forbidden", encoding="utf-8")
    code_write_errno = None
except OSError as error:
    code_write_errno = error.errno
try:
    log_target.write_text("forbidden", encoding="utf-8")
    log_write_errno = None
except OSError as error:
    log_write_errno = error.errno
try:
    escape_link.write_text("forbidden", encoding="utf-8")
    symlink_write_errno = None
except OSError as error:
    symlink_write_errno = error.errno
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        network_errno = connection.connect_ex(("1.1.1.1", 53))
except OSError as error:
    network_errno = error.errno
try:
    subprocess.run(["/usr/bin/true"], check=False, timeout=2)
    spawn_errno = None
except OSError as error:
    spawn_errno = error.errno
print(json.dumps({"data_read": data_read, "secret_errno": secret_errno,
                  "code_write_errno": code_write_errno,
                  "log_write_errno": log_write_errno,
                  "symlink_write_errno": symlink_write_errno,
                  "network_errno": network_errno, "spawn_errno": spawn_errno,
                  "nproc_limit": resource.getrlimit(resource.RLIMIT_NPROC)[1],
                  "tool_count": tool_count}))
'''


def verify_sandbox_enforcement(
    python: str | os.PathLike[str], writable_root: str | os.PathLike[str],
    allowed_read_roots: tuple[str | os.PathLike[str], ...] | list[str | os.PathLike[str]],
) -> None:
    """Prove permitted reads/writes/imports and denied private read/network.

    Raises SandboxUnavailableError on any unexpected outcome, including an
    outer container forbidding sandbox_apply. No analysis code is then run.
    """
    if platform.system() == "Windows":
        from .executor import _child_env
        from .windows_appcontainer import WindowsAppContainer

        root = _path(writable_root, directory=True)
        data = [_path(path) for path in allowed_read_roots]
        sandbox = WindowsAppContainer(root, data, _child_env(root))
        sandbox.close()
        return
    linux = platform.system() == "Linux"
    if linux:
        executable, root, data = _linux_configuration(python, writable_root, allowed_read_roots)
    else:
        executable, root, data, profile = _configuration(python, writable_root, allowed_read_roots)
    with tempfile.TemporaryDirectory(prefix="oceanmind_sandbox_probe_") as temp:
        secret = Path(temp) / "unrelated_secret"
        secret.write_text("must not be readable", encoding="utf-8")
        if any(secret.is_relative_to(path) for path in (*data, root, *_SYSTEM_READ_ROOTS)):
            raise SandboxUnavailableError("Probe secret lies within an allowed read root")
        output = root / f".sandbox_probe_{uuid4().hex}"
        code_root = root / "code"
        if code_root.is_symlink():
            raise SandboxUnavailableError("Code directory cannot be a symlink")
        code_root.mkdir(exist_ok=True)
        code_target = code_root / f".sandbox_write_probe_{uuid4().hex}"
        log_root = root / "logs"
        if log_root.is_symlink():
            raise SandboxUnavailableError("Log directory cannot be a symlink")
        log_root.mkdir(exist_ok=True)
        log_target = log_root / f".sandbox_write_probe_{uuid4().hex}"
        escape_link = root / f".sandbox_escape_probe_{uuid4().hex}"
        escape_link.symlink_to(secret)
        # If no dataset is supplied, test read access to the task directory.
        readable = data[0] if data else root
        env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(_PROJECT_ROOT),
               "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(root), "HOME": str(root)}
        if linux:
            env.update(OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4",
                       MKL_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4",
                       DASK_NUM_WORKERS="8")
        probe_args = ["-c", _PROBE, str(readable), str(secret), str(output),
                      str(code_target), str(log_target), str(escape_link)]
        command = (_linux_command(executable, root, data, probe_args) if linux else
                   [str(_SANDBOX_EXEC), "-p", profile, str(executable), *probe_args])
        try:
            result = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                    text=True, timeout=15, check=False)
            if result.returncode != 0:
                raise SandboxUnavailableError(
                    f"Sandbox probe failed ({result.returncode}): "
                    f"stdout={result.stdout[-800:]} stderr={result.stderr[-800:]}")
            payload = json.loads(result.stdout.strip().splitlines()[-1])
            denied = {errno.EPERM, errno.EACCES}
            hidden = denied | ({errno.ENOENT} if linux else set())
            read_only = denied | ({errno.EROFS} if linux else set())
            no_network = denied | ({errno.ENETUNREACH, errno.EHOSTUNREACH} if linux else set())
            if (payload.get("data_read") is not True or
                    payload.get("secret_errno") not in hidden or
                    payload.get("code_write_errno") not in read_only or
                    payload.get("log_write_errno") not in read_only or
                    payload.get("symlink_write_errno") not in hidden or
                    payload.get("network_errno") not in no_network or
                    (payload.get("spawn_errno") is not None if linux else
                     payload.get("spawn_errno") not in denied) or
                    (linux and payload.get("nproc_limit") != _LINUX_PROCESS_LIMIT) or
                    not isinstance(payload.get("tool_count"), int) or
                    payload["tool_count"] < 1 or
                    not output.is_file()):
                raise SandboxUnavailableError(f"Sandbox enforcement probe failed: {payload}")
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as error:
            raise SandboxUnavailableError(f"Sandbox probe could not complete: {error}") from error
        finally:
            output.unlink(missing_ok=True)
            code_target.unlink(missing_ok=True)
            log_target.unlink(missing_ok=True)
            escape_link.unlink(missing_ok=True)


def build_sandbox_command(
    python: str | os.PathLike[str], worker_args: list[str],
    writable_root: str | os.PathLike[str],
    allowed_read_roots: tuple[str | os.PathLike[str], ...] | list[str | os.PathLike[str]],
) -> list[str]:
    """Return a verified host sandbox worker argv; never a bare Python argv."""
    if not isinstance(worker_args, list) or not all(isinstance(arg, str) for arg in worker_args):
        raise TypeError("worker_args must be a list of strings")
    if platform.system() == "Windows":
        raise SandboxUnavailableError(
            "Windows AppContainer sessions are launched by SessionRunner, not by an argv wrapper")
    if platform.system() == "Linux":
        executable, root, data = _linux_configuration(python, writable_root, allowed_read_roots)
        verify_sandbox_enforcement(executable, root, data)
        return _linux_command(executable, root, data, worker_args)
    executable, root, data, profile = _configuration(python, writable_root, allowed_read_roots)
    verify_sandbox_enforcement(executable, root, data)
    return [str(_SANDBOX_EXEC), "-p", profile, str(executable), *worker_args]


__all__ = [
    "SandboxUnavailableError", "build_sandbox_command", "verify_sandbox_enforcement",
    "validate_data_roots",
]
