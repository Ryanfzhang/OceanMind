"""Native Windows AppContainer launcher for untrusted analysis code.

The task directory is granted write access. Python, OceanMind source, and the
selected datasets are readable. No network capability is granted. A one-process
Job Object terminates the worker and prevents it from spawning child processes.
"""

from __future__ import annotations

import ctypes
import json
import os
import stat
import subprocess
import sys
import tempfile
from ctypes import wintypes
from pathlib import Path
from uuid import uuid4


class WindowsSandboxUnavailable(RuntimeError):
    """The AppContainer policy could not be established or verified."""


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_CREATE_SUSPENDED = 0x00000004
_CREATE_NO_WINDOW = 0x08000000
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_SECURITY_CAPABILITIES_ATTRIBUTE = 0x00020009
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_WAIT_OBJECT_0 = 0
_WAIT_TIMEOUT = 258
_DRIVE_REMOTE = 4


class _SecurityCapabilities(ctypes.Structure):
    _fields_ = [("AppContainerSid", wintypes.LPVOID),
                ("Capabilities", wintypes.LPVOID),
                ("CapabilityCount", wintypes.DWORD),
                ("Reserved", wintypes.DWORD)]


class _StartupInfo(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR),
                ("lpDesktop", wintypes.LPWSTR), ("lpTitle", wintypes.LPWSTR),
                ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
                ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD),
                ("dwXCountChars", wintypes.DWORD), ("dwYCountChars", wintypes.DWORD),
                ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
                ("lpReserved2", wintypes.LPVOID), ("hStdInput", wintypes.HANDLE),
                ("hStdOutput", wintypes.HANDLE), ("hStdError", wintypes.HANDLE)]


class _StartupInfoEx(ctypes.Structure):
    _fields_ = [("StartupInfo", _StartupInfo), ("lpAttributeList", wintypes.LPVOID)]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]


class _JobBasicLimitInformation(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _JobExtendedLimitInformation(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _JobBasicLimitInformation),
                ("IoInfo", _IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


def _win_error() -> OSError:
    return ctypes.WinError(ctypes.get_last_error())


def _api():
    if os.name != "nt":
        raise WindowsSandboxUnavailable("AppContainer requires Windows")
    if (ctypes.sizeof(_StartupInfoEx), ctypes.sizeof(_SecurityCapabilities),
            ctypes.sizeof(_JobExtendedLimitInformation)) != (112, 24, 144):
        raise WindowsSandboxUnavailable("Unsupported Win32 structure layout")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    userenv = ctypes.WinDLL("userenv", use_last_error=True)
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel.InitializeProcThreadAttributeList.argtypes = [
        wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.c_size_t)]
    kernel.InitializeProcThreadAttributeList.restype = wintypes.BOOL
    kernel.UpdateProcThreadAttribute.argtypes = [
        wintypes.LPVOID, wintypes.DWORD, ctypes.c_size_t, wintypes.LPVOID,
        ctypes.c_size_t, wintypes.LPVOID, wintypes.LPVOID]
    kernel.UpdateProcThreadAttribute.restype = wintypes.BOOL
    kernel.DeleteProcThreadAttributeList.argtypes = [wintypes.LPVOID]
    kernel.CreateProcessW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.LPVOID, wintypes.LPVOID,
        wintypes.BOOL, wintypes.DWORD, wintypes.LPVOID, wintypes.LPCWSTR,
        ctypes.POINTER(_StartupInfoEx), ctypes.POINTER(_ProcessInformation)]
    kernel.CreateProcessW.restype = wintypes.BOOL
    kernel.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel.ResumeThread.restype = wintypes.DWORD
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateProcess.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [wintypes.LPVOID]
    kernel.LocalFree.restype = wintypes.LPVOID
    userenv.CreateAppContainerProfile.argtypes = [
        wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPVOID,
        wintypes.DWORD, ctypes.POINTER(wintypes.LPVOID)]
    userenv.CreateAppContainerProfile.restype = ctypes.c_long
    userenv.DeriveAppContainerSidFromAppContainerName.argtypes = [
        wintypes.LPCWSTR, ctypes.POINTER(wintypes.LPVOID)]
    userenv.DeriveAppContainerSidFromAppContainerName.restype = ctypes.c_long
    userenv.DeleteAppContainerProfile.argtypes = [wintypes.LPCWSTR]
    userenv.DeleteAppContainerProfile.restype = ctypes.c_long
    security.ConvertSidToStringSidW.argtypes = [
        wintypes.LPVOID, ctypes.POINTER(wintypes.LPWSTR)]
    security.ConvertSidToStringSidW.restype = wintypes.BOOL
    security.FreeSid.argtypes = [wintypes.LPVOID]
    security.FreeSid.restype = wintypes.LPVOID
    return kernel, userenv, security


def _checked_path(path: Path) -> Path:
    attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
        raise WindowsSandboxUnavailable(f"Reparse-point sandbox root: {path}")
    return path.resolve(strict=True)


def _is_network_path(path: Path) -> bool:
    # Detect UNC paths before any SMB stat or ACL operation, and mapped drives
    # before attempting to grant an AppContainer SID on the remote server.
    if str(path).startswith("\\\\"):
        return True
    if os.name != "nt" or not path.anchor:
        return False
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
    kernel.GetDriveTypeW.restype = wintypes.UINT
    return kernel.GetDriveTypeW(path.anchor) == _DRIVE_REMOTE


def _acl(path: Path, action: str, value: str) -> None:
    result = subprocess.run(
        [_icacls(), str(path), action, value, "/Q"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if result.returncode:
        raise WindowsSandboxUnavailable(
            f"Cannot apply AppContainer access to {path}: {result.stderr[-300:]}")


def _icacls() -> str:
    # ACL operations must not resolve an executable from the server's PATH.
    return str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "icacls.exe")


class WindowsProcess:
    def __init__(self, kernel, process_handle, job_handle):
        self.kernel = kernel
        self.process_handle = process_handle
        self.job_handle = job_handle

    def poll(self) -> int | None:
        state = self.kernel.WaitForSingleObject(self.process_handle, 0)
        if state == _WAIT_TIMEOUT:
            return None
        if state != _WAIT_OBJECT_0:
            raise _win_error()
        code = wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.process_handle, ctypes.byref(code)):
            raise _win_error()
        return int(code.value)

    def wait(self, seconds: float | None = None) -> int:
        millis = 0xFFFFFFFF if seconds is None else min(int(seconds * 1000), 0xFFFFFFFE)
        state = self.kernel.WaitForSingleObject(self.process_handle, millis)
        if state == _WAIT_TIMEOUT:
            raise TimeoutError("AppContainer worker timed out")
        if state != _WAIT_OBJECT_0:
            raise _win_error()
        return self.poll() or 0

    def close(self) -> None:
        if self.process_handle:
            if self.poll() is None:
                self.kernel.TerminateJobObject(self.job_handle, 1)
                self.kernel.WaitForSingleObject(self.process_handle, 5000)
            self.kernel.CloseHandle(self.process_handle)
            self.process_handle = None
        if self.job_handle:
            self.kernel.CloseHandle(self.job_handle)
            self.job_handle = None


class WindowsAppContainer:
    """One fresh AppContainer identity per query, shared across its attempts."""

    def __init__(self, root: Path, data_roots: list[Path], env: dict[str, str]):
        for path in data_roots:
            if _is_network_path(path):
                raise WindowsSandboxUnavailable(
                    f"The Windows AppContainer cannot authorize the network-share dataset {path}. "
                    "For a trusted local deployment, set "
                    "OCEANMIND_WINDOWS_ANALYSIS_MODE=unsandboxed in .env and restart OceanMind.")
        self.root = _checked_path(root)
        self.data_roots = [_checked_path(path) for path in data_roots]
        self.env = env
        self.kernel, self.userenv, self.security = _api()
        self.name = f"OceanMind_{uuid4().hex}"
        self.sid = wintypes.LPVOID()
        self.sid_string = ""
        self.acl_paths: list[Path] = []
        self.low_integrity_paths: list[Path] = []
        self.closed = False
        try:
            result = self.userenv.CreateAppContainerProfile(
                self.name, "OceanMind analysis", "Isolated ocean analysis",
                None, 0, ctypes.byref(self.sid))
            if result:
                raise WindowsSandboxUnavailable(f"CreateAppContainerProfile failed: {result:#x}")
            sid_text = wintypes.LPWSTR()
            if not self.security.ConvertSidToStringSidW(self.sid, ctypes.byref(sid_text)):
                raise _win_error()
            try:
                self.sid_string = sid_text.value
            finally:
                self.kernel.LocalFree(ctypes.cast(sid_text, wintypes.LPVOID))
            self._prepare_acl()
            self.verify()
        except BaseException:
            self.close()
            raise

    def _grant(self, path: Path, rights: str) -> None:
        target = _checked_path(path)
        _acl(target, "/grant", f"*{self.sid_string}:{rights}")
        self.acl_paths.append(target)

    def _prepare_acl(self) -> None:
        from .sandbox import _validate_data_roots, _validate_writable_root

        _validate_writable_root(self.root)
        _validate_data_roots(tuple(self.data_roots))
        if any(self.root.is_relative_to(path) for path in self.data_roots):
            raise WindowsSandboxUnavailable("Dataset root contains the task directory")
        runtime = {Path(sys.prefix), Path(sys.base_prefix), Path(sys.executable).parent}
        code = {_PROJECT_ROOT / "packages", _PROJECT_ROOT / "domain",
                _PROJECT_ROOT / "configs" / "dataset_config.yaml"}
        for path in sorted(runtime | code | set(self.data_roots), key=str):
            target = _checked_path(path)
            self._grant(target, "(OI)(CI)RX" if target.is_dir() else "R")
        self._grant(self.root, "(OI)(CI)M")
        # M includes DELETE_CHILD on the parent. Without this restriction the
        # worker could rename the protected code/logs directories themselves.
        _acl(self.root, "/deny", f"*{self.sid_string}:(DE,DC)")
        self._set_low_integrity(self.root)
        (self.root / "records" / "run").mkdir(parents=True, exist_ok=True)
        # The run and attempt records already exist before the sandbox starts.
        # A Low-IL worker must be able to update them and create descendants.
        for name in ("records", "artifacts", "tmp", ".mplconfig", "appdata"):
            path = self.root / name
            if path.exists():
                self._set_low_integrity(path, recursive=True)
        for name in ("code", "logs"):
            protected = self.root / name
            protected.mkdir(exist_ok=True)
            _acl(protected, "/deny",
                 f"*{self.sid_string}:(OI)(CI)(WD,AD,WEA,WA,DE,DC,WDAC,WO)")
            self.acl_paths.append(protected)

    def _set_low_integrity(self, path: Path, *, recursive: bool = False) -> None:
        command = [_icacls(), str(path), "/setintegritylevel", "(OI)(CI)L"]
        if recursive:
            command.extend(("/T", "/L"))
        # icacls can change some entries before failing on a later entry.
        self.low_integrity_paths.append(path)
        result = subprocess.run([*command, "/Q"], capture_output=True,
                                text=True, timeout=30, check=False)
        if result.returncode:
            raise WindowsSandboxUnavailable(
                f"Cannot set Low integrity on {path}: {result.stderr[-300:]}")

    def spawn(self, worker_args: list[str]) -> WindowsProcess:
        attrs_size = ctypes.c_size_t()
        self.kernel.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(attrs_size))
        if not attrs_size.value:
            raise _win_error()
        attrs = ctypes.create_string_buffer(attrs_size.value)
        if not self.kernel.InitializeProcThreadAttributeList(
                attrs, 1, 0, ctypes.byref(attrs_size)):
            raise _win_error()
        try:
            capabilities = _SecurityCapabilities(self.sid, None, 0, 0)
            if not self.kernel.UpdateProcThreadAttribute(
                    attrs, 0, _SECURITY_CAPABILITIES_ATTRIBUTE,
                    ctypes.byref(capabilities), ctypes.sizeof(capabilities), None, None):
                raise _win_error()
            info = _StartupInfoEx()
            info.StartupInfo.cb = ctypes.sizeof(info)
            info.lpAttributeList = ctypes.cast(attrs, wintypes.LPVOID)
            process = _ProcessInformation()
            command = ctypes.create_unicode_buffer(subprocess.list2cmdline(
                [sys.executable, *worker_args]))
            variables = ctypes.create_unicode_buffer("\0".join(
                f"{key}={value}" for key, value in sorted(self.env.items())) + "\0\0")
            flags = (_EXTENDED_STARTUPINFO_PRESENT | _CREATE_SUSPENDED |
                     _CREATE_NO_WINDOW | _CREATE_UNICODE_ENVIRONMENT)
            if not self.kernel.CreateProcessW(
                    sys.executable, command, None, None, False, flags, variables,
                    str(self.root), ctypes.byref(info), ctypes.byref(process)):
                raise _win_error()
            job = None
            try:
                job = self.kernel.CreateJobObjectW(None, None)
                if not job:
                    raise _win_error()
                limits = _JobExtendedLimitInformation()
                limits.BasicLimitInformation.LimitFlags = (
                    _JOB_OBJECT_LIMIT_ACTIVE_PROCESS | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
                limits.BasicLimitInformation.ActiveProcessLimit = 1
                if not self.kernel.SetInformationJobObject(
                        job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                        ctypes.byref(limits), ctypes.sizeof(limits)):
                    raise _win_error()
                if not self.kernel.AssignProcessToJobObject(job, process.hProcess):
                    raise _win_error()
                if self.kernel.ResumeThread(process.hThread) == 0xFFFFFFFF:
                    raise _win_error()
                return WindowsProcess(self.kernel, process.hProcess, job)
            except BaseException:
                self.kernel.TerminateProcess(process.hProcess, 1)
                self.kernel.CloseHandle(process.hProcess)
                if job:
                    self.kernel.CloseHandle(job)
                raise
            finally:
                self.kernel.CloseHandle(process.hThread)
        finally:
            self.kernel.DeleteProcThreadAttributeList(attrs)

    def verify(self) -> None:
        """Run real import, access, and network checks before any analysis cell."""
        output = self.root / f".sandbox_probe_{uuid4().hex}.json"
        with tempfile.TemporaryDirectory(prefix="oceanmind_private_") as private:
            secret = Path(private) / "secret"
            secret.write_text("private", encoding="utf-8")
            code = r'''
import ctypes, json, os, socket, subprocess, sys
from pathlib import Path
import numpy, xarray
from packages.tool_loader.introspect import get_tools_cached
import packages.analysis_runtime.session_worker
root, secret, project_env, output, code_target, log_target = map(Path, sys.argv[1:7])
data = [Path(value) for value in sys.argv[7:]]
def check(action):
    try:
        action()
    except OSError as exc:
        return exc.errno or -1
    return None
token = ctypes.c_void_p()
is_container = ctypes.c_uint32()
size = ctypes.c_uint32()
advapi = ctypes.WinDLL('advapi32', use_last_error=True)
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetCurrentProcess.restype = ctypes.c_void_p
advapi.OpenProcessToken.argtypes = [
    ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
advapi.GetTokenInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                     ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
assert advapi.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token))
assert advapi.GetTokenInformation(token, 29, ctypes.byref(is_container), 4, ctypes.byref(size))
readable = [check(lambda path=path: path.open('rb').read(1) if path.is_file()
                  else next(path.iterdir(), None)) for path in data]
writeable = check(lambda: output.write_text('test', encoding='utf-8'))
nested = root / 'records' / 'run' / 'sandbox_probe'
nested_error = check(lambda: nested.write_text('test', encoding='utf-8'))
if nested_error is None:
    nested.unlink()
secret_error = check(lambda: secret.read_bytes())
project_env_error = check(lambda: project_env.read_bytes()) if project_env.exists() else -1
code_error = check(lambda: code_target.write_text('bad', encoding='utf-8'))
log_error = check(lambda: log_target.write_text('bad', encoding='utf-8'))
rename_error = check(lambda: (root / 'code').rename(root / 'renamed_code'))
def connect():
    with socket.socket() as sock:
        sock.settimeout(1)
        sock.connect(('1.1.1.1', 53))
network_error = check(connect)
spawn_error = check(lambda: subprocess.run([sys.executable, '-c', 'pass'], timeout=2))
output.write_text(json.dumps(dict(appcontainer=is_container.value == 1,
    data_read=all(error is None for error in readable),
    task_write=writeable is None, nested_write=nested_error is None,
    secret_denied=secret_error is not None,
    project_env_denied=project_env_error is not None,
    code_denied=code_error is not None,
    log_denied=log_error is not None, rename_denied=rename_error is not None,
    network_denied=network_error is not None,
    spawn_denied=spawn_error is not None, tool_count=len(get_tools_cached()))), encoding='utf-8')
'''
            readable = self.data_roots or [self.root]
            code_target = self.root / "code" / f"probe_{uuid4().hex}"
            log_target = self.root / "logs" / f"probe_{uuid4().hex}"
            process = self.spawn(["-c", code, str(self.root), str(secret),
                                  str(_PROJECT_ROOT / ".env"), str(output),
                                  str(code_target), str(log_target),
                                  *(str(path) for path in readable)])
            try:
                exit_code = process.wait(60)
            except TimeoutError as exc:
                raise WindowsSandboxUnavailable("AppContainer probe timed out") from exc
            finally:
                process.close()
            try:
                result = json.loads(output.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise WindowsSandboxUnavailable(
                    f"AppContainer probe exited {exit_code} without results") from exc
            finally:
                output.unlink(missing_ok=True)
                code_target.unlink(missing_ok=True)
                log_target.unlink(missing_ok=True)
            if exit_code or not all(result.get(key) is True for key in (
                    "appcontainer", "data_read", "task_write", "nested_write",
                    "secret_denied", "project_env_denied", "code_denied",
                    "log_denied", "rename_denied",
                    "network_denied", "spawn_denied")) or \
                    result.get("tool_count", 0) < 1:
                raise WindowsSandboxUnavailable(f"AppContainer probe failed: {result}")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.sid_string:
            for path in reversed(self.acl_paths):
                if path.exists():
                    subprocess.run([_icacls(), str(path), "/remove",
                                    f"*{self.sid_string}", "/Q"],
                                   capture_output=True, timeout=30, check=False)
        # Files created by the worker inherit Low IL. Restore the whole task
        # tree, including code, logs, and IPC, before deleting the profile.
        if self.low_integrity_paths and self.root.exists():
            subprocess.run([_icacls(), str(self.root), "/setintegritylevel",
                            "(OI)(CI)M", "/T", "/L", "/Q"],
                           capture_output=True, timeout=30, check=False)
        if self.sid:
            self.security.FreeSid(self.sid)
            self.sid = wintypes.LPVOID()
        if hasattr(self, "userenv"):
            self.userenv.DeleteAppContainerProfile(self.name)


__all__ = ["WindowsAppContainer", "WindowsSandboxUnavailable", "WindowsProcess"]
