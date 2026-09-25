"""Run one saved analysis version in a bounded child process."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Sequence

from .artifacts import ArtifactStore, _stored_payload
from .code_store import CodeStore
from .records import RunRecords


EventCallback = Callable[[dict], None]
Launcher = Callable[[str, list[str], Path, Sequence[Path]], list[str]]
PREVIEW_LIMIT = 5
MAX_EVENT_LINE = 128 * 1024


class _LogBuffer:
    def __init__(self, limit: int):
        self.limit = limit
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def add(self, chunk: bytes) -> None:
        self.total += len(chunk)
        head_limit = self.limit // 2
        if len(self.head) < head_limit:
            taken = min(len(chunk), head_limit - len(self.head))
            self.head.extend(chunk[:taken])
            chunk = chunk[taken:]
        if chunk:
            self.tail.extend(chunk)
            if len(self.tail) > self.limit - head_limit:
                del self.tail[:len(self.tail) - (self.limit - head_limit)]

    def result(self) -> tuple[str, bool]:
        if self.total <= self.limit:
            raw = self.head + self.tail
        else:
            raw = self.head + b"\n...[log truncated]...\n" + self.tail
        return raw.decode("utf-8", errors="replace"), self.total > self.limit


def _read_log(pipe, output: _LogBuffer) -> None:
    try:
        while chunk := os.read(pipe.fileno(), 8192):
            output.add(chunk)
    finally:
        pipe.close()


def _read_events(fd: int, callback: EventCallback | None, counts: dict) -> None:
    pending = bytearray()
    try:
        while chunk := os.read(fd, 8192):
            pending.extend(chunk)
            while b"\n" in pending:
                line, _, remainder = pending.partition(b"\n")
                pending = bytearray(remainder)
                if len(line) > MAX_EVENT_LINE:
                    counts["invalid"] += 1
                    continue
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                        raise ValueError("Invalid event")
                except (ValueError, UnicodeDecodeError):
                    counts["invalid"] += 1
                    continue
                counts["received"] += 1
                if callback:
                    try:
                        callback(event)
                    except Exception:
                        # A UI subscriber must not stop draining the worker pipe.
                        counts["callback_errors"] += 1
            if len(pending) > MAX_EVENT_LINE:
                pending.clear()
                counts["invalid"] += 1
    finally:
        if pending:
            counts["invalid"] += 1
        os.close(fd)


def _child_env(root: Path) -> dict[str, str]:
    """Explicit allowlist: no inherited API keys, proxy settings, or credentials."""
    for name in ("tmp", ".mplconfig"):
        (root / name).mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    env = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "HOME": str(root),
        "TMPDIR": str(root / "tmp"),
        "MPLCONFIGDIR": str(root / ".mplconfig"),
        "PYTHONPATH": str(repo),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    if sys.platform == "linux":
        env.update(OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4",
                   MKL_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4",
                   DASK_NUM_WORKERS="8")
    return env


def _kill_process_group(pid: int) -> None:
    """Stop descendants that may still hold event or log pipe descriptors."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _records_for(records: RunRecords, kind: str, attempt_id: str) -> list[dict]:
    directory = records.root / "records" / kind
    if not directory.exists():
        return []
    result = []
    for path in directory.glob(f"{attempt_id}_{kind}_*.json"):
        item = records.read(kind, path.stem)
        if (item.get(f"{kind}_id") != path.stem or item.get("run_id") != records.run_id
                or item.get("attempt_id") != attempt_id):
            raise ValueError(f"{kind} record has inconsistent ownership")
        result.append(item)
    return sorted(result, key=lambda item: item.get("created_at", ""))


def _close_unfinished(records: RunRecords, attempt_id: str, reason: str,
                      callback: EventCallback | None) -> None:
    def notify(event: dict) -> None:
        if callback:
            try:
                callback(event)
            except Exception:
                pass

    for call in _records_for(records, "call", attempt_id):
        if call["status"] == "running":
            records.update("call", call["call_id"], status="failed", error=reason)
            notify({"type": "call_failed", "stage_id": call["stage_id"],
                    "call_id": call["call_id"], "error": reason})
    for stage in _records_for(records, "stage", attempt_id):
        if stage["status"] == "running":
            records.update("stage", stage["stage_id"], status="failed", error=reason)
            notify({"type": "step_failed", "step_id": stage["stage_id"],
                    "stage_id": stage["stage_id"], "title": stage["name"],
                    "completed_units": stage.get("completed_units", 0),
                          "error": reason})


def _unavailable_results(root: Path, run_id: str, attempt_id: str,
                         read_roots: Sequence[Path], entries: list[dict]) -> list[str]:
    """Check that completed references still point to readable artifact files."""
    artifacts = ArtifactStore(root, allowed_source_roots=read_roots)
    unavailable = []
    for entry in entries:
        artifact_id = entry.get("artifact_id")
        try:
            metadata = artifacts.read_artifact(artifact_id)
            if (metadata["status"] != "completed" or metadata.get("run_id") != run_id
                    or metadata.get("attempt_id") != attempt_id):
                raise ValueError("artifact is not completed")
            if metadata["kind"] in {"json", "dataarray_netcdf"}:
                _stored_payload(root, metadata["payload"])
            elif metadata["kind"] == "image_png":
                artifacts.read_image(artifact_id)
            elif metadata["kind"] == "source_netcdf":
                artifacts._source(metadata["source_path"])
            else:
                raise ValueError("unknown artifact format")
        except (KeyError, OSError, TypeError, ValueError):
            unavailable.append(str(artifact_id))
    return unavailable


def run_script(
    *, root: str | Path, run_id: str, code_id: str, launcher: Launcher,
    code_store: CodeStore | None = None,
    allowed_read_roots: Sequence[str | Path] = (),
    timeout_seconds: float = 60,
    on_event: EventCallback | None = None,
    max_log_bytes: int = 65_536,
) -> dict:
    """Run a verified source version; launcher must enforce process isolation.

    The caller supplies a sandbox command builder. No bare Python command is
    constructed as a fallback. Local tests inject an explicit launcher; the
    agent graph must provide the validated sandbox builder.
    """
    if not callable(launcher):
        raise ValueError("An isolation launcher is required")
    if not 0 < timeout_seconds <= 3600:
        raise ValueError("timeout_seconds must be within 0..3600")
    if not 1024 <= max_log_bytes <= 1_048_576:
        raise ValueError("max_log_bytes must be within 1024..1048576")
    root = Path(root).expanduser().resolve()
    records = RunRecords(root, run_id=run_id)
    store = code_store or CodeStore(root, run_id)
    if store.run_id != run_id or store.root != root:
        raise ValueError("Code store must belong to this run directory")
    script = store.get_path(code_id)  # validates path, ownership, and hash
    roots = [Path(path).expanduser().resolve() for path in allowed_read_roots]
    attempt_id = records.new_attempt(code_version=code_id)
    worker_args = ["-m", "packages.analysis_runtime.worker", "--root", str(root),
                   "--run-id", run_id, "--attempt-id", attempt_id,
                   "--script", str(script)]
    read_fd, write_fd = os.pipe()
    worker_args.extend(["--event-fd", str(write_fd)])
    stdout_log, stderr_log = _LogBuffer(max_log_bytes), _LogBuffer(max_log_bytes)
    event_counts = {"received": 0, "invalid": 0, "callback_errors": 0}
    process = None
    timed_out = False
    launch_error = None
    try:
        command = launcher(sys.executable, worker_args, root, roots)
        if not isinstance(command, list) or not command or not all(
                isinstance(item, str) for item in command):
            raise ValueError("Launcher must return a nonempty argv list")
        process = subprocess.Popen(command, cwd=root, env=_child_env(root),
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, pass_fds=(write_fd,),
                                   start_new_session=True)
    except Exception as exc:
        launch_error = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        os.close(write_fd)

    if process is None:
        os.close(read_fd)
        records.update("attempt", attempt_id, status="failed", error=launch_error)
        return {"run_id": run_id, "attempt_id": attempt_id, "code_id": code_id,
                "status": "failed", "exit_code": None, "timed_out": False,
                "error": launch_error, "stdout": "", "stderr": "",
                "stdout_truncated": False, "stderr_truncated": False,
                "event_count": 0, "stage_count": 0, "call_count": 0,
                "result_count": 0, "stages": [], "result_preview": [],
                "failed_call_preview": []}

    threads = [
        threading.Thread(target=_read_log, args=(process.stdout, stdout_log), daemon=True),
        threading.Thread(target=_read_log, args=(process.stderr, stderr_log), daemon=True),
        threading.Thread(target=_read_events, args=(read_fd, on_event, event_counts), daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(process.pid)
        process.wait()
    # A script can return while a subprocess is still alive. Drain pipes only
    # after stopping the entire group, otherwise inherited FDs keep them open.
    _kill_process_group(process.pid)
    for thread in threads:
        thread.join(timeout=5)

    attempt = records.read("attempt", attempt_id)
    if timed_out:
        status, reason = "timed_out", f"Script exceeded {timeout_seconds:g} seconds"
    elif process.returncode != 0:
        status, reason = "failed", attempt.get("error") or f"Worker exited {process.returncode}"
    elif (attempt.get("run_id") != run_id or attempt.get("attempt_id") != attempt_id
          or attempt.get("code_version") != code_id):
        status, reason = "failed", "Attempt record changed its run or code version"
    elif attempt["status"] != "completed":
        status, reason = "failed", "Worker exited before completing the attempt"
    else:
        status, reason = "completed", None
    try:
        store.get_path(code_id)
    except (OSError, ValueError) as exc:
        status, reason = "failed", f"Saved code changed during execution: {exc}"
    if status != "completed":
        _close_unfinished(records, attempt_id, reason, on_event)
        records.update("attempt", attempt_id, status=status, error=reason)

    stages = _records_for(records, "stage", attempt_id)
    calls = _records_for(records, "call", attempt_id)
    results = [entry for item in stages for entry in item.get("result_index", [])]
    successful = [entry for entry in results if entry["status"] == "completed"]
    unavailable = _unavailable_results(root, run_id, attempt_id, roots, successful)
    if unavailable:
        status, reason = "failed", f"Saved result artifacts unavailable: {len(unavailable)}"
        records.update("attempt", attempt_id, status=status, error=reason)
    failed = [call for call in calls if call["status"] == "failed"]
    failed_stages = [item for item in stages if item["status"] != "completed"]
    stage_preview = (list(reversed(failed_stages)) + [item for item in reversed(stages)
                                      if item["status"] == "completed"])[:PREVIEW_LIMIT]
    stdout, stdout_truncated = stdout_log.result()
    stderr, stderr_truncated = stderr_log.result()
    return {
        "run_id": run_id, "attempt_id": attempt_id, "code_id": code_id,
        "status": status, "exit_code": process.returncode, "timed_out": timed_out,
        "error": reason, "stdout": stdout, "stderr": stderr,
        "stdout_truncated": stdout_truncated, "stderr_truncated": stderr_truncated,
        "event_count": event_counts["received"],
        "invalid_event_count": event_counts["invalid"],
        "stage_count": len(stages), "call_count": len(calls),
        "result_count": len(successful),
        "invalid_result_count": len(unavailable),
        "invalid_result_preview": unavailable[:PREVIEW_LIMIT],
        "stages": [{"stage_id": item["stage_id"], "title": item["name"],
                    "status": item["status"],
                    "completed_units": item.get("completed_units", 0)}
                   for item in stage_preview],
        "result_preview": [{"artifact_id": entry["artifact_id"],
                            "call_id": entry["call_id"], "stage_id": entry["stage_id"],
                            "status": entry["status"], "summary": entry["summary"]}
                           for entry in successful[-PREVIEW_LIMIT:]],
        "failed_call_preview": [{"call_id": item["call_id"], "stage_id": item["stage_id"],
                                 "name": item["name"], "error": item.get("error", "")}
                                for item in failed[-PREVIEW_LIMIT:]],
    }
