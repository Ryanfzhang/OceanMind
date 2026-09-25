"""Execute successive analysis cells in one sandboxed Python namespace."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType

from .artifacts import ArtifactStore
from .code_store import CodeStore
from .executor import _LogBuffer
from .records import RunRecords
from .stages import StageManager, stage
from .tools import AnalysisTools


class _BoundedTextLog:
    def __init__(self, limit: int) -> None:
        self.buffer = _LogBuffer(limit)

    def write(self, value: str) -> int:
        self.buffer.add(value.encode("utf-8", errors="replace"))
        return len(value)

    def flush(self) -> None:
        pass

    def result(self) -> tuple[str, bool]:
        return self.buffer.result()


def _send(fd: int, event: dict) -> None:
    packet = (json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    while packet:
        packet = packet[os.write(fd, packet):]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--event-fd", required=True, type=int)
    parser.add_argument("--max-log-bytes", required=True, type=int)
    args = parser.parse_args(argv)

    root = Path(args.root)
    records = RunRecords(root, run_id=args.run_id)
    codes = CodeStore(root, args.run_id)
    artifacts = ArtifactStore(root)
    runtime = ModuleType("oceanmind_runtime")
    sys.modules[runtime.__name__] = runtime
    namespace: dict = {"__name__": "__main__", "__package__": None}
    try:
        for line in sys.stdin:
            command = json.loads(line)
            if command.get("shutdown"):
                break
            attempt_id, code_id = command["attempt_id"], command["code_id"]
            stdout = _BoundedTextLog(args.max_log_bytes)
            stderr = _BoundedTextLog(args.max_log_bytes)
            status = "completed"
            try:
                attempt = records.read("attempt", attempt_id)
                if attempt["run_id"] != args.run_id or attempt["code_version"] != code_id:
                    raise ValueError("Attempt does not own this code version")
                script = codes.get_path(code_id)
                stages = StageManager(
                    args.run_id, attempt_id, records=records, retain_events=False,
                    callback=lambda event: _send(args.event_fd, event),
                )
                tools = AnalysisTools(records, artifacts, stages)
                runtime.stage, runtime.tools = stage, tools
                runtime.publish, runtime.load_result = tools.publish, tools.load_result
                namespace.update(__file__=str(script), stage=stage, tools=tools,
                                 publish=tools.publish, load_result=tools.load_result)
                with redirect_stdout(stdout), redirect_stderr(stderr), stages.activate():
                    exec(compile(script.read_bytes(), str(script), "exec"), namespace)
            except BaseException as exc:
                status = "failed"
                traceback.print_exception(exc, file=stderr)
                records.update("attempt", attempt_id, status=status,
                               error=f"{type(exc).__name__}: {exc}"[:500])
            else:
                records.update("attempt", attempt_id, status=status)
            out, out_truncated = stdout.result()
            err, err_truncated = stderr.result()
            _send(args.event_fd, {
                "type": "attempt_finished", "attempt_id": attempt_id,
                "exit_code": 0 if status == "completed" else 1,
                "stdout": out, "stderr": err,
                "stdout_truncated": out_truncated,
                "stderr_truncated": err_truncated,
            })
    finally:
        sys.modules.pop(runtime.__name__, None)
        os.close(args.event_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
