"""Child-process entry point for one saved analysis script."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import traceback
from pathlib import Path
from types import ModuleType

from .artifacts import ArtifactStore
from .code_store import CodeStore
from .records import RunRecords
from .stages import StageManager, stage
from .tools import AnalysisTools


MAX_EVENT_BYTES = 64 * 1024


def _send_event(fd: int, event: dict) -> None:
    payload = json.dumps(event, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(payload) > MAX_EVENT_BYTES:
        payload = b'{"type":"event_omitted","reason":"event too large"}'
    packet = payload + b"\n"
    while packet:
        packet = packet[os.write(fd, packet):]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--event-fd", required=True, type=int)
    args = parser.parse_args(argv)

    root = Path(args.root)
    records = RunRecords(root, run_id=args.run_id)
    attempt = records.read("attempt", args.attempt_id)
    if attempt["run_id"] != args.run_id:
        raise ValueError("Attempt belongs to another run")
    verified_script = CodeStore(root, args.run_id).get_path(attempt["code_version"])
    if verified_script != Path(args.script).resolve(strict=True):
        raise ValueError("Attempt code path does not match its saved version")
    stages = StageManager(args.run_id, args.attempt_id, records=records,
                          retain_events=False,
                          callback=lambda event: _send_event(args.event_fd, event))
    tools = AnalysisTools(records, ArtifactStore(root), stages)

    # The script-facing import is assembled for this attempt; no global runtime
    # singleton survives between scripts or child processes.
    module = ModuleType("oceanmind_runtime")
    module.stage = stage
    module.tools = tools
    module.publish = tools.publish
    module.load_result = tools.load_result
    sys.modules[module.__name__] = module
    try:
        with stages.activate():
            runpy.run_path(args.script, run_name="__main__", init_globals={
                "stage": stage, "tools": tools,
                "publish": tools.publish, "load_result": tools.load_result,
            })
    except BaseException as exc:
        traceback.print_exc()
        records.update("attempt", args.attempt_id, status="failed",
                       error=f"{type(exc).__name__}: {exc}"[:500])
        return 1
    finally:
        sys.modules.pop(module.__name__, None)
        os.close(args.event_fd)
    records.update("attempt", args.attempt_id, status="completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
