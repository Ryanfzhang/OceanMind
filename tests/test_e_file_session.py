"""File event transport used by the native Windows analysis worker."""

import json
import subprocess
import sys
import time
from pathlib import Path

from packages.analysis_runtime.code_store import CodeStore
from packages.analysis_runtime.records import RunRecords


def test_file_transport_preserves_namespace_and_events(tmp_path: Path) -> None:
    records = RunRecords(tmp_path)
    codes = CodeStore(tmp_path, records.run_id)
    first = codes.write_analysis("value = 7\n")["code_id"]
    second = codes.write_analysis('''from oceanmind_runtime import stage, publish
with stage("Reuse value"):
    publish("answer", {"value": value + 1}, inputs=[])
''')["code_id"]
    commands = tmp_path / "commands.jsonl"
    events = tmp_path / "events.jsonl"
    commands.touch()
    events.touch()
    repo = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "packages.analysis_runtime.session_worker",
         "--root", str(tmp_path), "--run-id", records.run_id,
         "--command-file", str(commands), "--event-file", str(events),
         "--max-log-bytes", "4096"],
        cwd=repo, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        for code_id in (first, second):
            attempt = records.new_attempt(code_version=code_id)
            with commands.open("a", encoding="utf-8") as file:
                file.write(json.dumps({"attempt_id": attempt, "code_id": code_id}) + "\n")
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                lines = events.read_text(encoding="utf-8").splitlines()
                if any(json.loads(line).get("type") == "attempt_finished" and
                       json.loads(line).get("attempt_id") == attempt for line in lines):
                    break
                time.sleep(0.05)
            else:
                raise AssertionError(f"Worker did not complete: {process.poll()}")
            assert records.read("attempt", attempt)["status"] == "completed"
        assert any(json.loads(line).get("type") == "step_completed"
                   for line in events.read_text(encoding="utf-8").splitlines())
    finally:
        with commands.open("a", encoding="utf-8") as file:
            file.write('{"shutdown":true}\n')
        process.wait(timeout=10)
        process.stderr.close()
