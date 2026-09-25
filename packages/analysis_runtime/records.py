"""IDs and minimal run, attempt, stage, and call records."""

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


KINDS = {"run", "attempt", "stage", "call", "artifact"}
SAFE_ID = re.compile(r"^[a-z0-9_]+$")
MAX_RECORD_BYTES = 8 * 1024 * 1024


def new_id(kind: str, *, run_id: str | None = None, attempt_id: str | None = None) -> str:
    if kind not in KINDS:
        raise ValueError(f"Unknown ID kind: {kind}")
    if kind == "run":
        prefix = ""
    elif kind == "attempt":
        if not run_id or not SAFE_ID.fullmatch(run_id):
            raise ValueError("attempt requires a valid run_id")
        prefix = f"{run_id}_"
    else:
        if not attempt_id or not SAFE_ID.fullmatch(attempt_id):
            raise ValueError(f"{kind} requires a valid attempt_id")
        prefix = f"{attempt_id}_"
    return f"{prefix}{kind}_{uuid4().hex}"


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file, ensure_ascii=False, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunRecords:
    """JSON records keyed by opaque IDs; values never contain large data."""

    def __init__(self, root: str | Path, run_id: str | None = None):
        self.root = Path(root)
        if run_id is None:
            self.run_id = new_id("run")
            self._create("run", self.run_id, {"status": "running", "created_at": _now()})
        else:
            self.run_id = run_id
            self.read("run", run_id)

    def _path(self, kind: str, record_id: str) -> Path:
        if kind not in KINDS or not SAFE_ID.fullmatch(record_id):
            raise ValueError("Invalid record kind or ID")
        records = self.root / "records"
        directory = records / kind
        path = directory / f"{record_id}.json"
        if records.is_symlink() or directory.is_symlink() or path.is_symlink():
            raise ValueError("Record path cannot be a symlink")
        return path

    def _create(self, kind: str, record_id: str, fields: dict) -> None:
        path = self._path(kind, record_id)
        if path.exists():
            raise FileExistsError(record_id)
        write_json_atomic(path, {f"{kind}_id": record_id, **fields})

    def read(self, kind: str, record_id: str) -> dict:
        path = self._path(kind, record_id)
        if path.stat().st_size > MAX_RECORD_BYTES:
            raise ValueError("Record is too large")
        with path.open(encoding="utf-8") as file:
            return json.load(file)

    def new_attempt(self, *, code_version: str | None = None) -> str:
        attempt_id = new_id("attempt", run_id=self.run_id)
        self._create("attempt", attempt_id, {
            "run_id": self.run_id, "code_version": code_version,
            "status": "running", "created_at": _now(),
        })
        return attempt_id

    def new_stage(self, attempt_id: str, name: str) -> str:
        self._check_attempt(attempt_id)
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ValueError("Stage name must contain 1..120 characters")
        stage_id = new_id("stage", attempt_id=attempt_id)
        self._create("stage", stage_id, {
            "run_id": self.run_id, "attempt_id": attempt_id, "name": name,
            "status": "running", "created_at": _now(),
        })
        return stage_id

    def new_call(self, attempt_id: str, stage_id: str, name: str,
                 *, inputs: list[str] | None = None, parameters: dict | None = None) -> str:
        self._check_attempt(attempt_id)
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ValueError("Call name must contain 1..120 characters")
        stage = self.read("stage", stage_id)
        if stage["attempt_id"] != attempt_id:
            raise ValueError("Call stage belongs to another attempt")
        call_id = new_id("call", attempt_id=attempt_id)
        self._create("call", call_id, {
            "run_id": self.run_id, "attempt_id": attempt_id, "stage_id": stage_id,
            "name": name, "inputs": inputs or [], "parameters": parameters or {},
            "artifact_ids": [], "status": "running", "created_at": _now(),
        })
        return call_id

    def update(self, kind: str, record_id: str, **fields) -> dict:
        record = self.read(kind, record_id)
        if any(key.endswith("_id") for key in fields):
            raise ValueError("Record ownership cannot change")
        record.update(fields)
        record["updated_at"] = _now()
        write_json_atomic(self._path(kind, record_id), record)
        return record

    def _check_attempt(self, attempt_id: str) -> None:
        if self.read("attempt", attempt_id)["run_id"] != self.run_id:
            raise ValueError("Attempt belongs to another run")
