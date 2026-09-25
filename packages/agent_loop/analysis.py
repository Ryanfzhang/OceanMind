"""Run-scoped analysis actions exposed to one LangGraph agent."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from packages.analysis_runtime.artifacts import ArtifactStore
from packages.analysis_runtime.code_store import CodeStore
from packages.analysis_runtime.executor import run_script
from packages.analysis_runtime.index import list_results
from packages.analysis_runtime.records import RunRecords, write_json_atomic
from packages.analysis_runtime.sandbox import build_sandbox_command, validate_data_roots


WRITE_ANALYSIS_SCHEMA = {
    "type": "function", "function": {
        "name": "write_analysis",
        "description": "Save a complete ordinary Python analysis script as a new immutable version.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"},
            "previous_version": {"type": "string"},
        }, "required": ["code"], "additionalProperties": False},
    },
}
READ_ARTIFACT_SCHEMA = {
    "type": "function", "function": {
        "name": "read_artifact",
        "description": "Read a bounded excerpt of saved code, its diff, or a result artifact.",
        "parameters": {"type": "object", "properties": {
            "ref": {"type": "string"},
            "offset": {"type": "integer", "minimum": 0},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": 10000},
            "view": {"type": "string", "enum": ["content", "diff"]},
        }, "required": ["ref"], "additionalProperties": False},
    },
}
RUN_ANALYSIS_SCHEMA = {
    "type": "function", "function": {
        "name": "run_analysis",
        "description": "Run one saved script version in a checked process sandbox; return stage and result references.",
        "parameters": {"type": "object", "properties": {
            "code_id": {"type": "string"},
            "timeout_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 600},
        }, "required": ["code_id"], "additionalProperties": False},
    },
}
LIST_RESULTS_SCHEMA = {
    "type": "function", "function": {
        "name": "list_results",
        "description": "Page every saved call result in an analysis attempt by stage, time, and depth.",
        "parameters": {"type": "object", "properties": {
            "attempt_id": {"type": "string"},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        }, "required": ["attempt_id"], "additionalProperties": False},
    },
}


class AnalysisSession:
    """Own one task directory; scripts may only write inside this directory."""

    def __init__(
        self, workspace: str | Path, *, data_roots: Iterable[str | Path] = (),
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        base = Path(workspace).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        self.root = base / f"task_{uuid4().hex}"
        self.root.mkdir(mode=0o700)
        (self.root / "logs").mkdir()
        self.records = RunRecords(self.root)
        self.run_id = self.records.run_id
        self.codes = CodeStore(self.root, self.run_id)
        self.data_roots = tuple(Path(path).expanduser().resolve(strict=True) for path in data_roots)
        validate_data_roots(self.data_roots)
        if any(path.is_relative_to(self.root) for path in self.data_roots):
            raise ValueError("Data sources must be outside the writable task directory")
        self.artifacts = ArtifactStore(self.root, allowed_source_roots=self.data_roots)
        self.on_event = on_event

    @classmethod
    def open_existing(
        cls, root: str | Path, run_id: str, *,
        data_roots: Iterable[str | Path] = (),
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> "AnalysisSession":
        """Resume a verified run without creating a second task or run record."""
        task_root = Path(root).expanduser()
        if task_root.is_symlink() or not task_root.is_dir():
            raise ValueError("Existing task directory is unavailable")
        task_root = task_root.resolve(strict=True)
        logs = task_root / "logs"
        if logs.is_symlink() or not logs.is_dir():
            raise ValueError("Existing task log directory is unavailable")
        roots = tuple(Path(path).expanduser().resolve(strict=True) for path in data_roots)
        validate_data_roots(roots)
        if any(path.is_relative_to(task_root) for path in roots):
            raise ValueError("Data sources must be outside the writable task directory")
        session = cls.__new__(cls)
        session.root = task_root
        session.records = RunRecords(task_root, run_id)
        session.run_id = session.records.run_id
        session.codes = CodeStore(task_root, session.run_id)
        session.data_roots = roots
        session.artifacts = ArtifactStore(task_root, allowed_source_roots=roots)
        session.on_event = on_event
        return session

    def write_analysis(self, code: str, previous_version: str | None = None) -> dict:
        return self.codes.write_analysis(code, previous_version)

    def read_artifact(
        self, ref: str, offset: int = 0, max_chars: int = 2000,
        view: str = "content",
    ) -> dict:
        if view not in {"content", "diff"}:
            raise ValueError("view must be content or diff")
        if ref.startswith("log_"):
            if view != "content" or offset < 0 or not 1 <= max_chars <= 10000:
                raise ValueError("Log reads require content view and bounded offsets")
            attempt_id = ref[4:]
            attempt = self.records.read("attempt", attempt_id)
            if attempt["run_id"] != self.run_id:
                raise ValueError("Log belongs to another run")
            with (self.root / "logs" / f"{attempt_id}.json").open(encoding="utf-8") as file:
                content = json.dumps(json.load(file), ensure_ascii=False)
            return {"ref": ref, "status": "completed", "content": content[offset:offset + max_chars],
                    "truncated": offset + max_chars < len(content),
                    "next_offset": min(offset + max_chars, len(content))}
        if ref.startswith("code_"):
            return self.codes.read_artifact(
                ref, offset=offset, max_chars=max_chars,
                view="diff" if view == "diff" else "code",
            )
        if view != "content":
            raise ValueError("Diff view is only available for code versions")
        metadata = self.artifacts.read_artifact(ref)
        if metadata["status"] != "missing" and metadata.get("run_id") != self.run_id:
            raise ValueError("Artifact belongs to another run")
        if metadata["status"] != "missing":
            if metadata.get("artifact_id") != ref:
                raise ValueError("Artifact index has the wrong ID")
            attempt = self.records.read("attempt", metadata["attempt_id"])
            if attempt["run_id"] != self.run_id:
                raise ValueError("Artifact attempt belongs to another run")
        result = self.artifacts.read_artifact(
            ref, include_content=True, offset=offset, max_chars=max_chars,
        )
        visible = {key: result[key] for key in (
            "artifact_id", "run_id", "attempt_id", "stage_id", "call_id", "name",
            "kind", "status", "summary", "inputs", "error", "content", "truncated",
        ) if key in result}
        if isinstance(visible.get("name"), str):
            visible["name"] = visible["name"][:120]
        if isinstance(visible.get("inputs"), list):
            visible["inputs"] = visible["inputs"][:20]
        if isinstance(visible.get("error"), str):
            visible["error"] = visible["error"][:500]
        if "summary" in visible and len(json.dumps(visible["summary"], ensure_ascii=False)) > 1200:
            visible["summary"] = {"truncated": True}
        return visible

    def list_results(self, attempt_id: str, offset: int = 0, limit: int = 10) -> dict:
        return list_results(self.root, self.run_id, attempt_id, offset=offset, limit=limit)

    def run_analysis(self, code_id: str, timeout_seconds: float = 60) -> dict:
        result = run_script(
            root=self.root, run_id=self.run_id, code_id=code_id,
            code_store=self.codes, launcher=build_sandbox_command,
            allowed_read_roots=self.data_roots, timeout_seconds=timeout_seconds,
            on_event=self.on_event, max_log_bytes=16_384,
        )
        attempt_id = result["attempt_id"]
        if (self.root / "logs").is_symlink() or not (self.root / "logs").is_dir():
            raise ValueError("Task log directory is unavailable")
        write_json_atomic(self.root / "logs" / f"{attempt_id}.json", {
            "run_id": self.run_id, "attempt_id": attempt_id,
            "stdout": result["stdout"], "stderr": result["stderr"],
            "stdout_truncated": result["stdout_truncated"],
            "stderr_truncated": result["stderr_truncated"],
        })
        visible = {**result, "log_ref": f"log_{attempt_id}"}
        for stream in ("stdout", "stderr"):
            visible[f"{stream}_excerpt_truncated"] = len(result[stream]) > 2000
            visible[stream] = result[stream][-2000:]
        return visible
