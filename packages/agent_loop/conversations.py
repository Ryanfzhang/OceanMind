"""Small durable conversation index for the LangGraph API."""

from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from packages.agent_loop.analysis import AnalysisSession
from packages.analysis_runtime.records import RunRecords, SAFE_ID, write_json_atomic


CONVERSATION_ID = re.compile(r"^conv_[0-9a-f]{32}$")
TASK_NAME = re.compile(r"^task_[0-9a-f]{32}$")
MAX_CONVERSATION_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class ConversationRecord:
    conversation_id: str
    run_id: str
    run_root: Path
    data_roots: tuple[Path, ...]
    messages: list[dict[str, Any]]
    status: str


class ConversationStore:
    """Map an opaque bearer ID to exactly one analysis run and message history."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.directory = self.workspace / "conversations"
        if self.directory.is_symlink():
            raise ValueError("Conversation index cannot be a symlink")
        self.directory.mkdir(exist_ok=True)
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def _path(self, conversation_id: str) -> Path:
        if not isinstance(conversation_id, str) or not CONVERSATION_ID.fullmatch(conversation_id):
            raise ValueError("Invalid conversation ID")
        if self.directory.is_symlink():
            raise ValueError("Conversation index cannot be a symlink")
        path = self.directory / f"{conversation_id}.json"
        if path.is_symlink():
            raise ValueError("Conversation record cannot be a symlink")
        return path

    @contextmanager
    def locked(self, conversation_id: str) -> Iterator[None]:
        self._path(conversation_id)
        with self._guard:
            lock = self._locks.setdefault(conversation_id, threading.Lock())
        with lock:
            yield

    def create(self, *, data_roots: tuple[str | Path, ...] = ()) -> tuple[ConversationRecord, AnalysisSession]:
        session = AnalysisSession(self.workspace, data_roots=data_roots)
        record = ConversationRecord(
            conversation_id=f"conv_{uuid4().hex}",
            run_id=session.run_id,
            run_root=session.root,
            data_roots=session.data_roots,
            messages=[],
            status="running",
        )
        self.save(record)
        return record, session

    def save(self, record: ConversationRecord) -> None:
        path = self._path(record.conversation_id)
        if record.run_root.parent != self.workspace or not TASK_NAME.fullmatch(record.run_root.name):
            raise ValueError("Conversation task is outside the workspace")
        if not SAFE_ID.fullmatch(record.run_id):
            raise ValueError("Invalid conversation run ID")
        RunRecords(record.run_root, record.run_id)
        payload = {
            "conversation_id": record.conversation_id,
            "run_id": record.run_id,
            "task_name": record.run_root.name,
            "data_roots": [str(path) for path in record.data_roots],
            "messages": record.messages,
            "status": record.status,
        }
        if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > MAX_CONVERSATION_BYTES:
            raise ValueError("Conversation history is too large")
        write_json_atomic(path, payload)

    def resolve(self, conversation_id: str) -> ConversationRecord:
        path = self._path(conversation_id)
        if path.stat().st_size > MAX_CONVERSATION_BYTES:
            raise ValueError("Conversation record is too large")
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
        if payload.get("conversation_id") != conversation_id:
            raise ValueError("Conversation ID does not match its record")
        task_name = payload.get("task_name")
        run_id = payload.get("run_id")
        if not isinstance(task_name, str) or not TASK_NAME.fullmatch(task_name):
            raise ValueError("Invalid conversation task")
        if not isinstance(run_id, str) or not SAFE_ID.fullmatch(run_id):
            raise ValueError("Invalid conversation run ID")
        task_root = self.workspace / task_name
        if task_root.is_symlink() or not task_root.is_dir() or task_root.resolve() != task_root:
            raise ValueError("Conversation task is unavailable")
        RunRecords(task_root, run_id)
        roots = payload.get("data_roots", [])
        if not isinstance(roots, list) or not all(isinstance(item, str) for item in roots):
            raise ValueError("Invalid conversation data roots")
        messages = payload.get("messages", [])
        if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
            raise ValueError("Invalid conversation messages")
        status = payload.get("status")
        if status not in {"running", "completed", "needs_input", "incomplete", "failed"}:
            raise ValueError("Invalid conversation status")
        return ConversationRecord(
            conversation_id=conversation_id,
            run_id=run_id,
            run_root=task_root,
            data_roots=tuple(Path(item) for item in roots),
            messages=messages,
            status=status,
        )

    def open_analysis(
        self, record: ConversationRecord, *,
        data_roots: tuple[str | Path, ...] | None = None,
    ) -> AnalysisSession:
        current = tuple(Path(path).expanduser().resolve(strict=True) for path in (
            record.data_roots if data_roots is None else data_roots
        ))
        if current != record.data_roots:
            raise ValueError("Configured data roots changed for this conversation")
        return AnalysisSession.open_existing(
            record.run_root, record.run_id, data_roots=current,
        )
