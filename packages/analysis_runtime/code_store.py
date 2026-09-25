"""Immutable Python source versions for one analysis run."""

import difflib
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .records import SAFE_ID, write_json_atomic


MAX_CODE_BYTES = 256 * 1024
MAX_READ_CHARS = 10_000
DIFF_PREVIEW_CHARS = 2_000


class CodeStore:
    """Keep source revisions separately from result artifacts and execution attempts."""

    def __init__(self, root: str | Path, run_id: str):
        if not isinstance(run_id, str) or not SAFE_ID.fullmatch(run_id):
            raise ValueError("Invalid run ID")
        self.root = Path(root).expanduser().resolve()
        self.run_id = run_id
        code_root = self.root / "code"
        if code_root.is_symlink():
            raise ValueError("Code directory cannot be a symlink")
        code_root.mkdir(parents=True, exist_ok=True)
        self.directory = code_root / run_id
        if self.directory.is_symlink():
            raise ValueError("Run code directory cannot be a symlink")
        self.directory.mkdir(exist_ok=True)

    def _paths(self, code_id: str) -> tuple[Path, Path]:
        if (self.directory.parent.is_symlink() or self.directory.is_symlink() or
                self.directory.resolve() != self.directory):
            raise ValueError("Code path escapes its run directory")
        if (not isinstance(code_id, str) or not code_id.startswith("code_") or
                len(code_id) != 37 or not all(c in "0123456789abcdef" for c in code_id[5:])):
            raise ValueError("Invalid code ID")
        return self.directory / f"{code_id}.py", self.directory / f"{code_id}.json"

    def _verified(self, code_id: str) -> tuple[dict, bytes, Path]:
        path, index = self._paths(code_id)
        if path.is_symlink() or index.is_symlink():
            raise ValueError("Code versions cannot be symlinks")
        with index.open(encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("code_id") != code_id or metadata.get("run_id") != self.run_id:
            raise ValueError("Code version belongs to another run")
        if path.resolve(strict=True).parent != self.directory.resolve(strict=True):
            raise ValueError("Code path escapes its run directory")
        if path.stat().st_size != metadata.get("size_bytes"):
            raise ValueError("Saved code has changed")
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != metadata.get("sha256"):
            raise ValueError("Saved code has changed")
        return metadata, source, path

    def get_path(self, code_id: str) -> Path:
        """Return a verified source path for a later executor."""
        return self._verified(code_id)[2]

    def _diff(self, parent_id: str | None, code_id: str, source: str) -> str:
        if parent_id is None:
            return ""
        _, parent, _ = self._verified(parent_id)
        return "".join(difflib.unified_diff(
            parent.decode("utf-8").splitlines(keepends=True),
            source.splitlines(keepends=True),
            fromfile=parent_id, tofile=code_id,
        ))

    def write_analysis(self, code: str, previous_version: str | None = None) -> dict:
        """Save a complete source file; a revision points to an existing version."""
        if not isinstance(code, str):
            raise TypeError("code must be Python source text")
        encoded = code.encode("utf-8")
        if not encoded or len(encoded) > MAX_CODE_BYTES:
            raise ValueError(f"code must contain 1..{MAX_CODE_BYTES} UTF-8 bytes")
        if previous_version is not None:
            self._verified(previous_version)

        code_id = f"code_{uuid4().hex}"
        path, index = self._paths(code_id)
        diff = self._diff(previous_version, code_id, code)
        metadata = {
            "code_id": code_id,
            "run_id": self.run_id,
            "previous_version": previous_version,
            "path": str(path),
            "relative_path": str(path.relative_to(self.root)),
            "size_bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with path.open("xb") as file:
                file.write(encoded)
                file.flush()
                os.fsync(file.fileno())
            write_json_atomic(index, metadata)
        except Exception:
            # A version is visible only after its metadata exists.
            if not index.exists():
                path.unlink(missing_ok=True)
            raise
        return {**metadata, "diff": diff[:DIFF_PREVIEW_CHARS],
                "diff_truncated": len(diff) > DIFF_PREVIEW_CHARS}

    def read_artifact(self, code_id: str, offset: int = 0, max_chars: int = 2000,
                      view: str = "code") -> dict:
        """Read a bounded slice of source or its exact unified diff to its parent."""
        if (type(offset) is not int or type(max_chars) is not int or
                offset < 0 or not 1 <= max_chars <= MAX_READ_CHARS):
            raise ValueError(f"offset must be nonnegative and max_chars 1..{MAX_READ_CHARS}")
        if view not in {"code", "diff"}:
            raise ValueError("view must be code or diff")
        metadata, source, _ = self._verified(code_id)
        content = (source.decode("utf-8") if view == "code" else
                   self._diff(metadata["previous_version"], code_id,
                              source.decode("utf-8")))
        chunk = content[offset:offset + max_chars]
        return {**metadata, "view": view, "content": chunk,
                "truncated": offset + len(chunk) < len(content),
                "next_offset": offset + len(chunk), "total_chars": len(content)}
