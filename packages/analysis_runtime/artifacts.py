"""Durable, individually addressable results for analysis attempts."""

import hashlib
import json
import os
import tempfile
import base64
import math
import struct
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from .records import SAFE_ID, new_id, write_json_atomic


class SourceChangedError(RuntimeError):
    """A referenced source no longer has the recorded content."""


INLINE_ARRAY_BYTES = 64 * 1024
FULL_HASH_BYTES = 16 * 1024 * 1024
SAMPLE_BYTES = 1024 * 1024
MAX_ARTIFACT_INDEX_BYTES = 32 * 1024
MAX_IMAGE_BYTES = 16 * 1024 * 1024


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if (not isinstance(data, bytes) or len(data) < 24 or
            data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR"):
        raise ValueError("Image must contain PNG bytes")
    width, height = struct.unpack(">II", data[16:24])
    if not 1 <= width <= 8192 or not 1 <= height <= 8192:
        raise ValueError("PNG dimensions are outside supported limits")
    return width, height


def _stored_payload(root: Path, relative: str) -> Path:
    """Resolve an artifact payload without following it outside its data store."""
    if not isinstance(relative, str):
        raise ValueError("Invalid artifact payload path")
    path = Path(relative)
    if (path.is_absolute() or ".." in path.parts or
            path.parts[:2] != ("artifacts", "data")):
        raise ValueError("Invalid artifact payload path")
    if (root / "artifacts").is_symlink() or (root / "artifacts" / "data").is_symlink():
        raise ValueError("Artifact data store cannot be a symlink")
    data_root = root.resolve() / "artifacts" / "data"
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(data_root) or not resolved.is_file():
        raise ValueError("Artifact payload escapes its data store")
    return resolved


def _encoded(value: Any, *, root: Path | None = None,
             artifact_id: str | None = None, sidecars: list[Path] | None = None) -> Any:
    import numpy as np
    import xarray as xr

    if isinstance(value, xr.DataArray):
        if root is None or artifact_id is None or sidecars is None:
            raise ValueError("Nested DataArray requires artifact storage")
        path = root / "artifacts" / "data" / f"{artifact_id}_field_{len(sidecars)}.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".nc", delete=False) as file:
            temporary = Path(file.name)
        try:
            value.to_netcdf(temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        sidecars.append(path)
        return {"__dataarray_file__": str(path.relative_to(root))}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot be persisted")
        contiguous = np.ascontiguousarray(value)
        if value.nbytes > INLINE_ARRAY_BYTES:
            if root is None or artifact_id is None or sidecars is None:
                raise ValueError("Large arrays require artifact storage")
            path = root / "artifacts" / "data" / f"{artifact_id}_array_{len(sidecars)}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as file:
                temporary = Path(file.name)
                try:
                    np.save(file, contiguous, allow_pickle=False)
                    file.flush()
                    os.fsync(file.fileno())
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            sidecars.append(path)
            return {"__ndarray_file__": str(path.relative_to(root))}
        return {"__ndarray__": True, "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data_b64": base64.b64encode(contiguous.tobytes()).decode("ascii")}
    if isinstance(value, np.generic):
        return _encoded(value.item(), root=root, artifact_id=artifact_id, sidecars=sidecars)
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite__": repr(value)}
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, slice):
        return {"__slice__": [_encoded(item, root=root, artifact_id=artifact_id,
                                      sidecars=sidecars)
                              for item in (value.start, value.stop, value.step)]}
    if isinstance(value, dict):
        return {str(key): _encoded(item, root=root, artifact_id=artifact_id,
                                   sidecars=sidecars) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encoded(item, root=root, artifact_id=artifact_id,
                         sidecars=sidecars) for item in value]
    return value


def _decoded(value: Any, *, root: Path | None = None) -> Any:
    if isinstance(value, dict) and set(value) == {"__dataarray_file__"}:
        import xarray as xr

        if root is None:
            raise ValueError("DataArray sidecar requires artifact storage")
        with xr.open_dataarray(_stored_payload(root, value["__dataarray_file__"])) as array:
            return array.load()
    if isinstance(value, dict) and set(value) == {"__ndarray_file__"}:
        import numpy as np

        if root is None:
            raise ValueError("Array sidecar requires artifact storage")
        return np.load(_stored_payload(root, value["__ndarray_file__"]), allow_pickle=False)
    if (isinstance(value, dict) and value.get("__ndarray__") is True and
            set(value) == {"__ndarray__", "dtype", "shape", "data_b64"}):
        import numpy as np

        raw = base64.b64decode(value["data_b64"])
        return np.frombuffer(raw, dtype=value["dtype"]).copy().reshape(value["shape"])
    if isinstance(value, dict) and set(value) == {"__nonfinite__"}:
        return float(value["__nonfinite__"])
    if isinstance(value, dict) and set(value) == {"__slice__"}:
        return slice(*(_decoded(item, root=root) for item in value["__slice__"]))
    if isinstance(value, dict):
        return {key: _decoded(item, root=root) for key, item in value.items()}
    if isinstance(value, list):
        return [_decoded(item, root=root) for item in value]
    return value


def _version(path: Path) -> dict:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as file:
        if stat.st_size <= FULL_HASH_BYTES:
            for chunk in iter(lambda: file.read(SAMPLE_BYTES), b""):
                digest.update(chunk)
            method = "sha256_full"
        else:
            for offset in (0, max(0, stat.st_size // 2 - SAMPLE_BYTES // 2),
                           stat.st_size - SAMPLE_BYTES):
                file.seek(offset)
                digest.update(file.read(SAMPLE_BYTES))
            method = "sha256_three_samples"
    after = path.stat()
    if (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_dev, after.st_ino) != (
            stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino):
        raise SourceChangedError(f"Source changed while recording: {path}")
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "device": stat.st_dev, "inode": stat.st_ino,
            "fingerprint_method": method, "fingerprint": digest.hexdigest()}


class ArtifactStore:
    def __init__(self, root: str | Path,
                 allowed_source_roots: Iterable[str | Path] | None = None):
        self.root = Path(root)
        self.allowed_source_roots = (None if allowed_source_roots is None else tuple(
            Path(path).expanduser().resolve(strict=True) for path in allowed_source_roots
        ))

    def _source(self, path: str | Path) -> Path:
        source = Path(path).expanduser().resolve(strict=True)
        if self.allowed_source_roots is not None and not any(
            source == root or root.is_dir() and source.is_relative_to(root)
            for root in self.allowed_source_roots
        ):
            raise ValueError("Source is outside authorized data roots")
        return source

    def _index(self, artifact_id: str) -> Path:
        if not SAFE_ID.fullmatch(artifact_id):
            raise ValueError("Invalid artifact ID")
        return self.root / "artifacts" / "index" / f"{artifact_id}.json"

    def _payload(self, artifact_id: str, suffix: str) -> Path:
        return self.root / "artifacts" / "data" / f"{artifact_id}{suffix}"

    def _start(self, name: str, *, run_id: str, attempt_id: str, stage_id: str,
               inputs: list[str], call_id: str | None, kind: str, summary: dict) -> dict:
        if not isinstance(inputs, list) or not all(isinstance(item, str) for item in inputs):
            raise TypeError("inputs must be an explicit list of artifact IDs")
        for source_id in inputs:
            source = self.read_artifact(source_id)
            if source["status"] != "completed" or source.get("run_id") != run_id:
                raise ValueError(f"Input artifact is unavailable: {source_id}")
        artifact_id = new_id("artifact", attempt_id=attempt_id)
        metadata = {"artifact_id": artifact_id, "run_id": run_id,
                    "attempt_id": attempt_id, "stage_id": stage_id,
                    "call_id": call_id, "name": name, "inputs": inputs,
                    "kind": kind, "summary": summary, "status": "pending"}
        write_json_atomic(self._index(artifact_id), metadata)
        return metadata

    def _finish(self, metadata: dict, **fields) -> str:
        metadata.update(status="completed", **fields)
        write_json_atomic(self._index(metadata["artifact_id"]), metadata)
        return metadata["artifact_id"]

    def _fail(self, metadata: dict, error: Exception) -> None:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json_atomic(self._index(metadata["artifact_id"]), metadata)

    def publish(self, name: str, value: Any, *, run_id: str, attempt_id: str,
                stage_id: str, inputs: list[str], call_id: str | None = None,
                presentation: str = "auto") -> str:
        import xarray as xr
        from .figures import PngFigure

        if presentation not in {"auto", "summary", "map"}:
            raise ValueError("presentation must be auto, summary, or map")
        is_array = isinstance(value, xr.DataArray)
        is_image = isinstance(value, PngFigure)
        if is_image:
            if not 0 < len(value.png_bytes) <= MAX_IMAGE_BYTES:
                raise ValueError("PNG must be nonempty and at most 16 MiB")
            width, height = _png_dimensions(value.png_bytes)
            context = dict(value.metadata)
            encoded_context = json.dumps(context, ensure_ascii=False, allow_nan=False)
            if len(encoded_context.encode("utf-8")) > 4096:
                raise ValueError("Image metadata is too large")
            summary = {"width": width, "height": height, "mime_type": "image/png",
                       "context": context}
        elif is_array:
            summary = {"dims": list(value.dims), "shape": list(value.shape),
                       "units": value.attrs.get("units"), "coordinates": list(value.coords)}
        else:
            summary = {"type": type(value).__name__}
        kind = "image_png" if is_image else "dataarray_netcdf" if is_array else "json"
        metadata = self._start(name, run_id=run_id, attempt_id=attempt_id,
                               stage_id=stage_id, inputs=inputs, call_id=call_id,
                               kind=kind, summary=summary)
        metadata["presentation"] = presentation
        write_json_atomic(self._index(metadata["artifact_id"]), metadata)
        artifact_id = metadata["artifact_id"]
        path = self._payload(artifact_id, ".png" if is_image else ".nc" if is_array else ".json")
        sidecars: list[Path] = []
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if is_image:
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as file:
                    temporary = Path(file.name)
                    try:
                        file.write(value.png_bytes)
                        file.flush()
                        os.fsync(file.fileno())
                        os.replace(temporary, path)
                    finally:
                        temporary.unlink(missing_ok=True)
            elif is_array:
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".nc", delete=False) as file:
                    temporary = Path(file.name)
                try:
                    value.to_netcdf(temporary)
                    with temporary.open("rb") as file:
                        os.fsync(file.fileno())
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            else:
                encoded = _encoded(value, root=self.root, artifact_id=artifact_id,
                                   sidecars=sidecars)
                write_json_atomic(path, {"value": encoded})
            fields = {"payload": str(path.relative_to(self.root))}
            if is_image:
                fields["sha256"] = hashlib.sha256(value.png_bytes).hexdigest()
            return self._finish(metadata, **fields)
        except Exception as error:
            for sidecar in sidecars:
                sidecar.unlink(missing_ok=True)
            self._fail(metadata, error)
            raise

    def publish_source(self, name: str, path: str | Path, *, variable: str,
                       selection: dict, run_id: str, attempt_id: str,
                       stage_id: str, inputs: list[str]) -> str:
        import xarray as xr

        source_path = self._source(path)
        if not source_path.is_file():
            raise ValueError("Only NetCDF files can be referenced")
        source_version = _version(source_path)
        with xr.open_dataset(source_path) as dataset:
            selected = dataset[variable]
            selected = selected.sel(**selection.get("sel", {}))
            selected = selected.isel(**selection.get("isel", {}))
            summary = {"dims": list(selected.dims), "shape": list(selected.shape),
                       "units": selected.attrs.get("units"), "coordinates": list(selected.coords)}
        if _version(source_path) != source_version:
            raise SourceChangedError(f"Source changed while recording: {source_path}")
        metadata = self._start(name, run_id=run_id, attempt_id=attempt_id,
                               stage_id=stage_id, inputs=inputs, call_id=None,
                               kind="source_netcdf", summary=summary)
        try:
            return self._finish(metadata, source_path=str(source_path),
                                source_version=source_version, variable=variable,
                                selection=_encoded(selection))
        except Exception as error:
            self._fail(metadata, error)
            raise

    def read_artifact(self, artifact_id: str, *, include_content: bool = False,
                      offset: int = 0, max_chars: int = 2000) -> dict:
        if max_chars < 1 or max_chars > 10000 or offset < 0:
            raise ValueError("max_chars must be 1..10000 and offset nonnegative")
        path = self._index(artifact_id)
        if ((self.root / "artifacts").is_symlink() or path.parent.is_symlink()
                or path.is_symlink()):
            raise ValueError("Artifact index cannot be a symlink")
        if not path.exists():
            return {"artifact_id": artifact_id, "status": "missing"}
        if path.stat().st_size > MAX_ARTIFACT_INDEX_BYTES:
            raise ValueError("Artifact index is too large")
        with path.open(encoding="utf-8") as file:
            metadata = json.load(file)
        if include_content and metadata["status"] == "completed" and metadata["kind"] == "json":
            payload = _stored_payload(self.root, metadata["payload"])
            with payload.open("rb") as file:
                file.seek(offset)
                content = file.read(max_chars + 1)
            metadata["content"] = content[:max_chars].decode("utf-8", errors="replace")
            metadata["truncated"] = len(content) > max_chars
        return metadata

    def load_result(self, artifact_id: str) -> Any:
        metadata = self.read_artifact(artifact_id)
        if metadata["status"] == "missing":
            raise FileNotFoundError(artifact_id)
        if metadata["status"] != "completed":
            raise RuntimeError(f"Artifact is {metadata['status']}: {artifact_id}")
        if metadata["kind"] == "json":
            with _stored_payload(self.root, metadata["payload"]).open(encoding="utf-8") as file:
                return _decoded(json.load(file)["value"], root=self.root)
        if metadata["kind"] == "image_png":
            from .figures import PngFigure

            return PngFigure(self.read_image(artifact_id),
                             metadata["summary"]["context"])
        import xarray as xr

        if metadata["kind"] == "dataarray_netcdf":
            with xr.open_dataarray(_stored_payload(self.root, metadata["payload"])) as array:
                return array.load()
        if metadata["kind"] != "source_netcdf":
            raise ValueError(f"Unknown artifact format: {metadata['kind']}")
        source_path = self._source(metadata["source_path"])
        if not source_path.is_file() or _version(source_path) != metadata["source_version"]:
            raise SourceChangedError(f"Source changed: {source_path}")
        selection = _decoded(metadata["selection"])
        with xr.open_dataset(source_path) as dataset:
            array = dataset[metadata["variable"]]
            array = array.sel(**selection.get("sel", {}))
            result = array.isel(**selection.get("isel", {})).load()
        if _version(source_path) != metadata["source_version"]:
            raise SourceChangedError(f"Source changed while reading: {source_path}")
        return result

    def read_image(self, artifact_id: str) -> bytes:
        """Read the saved PNG after checking its kind, size, path and digest."""
        metadata = self.read_artifact(artifact_id)
        if metadata["status"] != "completed" or metadata["kind"] != "image_png":
            raise ValueError("Artifact is not a completed PNG image")
        path = _stored_payload(self.root, metadata["payload"])
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("PNG exceeds image size limit")
        data = path.read_bytes()
        width, height = _png_dimensions(data)
        if (hashlib.sha256(data).hexdigest() != metadata.get("sha256") or
                metadata["summary"].get("width") != width or
                metadata["summary"].get("height") != height):
            raise ValueError("PNG content differs from its saved index")
        return data
