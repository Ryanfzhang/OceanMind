"""A script-writable artifact index cannot redirect a parent read."""

import json

import pytest

from packages.agent_loop.analysis import AnalysisSession
from packages.analysis_runtime.records import write_json_atomic


def _published(session):
    attempt = session.records.new_attempt()
    stage = session.records.new_stage(attempt, "result")
    artifact_id = session.artifacts.publish(
        "value", {"number": 1}, run_id=session.run_id,
        attempt_id=attempt, stage_id=stage, inputs=[],
    )
    return artifact_id, session.artifacts.read_artifact(artifact_id)


def test_forged_absolute_and_symlink_payloads_cannot_be_read(tmp_path):
    session = AnalysisSession(tmp_path / "work")
    artifact_id, metadata = _published(session)
    secret = tmp_path / "secret.json"
    secret.write_text('{"secret":"do not disclose"}')
    index = session.root / "artifacts" / "index" / f"{artifact_id}.json"

    write_json_atomic(index, {**metadata, "payload": str(secret)})
    with pytest.raises(ValueError, match="payload path"):
        session.read_artifact(artifact_id)

    write_json_atomic(index, metadata)
    payload = session.root / metadata["payload"]
    payload.unlink()
    payload.symlink_to(secret)
    with pytest.raises(ValueError, match="escapes"):
        session.read_artifact(artifact_id)


def test_symlinked_store_directory_and_index_are_rejected(tmp_path):
    session = AnalysisSession(tmp_path / "work")
    artifact_id, metadata = _published(session)
    data_dir = session.root / "artifacts" / "data"
    data_dir.rename(session.root / "artifacts" / "data_original")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / (session.root / metadata["payload"]).name).write_text(
        json.dumps({"value": "secret"})
    )
    data_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="data store cannot be a symlink"):
        session.read_artifact(artifact_id)

    data_dir.unlink()
    (session.root / "artifacts" / "data_original").rename(data_dir)
    index = session.root / "artifacts" / "index" / f"{artifact_id}.json"
    index.unlink()
    index.symlink_to(tmp_path / "secret_index.json")
    with pytest.raises(ValueError, match="index cannot be a symlink"):
        session.read_artifact(artifact_id)


def test_parent_log_write_rejects_script_created_symlink(tmp_path):
    session = AnalysisSession(tmp_path / "work")
    saved = session.write_analysis('print("hello")\n')
    outside = tmp_path / "outside"
    outside.mkdir()
    logs = session.root / "logs"
    logs.rmdir()
    logs.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="log directory"):
        session.run_analysis(saved["code_id"])
    assert list(outside.iterdir()) == []
