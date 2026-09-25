import numpy as np
import pytest
import xarray as xr

from packages.analysis_runtime.artifacts import ArtifactStore, SourceChangedError
from packages.analysis_runtime.records import RunRecords, new_id


def context(tmp_path):
    records = RunRecords(tmp_path)
    attempt_id = records.new_attempt(code_version="v1")
    stage_id = records.new_stage(attempt_id, "Detect")
    return records, attempt_id, stage_id, ArtifactStore(tmp_path)


def test_ids_are_unique_and_records_survive_reopen(tmp_path):
    records, first, stage, _ = context(tmp_path)
    another_stage = records.new_stage(first, "Detect")
    call_a = records.new_call(first, stage, "detect_eddies")
    call_b = records.new_call(first, stage, "detect_eddies")
    retry = records.new_attempt(code_version="v2")
    retry_stage = records.new_stage(retry, "Detect")
    ids = {records.run_id, first, retry, stage, another_stage, retry_stage, call_a, call_b}
    assert len(ids) == 8
    assert first.startswith(records.run_id)
    assert stage.startswith(first)
    assert retry_stage.startswith(retry)
    assert new_id("artifact", attempt_id=first) != new_id("artifact", attempt_id=first)
    records.update("call", call_a, status="completed", artifact_ids=["a"])
    reopened = RunRecords(tmp_path, records.run_id)
    assert reopened.read("call", call_a)["artifact_ids"] == ["a"]


def test_json_roundtrip_provenance_and_bounded_read(tmp_path):
    records, attempt, stage, store = context(tmp_path)
    first = store.publish("custom statistic", {"mask": np.array([[True, False]]),
                                             "score": float("nan"), "text": "x" * 6000},
                          run_id=records.run_id, attempt_id=attempt, stage_id=stage, inputs=[])
    second = store.publish("followup", {"count": 2}, run_id=records.run_id,
                           attempt_id=attempt, stage_id=stage, inputs=[first])
    reopened = ArtifactStore(tmp_path)
    value = reopened.load_result(first)
    np.testing.assert_array_equal(value["mask"], np.array([[True, False]]))
    assert np.isnan(value["score"])
    assert reopened.read_artifact(second)["inputs"] == [first]
    excerpt = reopened.read_artifact(first, include_content=True, max_chars=50)
    assert len(excerpt["content"]) == 50 and excerpt["truncated"]
    assert reopened.read_artifact("missing_artifact")["status"] == "missing"
    with pytest.raises(ValueError, match="unavailable"):
        store.publish("bad input", {}, run_id=records.run_id, attempt_id=attempt,
                      stage_id=stage, inputs=["missing_artifact"])


def test_failed_write_never_publishes_completed_artifact(tmp_path, monkeypatch):
    import packages.analysis_runtime.artifacts as module

    records, attempt, stage, store = context(tmp_path)
    original = module.write_json_atomic

    def fail_payload(path, value):
        if path.parent.name == "data":
            raise OSError("disk unavailable")
        original(path, value)

    monkeypatch.setattr(module, "write_json_atomic", fail_payload)
    with pytest.raises(OSError, match="disk unavailable"):
        store.publish("broken", {"x": 1}, run_id=records.run_id,
                      attempt_id=attempt, stage_id=stage, inputs=[])
    indexes = list((tmp_path / "artifacts" / "index").glob("*.json"))
    assert len(indexes) == 1
    artifact_id = indexes[0].stem
    assert store.read_artifact(artifact_id)["status"] == "failed"
    with pytest.raises(RuntimeError, match="failed"):
        store.load_result(artifact_id)
    pending = store.read_artifact(artifact_id)
    pending["status"] = "pending"
    original(indexes[0], pending)
    assert store.read_artifact(artifact_id)["status"] == "pending"


def test_dataarray_snapshot_survives_mutation_and_reopen(tmp_path):
    records, attempt, stage, store = context(tmp_path)
    array = xr.DataArray(np.array([[1.0, 2.0], [3.0, 4.0]]),
                         dims=("lat", "lon"), coords={"lat": [10, 11], "lon": [120, 121]},
                         attrs={"units": "m/s"}, name="velocity")
    array.coords["lat"].attrs["units"] = "degrees_north"
    artifact_id = store.publish("velocity", array, run_id=records.run_id,
                                attempt_id=attempt, stage_id=stage, inputs=[])
    array.values[:] = 99
    restored = ArtifactStore(tmp_path).load_result(artifact_id)
    assert restored.dims == ("lat", "lon")
    assert restored.attrs["units"] == "m/s"
    assert restored.coords["lat"].attrs["units"] == "degrees_north"
    assert restored.sel(lat=10, lon=120).item() == 1
    assert store.read_artifact(artifact_id)["summary"]["shape"] == [2, 2]


def test_large_nested_array_uses_sidecar_not_json(tmp_path):
    records, attempt, stage, store = context(tmp_path)
    values = np.arange(30_000, dtype=np.float64)
    artifact_id = store.publish("eddy fields", {"ow_field": values},
                                run_id=records.run_id, attempt_id=attempt,
                                stage_id=stage, inputs=[])
    metadata = store.read_artifact(artifact_id)
    payload = tmp_path / metadata["payload"]
    assert payload.stat().st_size < 1000
    assert list((tmp_path / "artifacts" / "data").glob("*_array_*.npy"))
    np.testing.assert_array_equal(ArtifactStore(tmp_path).load_result(artifact_id)["ow_field"],
                                  values)


def test_nested_dataarray_roundtrips_for_structured_analysis_result(tmp_path):
    records, attempt, stage, store = context(tmp_path)
    pattern = xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("lat", "lon"),
                           coords={"lat": [20, 21], "lon": [120, 121]})
    artifact_id = store.publish("eof", {"modes": [{"spatial_pattern": pattern}]},
                                run_id=records.run_id, attempt_id=attempt,
                                stage_id=stage, inputs=[])
    restored = ArtifactStore(tmp_path).load_result(artifact_id)
    xr.testing.assert_identical(restored["modes"][0]["spatial_pattern"], pattern)


def test_source_reference_checks_content_before_reopen(tmp_path):
    records, attempt, stage, store = context(tmp_path)
    path = tmp_path / "ocean.nc"
    data = xr.DataArray(np.arange(12).reshape(2, 2, 3), dims=("time", "lat", "lon"),
                        coords={"time": [0, 1], "lat": [10, 11], "lon": [120, 121, 122]},
                        attrs={"units": "degC"}, name="temp")
    data.to_dataset().to_netcdf(path)
    artifact_id = store.publish_source("source", path, variable="temp",
                                       selection={"sel": {"time": 1}, "isel": {"lon": slice(0, 2)}},
                                       run_id=records.run_id, attempt_id=attempt,
                                       stage_id=stage, inputs=[])
    assert store.read_artifact(artifact_id)["source_version"]["fingerprint_method"] == "sha256_full"
    restored = ArtifactStore(tmp_path).load_result(artifact_id)
    assert restored.dims == ("lat", "lon")
    assert restored.shape == (2, 2)
    assert restored.attrs["units"] == "degC"
    changed = data + 100
    changed.to_dataset().to_netcdf(path)
    with pytest.raises(SourceChangedError, match="Source changed"):
        ArtifactStore(tmp_path).load_result(artifact_id)


def test_large_source_fingerprint_reads_fixed_samples(tmp_path):
    from packages.analysis_runtime.artifacts import _version

    path = tmp_path / "large.nc"
    with path.open("wb") as file:
        file.write(b"header")
        file.truncate(20 * 1024 * 1024)
    version = _version(path)
    assert version["fingerprint_method"] == "sha256_three_samples"
    assert version["size"] == 20 * 1024 * 1024
