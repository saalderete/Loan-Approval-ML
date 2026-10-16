import json
import shutil

import pytest

from loan_approval.registry import artifacts


def _copy(trained_env, tmp_path):
    base = tmp_path / "models"
    shutil.copytree(trained_env["base"], base)
    return base, trained_env["version"]


def test_verified_version_loads(trained_env, tmp_path):
    base, version = _copy(trained_env, tmp_path)
    loaded = artifacts.load_version(version, base)
    assert loaded.manifest["selected_algorithm"] in loaded.pipelines


def test_tampered_artifact_is_refused(trained_env, tmp_path):
    base, version = _copy(trained_env, tmp_path)
    manifest = artifacts.load_manifest(version, base)
    path = base / version / next(iter(manifest["artifacts"].values()))["path"]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(artifacts.ArtifactError, match="SHA-256"):
        artifacts.load_version(version, base)


def test_tampered_manifest_is_refused(trained_env, tmp_path):
    base, version = _copy(trained_env, tmp_path)
    mpath = base / version / "manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["selected_algorithm"] = "something_else"
    mpath.write_text(json.dumps(manifest))
    with pytest.raises(artifacts.ArtifactError, match="checksum"):
        artifacts.verify_version(version, base)


def test_unlisted_version_and_path_traversal_are_refused(trained_env, tmp_path):
    base, _ = _copy(trained_env, tmp_path)
    with pytest.raises(artifacts.ArtifactError):
        artifacts.verify_version("not-registered", base)
    with pytest.raises(artifacts.ArtifactError):
        artifacts.load_manifest("../../etc", base)


def test_promotion_requires_passing_gates(trained_env, tmp_path):
    base, version = _copy(trained_env, tmp_path)
    mpath = base / version / "manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["promotion_gates"][0]["passed"] = False
    mpath.write_text(json.dumps(manifest))
    with pytest.raises(artifacts.ArtifactError, match="gates"):
        artifacts.promote(version, actor="test", base=base)


def test_retiring_active_version_clears_it(trained_env, tmp_path):
    base, version = _copy(trained_env, tmp_path)
    artifacts.set_status(version, "retired", actor="test", base=base)
    assert artifacts.read_registry(base)["active_version"] is None
