"""Model registry: versioned artifacts, integrity checks and the promotion lifecycle.

Security: joblib/pickle files can execute code when loaded. We only load files that
  1. live inside the configured models directory,
  2. are listed in a version manifest written by our own training pipeline, and
  3. match the SHA-256 recorded in ``registry.json`` and the manifest.
Uploaded or unknown model files are never loaded.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib

from loan_approval.config import get_settings

LIFECYCLE = [
    "draft_experiment",
    "training",
    "evaluated",
    "review_required",
    "demo_approved",
    "retired",
]
REGISTRY_FILE = "registry.json"
_LOCK = threading.Lock()


class ArtifactError(RuntimeError):
    """A model artifact is missing, unlisted or failed its integrity check."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def models_dir() -> Path:
    return get_settings().models_dir


def registry_path(base: Path | None = None) -> Path:
    return (base or models_dir()) / REGISTRY_FILE


def read_registry(base: Path | None = None) -> dict:
    path = registry_path(base)
    if not path.exists():
        return {"active_version": None, "versions": {}, "audit_log": []}
    return json.loads(path.read_text())


def write_registry(registry: dict, base: Path | None = None) -> None:
    path = registry_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=2))
    tmp.replace(path)


def _safe_child(base: Path, relative: str) -> Path:
    target = (base / relative).resolve()
    if base.resolve() not in target.parents and target != base.resolve():
        raise ArtifactError(f"Refusing to access {relative!r}: outside the models directory")
    return target


def register_version(version: str, manifest: dict, status: str, base: Path | None = None) -> None:
    if status not in LIFECYCLE:
        raise ValueError(f"Unknown lifecycle status {status!r}")
    base = base or models_dir()
    with _LOCK:
        registry = read_registry(base)
        manifest_path = _safe_child(base, f"{version}/manifest.json")
        registry["versions"][version] = {
            "status": status,
            "created_at": manifest["created_at"],
            "selected_algorithm": manifest["selected_algorithm"],
            "dataset": manifest["dataset"]["name"],
            "headline_metrics": manifest["headline_metrics"],
            "manifest_sha256": sha256_path(manifest_path),
            "artifact_sha256": {k: v["sha256"] for k, v in manifest["artifacts"].items()},
            "gates_passed": all(g["passed"] for g in manifest["promotion_gates"]),
        }
        registry["audit_log"].append(
            {"at": now_iso(), "action": "register", "version": version, "status": status}
        )
        write_registry(registry, base)


def set_status(version: str, status: str, actor: str, reason: str = "", base: Path | None = None):
    if status not in LIFECYCLE:
        raise ValueError(f"Unknown lifecycle status {status!r}")
    base = base or models_dir()
    with _LOCK:
        registry = read_registry(base)
        if version not in registry["versions"]:
            raise ArtifactError(f"Unknown model version {version!r}")
        registry["versions"][version]["status"] = status
        if status == "retired" and registry.get("active_version") == version:
            registry["active_version"] = None
        registry["audit_log"].append(
            {
                "at": now_iso(),
                "action": f"status:{status}",
                "version": version,
                "actor": actor,
                "reason": reason,
            }
        )
        write_registry(registry, base)


def promote(version: str, actor: str, reason: str = "", base: Path | None = None) -> dict:
    """Make a version the active demo model. Requires every promotion gate to have passed."""
    base = base or models_dir()
    manifest = load_manifest(version, base)
    failed = [g["name"] for g in manifest["promotion_gates"] if not g["passed"]]
    if failed:
        raise ArtifactError(f"Version {version} failed promotion gates: {failed}")
    verify_version(version, base)
    with _LOCK:
        registry = read_registry(base)
        previous = registry.get("active_version")
        if previous and previous != version and previous in registry["versions"]:
            registry["versions"][previous]["status"] = "review_required"
        registry["versions"][version]["status"] = "demo_approved"
        registry["active_version"] = version
        registry["audit_log"].append(
            {
                "at": now_iso(),
                "action": "promote",
                "version": version,
                "previous": previous,
                "actor": actor,
                "reason": reason,
            }
        )
        write_registry(registry, base)
    return {"active_version": version, "previous": previous}


def load_manifest(version: str, base: Path | None = None) -> dict:
    base = base or models_dir()
    path = _safe_child(base, f"{version}/manifest.json")
    if not path.exists():
        raise ArtifactError(f"No manifest for model version {version!r}")
    return json.loads(path.read_text())


def verify_version(version: str, base: Path | None = None) -> None:
    base = base or models_dir()
    registry = read_registry(base)
    entry = registry["versions"].get(version)
    if entry is None:
        raise ArtifactError(f"Model version {version!r} is not in the registry")
    manifest_path = _safe_child(base, f"{version}/manifest.json")
    if sha256_path(manifest_path) != entry["manifest_sha256"]:
        raise ArtifactError(f"Manifest for {version} does not match the registry checksum")
    manifest = json.loads(manifest_path.read_text())
    for name, art in manifest["artifacts"].items():
        path = _safe_child(base, f"{version}/{art['path']}")
        if not path.exists():
            raise ArtifactError(f"Artifact {name} for {version} is missing")
        if sha256_path(path) != art["sha256"] or entry["artifact_sha256"].get(name) != art["sha256"]:
            raise ArtifactError(f"Artifact {name} for {version} failed its SHA-256 check")


@dataclass
class LoadedVersion:
    version: str
    manifest: dict
    pipelines: dict  # algorithm -> fitted (calibrated) pipeline
    base: Path

    @property
    def selected(self):
        return self.pipelines[self.manifest["selected_algorithm"]]

    def report(self, name: str) -> dict:
        path = _safe_child(self.base, f"{self.version}/reports/{name}.json")
        return json.loads(path.read_text()) if path.exists() else {}


def load_version(version: str, base: Path | None = None) -> LoadedVersion:
    base = base or models_dir()
    verify_version(version, base)
    manifest = load_manifest(version, base)
    pipelines = {}
    for name, art in manifest["artifacts"].items():
        if art.get("kind") != "pipeline":
            continue
        path = _safe_child(base, f"{version}/{art['path']}")
        pipelines[name] = joblib.load(path)  # trusted: hash-verified, produced by our pipeline
    return LoadedVersion(version=version, manifest=manifest, pipelines=pipelines, base=base)


def active_version(base: Path | None = None) -> str | None:
    return read_registry(base).get("active_version")
