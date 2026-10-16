"""Download the public HMDA file, record its provenance, and build the processed model table."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import yaml

from loan_approval.config import ROOT, get_settings
from loan_approval.data import hmda
from loan_approval.data.validate import QualityReport, build_model_frame

DEFAULT_STATE = "TX"
DEFAULT_YEAR = 2025
MANIFEST_PATH = ROOT / "data" / "data_manifest.yaml"


def raw_filename(state: str, year: int) -> str:
    return f"hmda_{year}_{state}_actions_1_2_3.csv"


def processed_filename(state: str, year: int) -> str:
    return f"hmda_{year}_{state}_model.parquet"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ca_bundle() -> str | bool:
    for env in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        value = os.environ.get(env)
        if value and Path(value).exists():
            return value
    return True


def fetch(state: str = DEFAULT_STATE, year: int = DEFAULT_YEAR, force: bool = False) -> Path:
    """Download approved/denied applications for one state and year from the FFIEC Data Browser."""
    settings = get_settings()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    target = settings.raw_dir / raw_filename(state, year)
    params = {"states": state, "years": str(year), "actions_taken": "1,2,3"}
    if target.exists() and not force:
        print(f"Using existing {target} (pass --force to re-download)")
    else:
        tmp = target.with_suffix(".part")
        with (
            httpx.Client(follow_redirects=True, timeout=600, verify=_ca_bundle()) as client,
            client.stream("GET", hmda.HMDA_SOURCE["api"], params=params) as resp,
        ):
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes(1 << 20):
                    fh.write(chunk)
        tmp.replace(target)
        print(f"Downloaded {target.stat().st_size / 1e6:.1f} MB to {target}")

    digest = sha256_file(target)
    manifest = load_manifest()
    key = f"{state}_{year}"
    previous = manifest.get("files", {}).get(key, {})
    if previous.get("sha256") and previous["sha256"] != digest:
        print(
            "WARNING: the file checksum differs from data/data_manifest.yaml. HMDA snapshots are "
            "occasionally re-published; metrics from earlier model versions may not reproduce."
        )
    header = pd.read_csv(target, nrows=0).columns.tolist()
    with target.open("rb") as fh:
        rows = sum(1 for _ in fh) - 1
    manifest.setdefault("files", {})[key] = {
        "filename": target.name,
        "source": hmda.HMDA_SOURCE["api"],
        "query": params,
        "retrieved_at": previous.get("retrieved_at")
        if previous.get("sha256") == digest
        else datetime.now(UTC).isoformat(timespec="seconds"),
        "sha256": digest,
        "bytes": target.stat().st_size,
        "rows": rows,
        "columns": len(header),
    }
    manifest["dataset"] = hmda.HMDA_SOURCE
    manifest["license"] = (
        "Public data released by U.S. federal agencies (FFIEC/CFPB) under HMDA. The modified LAR "
        "is published for public use; re-check the FFIEC terms before redistributing raw files."
    )
    save_manifest(manifest)
    return target


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return yaml.safe_load(MANIFEST_PATH.read_text()) or {}
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(yaml.safe_dump(manifest, sort_keys=False))


def read_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def prepare(
    state: str = DEFAULT_STATE, year: int = DEFAULT_YEAR, raw_path: Path | None = None
) -> tuple[Path, QualityReport]:
    """Raw CSV -> validated, filtered parquet table plus a quality report next to it."""
    settings = get_settings()
    raw_path = raw_path or settings.raw_dir / raw_filename(state, year)
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. Run `loan-approval fetch` first.")
    frame, report = build_model_frame(read_raw(raw_path))
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    out = settings.processed_dir / processed_filename(state, year)
    frame.to_parquet(out, index=False)
    meta = {
        "raw_file": raw_path.name,
        "raw_sha256": sha256_file(raw_path),
        "rows": len(frame),
        "quality": report.to_dict(),
    }
    out.with_suffix(".quality.json").write_text(json.dumps(meta, indent=2))
    return out, report
