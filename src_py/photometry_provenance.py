"""Moved from photometry_core.py (CONSOLIDATE-01E2). Facade re-exports these names."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
import functools
import json
import subprocess

from photometry_defs import LOGGER

# src_py/photometry_provenance.py -> repo root is parent.parent (git cwd + porcelain path base).
_REPO_ROOT_FOR_PROVENANCE = Path(__file__).resolve().parent.parent

def _is_import_relevant_py_path(path: str) -> bool:
    """True for VYVAR modules imported by the pipeline: ``src_py/*.py`` plus the root ``app.py`` shim.

    Everything under ``dev/`` (tests, scripts, tools, validation, sandbox, orchestrator) and
    ``tmp/`` / ``docs/`` is scratch: it never trips the FAIL-CLOSED dirty-code gate (T3 FIX B).
    """
    p = path.replace("\\", "/").lstrip("./")
    if not p.endswith(".py"):
        return False
    if p == "app.py":  # thin root Streamlit shim (the only import-relevant module at repo root)
        return True
    return p.startswith("src_py/")

def _porcelain_paths(porcelain: str) -> list[str]:
    """Paths of ``git status --porcelain`` lines (``XY PATH`` or ``XY ORIG -> PATH``), "/" separated."""
    out: list[str] = []
    for line in porcelain.splitlines():
        if len(line) < 4 or not line.strip():
            continue
        path_part = line[3:].strip()
        if " -> " in path_part:
            path_part = path_part.split(" -> ", 1)[-1].strip()
        out.append(path_part.strip('"').replace("\\", "/"))
    return out


def classify_git_dirty_paths(dirty_files: Sequence[dict[str, str]]) -> tuple[bool, list[str], list[str]]:
    """Split dirty paths into import-relevant code vs scratch (F-431 / T3 dirty-gate).

    ``dirty_code`` = any changed or untracked import-relevant ``*.py`` (``src_py/*.py``, root ``app.py``);
    everything else is ``dirty_scratch``. (G11-39: the git status letter never changed the verdict, so the
    second ``git status`` call and its parser are gone.)
    """
    code_paths: list[str] = []
    scratch_paths: list[str] = []
    for entry in dirty_files:
        path = str(entry.get("path") or "").replace("\\", "/")
        if not path or path == "...truncated...":
            continue
        (code_paths if _is_import_relevant_py_path(path) else scratch_paths).append(path)
    return bool(code_paths), code_paths, scratch_paths


@functools.cache
def _warn_git_unavailable() -> None:
    """Once per process (G11-39: replaces the module-global flag)."""
    LOGGER.warning("[PHOT] pipeline provenance: git unavailable; git_hash/git_dirty set to null")


def git_head_hash(*, short: bool = False) -> str | None:
    """HEAD of the VYVAR source tree (G10-37: run in the VYVAR folder, never the process CWD, which may be
    another repository or none). None when git or the repository is unavailable."""
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.check_output(cmd, cwd=_REPO_ROOT_FOR_PROVENANCE, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.strip() or None


def _resolve_git_provenance() -> tuple[str | None, bool | None, list[dict[str, str]]]:
    """Return (HEAD hash, dirty flag, dirty file rows) from repo root; nulls when git unavailable.

    When dirty, each row is ``{path, content_sha256}`` for tracked/untracked paths listed by
    ``git status --porcelain`` (F-431 provenance hardening).
    """
    try:
        head = git_head_hash()
        if head is None:
            raise OSError("git rev-parse HEAD failed")
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT_FOR_PROVENANCE,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        dirty = bool(status.strip())
        dirty_files: list[dict[str, str]] = []
        if dirty:
            import hashlib

            for path_part in _porcelain_paths(status):
                fp = _REPO_ROOT_FOR_PROVENANCE / path_part
                sha = ""
                try:
                    if fp.is_file():
                        h = hashlib.sha256()
                        with fp.open("rb") as fh:
                            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                                h.update(chunk)
                        sha = h.hexdigest()
                    elif fp.is_dir():
                        sha = "DIR"
                    else:
                        sha = "MISSING"
                except OSError:
                    sha = "UNREADABLE"
                dirty_files.append({"path": path_part, "content_sha256": sha})
                if len(dirty_files) >= 200:
                    dirty_files.append({"path": "...truncated...", "content_sha256": ""})
                    break
        return (head or None), dirty, dirty_files
    except Exception:  # noqa: BLE001
        _warn_git_unavailable()
        return None, None, []

def _json_safe_snapshot_value(v: Any) -> Any:
    """Coerce an AppConfig field value to a JSON-serializable form for the snapshot."""
    from pathlib import Path as _P

    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, _P):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _json_safe_snapshot_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe_snapshot_value(x) for x in v]
    return str(v)

def _complete_config_snapshot(cfg: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Backfill the provenance snapshot so it covers EVERY public AppConfig field.

    ``AppConfig.to_json()`` carries only the config.json keys; derived/runtime fields
    (project_root, the observer location mirror, gain, read_noise) are
    not in it. For an honest, complete provenance snapshot we backfill any missing public field
    from ``getattr(cfg, name)`` (JSON-coerced). This does NOT touch ``to_json()`` / config.json
    (save/load semantics unchanged) and is metadata only -- the anchor comparator ignores
    pipeline_meta.json, so fresh snapshots become complete without any numeric behaviour change.
    """
    try:
        import dataclasses

        names = [f.name for f in dataclasses.fields(cfg) if not f.name.startswith("_")]
    except Exception:  # noqa: BLE001
        return snapshot
    out = dict(snapshot)
    for name in names:
        if name in out:
            continue
        try:
            out[name] = _json_safe_snapshot_value(getattr(cfg, name))
        except Exception:  # noqa: BLE001
            continue
    return out

def _build_pipeline_provenance_block(cfg: Any, *, entry_point: str) -> dict[str, Any]:
    """Run provenance stamped into ``pipeline_meta.json`` (last writer wins)."""
    git_hash, git_dirty, dirty_files = _resolve_git_provenance()
    if hasattr(cfg, "to_dict"):
        config_snapshot = cfg.to_dict()
    elif hasattr(cfg, "to_json"):
        config_snapshot = cfg.to_json()
    else:
        from dataclasses import asdict

        config_snapshot = asdict(cfg)
    config_snapshot = _complete_config_snapshot(cfg, config_snapshot)
    block: dict[str, Any] = {
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "config_snapshot": config_snapshot,
        "stamped_at_utc": datetime.now(timezone.utc).isoformat(),
        "entry_point": entry_point,
        "labbe_rng_seed_policy": "content_frame_hash_v1",
    }
    try:
        from catalog_provenance import build_catalog_provenance_block  # noqa: PLC0415

        block["catalog_databases"] = build_catalog_provenance_block(cfg)
    except Exception as exc:  # noqa: BLE001
        block["catalog_databases_error"] = str(exc)
    if git_dirty is True:
        block["git_dirty_files"] = dirty_files
        code_dirty, code_paths, scratch_paths = classify_git_dirty_paths(dirty_files)
        block["git_dirty_code"] = code_dirty
        block["git_dirty_code_files"] = code_paths
        block["git_dirty_scratch_files"] = scratch_paths
    elif git_dirty is False:
        block["git_dirty_code"] = False
    else:
        block["git_dirty_code"] = None
    return block

def _json_default(o: Any) -> Any:
    """numpy scalars / arrays and paths in pipeline_meta values."""
    import numpy as np  # noqa: PLC0415

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"pipeline_meta: value of type {type(o).__name__} is not JSON serialisable")


def merge_photometry_pipeline_meta(
    photometry_dir: Path | str,
    updates: dict[str, Any],
    cfg: Any = None,
    *,
    entry_point: str | None = None,
) -> None:
    """Merge keys into ``photometry/pipeline_meta.json`` (MASTERSTAR + aperture photometry).

    FAIL-LOUD-2026-10 (code review 2 G11-07): an unreadable existing file is kept as ``.corrupt`` and the error
    raised (it used to be replaced by the new keys only, losing provenance, site and sigma_sys); numpy values are
    serialised; the write is atomic (tmp + os.replace); a write error is raised, not logged at DEBUG.
    """
    import os  # noqa: PLC0415

    _meta_path = Path(photometry_dir) / "pipeline_meta.json"
    _meta_path.parent.mkdir(parents=True, exist_ok=True)
    _existing: dict[str, Any] = {}
    if _meta_path.is_file():
        try:
            _existing = json.loads(_meta_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            _bad = _meta_path.with_suffix(".json.corrupt")
            os.replace(_meta_path, _bad)
            raise RuntimeError(f"pipeline_meta.json unreadable (kept as {_bad.name}): {exc}") from exc
    _merged = dict(updates)
    if cfg is not None and entry_point:
        _merged["provenance"] = _build_pipeline_provenance_block(cfg, entry_point=entry_point)
    _existing.update(_merged)
    _tmp = _meta_path.with_suffix(".json.tmp")
    _tmp.write_text(json.dumps(_existing, indent=2, default=_json_default), encoding="utf-8")
    os.replace(_tmp, _meta_path)
