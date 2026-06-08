from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from catalog_crossmatch import CrossmatchResult, check_candidate_in_catalogs
from utils import _NumpyEncoder

LOGGER = logging.getLogger(__name__)

CROSSMATCH_DELAY_S = 1.0


def _fmt_cell_existing(v: Any) -> str:
    s = str(v if v is not None else "").strip()
    return s


def _has_positive_catalog_match(katalogy_text: str) -> bool:
    """True only if at least one catalog line has a real match (not 'žiadny záznam')."""
    if not katalogy_text or not str(katalogy_text).strip():
        return False
    for line in str(katalogy_text).splitlines():
        line = line.strip()
        if not line:
            continue
        if "žiadny záznam" in line:
            continue
        if line.startswith("🔭"):
            continue
        return True
    return False


def _format_katalogy_cell(result: CrossmatchResult | dict[str, Any]) -> str:
    """Format crossmatch result into multiline cell (one catalog per line)."""
    if isinstance(result, CrossmatchResult):
        return "\n".join(result.catalog_summary_bullets())
    # best-effort: expect catalog_crossmatch.asdict(CrossmatchResult)
    try:
        matches = result.get("matches", {}) if isinstance(result, dict) else {}
        errors = result.get("errors", {}) if isinstance(result, dict) else {}
        ra = float(result.get("ra"))
        dec = float(result.get("dec"))
        radius = float(result.get("radius_arcsec", 10.0))
        mag = result.get("mag", None)
        # Rehydrate minimal CrossmatchResult to reuse bullet formatter.
        cm = CrossmatchResult(
            ra=ra,
            dec=dec,
            mag=(float(mag) if mag is not None and math.isfinite(float(mag)) else None),
            radius_arcsec=radius,
            matches={},  # filled below
            errors={str(k): str(v) for k, v in (errors or {}).items()},
        )
        # Rehydrate matches -> CatalogMatch-like dicts are ok if empty; we only need summary lines.
        # If structure doesn't match, just fall back to JSON text.
        from catalog_crossmatch import CatalogMatch  # noqa: PLC0415

        out_matches: dict[str, list[CatalogMatch]] = {}
        for cat, lst in (matches or {}).items():
            items: list[CatalogMatch] = []
            for m in (lst or []):
                try:
                    items.append(
                        CatalogMatch(
                            catalog=str(m.get("catalog", cat) or cat),
                            name=str(m.get("name", "") or "—"),
                            var_type=str(m.get("var_type", "") or ""),
                            period=(float(m["period"]) if m.get("period") is not None else None),
                            amplitude=(float(m["amplitude"]) if m.get("amplitude") is not None else None),
                            delta_r=(float(m["delta_r"]) if m.get("delta_r") is not None else None),
                            mag=(float(m["mag"]) if m.get("mag") is not None else None),
                            epoch=(float(m["epoch"]) if m.get("epoch") is not None else None),
                            extra=(m.get("extra") if isinstance(m.get("extra"), dict) else {}),
                        )
                    )
                except Exception:  # noqa: BLE001
                    continue
            out_matches[str(cat)] = items
        cm.matches = out_matches
        return "\n".join(cm.catalog_summary_bullets())
    except Exception:  # noqa: BLE001
        try:
            return json.dumps(result, ensure_ascii=False, cls=_NumpyEncoder)
        except Exception:  # noqa: BLE001
            return "—"


def _load_radec_map(output_dir: Path) -> dict[str, tuple[float, float]]:
    """Load catalog_id -> (ra_deg, dec_deg) from available CSVs."""
    out: dict[str, tuple[float, float]] = {}
    candidates = [
        Path(output_dir) / "active_targets.csv",
        Path(output_dir).parent / "masterstars_full_match.csv",
        Path(output_dir) / "photometry_summary.csv",
    ]
    for csv_path in candidates:
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, dtype={"catalog_id": str, "name": str}, low_memory=False)
        except Exception:  # noqa: BLE001
            continue
        ra_col = "ra_deg" if "ra_deg" in df.columns else ("ra" if "ra" in df.columns else None)
        de_col = "dec_deg" if "dec_deg" in df.columns else ("dec" if "dec" in df.columns else None)
        if not ra_col or not de_col or "catalog_id" not in df.columns:
            continue
        for _, r in df.iterrows():
            cid = str(r.get("catalog_id") or "").strip()
            if not cid:
                continue
            ra = pd.to_numeric(r.get(ra_col), errors="coerce")
            de = pd.to_numeric(r.get(de_col), errors="coerce")
            try:
                raf = float(ra)
                def_ = float(de)
            except Exception:  # noqa: BLE001
                continue
            if not (math.isfinite(raf) and math.isfinite(def_)):
                continue
            out[cid] = (raf, def_)
        if out:
            return out
    LOGGER.warning("[RADEC] Nenajdeny subor s RA/Dec v %s", str(output_dir))
    return out


def _load_radec_map_from_df(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Load catalog_id -> (ra, dec) directly from a DataFrame (candidates CSV)."""
    out: dict[str, tuple[float, float]] = {}
    if df is None or df.empty or "catalog_id" not in df.columns:
        return out
    ra_col = "ra_deg" if "ra_deg" in df.columns else ("ra" if "ra" in df.columns else None)
    de_col = "dec_deg" if "dec_deg" in df.columns else ("dec" if "dec" in df.columns else None)
    if not ra_col or not de_col:
        return out
    for _, r in df.iterrows():
        cid = str(r.get("catalog_id") or "").strip()
        if not cid:
            continue
        ra = pd.to_numeric(r.get(ra_col), errors="coerce")
        de = pd.to_numeric(r.get(de_col), errors="coerce")
        try:
            raf = float(ra)
            def_ = float(de)
        except Exception:  # noqa: BLE001
            continue
        if not (math.isfinite(raf) and math.isfinite(def_)):
            continue
        out[cid] = (raf, def_)
    return out


def _run_crossmatch(
    *,
    ra: float,
    dec: float,
    mag: float | None,
    radius_arcsec: float = 10.0,
    vsx_local_db_path: str | None = None,
) -> CrossmatchResult:
    return check_candidate_in_catalogs(
        ra=float(ra),
        dec=float(dec),
        mag=mag,
        radius_arcsec=float(radius_arcsec),
        vsx_local_db_path=vsx_local_db_path,
    )


def auto_crossmatch_candidates(
    *,
    candidates_csv: Path,
    output_dir: Path,
    cfg: Any,
) -> None:
    """Fill 'katalogy' column for all candidates and write cache JSON per catalog_id."""
    candidates_csv = Path(candidates_csv)
    output_dir = Path(output_dir)
    if not candidates_csv.exists():
        LOGGER.info("[CROSSMATCH] candidates_csv missing: %s", str(candidates_csv))
        return
    try:
        df = pd.read_csv(candidates_csv, dtype={"catalog_id": str, "name": str}, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[CROSSMATCH] read failed: %s (%s)", str(candidates_csv), exc)
        return
    if df.empty or "catalog_id" not in df.columns:
        return

    cache_dir = output_dir / "_crossmatch"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Prefer RA/Dec already present in candidates CSV, fallback to other CSVs in output_dir.
    radec_map = _load_radec_map_from_df(df)
    if not radec_map:
        radec_map = _load_radec_map(output_dir)

    # Primárny zdroj súradníc: active_targets.csv (WCS-overené); kandidátsky CSV len fallback.
    _at_path = Path(output_dir) / "active_targets.csv"
    _at_radec: dict[str, tuple[float, float]] = {}
    if _at_path.is_file():
        try:
            _at_df = pd.read_csv(_at_path, dtype=str, low_memory=False)
            for _, r in _at_df.iterrows():
                cid = str(r.get("catalog_id", "") or "").strip()
                ra = pd.to_numeric(r.get("ra_deg", ""), errors="coerce")
                dec = pd.to_numeric(r.get("dec_deg", ""), errors="coerce")
                if cid and math.isfinite(float(ra)) and math.isfinite(float(dec)):
                    _at_radec[cid] = (float(ra), float(dec))
        except Exception:  # noqa: BLE001
            pass

    radec_map = {**radec_map, **_at_radec}
    LOGGER.info("[CROSSMATCH] radec_map: %d zo active_targets, %d celkom", len(_at_radec), len(radec_map))

    changed = False
    created_col = False

    # Column name preference: keep existing, else create ASCII-friendly 'katalogy'.
    col_kat = "katalogy"
    for c in ("katalogy", "katalógy", "katalog", "catalogs"):
        if c in df.columns:
            col_kat = c
            break
    if col_kat not in df.columns:
        df[col_kat] = "—"
        created_col = True

    for idx, row in df.iterrows():
        cid = str(row.get("catalog_id") or "").strip()
        if not cid:
            continue

        existing = _fmt_cell_existing(row.get(col_kat, ""))
        if _has_positive_catalog_match(existing):
            continue

        cache_path = cache_dir / f"{cid}.json"
        if cache_path.exists():
            cache_ok = False
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                cached_text = _format_katalogy_cell(raw)
                if _has_positive_catalog_match(cached_text):
                    df.at[idx, col_kat] = cached_text
                    changed = True
                    cache_ok = True
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("[CROSSMATCH] Cache write/read failed (non-critical): %s", exc)
            if cache_ok:
                continue
            try:
                cache_path.unlink()
            except OSError:
                pass
            # Stale or unreadable cache — fall through to API crossmatch.

        radec = radec_map.get(cid)
        if radec is None:
            LOGGER.warning("[CROSSMATCH] RA/Dec not found for %s", cid)
            continue
        ra, dec = radec

        mag = None
        if "mag" in df.columns:
            try:
                mv = float(pd.to_numeric(row.get("mag"), errors="coerce"))
                mag = mv if math.isfinite(mv) else None
            except Exception:  # noqa: BLE001
                mag = None

        try:
            _vsx_p = str(getattr(cfg, "vsx_local_db_path", "") or "").strip() or None
            res = _run_crossmatch(ra=ra, dec=dec, mag=mag, radius_arcsec=10.0, vsx_local_db_path=_vsx_p)
            cache_path.write_text(
                json.dumps(asdict(res), ensure_ascii=False, indent=2, cls=_NumpyEncoder),
                encoding="utf-8",
            )
            df.at[idx, col_kat] = _format_katalogy_cell(res)
            changed = True
            LOGGER.info("[CROSSMATCH] OK: %s", cid)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[CROSSMATCH] FAILED %s: %s", cid, exc)

        time.sleep(float(CROSSMATCH_DELAY_S))

    if changed or created_col:
        df.to_csv(candidates_csv, index=False)
        LOGGER.info("[CROSSMATCH] Updated: %s", str(candidates_csv))

