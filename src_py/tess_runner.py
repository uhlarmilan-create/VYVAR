from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from tess_verify import run_tess_analysis
from crossmatch_runner import _load_radec_map

LOGGER = logging.getLogger(__name__)


def _fmt_period(p: Any) -> str:
    if p is None:
        return "   -   "
    try:
        f = float(p)
    except (TypeError, ValueError):
        return "   -   "
    return f"{f:.6f}d" if math.isfinite(f) else "   -   "


def _write_tess_result_txt(result_json_path: Path, result_txt_path: Path, catalog_id: str) -> None:
    try:
        data = json.loads(Path(result_json_path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[TESS TXT] read failed: %s", exc)
        return

    def _fmt_coord(x: Any) -> str:
        try:
            return f"{float(x):.4f}"
        except Exception:  # noqa: BLE001
            # EXC-0469: ? -- intent unclear (try: / return f'{float(x):.4f}' / except Exception:  # noqa: BLE001 / r... (EXCEPT-BULK 2026-07-08)
            return "?"

    lines: list[str] = [
        f"TESS Analysis Summary - {catalog_id}",
        f"RA={_fmt_coord(data.get('ra'))}  Dec={_fmt_coord(data.get('dec'))}  mag={data.get('mag', '?')}",
        f"Sectors found: {data.get('total_sectors_found', 0)}  |  OK: {data.get('total_sectors_ok', 0)}",
        "",
        "Periods (consensus across sectors):",
        f"  LS consensus:    {_fmt_period(data.get('period_consensus'))}",
        f"  ANOVA consensus: {_fmt_period(data.get('period_anova_consensus'))}",
        f"  2P consensus:    {_fmt_period(data.get('period_2p_consensus'))}",
        "",
        "Per-sector detail:",
    ]

    for s in data.get("sectors", []) or []:
        err = s.get("error")
        sec = s.get("sector", "?")
        if err:
            lines.append(f"  Sector {sec:>3}: ERROR - {err}")
            continue
        lines.append(
            f"  Sector {sec:>3}: "
            f"n={s.get('n_points', '?'):>5}  "
            f"P_ls={_fmt_period(s.get('period_ls'))}  "
            f"P_anova={_fmt_period(s.get('period_anova'))}  "
            f"P_con={_fmt_period(s.get('period_consensus'))}  "
            f"method={s.get('period_method_used', '?')}  "
            f"{s.get('harmonic_note', '') or ''}"
        )

    lines += ["", "Files generated:"]
    for s in data.get("sectors", []) or []:
        for key in ("plot_raw_path", "plot_phased_p_path", "plot_phased_2p_path"):
            p = s.get(key)
            if p:
                lines.append(f"  {Path(p).name}")

    try:
        Path(result_txt_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        LOGGER.info("[TESS TXT] Wrote: %s", str(result_txt_path))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[TESS TXT] write failed: %s", exc)


def _period_hint_from_crossmatch_cache(cache_path: Path) -> float | None:
    try:
        raw = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        # EXC-0470: ? -- intent unclear (try: / raw = json.loads(Path(cache_path).read_text(encoding='utf-8')) /... (EXCEPT-BULK 2026-07-08)
        return None
    try:
        matches = raw.get("matches", {}) if isinstance(raw, dict) else {}
        vsx = matches.get("VSX", []) if isinstance(matches, dict) else []
        if vsx:
            p = vsx[0].get("period")
            if p is not None:
                pf = float(p)
                return pf if math.isfinite(pf) and pf > 0 else None
    except Exception:  # noqa: BLE001
        # EXC-0471: ? -- intent unclear (pf = float(p) / return pf if math.isfinite(pf) and pf > 0 else None / e... (EXCEPT-BULK 2026-07-08)
        return None
    return None


def auto_tess_verify_candidates(
    *,
    candidates_csv: Path,
    output_dir: Path,
    cfg: Any,
) -> None:
    """Run TESS analysis for all candidates (best-effort) and write result.txt next to result.json."""
    candidates_csv = Path(candidates_csv)
    output_dir = Path(output_dir)
    if not candidates_csv.exists():
        return
    try:
        df = pd.read_csv(
            candidates_csv,
            dtype={"catalog_id": str, "name": str},  # Gaia ID musi byt str - float64 straca cifry
            low_memory=False,
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning('[EXC-0472] intent unclear (low_memory=False, / ) / except Exception:  # noqa: BLE001 / return / if...: %s', exc)
        return
    if df.empty or "catalog_id" not in df.columns:
        return

    if not bool(getattr(cfg, "tess_enabled", False)):
        LOGGER.info("[TESS] preskocene - tess_enabled=False (auto_tess_verify_candidates)")
        return

    # Prefer RA/Dec already present in candidates CSV, fallback to other CSVs in output_dir.
    radec_map: dict[str, tuple[float, float]] = {}
    try:
        if "ra_deg" in df.columns and "dec_deg" in df.columns:
            for _, r in df.iterrows():
                cid0 = str(r.get("catalog_id") or "").strip()
                if not cid0:
                    continue
                ra0 = pd.to_numeric(r.get("ra_deg"), errors="coerce")
                de0 = pd.to_numeric(r.get("dec_deg"), errors="coerce")
                try:
                    raf = float(ra0)
                    def_ = float(de0)
                except Exception:  # noqa: BLE001
                    continue
                if math.isfinite(raf) and math.isfinite(def_):
                    radec_map[cid0] = (raf, def_)
        elif "ra" in df.columns and "dec" in df.columns:
            for _, r in df.iterrows():
                cid0 = str(r.get("catalog_id") or "").strip()
                if not cid0:
                    continue
                ra0 = pd.to_numeric(r.get("ra"), errors="coerce")
                de0 = pd.to_numeric(r.get("dec"), errors="coerce")
                try:
                    raf = float(ra0)
                    def_ = float(de0)
                except Exception:  # noqa: BLE001
                    continue
                if math.isfinite(raf) and math.isfinite(def_):
                    radec_map[cid0] = (raf, def_)
    except Exception:  # noqa: BLE001
        radec_map = {}
    if not radec_map:
        radec_map = _load_radec_map(output_dir)
    cache_dir = output_dir / "_crossmatch"
    tess_dir = output_dir / "_tess"

    kat_col = None
    for c in df.columns:
        cl = str(c).strip().lower().replace("o", "o")
        if cl in ("katalogy", "katalog"):
            kat_col = str(c)
            break

    def _row_tess_eligible(row: Any) -> bool:
        if "vsx_known_variable" in df.columns:
            try:
                if bool(pd.to_numeric(row.get("vsx_known_variable"), errors="coerce")):
                    return False
            except Exception:  # noqa: BLE001
                pass
        if "vsx_match" in df.columns:
            try:
                if bool(pd.to_numeric(row.get("vsx_match"), errors="coerce")):
                    return False
            except Exception:  # noqa: BLE001
                pass
        if "gaia_dr3_variable_catalog" in df.columns:
            try:
                if bool(pd.to_numeric(row.get("gaia_dr3_variable_catalog"), errors="coerce")):
                    return False
            except Exception:  # noqa: BLE001
                pass
        if kat_col:
            txt = str(row.get(kat_col, "") or "")
            for line in txt.splitlines():
                s = line.strip()
                if not s or s.startswith("[telescope]"):
                    continue
                if "ziadny zaznam" in s.lower() or "no match" in s.lower():
                    continue
                return False
        return True

    for _, row in df.iterrows():
        cid = str(row.get("catalog_id") or "").strip()
        if not cid:
            continue
        if not _row_tess_eligible(row):
            LOGGER.info("[TESS] Skip %s - known variable or catalog match", cid)
            continue
        out_base = tess_dir / cid
        result_json = out_base / "result.json"
        result_txt = out_base / "result.txt"

        if result_json.exists():
            LOGGER.info("[TESS] Skip %s - result.json exists", cid)
            if (not result_txt.exists()) and result_json.exists():
                _write_tess_result_txt(result_json, result_txt, cid)
            continue

        radec = radec_map.get(cid)
        if radec is None:
            continue
        ra, dec = radec

        period_hint = None
        cache_path = cache_dir / f"{cid}.json"
        if cache_path.exists():
            period_hint = _period_hint_from_crossmatch_cache(cache_path)

        mag = None
        if "mag" in df.columns:
            try:
                mv = float(pd.to_numeric(row.get("mag"), errors="coerce"))
                mag = mv if math.isfinite(mv) else None
            except Exception:  # noqa: BLE001
                mag = None

        LOGGER.info("[TESS] Run %s (period_hint=%s)", cid, str(period_hint) if period_hint is not None else "-")
        try:
            run_tess_analysis(
                catalog_id=cid,
                ra=float(ra),
                dec=float(dec),
                mag=mag,
                photometry_dir=str(output_dir),
                period_hint=period_hint,
                cfg=cfg,
            )
            if result_json.exists():
                _write_tess_result_txt(result_json, result_txt, cid)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[TESS] FAILED %s: %s", cid, exc)

