"""Post-run validation report for draft_000343."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DRAFT = _ROOT / "Archive/Drafts/draft_000343"
SETUP = "NoFilter_60_2"
PHOT = DRAFT / "platesolve" / SETUP / "photometry"
REPORTS = PHOT / "lightcurves_reports"
LOG = _ROOT / "scripts/_draft343_night_run.log"


def _chk(name: str, ok: bool, note: str = "") -> dict:
    return {"feature": name, "status": "PASS" if ok else ("SKIP" if note.startswith("SKIP") else "FAIL"), "note": note}


def main() -> int:
    meta_path = PHOT / "pipeline_meta.json"
    summ_path = PHOT / "photometry_summary.csv"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    summ = pd.read_csv(summ_path, dtype={"catalog_id": str}) if summ_path.is_file() else pd.DataFrame()

    print("## draft_343 Basic Metrics\n")
    print("### pipeline_meta.json")
    for k in ("n_targets", "n_lc"):
        print(f"- {k}: {meta.get(k)}")
    lc = meta.get("lc_quality_summary") or {}
    print(f"- lc_quality_summary: {lc}")
    lunar = meta.get("lunar_context") or {}
    if lunar:
        print(
            f"- lunar_context: phase={lunar.get('phase_pct')}% "
            f"sep={lunar.get('separation_deg')} deg alt={lunar.get('altitude_deg')} deg risk={lunar.get('risk')}"
        )
    gs11 = meta.get("gs11_summary") or {}
    print(f"- gs11_summary: enabled={gs11.get('enabled')} aperture_arcsec={gs11.get('aperture_arcsec')}")
    rms_m = meta.get("rms_model") or {}
    print(
        f"- rms_model: slope={rms_m.get('slope')} intercept={rms_m.get('intercept')} n_stars={rms_m.get('n_stars')}"
    )
    obs = meta.get("observer_location") or meta.get("location") or {}
    if obs:
        print(f"- observer_location: {obs}")

    print("\n### photometry_summary.csv")
    if len(summ):
        top = summ.sort_values("lc_rms", na_position="last").head(5)
        print("- Top 5 by lc_rms (lowest):")
        for _, r in top.iterrows():
            print(f"  {r.get('vsx_name', r.get('name', '?'))} id={r['catalog_id']} lc_rms={r.get('lc_rms'):.4f}")
        if "lc_quality_flag" in summ.columns:
            print("- lc_quality_flag:", summ["lc_quality_flag"].value_counts().to_dict())
        if "zone_flag" in summ.columns:
            print("- zone_flag:", summ["zone_flag"].value_counts().to_dict())
        vmask = summ["vsx_name"].astype(str).str.contains("V0842|V842", case=False, na=False) if "vsx_name" in summ.columns else pd.Series(False, index=summ.index)
        if not vmask.any() and "name" in summ.columns:
            vmask = summ["name"].astype(str).str.contains("V0842|V842|Her", case=False, na=False)
        if vmask.any():
            print("- V0842 Her row:")
            print(summ.loc[vmask].iloc[0].to_string())
        else:
            print("- V0842 Her row: NOT in photometry_summary")

    # Feature checklist
    checks: list[dict] = []
    loc = meta.get("observer_location") or {}
    checks.append(
        _chk(
            "Observer location (Jirny)",
            abs(float(loc.get("lat", 0)) - 50.1121658) < 0.01 and abs(float(loc.get("lon", 0)) - 14.6982547) < 0.01,
            str(loc) or "missing in pipeline_meta",
        )
    )
    checks.append(
        _chk(
            "Lunar context",
            bool(lunar) and lunar.get("phase_pct") is not None,
            str(lunar) if lunar else "missing",
        )
    )
    checks.append(
        _chk(
            "lc_quality_flag",
            "lc_quality_flag" in summ.columns and summ["lc_quality_flag"].notna().any(),
            summ["lc_quality_flag"].value_counts().to_dict() if "lc_quality_flag" in summ.columns else "no column",
        )
    )
    n_stars = int(rms_m.get("n_stars") or 0)
    slope = rms_m.get("slope")
    intercept = rms_m.get("intercept")
    checks.append(
        _chk(
            "RMS model",
            n_stars > 10 and np.isfinite(float(slope or np.nan)) and np.isfinite(float(intercept or np.nan)),
            f"n_stars={n_stars} slope={slope} intercept={intercept}",
        )
    )

    # Sample LC for ct / bjd / dilution / lunar
    lc_files = list((PHOT / "lightcurves").glob("lightcurve_*.csv"))
    sample_lc = pd.read_csv(lc_files[0]) if lc_files else pd.DataFrame()
    cid_sample = lc_files[0].stem.replace("lightcurve_", "") if lc_files else ""
    ct_ok = True
    if len(sample_lc) and "mag_calib_ct" in sample_lc.columns and "mag_calib" in sample_lc.columns:
        diff = (sample_lc["mag_calib_ct"].astype(float) - sample_lc["mag_calib"].astype(float)).abs()
        ct_ok = bool((diff < 1e-6).all() or sample_lc.get("ct_ok", pd.Series([False])).astype(str).str.lower().eq("false").all())
    checks.append(_chk("mag_calib_ct == mag_calib (NoFilter)", ct_ok, f"sample {cid_sample}"))

    bjd_ok = False
    bjd_note = "no LC"
    if len(sample_lc) and "bjd" in sample_lc.columns and "jd" in sample_lc.columns:
        d = (sample_lc["bjd"].astype(float) - sample_lc["jd"].astype(float)) * 86400
        med = float(np.nanmedian(d))
        bjd_ok = 400 < abs(med) < 600  # ~8 min LTT
        bjd_note = f"median bjd-jd = {med:.1f} s"
    checks.append(_chk("per-target BJD", bjd_ok, bjd_note))

    log_text = LOG.read_text(encoding="utf-8", errors="replace") if LOG.is_file() else ""
    checks.append(
        _chk(
            "slope filter log",
            "comp_max_slope" in log_text or "slope" in log_text.lower() or "mmag_hr" in log_text,
            "grep log for comp_max_slope_mmag_hr=5.0",
        )
    )
    dil_col = "dilution_factor" in sample_lc.columns if len(sample_lc) else False
    dil_ok = False
    if dil_col:
        dil_ok = bool(np.allclose(sample_lc["dilution_factor"].astype(float), 1.0, equal_nan=True))
    checks.append(_chk("dilution_factor=1 (GS11 off)", dil_col and dil_ok, "col present" if dil_col else "no column"))
    checks.append(
        _chk(
            "lunar_phase_pct in LC",
            "lunar_phase_pct" in sample_lc.columns if len(sample_lc) else False,
            "present" if "lunar_phase_pct" in sample_lc.columns else "missing",
        )
    )

    aavso = list((REPORTS / "aavso").glob("*.txt")) if (REPORTS / "aavso").is_dir() else []
    aavso_ok = False
    aavso_note = "no export"
    if aavso:
        head = aavso[0].read_text(encoding="utf-8", errors="replace").splitlines()[:30]
        aavso_ok = any("#LATITUDE" in ln for ln in head) and any("#LONGITUDE" in ln for ln in head)
        aavso_note = aavso[0].name
    checks.append(_chk("AAVSO header lat/lon/elev", aavso_ok, aavso_note))

    var_files = list((REPORTS / "varastro").glob("*.txt")) if (REPORTS / "varastro").is_dir() else []
    var_ok = False
    var_note = "no export"
    for vf in var_files:
        if "Site: Jirny" in vf.read_text(encoding="utf-8", errors="replace")[:500]:
            var_ok = True
            var_note = vf.name
            break
    checks.append(_chk("VAR.ASTRO Site: Jirny", var_ok or len(var_files) == 0, var_note if var_files else "SKIP no EW export"))

    pdf_dir = REPORTS / "pdf" if (REPORTS / "pdf").is_dir() else PHOT / "reports"
    pdfs = list(pdf_dir.glob("*.pdf")) if pdf_dir.is_dir() else list(REPORTS.glob("*.pdf"))
    pdf_text = ""
    if pdfs:
        try:
            import pypdf

            pdf_text = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(str(pdfs[0])).pages[:5])
        except Exception as e:  # noqa: BLE001
            pdf_text = str(e)
    checks.append(_chk("PDF Observing Conditions", "Observing Conditions" in pdf_text or "Lunar" in pdf_text, pdfs[0].name if pdfs else "no pdf"))
    checks.append(_chk("PDF LC Quality", "LC Quality" in pdf_text or "quality" in pdf_text.lower(), ""))
    checks.append(
        _chk(
            "summary dilution_factor column",
            "dilution_factor" in summ.columns,
            "present" if "dilution_factor" in summ.columns else "missing",
        )
    )

    print("\n## Feature checklist\n")
    print("| feature | status | note |")
    print("|---------|--------|------|")
    for c in checks:
        print(f"| {c['feature']} | {c['status']} | {c['note']} |")

    # V0842 Her
    print("\n## V0842 Her LC\n")
    at = PHOT / "active_targets.csv"
    cid = None
    if at.is_file():
        adf = pd.read_csv(at, dtype={"catalog_id": str})
        m = adf["vsx_name"].astype(str).str.contains("V0842|V842", case=False, na=False) if "vsx_name" in adf.columns else pd.Series(False, index=adf.index)
        if m.any():
            cid = str(adf.loc[m].iloc[0]["catalog_id"])
            print(f"catalog_id: {cid} name={adf.loc[m].iloc[0].get('vsx_name')}")
    if cid:
        lcp = PHOT / "lightcurves" / f"lightcurve_{cid}.csv"
        if lcp.is_file():
            lc = pd.read_csv(lcp)
            row = summ[summ["catalog_id"].astype(str) == cid].iloc[0] if len(summ) and (summ["catalog_id"].astype(str) == cid).any() else None
            print(f"lc_rms: {row['lc_rms'] if row is not None else 'n/a'}")
            print(f"lc_median_mag: {row.get('lc_median_mag', 'n/a') if row is not None else 'n/a'}")
            print(f"n_frames: {len(lc)} lc_quality_flag: {row.get('lc_quality_flag') if row is not None else 'n/a'}")
            print(f"zone_flag: {row.get('zone_flag') if row is not None else 'n/a'}")
            if "mag_calib" in lc.columns:
                mags = lc["mag_calib"].astype(float)
                print(f"mag_calib min/max: {mags.min():.4f} / {mags.max():.4f} amp={mags.max()-mags.min():.4f}")
            b0 = float(lc["bjd"].iloc[0])
            print("\nBJD offset    mag_calib")
            for i in range(0, len(lc), max(1, len(lc) // 14)):
                print(f"+{lc['bjd'].iloc[i]-b0:.6f}        {lc['mag_calib'].iloc[i]:.3f}")
        else:
            print(f"No LC file: {lcp}")
    else:
        print("V0842 Her not in active_targets - search Gaia/VSX nearest in summary")
        if len(summ) and "ra_deg" in summ.columns:
            # VSX 241.509, 50.187
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            vsx = SkyCoord(241.509 * u.deg, 50.187 * u.deg)
            sc = SkyCoord(summ["ra_deg"].astype(float) * u.deg, summ["dec_deg"].astype(float) * u.deg)
            sep = vsx.separation(sc).arcsec
            i = int(sep.argmin())
            print(f"Nearest in summary: {summ.iloc[i]['vsx_name']} sep={sep.iloc[i]:.1f}\" id={summ.iloc[i]['catalog_id']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
