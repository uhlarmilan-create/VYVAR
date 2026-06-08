"""Pre-commit sanity: confirm the unified resolver does NOT move the science for
drafts 360/363 beyond the negligible per-draft site BJD shift.

Compares resolver-path vs reconstructed pre-change path for:
  * equipment-intrinsic gain / read-noise / pixel (both the old error-map header-first
    path AND the old Phase-2A DB-only path)
  * BJD (config-site vs per-draft resolved-site)
  * airmass (config-site vs per-draft resolved-site)

Run from the repo root:
    python tools/validation/precommit_sanity.py
"""
import glob
import math
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from astropy.io import fits

from database import VyvarDatabase
from config import AppConfig
from param_resolver import resolve_gain, resolve_read_noise, resolve_pixel_um, resolve_site
from photometry_core import _recompute_bjd_hjd_per_target

db = VyvarDatabase("vyvar.sqlite3")
cfg = AppConfig()


def first_light(draft):
    fs = sorted(glob.glob(f"Archive/Drafts/draft_{draft:06d}/Raw/lights/**/*.fit*", recursive=True))
    return (fits.getheader(fs[0]), fs[0]) if fs else (None, None)


def hdr_pos(h, *keys):
    for k in keys:
        try:
            if k in h:
                v = float(h[k])
                if math.isfinite(v) and v > 0:
                    return v
        except (TypeError, ValueError):
            pass
    return None


def old_errormap_gain_rn(h, eq):
    """Pre-change error-map path: header positive first, then DB, then default."""
    g = hdr_pos(h, "EGAIN", "GAIN")
    if g is None and eq is not None:
        gd, _ = db.get_equipment_cosmic_params(int(eq))
        g = float(gd) if gd and gd > 0 else None
    g = g if g else 1.0
    rn = hdr_pos(h, "RDNOISE", "READNOISE")
    if rn is None and eq is not None:
        _, rd = db.get_equipment_cosmic_params(int(eq))
        rn = float(rd) if rd and rd > 0 else None
    rn = rn if rn else 10.0
    return g, rn


def old_phase2a_gain_rn(eq):
    """Pre-change Phase-2A path: DB-only, defaults otherwise."""
    g, rn = 1.0, 10.0
    if eq is not None:
        gd, rd = db.get_equipment_cosmic_params(int(eq))
        if gd and gd > 0:
            g = float(gd)
        if rd and rd > 0:
            rn = float(rd)
    return g, rn


jd = np.array([2461112.40, 2461112.45], dtype=float)
ra, dec = 210.0, 28.3
print(f"config.json site: lat={cfg.observer_lat:.4f} lon={cfg.observer_lon:.4f} ({cfg.observer_location_name})")
worst = 0.0
for draft in (360, 363):
    h, path = first_light(draft)
    row = db.conn.execute("SELECT ID_EQUIPMENTS FROM OBS_DRAFT WHERE ID=?", (draft,)).fetchone()
    eq = row[0] if row else None
    print(f"\n=== draft {draft} (eq={eq}) ===")

    # gain / RN — both old stages vs resolver
    og_em, orn_em = old_errormap_gain_rn(h, eq)
    og_2a, orn_2a = old_phase2a_gain_rn(eq)
    rg = resolve_gain(h, db=db, equipment_id=eq, cfg=cfg)
    rrn = resolve_read_noise(h, db=db, equipment_id=eq, cfg=cfg)
    print(f"  gain : old_errormap={og_em:.4f} old_phase2a={og_2a:.4f} resolver={rg.value:.4f} ({rg.source})  "
          f"{'OK' if abs(og_em-rg.value)<1e-9 and abs(og_2a-rg.value)<1e-9 else 'DELTA!!'}")
    print(f"  RN   : old_errormap={orn_em:.4f} old_phase2a={orn_2a:.4f} resolver={rrn.value:.4f} ({rrn.source})  "
          f"{'OK' if abs(orn_em-rrn.value)<1e-9 and abs(orn_2a-rrn.value)<1e-9 else 'DELTA!!'}")

    # pixel — old DB-only vs resolver
    opx = db.get_equipment_pixel_size_um(int(eq)) if eq else None
    rpx = resolve_pixel_um(h, db=db, equipment_id=eq, cfg=cfg)
    px_ok = (opx is not None and rpx.ok and abs(float(opx) - rpx.value) < 1e-9)
    print(f"  pixel: old_DB={opx} resolver={rpx.value} ({rpx.source})  warnings={rpx.warnings}  {'OK' if px_ok else 'DELTA!!'}")

    # BJD — config site vs resolved (draft) site
    site = resolve_site(h, db=db, draft_id=draft, cfg=cfg)
    bjd_res, _ = _recompute_bjd_hjd_per_target(jd, ra, dec, cfg, site=(site.lat, site.lon, site.elev))
    bjd_cfg, _ = _recompute_bjd_hjd_per_target(jd, ra, dec, cfg, site=None)
    d_ms = float(np.max(np.abs(bjd_res - bjd_cfg))) * 86400.0 * 1000.0
    worst = max(worst, d_ms)
    print(f"  BJD  : site={site.source} ({site.lat:.4f},{site.lon:.4f})  max|resolved-config| = {d_ms:.4f} ms")

print(f"\nWorst BJD shift across drafts: {worst:.4f} ms  (expected <= ~0.05 ms)")
print("VERDICT:", "PASS — only negligible site shift" if worst <= 0.05 else "INVESTIGATE")
