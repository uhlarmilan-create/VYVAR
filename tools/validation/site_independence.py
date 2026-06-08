"""Validate the Phase-1 SITE fix: BJD/airmass site must come from per-draft
ID_LOCATION (or header), INDEPENDENT of config.json (the config-drift trap).

The Sydney config-independence guard: forcing config.json to a bogus site must leave
every draft's resolved lat/lon unchanged. Run from the repo root:
    python tools/validation/site_independence.py
"""
import json, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
from astropy.io import fits

from database import VyvarDatabase
from param_resolver import resolve_site
from time_utils import resolve_observer_location
from photometry_core import _recompute_bjd_hjd_per_target

cfgj = json.load(open("config.json"))

class Shim:
    observer_lat = float(cfgj.get("observer_lat", 0.0) or 0.0)
    observer_lon = float(cfgj.get("observer_lon", 0.0) or 0.0)
    observer_alt_m = float(cfgj.get("observer_alt_m", 0.0) or 0.0)
    observer_location_id = int(cfgj.get("observer_location_id", 0) or 0)
    observer_location_name = str(cfgj.get("observer_location_name", "") or "")

cfg = Shim()
print(f"config.json observer: id={cfg.observer_location_id} "
      f"lat={cfg.observer_lat:.4f} lon={cfg.observer_lon:.4f} ({cfg.observer_location_name})")

db = VyvarDatabase("vyvar.sqlite3")

def first_light_header(draft):
    pats = [f"Archive/Drafts/draft_{draft:06d}/Raw/lights/**/*.fit*"]
    for p in pats:
        files = sorted(glob.glob(p, recursive=True))
        if files:
            return fits.getheader(files[0]), files[0]
    return None, None

LOC = {r[0]: (r[1], float(r[2]), float(r[3])) for r in
       db.conn.execute("SELECT ID,PLACENAME,LATITUDE,LONGITUDE FROM LOCATION")}

print("\n%-6s %-10s %-8s %-9s %-9s %-22s %s" %
      ("draft","DB loc","src","lat","lon","name(DB ID_LOCATION)","matches draft?"))
print("-"*95)
jd = np.array([2461112.40, 2461112.45], dtype=float)
ra, dec = 210.0, 28.3

for draft in (360, 361, 362, 363):
    row = db.conn.execute("SELECT ID_LOCATION FROM OBS_DRAFT WHERE ID=?", (draft,)).fetchone()
    loc_id = row[0] if row else None
    locname, loclat, loclon = LOC.get(loc_id, ("?", float("nan"), float("nan")))
    hdr, path = first_light_header(draft)
    site = resolve_site(hdr, db=db, draft_id=draft, cfg=cfg)
    match = (site.lat is not None and abs(site.lat - loclat) < 1e-3
             and abs(site.lon - loclon) < 1e-3)
    print("%-6s %-10s %-8s %-9.4f %-9.4f %-22s %s" %
          (draft, f"{loc_id}:{locname}", site.source,
           site.lat or 0, site.lon or 0, f"{locname}", "YES" if match else "NO <<<"))

print("\n--- INDEPENDENCE: force config to a BOGUS location; drafts must be unaffected ---")
class Bogus:
    observer_lat = -33.8688   # Sydney
    observer_lon = 151.2093
    observer_alt_m = 58.0
    observer_location_id = 999
    observer_location_name = "BOGUS-Sydney"
bogus = Bogus()
for draft in (360, 363):
    hdr, _ = first_light_header(draft)
    site = resolve_site(hdr, db=db, draft_id=draft, cfg=bogus)
    print(f"  draft {draft}: config=BOGUS({bogus.observer_lat:.2f}) -> resolved src={site.source} "
          f"lat={site.lat:.4f} lon={site.lon:.4f}  (config IGNORED: {'OK' if site.lat>0 else 'LEAK'})")

print("\n--- Phase 2A BJD: per-draft resolved site vs config-only (the OLD bug) ---")
for draft in (360, 363):
    hdr, _ = first_light_header(draft)
    site = resolve_site(hdr, db=db, draft_id=draft, cfg=cfg)
    bjd_resolved, _ = _recompute_bjd_hjd_per_target(jd, ra, dec, cfg,
                          site=(site.lat, site.lon, site.elev))
    bjd_cfg, _ = _recompute_bjd_hjd_per_target(jd, ra, dec, cfg, site=None)  # legacy cfg-only
    d_ms = abs(bjd_resolved[0] - bjd_cfg[0]) * 86400.0 * 1000.0
    print(f"draft {draft}: site={site.source} ({site.lat:.4f},{site.lon:.4f})  "
          f"BJD_resolved-BJD_cfg = {d_ms:.3f} ms  (config={cfg.observer_lat:.4f},{cfg.observer_lon:.4f})")
