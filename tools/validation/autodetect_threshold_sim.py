"""Re-run 360/363 optics auto-detect under the PROPOSED bands/weights and report the
confidence scores + auto-fill / pre-fill / prompt decisions. Run from the repo root:
    python tools/validation/autodetect_threshold_sim.py
"""
import warnings
warnings.filterwarnings("ignore")
from database import VyvarDatabase, get_observer_locations
from optics_autodetect import (
    autodetect_from_source, AUTOFILL_THRESHOLD, PREFILL_THRESHOLD,
)

db = VyvarDatabase("vyvar.sqlite3")
act = db.sql_expr_active_is_true("ACTIVE")
eqs = [dict(r) for r in db.conn.execute(
    f"SELECT ID,CAMERANAME,ALIAS,SENSORTYPE,SENSORSIZE,PIXELSIZE,GAIN_ADU FROM EQUIPMENTS WHERE {act}")]
tels = [dict(r) for r in db.conn.execute(
    f"SELECT ID,TELESCOPENAME,ALIAS,DIAMETER,FOCAL FROM TELESCOPE WHERE {act}")]
locs = get_observer_locations("vyvar.sqlite3", active_only=True)

print(f"bands: high >= {AUTOFILL_THRESHOLD:.2f} (auto-fill) | "
      f"medium >= {PREFILL_THRESHOLD:.2f} (pre-fill+verify) | low < {PREFILL_THRESHOLD:.2f} (prompt)\n")


def decision(d):
    if d.matched_id is None or d.band() == "none":
        return "PROMPT (no/low match)"
    if d.autofill:
        return f"AUTO-FILL id={d.matched_id}"
    return f"PRE-FILL id={d.matched_id} (UNCONFIRMED — verify)"


for draft in (360, 363):
    rep = autodetect_from_source(
        f"Archive/Drafts/draft_{draft:06d}/Raw/lights",
        equipments=eqs, telescopes=tels, locations=locs)
    print(f"=== draft {draft} ===")
    for nm, det in (("Equipment", rep.equipment), ("Telescope", rep.telescope), ("Location", rep.location)):
        print(f"  {nm:9s} conf={det.confidence:.2f} [{det.band():6s}] -> {decision(det)}")
        if det.reasons:
            print(f"            evidence: {'; '.join(det.reasons)}")
    print(f"  poor-FITS gaps: {[g['field'] for g in rep.unresolved]}\n")
