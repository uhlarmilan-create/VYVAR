"""Merge the vendor parts into src_py/camera_catalog.json (CAMERA-PROFILE-2026-10).

usage: python dev/tools/camera_catalog/merge_catalog.py dev/tools/camera_catalog/parts src_py/camera_catalog.json
The parts are the per-vendor research files (every value with its source URL)."""
import json, glob, re, sys
FAST = ("IMX174", "IMX249", "IMX250", "IMX252", "IMX253", "IMX264", "IMX265", "IMX273", "IMX287", "IMX304",
        "IMX428", "IMX429", "IMX432", "IMX290", "IMX462")
SCIENTIFIC_VENDORS = ("PCO", "Excelitas PCO", "Andor", "Teledyne Photometrics", "Hamamatsu")
LEGACY = {"qhy_qhy294m_pro": ["qhy294m_pro"], "qhy_qhy600m_ph": ["qhy600m"]}
parts = sorted(glob.glob(sys.argv[1] + "/*.json"))
allm = [m for f in parts for m in json.load(open(f))["models"]]

def richness(m):
    n = sum(len(mo.get("curve") or []) * 3 for mo in m["modes"])
    n += sum(1 for mo in m["modes"] for k, v in mo.items() if v not in (None, [], ""))
    return n

by = {}
for m in allm:
    k = m["key"]
    if k not in by or richness(m) > richness(by[k]):
        by[k] = m
keep, dropped = [], []
for m in by.values():
    sensor = str(m.get("sensor") or "").upper()
    name = (m["key"] + " " + m["model"]).lower()
    swir = "ingaas" in sensor or re.search(r"(^|_)99[012]", m["key"])
    guide = "guide" in name or "_oag" in name
    empty = not m.get("sources") or not any(mo.get("pixel_um") or mo.get("width") for mo in m["modes"])
    if swir or guide or empty:
        dropped.append(m["key"]); continue
    if m.get("cooled") is False:
        fast = m["color"] == "mono" and any(s in sensor for s in FAST)
        sci = m["vendor"] in SCIENTIFIC_VENDORS and m["color"] == "mono"
        if not (fast or sci):
            dropped.append(m["key"]); continue
    m = dict(m)
    if m["vendor"].startswith("FLI"):
        m["vendor"] = "FLI"
    if m["vendor"].upper().startswith("QHY"):
        # the QHY SDK names cameras e.g. QHY294PROM, QHY268PROC (INSTRUME "QHY CCD QHY294PROM-<serial>")
        mm = re.match(r"^(QHY\d+)([MC])\s*PRO\b", m["model"].upper())
        if mm:
            m["aliases"] = list(dict.fromkeys([*m["aliases"], f"{mm.group(1)}PRO{mm.group(2)}"]))
    if m["key"] in LEGACY:
        m["legacy_keys"] = LEGACY[m["key"]]
    keep.append(m)
keep.sort(key=lambda m: (m["vendor"].lower(), m["model"].lower()))
out = {
    "schema": 2,
    "version": "2026-10-23",
    "about": ("VYVAR camera catalog (CAMERA-PROFILE-2026-10): CMOS cameras used for photometry - cooled models "
              "of all major vendors plus uncooled fast mono models (global shutter / occultation sensors). "
              "Typical manufacturer values; every curve point names its source; values the vendor does not "
              "publish as numbers are left out. header_gain: what the FITS GAIN card means (e_per_adu = "
              "effective e-/ADU, setting = driver gain setting). binning (only when documented): average "
              "(gain x N^2, read noise x N), sum (gain x 1, read noise x N), charge (both x 1). e_per_adu is in "
              "native ADC units of the mode (adc_bits). Measured or manual points of your own camera always win."),
    "models": keep,
}
open(sys.argv[2], "w").write(json.dumps(out, indent=1, ensure_ascii=True) + "\n")
print("kept", len(keep), "dropped", len(dropped))
import collections
print(collections.Counter(m["vendor"] for m in keep).most_common())
print("with curve", sum(1 for m in keep if any(mo.get("curve") or mo.get("e_per_adu") or mo.get("read_noise_e") for mo in m["modes"])))
print("with e/ADU", sum(1 for m in keep if any(mo.get("e_per_adu") or any(p.get("e_per_adu") for p in (mo.get("curve") or [])) for mo in m["modes"])))
