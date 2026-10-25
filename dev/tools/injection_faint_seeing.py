# -*- coding: ascii -*-
"""Injection step D: amplitude of injected sines and seeing dependence of the light curves.

Usage: python injection_faint_seeing.py <repo> <plan.json> <injected processed-field dir>

For every synthetic star of the plan and both paths (aperture ``mag_calib``, ePSF ``psf_delta_mag``):
amplitude ratio of the injected sine (offset free), the same with a term linear in the frame FWHM
(aperture_r_px / its night median - 1: r = k x FWHM_frame), the slope of that term, rms with and
without it, and the median residual in the 20 % worst-seeing frames. Printed per magnitude.
"""
import sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1] + "/dev/tools")
from injection_recovery import Signal
plan = json.load(open(sys.argv[2])); L = glob.glob(sys.argv[3] + "/platesolve/*/photometry/lightcurves")[0]
rows = []
for st in plan["stars"]:
    sig = Signal(**st["signal"]); cid = st["catalog_id"]
    ap = pd.read_csv(f"{L}/lightcurve_{cid}.csv", comment="#")
    fw = ap.set_index("source_file")["aperture_r_px"]
    for meth in ("aperture", "epsf"):
        if meth == "epsf":
            lc = pd.read_csv(f"{L}/lightcurve_{cid}_psf.csv", comment="#"); lc = lc[lc.psf_fit_ok.astype(str).str.lower() == "true"]; m = lc.psf_delta_mag
        else:
            lc = ap; m = lc.mag_calib
        ok = np.isfinite(m.to_numpy(float)); lc = lc[ok]; m = m[ok].to_numpy(float); t = lc.jd.to_numpy(float)
        f = fw.reindex(lc.source_file).to_numpy(float); f = f / np.nanmedian(f) - 1.0
        s = sig.dm(t) / sig.amp if sig.kind == "sine" else np.zeros_like(t)
        A1 = np.c_[s, np.ones_like(t)]; A2 = np.c_[s, np.ones_like(t), f]
        if sig.kind != "sine": A1, A2 = A1[:, 1:], A2[:, 1:]
        c1 = np.linalg.lstsq(A1, m, rcond=None)[0]; c2 = np.linalg.lstsq(A2, m, rcond=None)[0]
        r = m - A1 @ c1; r2 = m - A2 @ c2
        late = f > np.percentile(f, 80)
        rows.append(dict(meth=meth, mag=st["mag"], sine=sig.kind == "sine", amp1=c1[0] / sig.amp if sig.kind == "sine" else np.nan,
                         amp2=c2[0] / sig.amp if sig.kind == "sine" else np.nan, slope=c2[-1] * 1000, rms=1000 * r.std(), rms2=1000 * r2.std(),
                         late=1000 * np.median(r[late])))
d = pd.DataFrame(rows)
for (meth, mag), g in d.groupby(["meth", "mag"]):
    s = g[g.sine]
    print(f"{meth:8s} G{mag:4.1f}  amp {s.amp1.mean():.3f}+-{s.amp1.std()/np.sqrt(len(s)):.3f} (spread {s.amp1.std():.2f})  with FWHM term {s.amp2.mean():.3f}+-{s.amp2.std()/np.sqrt(len(s)):.3f} (spread {s.amp2.std():.2f})"
          f"  | slope dmag per +100% FWHM: median {g.slope.median():+.0f} mmag (|.| {g.slope.abs().median():.0f})  rms {g.rms.median():.1f}->{g.rms2.median():.1f}  worst-seeing 20% offset {g.late.median():+.0f} mmag")
