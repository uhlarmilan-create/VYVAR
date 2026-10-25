# -*- coding: ascii -*-
"""INJECTION-RECOVERY-2026-10: known light-curve signals injected into real raw frames.

Method (step A, "signal on a host star"): pick isolated, unsaturated Gaia stars of a finished
night (the hosts) and multiply each host's own image in every raw LIGHT by a known factor
(1 + k(t)), k(t) = 10**(-0.4 dm(t)) - 1. The host's own image carries the frame's real PSF, flat
and position, so no PSF model is involved (the test cannot measure a model with itself). Both
nights - the control (untouched raw) and the injected one - run through the normal night pipeline
with the hosts added as targets; the difference of the two light curves of a host is the signal
VYVAR recovered, to be compared with dm(t).

Subcommands (run them in this order):
  plan     control draft -> plan.json (hosts, signals; seeded)
  inject   raw LIGHT dir + plan + control draft + master dark -> injected raw LIGHT dir
  targets  add the hosts to a draft's variable_targets.csv (catalog INJECT) before a photometry rerun
  compare  control draft + injected draft + plan -> results.csv, summary.md, plots

Step B ("new stars"): synthetic stars at positions where Gaia has none - in empty sky (error-bar
check: pulls; detection limit) and next to isolated real stars (blend bias; leakage of the
neighbour's variability). A star is a real star of the same frame (a template: dark- and
sky-subtracted, isolated, bright) shifted to the new position and scaled, plus photon noise.
VYVAR measures catalogue stars only, so the synthetic stars are listed in a CSV that the night
run reads through the test hook ``$VYVAR_INJECT_CATALOG`` (pipeline_catalog).
  plan-new    control draft -> plan.json + inject_catalog.csv
  inject-new  raw LIGHT dir + plan + control draft + master dark -> injected raw LIGHT dir
  targets     (as above; adds the synthetic stars)
  compare-new injected draft + plan -> results.csv, summary.md, plots

The raw frames and the drafts are never touched in place except ``targets`` on a draft copy.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

TOOL = "injection_recovery v1"
INJECT_CATALOG = "INJECT"

# --------------------------------------------------------------------------- signals


@dataclass
class Signal:
    """dm(t) in magnitudes (positive = fainter) at UTC JD ``t``."""

    kind: str  # "null" | "sine" | "eclipse" | "step" | "rise" | "flare"
    amp: float = 0.0  # sine: semi-amplitude; eclipse: depth (mag); rise: magnitudes gained
    period_d: float = 0.0  # sine
    t0: float = 0.0  # sine: zero phase; eclipse: time of minimum; step / rise / flare: onset
    width_d: float = 0.0  # eclipse: Gaussian sigma; rise: duration; flare: e-folding time (days)

    def dm(self, t: Any) -> np.ndarray:
        """TRANSIENT-2026-10: step / rise / flare are absent (dm = OFF_MAG) before ``t0``; then
        step: constant; rise: amp magnitudes fainter at t0, linear in magnitude to 0 at t0 + width_d
        (a nova's final rise); flare: exponential decay of the flux with e-folding ``width_d``."""
        t = np.asarray(t, dtype=np.float64)
        if self.kind == "sine":
            return self.amp * np.sin(2.0 * np.pi * (t - self.t0) / self.period_d)
        if self.kind == "eclipse":
            return self.amp * np.exp(-0.5 * ((t - self.t0) / self.width_d) ** 2)
        if self.kind in ("step", "rise", "flare"):
            on = t >= self.t0
            dt = np.clip(t - self.t0, 0.0, None)
            if self.kind == "step":
                v = np.zeros_like(t)
            elif self.kind == "rise":
                v = self.amp * np.clip(1.0 - dt / self.width_d, 0.0, None)
            else:
                v = 1.0857362 * dt / self.width_d
            return np.where(on, np.minimum(v, OFF_MAG), OFF_MAG)
        return np.zeros_like(t)


#: TRANSIENT-2026-10: "absent" for the transient signals (flux factor 1e-40)
OFF_MAG = 100.0


def flux_factor(dm: Any) -> np.ndarray:
    """(1 + k) for a magnitude change ``dm``."""
    return np.power(10.0, -0.4 * np.asarray(dm, dtype=np.float64))


SIGNAL_SET: tuple[tuple[str, float], ...] = (
    ("null", 0.0),
    ("sine", 0.01),
    ("sine", 0.05),
    ("sine", 0.20),
    ("eclipse", 0.05),
    ("eclipse", 0.30),
)


# --------------------------------------------------------------------------- helpers


def _ps_dir(draft: Path) -> Path:
    """The single platesolve/<setup> directory of a draft."""
    dirs = [p for p in (Path(draft) / "platesolve").iterdir() if (p / "MASTERSTAR.fits").is_file()]
    if len(dirs) != 1:
        raise SystemExit(f"expected one platesolve setup with MASTERSTAR.fits in {draft}, found {len(dirs)}")
    return dirs[0]


def _aligned_dir(draft: Path, setup: str) -> Path:
    """Aligned frames of the control night: their headers carry the alignment matrix (VY_AM0..8)."""
    d = Path(draft) / "detrended_aligned" / "lights" / setup
    if not any(d.glob("*.fits")):
        # without the matrices every frame would be copied untouched and the test would run on
        # uninjected data (happened once on 534 after the control frames were deleted for space)
        raise SystemExit(f"no aligned frames of the control night in {d}")
    return d


def _frame_jd(hdr: fits.Header) -> float:
    """UTC JD at mid-exposure (DATE-OBS = shutter open)."""
    from astropy.time import Time

    t = Time(str(hdr["DATE-OBS"]), format="isot", scale="utc")
    return float(t.jd + float(hdr.get("EXPTIME", 0.0)) / 2.0 / 86400.0)


def _align_matrix(hdr: fits.Header) -> np.ndarray | None:
    """Frame -> reference 3x3 matrix stamped by the alignment (VY_AM0..8)."""
    try:
        m = np.array([float(hdr[f"VY_AM{i}"]) for i in range(9)], dtype=np.float64).reshape(3, 3)
    except KeyError:
        return None
    return m if np.all(np.isfinite(m)) else None


def _isolated(ms: pd.DataFrame, field_xy: np.ndarray, field_mag: np.ndarray, radius_px: float, dmag: float) -> np.ndarray:
    """True where no catalogue star brighter than mag + ``dmag`` lies within ``radius_px`` (other than the star)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(field_xy)
    out = np.ones(len(ms), dtype=bool)
    for i, (x, y, m) in enumerate(zip(ms["x"].to_numpy(), ms["y"].to_numpy(), ms["mag"].to_numpy())):
        for j in tree.query_ball_point([x, y], radius_px):
            d = math.hypot(field_xy[j, 0] - x, field_xy[j, 1] - y)
            if d > 0.5 and field_mag[j] < m + dmag:
                out[i] = False
                break
    return out


# --------------------------------------------------------------------------- plan


def cmd_plan(a: argparse.Namespace) -> None:
    from astropy.wcs import WCS

    draft = Path(a.control_draft)
    ps = _ps_dir(draft)
    ms_hdr = fits.getheader(ps / "MASTERSTAR.fits")
    fwhm = float(ms_hdr.get("VY_FWHM") or 2.5)
    w, h = int(ms_hdr["NAXIS1"]), int(ms_hdr["NAXIS2"])
    ms = pd.read_csv(ps / "masterstars_full_match.csv", low_memory=False, dtype={"catalog_id": str, "name": str})
    ms["cid"] = ms["name"].astype(str).str.strip()
    flux = pd.to_numeric(ms["flux"], errors="coerce")
    ok = (
        np.isfinite(flux)
        & (ms["source_state"].astype(str).str.lower() != "catalog_membership")
        & (ms["zone"].astype(str).str.lower() == "linear")
        & (pd.to_numeric(ms["snr_ap_pixscaled"], errors="coerce") >= a.snr_min)
        & (pd.to_numeric(ms["peak_max_adu"], errors="coerce") < a.peak_frac * pd.to_numeric(ms["saturate_limit_adu"], errors="coerce"))
        & ms["x"].between(a.edge_px, w - a.edge_px)
        & ms["y"].between(a.edge_px, h - a.edge_px)
        & ms["cid"].str.fullmatch(r"\d{6,}")
        & ~ms["phot_variable_flag"].astype(str).str.upper().eq("VARIABLE")
        & ~ms["non_single_star"].fillna(0).astype(float).astype(bool)
    )
    vt = pd.read_csv(ps / "variable_targets.csv", low_memory=False, dtype={"catalog_id": str})
    ok &= ~ms["cid"].isin(set(vt["catalog_id"].astype(str)))
    cand = ms.loc[ok].copy()
    fc = pd.read_csv(ps / "field_catalog_cone.csv", low_memory=False)
    wcs = WCS(ms_hdr)
    fx, fy = wcs.all_world2pix(fc["ra_deg"].to_numpy(), fc["dec_deg"].to_numpy(), 0)
    cand = cand.loc[_isolated(cand, np.c_[fx, fy], fc["mag"].to_numpy(), a.iso_fwhm * fwhm, a.iso_dmag)]
    if len(cand) < len(SIGNAL_SET):
        raise SystemExit(f"only {len(cand)} isolated hosts - loosen the selection")
    rng = np.random.default_rng(a.seed)
    cand = cand.sort_values("mag").reset_index(drop=True)
    bins = np.array_split(cand.index.to_numpy(), a.bins)
    jds = sorted(_frame_jd(fits.getheader(p)) for p in sorted(Path(a.raw_dir).glob("*.fit*")))
    t_lo, t_hi = jds[0], jds[-1]
    span = t_hi - t_lo
    hosts: list[dict[str, Any]] = []
    for b, idx in enumerate(bins):
        pick = rng.choice(idx, size=min(len(SIGNAL_SET), len(idx)), replace=False)
        for (kind, amp), i in zip(SIGNAL_SET, pick):
            r = cand.loc[i]
            if kind == "sine":
                sig = Signal("sine", amp, period_d=0.6 * span, t0=t_lo + rng.uniform(0, 0.6 * span))
            elif kind == "eclipse":
                sig = Signal("eclipse", amp, t0=t_lo + rng.uniform(0.3, 0.7) * span, width_d=span / 12.0)
            else:
                sig = Signal("null")
            hosts.append({
                "catalog_id": str(r["cid"]), "ra_deg": float(r["ra_deg"]), "dec_deg": float(r["dec_deg"]),
                "x_ref": float(r["x"]), "y_ref": float(r["y"]), "mag": float(r["mag"]),
                "bp_rp": float(pd.to_numeric(r.get("bp_rp"), errors="coerce")), "mag_bin": b, "signal": asdict(sig),
            })
    plan = {
        "tool": TOOL, "seed": a.seed, "control_draft": str(draft), "setup": ps.name, "fwhm_px": fwhm,
        "stamp_r_fwhm": a.stamp_r_fwhm, "jd_first": t_lo, "jd_last": t_hi, "hosts": hosts,
    }
    Path(a.out).write_text(json.dumps(plan, indent=2), encoding="ascii")
    print(f"plan: {len(hosts)} hosts in {a.bins} magnitude bins ({cand['mag'].min():.2f}..{cand['mag'].max():.2f}) -> {a.out}")


# --------------------------------------------------------------------------- inject


def _dark_for(raw_shape: tuple[int, int], dark_path: Path, mode: str) -> np.ndarray:
    d = np.asarray(fits.getdata(dark_path), dtype=np.float64)
    if d.shape == raw_shape:
        return d
    fy, fx = d.shape[0] // raw_shape[0], d.shape[1] // raw_shape[1]
    if fy != fx or fy < 1:
        raise SystemExit(f"dark {d.shape} does not bin to {raw_shape}")
    d = d[: raw_shape[0] * fy, : raw_shape[1] * fx].reshape(raw_shape[0], fy, raw_shape[1], fx)
    return d.sum(axis=(1, 3)) if mode == "sum" else d.mean(axis=(1, 3))


def inject_into_frame(raw: np.ndarray, dark: np.ndarray, x: float, y: float, factor: float, r_px: float) -> tuple[np.ndarray, float]:
    """Raw frame with the star at native (x, y) scaled by ``factor``; returns (frame, added ADU).

    The star = raw - dark - local sky (median of an annulus 2..3 x r) inside a circle of ``r_px``;
    ``(factor - 1) * star`` is added (flat and PSF of that frame are in the star itself)."""
    h, w = raw.shape
    R = int(math.ceil(3.0 * r_px)) + 1
    x0, x1 = max(0, int(x) - R), min(w, int(x) + R + 1)
    y0, y1 = max(0, int(y) - R), min(h, int(y) + R + 1)
    sub = raw[y0:y1, x0:x1].astype(np.float64) - dark[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - x, yy - y)
    sky = float(np.median(sub[(rr >= 2.0 * r_px) & (rr <= 3.0 * r_px)]))
    star = np.where(rr <= r_px, sub - sky, 0.0)
    add = (float(factor) - 1.0) * star
    out = raw.astype(np.float64)
    out[y0:y1, x0:x1] += add
    return out, float(add.sum())


def cmd_inject(a: argparse.Namespace) -> None:
    plan = json.loads(Path(a.plan).read_text(encoding="ascii"))
    aligned = _aligned_dir(Path(plan["control_draft"]), plan["setup"])
    r_px = float(plan["stamp_r_fwhm"]) * float(plan["fwhm_px"])
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    dark = None
    log: list[dict[str, Any]] = []
    for fp in sorted(Path(a.raw_dir).glob("*.fit*")):
        with fits.open(fp) as hd:
            hdr = hd[0].header.copy()
            raw = np.asarray(hd[0].data)
        al = aligned / f"{fp.stem}.fits"
        m = _align_matrix(fits.getheader(al)) if al.is_file() else None
        if m is None:
            # not aligned in the control run -> not measured there either; copied unchanged
            fits.writeto(out_dir / fp.name, raw, hdr, output_verify="silentfix")
            log.append({"file": fp.name, "status": "not_aligned_copied"})
            continue
        if dark is None:
            dark = _dark_for(raw.shape, Path(a.dark), a.dark_binning)
        jd = _frame_jd(hdr)
        minv = np.linalg.inv(m)
        img = raw.astype(np.float64)
        for host in plan["hosts"]:
            sig = Signal(**host["signal"])
            if sig.kind == "null":
                continue
            fac = float(flux_factor(sig.dm(jd)))
            xn, yn, _ = minv @ np.array([host["x_ref"], host["y_ref"], 1.0])
            img, added = inject_into_frame(img, dark, float(xn), float(yn), fac, r_px)
            log.append({"file": fp.name, "catalog_id": host["catalog_id"], "jd": jd, "x": xn, "y": yn,
                        "factor": fac, "added_adu": added})
        info = np.iinfo(raw.dtype) if np.issubdtype(raw.dtype, np.integer) else None
        if info is not None:
            img = np.clip(np.rint(img), info.min, info.max).astype(raw.dtype)
        else:
            img = img.astype(raw.dtype)
        hdr["VYINJECT"] = (TOOL, "injection-recovery test frame")
        fits.writeto(out_dir / fp.name, img, hdr, output_verify="silentfix")
    pd.DataFrame(log).to_csv(out_dir.parent / f"{out_dir.name}_inject_log.csv", index=False)
    print(f"injected {sum(1 for r in log if 'factor' in r)} star-frames into {out_dir}")


# --------------------------------------------------------------------------- targets


def cmd_targets(a: argparse.Namespace) -> None:
    plan = json.loads(Path(a.plan).read_text(encoding="ascii"))
    vt_path = _ps_dir(Path(a.draft)) / "variable_targets.csv"
    vt = pd.read_csv(vt_path, low_memory=False, dtype={"catalog_id": str})
    vt = vt.loc[vt.get("target_origin", pd.Series("", index=vt.index)).astype(str) != INJECT_CATALOG]
    # --as-vsx: listed like a VSX target (RUN ePSF fits VSX targets only; epsf_science_set)
    cat = "VSX" if a.as_vsx else INJECT_CATALOG
    rows = [{
        "name": f"INJ {h['signal']['kind']} {h['signal']['amp']:g} {h['catalog_id']}",
        "catalog_id": h["catalog_id"], "catalog": cat, "ra_deg": h["ra_deg"], "dec_deg": h["dec_deg"],
        "x": h["x_ref"], "y": h["y_ref"], "mag": h["mag"], "target_origin": INJECT_CATALOG,
        **({"vsx_name": f"INJ {h['catalog_id']}", "vsx_type": "", "gaia_match_source": "masterstars"} if a.as_vsx else {}),
    } for h in plan.get("stars", plan["hosts"])]
    new = pd.DataFrame(rows)
    if "catalog_id" in vt.columns:  # a repeated call replaces its own rows instead of doubling them
        vt = vt.loc[~vt["catalog_id"].astype(str).isin(set(new["catalog_id"].astype(str)))]
    out = pd.concat([vt, new], ignore_index=True)
    out.to_csv(vt_path, index=False)
    print(f"{vt_path}: {len(rows)} INJECT targets added")


# --------------------------------------------------------------------------- compare


def _lc(ps: Path, cid: str) -> pd.DataFrame | None:
    p = ps / "photometry" / "lightcurves" / f"lightcurve_{cid}.csv"
    return pd.read_csv(p, low_memory=False) if p.is_file() else None


def _roles(ps: Path) -> tuple[set[str], set[str]]:
    """(active target ids, comparison star ids) of a photometry run."""
    ph = ps / "photometry"
    at = pd.read_csv(ph / "active_targets.csv", low_memory=False, dtype={"catalog_id": str})
    tg = set(at.loc[~at["skip_photometry"].astype(str).str.lower().isin(("true", "1")), "catalog_id"].astype(str))
    comps: set[str] = set()
    cp = ph / "comparison_stars_per_target.csv"
    if cp.is_file():
        c = pd.read_csv(cp, low_memory=False, dtype=str)
        col = next((k for k in ("comp_catalog_id", "catalog_id", "comp_id") if k in c.columns), None)
        if col:
            comps = set(c[col].astype(str))
    return tg, comps


def _fit_scale(d: np.ndarray, tmpl: np.ndarray) -> float:
    """Least-squares scale s for d = s * tmpl + c."""
    A = np.c_[tmpl, np.ones_like(tmpl)]
    s, _c = np.linalg.lstsq(A, d, rcond=None)[0]
    return float(s)


def _fit_tmin_shift(d: np.ndarray, t: np.ndarray, sig: Signal) -> float:
    """Shift (minutes) of the eclipse template that best fits ``d`` (scale and offset free)."""
    best, best_res = 0.0, float("inf")
    for sh in np.arange(-20.0, 20.001, 0.1):
        tm = Signal("eclipse", 1.0, t0=sig.t0 + sh / 1440.0, width_d=sig.width_d).dm(t)
        A = np.c_[tm, np.ones_like(tm)]
        coef, res, *_ = np.linalg.lstsq(A, d, rcond=None)
        r = float(np.sum((d - A @ coef) ** 2))
        if r < best_res:
            best, best_res = float(sh), r
    return best


def cmd_compare(a: argparse.Namespace) -> None:
    plan = json.loads(Path(a.plan).read_text(encoding="ascii"))
    ps_c, ps_i = _ps_dir(Path(a.control_draft)), _ps_dir(Path(a.injected_draft))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    series: list[tuple[dict[str, Any], pd.DataFrame]] = []
    # A host must be a measured target and never a comparison star in either run (an injected
    # comparison star would leak its signal into the real targets' ensembles).
    roles = {name: _roles(ps) for name, ps in (("control", ps_c), ("injected", ps_i))}
    for h in plan["hosts"]:
        sig = Signal(**h["signal"])
        lc_c, lc_i = _lc(ps_c, h["catalog_id"]), _lc(ps_i, h["catalog_id"])
        base = {"catalog_id": h["catalog_id"], "mag": h["mag"], "bp_rp": h["bp_rp"], "signal": sig.kind,
                "amp": sig.amp}
        bad_role = [f"{n}: {'comparison star' if h['catalog_id'] in r[1] else 'not a target'}"
                    for n, r in roles.items() if h["catalog_id"] in r[1] or h["catalog_id"] not in r[0]]
        if bad_role:
            rows.append({**base, "status": "; ".join(bad_role)})
            continue
        if lc_c is None or lc_i is None:
            rows.append({**base, "status": "no light curve in " + ("control" if lc_c is None else "injected")})
            continue
        m = lc_c.merge(lc_i, on="source_file", suffixes=("_c", "_i"))
        m = m[np.isfinite(m["mag_calib_c"]) & np.isfinite(m["mag_calib_i"])]
        t = m["jd_i"].to_numpy(dtype=float)
        truth = sig.dm(t)
        # full chain: calibrated magnitudes (comparison ensemble included; the two runs may pick a
        # slightly different ensemble -> a zero-point offset); measurement only: instrumental
        # magnitudes of the same frames (no ensemble involved)
        d = (m["mag_calib_i"] - m["mag_calib_c"]).to_numpy(dtype=float)
        d_inst = (m["mag_inst_i"] - m["mag_inst_c"]).to_numpy(dtype=float)
        resid = d - truth
        resid_i = d_inst - truth
        rec = {
            **base, "status": "ok", "n_epochs": int(len(m)), "n_control": int(len(lc_c)), "n_injected": int(len(lc_i)),
            "bias_mmag": 1000.0 * float(np.mean(resid)), "resid_rms_mmag": 1000.0 * float(np.std(resid)),
            "inst_bias_mmag": 1000.0 * float(np.mean(resid_i)), "inst_resid_rms_mmag": 1000.0 * float(np.std(resid_i)),
            "err_median_mmag": 1000.0 * float(np.median(m["err_i"])),
            "lc_rms_control_mmag": 1000.0 * float(np.std(m["mag_calib_c"])),
            "flagged_control": int((m["flag_c"].astype(str) != "normal").sum()),
            "flagged_injected": int((m["flag_i"].astype(str) != "normal").sum()),
        }
        if sig.kind in ("sine", "eclipse") and sig.amp > 0:
            rec["amp_ratio"] = _fit_scale(d, sig.dm(t) / sig.amp) / sig.amp
            rec["inst_amp_ratio"] = _fit_scale(d_inst, sig.dm(t) / sig.amp) / sig.amp
        if sig.kind == "eclipse":
            rec["tmin_shift_min"] = _fit_tmin_shift(d, t, sig)
        rows.append(rec)
        series.append((h, m.assign(d=d, truth=truth)))
    res = pd.DataFrame(rows)
    res.to_csv(out / "results.csv", index=False)
    _summary(res, plan, out, a.control_draft, a.injected_draft)
    _plots(series, out)
    print(f"compare: {len(res)} hosts -> {out}")


def _summary(res: pd.DataFrame, plan: dict[str, Any], out: Path, control: str, injected: str) -> None:
    lines = [f"# Injection-recovery - {plan['setup']} ({TOOL}, seed {plan['seed']})", "",
             f"Plan from: `{plan['control_draft']}`", f"Control run: `{control}`", f"Injected run: `{injected}`", f"Hosts: {len(res)}, stamp radius {plan['stamp_r_fwhm']} x FWHM "
             f"({plan['fwhm_px']:.2f} px)", ""]
    ok = res[res["status"] == "ok"]
    cols = [c for c in ("signal", "amp", "mag", "n_epochs", "err_median_mmag", "inst_amp_ratio", "inst_resid_rms_mmag",
                        "amp_ratio", "bias_mmag", "resid_rms_mmag", "tmin_shift_min", "flagged_control",
                        "flagged_injected") if c in ok.columns]
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in ok.sort_values(["signal", "amp", "mag"]).iterrows():
        vals = []
        for c in cols:
            v = r.get(c)
            vals.append(f"{v:.3f}" if isinstance(v, float) and math.isfinite(v) else ("" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(vals) + " |")
    bad = res[res["status"] != "ok"]
    if len(bad):
        lines += ["", "Not compared:"] + [f"- {r.catalog_id}: {r.status}" for r in bad.itertuples()]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="ascii")


def _plots(series: list[tuple[dict[str, Any], pd.DataFrame]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not series:
        return
    n = len(series)
    cols = 3
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 2.6 * rows), squeeze=False)
    for ax, (h, m) in zip(axes.ravel(), series):
        t = (m["jd_i"] - m["jd_i"].min()) * 24.0
        ax.plot(t, 1000 * m["d"], ".", ms=3, label="recovered (injected - control)")
        order = np.argsort(t.to_numpy())
        ax.plot(t.to_numpy()[order], 1000 * m["truth"].to_numpy()[order], "-", lw=1, label="injected")
        s = h["signal"]
        ax.set_title(f"{s['kind']} {s['amp']:g} mag, G={h['mag']:.1f}", fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("hours", fontsize=8)
        ax.set_ylabel("mmag", fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "recovered_vs_injected.png", dpi=110)
    plt.close(fig)



# --------------------------------------------------------------------------- step B: new stars

SYNTH_ID0 = 9_100_000_000_000_000_000  # above the Gaia DR3 source_id range (< 6.92e18)


def _select_isolated(draft: Path, *, snr_min: float, peak_frac: float, edge_px: float, iso_fwhm: float,
                     iso_dmag: float) -> tuple[pd.DataFrame, pd.DataFrame, Any, float, int, int]:
    """(isolated unsaturated catalogue stars, field cone with x/y, MASTERSTAR WCS, FWHM, w, h)."""
    from astropy.wcs import WCS

    ps = _ps_dir(draft)
    hdr = fits.getheader(ps / "MASTERSTAR.fits")
    fwhm = float(hdr.get("VY_FWHM") or 2.5)
    w, h = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
    wcs = WCS(hdr)
    ms = pd.read_csv(ps / "masterstars_full_match.csv", low_memory=False, dtype={"catalog_id": str, "name": str})
    ms["cid"] = ms["name"].astype(str).str.strip()
    ok = (
        np.isfinite(pd.to_numeric(ms["flux"], errors="coerce"))
        & (ms["source_state"].astype(str).str.lower() != "catalog_membership")
        & (ms["zone"].astype(str).str.lower() == "linear")
        & (pd.to_numeric(ms["snr_ap_pixscaled"], errors="coerce") >= snr_min)
        & (pd.to_numeric(ms["peak_max_adu"], errors="coerce") < peak_frac * pd.to_numeric(ms["saturate_limit_adu"], errors="coerce"))
        & ms["x"].between(edge_px, w - edge_px) & ms["y"].between(edge_px, h - edge_px)
        & ms["cid"].str.fullmatch(r"\d{6,}")
        & ~ms["phot_variable_flag"].astype(str).str.upper().eq("VARIABLE")
        & ~ms["non_single_star"].fillna(0).astype(float).astype(bool)
    )
    vt = pd.read_csv(ps / "variable_targets.csv", low_memory=False, dtype={"catalog_id": str})
    ok &= ~ms["cid"].isin(set(vt["catalog_id"].astype(str)))
    fc = pd.read_csv(ps / "field_catalog_cone.csv", low_memory=False, dtype={"catalog_id": str})
    fx, fy = wcs.all_world2pix(fc["ra_deg"].to_numpy(dtype=float), fc["dec_deg"].to_numpy(dtype=float), 0)
    fc = fc.assign(x=fx, y=fy)
    cand = ms.loc[ok].copy()
    cand = cand.loc[_isolated(cand, np.c_[fx, fy], fc["mag"].to_numpy(dtype=float), iso_fwhm * fwhm, iso_dmag)]
    return cand.sort_values("mag").reset_index(drop=True), fc, wcs, fwhm, w, h


def _clear_of(fc: pd.DataFrame, x: float, y: float, mag: float, *, r_any: float, r_bright: float, dmag: float,
              skip: set[str] = frozenset()) -> bool:
    d = np.hypot(fc["x"].to_numpy() - x, fc["y"].to_numpy() - y)
    m = fc["mag"].to_numpy(dtype=float)
    keep = ~fc["catalog_id"].astype(str).isin(skip).to_numpy()
    return not (np.any(keep & (d < r_any)) or np.any(keep & (d < r_bright) & (m < mag + dmag)))


def cmd_plan_new(a: argparse.Namespace) -> None:
    draft = Path(a.control_draft)
    cand, fc, wcs, fwhm, w, h = _select_isolated(draft, snr_min=a.snr_min, peak_frac=0.6, edge_px=a.edge_px,
                                                  iso_fwhm=a.stamp_r_fwhm, iso_dmag=3.0)
    rng = np.random.default_rng(a.seed)
    r_st = a.stamp_r_fwhm * fwhm
    jds = sorted(_frame_jd(fits.getheader(p)) for p in sorted(Path(a.raw_dir).glob("*.fit*")))
    t_lo, t_hi = jds[0], jds[-1]
    span = t_hi - t_lo
    # templates: the brightest isolated stars, spread over the chip
    templates = cand.nsmallest(min(a.n_templates, len(cand)), "mag")
    # flux scale of the field: median of MASTERSTAR aperture flux x 10**(0.4 G) over the isolated
    # stars. A synthetic star of magnitude G gets flux zp x 10**(-0.4 G), whatever the colour of its
    # template (scaling by the template's own G would carry the template's colour term: night 537,
    # first try, two G 11 stars 0.3 mag apart).
    fl = pd.to_numeric(cand["flux_ap"], errors="coerce").to_numpy(dtype=float)
    gm = pd.to_numeric(cand["mag"], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(fl) & (fl > 0) & np.isfinite(gm)
    flux_zp = float(np.median(fl[good] * 10.0 ** (0.4 * gm[good])))
    tmpl_ids = set(templates["cid"].astype(str))
    # neighbour hosts: isolated stars of medium brightness, not templates
    hosts_pool = cand.loc[~cand["cid"].isin(tmpl_ids) & cand["mag"].between(a.host_mag_min, a.host_mag_max)]
    hosts_pool = hosts_pool.sample(frac=1.0, random_state=a.seed).head(a.n_hosts)
    stars: list[dict[str, Any]] = []
    hosts: list[dict[str, Any]] = []
    placed: list[tuple[float, float]] = []

    def _add(kind: str, x: float, y: float, g: float, sig: Signal, extra: dict[str, Any]) -> None:
        ra, dec = wcs.all_pix2world(x, y, 0)
        stars.append({"catalog_id": str(SYNTH_ID0 + len(stars) + 1), "kind": kind, "ra_deg": float(ra),
                      "dec_deg": float(dec), "x_ref": float(x), "y_ref": float(y), "mag": float(g),
                      "bp_rp": float(a.bp_rp), "signal": asdict(sig), **extra})
        placed.append((x, y))

    def _far_from_placed(x: float, y: float) -> bool:
        return all(math.hypot(x - px, y - py) > 2.5 * r_st for px, py in placed)

    for i, (_, hr) in enumerate(hosts_pool.iterrows()):
        hsig = Signal("sine", a.host_amp, period_d=0.6 * span, t0=t_lo + rng.uniform(0, 0.6 * span)) if i % 2 == 0 \
            else Signal("null")
        hosts.append({"catalog_id": str(hr["cid"]), "ra_deg": float(hr["ra_deg"]), "dec_deg": float(hr["dec_deg"]),
                      "x_ref": float(hr["x"]), "y_ref": float(hr["y"]), "mag": float(hr["mag"]),
                      "bp_rp": float(pd.to_numeric(hr.get("bp_rp"), errors="coerce")), "mag_bin": 0,
                      "signal": asdict(hsig)})
        placed.append((float(hr["x"]), float(hr["y"])))
        ang0 = rng.uniform(0, 2 * np.pi)
        for j, sep in enumerate(a.sep_fwhm):
            ang = ang0 + 2 * np.pi * j / len(a.sep_fwhm)
            x = float(hr["x"]) + sep * fwhm * math.cos(ang)
            y = float(hr["y"]) + sep * fwhm * math.sin(ang)
            _add("neighbour", x, y, float(hr["mag"]) + a.dmag, Signal("null"),
                 {"host_id": str(hr["cid"]), "sep_fwhm": float(sep), "dmag": float(a.dmag),
                  "host_signal": asdict(hsig)})
    # empty-sky stars on a magnitude grid
    n_try = 0
    for g in a.empty_mags:
        for k in range(a.per_mag):
            sig = Signal("sine", a.empty_amp, period_d=0.6 * span, t0=t_lo + rng.uniform(0, 0.6 * span)) \
                if (k >= a.per_mag - a.n_sine and g <= a.sine_mag_max) else Signal("null")
            while True:
                n_try += 1
                if n_try > 200000:
                    raise SystemExit("no free positions left - fewer stars or a sparser field")
                x, y = rng.uniform(a.edge_px, w - a.edge_px), rng.uniform(a.edge_px, h - a.edge_px)
                if _clear_of(fc, x, y, g, r_any=2.0 * fwhm, r_bright=r_st, dmag=3.0) and _far_from_placed(x, y):
                    break
            _add("empty", x, y, float(g), sig, {})
    plan = {
        "tool": TOOL, "step": "B", "seed": a.seed, "control_draft": str(draft), "setup": _ps_dir(draft).name,
        "fwhm_px": fwhm, "stamp_r_fwhm": a.stamp_r_fwhm, "jd_first": t_lo, "jd_last": t_hi,
        "flux_zp": flux_zp,
        "templates": [{"catalog_id": str(r["cid"]), "x_ref": float(r["x"]), "y_ref": float(r["y"]),
                       "mag": float(r["mag"]), "flux_ap": float(pd.to_numeric(r["flux_ap"], errors="coerce"))}
                      for _, r in templates.iterrows()],
        "hosts": hosts, "stars": stars,
    }
    out = Path(a.out)
    out.write_text(json.dumps(plan, indent=2), encoding="ascii")
    g = np.array([st["mag"] for st in stars])
    cat = pd.DataFrame({
        "catalog_id": [st["catalog_id"] for st in stars], "ra_deg": [st["ra_deg"] for st in stars],
        "dec_deg": [st["dec_deg"] for st in stars], "mag": g, "bp_mag": g + 0.45 * a.bp_rp,
        "rp_mag": g - 0.55 * a.bp_rp, "bp_rp": a.bp_rp, "var_flag": "NOT_AVAILABLE", "g_flux_error_rel": 0.001,
        "non_single_star": 0, "b_v": np.nan,
    })
    cat.to_csv(out.with_name("inject_catalog.csv"), index=False)
    print(f"plan-new: {sum(s_['kind'] == 'empty' for s_ in stars)} empty-sky + "
          f"{sum(s_['kind'] == 'neighbour' for s_ in stars)} neighbour stars, {len(hosts)} hosts, "
          f"{len(templates)} templates -> {out}")


def _stamp(img: np.ndarray, dark: np.ndarray, x: float, y: float, r_px: float) -> tuple[np.ndarray, int, int]:
    """Dark- and sky-subtracted star image (circle r_px) around native (x, y); (stamp, y0, x0)."""
    R = int(math.ceil(3.0 * r_px)) + 1
    hh, ww = img.shape
    x0, x1 = max(0, int(x) - R), min(ww, int(x) + R + 1)
    y0, y1 = max(0, int(y) - R), min(hh, int(y) + R + 1)
    sub = img[y0:y1, x0:x1].astype(np.float64) - dark[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - x, yy - y)
    sky = float(np.median(sub[(rr >= 2.0 * r_px) & (rr <= 3.0 * r_px)]))
    return np.where(rr <= r_px, sub - sky, 0.0), y0, x0


def _raw_gain(raw_dir: Path, control_draft: Path) -> float:
    """e-/ADU of the raw frames as VYVAR resolves it: header (VY_EGAIN, EGAIN, GAIN) first, then the
    control run's resolved value. Night 534 (C5A-150M, bin 4): GAIN = 12.48 = 16 x 0.78 because the
    camera averages the binned pixels; the database value 0.78 is the unbinned gain (the first run of
    step B on 534 used it and injected 4 x too much photon noise - an error of the test)."""
    fp = next(iter(sorted(raw_dir.glob("*.fit*"))), None)
    if fp is not None:
        hdr = fits.getheader(fp)
        for k in ("VY_EGAIN", "EGAIN", "GAIN"):  # param_resolver.HEADER_KEYS["gain"]
            try:
                v = float(hdr[k])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(v) and v > 0:
                return v
    meta = json.loads((_ps_dir(control_draft) / "photometry" / "pipeline_meta.json").read_text())
    if "resolved_facts" not in meta:
        raise SystemExit("no gain in the raw headers nor in the control run - pass --gain")
    return float(meta["resolved_facts"]["gain"]["value"])


def cmd_inject_new(a: argparse.Namespace) -> None:
    from scipy.ndimage import shift as nd_shift

    plan = json.loads(Path(a.plan).read_text(encoding="ascii"))
    aligned = _aligned_dir(Path(plan["control_draft"]), plan["setup"])
    gain = float(a.gain) if a.gain else _raw_gain(Path(a.raw_dir), Path(plan["control_draft"]))
    r_px = float(plan["stamp_r_fwhm"]) * float(plan["fwhm_px"])
    rng = np.random.default_rng(int(plan["seed"]) + 1)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    tx = np.array([[t["x_ref"], t["y_ref"]] for t in plan["templates"]])
    dark = None
    log: list[dict[str, Any]] = []
    for fp in sorted(Path(a.raw_dir).glob("*.fit*")):
        with fits.open(fp) as hd:
            hdr = hd[0].header.copy()
            raw = np.asarray(hd[0].data)
        al = aligned / f"{fp.stem}.fits"
        m = _align_matrix(fits.getheader(al)) if al.is_file() else None
        if m is None:
            fits.writeto(out_dir / fp.name, raw, hdr, output_verify="silentfix")
            log.append({"file": fp.name, "status": "not_aligned_copied"})
            continue
        if dark is None:
            dark = _dark_for(raw.shape, Path(a.dark), a.dark_binning)
        jd = _frame_jd(hdr)
        minv = np.linalg.inv(m)
        orig = raw.astype(np.float64)
        img = orig.copy()

        def _native(xr: float, yr: float) -> tuple[float, float]:
            v = minv @ np.array([xr, yr, 1.0])
            return float(v[0]), float(v[1])

        # 1) neighbour hosts with a signal (step-A injection on the host's own image)
        for hst in plan["hosts"]:
            sig = Signal(**hst["signal"])
            if sig.kind == "null":
                continue
            xn, yn = _native(hst["x_ref"], hst["y_ref"])
            img, _ = inject_into_frame(img, dark, xn, yn, float(flux_factor(sig.dm(jd))), r_px)
        # 2) synthetic stars: nearest template (from the untouched frame), shifted, scaled, + noise
        for st in plan["stars"]:
            k = int(np.argmin(np.hypot(tx[:, 0] - st["x_ref"], tx[:, 1] - st["y_ref"])))
            tp = plan["templates"][k]
            txn, tyn = _native(tp["x_ref"], tp["y_ref"])
            vx, vy = st.get("rate_px_day", (0.0, 0.0))  # TRANSIENT-2026-10: moving synthetic object
            sxn, syn = _native(st["x_ref"] + vx * (jd - plan["jd_first"]), st["y_ref"] + vy * (jd - plan["jd_first"]))
            stamp, y0, x0 = _stamp(orig, dark, txn, tyn, r_px)
            # target flux from the field's flux scale, relative to the template's measured flux
            scale = (float(plan["flux_zp"]) * 10.0 ** (-0.4 * st["mag"]) / float(tp["flux_ap"])) \
                * float(flux_factor(Signal(**st["signal"]).dm(jd)))
            if scale <= 0 or float(Signal(**st["signal"]).dm(jd)) >= OFF_MAG:
                continue
            dx, dy = sxn - txn, syn - tyn
            ix, iy = int(math.floor(dx)), int(math.floor(dy))
            moved = nd_shift(stamp * scale, (dy - iy, dx - ix), order=3, mode="constant", cval=0.0)
            ty0, tx0 = y0 + iy, x0 + ix
            hh, ww = img.shape
            ya, yb = max(0, ty0), min(hh, ty0 + moved.shape[0])
            xa, xb = max(0, tx0), min(ww, tx0 + moved.shape[1])
            add = moved[ya - ty0:yb - ty0, xa - tx0:xb - tx0]
            noise = rng.normal(0.0, 1.0, add.shape) * np.sqrt(np.clip(add, 0.0, None) / gain)
            img[ya:yb, xa:xb] += add + noise
            log.append({"file": fp.name, "catalog_id": st["catalog_id"], "jd": jd, "x": sxn, "y": syn,
                        "template": tp["catalog_id"], "scale": scale, "added_adu": float(add.sum())})
        info = np.iinfo(raw.dtype) if np.issubdtype(raw.dtype, np.integer) else None
        img = np.clip(np.rint(img), info.min, info.max).astype(raw.dtype) if info is not None else img.astype(raw.dtype)
        hdr["VYINJECT"] = (TOOL, "injection-recovery test frame (step B)")
        fits.writeto(out_dir / fp.name, img, hdr, output_verify="silentfix")
    pd.DataFrame(log).to_csv(out_dir.parent / f"{out_dir.name}_inject_log.csv", index=False)
    print(f"inject-new: {sum(1 for r in log if 'scale' in r)} star-frames (gain {gain:g} e-/ADU) into {out_dir}")


def cmd_plan_transients(a: argparse.Namespace) -> None:
    """TRANSIENT-2026-10: a step-B plan of uncatalogued sources (no catalogue rows): per magnitude
    one source present all night, one switching on, one nova-like rise, one flare; two moving."""
    a.n_hosts, a.sep_fwhm = 0, []
    a.empty_mags = list(a.mags)
    a.per_mag, a.n_sine, a.sine_mag_max, a.empty_amp = 4, 0, -99.0, 0.0
    cmd_plan_new(a)
    out = Path(a.out)
    plan = json.loads(out.read_text(encoding="ascii"))
    t_lo, t_hi = float(plan["jd_first"]), float(plan["jd_last"])
    span = t_hi - t_lo
    rng = np.random.default_rng(a.seed + 7)
    kinds = ("null", "step", "rise", "flare")
    for i, st in enumerate(plan["stars"]):
        k = kinds[i % 4]
        t0 = t_lo + rng.uniform(0.25, 0.6) * span
        sig = {"null": Signal("null"), "step": Signal("step", t0=t0),
               "rise": Signal("rise", amp=3.0, t0=t0, width_d=1.5 / 24.0),
               "flare": Signal("flare", t0=t0, width_d=20.0 / 1440.0)}[k]
        st["signal"] = asdict(sig)
        st["kind"] = "transient"
    fw = float(plan["fwhm_px"])
    for j in range(2):  # moving objects: ~30 and ~60 "/h at the plan's pixel scale, along random directions
        src = plan["stars"][j]
        ang = rng.uniform(0, 2 * np.pi)
        v = (8.0 + 8.0 * j) * fw / max(span, 1e-3)
        mv = dict(src, catalog_id=str(SYNTH_ID0 + 900 + j), kind="moving", signal=asdict(Signal("null")),
                  rate_px_day=[float(v * math.cos(ang)), float(v * math.sin(ang))],
                  x_ref=src["x_ref"] + 3 * fw, y_ref=src["y_ref"] + 3 * fw)
        plan["stars"].append(mv)
    out.write_text(json.dumps(plan, indent=2), encoding="ascii")
    # the transients are NOT catalogued: empty test catalogue
    cat = out.with_name("inject_catalog.csv")
    pd.read_csv(cat).iloc[0:0].to_csv(cat, index=False)
    print(f"plan-transients: {len(plan['stars'])} uncatalogued sources -> {out}")


def cmd_compare_transients(a: argparse.Namespace) -> None:
    """TRANSIENT-2026-10: injected transients vs ``transient_candidates.csv`` / ``moving_objects.csv``."""
    plan = json.loads(Path(a.plan).read_text(encoding="ascii"))
    ph = _ps_dir(Path(a.injected_draft)) / "photometry"
    T = pd.read_csv(ph / "transient_candidates.csv")
    M = pd.read_csv(ph / "moving_objects.csv") if (ph / "moving_objects.csv").stat().st_size > 2 else pd.DataFrame()
    fr = pd.read_csv(next((_ps_dir(Path(a.injected_draft)).parent.parent / "detrended_aligned" / "lights").glob("*/frames.csv")))
    times = np.sort(fr["jd_mid"].to_numpy(float))
    fw = float(plan["fwhm_px"])
    rows = []
    for st in plan["stars"]:
        sig = Signal(**st["signal"])
        if st["kind"] == "moving":
            vx, vy = st["rate_px_day"]
            hit = ""
            if not M.empty:
                tm = 0.5 * (times[0] + times[-1]) - plan["jd_first"]
                xb, yb = st["x_ref"] + vx * tm, st["y_ref"] + vy * tm
                xm = M["x_t0"] + (M["x_t1"] - M["x_t0"]) * (0.5 * (times[0] + times[-1]) - M["jd0"]) / (M["jd1"] - M["jd0"])
                ym = M["y_t0"] + (M["y_t1"] - M["y_t0"]) * (0.5 * (times[0] + times[-1]) - M["jd0"]) / (M["jd1"] - M["jd0"])
                d = np.hypot(xm - xb, ym - yb)
                hit = "found" if (d < 2 * fw).any() else ""
            rows.append({"catalog_id": st["catalog_id"], "kind": "moving", "mag": st["mag"], "signal": "moving",
                         "rate_px_h": math.hypot(vx, vy) / 24.0, "status": hit or "missed"})
            continue
        d = np.hypot(T["x"] - st["x_ref"], T["y"] - st["y_ref"]) if not T.empty else np.array([])
        j = int(np.argmin(d)) if len(d) else -1
        rec = {"catalog_id": st["catalog_id"], "kind": "transient", "mag": st["mag"], "signal": sig.kind,
               "t0": sig.t0 if sig.kind != "null" else float("nan")}
        if j >= 0 and d.iloc[j] < fw:
            r = T.iloc[j]
            rec.update(status="candidate" if bool(r["candidate"]) else f"rejected: {r['reject_reason']}",
                       found_kind=r["kind"], mag_g_est=r["mag_g_est"], first_time=r["first_time"],
                       delay_min=(r["first_time"] - sig.t0) * 1440.0 if sig.kind != "null" else float("nan"))
        else:
            rec["status"] = "not found"
        rows.append(rec)
    res = pd.DataFrame(rows)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "results_transients.csv", index=False)
    inj = {(round(s["x_ref"]), round(s["y_ref"])) for s in plan["stars"]}
    n_fp = int(sum(1 for _, r in T[T["candidate"]].iterrows()
                   if min((math.hypot(r["x"] - x, r["y"] - y) for x, y in inj), default=1e9) > fw)) if not T.empty else 0
    print(res.to_string())
    print(f"candidates not injected: {n_fp}")


def _excluded_reason(ph: Path, cid: str) -> str:
    p = ph / "excluded_targets.csv"
    at = ph / "active_targets.csv"
    if at.is_file():
        a_ = pd.read_csv(at, low_memory=False, dtype={"catalog_id": str})
        hit = a_.loc[a_["catalog_id"].astype(str) == cid]
        if len(hit) and str(hit.iloc[0].get("skip_reason") or "") not in ("", "nan"):
            return str(hit.iloc[0]["skip_reason"])
    if p.is_file():
        e = pd.read_csv(p, low_memory=False, dtype=str)
        hit = e.loc[e["name"].astype(str).str.endswith(cid)] if "name" in e.columns else e.iloc[0:0]
        if len(hit):
            return str(hit.iloc[0].get("reason"))
    return "not measured"


def cmd_compare_new(a: argparse.Namespace) -> None:
    plan = json.loads(Path(a.plan).read_text(encoding="ascii"))
    ps = _ps_dir(Path(a.injected_draft))
    ph = ps / "photometry"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path_kind, suffix in (("aperture", ""), ("epsf", "_psf")):
        for st in plan["stars"]:
            sig = Signal(**st["signal"])
            base = {"method": path_kind, "catalog_id": st["catalog_id"], "kind": st["kind"], "mag": st["mag"],
                    "signal": sig.kind, "amp": sig.amp, "sep_fwhm": st.get("sep_fwhm"), "dmag": st.get("dmag"),
                    "host_signal": (st.get("host_signal") or {}).get("kind")}
            p = ph / "lightcurves" / f"lightcurve_{st['catalog_id']}{suffix}.csv"
            if not p.is_file():
                if path_kind == "aperture":
                    rows.append({**base, "status": _excluded_reason(ph, st["catalog_id"])})
                continue
            lc = pd.read_csv(p, low_memory=False, comment="#")  # ePSF curves carry a '#' header
            if path_kind == "epsf":
                # differential ePSF magnitude against the ensemble; its error is in magnitudes
                lc = lc[lc["psf_fit_ok"].astype(str).str.lower() == "true"]
                mag_s, err_s = lc["psf_delta_mag"], lc["psf_delta_mag_err"]
            else:
                mag_s, err_s = lc["mag_calib"], 1.0857362 * lc["err"]  # err: relative flux error
            keep = np.isfinite(pd.to_numeric(mag_s, errors="coerce"))
            lc, mag_s, err_s = lc[keep], mag_s[keep], err_s[keep]
            if not len(lc):  # e.g. aperture curve not calibrated (no comparison star with the band)
                rows.append({**base, "status": "no finite magnitude"})
                continue
            t = lc["jd"].to_numpy(dtype=float)
            mag = pd.to_numeric(mag_s, errors="coerce").to_numpy(dtype=float)
            err_mag = pd.to_numeric(err_s, errors="coerce").to_numpy(dtype=float)
            err_col = True
            dm = sig.dm(t)
            if sig.kind == "sine" and sig.amp > 0:
                A = np.c_[sig.dm(t) / sig.amp, np.ones_like(t)]
                coef = np.linalg.lstsq(A, mag, rcond=None)[0]
                amp_ratio, off = float(coef[0]) / sig.amp, float(coef[1])
                resid = mag - A @ coef
            else:
                amp_ratio, off = float("nan"), float(np.median(mag - dm))
                resid = mag - dm - off
            rec = {**base, "status": "ok", "n_epochs": int(len(lc)), "offset_mag": off,
                   "rms_mmag": 1000.0 * float(np.std(resid)), "err_median_mmag": 1000.0 * float(np.nanmedian(err_mag)),
                   "pull_rms": float(np.sqrt(np.nanmean((resid / err_mag) ** 2))) if err_col else float("nan"),
                   "amp_ratio": amp_ratio, "flagged": int((lc["flag"].astype(str) != "normal").sum()) if "flag" in lc else 0}
            hs = st.get("host_signal")
            if hs and hs.get("kind") == "sine":
                hsig = Signal(**hs)
                tm = hsig.dm(t) / hsig.amp
                c = np.linalg.lstsq(np.c_[tm, np.ones_like(t)], mag, rcond=None)[0]
                rec["leak_ratio"] = float(c[0]) / hsig.amp  # fraction of the host's amplitude seen in the star
            rows.append(rec)
    res = pd.DataFrame(rows)
    # zero point: empty-sky constant stars of each method; blend bias of a star = its offset - ZP
    for meth in res["method"].unique() if len(res) else []:
        ok = (res["method"] == meth) & (res["status"] == "ok")
        emp = ok & (res["kind"] == "empty") & (res["signal"] == "null")
        if emp.any():
            zp = float(np.median(res.loc[emp, "offset_mag"] - res.loc[emp, "mag"]))
            res.loc[ok, "bias_mmag"] = 1000.0 * (res.loc[ok, "offset_mag"] - res.loc[ok, "mag"] - zp)
    res.to_csv(out / "results_new.csv", index=False)
    _summary_new(res, plan, out, a.injected_draft)
    print(f"compare-new: {len(res)} rows -> {out}")


def _summary_new(res: pd.DataFrame, plan: dict[str, Any], out: Path, injected: str) -> None:
    def fmt(v: Any, d: int = 2) -> str:
        return f"{v:.{d}f}" if isinstance(v, (int, float)) and math.isfinite(float(v)) else ("" if v is None or (isinstance(v, float) and not math.isfinite(v)) else str(v))

    lines = [f"# Injection-recovery step B - {plan['setup']} ({TOOL}, seed {plan['seed']})", "",
             f"Injected run: `{injected}`", f"Stars: {len(plan['stars'])}, templates {len(plan['templates'])}, "
             f"stamp radius {plan['stamp_r_fwhm']} x FWHM ({plan['fwhm_px']:.2f} px)", ""]
    for meth in ("aperture", "epsf"):
        sub = res[res["method"] == meth]
        if not len(sub):
            continue
        lines += [f"## {meth}", "", "### Empty sky", "",
                  "| G | signal | status | n | err (mmag) | rms (mmag) | pull rms | amp ratio | bias (mmag) | flagged |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub[sub["kind"] == "empty"].sort_values(["mag", "signal"]).iterrows():
            lines.append(f"| {fmt(r.mag)} | {r.signal} | {r.status} | {fmt(r.get('n_epochs'), 0)} | {fmt(r.get('err_median_mmag'), 1)} | "
                         f"{fmt(r.get('rms_mmag'), 1)} | {fmt(r.get('pull_rms'))} | {fmt(r.get('amp_ratio'), 3)} | {fmt(r.get('bias_mmag'), 1)} | {fmt(r.get('flagged'), 0)} |")
        lines += ["", "### Next to a brighter star", "",
                  "| sep (FWHM) | dmag | host signal | G | status | rms (mmag) | pull rms | bias (mmag) | leak ratio | flagged |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub[sub["kind"] == "neighbour"].sort_values(["sep_fwhm", "host_signal"]).iterrows():
            lines.append(f"| {fmt(r.sep_fwhm, 1)} | {fmt(r.dmag, 1)} | {r.host_signal} | {fmt(r.mag)} | {r.status} | {fmt(r.get('rms_mmag'), 1)} | "
                         f"{fmt(r.get('pull_rms'))} | {fmt(r.get('bias_mmag'), 1)} | {fmt(r.get('leak_ratio'), 3)} | {fmt(r.get('flagged'), 0)} |")
        lines.append("")
    (out / "summary_new.md").write_text("\n".join(lines) + "\n", encoding="ascii")


# --------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--control-draft", required=True)
    pl.add_argument("--raw-dir", required=True, help="raw LIGHT directory of the night (frame times)")
    pl.add_argument("--out", required=True)
    pl.add_argument("--seed", type=int, default=20261023)
    pl.add_argument("--bins", type=int, default=4, help="magnitude bins (hosts per bin = signal set)")
    pl.add_argument("--snr-min", type=float, default=10.0)
    pl.add_argument("--peak-frac", type=float, default=0.6, help="host peak below this fraction of saturation")
    pl.add_argument("--edge-px", type=float, default=60.0)
    pl.add_argument("--iso-fwhm", type=float, default=6.5, help="isolation radius in FWHM")
    pl.add_argument("--iso-dmag", type=float, default=3.0, help="neighbours fainter than host + this are ignored")
    pl.add_argument("--stamp-r-fwhm", type=float, default=6.5,
                    help="injected star radius; must exceed the photometry sky annulus (annulus_outer_fwhm)")
    pl.set_defaults(fn=cmd_plan)
    ij = sub.add_parser("inject")
    ij.add_argument("--plan", required=True)
    ij.add_argument("--raw-dir", required=True)
    ij.add_argument("--dark", required=True, help="master dark used for the night (raw stamps are dark-subtracted)")
    ij.add_argument("--dark-binning", choices=("sum", "average"), default="sum")
    ij.add_argument("--out-dir", required=True)
    ij.set_defaults(fn=cmd_inject)
    tg = sub.add_parser("targets")
    tg.add_argument("--plan", required=True)
    tg.add_argument("--draft", required=True)
    tg.add_argument("--as-vsx", action="store_true", help="list as VSX targets so that RUN ePSF fits them too")
    tg.set_defaults(fn=cmd_targets)
    cp = sub.add_parser("compare")
    cp.add_argument("--plan", required=True)
    cp.add_argument("--control-draft", required=True)
    cp.add_argument("--injected-draft", required=True)
    cp.add_argument("--out-dir", required=True)
    cp.set_defaults(fn=cmd_compare)
    pn = sub.add_parser("plan-new")
    pn.add_argument("--control-draft", required=True)
    pn.add_argument("--raw-dir", required=True)
    pn.add_argument("--out", required=True)
    pn.add_argument("--seed", type=int, default=20261023)
    pn.add_argument("--snr-min", type=float, default=10.0)
    pn.add_argument("--edge-px", type=float, default=80.0)
    pn.add_argument("--stamp-r-fwhm", type=float, default=6.5)
    pn.add_argument("--n-templates", type=int, default=12)
    pn.add_argument("--n-hosts", type=int, default=8)
    pn.add_argument("--host-mag-min", type=float, default=9.0)
    pn.add_argument("--host-mag-max", type=float, default=11.0)
    pn.add_argument("--host-amp", type=float, default=0.2, help="sine on every other host (leakage source)")
    pn.add_argument("--sep-fwhm", type=float, nargs="+", default=[1.5, 2.5, 4.0])
    pn.add_argument("--dmag", type=float, default=3.0, help="neighbour stars this much fainter than the host")
    pn.add_argument("--empty-mags", type=float, nargs="+", default=[10, 11, 12, 13, 13.5, 14, 14.5, 15])
    pn.add_argument("--per-mag", type=int, default=3)
    pn.add_argument("--empty-amp", type=float, default=0.05)
    pn.add_argument("--n-sine", type=int, default=1, help="empty-sky stars per magnitude that carry the sine (the last n)")
    pn.add_argument("--sine-mag-max", type=float, default=13.0)
    pn.add_argument("--bp-rp", type=float, default=0.9)
    pn.set_defaults(fn=cmd_plan_new)
    inn = sub.add_parser("inject-new")
    inn.add_argument("--plan", required=True)
    inn.add_argument("--raw-dir", required=True)
    inn.add_argument("--dark", required=True)
    inn.add_argument("--dark-binning", choices=("sum", "average"), default="sum")
    inn.add_argument("--gain", type=float, default=None, help="e-/ADU of the raw files (default: raw header gain, else the control run's)")
    inn.add_argument("--out-dir", required=True)
    inn.set_defaults(fn=cmd_inject_new)
    cn = sub.add_parser("compare-new")
    cn.add_argument("--plan", required=True)
    cn.add_argument("--injected-draft", required=True)
    cn.add_argument("--out-dir", required=True)
    cn.set_defaults(fn=cmd_compare_new)
    pt = sub.add_parser("plan-transients")
    pt.add_argument("--control-draft", required=True)
    pt.add_argument("--raw-dir", required=True)
    pt.add_argument("--out", required=True)
    pt.add_argument("--seed", type=int, default=20261025)
    pt.add_argument("--snr-min", type=float, default=10.0)
    pt.add_argument("--edge-px", type=float, default=80.0)
    pt.add_argument("--stamp-r-fwhm", type=float, default=6.5)
    pt.add_argument("--n-templates", type=int, default=12)
    pt.add_argument("--host-mag-min", type=float, default=9.0)
    pt.add_argument("--host-mag-max", type=float, default=11.0)
    pt.add_argument("--host-amp", type=float, default=0.0)
    pt.add_argument("--dmag", type=float, default=3.0)
    pt.add_argument("--bp-rp", type=float, default=0.9)
    pt.add_argument("--mags", type=float, nargs="+", default=[10, 11, 12, 13, 14])
    pt.set_defaults(fn=cmd_plan_transients)
    ct = sub.add_parser("compare-transients")
    ct.add_argument("--plan", required=True)
    ct.add_argument("--injected-draft", required=True)
    ct.add_argument("--out-dir", required=True)
    ct.set_defaults(fn=cmd_compare_transients)
    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
