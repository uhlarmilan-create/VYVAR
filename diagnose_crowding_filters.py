"""Diagnostika crowding/blend filtrov — kalibrácia parametrov pre draft_000193."""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import math

DRAFT = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000193")

# Poznámka: v tomto drafte je masterstars CSV uložené v platesolve root
# ako masterstars_full_match.csv (nie v photometry/).
MASTERSTAR_CSV = DRAFT / "platesolve" / "masterstars_full_match.csv"
COMP_CSV = DRAFT / "platesolve" / "NoFilter_120_2" / "photometry" / "comparison_stars_per_target.csv"
PERFRAME_DIR = DRAFT / "detrended_aligned" / "lights" / "NoFilter_120_2"


def _p(s: str = "") -> None:
    print(s)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ── 1. Masterstars: Filter A stĺpce ──
_p("=== Filter A: Gaia flagy v masterstars ===")
ms = pd.read_csv(MASTERSTAR_CSV, low_memory=False)
_p(f"  Masterstars riadkov: {len(ms)}")
for col in ("gaia_nss", "gaia_qso", "gaia_gal"):
    if col in ms.columns:
        n_true = int(
            (
                ms[col]
                .astype(str)
                .str.lower()
                .isin(
                    ["true", "1", "yes"],
                )
            ).sum()
        )
        _p(f"  {col}: PRÍTOMNÝ, True={n_true}/{len(ms)}")
    else:
        _p(f"  {col}: CHÝBA v masterstars CSV")


# ── 2. Per-frame CSV: Filter B stĺpce ──
_p()
_p("=== Filter B: PSF/FWHM stĺpce v per-frame CSV ===")
sample_csvs = sorted(PERFRAME_DIR.glob("proc_*.csv"))[:3]
for csv_path in sample_csvs:
    df = pd.read_csv(csv_path, nrows=5)
    has_chi2 = "psf_chi2" in df.columns
    has_fwhm = "fwhm_estimate_px" in df.columns
    _p(
        f"  {csv_path.name}: psf_chi2={'OK' if has_chi2 else 'CHÝBA'}, "
        f"fwhm_estimate_px={'OK' if has_fwhm else 'CHÝBA'}"
    )
    if has_chi2:
        header_cols = pd.read_csv(csv_path, nrows=0).columns
        use = ["psf_chi2", "psf_fit_ok"] if "psf_fit_ok" in header_cols else ["psf_chi2"]
        df2 = pd.read_csv(csv_path, usecols=use)
        chi2 = pd.to_numeric(df2["psf_chi2"], errors="coerce").dropna()
        if len(chi2) > 0:
            _p(
                f"    psf_chi2: mean={chi2.mean():.2f}, median={chi2.median():.2f}, "
                f"p90={np.percentile(chi2, 90):.2f}, p99={np.percentile(chi2, 99):.2f}"
            )
    if has_fwhm:
        df3 = pd.read_csv(csv_path, usecols=["fwhm_estimate_px"])
        fwhm = pd.to_numeric(df3["fwhm_estimate_px"], errors="coerce").dropna()
        if len(fwhm) > 0:
            _p(
                f"    fwhm_estimate_px: mean={fwhm.mean():.2f}, median={fwhm.median():.2f}, "
                f"p90={np.percentile(fwhm, 90):.2f}, max={fwhm.max():.2f}"
            )


# ── 3. Filter C: distribúcia vzdialeností susedov ──
_p()
_p("=== Filter C: Distribúcia vzdialeností medzi hviezdami v masterstars ===")
if "x" in ms.columns and "y" in ms.columns:
    xs = pd.to_numeric(ms["x"], errors="coerce").dropna().to_numpy()
    ys = pd.to_numeric(ms["y"], errors="coerce").dropna().to_numpy()
    # Náhodná vzorka 200 hviezd — spočítaj nearest neighbor vzdialenosť
    rng = np.random.default_rng(42)
    idx = rng.choice(len(xs), size=min(200, len(xs)), replace=False)
    nn_dists: list[float] = []
    for i in idx:
        dx = xs - xs[i]
        dy = ys - ys[i]
        dists = np.sqrt(dx * dx + dy * dy)
        dists[i] = np.inf
        nn_dists.append(float(dists.min()))
    nn = np.array(nn_dists)
    _p("  Nearest-neighbor vzdialenosť (px) medzi hviezdami v masterstars:")
    _p(
        f"  min={nn.min():.1f}, p5={np.percentile(nn, 5):.1f}, "
        f"p25={np.percentile(nn, 25):.1f}, median={np.percentile(nn, 50):.1f}, "
        f"p75={np.percentile(nn, 75):.1f}, p95={np.percentile(nn, 95):.1f}"
    )
    _p()
    # Koľko hviezd má suseda bližšie ako rôzne prahy?
    _p("  Počet hviezd s najbližším susedom < prah (contamination detection):")
    _p(f"  {'Prah (px)':>12}  {'Počet':>8}  {'%':>6}")
    for thr in (10, 15, 20, 25, 30, 40, 50):
        n = int((nn < thr).sum())
        _p(f"  {thr:>12}  {n:>8}  {100 * n / len(nn):>5.1f}%")
else:
    _p("  x/y stĺpce chýbajú v masterstars!")


# ── 4. Contamination ratio analýza ──
_p()
_p("=== Filter C: Analýza contamination ratio ===")
if "x" in ms.columns and "y" in ms.columns:
    # Flux proxy
    flux_col = None
    for fc in ("dao_flux", "flux"):
        if fc in ms.columns:
            flux_col = fc
            break
    mag_col = None
    for mc in ("phot_g_mean_mag", "catalog_mag", "mag"):
        if mc in ms.columns:
            mag_col = mc
            break

    if flux_col:
        fluxes = pd.to_numeric(ms[flux_col], errors="coerce").to_numpy()
    elif mag_col:
        mags = pd.to_numeric(ms[mag_col], errors="coerce").to_numpy()
        fluxes = np.where(np.isfinite(mags), 10 ** (-0.4 * mags), np.nan)
    else:
        fluxes = np.ones(len(ms))

    xs2 = pd.to_numeric(ms["x"], errors="coerce").to_numpy()
    ys2 = pd.to_numeric(ms["y"], errors="coerce").to_numpy()

    # Pre 200 náhodných hviezd: contamination ratio od najjasnejšieho suseda
    # v rôznych polomeroch
    rng = np.random.default_rng(42)
    idx2 = rng.choice(len(xs2), size=min(200, len(xs2)), replace=False)
    _p(
        f"  {'Polomer (px)':>14}  {'max_ratio=0.10':>16}  {'max_ratio=0.20':>16}  "
        f"{'max_ratio=0.50':>16}"
    )
    _p(
        f"  {'':>14}  {'% vylúčených':>16}  {'% vylúčených':>16}  {'% vylúčených':>16}"
    )
    for radius in (15, 20, 25, 30, 40):
        n_excl = {0.10: 0, 0.20: 0, 0.50: 0}
        for i in idx2:
            if not math.isfinite(float(fluxes[i])) or fluxes[i] <= 0:
                continue
            dx = xs2 - xs2[i]
            dy = ys2 - ys2[i]
            dists = np.sqrt(dx * dx + dy * dy)
            mask = (dists > 0.5) & (dists <= radius) & np.isfinite(fluxes)
            if not np.any(mask):
                continue
            max_ratio = float(np.nanmax(fluxes[mask] / fluxes[i]))
            for thr in (0.10, 0.20, 0.50):
                if max_ratio > thr:
                    n_excl[thr] += 1
        pct = {t: 100 * n / len(idx2) for t, n in n_excl.items()}
        _p(
            f"  {radius:>14}  {pct[0.10]:>15.1f}%  {pct[0.20]:>15.1f}%  "
            f"{pct[0.50]:>15.1f}%"
        )


# ── 5. Comp hviezdy: koľko zostalo per target ──
_p()
_p("=== Comp hviezdy po filtroch ===")
if COMP_CSV.exists():
    comp = pd.read_csv(COMP_CSV)
    id_col = "target_catalog_id" if "target_catalog_id" in comp.columns else comp.columns[0]
    counts = comp.groupby(id_col).size()
    _p(f"  Targetov s comp hviezdami: {len(counts)}")
    _p(
        f"  Comp per target: min={counts.min()}, median={counts.median():.0f}, "
        f"max={counts.max()}, mean={counts.mean():.1f}"
    )
    _p(f"  Targetov s < 5 comp: {(counts < 5).sum()}")
    _p(f"  Targetov s >= 5 comp: {(counts >= 5).sum()}")
else:
    _p(f"  {COMP_CSV} neexistuje — spusti najprv run_phase0_and_phase1")

_p()
_p("=== Odporúčané parametre pre tento dataset ===")
_p("  (vyplní sa po analýze vyššie)")
