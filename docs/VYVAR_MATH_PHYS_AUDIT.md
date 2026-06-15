# VYVAR — Mathematical / Physical Code Audit (first pass)

**Filed:** 2026-06-11 · **Line refs:** commit `ad6e788` (live tree).  
**Purpose:** verify physical/statistical computations against primary literature; flag sound vs
discuss vs follow-up. Byte-identity-neutral hygiene landed in same session (citations, BJD
warning guard, MAD constant).

---

## A. Verified sound

| Topic | Live refs | Literature |
|-------|-----------|------------|
| Pogson flux→mag | `comp_qa_core.py:52`; `check_star_kmag.py:439–446,493`; `psf_neighbor_sub.py:26` | Standard |
| BJD_TDB | `time_utils.py:124–151` (`astropy` UTC→TDB + barycentric LTT) | Eastman et al. 2010 |
| MAD→σ | `comp_qa_core.py:19,34` (`/0.6745` unified) | Gaussian consistency |
| σ_IQR | `comp_qa_core.py:112` (`/1.349`) | Sokolovsky 2017 |
| inv_nv (1/η) | `comp_qa_core.py:114–117` (`ddof=1`) | Sokolovsky 2017; von Neumann 1941 |
| Spike index | `comp_qa_core.py:118–123` | std vs robust σ |
| HRD distance modulus | `hrd_analysis.py:270` | Standard |
| Trust check scatter | `trust_flag_core.py:88` (`ddof=1`) | Consistent with comp-QA |

---

## B. Discuss (documented; deep physics parked)

### D1 — Ensemble combination is AIJ flux-sum, not Broeg-weighted ALC

- **Fn:** `ensemble_normalize` `photometry_core.py:2285`
- **Combination:** flux sum `photometry_core.py:2387–2394` (`-2.5 log10 Σ 10^{-0.4 m}`) — AIJ
  `tot_C_cnts` / Honeycutt 1992.
- **Broeg scope:** comp **selection** ordering (`comp_rms`) + **zeropoint** offset
  `1/rms²` at `photometry_core.py:2418–2420` — not `ens_med`.
- **Citation fix (2026-06-11):** `CITATIONS.bib` + `citations.py` — Broeg → selection/zeropoint;
  Collins 2017 (AIJ) + Honeycutt 1992 → combination. No numeric change.

### D2 — BJD requires mid-exposure JD

- **Primary path:** `compute_time_columns` → `mid_exposure_jd` (`time_utils.py:62–119`) adds
  `EXPTIME/2`; pipeline inserts `jd_mid` / `bjd_tdb_mid` (`pipeline.py:~8744`).
- **Guard (2026-06-11):** loud `log_event` when `EXPTIME`/`EXPOSURE` missing or ≤0; **JD
  unchanged** (still shutter-open) — surfaces silent offset risk without moving outputs.
- **Test:** `tests/test_time_utils_mid_exposure.py` — offset = `EXPTIME/2`.
- **Non-production parity:** `scripts/cross_validate_draft342.py:_mid_exposure_jd` duplicates
  logic (scratch script); no production caller passes raw `DATE-OBS` into `compute_hjd_bjd`
  without `mid_exposure_jd`.

### D3 — Extinction / colour / standard system

Known limitation (APCORR-COLOR parked; AAVSO #4 open). Same physics as D1 slope debate.
**Parked** — ROADMAP D3 + #4 workstream.

---

## C. Follow-up (second pass — parked)

- **Howell 1989** — aperture-path error budget + scintillation (gain/read-noise located in
  `psf_photometry.py`; aperture uncertainty not traced).
- **Stetson / aperture correction** — APCORR-MIXEDFRAME coupling.
- **Mighell 1999 χ²-gamma** — export-only; **new reduced-χ²/dof gate** scoped in
  `docs/VYVAR_SIGMA_BUDGET_SPEC.md` + `tmp/phase12/chi2_sigma_gate.py` (sandbox, 2026-06-15).
- **PM epoch** — `vyvar_platesolver.py:63` `GAIA_EPOCH=2016.0` → DR4 `J2017.5`.

---

## D. Minor (done)

- MAD constant unified to `/0.6745` (`_MAD_SCALE`) in `build_locus` (`comp_qa_core.py:154,161`).

---

## Citation integrity (`CITATIONS.bib` vs code)

| Key | Status | Code / export anchor |
|-----|--------|----------------------|
| `broeg2005` | **used** | `citations.py` core; zeropoint weights `photometry_core.py:~2418` |
| `collins2017` | **used** | `citations.py` core (2026-06-11) |
| `honeycutt1992` | **used** | `citations.py` core; `ensemble_normalize` flux sum |
| `howell1989` | **used** | `citations.py` core |
| `stetson1987` | **used** | `citations.py`; PSF/aperture context |
| `sokolovsky2017`, `vonneumann1941` | **used** | `comp_qa_core.py` indices; `citations.py` comp_qa |
| `henden_kaitchuck1982`, `aavso_ccd_guide` | **used** | `citations.py` core; comp selection policy |
| `eastman2010` | **used** | `time_utils.py`; `citations.py` |
| `gaia2023`, `lindegren2021` | **used** | Catalog ingest / `citations.py` |
| `barbary2016`, `bertin1996` | **used** | SEP/xval; `citations.py` |
| `astier2013`, `lacroix2025`, `guy2010` | **used** | PSF weights / exports when PSF on |
| `anderson2000`, `moffat1969` | **used** | PSF path |
| **`mighell1999`** | **export-only / aspirational** | `citations.py` PSF block only; **no χ²-gamma in production** |
| `pont2006`, `tamuz2005`, `savitzky1964`, `aigrain2004`, `hippke2024`, `marconi2026` | **used** | Conditional export when flags on |
| `lomb1976` … `stellingwerf1978` | **used** | Period analysis exports |
| `seager2003`, `ciardi2015` | **used** | GS11 dilution exports |
| `photutils`, `astropy2022`, `numpy2020`, `scipy2020`, `lightkurve2018`, `astroquery2019` | **used** | Software stack imports + exports |
| `watson2006` | **used** | VSX when configured |
| `riello2021` | **used** | Colour tiers when on |

---

## Parked deep work (ROADMAP)

1. **D1-combination** — Broeg-weighted `ens_med` re-test after colour/extinction (**moves anchor**).
2. **D3 + AAVSO #4** — extinction/colour → standard mags.
3. **C second pass** — Howell error budget + aperture correction audit.

---

## Suggested order (unchanged)

1. ~~D2 guard + test~~ **done (warning-only)**.
2. ~~D1 citation scoping~~ **done**.
3. D1+D3+#4 physics workstream.
4. C second pass.
