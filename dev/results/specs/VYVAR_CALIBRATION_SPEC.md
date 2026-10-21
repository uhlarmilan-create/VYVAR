# VYVAR Calibration Specification (CAL-2026-10)

Agreed with Milan 2026-10-09 (decisions K1-K9 + refinements). This file is the
contract for the calibration code, the GUI and the night runner. English, ASCII.

## 0. Scope and hard rules

- CMOS cameras only, mono and one-shot-colour (OSC). **No bias frames anywhere**
  (no code path, no UI text, no report text).
- **No sigma clipping anywhere.** Master combination is a plain per-pixel
  median or mean.
- The GUI and the night runner build the **same calibration plan** with the same
  function and apply it with the same function. Any choice the GUI offers has a
  night-runner equivalent (config or CLI flag).
- Formula applied to every LIGHT: `cal = (light - dark) / flat_norm`.

## 1. Master kinds

| kind     | built from                         | used for          |
|----------|------------------------------------|-------------------|
| DARK     | darks at the LIGHT exposure        | LIGHT             |
| DARKFLAT | darks at the FLAT exposure         | FLAT (optional)   |
| FLAT     | flats (offset-corrected, see 3)    | LIGHT             |

## 2. Library

- Every master row is linked to a **camera** (EQUIPMENTS). FLAT is also linked
  to a **telescope** (TELESCOPE). DARK/DARKFLAT do not depend on the telescope
  (the link may be shown, it is not a match key).
- Every row records its **origin**: `VYVAR` (built by VYVAR) or `IMPORTED`
  (+ software name, e.g. SIPS, AstroImageJ, PixInsight, N.I.N.A., from header
  SWCREATE/CREATOR/PROGRAM/SOFTWARE/ORIGIN or typed by the user), the **combine method**
  (median / mean / unknown for imports), number of frames, capture date, and
  for FLAT the **offset handling** (darkflat / constant / none / legacy-unknown).
- Import of an external master requires the user to choose the camera (and the
  telescope for FLAT). For an imported FLAT the user states whether the offset
  was already removed, or lets VYVAR read it from the header (`VYOFFSUB`, or
  HISTORY "Dark frame subtracted." as written by C-Munipack -> removed).
  A folder can be imported at once (kind from IMAGETYP). Re-importing an
  identical file keeps the existing library copy.
- The GUI shows the library as a table with camera, telescope, kind, binning,
  exposure, gain, offset, temperature, filter, capture date, valid-until,
  origin and method.

## 3. Master building (Calibration Library tab only)

- Masters are made **only** in the Calibration Library tab (Build: Auto /
  DARK / DARKFLAT / FLAT). Scan Source and RUN VYVAR never build masters; raw
  calibration frames found next to the LIGHTs are only reported.

- Combine method: **median (default)** or mean, chosen by the user in the GUI;
  night runner uses the config value (default median). Stored in the header
  (`VYSTKMOD`) and in the library row.
- DARK, DARKFLAT: per-pixel median/mean of the raw frames.
- FLAT: for each raw flat `f_i`:
  1. subtract the offset (in this order of preference):
     a. master DARKFLAT matching camera/gain/offset/binning/temperature and the
        flat exposure;
     b. otherwise a **constant** offset P measured from DARK masters of the same
        camera/gain/offset/binning: P = intercept at t=0 of the median dark level
        vs exposure (>= 2 exposures), else the median of the shortest dark
        (recorded as an upper bound);
     c. otherwise nothing; the plan records the estimated error.
     A long-exposure dark is **never** subtracted from a flat.
  2. normalise `f_i` by its own median;
  3. combine (median/mean); normalise the master by its median.
  OSC: steps 2-3 normalise each CFA channel separately (equalised CFA).
- Header stamps: `VYOFFSUB` (darkflat|constant|none), `VYOFFADU` (P), source
  list, `NCOMBINE`, `VY_CDATE` (capture date), origin.
- Measured on Milan's QHY294MM library: P = 24.5 ADU on a 32 968 ADU flat
  (0.07 %), dark current 60 s vs 120 s = 0.000 ADU/s, impact 0.0 mmag; the
  rule matters for cameras with a large pedestal, and costs nothing otherwise.
- Legacy VYVAR flats (median of raw flats, no `VYOFFSUB`): subtracting a
  constant commutes with the median, so they are corrected at use time by
  subtracting P (2b) and renormalising; flagged `legacy-corrected`.

## 4. Matching a master to a LIGHT group

Keys must match exactly unless stated:

| kind     | keys                                                        |
|----------|-------------------------------------------------------------|
| DARK     | camera, gain, offset*, exposure, binning, temperature +- tol |
| DARKFLAT | camera, gain, offset*, flat exposure, binning, temp +- tol   |
| FLAT     | camera, telescope, filter, gain, binning                    |

\* offset compared only when known on both sides.

- **Binning**: the standard is masters in the same binning as the LIGHTs.
  Software rebinning of a 1x1 master to NxN is allowed as a fallback, never the
  reverse. DARK rebin uses the camera's binning mode (sum or mean), which VYVAR
  measures once from data and stores with the camera; if the mode is unknown,
  no DARK rebin is done (reported). FLAT rebin uses the mean (the mode cancels
  after normalisation).
- **Validity window**: a master with capture date D and validity V days is valid
  for a LIGHT with date L when `D - 1 day <= L <= D + V`. Outside the window it
  is **never used**, in the GUI or the night runner; the user must make new
  masters. (Validity per kind from config: dark 90 d, flat 200 d.)
- Optics changes (cleaning, refocus, new filter) are the observer's
  responsibility: make new flats.
- Among valid candidates: closest temperature (darks), then newest capture date.

## 5. Plan

- `build_calibration_plan(...)` returns, per LIGHT group (setup): chosen DARK,
  FLAT (+ offset handling, P), rebin factors and modes, validity status,
  messages. Status per group: OK / MISSING_DARK / MISSING_FLAT / EXPIRED /
  UNCALIBRATED (user choice) / PRECALIBRATED.
- **Masters next to the LIGHTs** (2026-10-09, Milan): master files in the
  source tree (name MD_/MF_/MASTER*, or NCOMBINE / HISTORY master) are
  recognised from their headers (IMAGETYP kind, EXPTIME, binning, gain, offset,
  temperature, filter, DATE-OBS, program from ORIGIN/SWCREATE). A master whose
  INSTRUME differs from the LIGHTs' INSTRUME is not used. They are matched with
  exactly the same rules as library masters (section 4).
- **Source priority**: `masters_from = lights` (default) tries the masters next
  to the LIGHTs first and the library as fallback; `library` the reverse. GUI:
  switch above Scan Source; night runner: `--masters-from lights|library`.
- **Scan Source is information only** - it writes nothing. RUN VYVAR (GUI and
  night runner, same code) imports the LIGHT-folder masters it uses into the
  library (origin IMPORTED + program, linked to the selected camera, and
  telescope for FLAT) and calibrates with the library copies.
- The GUI shows one plan table: LIGHT group, frames, status, DARK + source,
  FLAT + source, note; the night runner logs it and writes
  `calibration_plan.json` into the draft.

## 6. Apply

- On the raw frame (mono) or raw CFA (OSC): `(light - dark) / flat_norm`.
- OSC: after calibration, superpixel extraction (R, G=(G1+G2)/2, B) and binning;
  channel headers keep all calibration stamps; gain/read noise are not guessed.
- Pre-calibrated LIGHTs (made by another program) and deliberately
  uncalibrated LIGHTs go through the same branch with calibration skipped; the
  user is aware, and the plan, headers and report state it plainly. A run is
  never reported as calibrated when it was not.
- No hot-pixel flagging or interpolation (as SIPS / AIJ): the master DARK
  removes hot pixels from the LIGHTs. Residual hot pixels / cosmics are
  rejected later by star detection (DAO stage: point-source shape test against
  the image PSF), not by calibration.
