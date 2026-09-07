# PythonPhot vendor provenance (EPSF-XVAL-A1)

Upstream: https://github.com/djones1040/PythonPhot
License: BSD-3-Clause (see LICENSE.rst; upstream `licenses/LICENSE.rst`)
ASCL: 1501.010
Upstream commit SHA: `7992eb6ba98ee946a4b8d27eb55ce0028f76374e` (2023-07-05)

This tree is a **minimal** copy of the DAOPHOT/Stetson lineage used by
the EPSF-XVAL-A1 harness. It is **dev-only**. `src_py` must never import
`dev.xval_pythonphot` or this vendor package.

## Algorithm (literature check)

`pkfit.py` / `pkfit_norecenter.py` are ports of the IDL Astronomy Users
Library routine adapted from the official DAO version of 1985 January 25
(Stetson 1987, PASP 99, 191 lineage): weighted least-squares / chi-square
fit of `scale * (integrated Gaussian core + residual lookup table)`.
`getpsf.py` builds that Gaussian + LUT from PSF stars. This is
Stetson-style chi-square PSF fitting, not an Anderson & King ePSF.

## Modules copied (no `__init__.py` from upstream)

getpsf.py, pkfit.py, pkfit_norecenter.py, pkfit_noise.py, aper.py,
dao_value.py, rdpsf.py, daoerf.py, mmm.py, pixwt.py, make_2d.py,
rinter.py, rebin.py, LICENSE.rst

Upstream `PythonPhot/__init__.py` was **not** copied (it pulls
`_astropy_init`). A thin local `__init__.py` is package-only.

## Py3 / modern-stack compatibility edits (line-level)

All other files are byte copies of the upstream SHA above except the
edits listed here. None change the Stetson/DAO least-squares math.

1. `getpsf.py:413` (upstream `getpsf.py:412`)
   - Was: `if type(goodstar) == np.int:`
   - Now: `if isinstance(goodstar, (int, np.integer)):`
   - Why: `np.int` was removed in NumPy 1.24. Same scalar-vs-array
     branch.

2. `getpsf.py:419` and `getpsf.py:427` (upstream `getpsf.py:418` and `:426`)
   - Was: `hdu.writeto(psfname, clobber=True)`
   - Now: `hdu.writeto(psfname, overwrite=True)`
   - Why: astropy.io.fits dropped `clobber` in favor of `overwrite`.
     Same overwrite semantics.

3. `aper.py:25-27` (upstream `aper.py:24` used `np.asfarray` in the
   name bind). Local `asfarray(a)` is `np.asarray(a, dtype=np.float64)`.
   Call site still `aper.py:161` (`skyrad = asfarray(skyrad)`).

4. `aper.py:29-34` plus call sites `aper.py:148`, `:155`, `:169`
   (upstream `np.iterable` at those lines). Local `_iterable` is the
   NumPy 1.x implementation (`try: iter(y)`). No change to aperture
   photometry.

5. `pkfit_norecenter.py:226` and `:436` (upstream `:226` used
   `(i_tofit / ixx).astype(int)`; `:435` used `good / ixx`).
   Now `//` floor division. In Python 2 `/` on integer index arrays
   was floor division; Python 3 `/` is true division and raises
   `IndexError`. Same pixel-index mapping as DAO/IDL.

## Deliberate non-edits

- Docstring examples still say `from PythonPhot import ...` (not
  executable).
- `pkfit.py` `if fnoise:` (ndarray boolean) is unchanged. The harness
  omits `noiseim` so `fnoise` is `None` and the test is safe. Weights
  come from `ronois` / `phpadu` (Stetson path).
- Relative imports (`. import daoerf`, etc.) were already present
  upstream and are kept.

## Usage constraint

PSF FITS written by `getpsf` must land under a sandbox / tmp path.
Never write into `Archive/`.
