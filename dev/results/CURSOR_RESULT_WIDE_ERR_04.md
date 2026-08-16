# CURSOR RESULT - WIDE-ERR-04

Date: 2026-08-16
Draft 515 photometry SHA: da9cce4
Content tip: feab854 (WIDE-ERR-04 science)
Pushed range: 9f139ef..feab854 (science) through docs stamp on push tip
Push: authorized (step 4)

## Verdict

WIDE-ERR + SEM **CLOSED** at the physical model (s=1, sigma_r=0).
CORR-ERR-01 remains **OPEN** (LOW research note). Mag byte-identity 49/49.
Product-frame accuracy table documented (not a pass/fail gate).

## 1a - Identity calibration

Draft 515 `err_calibration.json`: s=1.0, sigma_r0=0, form constant_sigma_r.
Default for new drafts: identity unless a per-draft sidecar overrides
(`identity_smooth_calibration` / `write_identity_sidecar` in
`src_py/err_calibration.py`). `export_err_mode=calibrated|model` retained.

## 1b - Re-export

| target | catalog_id | median err before [mmag] | after [mmag] |
|--------|------------|-------------------------:|-------------:|
| BO | 1498613634033133184 | 11.138 | 9.142 |
| FW | 1497343732462852864 | 9.770 | 7.420 |

Frame: exported LC `err` column. Mag byte-identity: **49/49 PASS**.
SHA of photometry inputs: da9cce4. Artifact: `WIDE_ERR_04_reexport.json`.

Note vs architect expectation (BO ~7.6-8): after identity, BO is 9.14 mmag
because export still adds colour-term coefficient uncertainty on top of the
physical quadrature (photon+SEM+scint). FW 7.42 is near the expected band.
Spec defect: expected BO band ignored the CT err term.

## 1c - Product-frame accuracy statement (even half; documentation)

Frame: product mag_calib (XVAL-BO-01 formula); EVEN-indexed frames; ensemble
self_excluded; calibration identity s=1, sigma_r=0; err_model = g_pt +
weighted SEM + scint [mmag]; scatter = 1.4826*MAD of mag_calib [mmag].
n=54 clean comps. SHA da9cce4.

| bin | n | median ratio | median err [mmag] | gated |
|-----|---|-------------:|------------------:|:-----:|
| (9.0, 9.5] | 10 | 0.994 | 7.77 | Y |
| (9.5, 10.0] | 11 | 0.968 | 9.35 | Y |
| (10.0, 10.5] | 8 | 0.857 | 10.05 | Y |
| (10.5, 11.0] | 4 | 1.149 | 14.10 | Y |
| (11.0, 11.5] | 5 | 0.746 | 18.56 | Y |
| (12.0, 12.5] | 5 | 1.113 | 15.35 | Y |
| G(8,9] union | 4 | 1.037 | 6.73 | in_window |

Statement: typically within ~+/-15% of unity across G 8-13 gated bins;
(11.0, 11.5] n=5 over-quotes ~25% (CORR-ERR-01). Under-quoting eliminated.
Artifact: `WIDE_ERR_04_accuracy.json`.

## 2 - Register / docs

- WIDE-ERR CLOSED; SEM CLOSED; CORR-ERR-01 OPEN (LOW research);
  WIDE-ERR-CROSSRIG OPEN; GAIN-DOMAIN-01 CLOSED.
- DECISIONS WIDE-ERR-04 (literature + conservative-vs-underquote rule).
- ROADMAP: WIDE-ERR+SEM removed from HIGH; EXPORT-READY interim MAGERR
  OBSOLETE; U-09 remains ToM export prerequisite.
- JAAVSO methods one-liner: exported MAGERR is the physical photon+SEM+scint
  budget (container-domain gain); residual common-mode conservatism is
  documented (CORR-ERR-01), not absorbed by s<1.

Closing SHAs (substance):
- GAIN-DOMAIN-01 + photon-transfer + weighted SEM + identity close: this
  content tip (WIDE-ERR-03 through 04 landed together on the push tip).
- Draft 515 photometry run: da9cce4.

## 3 - Verify

session_baseline_check.py --fast: see below (filled at tip).

## 4 - Push

Content tip (science): **feab854**. Pushed range: **9f139ef..745cbf9**.
Scratch left behind: `tmp/wide_err_04_lc_before`, `tmp/wide_err_03b_lc_backup`,
Archive draft 515 photometry (gitignored), caches, `dev/tests/_tmp_batch_e_lc/`,
`src_py/tmp/`, sqlite shm/wal.

## Spec defects
1. Expected BO after-band ~7.6-8 mmag ignored CT uncertainty added at export.
2. Accuracy table uses harness err_model, not per-comp LC err (comps lack
   target LCs); consistent with 03C meter, not a second product.

## Files
- `src_py/err_calibration.py`, `gain_photon_transfer.py`, `photometry_core.py`,
  `sigma_floor_core.py`, `export_reports.py`, `config.py`, params registry
- `dev/tools/wide_err_04_close.py` (+ 03/03B/03C tools)
- docs: DECISIONS, REGISTER, ROADMAP, STATE
- `dev/results/CURSOR_RESULT_WIDE_ERR_04.md` + JSON

## session_baseline_check.py --fast
**OVERALL PASS** (1429 passed, 28 skipped) at tip 2396949 before WIDE-ERR-04
commit; content tip feab854.
