CURSOR RESULT — 2026-07-08 (EXCEPT-RETRIAGE-1)

What I did
Marked census tiers PROVISIONAL-MECHANICAL; evidence-based re-triage of tranche 1
science kernel (144 sites); TOP-10 T1 fix batch flagged. No code changes.

## Step 0
Census header: **PROVISIONAL-MECHANICAL** / **EVIDENCE** policy note added.

## Tranche 1 (EVIDENCE, 2026-07-08)

**Scope:** `photometry_core.py`, `comp_selection_per_target.py`, `psf_runner.py`,
`psf_photometry.py` — **144** sites.

| Tier | Count |
|------|------:|
| T1-SCIENCE | 51 |
| T2-INTEGRITY | 41 |
| T3-UI | 27 |
| T4-LEGIT | 25 |
| ? | 0 |

| Disposition | Count |
|-------------|------:|
| fix-now | 10 |
| narrow+log-ERROR | 75 |
| narrow+comment(T4) | 32 |
| delete-dead | 27 |

**Key reclassifications:** EXC-0212 ? T3 (Streamlit perf cache); EXC-0123/0131/0152/0193
? T2/T4 (`con.close()` cleanup); EXC-0033–0042 ? T3 (BO CVn DEBUG prints).

## TOP-10 T1 (first fix batch)

1. **EXC-0132** — sky_pp=0 ? no sky subtraction on aperture flux path
2. **EXC-0166** — proc CSV cache miss ? frame dropped from all LCs
3. **EXC-0044** — frame skipped in comp BJD/flux accumulation
4. **EXC-0043** — proc CSV unreadable in phase1 comp cache
5. **EXC-0136** — comp-pool mag aggregation frame skip
6. **EXC-0455** — PSF grouped fit failure drops star silently
7. **EXC-0449** — ePSF trained on non-sky-sub cutout
8. **EXC-0452** — PSF local sky NaN ? wrong background
9. **EXC-0045** — comp detrend failure ? biased RMS map
10. **EXC-0198** — variable_targets WCS x/y refresh failure

Evidence source: `tmp/retriage1_evidence.json` (144 entries).

## Commit
(pending)
