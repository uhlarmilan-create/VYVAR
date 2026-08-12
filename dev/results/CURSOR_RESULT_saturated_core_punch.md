CURSOR RESULT - 2026-08-12 saturated star core punch investigation

## Verdict

No remaining code path on calibrate / preprocess / align alters saturated
star-core pixels. The historical punch was L.A.Cosmic
(`pipeline.py::_remove_cosmics_lacosmic` via astroscrappy), already removed in
commit `0ab686f` (2026-08-12). No further pixel-alteration site to delete.

## What produces white/low cores (historical)

| Item | Role |
|------|------|
| `pipeline.py::_remove_cosmics_lacosmic` (deleted in `0ab686f`) | **PIXEL ALTERATION** -- replaced sharp/peak pixels with interpolated background; headers `VY_COSM` / `VY_COSMNPX` |
| On-disk drafts 505/506 (`VY_COSM=True`) | Still carry CR-eaten cores; e.g. cal frame 012 star near (503,257): draft 508 peak ~47460 ADU vs draft 506 ~1847 ADU |

## Flagging kept (metadata only -- no pixel change)

| Site | What it does |
|------|----------------|
| `pipeline.py::_star_saturation_flags` / `_vectorized_star_saturation_columns` / `_saturated_core_plateau*` | Sets `likely_saturated`, `saturated_from_peak`, `saturated_plateau`, `photometry_ok` on catalogs |
| `pipeline.py` zone assignment (`zone="saturated"`, `is_saturated`) | Excludes saturated stars from comps / marks photometry |
| `photometry_core.enhance_catalog_dataframe_aperture_bpm` | `likely_nonlinear`, `on_bad_column` flags from dark BPM sidecar; does **not** rewrite light pixels |
| `importer.write_dark_bpm_json` | Writes `*_dark_bpm.json` sidecar only |

## Path audited (no peak/sat pixel rewrite found)

- `_calibrate_one_light_apply_masters_in_ram` -- dark/flat only; `nan_to_num` for non-finite after flat
- `_fit_subtract_preprocess_sky_surface` -- smooth polynomial sky; star-masked fit
- `_qc_enrich_one_frame` -- sky + QC headers; strips legacy `VY_COSM*` only
- `vyvar_alignment_frame` -- detection-copy clips only; science warp via astroalign / reproject / ndimage_shift
- BPM -- column flags in photometry catalogs only

## Verify (draft 508 detrend_aligned, post-lacosmic removal)

- Brightest cores solid: e.g. `BO_CVn_Light_001.fits` peaks at ~68566 ADU with center/max = 1.0
- Bright-ring local-min scan (rmed > 15000, center < 0.4*rmed): **0** hits on cal and aln
- Raw saturated plateaus (65535) remain uniformly high after dark+flat (within-plateau odd-low = 0)
- Drafts 505/506 still show CR-eaten sharp stars -- need Milan re-run from raw for those drafts

## Code change this arc

None. Fix already landed in `0ab686f`. Fresh BO CVn re-run still required for any draft whose calibrated/aligned FITS still have `VY_COSM=True`.
