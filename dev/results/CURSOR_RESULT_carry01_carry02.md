CURSOR RESULT - CARRY-01 + CARRY-02 (2026-07-30)

What I did
Part A (CARRY-01) read-only diagnosis complete. STOP GATE A  awaiting Milan clearance before Part B.

---

## PART A  CARRY-01 diagnosis

### A1  Producers (file:line)

#### NEW profile (`session_20260730_preprocess_profile/preprocess_profile.csv`)

| Role | Location |
|------|----------|
| Emitter script | `dev/scripts/post453_preprocess_bench.py` |
| Timing step `mask+fit+eval (combined)` | `profile_one_frame()` lines **3840** (`_fit_subtract_preprocess_sky_surface`) |
| Metric `max_abs_diff` | `byte_compare()` lines **7280**, returned line **86** |
| CSV writer | `main()` lines **106113** |
| Committed as evidence | commit **`83ee002`** (copied from bench run that overwrote post453 folder during release gates; see sync session transcript) |

Introduced in commit **`4103e56`** with the collapsed schema from the first commit of this script.

#### OLD profile (`session_20260727_post453/preprocess_profile.csv`)

| Role | Location |
|------|----------|
| **No committed Python producer** with the 7-step breakdown |  |
| CSV committed as session evidence | commit **`4103e56`**, file `dev/results/context/session_20260727_post453/preprocess_profile.csv` |
| Assembled manually during POST-453 session | agent transcript [POST-453 consolidation](9b47e6eb-9139-43f6-8a19-e31adbdb38aa)  `Write` to preprocess_profile.csv with 7-step rows + `max_abs_diff_data` |
| Methodology documented | `dev/results/CURSOR_RESULT_post453.md` lines **4648** |
| Per-frame identity measurements (451 vs 452) | ad-hoc inline Python during POST-453-fixes session ? `dev/results/context/session_20260728_post453_fixes/frame001_investigation.csv` (commit **`6c0a524`**) |

`git log --all --oneline -S "max_abs_diff_data"` ? **`4103e56`**, **`6c0a524`**, **`c7d27de`** (docs only).  
`git log --all --oneline -S "source masking (DAOStarFinder bbox)"` ? same; **no `.py` file ever contained that step label**.

---

### A2  What the new metric compares (from code)

Source: `byte_compare()` in `dev/scripts/post453_preprocess_bench.py`.

| Question | Answer |
|----------|--------|
| **Array A (reference)** | `Archive/Drafts/draft_000452/calibrated/lights/NoFilter_60_2/BO_CVn_Light_*.fits`  draft **452 calibrated** data as stored on disk (lines **5859**, **74**) |
| **Array B (after bench run)** | Copy of the same 452 file after `_qc_enrich_calibrated_in_place(bench_root)` (lines **6469**, **75**) |
| **Comparison type** | **Same-draft idempotency check**: 452 output vs 452 re-preprocessed in place. **Not** the old cross-draft check (451-cal input + preprocess vs 452 output). |
| **Units** | **ADU**  raw float32 FITS data arrays differenced as float64 (lines **7778**); not normalised. |
| **Frames included** | First **10** sorted `BO_CVn_Light_*.fits` (lines **5859**); **frame001 included**, no exclusion logic. |
| **Reduction** | Per frame: `max(abs(diff))` over finite pixels (lines **7880**); report: **global max** over all frames (`max_diff = max(max_diff, )`, line **80**). |

Timing block in the new profile uses `profile_one_frame()` on **452 Light_001** only (lines **2250**); QC timing in `byte_compare` covers all 10 frames.

---

### A3  Per-frame recomputation

See **`dev/results/CARRY01_per_frame_diff.md`**.

**Status: BLOCKED**  `draft_000452` FITS absent locally (`Archive/` gitignored; only `draft_000435` remains).

Committed aggregate for new comparison: **`max_abs_diff = 508.969482421875`** (`session_20260730_preprocess_profile/preprocess_profile.csv`).

Proxy table for the **old** comparison (451+preprocess vs 452) is in CARRY01 file  frames 002010 are **0.0 ADU**, frame001 **533.450 ADU**.

---

### A4  Verdict

**Primary finding: the new and old profiles measure different quantities.**

| Profile | Comparison | Reported diff | frame001 handling |
|---------|------------|--------------:|-------------------|
| Old (`post453` session) | draft **451** calibrated ? new preprocess vs draft **452** calibrated output | `max_abs_diff_data = 0.0` (9/10 identical) | **Excluded** from aggregate; note cites ~660 ADU pre-preprocess 451/452 cal input mismatch |
| New (`60730` session) | draft **452** calibrated vs draft **452** re-preprocessed (double sky-surface pass) | `max_abs_diff = 508.97` | **Included**; no exclusion field |

**Verdict branch: cannot assign V1/V2/V3 on the new comparison without live per-frame recompute.**

Inferred (pending draft restore):

- **Likely V1-variant** for the new comparison: frame001-only residual from re-preprocessing already-preprocessed 452 data (`_qc_enrich_one_frame` always re-applies sky surface when `sky_order > 0`, `pipeline.py` **1694616947**  no idempotency guard).
- **Not** same metric, lost exclusion alone  the comparison **method changed** between the hand-written old CSV and `post453_preprocess_bench.py`.

**660 ADU vs 508.97 ADU  same quantity?**

| Value | What it measures |
|------:|------------------|
| **~660 ADU** | Pre-preprocess calibrated **input** difference: draft 451 vs draft 452 pixel arrays before any new preprocess (`frame001_investigation.csv`: **659.646** for frame001). |
| **533.45 ADU** | Cross-draft identity: draft **451** cal input + **single** new preprocess vs draft **452** cal output (frame001). |
| **508.97 ADU** | Same-draft re-preprocess: draft **452** cal vs draft **452** after **second** in-place preprocess pass (aggregate from new profiler). |

**No  660 and 508.97 are not the same quantity.** 660 is raw cal-input mismatch; 508.97 is idempotency residual on 452-only data. 508.97 is closer to 533.45 (same order of magnitude, different mechanism).

---

## STOP GATE A

**Posted. Gate A partially cleared for A5.**

---

## PART A5 — Idempotency reachability (code only)

### A5.1 — Idempotency marker

**Write path** (`_qc_enrich_one_frame`, after sky subtract):

| Marker | file:line | When written |
|--------|-----------|--------------|
| `VY_SKYSF` | `pipeline.py:16959` | `sky_stats["sky_surface_applied"]` is true |
| `VYSKYORD` | `pipeline.py:16960-16963` | same |
| `VYSKYP2P` | `pipeline.py:16964-16969` | when p2p finite |
| `VYVARPR` | `pipeline.py:16976` | **always** on any QC-enrich pass (sky or not) |
| `qc_metrics.csv` column `sky_surface_applied` | row `**sky_stats` at `16991-17000`, CSV at `17163-17167` | when subtract ran |

**Read path before subtract: none.** Gate at `pipeline.py:16946-16947`:

```python
if sky_order > 0 and not is_mosaic:
    data, sky_stats = _fit_subtract_preprocess_sky_surface(data, order=sky_order)
```

No read of `VY_SKYSF`, `VYSKYORD`, `VYVARPR`, or `qc_metrics.csv`.

**Verdict: marker exists but is never checked** (different defect class from no marker at all).

Note: `apply_sky_surface` on `_qc_enrich_calibrated_in_place` is discarded at `pipeline.py:17045`.

---

### A5.2 — Caller trace

#### `_qc_enrich_calibrated_in_place`

| file:line | Entry point | Twice on same calibrated dir? |
|-----------|-------------|-------------------------------|
| `pipeline.py:17236` | **UI** `_vyvar_execute_preprocess_pending` (`app.py:837`, `:857`; jobs at `:1536-1538`, `:1542-1543`; RUN VYVAR `:587`) | **Yes** — in-place, no skip |
| `pipeline.py:17236` | **Headless** `_night_run_preprocess` (`night_run.py:273`, `:288`) | **Yes** if preprocess/MAKE MASTERSTAR re-run on existing draft without recalibration |
| `pipeline.py:18310` | `quick_preprocess_last_import(run=True)` | **Yes** if called (prod uses `run=False`) |
| `pipeline.py:16893` | **OSC extraction** (fresh channel FITS) | **Yes** if extraction re-run on same paths |
| `dev/scripts/post453_preprocess_bench.py:69` | Bench | Intentional |
| `dev/tests/*` | Tests | Bench-only |

#### `_fit_subtract_preprocess_sky_surface` (production)

Only via `_qc_enrich_one_frame` at `pipeline.py:16947`.

`analyze_calibrated_qc` (`pipeline.py:17252`) does **not** subtract sky surface.

Resume notes: `PROCESSED` draft status written after preprocess (`app.py:897`) but **not** checked before re-entry. No `VY_SKYSF` skip anywhere in `src_py/`.

---

### A5.3 — Reachability verdict: **R2 — REACHABLE**

**Concrete production sequences:**

1. **UI preprocess twice:** `kind == "preprocess"` job executed twice on same draft (`app.py:1536-1538`) without recalibration.
2. **UI MAKE MASTERSTAR twice:** each job calls `_vyvar_execute_preprocess_pending` first (`app.py:1542-1543`; RUN VYVAR `:587`) — second run on same draft subtracts again.
3. **Interrupted re-run:** preprocess completes (`VY_SKYSF` set); operator re-triggers preprocess or MAKE MASTERSTAR — all frames get second pass.

---

### A5.4 — Magnitude sanity (508.97 ADU; code only)

**A4 frame001-only attribution: UNVERIFIED inference, not measurement.**

| # | Candidate | file:line basis |
|---|-----------|----------------|
| C1 | Full order-N surface incl. constant on already-subtracted data | `16753-16759`, `16798-16799` |
| C2 | Different DAOStarFinder mask second pass | `16729-16751` |
| C3 | Different calm-pixel fit set (`calm_adu` gate) | `16755-16759` |
| C4 | First pass skipped (`sky_surface_applied: False`) — second pass is effectively first subtract | `16763-16768`, `16776-16781` |
| C5 | float32 write-back quantization | `16956` — unlikely alone for 508 ADU |
| C6 | Production R2 double-entry (any sequence above) | `16946-16947` |
| C7 | Bench idempotency compare (ref vs copy), not cross-draft identity | `post453_preprocess_bench.py:73-75` |

---

## STOP GATE A5

**Posted. Verdict R2 — Part B NOT started** (pipeline defect outranks profiler hygiene).

### D1 preview (for future Part B on R1)

- CARRY-01 inline recompute with absent drafts: no CSV.
- `post453_preprocess_bench.py` with missing `REF_LIGHT`: **exit 1, no CSV** (verified).
- Latent: `byte_compare()` with zero glob matches would still write `n_frames,0.0` / `max_abs_diff,0.0` if `main()` reached it (`:106-113`).

---

## PART B — CARRY-02 (blocked on R2)

*(Not executed)*
