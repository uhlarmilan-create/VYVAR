# CURSOR RESULT - HANG-P1-T16-01

Date: 2026-08-16
Draft: 515
Push: NO
Baseline tip at report: see git HEAD after local commit

## Verdict

The architect's Gaia box-scan hypothesis is **DEAD** on this machine.

- Production DB has `idx_ra`, `idx_dec`, and composite `idx_ra_dec`.
- Exact box query for target 16 coords: `SEARCH ... USING INDEX idx_ra_dec`, time **0.401 ms**.
- Target 16 has digit Gaia `catalog_id` and finite `bp_rp`; C2 count = **0** of 97.
- Live process is **idle** (~0 CPU); py-spy shows **no ScriptRunner / no photometry stack**.
- Last log line after Phase 1 target 16 is Variability **TESS skip + implied UI rerun**, not a Gaia stall.

Part D choice: **D3** (document; propose follow-up; do not implement D1/D2/E in this task).

---

## Part A - Live process evidence

### A1 - PID

| | |
|--|--|
| PID | **27300** |
| Command | `python.exe -m streamlit run .\app.py` |
| Parent | 32000 |
| Children | none (no separate RUN worker) |
| Relation | RUN is **in-process** on the Streamlit script path (`_run_vyvar_full_pipeline`), not a separate process |

### A2 - py-spy dump (verbatim)

Captured with `py-spy dump --pid 27300` while the UI still claimed a live RUN.
Saved also at `tmp/hang_p1_t16_pyspy_dump.txt`.

```
Process 27300: "C:\Users\uhlar\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run .\app.py
Python v3.12.10 (C:\Users\uhlar\AppData\Local\Programs\Python\Python312\python.exe)

Thread 8188 (idle): "MainThread"
    _select (selectors.py:314)
    select (selectors.py:323)
    _run_once (asyncio\base_events.py:1961)
    run_forever (asyncio\base_events.py:645)
    run_until_complete (asyncio\base_events.py:678)
    run (asyncio\runners.py:118)
    run (asyncio\runners.py:195)
    run (streamlit\web\bootstrap.py:466)
    _main_run (streamlit\web\cli.py:325)
    main_run (streamlit\web\cli.py:251)
    invoke (click\core.py:824)
    invoke (click\core.py:1269)
    invoke (click\core.py:1873)
    main (click\core.py:1406)
    __call__ (click\core.py:1485)
    <module> (streamlit\__main__.py:20)
    _run_code (<frozen runpy>:88)
    _run_module_as_main (<frozen runpy>:198)
Thread 22220 (idle): "ThreadPoolExecutor-0_0"
    task (streamlit\watcher\polling_path_watcher.py:90)
    run (concurrent\futures\thread.py:59)
    _worker (concurrent\futures\thread.py:93)
    run (threading.py:1012)
    _bootstrap_inner (threading.py:1075)
    _bootstrap (threading.py:1032)
Thread 21036 (idle): "ThreadPoolExecutor-0_1"
    _worker (concurrent\futures\thread.py:90)
    run (threading.py:1012)
    _bootstrap_inner (threading.py:1075)
    _bootstrap (threading.py:1032)
Thread 31112 (idle): "ThreadPoolExecutor-0_2"
    task (streamlit\watcher\polling_path_watcher.py:90)
    run (concurrent\futures\thread.py:59)
    _worker (concurrent\futures\thread.py:93)
    run (threading.py:1012)
    _bootstrap_inner (threading.py:1075)
    _bootstrap (threading.py:1032)
Thread 18200 (idle): "ThreadPoolExecutor-0_3"
    _worker (concurrent\futures\thread.py:90)
    run (threading.py:1012)
    _bootstrap_inner (threading.py:1075)
    _bootstrap (threading.py:1032)
Thread 31200 (idle): "Thread-1"
    wait (threading.py:355)
    get (queue.py:171)
    dispatch_events (watchdog\observers\api.py:379)
    run (watchdog\observers\api.py:213)
    _bootstrap_inner (threading.py:1075)
    _bootstrap (threading.py:1032)
Thread 20316 (idle): "Thread-9"
    read_directory_changes (watchdog\observers\winapi.py:334)
    read_events (watchdog\observers\winapi.py:380)
    _read_events (watchdog\observers\read_directory_changes.py:67)
    queue_events (watchdog\observers\read_directory_changes.py:70)
    run (watchdog\observers\api.py:158)
    _bootstrap_inner (threading.py:1075)
    _bootstrap (threading.py:1032)
```

Deciding fact: **no thread** is inside `photometry_core`, `comp_selection_per_target`, or sqlite. Native dump likewise shows MainThread in `select` / asyncio only. Windows reports ~38 OS threads, all Wait; Python-visible work is idle Streamlit.

### A3 - CPU / I/O

| Metric | Value |
|--------|-------|
| CPU over 60 s | delta **0.08 s** CPU / 60.0 s wall = **0.00 cores** (idle, not one-core busy) |
| Working set | ~777 MB |
| Gaia DB size | 53137264640 bytes (~53.1 GB) |
| Gaia LastAccessTime | **2026-08-16 1:55:28 PM** (local) = freeze window; no later access observed |
| handle.exe | not installed; module scan found no open sqlite/gaia names |

Idle + stale Gaia atime contradicts an ongoing full-table scan.

### A4 - Draft 515 mtimes

Newest science/log writes stopped at the freeze:

| Age at sample | File |
|---------------|------|
| ~88.6 min | `infolog_20260816_131411.txt` (LastWriteTime 1:55:27 PM) |
| ~94 min | `variability_candidates.csv` |
| ~96 min | `draft_manifest.json` |

Nothing under `draft_000515` was still being written during Part A.

### A5 - Kill request (Milan)

**Not killed by Cursor.** The RUN is already **not executing** (idle Streamlit only).

Milan: optional to stop Streamlit PID 27300 to clear a stale footer/"8/97" UI. That is cleanup, not unblocking a hung photometry thread. Please confirm if you want it stopped.

Infolog tail (frozen):

```
11:55:24  [COMP] 1498613634033133184: selected=5 T1=5 T2=0 T3=0 T4=0 note=color_rms_t1
11:55:24  [RUN VYVAR] Phase 1: target 16/97: ASASSN-V J140843.29+402701.8
11:55:27  INFO  [pipeline]  [TESS] preskocene - tess_enabled=False (Variability auto vetva)
```

The TESS line is from `ui_variability.py` (Variability auto branch), which sets `tess_auto_done` and calls `st.rerun()`. That is UI script-cycle activity after Phase 1 stopped, not Phase 1 Gaia code.

---

## Part B - Database premise

### B1 - PRAGMA index_list (verbatim)

Read-only URI on live `GAIA_DR3/vyvar_gaia_dr3.db`:

```
PRAGMA index_list(gaia_dr3):
(0, 'idx_teff', 0, 'c', 0)
  index_info: [(0, 11, 'teff_gspphot')]
(1, 'idx_parallax_snr', 0, 'c', 0)
  index_info: [(0, 10, 'parallax_over_error')]
(2, 'idx_g_mag', 0, 'c', 0)
  index_info: [(0, 3, 'g_mag')]
(3, 'idx_ra_dec', 0, 'c', 0)
  index_info: [(0, 1, 'ra'), (1, 2, 'dec')]
(4, 'idx_dec', 0, 'c', 0)
  index_info: [(0, 2, 'dec')]
(5, 'idx_ra', 0, 'c', 0)
  index_info: [(0, 1, 'ra')]
```

`idx_ra` / `idx_dec` / `idx_ra_dec` **exist**. Premise "indexes may be absent" does **not** hold here.

### B2 - EXPLAIN QUERY PLAN (target 16 coords)

Params from active_targets row: ra=212.180374, dec=40.450508, r=10 arcsec box (same SQL as `comp_selection_per_target.py`).

```
(5, 0, 163, 'SEARCH gaia_dr3 USING INDEX idx_ra_dec (ra>? AND ra<?)')
(46, 0, 0, 'USE TEMP B-TREE FOR ORDER BY')
```

**SEARCH ... USING INDEX**, not SCAN.

### B3 - Standalone timing

| Query | Result | Time |
|-------|--------|------|
| Box query (exact production SQL) | `(1498027456896444928, 1.9690628051757812)` | **0.401 ms** |
| `source_id` lookup | `(1.9690628051757812,)` | **0.015 ms** |

Hypothesis that target 16 is stuck on a full Gaia scan is **DEAD**. Return to Part A: the real location is "Phase 1 script no longer running."

### B4 - DB size

| | |
|--|--|
| Rows `gaia_dr3` | **211712600** (COUNT ~2.3 s) |
| File size | **53137264640** bytes |

---

## Part C - Why target 16?

### C1 - Target 16 row (Phase 1 order on `active_targets.csv`)

| Field | Value |
|-------|-------|
| phase1 index | 16 / 97 |
| vsx_name / name | ASASSN-V J140843.29+402701.8 |
| catalog_id | **1498027456896444928** (digit Gaia source_id: **yes**) |
| ra_deg | 212.180374 |
| dec_deg | 40.450508 |
| bp_rp | **1.969063** (finite: **yes**) |
| mag | 13.175178 |
| skip_reason | (empty / NaN) |
| zone_flag | linear |
| catalog | VSX |

It does **not** match the "no bp_rp AND no digit Gaia id" signature. It would take the source_id path (or already use CSV bp_rp), not the emergency box path for missing id/colour.

### C2 - Shared signature count

Among 97 Phase 1 targets (`skip_reason != vsx_type_out_of_scope`):

**C2_count = 0**

A resumed run would **not** hang again on that missing-index/box branch for the same reason (branch not entered; indexes present anyway).

---

## Part D - Fix choice

| Option | Decision |
|--------|----------|
| D1 create indexes + startup guard | **Rejected** - indexes already present; B3 is sub-millisecond |
| D2 log before box query | **Deferred** - useful observability, but D3 applies; no implement without follow-up |
| D3 hang elsewhere | **Selected** - dump shows idle Streamlit; TESS Variability `st.rerun()` after Phase 1 stop |

### Proposed follow-up (not implemented)

1. Treat this as **Streamlit script interruption / abandoned RUN**, not Gaia scan.
   Evidence: TESS Variability auto log + `st.rerun()` three seconds after Phase 1 T16 status; no ScriptRunner; CPU 0; no further Phase 1 lines.
2. Hardening candidates for a later task:
   - Run Phase 0+1+2A in a **subprocess / job worker** immune to UI reruns.
   - Or disable Variability auto-TESS / other `st.rerun()` paths while `vyvar_footer_state.running`.
   - Log an explicit `[RUN] interrupted` / finally-block line when Streamlit stops the script.
3. Resume draft 515 Phase 1+2A from a **headless** entry (or a fresh RUN after UI settle), not by waiting on the current idle Streamlit.

### Part E

**Not applied** (D did not land a code fix). Still valid: progress every `n//12` caused stale "8/97" while log reached 16/97 - false alarm amplifier; include in the follow-up with D2.

---

## Spec defects (named)

1. Architect hypothesis assumed missing `idx_ra`/`idx_dec` - **false** on production DB (also has `idx_ra_dec`).
2. Assumed target 16 lacks finite bp_rp / digit Gaia id - **false** (both present; C2=0).
3. Assumed hang is between Phase 1 status and `[COMP] BP-RP=` inside Gaia enrich/box - **not confirmed**; live stack is idle UI, and the next log line is Variability TESS, not COMP.
4. Spec treated "frozen log >1.5 h" as proof of a live blocked thread - need py-spy first (done); here the process had already left photometry.
5. Part E UI "8/97" is explained by progress stride, but target 16 **does** fire the callback (`16 % 8 == 0`); stale 8/97 means the footer did not refresh for the 16/97 event (Streamlit interrupt / render), not only the stride bug.

---

## Commits / --fast

Local result commit only (no science/code change). `--fast` recorded below after commit.
