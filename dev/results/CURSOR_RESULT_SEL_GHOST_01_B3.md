CURSOR RESULT - 2026-08-25 (SEL-GHOST-01 B-STOP-3)

What I did
T1 D1 radius tidy and T2 aperture SNR committed. Production-path
three-way on 516 (R0 live frozen / R1 c592ecf / R2 HEAD after T1/T2).
520 V0612 rms-ceiling cost measured (no wiring). Part A 2.5 honest
match rate redefined on DETECTED rows. No re-cut. No push. Live
516/520 read-only.

HEAD after T1/T2/INV-CAL: `6950495` on local main (ahead of
origin/main `b1f5b8c`). Session:
`dev/results/session_20260825_sel_ghost_01_b3/` plus Rule 0.2 copy
`dev/results/context/session_20260825_sel_ghost_01_b3/`.

## Premise (Rule 0.1)

**What is compared:** three production-path 516 chains from raw lights
through MASTERSTAR, matching, comps, 60 aperture LCs, AAVSO, VarAstro.
R0 is the live frozen draft_000516 products (no run). R1 is a fresh
full chain at c592ecf (pre-B1) in a worktree. R2 is a fresh full chain
at HEAD after T1/T2. T4 compares a 520 production-path photometry run
on the S6-retry/T1 sandbox MASTERSTAR against the 0.1 mag rms ceiling,
not against 516. T2-P1 joins the 67 unique comps of the 60
skip_photometry=False live 516 LC targets onto the T1/R2 MASTERSTAR
aperture SNR.

**How they differ:** R0 is a historical freeze; a fresh run has never
reproduced it. R1 lacks B1-S5 identity/D3 stamps. R2 has D1 radius
without solve-rms in the formula, D3 aperture SNR, and B1 name=cid
export. 520 is pre_calibrated (no cal_diag.json on the live draft).
Sandbox photometry from B-STOP-2 is not this experiment.

## Commits

| Hash | Item | What |
|------|------|------|
| `e410130` | T1 | D1 radius = max(12 arcsec, 3 x FWHM_dao_px x plate_scale); solve_rms_px stamp diagnostic only |
| `6e0fd5c` | T2 | MASTERSTAR snr = flux_ap/err_ap; snr_peak = peak_dao/sky sigma (not gated) |
| `6950495` | T4 harness | INV-CAL-01: pre_calibrated did not run dark calibration, so cal_diag is not required |

T2 columns: flux_ap from CircularAperture+annulus on the MASTERSTAR
image when (x,y) finite, else the table flux column. err_ap is not in
the live MS CSV; computed with the production empirical path
sqrt(F/g + sigma_bkg_ap^2), sigma_bkg_ap = bg_sigma * sqrt(pi r^2),
r = aperture_fwhm_factor * FWHM (1.9). Threshold stays 10.

## T1

516 sandbox post-gate catalog_id set identical to S3 (n=3583),
radius 152.32 arcsec, 101.7 s. 520 identical to S6 (n=111), radius
12.0 arcsec floor, 72.2 s. No IDs moved. `t1_516.json`, `t1_520.json`.

## T2-P1

**FALSE.** 65/67 pass D3 on aperture SNR. Failures:

| catalog_id | snr_ap | snr_peak | D3 |
|------------|--------|----------|----|
| 1500579870061241088 | 8.290 | 6.478 | FAIL (predicted star; still <10) |
| 1498964240802993408 | 7.572 | 17.90 | FAIL (peak would pass; aperture background) |

Do not retune. Full 67-row table:

| catalog_id | in_ms | snr_ap | snr_peak | d3_pass | source_state | vy_identity_gate | gaia_dao_resid_px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1485483541052276480 | True | 69.1067 | 224.827 | True | DETECTED_P1 | ok | 0.305168 |
| 1485609641291887872 | True | 66.3892 | 228.732 | True | DETECTED_P1 | ok | 0.475398 |
| 1496315070616056064 | True | 124.306 | 270.507 | True | DETECTED_P1 | ok | 0.540978 |
| 1496786967262035712 | True | 22.9665 | 39.5703 | True | DETECTED_P1 | ok | 0.203783 |
| 1496798993170465536 | True | 207.168 | 302.156 | True | DETECTED_P1 | ok | 0.131063 |
| 1496874408500111104 | True | 58.3111 | 133.911 | True | DETECTED_P1 | ok | 0.187674 |
| 1496948904709272576 | True | 224.767 | 523.4 | True | DETECTED_P1 | ok | 0.537219 |
| 1496984467038399744 | True | 191.926 | 491.89 | True | DETECTED_P1 | ok | 0.305045 |
| 1496994156484645632 | True | 173.022 | 317.598 | True | DETECTED_P1 | ok | 0.410954 |
| 1497014390074509312 | True | 94.3722 | 244.289 | True | DETECTED_P1 | ok | 0.578222 |
| 1497119157212720896 | True | 348.36 | 1397.65 | True | DETECTED_P1 | ok | 0.555799 |
| 1497145751650265600 | True | 170.36 | 352.076 | True | DETECTED_P1 | ok | 0.327684 |
| 1497196054307837696 | True | 157.468 | 349.291 | True | DETECTED_P1 | ok | 0.506867 |
| 1497203132413443328 | True | 277.95 | 779.662 | True | DETECTED_P1 | ok | 0.593429 |
| 1497313255374892800 | True | 163.137 | 440.425 | True | DETECTED_P1 | ok | 0.537113 |
| 1497326621313118976 | True | 74.9659 | 179.884 | True | DETECTED_P1 | ok | 0.598321 |
| 1497368849430107904 | True | 90.208 | 198.615 | True | DETECTED_P1 | ok | 0.47202 |
| 1497370563121917952 | True | 264.74 | 574.159 | True | DETECTED_P1 | ok | 0.377197 |
| 1497429837967596800 | True | 20.9042 | 51.9739 | True | DETECTED_P1 | ok | 0.613122 |
| 1497442379271632384 | True | 394.849 | 1343.6 | True | DETECTED_P1 | ok | 0.531654 |
| 1497528072458898432 | True | 292.837 | 813.994 | True | DETECTED_P1 | ok | 0.434209 |
| 1497613731286514432 | True | 466.574 | 1413.88 | True | DETECTED_P1 | ok | 0.537241 |
| 1497617407778562304 | True | 209.807 | 565.729 | True | DETECTED_P1 | ok | 0.443581 |
| 1497631048594737408 | True | 60.6861 | 154.033 | True | DETECTED_P1 | ok | 0.415303 |
| 1497674651102612992 | True | 286.095 | 939.705 | True | DETECTED_P1 | ok | 0.574052 |
| 1497676850124557312 | True | 53.0117 | 107.154 | True | DETECTED_P1 | ok | 0.571141 |
| 1497719902878053888 | True | 245.427 | 660.47 | True | DETECTED_P1 | ok | 0.603703 |
| 1497726465588057344 | True | 165.633 | 412.077 | True | DETECTED_P1 | ok | 0.695847 |
| 1497758935541325824 | True | 174.69 | 405.239 | True | DETECTED_P1 | ok | 0.630089 |
| 1497771992240531712 | True | 218.121 | 646.981 | True | DETECTED_P1 | ok | 0.593347 |
| 1497837207025312768 | True | 226.919 | 477.368 | True | DETECTED_P1 | ok | 0.573437 |
| 1497894828305963776 | True | 101.338 | 204.803 | True | DETECTED_P1 | ok | 0.608601 |
| 1497953377300128768 | True | 89.4629 | 203.822 | True | DETECTED_P1 | ok | 0.572072 |
| 1497974027502858240 | True | 68.445 | 181.427 | True | DETECTED_P1 | ok | 0.575471 |
| 1497976089087192832 | True | 128.273 | 247.519 | True | DETECTED_P1 | ok | 0.65933 |
| 1497977291678057472 | True | 24.7574 | 59.5314 | True | DETECTED_P1 | ok | 0.659059 |
| 1498020894186918144 | True | 151.657 | 557.176 | True | DETECTED_P1 | ok | 0.130941 |
| 1498062332030906880 | True | 229.52 | 457.087 | True | DETECTED_P1 | ok | 0.225647 |
| 1498222615914114816 | True | 25.5466 | 85.1557 | True | DETECTED_P1 | ok | 0.126124 |
| 1498244610443240192 | True | 27.4514 | 53.6367 | True | DETECTED_P1 | ok | 0.037425 |
| 1498326455340079616 | True | 253.263 | 633.639 | True | DETECTED_P1 | ok | 0.579574 |
| 1498626793812916480 | True | 100.465 | 212.89 | True | DETECTED_P1 | ok | 0.216714 |
| 1498645691668588288 | True | 13.214 | 26.6073 | True | DETECTED_P1 | ok | 0.208801 |
| 1498677611865494016 | True | 57.6238 | 115.468 | True | DETECTED_P1 | ok | 0.156339 |
| 1498735778606786816 | True | 296.379 | 740.382 | True | DETECTED_P1 | ok | 0.283748 |
| 1498812233320666368 | True | 140.642 | 323.685 | True | DETECTED_P1 | ok | 0.434481 |
| 1498964240802993408 | True | 7.57216 | 17.8993 | False | DETECTED_P1 | ok | 0.424141 |
| 1498974102047892224 | True | 24.1069 | 49.9966 | True | DETECTED_P1 | ok | 0.518946 |
| 1498994786610872576 | True | 101.826 | 171.163 | True | DETECTED_P1 | ok | 0.399041 |
| 1499200223486564608 | True | 217.593 | 625.962 | True | DETECTED_P1 | ok | 0.216638 |
| 1499867867561302272 | True | 30.1056 | 75.9967 | True | DETECTED_P1 | ok | 0.830109 |
| 1499906247391001088 | True | 413.358 | 1331.53 | True | DETECTED_P1 | ok | 0.145184 |
| 1500296402219939584 | True | 504.027 | 1211.37 | True | DETECTED_P1 | ok | 0.520415 |
| 1500355466608253184 | True | 99.3103 | 137.155 | True | DETECTED_P1 | ok | 0.232727 |
| 1500403089207449856 | True | 44.1084 | 93.7894 | True | DETECTED_P1 | ok | 0.587803 |
| 1500460813567859456 | True | 245.261 | 540.824 | True | DETECTED_P1 | ok | 0.579371 |
| 1500467303261764096 | True | 15.2104 | 18.3373 | True | DETECTED_P1 | ok | 0.798008 |
| 1500486102335278592 | True | 142.854 | 343.352 | True | DETECTED_P1 | ok | 0.526907 |
| 1500576537166572544 | True | 57.1191 | 116.525 | True | DETECTED_P1 | ok | 0.417527 |
| 1500579870061241088 | True | 8.2897 | 6.4783 | False | DETECTED_P1 | ok | 0.304933 |
| 1500602959805382656 | True | 10.5469 | 12.7855 | True | DETECTED_P1 | ok | 0.503541 |
| 1500664876053883904 | True | 163.239 | 221.015 | True | DETECTED_P1 | ok | 0.677814 |
| 1500688820495583488 | True | 81.4908 | 170.214 | True | DETECTED_P1 | ok | 0.093184 |
| 1500727513856914944 | True | 270.604 | 477.691 | True | DETECTED_P1 | ok | 0.162193 |
| 1501956561697995008 | True | 187.944 | 341.424 | True | DETECTED_P1 | ok | 0.575149 |
| 1502007585909464064 | True | 83.8976 | 112.645 | True | DETECTED_P1 | ok | 0.360064 |
| 1504489595970703872 | True | 103.549 | 267.858 | True | DETECTED_P1 | ok | 1.08427 |

## T3 run table (Rule 0.3)

| Run | What | copy_s | MS_s | phot_s | total_s | INV-CAL-01 | catalog_matched |
|-----|------|--------|------|--------|---------|------------|-----------------|
| R0 | live frozen; no run | - | - | - | - | - | frozen |
| R1 | c592ecf worktree | 1.2 | 83.2 | 2094.4 | 2178.8 | ok via cal_diag.json file presence (keys=0) | 3581 |
| R2 | HEAD after T1/T2 | 1.4 | 101.5 | 1470.4 | 1573.3 | same attachment | 3583 |
| T3-P5 | R2 freeze in session dir; photometry-only rerun | 1.9+1.7 | - | 1483.6 | ~1487 | cal_diag attached; err=None | 3583 |
| T4 | 520 production photometry | lights copy | - | 33.2 | 37.5 | WARN pre_calibrated skip (6950495) | - |

Positive controls: self-hash TRUE; two different live LCs differ TRUE.

### T3 R1 vs R0 (60 rows)

| target | ensemble_identical | n_comps_left | n_comps_right | ids_swapped | median_dmag_mmag | dRMS_mmag | LC_SHA_equal | AAVSO_VarAstro_SHA_equal | ms_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1485534187306501376 | Y | 8 | 8 |  | 0 | 61.3169 | N | N |  |
| 1485560025830226432 | Y | 8 | 8 |  |  |  | N | Y |  |
| 1485987151737107200 | Y | 8 | 8 |  | 44.461 | -56.6595 | N | N |  |
| 1485987254816323328 | Y | 8 | 8 |  | 0 | -10.15 | N | N |  |
| 1496037650087948160 | N | 8 | 8 | 1497125479404597248;1498974102047892224 |  |  | N | Y |  |
| 1496278752372040832 | Y | 8 | 8 |  | 0 | 0 | N | Y |  |
| 1496293286541396480 | Y | 8 | 8 |  | 0 | 0 | N | Y |  |
| 1496733984545821696 | N | 8 | 8 | 1497631048594737408;1500492871204256512 |  |  | N | Y |  |
| 1496795041799526400 | Y | 8 | 8 |  | 0 | 0 | N | Y |  |
| 1496998382733052928 | Y | 7 | 7 |  | 0 | 0 | N | Y |  |
| 1497096960821764224 | Y | 8 | 8 |  | 8.346 | 6.48585 | N | N |  |
| 1497132660589966976 | Y | 8 | 8 |  | 0 | 0 | N | Y |  |
| 1497154753901690624 | Y | 8 | 8 |  | 0 | 13.7336 | N | N |  |
| 1497169940906156032 | N | 8 | 0 | 1496315070616056064;1497429837967596800;1497442379271632384;1497674651102612992;1498244610443240192;1499200223486564608;1500460813567859456;1500576537166572544 |  |  | N | Y |  |
| 1497227287309482624 | Y | 3 | 3 |  | 0 | -1.09587 | N | N |  |
| 1497236186481686016 | Y | 8 | 8 |  | 0 | 0 | N | Y |  |
| 1497245497969274240 | Y | 3 | 3 |  | 0 | 0 | N | Y |  |
| 1497284015237511808 | N | 5 | 6 | 1500418379291011840 | 2.5095 | -22.8902 | N | N |  |
| 1497343732462852864 | Y | 8 | 8 |  | 0 | 0 | N | Y |  |
| 1497350638770267520 | Y | 8 | 8 |  | 0 | 0 | N | Y |  |
| 1497418258735289472 | Y | 8 | 8 |  | 0 | 0 | N | Y |  |
| 1497425371201155072 | Y | 8 | 8 |  | 0 | 0 | N | Y |  |
| 1497491273179203456 | Y | 8 | 8 |  | 0 | 0 | N | N |  |
| 1497561779362267392 | Y | 5 | 5 |  | 0 | 0 | N | Y |  |
| 1497603835681942400 | Y | 6 | 6 |  | 0 | 0 | N | N |  |
| 1497639123133258752 | Y | 5 | 5 |  | 0 | 0 | N | N |  |
| 1497683722074089728 | Y | 6 | 6 |  | 0 | 0 | N | Y |  |
| 1497683996951418880 | Y | 8 | 8 |  | 0.005 | -3.10723 | N | Y |  |
| 1497871669842349184 | Y | 6 | 6 |  | 0 | -44.4333 | N | N |  |
| 1498000793739050368 | Y | 8 | 8 |  | 0 | -156.804 | N | Y |  |
| 1498027456896444928 | Y | 8 | 8 |  | 0 | 0 | N | Y |  |
| 1498058827337611392 | Y | 8 | 8 |  | 0 | 0 | N | N |  |
| 1498278351706325248 | Y | 5 | 5 |  | 0 | 0 | N | N |  |
| 1498298211635183744 | Y | 8 | 8 |  | 0 | 0 | N | Y |  |
| 1498321301379345408 | Y | 8 | 8 |  | 0 | 5.34171 | N | Y |  |
| 1498425548825498112 | Y | 3 | 3 |  | 0 | 0 | N | Y |  |
| 1498486880958321024 | Y | 4 | 4 |  | 0 | 0 | N | Y |  |
| 1498613634033133184 | Y | 4 | 4 |  | 0 | 0 | N | Y |  |
| 1498617482323461376 | Y | 8 | 8 |  | 0 | 0 | N | Y |  |
| 1498699086702005376 | Y | 8 | 8 |  | 0 | 0 | N | Y |  |
| 1498752516095473664 | Y | 3 | 3 |  |  |  | N | Y |  |
| 1498783199341798016 | Y | 8 | 8 |  | 0 | 0 | N | N |  |
| 1498795809366255488 | Y | 7 | 7 |  | 0 | 0 | N | N |  |
| 1498804639818507904 | Y | 8 | 8 |  | 0 | 0 | N | Y |  |
| 1498842882207281152 | Y | 8 | 8 |  | 0 | -1.16728 | N | Y |  |
| 1499006984318088320 | Y | 8 | 8 |  | 0 | 0 | N | N |  |
| 1499021174889970816 | Y | 7 | 7 |  | 0 | 0 | N | Y |  |
| 1499081819828174080 | N | 8 | 8 | 1496874408500111104;1504489595970703872 | 0.77 | -90.1004 | N | N |  |
| 1499084499887740160 | Y | 8 | 8 |  | 0 | 0 | N | N |  |
| 1499209638054824320 | N | 8 | 8 | 1498964240802993408;1500418379291011840 | 0.5145 | -0.0214578 | N | Y |  |
| 1499210016011946496 | Y | 8 | 8 |  | 0 | 0.204776 | N | N |  |
| 1499842372636900992 | Y | 4 | 4 |  | 0.864 | -3.14018 | N | Y |  |
| 1500327978819506944 | Y | 8 | 8 |  | 0 | 0 | N | N |  |
| 1500410236033012352 | N | 8 | 8 | 1496874408500111104;1497145751650265600 |  |  | N | Y |  |
| 1500418894687086208 | N | 8 | 8 | 1498383080188862976;1500688820495583488 | 9.437 | -103.405 | N | N |  |
| 1500424804562041984 | Y | 8 | 8 |  | 0 | 0 | N | Y |  |
| 1500461157165243648 | Y | 8 | 8 |  | 0 | 0 | N | Y |  |
| 1500549977088828160 | Y | 5 | 5 |  | 0 | 0 | N | Y |  |
| 1500693841313325696 | Y | 8 | 8 |  | 0 | 0 | N | N |  |
| 1502012464992313088 | Y | 6 | 6 |  | 0 | -26.0268 | N | N |  |

### T3 R2 vs R1 (60 rows)

| target | ensemble_identical | n_comps_left | n_comps_right | ids_swapped | median_dmag_mmag | dRMS_mmag | LC_SHA_equal | AAVSO_VarAstro_SHA_equal | ms_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1485534187306501376 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1485560025830226432 | Y | 8 | 8 |  |  |  | N | Y | none (ensemble identical) |
| 1485987151737107200 | Y | 8 | 8 |  | 0 | 0 | N | N | none (ensemble identical) |
| 1485987254816323328 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1496037650087948160 | Y | 8 | 8 |  |  |  | N | Y | none (ensemble identical) |
| 1496278752372040832 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1496293286541396480 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1496733984545821696 | Y | 8 | 8 |  |  |  | N | Y | none (ensemble identical) |
| 1496795041799526400 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1496998382733052928 | Y | 7 | 7 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1497096960821764224 | Y | 8 | 8 |  | 0 | 0 | N | N | none (ensemble identical) |
| 1497132660589966976 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1497154753901690624 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1497169940906156032 | N | 0 | 8 | 1496315070616056064;1497429837967596800;1497442379271632384;1497674651102612992;1498244610443240192;1499200223486564608;1500460813567859456;1500576537166572544 |  |  | N | Y | other: swapped 1496315070616056064,1497429837967596800,1497442379271632384,1497674651102612992,1498244610443240192,1499200223486564608,1500460813567859456,1500576537166572544 (name the MS input; not adaptive re-selection) |
| 1497227287309482624 | Y | 3 | 3 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1497236186481686016 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1497245497969274240 | Y | 3 | 3 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1497284015237511808 | N | 6 | 5 | 1498964240802993408 | 2.459 | -0.0230339 | N | N | D3 predicate 1498964240802993408 D3 snr=7.572<10 |
| 1497343732462852864 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1497350638770267520 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1497418258735289472 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1497425371201155072 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1497491273179203456 | Y | 8 | 8 |  | 0 | 0 | N | N | none (ensemble identical) |
| 1497561779362267392 | Y | 5 | 5 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1497603835681942400 | Y | 6 | 6 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1497639123133258752 | Y | 5 | 5 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1497683722074089728 | Y | 6 | 6 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1497683996951418880 | Y | 8 | 8 |  | 0 | 0 | N | Y | none (ensemble identical) |
| 1497871669842349184 | Y | 6 | 6 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1498000793739050368 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1498027456896444928 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1498058827337611392 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1498278351706325248 | Y | 5 | 5 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1498298211635183744 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1498321301379345408 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1498425548825498112 | Y | 3 | 3 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1498486880958321024 | Y | 4 | 4 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1498613634033133184 | Y | 4 | 4 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1498617482323461376 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1498699086702005376 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1498752516095473664 | Y | 3 | 3 |  | 0 | 0 | N | Y | none (ensemble identical) |
| 1498783199341798016 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1498795809366255488 | Y | 7 | 7 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1498804639818507904 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1498842882207281152 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1499006984318088320 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1499021174889970816 | Y | 7 | 7 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1499081819828174080 | Y | 8 | 8 |  | 0 | 0 | N | N | none (ensemble identical) |
| 1499084499887740160 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1499209638054824320 | Y | 8 | 8 |  | 0 | 0 | N | Y | none (ensemble identical) |
| 1499210016011946496 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1499842372636900992 | Y | 4 | 4 |  | 0 | 0 | N | Y | none (ensemble identical) |
| 1500327978819506944 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1500410236033012352 | Y | 8 | 8 |  |  |  | N | Y | none (ensemble identical) |
| 1500418894687086208 | Y | 8 | 8 |  | 0 | 0 | N | N | none (ensemble identical) |
| 1500424804562041984 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1500461157165243648 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1500549977088828160 | Y | 5 | 5 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1500693841313325696 | Y | 8 | 8 |  | 0 | 0 | Y | Y | none (ensemble identical) |
| 1502012464992313088 | Y | 6 | 6 |  | 0 | 0 | Y | Y | none (ensemble identical) |

## T3-P1 .. P5 verdicts

| ID | Verdict | Evidence |
|----|---------|----------|
| T3-P1 | TRUE (drift exists). Identical LC SHA 0/60. Median abs dmag of the rest 0.0 mmag (46/54 finite dmag exactly 0; 7 with abs dmag > 0.01 mmag; max 44.46 mmag). SHA mismatch with dmag=0 is provenance / non-science columns plus real dRMS on a minority. R1 vs R0 also moves 8 ensembles. Fresh run does not reproduce the frozen anchor. |
| T3-P2 | FALSE for byte LC SHA. 58/60 ensembles identical; 45/58 of those have LC SHA equal; 13 do not. Aperture mag_calib is identical where finite (dmag=0, dRMS=0). SHA moves on AC overlay: ct_n_comp 2232 vs 2238 because D1 +2 catalog IDs change the auto-cal ref pool (ac_correction, ac_scatter, ac_n_ref, mag_calib_ac, mag_calib_final ~2.4 mmag). Four targets have all-NaN mag_calib in both R1 and R2. S1-S5 did not change the aperture ensemble kernel; they coupled into the AC ref pool. Not a silent photometry-kernel defect. |
| T3-P3 | Partial TRUE. Two ensemble changes, both named: (1) 1497284015237511808 6->5 comps, D3 predicate 1498964240802993408 snr_ap=7.572<10. (2) 1497169940906156032 R1 had 0 comps, R2 has 8 (same IDs as R0). R1 excluded CSS_J134925.3+393524 as no_dao_detection because MASTERSTAR name was DET_0784 at c592ecf; B1 export makes name=catalog_id so R2 recovers the target and the pinned ensemble. Not the 4-in/2-out D1 edge delta. Not adaptive re-selection. |
| T3-P4 | TRUE. R2 catalog_id set == S3 sandbox 3583. |
| T3-P5 | Science PASS / byte SHA FAIL. R2 MS carries vy_identity_gate, gaia_dao_resid_px, snr, snr_peak, flux_ap, err_ap; pipeline_meta has dao_gaia_tol (effective vs *_config_default) and match_sep_formula_inputs. Candidate snapshot written under the session dir (not Archive/Drafts). Photometry-only rerun err=None, INV-CAL-01 attached. Core SHA n=121 both sides; snap 360ef397... vs run 4f372c6c.... The only differing core file is comparison_stars_per_target.csv, column _dist_deg, 8 rows, ~1e-14 float noise. All lightcurve_*.csv byte-identical. A --full byte-SHA gate against this candidate would still FAIL until expected hashes are updated. |

Candidate snapshot (R2, session dir): core `360ef397dadae4175eb4f938507990dbc56da54af2973c24bf6627f3d06d0151` n=121; ext `c6299a894d308ef4de965c198149d261c595574777febbb14df7e4df167206be` n=180. Era03 frozen expected core 9902d918 n=121 / ext 472bc9e4 n=179.

## T4 (520 V0612; measure, no wiring)

Lights: **26 FITS** in draft and sandbox; **25 proc CSV** (B-STOP-2 loaded 25).
calibration_mode=pre_calibrated. INV-CAL-01 WARN after `6950495` (valid).
Selected ensemble still one star 1111737033143440768 G=13.87, pipeline
lc_rms=0.123, ooe=0.069. Forced ensemble of all 7 (script, no config
change): lc_rms=**0.053**, ooe=0.022 vs M3 bright-8 0.068 and one-comp 0.123.

phase01_comparison_max_comp_rms unchanged (0.1 on this T4 AppConfig).

| catalog_id | G | snr_ap | snr_peak | resid_px | comp_rms | verdict_ceiling_0.1 | photon_sigma_mag_from_snr | suspected_variable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1112113680298377344 | 7.6321 | 853.691 | 5366.85 | 0.267424 | 0.968725 | fail_ceiling | 0.00127181 | True |
| 1111920204908702336 | 10.0693 | 212.362 | 337.959 | 1.41163 | 4.33223 | fail_ceiling | 0.00511266 | True |
| 1112110695298081664 | 10.8974 | 171.889 | 224.205 | 2.34296 | 0.7577 | fail_ceiling | 0.00631649 | False |
| 1111749157833870208 | 11.2287 | 159.082 | 200.115 | 0.675088 | 3.81522 | fail_ceiling | 0.00682501 | True |
| 1112121862213003648 | 11.0963 | 155.835 | 184.585 | 1.85524 | 0.206738 | fail_ceiling | 0.00696722 | False |
| 1112121067641532160 | 11.6061 | 131.688 | 138.129 | 2.21646 | 0.174369 | fail_ceiling | 0.00824477 | True |
| 1111737033143440768 | 13.8699 | 43.0311 | 51.7428 | 0.442819 | 0.0511099 | pass_ceiling | 0.0252315 | True |

Photon-from-snr (1.086/snr_ap) vs selector comp_rms: 0.1 mag is far
above photon for the six bright pool stars (0.001-0.008 mag photon vs
0.17-4.3 mag empirical). The one star under the ceiling (G=13.87)
has photon 0.025 mag, selector rms 0.051, ceiling 0.1 (~4x photon).
0.1 mag is a rig constant on this night, not a photon-noise threshold.
Five of seven are also in suspected_variables.csv. Feeds COMP-RMS-DEF-01.
No decision here.

## Gates

| Gate | Result | Evidence |
|------|--------|----------|
| live 516 CSV | PASS | bfa24039778f437b2bf7ed37056b6b507e068d52d2c7b4a222a73002125b250a |
| live 516 FITS | PASS | 13e77cf8a1dcb4e73fae0558437d7234feeb70a5ae4aa85064a8316812b01345 |
| live 516 ePSF | PASS | 172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20 |
| live 520 CSV | PASS | 5ce9b07fe0490103b2e16f6fbe3b18ffc7cd987fbee8a334722cc2fd46c6a683 |
| no Archive/Drafts writes | PASS | sandbox under session dir only |
| --fast --clean | PASS | 1556 passed, 32 skipped; clean-tree worktree b1b_clean_d5e2c9da; 513 s; HEAD 6950495 |

## Errors

INV-CAL-01 on 516: B-STOP-2 sandbox omitted cal_diag.json. T3 copies
cal_diag.json, draft_manifest.json, sat_diag.json onto work_root and
the platesolve setup dir, the same way --full `_copy_frozen_anchor_inputs`
does. R1/R2 INV-CAL-01 ok via file presence.

INV-CAL-01 on 520: live draft has no cal_diag.json (pre_calibrated;
dark calibration never ran). Passing draft_id=520 set
calibration_mode=pre_calibrated, but check_cal_diag treated any mode
other than PASSTHROUGH/RAW as dark-applied and FAIL. That is not a
missing attachment; there is nothing to attach. Fix: `6950495` skips
INV-CAL-01 FAIL for pre_calibrated (WARN, same as no-dark). T4 retry
after that commit: err=None, INV-CAL-01 WARN. Invariant not bypassed.

T4 first retry also applied post-hoc D3 with FWHM 1.25 / resid ceil
3.75 px and counted 60 stars. The seven-star table uses
comparison_stars.csv from photometry-time D3 (ceil 7.5 px, n_out=7)
plus selector comp_rms. Forced-60 numbers in t4_520.json are not the
T4 table.

## T5

Part A 2.5 honest match rate is now DETECTED-only (DECISIONS
SEL-GHOST-01). B-STOP-2 P-520-4 restated: honest 1.0 / reported 0.082
on 742 (61/61 DETECTED-with-cid; 61/742 DAO retained).

## Re-cut proposal (Milan)

Re-cut proposal: candidate snapshot core `360ef397dadae417...` n=121,
ext `c6299a894d308ef4...` n=180 from R2; delta vs anchor explained
target by target: NO. R1 vs R0 is already 0/60 LC SHA (fresh vs freeze).
R2 vs R1 same-ensemble aperture mag is identical; SHA still moves on
the AC ref pool because matching added two catalog IDs. Two ensemble
membership changes are named (D3 snr 7.572; R1 no_dao_detection on
DET_0784). Authorizing a recut makes --full track R2. It does not make
R2 byte-reproduce the frozen anchor. No recut in this task. No push.
