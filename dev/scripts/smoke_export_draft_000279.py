from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _print_safe(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(str(msg).encode("ascii", "backslashreplace").decode("ascii"))


def check(condition: bool, msg_ok: str, msg_fail: str, *, ok: list[str], errors: list[str]) -> None:
    if condition:
        ok.append(f"[OK] {msg_ok}")
        _print_safe(f"[OK] {msg_ok}")
    else:
        errors.append(f"[FAIL] {msg_fail}")
        _print_safe(f"[FAIL] {msg_fail}")


def main() -> None:
    # -- Konfiguracia --------------------------------------------------------
    DRAFT = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000279")

    # Najdi platesolve subdir automaticky
    platesolve_dirs = list((DRAFT / "platesolve").glob("*/photometry"))
    assert platesolve_dirs, "Nenajdeny photometry adresar v draft_000279"
    PHOT = platesolve_dirs[0]
    _print_safe(f"Photometry dir: {PHOT}")

    REPORTS = PHOT / "lightcurves_reports"
    AAVSO = REPORTS / "aavso"
    VARASTRO = REPORTS / "varastro"
    BO_CID = "1498613634033133184"

    errors: list[str] = []
    ok: list[str] = []

    # -- Test 1 - Adresare existuju -----------------------------------------
    check(AAVSO.is_dir(), "aavso/ adresar existuje", "aavso/ adresar chyba", ok=ok, errors=errors)
    check(
        VARASTRO.is_dir(),
        "varastro/ adresar existuje",
        "varastro/ adresar chyba",
        ok=ok,
        errors=errors,
    )

    # -- Test 2 - BO CVn subory existuju ------------------------------------
    bo_aavso = list(AAVSO.glob("BO_CVn_*.txt"))
    bo_varastro = list(VARASTRO.glob("BO_CVn_*.txt"))
    check(
        len(bo_aavso) >= 1,
        f"aavso/BO_CVn_*.txt existuje ({bo_aavso[0].name if bo_aavso else ''})",
        "aavso/BO_CVn_*.txt chyba",
        ok=ok,
        errors=errors,
    )
    check(
        len(bo_varastro) >= 1,
        f"varastro/BO_CVn_*.txt existuje ({bo_varastro[0].name if bo_varastro else ''})",
        "varastro/BO_CVn_*.txt chyba",
        ok=ok,
        errors=errors,
    )

    # -- Test 3 - VAR.ASTRO filter (RRAB a Mira nesmu byt vo varastro) -----
    ss_varastro = list(VARASTRO.glob("SS_CVn_*.txt"))
    r_varastro = list(VARASTRO.glob("R_CVn_*.txt"))
    check(
        len(ss_varastro) == 0,
        "SS_CVn (RRAB) nie je vo varastro/ - spravne",
        f"SS_CVn (RRAB) sa objavil vo varastro/: {ss_varastro}",
        ok=ok,
        errors=errors,
    )
    check(
        len(r_varastro) == 0,
        "R_CVn (Mira/M) nie je vo varastro/ - spravne",
        f"R_CVn (M) sa objavil vo varastro/: {r_varastro}",
        ok=ok,
        errors=errors,
    )

    # SS_CVn musi byt v aavso
    ss_aavso = list(AAVSO.glob("SS_CVn_*.txt"))
    check(
        len(ss_aavso) >= 1,
        "SS_CVn (RRAB) je v aavso/ - spravne",
        "SS_CVn (RRAB) chyba v aavso/",
        ok=ok,
        errors=errors,
    )

    # -- Test 4 - AAVSO hlavicka BO CVn ------------------------------------
    if bo_aavso:
        txt = bo_aavso[0].read_text(encoding="utf-8", errors="replace")
        check("#OBSCODE=" in txt, "AAVSO: #OBSCODE=", "AAVSO: chyba #OBSCODE=", ok=ok, errors=errors)
        check(
            "#TYPE=Extended" in txt,
            "AAVSO: #TYPE=Extended",
            "AAVSO: chyba #TYPE=Extended",
            ok=ok,
            errors=errors,
        )
        check("#DATE=BJD" in txt, "AAVSO: #DATE=BJD", "AAVSO: chyba #DATE=BJD", ok=ok, errors=errors)
        check("ENSEMBLE" in txt, "AAVSO: CNAME=ENSEMBLE", "AAVSO: chyba ENSEMBLE", ok=ok, errors=errors)
        check(",CV," in txt, "AAVSO: FILTER=CV", "AAVSO: chyba FILTER CV", ok=ok, errors=errors)
        check(",NO," in txt, "AAVSO: TRANS=NO", "AAVSO: chyba TRANS=NO", ok=ok, errors=errors)
        # KMAG - nesmie byt vsade "na" (aspon prvy datovy riadok ma KMAG)
        data_lines = [l for l in txt.splitlines() if l and not l.startswith("#")]
        if data_lines:
            first = data_lines[0]
            fields = first.split(",")
            kmag = fields[10] if len(fields) > 10 else "na"
            check(
                kmag != "na",
                f"AAVSO: KMAG vyplnene ({kmag})",
                "AAVSO: KMAG='na' v prvom datovom riadku",
                ok=ok,
                errors=errors,
            )
        # Pocet datovych riadkov
        check(
            len(data_lines) >= 10,
            f"AAVSO: {len(data_lines)} datovych riadkov (min 10)",
            f"AAVSO: len {len(data_lines)} datovych riadkov",
            ok=ok,
            errors=errors,
        )

    # -- Test 5 - VAR.ASTRO hlavicka + COMP TABLE BO CVn -------------------
    if bo_varastro:
        txt = bo_varastro[0].read_text(encoding="utf-8", errors="replace")
        check("BJD(TDB)" in txt, "VAR.ASTRO: BJD(TDB) v hlavicke", "VAR.ASTRO: chyba BJD(TDB)", ok=ok, errors=errors)
        check(
            "Milan Uhlar" in txt,
            "VAR.ASTRO: Observer Milan Uhlar",
            "VAR.ASTRO: chyba observer",
            ok=ok,
            errors=errors,
        )
        check("# COMP TABLE" in txt, "VAR.ASTRO: COMP TABLE sekcia", "VAR.ASTRO: chyba COMP TABLE", ok=ok, errors=errors)
        check("Broeg" in txt, "VAR.ASTRO: Broeg referencia", "VAR.ASTRO: chyba Broeg ref", ok=ok, errors=errors)
        check(
            "Color system: Gaia BP-RP" in txt,
            "VAR.ASTRO: Color system Gaia BP-RP",
            "VAR.ASTRO: chyba color-system BP-RP riadok",
            ok=ok,
            errors=errors,
        )
        check(
            ("tier_weight" in txt) or ("Tier" in txt),
            "VAR.ASTRO: tier popis",
            "VAR.ASTRO: chyba tier info",
            ok=ok,
            errors=errors,
        )

        # COMP TABLE - pocet C0x riadkov
        comp_lines = [l for l in txt.splitlines() if l.startswith("# C") and l[3:5].isdigit()]
        check(
            len(comp_lines) >= 10,
            f"VAR.ASTRO: COMP TABLE ma {len(comp_lines)} riadkov (min 10)",
            f"VAR.ASTRO: COMP TABLE ma len {len(comp_lines)} riadkov - ocakavane 12",
            ok=ok,
            errors=errors,
        )

        # Datove riadky - 4 stlpce
        data_lines = [l for l in txt.splitlines() if l and not l.startswith("#") and not l.startswith(" ")]
        if data_lines:
            first_fields = data_lines[0].split()
            check(
                len(first_fields) == 4,
                f"VAR.ASTRO: datovy riadok ma 4 stlpce ({data_lines[0][:60]})",
                f"VAR.ASTRO: datovy riadok ma {len(first_fields)} stlpcov (ocakavane 4)",
                ok=ok,
                errors=errors,
            )
            try:
                bjd_val = float(first_fields[0])
                check(
                    bjd_val > 2_400_000,
                    f"VAR.ASTRO: BJD hodnota OK ({bjd_val:.3f})",
                    f"VAR.ASTRO: BJD hodnota podozriva ({bjd_val})",
                    ok=ok,
                    errors=errors,
                )
            except ValueError:
                errors.append("[FAIL] VAR.ASTRO: BJD nie je cislo")
                print("[FAIL] VAR.ASTRO: BJD nie je cislo")

    # -- Test 6 - Field image ----------------------------------------------
    field_imgs = list(VARASTRO.glob("BO_CVn_*_field.png"))
    check(
        len(field_imgs) >= 1,
        f"varastro/BO_CVn_*_field.png existuje ({field_imgs[0].name if field_imgs else ''})",
        "varastro/BO_CVn_*_field.png chyba",
        ok=ok,
        errors=errors,
    )

    # -- Test 7 - Konzistencia: comparison_stars_per_target vs comp_quality JSON
    comp_csv = PHOT / "comparison_stars_per_target.csv"
    json_path = PHOT / "lightcurves" / f"comp_quality_{BO_CID}.json"

    if comp_csv.exists() and json_path.exists():
        comp_df = pd.read_csv(
            comp_csv,
            dtype={"catalog_id": str, "name": str, "target_catalog_id": str},  # Gaia ID musi byt str - float64 straca cifry
        )
        bo_comp = comp_df[comp_df["target_catalog_id"] == BO_CID]

        data = json.loads(json_path.read_text(encoding="utf-8"))
        good_json = [k for k, v in data.items() if v == "good"]

        check(
            len(bo_comp) >= 10,
            f"comparison_stars_per_target.csv: {len(bo_comp)} comp pre BO CVn (min 10)",
            f"comparison_stars_per_target.csv: len {len(bo_comp)} comp - mozna nekonzistencia s JSON",
            ok=ok,
            errors=errors,
        )
        check(
            len(good_json) >= 10,
            f"comp_quality JSON: {len(good_json)} good comp (min 10)",
            f"comp_quality JSON: len {len(good_json)} good comp",
            ok=ok,
            errors=errors,
        )

        import datetime

        mtime_csv = datetime.datetime.fromtimestamp(comp_csv.stat().st_mtime)
        mtime_json = datetime.datetime.fromtimestamp(json_path.stat().st_mtime)
        diff_min = abs((mtime_csv - mtime_json).total_seconds()) / 60
        check(
            diff_min < 10,
            f"Subory z rovnakeho runu (rozdiel {diff_min:.1f} min)",
            f"Subory z roznych runov! comparison_stars={mtime_csv:%H:%M:%S}, "
            f"comp_quality={mtime_json:%H:%M:%S}, rozdiel={diff_min:.1f} min",
            ok=ok,
            errors=errors,
        )

    # -- Zaver --------------------------------------------------------------
    print(f"\n{'='*60}")
    _print_safe(f"Vysledok: {len(ok)} OK, {len(errors)} FAIL")
    if errors:
        _print_safe("\nFailed testy:")
        for e in errors:
            _print_safe(f"  {e}")
        raise SystemExit(2)
    else:
        _print_safe("Vsetky testy presli!")


if __name__ == "__main__":
    main()

