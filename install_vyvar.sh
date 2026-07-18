#!/usr/bin/env bash
# VYVAR installer for Linux (bash twin of install_vyvar.ps1).
#
# Same seven phases, each echoing [OK]/[FAIL]:
#   1 PYTHON  2 VENV  3 CATALOGS  4 PATHS  5 VALIDATE  6 SMOKE  7 FINISH
# Best-effort: the primary target is Windows (install_vyvar.ps1); this twin keeps
# a Linux box runnable. Idempotent: healthy .venv reused, existing catalog files
# verified rather than recopied, config.json rewritten through the canonical writer.
#
# Usage:
#   ./install_vyvar.sh                 # interactive
#   NONINTERACTIVE=1 ./install_vyvar.sh
#   CATALOG_SOURCE=/mnt/vyvar ./install_vyvar.sh
#   DATA_ROOT=/data/vyvar ./install_vyvar.sh
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT}"
VENV_DIR="$REPO_ROOT/.venv"
VENV_PY="$VENV_DIR/bin/python"
CONFIG_PATH="$REPO_ROOT/config.json"
NONINTERACTIVE="${NONINTERACTIVE:-0}"
CATALOG_SOURCE="${CATALOG_SOURCE:-}"

FAILURES=0
ok()   { printf '[OK]   %s\n' "$1"; }
warn() { printf '[WARN] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1"; FAILURES=$((FAILURES+1)); }
info() { printf '       %s\n' "$1"; }
head2(){ printf '\n== %s ==\n' "$1"; }

ask() { # ask "prompt" "default"
    local prompt="$1" def="$2" ans
    if [ "$NONINTERACTIVE" = "1" ]; then printf '%s' "$def"; return; fi
    read -r -p "$prompt [$def]: " ans
    if [ -z "$ans" ]; then printf '%s' "$def"; else printf '%s' "$ans"; fi
}

fsize() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0; }
gb() { awk -v b="$1" 'BEGIN{ printf "%.2f GB", b/1073741824 }'; }

# Catalog manifest: "name|cfg_arg|dest|src1[,src2]|approx_bytes". The zaloha
# (G<=16 subset) set is the anchor catalog and the recommended copy (~12.1 GB).
MANIFEST=(
  "Gaia DR3 catalog (G<=16 subset)|--gaia-db|GAIA_DR3/vyvar_gaia_dr3.db|GAIA_DR3/zaloha/vyvar_gaia_dr3.db,GAIA_DR3/vyvar_gaia_dr3.db|10066063360"
  "Blind-solve fine index|--blind-fine|GAIA_DR3/gaia_triangles_fine.pkl|GAIA_DR3/zaloha/gaia_triangles_fine.pkl,GAIA_DR3/gaia_triangles_fine.pkl|1406291841"
  "Blind-solve wide index|--blind-wide|GAIA_DR3/gaia_triangles_wide.pkl|GAIA_DR3/zaloha/gaia_triangles_wide.pkl,GAIA_DR3/gaia_triangles_wide.pkl|706509280"
  "AAVSO VSX variables|--vsx-db|VSX/vyvar_vsx_local_v2.db|VSX/vyvar_vsx_local_v2.db|908324864"
  "NASA exoplanet archive|--exoplanet-db|exoplanets/vyvar_exoplanet_local.db|exoplanets/vyvar_exoplanet_local.db|2334720"
)

declare -A CHOSEN_PATHS
CATALOG_MODE="skip"

# --- PHASE 1 PYTHON --------------------------------------------------------
PY_CMD=""
phase_python() {
    head2 "1/7 PYTHON (require 3.12)"
    if command -v python3.12 >/dev/null 2>&1; then
        PY_CMD="python3.12"; ok "found: $(python3.12 --version 2>&1)"; return 0
    fi
    if command -v python3 >/dev/null 2>&1 && python3 --version 2>&1 | grep -q '3\.12\.'; then
        PY_CMD="python3"; ok "found: $(python3 --version 2>&1)"; return 0
    fi
    fail "Python 3.12 not found."
    info "Install it, then re-run. E.g. (Debian/Ubuntu with deadsnakes):"
    info "  sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update"
    info "  sudo apt install python3.12 python3.12-venv"
    return 1
}

# --- PHASE 2 VENV ----------------------------------------------------------
phase_venv() {
    head2 "2/7 VENV + dependencies"
    if [ -x "$VENV_PY" ]; then
        ok ".venv already present - reusing"
    else
        info "Creating .venv..."
        "$PY_CMD" -m venv "$VENV_DIR" || { fail "venv creation failed."; return 1; }
        ok ".venv created"
    fi
    info "Upgrading pip..."
    "$VENV_PY" -m pip install --upgrade pip >/dev/null 2>&1 || warn "pip upgrade non-zero (continuing)."
    info "Installing requirements.txt (large wheels; several minutes)..."
    "$VENV_PY" -m pip install -r "$REPO_ROOT/requirements.txt" || { fail "pip install failed."; return 1; }
    ok "requirements installed"
    if "$VENV_PY" -m pip check; then ok "pip check clean"; else warn "pip check reported issues (usually safe)."; fi
    return 0
}

# --- PHASE 3 CATALOGS ------------------------------------------------------
phase_catalogs() {
    head2 "3/7 CATALOGS"
    local total=0 line approx
    for line in "${MANIFEST[@]}"; do
        approx="${line##*|}"; total=$((total+approx))
    done
    info "Catalog set (recommended copy): ~$(gb "$total") total."
    for line in "${MANIFEST[@]}"; do
        IFS='|' read -r name _arg _dest _src approx <<<"$line"
        info "  - $(printf '%-32s' "$name") ~$(gb "$approx")"
    done

    local choice="1"
    if [ -n "$CATALOG_SOURCE" ]; then choice="1"
    elif [ "$NONINTERACTIVE" = "1" ]; then choice="3"
    else
        echo ""
        echo "  [1] Copy from an existing VYVAR installation (recommended, default)"
        echo "  [2] Build from sources (hours to days, large downloads)"
        echo "  [3] Skip for now (LIMITED MODE)"
        choice="$(ask 'Choose catalog option' '1')"
    fi

    case "$choice" in
        1) catalog_copy ;;
        2) catalog_build ;;
        *) CATALOG_MODE="skip"
           warn "Catalogs SKIPPED -> LIMITED MODE (no Gaia matching until catalogs arrive)." ;;
    esac
}

catalog_copy() {
    CATALOG_MODE="copy"
    local src_root; src_root="$(ask 'Source VYVAR root (existing installation)' '/mnt/vyvar')"
    [ -d "$src_root" ] || { fail "source root not found: $src_root"; return; }
    local target_root; target_root="$(ask 'Target data root (catalogs copied under here)' "$DATA_ROOT")"

    local line name arg dest src approx found cand copy_bytes=0
    local -a plan=()
    for line in "${MANIFEST[@]}"; do
        IFS='|' read -r name arg dest src approx <<<"$line"
        found=""
        IFS=',' read -ra cands <<<"$src"
        for cand in "${cands[@]}"; do
            if [ -f "$src_root/$cand" ]; then found="$src_root/$cand"; break; fi
        done
        if [ -z "$found" ]; then warn "source missing for $name (skipping)"; continue; fi
        local b; b="$(fsize "$found")"
        plan+=("$name|$arg|$found|$target_root/$dest|$b")
        copy_bytes=$((copy_bytes+b))
    done
    [ "${#plan[@]}" -gt 0 ] || { fail "no catalog files found under source root."; return; }

    local free; free="$(df -P "$target_root" 2>/dev/null | awk 'NR==2{print $4*1024}')"
    if [ -n "$free" ]; then
        info "Copy size: $(gb "$copy_bytes"); free on target: $(gb "$free")"
        if [ "$free" -lt "$copy_bytes" ]; then fail "not enough free space on target."; return; fi
    fi

    local it iname iarg isrc idest ibytes got
    for it in "${plan[@]}"; do
        IFS='|' read -r iname iarg isrc idest ibytes <<<"$it"
        mkdir -p "$(dirname "$idest")"
        if [ -f "$idest" ] && [ "$(fsize "$idest")" = "$ibytes" ]; then
            ok "$iname already present + verified ($(gb "$ibytes"))"
        else
            info "Copying $iname ($(gb "$ibytes"))..."
            cp -f "$isrc" "$idest"
            got="$(fsize "$idest")"
            if [ "$got" != "$ibytes" ]; then fail "size mismatch for $iname ($got of $ibytes)"; continue; fi
            ok "$iname copied + size-verified"
        fi
        CHOSEN_PATHS["$iarg"]="$idest"
    done
}

catalog_build() {
    CATALOG_MODE="build"
    warn "BUILD path: large downloads, hours-to-days. Not the quick-setup path."
    info "  python GAIA_DR3/build_gaia_catalog.py"
    info "  python GAIA_DR3/build_blind_index.py"
    info "  python VSX/vsx_make.py"
    info "  python exoplanets/exoplanet_make.py"
    info "Then re-run this installer and choose option 1."
}

# --- PHASE 4 PATHS ---------------------------------------------------------
phase_paths() {
    head2 "4/7 PATHS (write machine-local config.json)"
    info "Location/telescope/camera facts are set in the app (Settings), not here."
    local archive calib db
    archive="$(ask 'Archive root' "$DATA_ROOT/Archive")"
    calib="$(ask 'Calibration library root' "$DATA_ROOT/CalibrationLibrary")"
    db="$(ask 'Main database path' "$DATA_ROOT/vyvar.sqlite3")"

    local -a a=(dev/scripts/apply_install_config.py --config "$CONFIG_PATH"
                --archive-root "$archive" --calibration-root "$calib" --database-path "$db")
    local arg
    for arg in "${!CHOSEN_PATHS[@]}"; do a+=("$arg" "${CHOSEN_PATHS[$arg]}"); done
    "$VENV_PY" "${a[@]}" || { fail "writing config.json failed."; return 1; }
    ok "config.json paths written (author absolute paths removed)"
    return 0
}

# --- PHASE 5 VALIDATE ------------------------------------------------------
phase_validate() {
    head2 "5/7 VALIDATE config.json"
    "$VENV_PY" "$REPO_ROOT/dev/scripts/validate_config.py" "$CONFIG_PATH" \
        || { fail "validate_config.py reported errors."; return 1; }
    ok "config.json valid"; return 0
}

# --- PHASE 6 SMOKE ---------------------------------------------------------
phase_smoke() {
    head2 "6/7 SMOKE (import app + database self-init)"
    "$VENV_PY" - <<'PYEOF'
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "src_py"))
import app  # import-only; main() runs only under `streamlit run`
from config import AppConfig
from database import VyvarDatabase
cfg = AppConfig()
db = VyvarDatabase(cfg.database_path)
tables = {r[0] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
db.conn.close()
missing = {"EQUIPMENTS","TELESCOPE","LOCATION","OBSERVATION","OBS_DRAFT"} - tables
assert not missing, f"DB self-init missing tables: {sorted(missing)}"
print(f"SMOKE OK: app import + DB self-init at {cfg.database_path} ({len(tables)} tables)")
PYEOF
    if [ $? -ne 0 ]; then fail "smoke run failed."; return 1; fi
    ok "app imports and the database self-initialises"
    return 0
}

# --- PHASE 7 FINISH --------------------------------------------------------
phase_finish() {
    head2 "7/7 FINISH - next steps"
    echo "Start the app from this folder:"
    echo "    streamlit run app.py"
    echo ""
    echo "Then, in the app:"
    echo "    1) Settings -> create your Location, Telescope and Equipment, then select them"
    echo "       (the DB ships with the author example rows, e.g. location 'Dablice' - do not"
    echo "        submit under those)."
    echo "    2) Import your first night and run the pipeline."
    echo ""
    info "Full walk-through:  VYVAR_INSTALL_GUIDE_CZ.pdf"
    info "Every config key:   docs/VYVAR_CONFIG_GUIDE_CZ.md / docs/VYVAR_CONFIG_GUIDE_EN.md"
    info "Install reference:   INSTALL.md"
    if [ "$CATALOG_MODE" = "skip" ]; then
        echo ""
        warn "LIMITED MODE: catalogs skipped - add them (re-run installer, option 1)."
    fi
}

# --- DRIVER ----------------------------------------------------------------
echo "==========================================="
echo " VYVAR installer (Linux / bash)"
echo "  repo: $REPO_ROOT"
echo "==========================================="

cd "$REPO_ROOT" || exit 1
phase_python  || { fail "Stopping: Python 3.12 required."; exit 1; }
phase_venv    || { fail "Stopping: environment setup failed."; exit 1; }
phase_catalogs
phase_paths   || { fail "Stopping: could not write config.json."; exit 1; }
phase_validate|| { fail "Stopping: config.json invalid."; exit 1; }
phase_smoke   || { fail "Stopping: smoke run failed."; exit 1; }
phase_finish

echo ""
if [ "$FAILURES" -eq 0 ]; then echo "INSTALL COMPLETE."; exit 0
else echo "INSTALL FINISHED WITH $FAILURES WARNING/FAILURE MARKER(S) - review [FAIL] lines above."; exit 1; fi
