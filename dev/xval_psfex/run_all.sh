#!/usr/bin/env bash
# EPSF-XVAL-A2: two-pass SExtractor + PSFEx (container-tested settings).
# Usage: bash run_all.sh SNAPSHOT_DIR OUT_DIR
set -u
set -o pipefail

APT_LINE="sudo apt install source-extractor psfex python3-astropy python3-pandas python3-numpy"

die() { echo "FAIL: $*" >&2; exit 1; }

if [ "$#" -lt 2 ]; then
  die "usage: bash run_all.sh SNAPSHOT_DIR OUT_DIR"
fi

SNAPSHOT_DIR=$(cd "$1" && pwd) || die "SNAPSHOT_DIR not a directory: $1"
OUT_DIR=$(mkdir -p "$2" && cd "$2" && pwd) || die "cannot create OUT_DIR: $2"
KIT_DIR=$(cd "$(dirname "$0")" && pwd)
LOG="$OUT_DIR/run_log.txt"
: > "$LOG"
log() { echo "$*" | tee -a "$LOG"; }

find_sex() {
  for c in source-extractor sextractor sex; do
    if command -v "$c" >/dev/null 2>&1; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

SEX=$(find_sex) || die "SExtractor not found (tried source-extractor | sextractor | sex). apt line: $APT_LINE"
command -v psfex >/dev/null 2>&1 || die "psfex not found. apt line: $APT_LINE"
command -v python3 >/dev/null 2>&1 || die "python3 not found. apt line: $APT_LINE"
log "binaries: sex=$SEX psfex=$(command -v psfex) python3=$(command -v python3)"

# Layout (C-EXPORT-GAP / _copy_frozen_anchor_inputs):
#   SNAPSHOT_DIR/detrended_aligned/lights/<setup>/*Light_*.fits
#   SNAPSHOT_DIR/calibrated/lights/qc_metrics.csv
# Filenames are BO_CVn_Light_NNN.fits, not Light_*.fits. MASTERSTAR excluded.
QC=""
if [ -f "$SNAPSHOT_DIR/calibrated/lights/qc_metrics.csv" ]; then
  QC="$SNAPSHOT_DIR/calibrated/lights/qc_metrics.csv"
else
  QC=$(find "$SNAPSHOT_DIR" -type f -name 'qc_metrics.csv' 2>/dev/null | head -n 1 || true)
fi
[ -n "$QC" ] && [ -f "$QC" ] || die "qc_metrics.csv not found under $SNAPSHOT_DIR (expected calibrated/lights/qc_metrics.csv)"
log "qc=$QC"

mapfile -t FRAMES < <(
  find "$SNAPSHOT_DIR/detrended_aligned" -type f \( -iname '*Light_*.fits' -o -iname '*Light_*.fit' \) \
    ! -iname 'MASTERSTAR.FITS' 2>/dev/null | sort
)
if [ "${#FRAMES[@]}" -eq 0 ]; then
  mapfile -t FRAMES < <(
    find "$SNAPSHOT_DIR" -type f \( -iname '*Light_*.fits' -o -iname '*Light_*.fit' \) \
      ! -iname 'MASTERSTAR.FITS' 2>/dev/null | sort
  )
fi
[ "${#FRAMES[@]}" -gt 0 ] || die "no *Light_*.fits under $SNAPSHOT_DIR (expected detrended_aligned/lights/<setup>/BO_CVn_Light_*.fits)"
log "n_frames=${#FRAMES[@]}"

P1="$KIT_DIR/pass1.param"
P2="$KIT_DIR/pass2.param"
[ -f "$P1" ] && [ -f "$P2" ] || die "missing pass1.param / pass2.param in $KIT_DIR"
META_PY="$KIT_DIR/frame_meta.py"
[ -f "$META_PY" ] || die "missing frame_meta.py in $KIT_DIR"

NFAIL=0

run_one() {
  local frame="$1"
  local outroot="$2"
  local deg="$3"
  local stem
  stem=$(basename "$frame")
  stem=${stem%.*}
  local work="$outroot/work/$stem"
  mkdir -p "$work"
  local meta_json="$work/meta.json"
  if ! python3 "$META_PY" --fits "$frame" --qc "$QC" > "$meta_json"; then
    log "FAIL $stem deg=$deg frame_meta"
    return 1
  fi
  local seeing gain satur
  seeing=$(python3 -c "import json; print(json.load(open('$meta_json'))['seeing_fwhm_arcsec'])")
  gain=$(python3 -c "import json; print(json.load(open('$meta_json'))['gain'])")
  satur=$(python3 -c "import json; print(json.load(open('$meta_json'))['satur_level'])")
  local gsrc
  gsrc=$(python3 -c "import json; print(json.load(open('$meta_json'))['gain_source'])")
  log "META $stem deg=$deg seeing=$seeing gain=$gain ($gsrc) satur=$satur"

  local cat1="$work/${stem}_pass1.cat"
  local psf="$work/${stem}.psf"
  local cat2="$work/${stem}_pass2.cat"

  if ! "$SEX" "$frame" \
      -CATALOG_NAME "$cat1" \
      -CATALOG_TYPE FITS_LDAC \
      -PARAMETERS_NAME "$P1" \
      -DETECT_THRESH 3.0 \
      -ANALYSIS_THRESH 3.0 \
      -PHOT_APERTURES 14 \
      -PIXEL_SCALE 0 \
      -FILTER N \
      -GAIN "$gain" \
      -SATUR_LEVEL "$satur" \
      -SEEING_FWHM "$seeing" \
      > "$work/pass1.stdout" 2> "$work/pass1.stderr"; then
    log "FAIL $stem deg=$deg pass1"
    return 1
  fi

  if ! (cd "$work" && psfex "$(basename "$cat1")" \
      -PSF_SIZE 31,31 \
      -PSFVAR_KEYS X_IMAGE,Y_IMAGE \
      -PSFVAR_GROUPS 1,1 \
      -PSFVAR_DEGREES "$deg" \
      -SAMPLE_FWHMRANGE 2.0,15.0 \
      -SAMPLE_MINSN 20 \
      -CHECKPLOT_TYPE NONE \
      -CHECKIMAGE_TYPE NONE \
      -WRITE_XML N \
      > psfex.stdout 2> psfex.stderr); then
    log "FAIL $stem deg=$deg psfex"
    return 1
  fi
  # PSFEx names the model from the catalog stem (typically *_pass1.psf).
  if [ ! -f "$psf" ]; then
    local found
    found=$(ls "$work"/*.psf 2>/dev/null | head -n 1 || true)
    if [ -n "$found" ]; then
      cp -f "$found" "$psf"
    fi
  fi
  if [ ! -f "$psf" ]; then
    log "FAIL $stem deg=$deg missing_psf"
    return 1
  fi

  if ! "$SEX" "$frame" \
      -CATALOG_NAME "$cat2" \
      -CATALOG_TYPE ASCII_HEAD \
      -PARAMETERS_NAME "$P2" \
      -PSF_NAME "$psf" \
      -DETECT_THRESH 3.0 \
      -ANALYSIS_THRESH 3.0 \
      -PHOT_APERTURES 14 \
      -PIXEL_SCALE 0 \
      -FILTER N \
      -GAIN "$gain" \
      -SATUR_LEVEL "$satur" \
      -SEEING_FWHM "$seeing" \
      > "$work/pass2.stdout" 2> "$work/pass2.stderr"; then
    log "FAIL $stem deg=$deg pass2"
    return 1
  fi
  log "OK $stem deg=$deg"
  return 0
}

sweep() {
  local outroot="$1"
  local deg="$2"
  mkdir -p "$outroot/work"
  local i=0
  local n=${#FRAMES[@]}
  for frame in "${FRAMES[@]}"; do
    i=$((i + 1))
    log "[$i/$n] $(basename "$frame") deg=$deg"
    if ! run_one "$frame" "$outroot" "$deg"; then
      NFAIL=$((NFAIL + 1))
    fi
  done
}

log "=== primary PSFVAR_DEGREES=2 ==="
sweep "$OUT_DIR" 2
log "=== sensitivity PSFVAR_DEGREES=3 -> $OUT_DIR/deg3 ==="
sweep "$OUT_DIR/deg3" 3
log "done n_frames=${#FRAMES[@]} n_fail=$NFAIL"
if [ "$NFAIL" -gt 0 ]; then
  echo "FAIL: $NFAIL frame/sweep errors (see $LOG)" >&2
  exit 2
fi
exit 0
