#Requires -Version 5.1
<#
.SYNOPSIS
    VYVAR installer for Windows (PowerShell). Interactive and idempotent.

.DESCRIPTION
    Sets up a runnable VYVAR on a fresh Windows machine, phase by phase, each
    echoing [OK] or [FAIL]:
      1 PYTHON   - require Python 3.12
      2 VENV     - create .venv, install pinned requirements
      3 CATALOGS - copy an existing catalog set (~12.5 GB), build, or skip
      4 PATHS    - write a machine-local paths block into config.json
                   (never keeps the author's C:\ASTRO paths)
      5 VALIDATE - dev/scripts/validate_config.py must pass
      6 SMOKE    - import the app (no server) + database self-init check
      7 FINISH   - print the exact first-run steps

    Safe to re-run: a healthy .venv is reused, existing catalog files are
    verified rather than recopied, and config.json is rewritten from its
    current contents through the canonical writer.

.PARAMETER NonInteractive
    Take defaults and skip all prompts (catalogs default to SKIP). For CI/smoke.

.PARAMETER CatalogSource
    Root of an existing VYVAR installation to copy the catalog set from
    (catalog option 1). Implies option 1 when set.

.PARAMETER DataRoot
    Target root for catalogs and data (default: this repo root).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install_vyvar.ps1
#>
[CmdletBinding()]
param(
    [switch]$NonInteractive,
    [string]$CatalogSource,
    [string]$DataRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- repo root is the script's own directory -------------------------------
$RepoRoot = $PSScriptRoot
if (-not $DataRoot) { $DataRoot = $RepoRoot }
$VenvDir = Join-Path $RepoRoot '.venv'
$VenvPy  = Join-Path $VenvDir 'Scripts\python.exe'
$ConfigPath = Join-Path $RepoRoot 'config.json'

$script:Failures = 0

function Write-Head($text) { Write-Host ""; Write-Host "== $text ==" -ForegroundColor Cyan }
function Write-Ok($text)   { Write-Host "[OK]   $text" -ForegroundColor Green }
function Write-Warn2($text) { Write-Host "[WARN] $text" -ForegroundColor Yellow }
function Write-Fail($text)  { Write-Host "[FAIL] $text" -ForegroundColor Red; $script:Failures++ }
function Write-Info($text)  { Write-Host "       $text" -ForegroundColor Gray }

function Read-Default([string]$prompt, [string]$default) {
    if ($NonInteractive) { return $default }
    $ans = Read-Host "$prompt [$default]"
    if ([string]::IsNullOrWhiteSpace($ans)) { return $default }
    return $ans.Trim()
}

# ---------------------------------------------------------------------------
# Catalog manifest (data). Sizes from DEPS/DOCS recon; verified on the author
# machine 2026-07-18. The "zaloha" (backup subset, G<=16) set is the anchor
# catalog and the recommended copy for a new machine (~12.1 GB total). Src
# candidates are tried in order under the chosen source root.
# ---------------------------------------------------------------------------
$Manifest = @(
    [pscustomobject]@{ Name='Gaia DR3 catalog (G<=16 subset)'; CfgArg='--gaia-db';
        Dest='GAIA_DR3\vyvar_gaia_dr3.db';
        Src=@('GAIA_DR3\zaloha\vyvar_gaia_dr3.db','GAIA_DR3\vyvar_gaia_dr3.db'); ApproxBytes=10066063360 }
    [pscustomobject]@{ Name='Blind-solve fine index'; CfgArg='--blind-fine';
        Dest='GAIA_DR3\gaia_triangles_fine.pkl';
        Src=@('GAIA_DR3\zaloha\gaia_triangles_fine.pkl','GAIA_DR3\gaia_triangles_fine.pkl'); ApproxBytes=1406291841 }
    [pscustomobject]@{ Name='Blind-solve wide index'; CfgArg='--blind-wide';
        Dest='GAIA_DR3\gaia_triangles_wide.pkl';
        Src=@('GAIA_DR3\zaloha\gaia_triangles_wide.pkl','GAIA_DR3\gaia_triangles_wide.pkl'); ApproxBytes=706509280 }
    [pscustomobject]@{ Name='AAVSO VSX variables'; CfgArg='--vsx-db';
        Dest='VSX\vyvar_vsx_local_v2.db';
        Src=@('VSX\vyvar_vsx_local_v2.db'); ApproxBytes=908324864 }
    [pscustomobject]@{ Name='NASA exoplanet archive'; CfgArg='--exoplanet-db';
        Dest='exoplanets\vyvar_exoplanet_local.db';
        Src=@('exoplanets\vyvar_exoplanet_local.db'); ApproxBytes=2334720 }
)

function Format-GB([long]$bytes) { return ('{0:N2} GB' -f ($bytes / 1GB)) }

# ===========================================================================
# PHASE 1 - PYTHON
# ===========================================================================
function Invoke-PhasePython {
    Write-Head '1/7 PYTHON (require 3.12)'
    $pyExe = $null
    # Prefer the launcher: py -3.12
    try {
        $v = & py -3.12 --version 2>$null
        if ($LASTEXITCODE -eq 0 -and $v) { $pyExe = 'py -3.12'; Write-Ok "found via launcher: $v" }
    } catch {}
    if (-not $pyExe) {
        try {
            $v = & python --version 2>&1
            if ($v -match '3\.12\.') { $pyExe = 'python'; Write-Ok "found: $v" }
        } catch {}
    }
    if (-not $pyExe) {
        Write-Fail 'Python 3.12 not found.'
        Write-Info 'Install it, then re-run this script:'
        Write-Info '  winget install -e --id Python.Python.3.12'
        Write-Info '  or download: https://www.python.org/downloads/release/python-3120/'
        Write-Info '  (During setup tick "Add python.exe to PATH".)'
        return $null
    }
    return $pyExe
}

# ===========================================================================
# PHASE 2 - VENV + DEPENDENCIES
# ===========================================================================
function Invoke-PhaseVenv([string]$pyExe) {
    Write-Head '2/7 VENV + dependencies'
    $healthy = Test-Path $VenvPy
    if ($healthy) {
        Write-Ok ".venv already present ($VenvPy) - reusing"
    } else {
        Write-Info 'Creating .venv (this can take a moment)...'
        if ($pyExe -eq 'py -3.12') { & py -3.12 -m venv $VenvDir } else { & python -m venv $VenvDir }
        if (-not (Test-Path $VenvPy)) { Write-Fail 'venv creation failed.'; return $false }
        Write-Ok '.venv created'
    }
    Write-Info 'Upgrading pip...'
    & $VenvPy -m pip install --upgrade pip 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Warn2 'pip upgrade returned non-zero (continuing).' }

    Write-Info 'Installing requirements.txt (large wheels: numpy/scipy/astropy/photutils; several minutes)...'
    & $VenvPy -m pip install -r (Join-Path $RepoRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { Write-Fail 'pip install -r requirements.txt failed.'; return $false }
    Write-Ok 'requirements installed'

    Write-Info 'Verifying dependency consistency (pip check)...'
    & $VenvPy -m pip check
    if ($LASTEXITCODE -ne 0) { Write-Warn2 'pip check reported issues (see above); usually safe to continue.' }
    else { Write-Ok 'pip check clean' }
    return $true
}

# ===========================================================================
# PHASE 3 - CATALOGS
# ===========================================================================
function Invoke-PhaseCatalogs {
    Write-Head '3/7 CATALOGS'
    $result = [ordered]@{ Mode='skip'; Paths=@{} }

    $totalApprox = ($Manifest | Measure-Object -Property ApproxBytes -Sum).Sum
    Write-Info ("Catalog set (recommended copy): ~{0} total." -f (Format-GB $totalApprox))
    foreach ($m in $Manifest) { Write-Info ("  - {0,-32} ~{1}" -f $m.Name, (Format-GB $m.ApproxBytes)) }

    $choice = '1'
    if ($CatalogSource) { $choice = '1' }
    elseif ($NonInteractive) { $choice = '3' }
    else {
        Write-Host ''
        Write-Host '  [1] Copy from an existing VYVAR installation (recommended, default)'
        Write-Host '  [2] Build from sources (astroquery/AAVSO/NASA; hours to days, large downloads)'
        Write-Host '  [3] Skip for now (LIMITED MODE - no Gaia matching until catalogs arrive)'
        $choice = Read-Default 'Choose catalog option' '1'
    }

    switch ($choice) {
        '1' { return (Invoke-CatalogCopy $result) }
        '2' { return (Invoke-CatalogBuild $result) }
        default {
            Write-Warn2 'Catalogs SKIPPED. VYVAR will run in LIMITED MODE:'
            Write-Info 'no Gaia identification/comparison selection -> not usable for science'
            Write-Info 'until catalogs are provided. Re-run this installer to add them later.'
            $result.Mode = 'skip'
            return $result
        }
    }
}

function Invoke-CatalogCopy($result) {
    $result.Mode = 'copy'
    $srcRoot = $CatalogSource
    if (-not $srcRoot) {
        $srcRoot = Read-Default 'Source VYVAR root (existing installation)' 'E:\VYVAR'
    }
    if (-not (Test-Path $srcRoot)) { Write-Fail "source root not found: $srcRoot"; return $result }

    $targetRoot = Read-Default 'Target data root (catalogs copied under here)' $DataRoot

    # Resolve source files + total copy size first.
    $plan = @()
    $copyBytes = 0
    foreach ($m in $Manifest) {
        $found = $null
        foreach ($cand in $m.Src) {
            $p = Join-Path $srcRoot $cand
            if (Test-Path $p) { $found = $p; break }
        }
        if (-not $found) { Write-Warn2 ("source missing for {0} (skipping)" -f $m.Name); continue }
        $srcLen = (Get-Item $found).Length
        $dest = Join-Path $targetRoot $m.Dest
        $plan += [pscustomobject]@{ M=$m; Src=$found; Dest=$dest; Bytes=$srcLen }
        $copyBytes += $srcLen
    }
    if ($plan.Count -eq 0) { Write-Fail 'no catalog files found under source root.'; return $result }

    # Free-space check on the target drive.
    $qualifier = (Split-Path -Qualifier (Resolve-Path -LiteralPath $targetRoot -ErrorAction SilentlyContinue | ForEach-Object Path) -ErrorAction SilentlyContinue)
    if (-not $qualifier) { $qualifier = (Split-Path -Qualifier $targetRoot) }
    try {
        $drive = New-Object System.IO.DriveInfo($qualifier)
        Write-Info ("Copy size: {0}; free on {1} {2}" -f (Format-GB $copyBytes), $qualifier, (Format-GB $drive.AvailableFreeSpace))
        if ($drive.AvailableFreeSpace -lt ($copyBytes * 1.05)) {
            Write-Fail ("not enough free space on {0} (need ~{1})." -f $qualifier, (Format-GB ($copyBytes * 1.05)))
            return $result
        }
    } catch { Write-Warn2 "could not check free space on $qualifier (continuing)." }

    foreach ($item in $plan) {
        $destDir = Split-Path -Parent $item.Dest
        if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir -Force | Out-Null }
        if ((Test-Path $item.Dest) -and ((Get-Item $item.Dest).Length -eq $item.Bytes)) {
            Write-Ok ("{0} already present + verified ({1})" -f $item.M.Name, (Format-GB $item.Bytes))
        } else {
            Write-Info ("Copying {0} ({1})..." -f $item.M.Name, (Format-GB $item.Bytes))
            Copy-Item -LiteralPath $item.Src -Destination $item.Dest -Force
            $got = (Get-Item $item.Dest).Length
            if ($got -ne $item.Bytes) {
                Write-Fail ("size mismatch for {0}: copied {1} of {2} bytes" -f $item.M.Name, $got, $item.Bytes)
                continue
            }
            Write-Ok ("{0} copied + size-verified" -f $item.M.Name)
        }
        $result.Paths[$item.M.CfgArg] = $item.Dest
    }
    return $result
}

function Invoke-CatalogBuild($result) {
    $result.Mode = 'build'
    Write-Warn2 'BUILD path: this downloads Gaia DR3 / VSX / exoplanet sources.'
    Write-Info 'Expect large downloads and hours-to-days depending on your link. This is'
    Write-Info 'NOT the recommended path for a quick setup. Builders (run inside .venv):'
    Write-Info '  python GAIA_DR3\build_gaia_catalog.py     (Gaia DR3 -> sqlite)'
    Write-Info '  python GAIA_DR3\build_blind_index.py      (triangle indexes from the DB)'
    Write-Info '  python VSX\vsx_make.py                    (AAVSO VSX -> sqlite)'
    Write-Info '  python exoplanets\exoplanet_make.py       (NASA archive -> sqlite)'
    Write-Info 'After building, re-run this installer and choose option 1 (or set paths by hand).'
    return $result
}

# ===========================================================================
# PHASE 4 - PATHS
# ===========================================================================
function Invoke-PhasePaths($catalog) {
    Write-Head '4/7 PATHS (write machine-local config.json)'
    Write-Info 'Location / telescope / camera facts are NOT set here - create them in the'
    Write-Info 'app (Settings) on first run. This phase only writes file/catalog paths.'

    $archive = Read-Default 'Archive root (raw + processed drafts)' (Join-Path $DataRoot 'Archive')
    $calib   = Read-Default 'Calibration library root' (Join-Path $DataRoot 'CalibrationLibrary')
    $db      = Read-Default 'Main database path (vyvar.sqlite3)' (Join-Path $DataRoot 'vyvar.sqlite3')

    $cfgArgs = @('dev\scripts\apply_install_config.py', '--config', $ConfigPath,
              '--archive-root', $archive, '--calibration-root', $calib, '--database-path', $db)
    foreach ($m in $Manifest) {
        if ($catalog.Paths.ContainsKey($m.CfgArg)) {
            $cfgArgs += @($m.CfgArg, $catalog.Paths[$m.CfgArg])
        }
    }
    # Any catalog path not chosen this run is sanitised (author C:\ASTRO paths blanked).
    & $VenvPy @cfgArgs
    if ($LASTEXITCODE -ne 0) { Write-Fail 'writing config.json failed.'; return $false }
    Write-Ok 'config.json paths written (author absolute paths removed)'
    return $true
}

# ===========================================================================
# PHASE 5 - VALIDATE
# ===========================================================================
function Invoke-PhaseValidate {
    Write-Head '5/7 VALIDATE config.json'
    & $VenvPy (Join-Path $RepoRoot 'dev\scripts\validate_config.py') $ConfigPath
    if ($LASTEXITCODE -ne 0) { Write-Fail 'validate_config.py reported errors.'; return $false }
    Write-Ok 'config.json valid'
    return $true
}

# ===========================================================================
# PHASE 6 - SMOKE
# ===========================================================================
function Invoke-PhaseSmoke {
    Write-Head '6/7 SMOKE (import app + database self-init)'
    $py = @'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "src_py"))
import app  # import-only; main() runs only under `streamlit run`
from config import AppConfig
from database import VyvarDatabase
cfg = AppConfig()
db = VyvarDatabase(cfg.database_path)
tables = {r[0] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
db.conn.close()
need = {"EQUIPMENTS","TELESCOPE","LOCATION","OBSERVATION","OBS_DRAFT"}
missing = need - tables
assert not missing, f"DB self-init missing tables: {sorted(missing)}"
print(f"SMOKE OK: app import + DB self-init at {cfg.database_path} ({len(tables)} tables)")
'@
    $tmp = Join-Path $env:TEMP ("vyvar_smoke_{0}.py" -f ([guid]::NewGuid().ToString('N')))
    Set-Content -LiteralPath $tmp -Value $py -Encoding UTF8
    try {
        & $VenvPy $tmp
        $rc = $LASTEXITCODE
    } finally {
        Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue
    }
    if ($rc -ne 0) { Write-Fail 'smoke run failed (see traceback above).'; return $false }
    Write-Ok 'app imports and the database self-initialises (empty of your data - see below)'
    return $true
}

# ===========================================================================
# PHASE 7 - FINISH
# ===========================================================================
function Invoke-PhaseFinish($catalog) {
    Write-Head '7/7 FINISH - next steps'
    Write-Host 'Start the app from this folder:' -ForegroundColor White
    Write-Host '    streamlit run app.py' -ForegroundColor White
    Write-Host ''
    Write-Host 'Then, in the app:' -ForegroundColor White
    Write-Host '    1) Settings -> create your Location, Telescope and Equipment,' -ForegroundColor White
    Write-Host '       then select them (the DB ships with the author example rows,' -ForegroundColor White
    Write-Host '       e.g. location "Dablice" - do not submit under those).' -ForegroundColor White
    Write-Host '    2) Import your first night and run the pipeline.' -ForegroundColor White
    Write-Host ''
    Write-Info 'Full walk-through:  VYVAR_INSTALL_GUIDE_CZ.pdf'
    Write-Info 'Every config key:   docs\VYVAR_CONFIG_GUIDE_CZ.md (CZ) / docs\VYVAR_CONFIG_GUIDE_EN.md (EN)'
    Write-Info 'Install reference:   INSTALL.md'
    if ($catalog.Mode -eq 'skip') {
        Write-Host ''
        Write-Warn2 'LIMITED MODE: catalogs were skipped - add them (re-run installer, option 1)'
        Write-Warn2 'before expecting Gaia identification or comparison-star selection to work.'
    }
}

# ===========================================================================
# DRIVER
# ===========================================================================
Write-Host '===========================================' -ForegroundColor Cyan
Write-Host ' VYVAR installer (Windows / PowerShell)' -ForegroundColor Cyan
Write-Host ('  repo: {0}' -f $RepoRoot) -ForegroundColor Cyan
Write-Host '===========================================' -ForegroundColor Cyan

$pyExe = Invoke-PhasePython
if (-not $pyExe) { Write-Host ''; Write-Fail 'Stopping: Python 3.12 is required.'; exit 1 }

if (-not (Invoke-PhaseVenv $pyExe)) { Write-Host ''; Write-Fail 'Stopping: environment setup failed.'; exit 1 }

$catalog = Invoke-PhaseCatalogs

if (-not (Invoke-PhasePaths $catalog)) { Write-Host ''; Write-Fail 'Stopping: could not write config.json.'; exit 1 }

if (-not (Invoke-PhaseValidate)) { Write-Host ''; Write-Fail 'Stopping: config.json invalid.'; exit 1 }

if (-not (Invoke-PhaseSmoke)) { Write-Host ''; Write-Fail 'Stopping: smoke run failed.'; exit 1 }

Invoke-PhaseFinish $catalog

Write-Host ''
if ($script:Failures -eq 0) {
    Write-Host 'INSTALL COMPLETE.' -ForegroundColor Green
    exit 0
} else {
    Write-Host ("INSTALL FINISHED WITH {0} WARNING/FAILURE MARKER(S) - review [FAIL] lines above." -f $script:Failures) -ForegroundColor Yellow
    exit 1
}
