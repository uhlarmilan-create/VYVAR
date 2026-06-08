$root = "C:\ASTRO\python\VYVAR"
$out = "$root\scripts\vyvar_audit.txt"

"=" * 60 | Out-File $out -Encoding UTF8
"VYVAR AUDIT - $(Get-Date)" | Out-File $out -Append -Encoding UTF8
"=" * 60 | Out-File $out -Append -Encoding UTF8

# 1. Zoznam suborov
"`n### SUBORY ###" | Out-File $out -Append -Encoding UTF8
Get-ChildItem "$root\*.py" | Select-Object Name, Length, LastWriteTime | Format-Table | Out-File $out -Append -Encoding UTF8

# 2. Funkcie a triedy
"`n### FUNKCIE A TRIEDY ###" | Out-File $out -Append -Encoding UTF8
Get-ChildItem "$root\*.py" | ForEach-Object {
    "`n--- $($_.Name) ---" | Out-File $out -Append -Encoding UTF8
    Select-String -Path $_.FullName -Pattern "^def |^class |^    def " |
        ForEach-Object { "  L$($_.LineNumber): $($_.Line.Trim())" } |
        Out-File $out -Append -Encoding UTF8
}

# 3. Interné importy
"`n### IMPORTY ###" | Out-File $out -Append -Encoding UTF8
Get-ChildItem "$root\*.py" | ForEach-Object {
    $imports = Select-String -Path $_.FullName -Pattern "^from vyvar|^import vyvar|^from pipeline|^from database|^from config|^from utils|^from photometry|^from calibration|^from importer|^from astrometry|^from infolog|^from time_utils|^from psf_photometry|^from masterstar"
    if ($imports) {
        "`n--- $($_.Name) ---" | Out-File $out -Append -Encoding UTF8
        $imports | ForEach-Object { "  $($_.Line.Trim())" } | Out-File $out -Append -Encoding UTF8
    }
}

# 4. Duplicitne funkcie
"`n### DUPLICITNE FUNKCIE ###" | Out-File $out -Append -Encoding UTF8
$allDefs = Get-ChildItem "$root\*.py" | ForEach-Object {
    $file = $_.Name
    Select-String -Path $_.FullName -Pattern "^def " | ForEach-Object {
        $fname = ($_.Line -replace "def ", "" -replace "\(.*", "").Trim()
        [PSCustomObject]@{ File = $file; Line = $_.LineNumber; FuncName = $fname }
    }
}
$allDefs | Group-Object FuncName | Where-Object { $_.Count -gt 1 } | ForEach-Object {
    "`nDUPLIKAT: $($_.Name)" | Out-File $out -Append -Encoding UTF8
    $_.Group | ForEach-Object { "  $($_.File) : L$($_.Line)" } | Out-File $out -Append -Encoding UTF8
}

# 5. config.json
"`n### CONFIG.JSON ###" | Out-File $out -Append -Encoding UTF8
if (Test-Path "$root\config.json") {
    Get-Content "$root\config.json" | Out-File $out -Append -Encoding UTF8
}

# 6. TODOs
"`n### TODO / FIXME / HACK ###" | Out-File $out -Append -Encoding UTF8
Select-String -Path "$root\*.py" -Pattern "TODO|FIXME|HACK|XXX|PENDING" |
    ForEach-Object { "  $($_.Filename):$($_.LineNumber): $($_.Line.Trim())" } |
    Out-File $out -Append -Encoding UTF8

# 7. Dlhe funkcie
"`n### DLHE FUNKCIE (viac ako 100 riadkov) ###" | Out-File $out -Append -Encoding UTF8
Get-ChildItem "$root\*.py" | ForEach-Object {
    $lines = Get-Content $_.FullName
    $file = $_.Name
    $start = 0
    $fname = ""
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^def ") {
            if ($fname -ne "" -and ($i - $start) -gt 100) {
                $len = $i - $start
                "  ${file} :: ${fname} :: ${len} riadkov :: L${start}" | Out-File $out -Append -Encoding UTF8
            }
            $fname = ($lines[$i] -replace "def ", "" -replace "\(.*", "").Trim()
            $start = $i + 1
        }
    }
}

# 8. Prvy 100 riadkov vyvar_platesolver.py
"`n### VYVAR_PLATESOLVER.PY (prvy 100 riadkov) ###" | Out-File $out -Append -Encoding UTF8
Get-Content "$root\vyvar_platesolver.py" | Select-Object -First 100 | Out-File $out -Append -Encoding UTF8

# 9. Prvy 50 riadkov pipeline.py
"`n### PIPELINE.PY (prvy 50 riadkov) ###" | Out-File $out -Append -Encoding UTF8
Get-Content "$root\pipeline.py" | Select-Object -First 50 | Out-File $out -Append -Encoding UTF8

"`n=== AUDIT DOKONCENY ===" | Out-File $out -Append -Encoding UTF8
Write-Host "Audit hotovy: $out" -ForegroundColor Green