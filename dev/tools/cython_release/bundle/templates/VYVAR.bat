@echo off
setlocal EnableExtensions
set "INSTALL_DIR=%~dp0"
set "INSTALL_DIR=%INSTALL_DIR:~0,-1%"
set "VYVAR_RELEASE_BUNDLE=1"
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONUSERBASE="
set "PATH=%INSTALL_DIR%\python;%INSTALL_DIR%\python\Scripts;%PATH%"
if "%~1"=="--selftest" (
  "%INSTALL_DIR%\python\python.exe" -I "%INSTALL_DIR%\vyvar_selftest.py"
  exit /b %ERRORLEVEL%
)
if "%~1"=="--tool" (
  set "TOOL=%~2"
  shift
  shift
  if "%~1"=="--" shift
  if /I "%TOOL%"=="build_gaia" set "CSCRIPT=build_gaia_catalog.py"
  if /I "%TOOL%"=="build_blind_index" set "CSCRIPT=build_blind_index.py"
  if /I "%TOOL%"=="build_vsx" set "CSCRIPT=vsx_make.py"
  if /I "%TOOL%"=="build_exoplanets" set "CSCRIPT=exoplanet_make.py"
  if not defined CSCRIPT (
    echo Unknown tool: %TOOL%
    exit /b 1
  )
  "%INSTALL_DIR%\python\python.exe" -I "%INSTALL_DIR%\scripts\catalogs\%CSCRIPT%" %*
  exit /b %ERRORLEVEL%
)
"%INSTALL_DIR%\python\python.exe" -I -m streamlit run "%INSTALL_DIR%\app.py" --server.headless true %*
exit /b %ERRORLEVEL%
