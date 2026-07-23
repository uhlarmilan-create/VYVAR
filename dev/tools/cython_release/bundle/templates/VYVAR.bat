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
"%INSTALL_DIR%\python\python.exe" -I -m streamlit run "%INSTALL_DIR%\app.py" --server.headless true %*
exit /b %ERRORLEVEL%
