@echo off
setlocal EnableExtensions
set "INSTALL_DIR=%~dp0"
set "INSTALL_DIR=%INSTALL_DIR:~0,-1%"
set "VYVAR_RELEASE_BUNDLE=1"
set "PYTHONHOME=%INSTALL_DIR%\python"
set "PYTHONPATH=%INSTALL_DIR%\src_py"
set "PATH=%INSTALL_DIR%\python;%INSTALL_DIR%\python\Scripts;%PATH%"
if "%~1"=="--selftest" (
  "%INSTALL_DIR%\python\python.exe" "%INSTALL_DIR%\vyvar_selftest.py"
  exit /b %ERRORLEVEL%
)
"%INSTALL_DIR%\python\python.exe" -m streamlit run "%INSTALL_DIR%\app.py" --server.headless true %*
exit /b %ERRORLEVEL%
