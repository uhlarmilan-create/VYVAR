@echo off
REM CYTHON-RELEASE-1 full MODULE_LIST build (MSVC). Run from VS Developer Command Prompt.
setlocal
call "%ProgramFiles(x86)%\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -no_logo -arch=amd64
where cl >nul 2>&1
if errorlevel 1 (
  echo ERROR: cl.exe not found. Install "Desktop development with C++" in VS Build Tools.
  exit /b 1
)
cd /d "%~dp0..\..\.."
echo === CYTHON-RELEASE full build ===
python build\setup_cython.py build
if errorlevel 1 exit /b 1
echo === smoke imports ===
python dev\tools\cython_release\smoke_imports.py
if errorlevel 1 exit /b 1
echo === MP spawn verify ===
python dev\tools\cython_release\verify_mp.py
exit /b %ERRORLEVEL%
