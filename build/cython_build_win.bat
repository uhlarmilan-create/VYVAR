@echo off
REM Run from "Visual Studio 2026 Developer Command Prompt" (Build Tools 18).
REM Requires cl.exe on PATH (Desktop development with C++ workload installed).
setlocal
call "%ProgramFiles(x86)%\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -no_logo -arch=amd64
where cl >nul 2>&1
if errorlevel 1 (
  echo.
  echo ERROR: cl.exe not found. The Build Tools shell is present but the C++ compiler
  echo workload is not installed. Open Visual Studio Installer, Modify Build Tools 2026,
  echo check "Desktop development with C++", then re-run this script.
  echo.
  exit /b 1
)
cd /d "%~dp0.."
echo === hello-world Cython ===
python tmp\cython_spike\hello_cython_test.py
if errorlevel 1 exit /b 1
echo === partial build (CYTHON_MODULES=buildable) ===
set CYTHON_MODULES=buildable
python build\setup_cython.py build_ext --inplace
if errorlevel 1 exit /b 1
echo === install .pyd beside sources ===
for %%F in (comp_selection_per_target photometry_phase2a) do (
  if exist "%%F.cp312-win_amd64.pyd" move /Y "%%F.cp312-win_amd64.pyd" "src_py\"
)
echo === import proof ===
python -c "import sys; sys.path.insert(0, r'src_py'); import comp_selection_per_target as m; import photometry_phase2a as p; assert m.__file__.endswith('.pyd'), m.__file__; assert p.__file__.endswith('.pyd'), p.__file__; print('OK comp', m.__file__); print('OK p2a', p.__file__)"
exit /b %ERRORLEVEL%
