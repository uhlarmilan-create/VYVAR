@echo off
call "%ProgramFiles(x86)%\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -no_logo -arch=amd64
cd /d "%~dp0.."
set CYTHON_MODULES=
python build\setup_cython.py build_ext --inplace
for %%F in (photometry_core comp_selection_per_target photometry_phase2a) do (
  if exist "%%F.cp312-win_amd64.pyd" move /Y "%%F.cp312-win_amd64.pyd" "src_py\"
)
exit /b %ERRORLEVEL%
