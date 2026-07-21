# VYVAR Cython build (release bundle)

Platform-neutral build scaffolding for compiling selected `src_py/` modules to
extension binaries (.pyd on Windows, .so on Linux). Pure-Python Cython mode:
no source edits required.

## Prerequisites

- Python 3.12+ with project deps installed (`pip install -r requirements.txt`)
- Cython 3.x: `pip install cython` (spike tested with **3.2.8**)
- C compiler:
  - **Windows:** [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
    with "Desktop development with C++" / MSVC v143+. Verify:
    `python tmp/cython_spike/hello_cython_test.py` -> RC 0 and a `.pyd` built.
  - **Linux:** `gcc`, `python3-dev`

## Build

From repo root:

```bash
python build/setup_cython.py build_ext --inplace
```

This writes compiled extensions next to their `.py` sources under `src_py/`.
Python imports the extension first (same module name, `.pyd`/`.so` shadows `.py`).

## Verify import

```python
import photometry_core
assert photometry_core.__file__.endswith((".pyd", ".so"))
print("OK:", photometry_core.__file__)
```

## Clean

Remove compiled binaries and Cython intermediates:

```bash
python build/setup_cython.py clean --all
# plus manual removal if needed:
#   src_py/*.pyd  src_py/*.so  src_py/*.c
#   rm -rf build/_cython_out/
```

On Windows PowerShell:

```powershell
Remove-Item src_py\*.pyd, src_py\*.c -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force build\_cython_out -ErrorAction SilentlyContinue
```

After clean, the tree is interpreted-only again; `--fast` baseline must stay PASS.

## Module list

Edit `MODULE_LIST` in `setup_cython.py`. Spike default (profile-driven):

- `photometry_core` -- **mandatory release target**; currently **STOP** at Cython
  translate (line 7419: undeclared `_get_lc_psf_strict`; needs source edit)
- `comp_selection_per_target` -- top PY-LOOP hotspot; translates OK
- `photometry_phase2a` -- phase2a split module; translates OK

## Git

Compiled artifacts are gitignored (see root `.gitignore`). Only this directory's
scripts are tracked.
