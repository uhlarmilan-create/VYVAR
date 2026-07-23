# CYTHON-RELEASE-1 build tooling

Release compilation is a **packaging step**, not a repo state change. Development
continues to run interpreted Python from `src_py/`; compiled `.pyd`/`.so` artifacts
are gitignored and shadow imports when present beside sources.

## Module list (S2)

`module_list.py` derives `MODULE_LIST` from all `src_py/*.py` except:

- `app.py` and `ui_*.py` (UI layer stays interpreted)
- entries in `EXPLICIT_EXCLUDE` (one-line reason each)

## Pinned flags (S3)

Plain compile with:

- `annotation_typing=False`
- `Options.docstrings=False`
- `language_level=3`

`build_release.py` refuses to run if these drift.

## Commands (repo root)

```bat
REM Windows (VS Developer Command Prompt)
dev\tools\cython_release\cython_build_release.bat

REM Or directly:
python build\setup_cython.py build
python build\setup_cython.py clean
python dev\tools\cython_release\latent_sweep.py
python dev\tools\cython_release\smoke_imports.py
python dev\tools\cython_release\verify_mp.py
```

Build log: `tmp/cython_release/build.log`

## pytest under compiled build

Ensure `src_py/` is on `PYTHONPATH` so `.pyd` shadows `.py`:

```bat
set PYTHONPATH=src_py
pytest dev\tests -q
```

Tests using `inspect.getsource` on compiled modules skip conditionally via
`dev/tests/cython_compat.py`.
