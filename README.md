# VYVAR

Automated differential photometry for variable-star observers: calibrate, plate-solve,
measure, trust-score, and export submission-ready light curves from a Streamlit app.

**Validation status.** Aperture photometry is validated end-to-end and cross-checked
against AstroImageJ (4.86 mmag RMS, 134 epochs, BO CVn). ePSF photometry is available
as **beta** (default OFF): internally consistent with aperture on the reference night
(era06); independent external validation and multi-rig validation are pending.

## Documentation

| Document | Description |
|----------|-------------|
| [Full overview (EN)](docs/README_FULL.md) | Pipeline capabilities, validation, config model |
| [Prehled (CZ)](docs/README_CZ.md) | Ceska verze |
| [Installation](docs/INSTALL.md) | Step-by-step install (mirrors `install_vyvar.ps1` / `install_vyvar.sh`) |
| [Alpha testing](docs/ALPHA_TESTING.md) | What to test, how to report issues (alpha testers) |
| [Development state](docs/VYVAR_STATE.md) | Current snapshot for contributors |

License: proprietary (see [LICENSE](LICENSE)). Alpha version: `0.10.0a1`.
