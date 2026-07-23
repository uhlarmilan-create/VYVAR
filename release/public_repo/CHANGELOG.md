# Changelog (VYVAR-release public repo)

## preview-20260723 (pre-release)

- Preview refreshed: launcher isolation fix (`python -I`; pin verification in selftest)
- First preview bundle: Windows win64 + Linux x64
- Bundled Python 3.12 runtime (embedded / python-build-standalone)
- 85 compiled science modules + interpreted Streamlit UI
- Separate data directory (`%LOCALAPPDATA%\VYVAR` / `~/.local/share/vyvar`)
- User-built catalogs (Gaia DR3, VSX, exoplanets) - not shipped
- `--selftest` launcher health check
