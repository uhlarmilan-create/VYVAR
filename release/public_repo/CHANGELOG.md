# Changelog (VYVAR-release public repo)

## preview-VYVAR.0.9.0 (pre-release)

**Important:** data processed with `preview-20260723` should be reprocessed. That build omitted
order-2 sky-surface subtraction on mono preprocess; detection catalogues were inflated (~40%
DAO_ONLY vs ~4% healthy) though photometry of well-detected targets was largely unaffected.

- Phase 0 target identity: Gaia source-id join (replaces proximity adoption of wrong stars)
- Exoplanet promotion restored; observer site resolution unified (fail-loud, no default)
- Preprocess ~60x faster (numerically neutral on BO CVn acceptance)
- Runtime guards: INV-PREP-01, INV-MS-01, INV-PHASE0-ID
- Catalogue provenance (Gaia + VSX) in run metadata; anchor gate compares fingerprints
- Durable operator infolog (full session log, not ring-buffer tail only)
- Light-curve schema consistent across UI and headless entry points
- 90 compiled science modules + interpreted Streamlit UI

**Retiring preview-20260723:** GitHub release/assets may be removed; keep git tag
`preview-20260723` as regression provenance (missing sky-surface on preprocess path).

## preview-20260723 (pre-release) [withdrawn - see preview-VYVAR.0.9.0]

- Preview refreshed: canonical first-run `config.json` (full registry defaults); Parameters dashboard None-safe
- Preview refreshed: Settings Paths tab shows install dir vs data dir
- Preview refreshed: launcher isolation fix (`python -I`; pin verification in selftest)
- Preview refreshed: runtime data files shipped (params_registry.json, CITATIONS.bib, logo)
- Preview refreshed: catalog builders in `scripts/catalogs/` + INSTALL catalog chapter
- First preview bundle: Windows win64 + Linux x64
- Bundled Python 3.12 runtime (embedded / python-build-standalone)
- 85 compiled science modules + interpreted Streamlit UI
- Separate data directory (`%LOCALAPPDATA%\VYVAR` / `~/.local/share/vyvar`)
- User-built catalogs (Gaia DR3, VSX, exoplanets) - not shipped
- `--selftest` launcher health check
