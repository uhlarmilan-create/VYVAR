"""Unified observer-site resolution for all VYVAR entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ObserverLocationSource = Literal["ui_selection", "cli_arg", "manifest", "config"]

_CONFIG_KEY = "observer_location_id"


@dataclass(frozen=True)
class ResolvedObserverLocation:
    location_id: int
    name: str
    lat: float
    lon: float
    alt_m: float
    source: ObserverLocationSource

    def as_provenance_dict(self) -> dict[str, Any]:
        return {
            "location_id": int(self.location_id),
            "name": str(self.name),
            "lat": float(self.lat),
            "lon": float(self.lon),
            "alt_m": float(self.alt_m),
            "source": str(self.source),
        }

    def milestone_line(self) -> str:
        return (
            f"[SITE] observer location id={self.location_id} name={self.name} "
            f"lat={self.lat} lon={self.lon} alt_m={self.alt_m} source={self.source}"
        )


def _cfg_location_id(cfg: Any | None) -> int:
    if cfg is None:
        return 0
    try:
        return max(0, int(getattr(cfg, "observer_location_id", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _positive_id(value: Any) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def location_preselect_id(
    locations: list[dict[str, Any]],
    cfg_observer_location_id: int,
) -> int | None:
    """UI selectbox pre-select: config if active, else IS_DEFAULT, else None.

    Product rule SITE-UI-01: what the UI shows must match what RUN will use.
    Prefer the durable config choice over DB IS_DEFAULT when both exist.
    """
    active_ids = {
        int(item["id"])
        for item in locations
        if int(item.get("id") or 0) > 0
    }
    cfg_id = 0
    try:
        cfg_id = int(cfg_observer_location_id or 0)
    except (TypeError, ValueError):
        cfg_id = 0
    if cfg_id > 0 and cfg_id in active_ids:
        return cfg_id
    for item in locations:
        if int(item.get("is_default", 0) or 0) == 1:
            lid = int(item.get("id") or 0)
            if lid > 0:
                return lid
    return None


def resolve_observer_location_for_run(
    db_path: str | Any,
    *,
    explicit_location_id: int | None = None,
    manifest_location_id: int | None = None,
    cfg: Any | None = None,
    source_hint: ObserverLocationSource | None = None,
) -> ResolvedObserverLocation:
    """Resolve observer site for this run.

    Precedence (no silent fallbacks):
    1. ``explicit_location_id`` (UI selection or CLI ``--location-id``)
    2. ``manifest_location_id`` (draft ``rig.location_id``)
    3. ``observer_location_id`` from config
    4. fail loud naming ``observer_location_id``

    ``source_hint`` is used only when ``explicit_location_id`` wins (ui_selection
    vs cli_arg). Manifest and config branches set source themselves.
    """
    from database import get_observer_location_by_id

    db_path_str = str(getattr(db_path, "database_path", db_path))

    explicit = _positive_id(explicit_location_id)
    manifest = _positive_id(manifest_location_id)
    cfg_id = _cfg_location_id(cfg)

    if explicit is not None:
        loc_id = explicit
        source: ObserverLocationSource = source_hint or "cli_arg"
        if source not in ("ui_selection", "cli_arg"):
            source = "cli_arg"
    elif manifest is not None:
        loc_id = manifest
        source = "manifest"
    elif cfg_id > 0:
        loc_id = cfg_id
        source = "config"
    else:
        raise ValueError(
            f"observer_location_id is unset (config key {_CONFIG_KEY}); "
            "select an observatory site in the UI or set it in config.json."
        )

    row = get_observer_location_by_id(db_path_str, loc_id)
    if not row:
        raise ValueError(
            f"observer_location_id={loc_id} not found in LOCATION table "
            f"(config key {_CONFIG_KEY})."
        )

    return ResolvedObserverLocation(
        location_id=int(row["id"]),
        name=str(row.get("name") or ""),
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        alt_m=float(row.get("alt_m") or 0.0),
        source=source,
    )


def apply_resolved_observer_location_to_config(cfg: Any, resolved: ResolvedObserverLocation) -> None:
    """Hydrate config observer fields from a resolved site (metadata consistency)."""
    cfg.observer_location_id = int(resolved.location_id)
    cfg.observer_lat = float(resolved.lat)
    cfg.observer_lon = float(resolved.lon)
    cfg.observer_alt_m = float(resolved.alt_m)
    cfg.observer_location_name = str(resolved.name)


def format_site_for_run_caption(resolved: ResolvedObserverLocation) -> str:
    """One-line caption for the UI RUN panel."""
    return (
        f"Site for this run: {resolved.name} "
        f"(lat={resolved.lat}, lon={resolved.lon})"
    )
