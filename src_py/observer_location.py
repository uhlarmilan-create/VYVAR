"""Unified observer-site resolution for all VYVAR entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ObserverLocationSource = Literal["ui_selection", "cli_arg", "config"]

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


def resolve_observer_location_for_run(
    db_path: str | Any,
    *,
    explicit_location_id: int | None = None,
    cfg: Any | None = None,
    source_hint: ObserverLocationSource | None = None,
) -> ResolvedObserverLocation:
    """Resolve observer site for this run.

    Precedence (no silent fallbacks):
    1. ``explicit_location_id`` (UI selection or CLI argument for this run)
    2. ``observer_location_id`` from config
    3. fail loud naming ``observer_location_id``
    """
    from database import get_observer_location_by_id

    db_path_str = str(getattr(db_path, "database_path", db_path))

    explicit: int | None = None
    if explicit_location_id is not None:
        try:
            cand = int(explicit_location_id)
        except (TypeError, ValueError):
            cand = 0
        if cand > 0:
            explicit = cand

    cfg_id = _cfg_location_id(cfg)

    if explicit is not None:
        loc_id = explicit
        source: ObserverLocationSource = source_hint or "cli_arg"
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
