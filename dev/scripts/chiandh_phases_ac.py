#!/usr/bin/env python3
"""h & χ Persei equipment IDs — C3-26000 (eq #2) + DDT 300/1200 + Dablice."""
from __future__ import annotations


def phase_a_register() -> dict[str, int]:
    """Return existing DB ids for equipment set #2 (no DB writes)."""
    return {
        "camera_id": 2,
        "telescope_id": 2,
        "location_id": 1,
        "camera_status": "reused",
        "telescope_status": "reused",
        "location_status": "reused",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(phase_a_register(), indent=2))
