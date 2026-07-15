"""Guard test for validation/VYVAR_VALIDATION_LEDGER.json."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

LEDGER_PATH = Path(__file__).resolve().parents[1] / "validation" / "VYVAR_VALIDATION_LEDGER.json"

REQUIRED_IDS = frozenset(
    {
        "VL-PYTEST-FULL",
        "VL-ANCHOR-424",
        "VL-COUNTERS-ZERO",
        "VL-K2-MATRIX",
        "VL-CALDIAG-424",
        "VL-AAVSO-EXPORT",
        "VL-XVAL-V0612",
        "VL-DETERMINISM-425",
        "VL-SHA-CUT1",
        "VL-PROVENANCE",
        "VL-TRUST-BASELINE",
    }
)

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@pytest.fixture(name="ledger")
def fixture_ledger() -> dict:
    assert LEDGER_PATH.is_file(), f"missing ledger: {LEDGER_PATH}"
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def test_ledger_parses_and_has_required_ids(ledger: dict) -> None:
    assert ledger.get("version") == 1
    items = ledger.get("items")
    assert isinstance(items, list)
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate ledger IDs"
    assert REQUIRED_IDS <= set(ids), f"missing required IDs: {sorted(REQUIRED_IDS - set(ids))}"


def test_ledger_item_schema(ledger: dict) -> None:
    required_fields = {
        "id": str,
        "area": str,
        "description": str,
        "verification": str,
        "passes": bool,
        "last_verified": (str, type(None)),
        "commit": (str, type(None)),
        "notes": str,
    }
    for it in ledger["items"]:
        for field, typ in required_fields.items():
            assert field in it, f"{it.get('id')}: missing field {field}"
            val = it[field]
            if isinstance(typ, tuple):
                assert isinstance(val, typ), f"{it['id']}.{field}: bad type {type(val)}"
            else:
                assert isinstance(val, typ), f"{it['id']}.{field}: bad type {type(val)}"
        assert isinstance(it["passes"], bool), f"{it['id']}.passes must be strict bool"
        lv = it["last_verified"]
        if lv is not None:
            assert ISO_DATE_RE.match(lv), f"{it['id']}.last_verified not ISO date: {lv!r}"
        if "status" in it:
            assert isinstance(it["status"], str), f"{it['id']}.status must be str"
        if "offline_backup" in it:
            ob = it["offline_backup"]
            assert isinstance(ob, dict), f"{it['id']}.offline_backup must be dict"
            for key in ("path", "sha256", "verified_utc"):
                assert key in ob, f"{it['id']}.offline_backup missing {key}"
