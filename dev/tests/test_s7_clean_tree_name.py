# -*- coding: ascii -*-
"""S7: params default_repr must not depend on the checkout folder name."""

from __future__ import annotations

from pathlib import Path

import params_registry as pr


def test_worktree_named_anything_project_root_repr() -> None:
    """Even if this checkout is not named VYVAR, the rendered default stays stable."""
    rendered = pr.default_repr(pr.appconfig_defaults()["project_root"])
    assert rendered == "(git toplevel)"
    assert rendered != Path.cwd().name or Path.cwd().name == "(git toplevel)"
    assert "b1b_clean" not in rendered
