"""Backward compatibility — pôvodný modul; implementácia je v ``photometry_core``."""

from photometry_core import *  # noqa: F401, F403
from photometry_core import _normalize_gaia_id  # noqa: F401 — import * neexportuje leading _

