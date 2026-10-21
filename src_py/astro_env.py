"""Astropy environment for VYVAR: no IERS downloads (IERS-OFF-2026-10).

Airmass, BJD_TDB and Moon geometry need UT1-UTC and polar motion only at the
sub-second / sub-arcsecond level, which has no measurable effect on photometry.
Astropy would otherwise download the IERS-A table in every process. Here the
bundled IERS-B table is used; dates beyond it are extrapolated silently.
"""

from __future__ import annotations


def configure_astropy_offline() -> None:
    try:
        from astropy.utils import iers  # noqa: PLC0415

        iers.conf.auto_download = False
        # Without downloads the bundled IERS-A predictions get older than auto_max_age
        # (30 d) and IERS_Auto raises - BJD_TDB failed for every target in draft 524.
        # UT1 error of ~1 s moves the observer by < 0.5 km -> < 2 us in light time.
        iers.conf.auto_max_age = None
        if hasattr(iers.conf, "iers_degraded_accuracy"):
            iers.conf.iers_degraded_accuracy = "ignore"
    except Exception:  # noqa: BLE001 - astropy missing or old: nothing to configure
        pass


configure_astropy_offline()
