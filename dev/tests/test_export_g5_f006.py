"""G5-F006: PDF time axis labeled BJD(TDB) to match LC column."""

from photometry_report import _pdf_time_axis_label


def test_pdf_time_axis_label_bjd_columns() -> None:
    assert _pdf_time_axis_label("bjd") == "BJD(TDB)"
    assert _pdf_time_axis_label("bjd_tdb") == "BJD(TDB)"
    assert _pdf_time_axis_label("bjd_tdb_mid") == "BJD(TDB)"


def test_pdf_time_axis_label_other_systems() -> None:
    assert _pdf_time_axis_label("hjd") == "HJD"
    assert _pdf_time_axis_label("jd") == "JD"


def test_pdf_time_axis_label_unknown_passthrough() -> None:
    assert _pdf_time_axis_label("frame_index") == "frame_index"
