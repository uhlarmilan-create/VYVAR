from utils import (
    infer_binning_xy_from_sensor_shape,
    parse_fits_binning_int,
    parse_sensor_naxis_from_text,
)


def test_parse_fits_binning_int_strings():
    assert parse_fits_binning_int("2x2") == 2
    assert parse_fits_binning_int("2X2") == 2
    assert parse_fits_binning_int(" 2 x 2 ") == 2
    assert parse_fits_binning_int(2) == 2
    assert parse_fits_binning_int("bogus", 1) == 1


def test_infer_binning_from_sensor_shape():
    native = (6280, 4176)
    x, y, inf = infer_binning_xy_from_sensor_shape(3140, 2088, native, (1, 1))
    assert inf is True
    assert (x, y) == (2, 2)
    x2, y2, inf2 = infer_binning_xy_from_sensor_shape(6252, 4176, native, (1, 1))
    assert inf2 is False
    assert (x2, y2) == (1, 1)


def test_parse_sensor_naxis():
    assert parse_sensor_naxis_from_text("6280 x 4176") == (6280, 4176)
