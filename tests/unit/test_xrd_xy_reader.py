from pathlib import Path

import numpy as np
import pytest

from ixdat.readers.xrd_xy import XRDXYReader, _parse_header_and_data

DATA_DIR = Path(__file__).parent.parent.parent / "test_data" / "xrd"


@pytest.mark.parametrize(
    "filename, expected",
    [
        (
            "hash_comment_twotheta.xy",
            dict(
                x_name="2theta",
                x_unit="degree",
                y_name="intensity",
                y_unit="counts",
                cols=2,
            ),
        ),
        (
            "excl_comment_twotheta.xye",
            dict(
                x_name="2theta",
                x_unit="degree",
                y_name="intensity",
                y_unit="counts",
                cols=3,
            ),
        ),
        (
            "semicolon_comment_twotheta.xy",
            dict(
                x_name="two theta",
                x_unit="degree",
                y_name="intensity",
                y_unit="counts",
                cols=2,
            ),
        ),
        (
            "apos_comment_twotheta.xy",
            dict(
                x_name="two theta",
                x_unit="degree",
                y_name="intensity",
                y_unit="counts",
                cols=2,
            ),
        ),
        (
            "bare_label_q.xy",
            dict(
                x_name="Q",
                x_unit="1/angstrom",
                y_name="I(Q)",
                y_unit="a.u.",
                cols=2,
                x_vals=[0.5, 1.0, 1.5],
            ),
        ),
        (
            "no_header.xye",
            dict(
                x_name="two theta",
                x_unit="degree",
                y_name="intensity",
                y_unit="counts",
                cols=3,
            ),
        ),
        (
            "multi_comment_twotheta.xy",
            dict(
                x_name="2theta",
                x_unit="degree",
                y_name="intensity",
                y_unit="counts",
                cols=2,
            ),
        ),
        (
            "comment_then_bare_label_q.xy",
            dict(
                x_name="Q",
                x_unit="1/angstrom",
                y_name="I(Q)",
                y_unit="a.u.",
                cols=2,
                x_vals=[0.5, 1.0, 1.5],
            ),
        ),
        # Q-space unit variants
        (
            "q_nm_bare_label.xy",
            dict(
                x_name="Q(nm^-1)",
                x_unit="1/nm",
                y_name="I(Q)",
                y_unit="a.u.",
                cols=2,
                x_vals=[0.5, 1.0, 1.5],
            ),
        ),
        (
            "q_loose_fallback.xy",
            dict(
                x_name="Q[1/A]",
                x_unit="1/angstrom",
                y_name="I(Q)",
                y_unit="a.u.",
                cols=2,
                x_vals=[0.5, 1.0, 1.5],
            ),
        ),
        # Alternative 2-theta spellings
        (
            "twotheta_hyphen.xy",
            dict(
                x_name="2-theta",
                x_unit="degree",
                y_name="intensity",
                y_unit="counts",
                cols=2,
            ),
        ),
        (
            "twotheta_2th.xy",
            dict(
                x_name="2th",
                x_unit="degree",
                y_name="intensity",
                y_unit="counts",
                cols=2,
            ),
        ),
        (
            "angle_label.xy",
            dict(
                x_name="angle",
                x_unit="degree",
                y_name="intensity",
                y_unit="counts",
                cols=2,
            ),
        ),
        # Structural header cases
        (
            "empty_lines_in_header.xy",
            dict(
                x_name="Q",
                x_unit="1/angstrom",
                y_name="I(Q)",
                y_unit="a.u.",
                cols=2,
                x_vals=[0.5, 1.0, 1.5],
            ),
        ),
        (
            "comma_data.xy",
            dict(
                x_name="Q",
                x_unit="1/angstrom",
                y_name="I(Q)",
                y_unit="a.u.",
                cols=2,
                x_vals=[0.5, 1.0, 1.5],
            ),
        ),
    ],
)
def test_parse_header_and_data(filename, expected):
    x_name, x_unit, y_name, y_unit, data = _parse_header_and_data(DATA_DIR / filename)

    assert x_name == expected["x_name"]
    assert x_unit == expected["x_unit"]
    assert y_name == expected["y_name"]
    assert y_unit == expected["y_unit"]
    assert data.shape == (3, expected["cols"])
    assert np.array_equal(data[:, 0], expected.get("x_vals", [10.0, 20.0, 30.0]))


@pytest.fixture
def reader():
    return XRDXYReader()


def test_xy_returns_no_error(reader):
    spectrum = reader.read(DATA_DIR / "hash_comment_twotheta.xy")
    assert spectrum.y_err is None


def test_xye_returns_error_array(reader):
    spectrum = reader.read(DATA_DIR / "excl_comment_twotheta.xye")
    assert spectrum.y_err is not None
    np.testing.assert_array_equal(spectrum.y_err, [5.0, 10.0, 7.0])


def test_q_space_labels(reader):
    spectrum = reader.read(DATA_DIR / "bare_label_q.xy")
    assert spectrum.x_name == "Q"
    assert spectrum.y_name == "I(Q)"


def test_no_header_defaults(reader):
    spectrum = reader.read(DATA_DIR / "no_header.xye")
    assert spectrum.x_name == "two theta"
    assert spectrum.y_name == "intensity"
    assert spectrum.y_err is not None
