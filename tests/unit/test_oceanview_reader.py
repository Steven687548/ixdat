"""Tests for OceanView optical spectrum-series integration."""

from pathlib import Path

import numpy as np

from ixdat import Spectrum
from ixdat.data_series import TimeSeries, ValueSeries
from ixdat.spectra import Spectrum as SingleSpectrum
from ixdat.techniques import ECMeasurement, TECHNIQUE_CLASSES
from ixdat.techniques.spectroelectrochemistry import (
    ECOpticalMeasurement,
    OpticalSpectrumSeries,
)


FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "test_data"
    / "oceanview_sec"
    / "mini_oceanview__0__15-02-35-123.txt"
)


def make_ec_measurement():
    """Return a minimal EC measurement for combination tests."""
    tseries = TimeSeries(
        name="time",
        unit_name="s",
        data=np.array([0.0, 1.0, 2.0, 3.0]),
        tstamp=1756386155.123,
    )
    potential = ValueSeries(
        name="raw_potential",
        unit_name="V",
        data=np.array([0.1, 0.2, 0.3, 0.4]),
        tseries=tseries,
    )
    current = ValueSeries(
        name="raw_current",
        unit_name="mA",
        data=np.array([1.0, 1.1, 1.2, 1.3]),
        tseries=tseries,
    )
    return ECMeasurement(
        name="test EC",
        technique="EC",
        series_list=[tseries, potential, current],
    )


def test_oceanview_reader_returns_optical_spectrum_series():
    optical = Spectrum.read(FIXTURE, reader="oceanview")

    assert isinstance(optical, OpticalSpectrumSeries)
    assert optical.technique == "Optical"
    assert optical.field.data.shape == (40, 11)
    np.testing.assert_allclose(
        optical.field.axes_series[1].data,
        [400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0],
    )
    np.testing.assert_allclose(optical.t, np.arange(40.0))


def test_optical_technique_mapping_keeps_spectrum_series_indexing():
    optical = Spectrum.read(FIXTURE, reader="oceanview")

    assert TECHNIQUE_CLASSES["Optical"] is OpticalSpectrumSeries
    spectrum = optical[0]
    assert isinstance(spectrum, SingleSpectrum)
    np.testing.assert_allclose(
        spectrum.x,
        [400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0],
    )
    np.testing.assert_allclose(
        spectrum.y,
        [220.0, 240.0, 270.0, 310.0, 360.0, 420.0, 360.0, 310.0, 270.0, 240.0, 220.0],
    )


def test_ec_plus_optical_spectrum_series_returns_ec_optical_measurement():
    ec = make_ec_measurement()
    optical = Spectrum.read(FIXTURE, reader="oceanview")

    ec_optical = ec + optical

    assert isinstance(ec_optical, ECOpticalMeasurement)
    assert ec_optical.technique == "EC-Optical"
    assert ec_optical.spectrum_series is optical
