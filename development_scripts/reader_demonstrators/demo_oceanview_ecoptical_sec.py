"""Demonstrate OceanView EC-optical data in ixdat.

The script reads a small OceanView optical spectrum-series fixture and a BioLogic EC
fixture, combines them into an ECOpticalMeasurement, and shows the basic EC-optical
plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from ixdat import Measurement, Spectrum


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DATA_DIR = REPO_ROOT / "test_data" / "oceanview_sec"

OPTICAL_FILE = DATA_DIR / "mini_oceanview__0__15-02-35-123.txt"
EC_FILE = DATA_DIR / "demo_oceanview_ecoptical_sec.mpt"


def read_ec_optical_measurement():
    """Read the EC and optical files and combine them into ECOpticalMeasurement."""
    optical = Spectrum.read(OPTICAL_FILE, reader="oceanview")
    ec = Measurement.read(EC_FILE, reader="biologic")
    ec_optical = ec + optical
    ec_optical.set_reference_spectrum(t_ref=55)
    return ec_optical


def main():
    print("reading EC and optical data")
    ec_optical = read_ec_optical_measurement()
    print(f"combined measurement technique: {ec_optical.technique}")
    print(
        "optical spectra: "
        f"{len(ec_optical.spectrum_series)} spectra x {len(ec_optical.wavelength.data)} wavelengths"
    )

    print("plotting EC-optical heat map")
    ec_optical.plot_measurement(tspan=[55, 95], wlspan=[400, 900], t_ref=55)

    print("plotting waterfall")
    ec_optical.plot_waterfall(t_ref=55)

    print("plotting wavelength tracking")
    ec_optical.plot_wavelengths(wavelengths=["w650", "w800"], tspan=[55, 95])

    plt.show()


if __name__ == "__main__":
    main()
