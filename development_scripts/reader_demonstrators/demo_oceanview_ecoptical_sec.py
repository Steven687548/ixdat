from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ixdat import Measurement, Spectrum
from ixdat.techniques import StaircaseSEC


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


def make_staircase_dataframe(
    ec_optical,
    potential_name="E step/V",
    settle_fraction=0.2,
):
    """Average optical spectra over each electrochemical step.

    The BioLogic DPV file stores each potential step twice. The interval from row i
    to row i+2 is therefore treated as one step, and the first part of the step
    can be skipped with settle_fraction to avoid transient spectra.
    """
    t_ec, potential = ec_optical.grab(potential_name)
    t_spectra = ec_optical.spectrum_series.t
    spectra = ec_optical.spectra.data

    averaged_spectra = []
    step_potentials = []
    for start_i in range(0, len(t_ec) - 2, 2):
        t_start = t_ec[start_i]
        t_stop = t_ec[start_i + 2]
        if t_stop <= t_start:
            continue

        t_min = t_start + settle_fraction * (t_stop - t_start)
        mask = (t_spectra >= t_min) & (t_spectra <= t_stop)
        if not np.any(mask):
            continue

        averaged_spectra.append(np.mean(spectra[mask, :], axis=0))
        step_potentials.append(float(potential[start_i]))

    if not averaged_spectra:
        raise ValueError("No optical spectra overlapped with the electrochemical steps.")

    return pd.DataFrame(
        data=np.asarray(averaged_spectra).T,
        index=ec_optical.wavelength.data,
        columns=np.round(step_potentials, 4),
    ).dropna(axis=1)


def main():
    print("reading EC and optical data")
    ec_optical = read_ec_optical_measurement()
    print(f"combined measurement technique: {ec_optical.technique}")
    print(
        "optical spectra: "
        f"{len(ec_optical.spectrum_series)} spectra x {len(ec_optical.wavelength.data)} wavelengths"
    )

    print("plotting wavelength tracking")
    ec_optical.plot_wavelengths(wavelengths=["w800"], tspan=[55, 650])

    print("averaging spectra over EC steps")
    staircase_df = make_staircase_dataframe(ec_optical)
    staircase = StaircaseSEC(
        SEC_intensity_dataFrame=staircase_df,
        global_WL_interval=[400, 800],
        global_V_interval=[0.38, 1.30],
    )

    print("plotting processed staircase SEC data")
    staircase.plot_waterfall()

    staircase.plot_differential(differential_interval=0.01)

    staircase.plot_spectrum(V_range=[0.38, 0.42])

    staircase.plot_stack_differential(
        differential_interval=0.01,
        make_colorbar=False,
        spacing=0.08,
    )
    plt.show()


if __name__ == "__main__":
    main()
