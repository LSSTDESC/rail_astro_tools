import numpy as np
import os


def ccm_alam_over_ebv(wl):
    """implement the Cardelli, Clayton, & Mathis 1989 dust model
    the model gives ax and bx in terms of inverse microns,  and
    gives A_lam/A_V = ax + bx/Rv
    This translates to
    Alam/E(B-V) = Rv * ax + bx
    CCM model assumes Rv = 3.1, so hardcode this!
    Input is wavelength in Angstroms
    Returns A_lambda/E(B-V) from the CCM model.
    """
    # assume input is in angstroms, for microns, need to divide by 10^-4
    # ax and bx are computed in inverse microns
    wl_micron = wl * 1.e-4
    x = 1. / wl_micron
    Rv = 3.1

    if x >= 0.3 and x <= 1.1:
        ax = 0.574 * np.power(x, 1.61)
        bx = -0.527 * np.power(x, 1.61)
    elif x > 1.1 and x <= 3.3:
        y = x - 1.82
        ax = 1 + 0.17699 * y - 0.50447 * np.power(y, 2.0) - 0.02427 * np.power(y, 3.0) + 0.72085 * np.power(y, 4.0) + 0.01979 * np.power(y, 5.0) - 0.77530 * np.power(y, 6.0) + 0.32999 * np.power(y, 7.0)
        bx = 1.41338 * y + 2.28305 * np.power(y, 2.0) + 1.07233 * np.power(y, 3.0) - 5.38434 * np.power(y, 4.0) - 0.62251 * np.power(y, 5.0) + 5.30260 * np.power(y, 6.0) - 2.09002 * np.power(y, 7.0)
    elif x > 3.3 and x <= 5.9:
        ax = 1.752 - 0.316 * x - 0.104 / ((x - 4.67) ** 2 + 0.341)
        bx = -3.090 + 1.825 * x + 1.206 / ((x - 4.62) ** 2 + 0.263)
    elif x > 5.9 and x <= 8.0:
        fax = -0.04473 * np.power(x - 5.9, 2.0) - 0.009779 * np.power(x - 5.9, 3.0)
        fbx = 0.2130 * np.power(x - 5.9, 2.0) + 0.1207 * np.power(x - 5.9, 3.0)
        ax = 1.752 - 0.316 * x - 0.104 / (np.power(x - 4.67, 2.0) + 0.341) + fax
        bx = -3.090 + 1.825 * x + 1.206 / (np.power(x - 4.62, 2.0) + 0.263) + fbx
    elif x > 8.0 and x < 11.0:
        y = x - 8.0
        ax = -1.073 - 0.628 * y + 0.137 * np.power(y, 2.0) - 0.070 * np.power(y, 3.0)
        bx = 13.670 + 4.257 * y - 0.420 * np.power(y, 2.0) + 0.374 * np.power(y, 3.0)
    else:
        raise ValueError(f"wavelength {wl} is outside of model range!")
    alam_ebv = Rv * ax + bx
    return alam_ebv


def calc_lambda_effective_from_file(filepath):
    """Given a filter file (usually in rail_base examples_data), open the file
    and compute effective wavelength
    """
    if os.path.exists(filepath):
        data = np.loadtxt(filepath)
        wl = data[:, 0]
        tr = data[:, 1]
    else:  # pragma: no cover
        raise FileNotFoundError(f"cannot open {filepath} with numpy.loadtxt, make sure that the file is an ascii filter file")
    leff_t = np.sum(wl * tr) / np.sum(tr)
    return leff_t


def calc_lambda_effective_from_arrays(wl, tr):
    """Given arrays of wavelengths and transmission, compute the effective wavelength
    """
    return np.sum(wl * tr) / np.sum(tr)
