"""
Lyman-alpha degrader due to IGM, following Crenshaw+ 2025 (https://arxiv.org/pdf/2503.06016).
Implementation of the IGM model from Inoue 2014 (arXiv:1402.0677).
Source code from: https://github.com/jfcrenshaw/lbg_tools/blob/main/src/lbg_tools/_igm_inoue.py
"""

import numpy as np
from rail.creation.noisifier import Noisifier
from rail.core.common_params import SHARED_PARAMS
from rail.utils.path_utils import RAILDIR
from ceci.config import StageParameter as Param
import os

class IGMExtinctionModel(Noisifier):
    """
    Degrader that simulates IGM extinction.
    
    """

    name = 'IGMExtinctionModel'
    config_options = Noisifier.config_options.copy()
    config_options.update(
        data_path=Param(str, "None", msg="data_path (str): file path to the "
                                          "FILTER directories.  If left to "
                                          "default `None` it will use the install "
                                          "directory for rail + rail/examples_data/estimation_data/data"),  
        filter_list=SHARED_PARAMS,
        bands=SHARED_PARAMS,
        redshift_col=SHARED_PARAMS,
        compute_uv_slope=Param(bool, True, msg="whether to compute the uv slope"
                                                "If not, the initial value of -2 will be used"),
    )

    def __init__(self, args, **kwargs):
        """
        """
        super().__init__(args, **kwargs)

        # Table 2 from arXiv:1402.0677
        _table = r"""
           2 (Ly$\alpha$) & 1215.67 & 1.690e-02 & 2.354e-03 & 1.026e-04 & 1.617e-04 & 5.390e-05 \\
           3 (Ly$\beta$) & 1025.72 & 4.692e-03 & 6.536e-04 & 2.849e-05 & 1.545e-04 & 5.151e-05 \\
           4 (Ly$\gamma$) &  972.537 & 2.239e-03 & 3.119e-04 & 1.360e-05 & 1.498e-04 & 4.992e-05 \\
           5 &  949.743 & 1.319e-03 & 1.837e-04 & 8.010e-06 & 1.460e-04 & 4.868e-05 \\
           6 &  937.803 & 8.707e-04 & 1.213e-04 & 5.287e-06 & 1.429e-04 & 4.763e-05 \\
           7 &  930.748 & 6.178e-04 & 8.606e-05 & 3.752e-06 & 1.402e-04 & 4.672e-05 \\
           8 &  926.226 & 4.609e-04 & 6.421e-05 & 2.799e-06 & 1.377e-04 & 4.590e-05 \\
           9 &  923.150 & 3.569e-04 & 4.971e-05 & 2.167e-06 & 1.355e-04 & 4.516e-05 \\
           10 &  920.963 & 2.843e-04 & 3.960e-05 & 1.726e-06 & 1.335e-04 & 4.448e-05 \\
           11 &  919.352 & 2.318e-04 & 3.229e-05 & 1.407e-06 & 1.316e-04 & 4.385e-05 \\
           12 &  918.129 & 1.923e-04 & 2.679e-05 & 1.168e-06 & 1.298e-04 & 4.326e-05 \\
           13 &  917.181 & 1.622e-04 & 2.259e-05 & 9.847e-07 & 1.281e-04 & 4.271e-05 \\
           14 &  916.429 & 1.385e-04 & 1.929e-05 & 8.410e-07 & 1.265e-04 & 4.218e-05 \\
           15 &  915.824 & 1.196e-04 & 1.666e-05 & 7.263e-07 & 1.250e-04 & 4.168e-05 \\
           16 &  915.329 & 1.043e-04 & 1.453e-05 & 6.334e-07 & 1.236e-04 & 4.120e-05 \\
           17 &  914.919 & 9.174e-05 & 1.278e-05 & 5.571e-07 & 1.222e-04 & 4.075e-05 \\
           18 &  914.576 & 8.128e-05 & 1.132e-05 & 4.936e-07 & 1.209e-04 & 4.031e-05 \\
           19 &  914.286 & 7.251e-05 & 1.010e-05 & 4.403e-07 & 1.197e-04 & 3.989e-05 \\
           20 &  914.039 & 6.505e-05 & 9.062e-06 & 3.950e-07 & 1.185e-04 & 3.949e-05 \\
           21 &  913.826 & 5.868e-05 & 8.174e-06 & 3.563e-07 & 1.173e-04 & 3.910e-05 \\
           22 &  913.641 & 5.319e-05 & 7.409e-06 & 3.230e-07 & 1.162e-04 & 3.872e-05 \\
           23 &  913.480 & 4.843e-05 & 6.746e-06 & 2.941e-07 & 1.151e-04 & 3.836e-05 \\
           24 &  913.339 & 4.427e-05 & 6.167e-06 & 2.689e-07 & 1.140e-04 & 3.800e-05 \\
           25 &  913.215 & 4.063e-05 & 5.660e-06 & 2.467e-07 & 1.130e-04 & 3.766e-05 \\
           26 &  913.104 & 3.738e-05 & 5.207e-06 & 2.270e-07 & 1.120e-04 & 3.732e-05 \\
           27 &  913.006 & 3.454e-05 & 4.811e-06 & 2.097e-07 & 1.110e-04 & 3.700e-05 \\
           28 &  912.918 & 3.199e-05 & 4.456e-06 & 1.943e-07 & 1.101e-04 & 3.668e-05 \\
           29 &  912.839 & 2.971e-05 & 4.139e-06 & 1.804e-07 & 1.091e-04 & 3.637e-05 \\
           30 &  912.768 & 2.766e-05 & 3.853e-06 & 1.680e-07 & 1.082e-04 & 3.607e-05 \\
           31 &  912.703 & 2.582e-05 & 3.596e-06 & 1.568e-07 & 1.073e-04 & 3.578e-05 \\
           32 &  912.645 & 2.415e-05 & 3.364e-06 & 1.466e-07 & 1.065e-04 & 3.549e-05 \\
           33 &  912.592 & 2.263e-05 & 3.153e-06 & 1.375e-07 & 1.056e-04 & 3.521e-05 \\
           34 &  912.543 & 2.126e-05 & 2.961e-06 & 1.291e-07 & 1.048e-04 & 3.493e-05 \\
           35 &  912.499 & 2.000e-05 & 2.785e-06 & 1.214e-07 & 1.040e-04 & 3.466e-05 \\
           36 &  912.458 & 1.885e-05 & 2.625e-06 & 1.145e-07 & 1.032e-04 & 3.440e-05 \\
           37 &  912.420 & 1.779e-05 & 2.479e-06 & 1.080e-07 & 1.024e-04 & 3.414e-05 \\
           38 &  912.385 & 1.682e-05 & 2.343e-06 & 1.022e-07 & 1.017e-04 & 3.389e-05 \\
           39 &  912.353 & 1.593e-05 & 2.219e-06 & 9.673e-08 & 1.009e-04 & 3.364e-05 \\
           40 &  912.324 & 1.510e-05 & 2.103e-06 & 9.169e-08 & 1.002e-04 & 3.339e-05 \\
        """
        self._lambda_L = 911.8
        
        _tls_params_list = []
        for row in _table.split(" \\")[:-1]:
            _tls_params_list.append([float(p) for p in row.split(" & ")[1:]])
        self._tls_params = np.array(_tls_params_list)

        self.beta_uv_init = -2

        datapath = self.config["data_path"]
        if datapath is None or datapath == "None":
            tmpdatapath = os.path.join(RAILDIR, "rail/examples_data/estimation_data/data")
            self.data_path = tmpdatapath
        else:  # pragma: no cover
            self.data_path = datapath
        if not os.path.exists(self.data_path):  # pragma: no cover
            raise FileNotFoundError(self.data_path + " does not exist! Check value of data_path in config file!")

    
    def _tls_laf(self, wavelen: np.ndarray, z: float) -> np.ndarray:
        """Calculate optical depth contribution from Lyman-series: Lya Forest
    
        Parameters
        ----------
        wavelen : np.ndarray
            Observed wavelength in Angstroms
        z : float
            Redshift
    
        Returns
        -------
        np.ndarray
            Optical depth contribution
        """
        # Evaluate power-law terms at every wavelength
        w = wavelen[:, None] / self._tls_params[None, :, 0]
        vals = self._tls_params[:, 1:4] * np.power(w[..., None], [1.2, 3.7, 5.5])
    
        # Zero points outside of appropriate wavelength ranges
        mask = (w > 1) & (w < 1 + z)
        vals *= mask[..., None]
    
        # Apply mask so only appropriate coefficient contributes to each
        mask = w < 2.2
        vals[..., 0] *= mask
    
        mask = (w >= 2.2) & (w < 5.7)
        vals[..., 1] *= mask
    
        mask = w >= 5.7
        vals[..., 2] *= mask
    
        # Sum over last axis. After masks, there is only non-zero element in each of
        # these sums, so the sum is really just to pick up that non-zero element
        vals = vals.sum(axis=-1)
    
        # Finally sum over the contributions from every Lyman transition
        # Note this could be combined with previous sum, but I separated them to
        # make the logic a little easier to parse.
        tau = vals.sum(axis=-1)
    
        return tau
    
    
    def _tls_dla(self, wavelen: np.ndarray, z: float) -> np.ndarray:
        """Calculate optical depth contribution from Lyman-series: Damped Lya Systems
    
        Parameters
        ----------
        wavelen : np.ndarray
            Observed wavelength in Angstroms
        z : float
            Redshift
    
        Returns
        -------
        np.ndarray
            Optical depth contribution
        """
        # Evaluate power-law terms at every wavelength
        w = wavelen[:, None] / self._tls_params[None, :, 0]
        vals = self._tls_params[:, 4:] * np.power(w[..., None], [2.0, 3.0])
    
        # Zero points outside of appropriate wavelength ranges
        mask = (w > 1) & (w < 1 + z)
        vals *= mask[..., None]
    
        # Apply mask so only appropriate coefficient contributes to each
        mask = w < 3.0
        vals[..., 0] *= mask
    
        mask = w >= 3.0
        vals[..., 1] *= mask
    
        # Sum over last axis. After masks, there is only non-zero element in each of
        # these sums, so the sum is really just to pick up that non-zero element
        vals = vals.sum(axis=-1)
    
        # Finally sum over the contributions from every Lyman transition
        # Note this could be combined with previous sum, but I separated them to
        # make the logic a little easier to parse.
        tau = vals.sum(axis=-1)
    
        return tau
    
    
    def _tlc_laf(self, wavelen: np.ndarray, z: float) -> np.ndarray:
        """Calculate optical depth contribution from Lyman-continuum: Lya Forest
    
        Parameters
        ----------
        wavelen : np.ndarray
            Observed wavelength in Angstroms
        z : float
            Redshift
    
        Returns
        -------
        np.ndarray
            Optical depth contribution
        """
        with np.errstate(divide="ignore"):
            w = wavelen / self._lambda_L
    
            tau = np.zeros_like(w)
    
            if z < 1.2:
                mask = (w >= 1) & (z < 1.2) & (w < 1 + z)
                vals = 0.325 * (w**1.2 - (1 + z) ** (-0.9) * w**2.1)
                tau[mask] = vals[mask]
    
            if (z >= 1.2) & (z < 4.7):
                mask = (w >= 1) & (z >= 1.2) & (z < 4.7) & (w < 2.2)
                vals = (
                    2.55e-2 * (1 + z) ** 1.6 * w**2.1 + 0.325 * w**1.2 - 0.25 * w**2.1
                )
                tau[mask] = vals[mask]
    
                mask = (w >= 1) & (z >= 1.2) & (z < 4.7) & (w >= 2.2) & (w < 1 + z)
                vals = 2.55e-2 * ((1 + z) ** 1.6 * w**2.1 - w**3.7)
                tau[mask] = vals[mask]
    
            else:
                mask = (w >= 1) & (z >= 4.7) & (w < 2.2)
                vals = (
                    5.22e-4 * (1 + z) ** 3.4 * w**2.1
                    + 0.325 * w**1.2
                    - 3.14e-2 * w**2.1
                )
                tau[mask] = vals[mask]
    
                mask = (w >= 1) & (z >= 4.7) & (w >= 2.2) & (w < 5.7)
                vals = (
                    5.22e-4 * (1 + z) ** 3.4 * w**2.1
                    + 0.218 * w**2.1
                    - 2.55e-2 * w**3.7
                )
                tau[mask] = vals[mask]
    
                mask = (w >= 1) & (z >= 4.7) & (w >= 5.7) & (w < 1 + z)
                vals = 5.22e-4 * ((1 + z) ** 3.4 * w**2.1 - w**5.5)
                tau[mask] = vals[mask]
    
        # Fix for low-wavelength continuum from FSPS
        # i.e., continuum fitting function decreases at low-wavelengths, which isn't
        # expected. As continuum tau starts to decrease towards lower wavelength,
        # we simply set values to the max value
        idx = np.nanargmax(tau)
        tau[:idx] = tau[idx]
    
        return tau
    
    
    def _tlc_dla(self, wavelen: np.ndarray, z: float) -> np.ndarray:
        """Calculate optical depth contribution from Lyman-continuum: Damped Lya Systems
    
        Parameters
        ----------
        wavelen : np.ndarray
            Observed wavelength in Angstroms
        z : float
            Redshift
    
        Returns
        -------
        np.ndarray
            Optical depth contribution
        """
        with np.errstate(divide="ignore"):
            w = wavelen / self._lambda_L
    
            tau = np.zeros_like(w)
    
            if z < 2.0:
                mask = (w >= 1) & (z < 2) & (w < 1 + z)
                vals = (
                    0.211 * (1 + z) ** 2
                    - 7.66e-2 * (1 + z) ** 2.3 * w ** (-0.3)
                    - 0.135 * w**2
                )
                tau[mask] = vals[mask]
    
            if z >= 2.0:
                mask = (w >= 1) & (z >= 2) & (w < 3)
                vals = (
                    0.634
                    + 4.7e-2 * (1 + z) ** 3
                    - 1.78e-2 * (1 + z) ** 3.3 * w ** (-0.3)
                    - 0.135 * w**2
                    - 0.291 * w ** (-0.3)
                )
                tau[mask] = vals[mask]
    
                mask = (w >= 1) & (z >= 2) & (w >= 3) & (w < 1 + z)
                vals = (
                    4.7e-2 * (1 + z) ** 3
                    - 1.78e-2 * (1 + z) ** 3.3 * w ** (-0.3)
                    - 2.92e-2 * w**3
                )
                tau[mask] = vals[mask]
    
        # Fix for low-wavelength continuum from FSPS
        # i.e., continuum fitting function decreases at low-wavelengths, which isn't
        # expected. As continuum tau starts to decrease towards lower wavelength,
        # we simply set values to the max value
        idx = np.nanargmax(tau)
        tau[:idx] = tau[idx]
    
        return tau
    
    
    def _igm_tau(self, wavelen: float | np.ndarray, z: float) -> np.ndarray:
        """Calculate optical depth of the IGM using the Inoue model.
    
        Parameters
        ----------
        wavelen : float | np.ndarray
            Observed wavelength in Angstroms
        z : float
            Redshift
    
        Returns
        -------
        np.ndarray
           IGM optical depth
        """
        # Make sure wavelength is an array
        wavelen = np.atleast_1d(wavelen).astype(float)
    
        # Optical depth of IGM due to Lyman transitions
        return (
            _tls_laf(wavelen, z)
            + _tls_dla(wavelen, z)
            + _tlc_laf(wavelen, z)
            + _tlc_dla(wavelen, z)
        )

    def _igm_transmission(self, wavelen: float | np.ndarray, z: float) -> np.ndarray:
        """Computes IGM transmission function using the Inoue model

        Parameters
        ----------
        wavelen : float | np.ndarray
            Observed wavelength in Angstroms
        z : float
            Redshift
    
        Returns
        -------
        np.ndarray
           IGM transmission T(wavelen, z)
        """
        tau = _igm_tau(wavelen,z)
        return np.exp(-tau)

    def _get_uv_slope(self, u, g, mean_wavelen_u, mean_wavelen_g):
        return (u - g)/(-2.5*np.log10(mean_wavelen_u/mean_wavelen_g)) - 2 

    def _compute_mean_wavelen_filter(self, band, beta_uv=None):
        if beta_uv == None:
            beta_uv = self.beta_uv_init
        return np.sum(self.wavelen**(beta_uv+2) * self.filters[band] * self.dwavelen) / np.sum( self.wavelen**(beta_uv+1) * self.filters[band] * self.dwavelen)
    
    def _initNoiseModel(self):

        filter_list = self.config.filter_list
        filters = {}
        for i, f in enumerate(filter_list[:2]):
            fin = np.loadtxt(self.data_path + "/FILTER/" + f + ".res")
            filters[self.config.bands[i]]=fin[:,1]

        self.filters = filters
        self.wavelen = fin[:,0]
        self.dwavelen = self.wavelen[1]- self.wavelen[0] 

        if self.config.compute_uv_slope == True:
            # these are computed using initial guess of beta_uv=-2
            self.mean_wavelen_u = self._compute_mean_wavelen_filter(self.config.bands[0])
            self.mean_wavelen_g = self._compute_mean_wavelen_filter(self.config.bands[1])
        else:
            self.mean_wavelen_u = None
            self.mean_wavelen_g = None   
        
    
    def _addNoise(self):
        # Note: we assume the order of band and filter_list are 
        # corresponding to each other, and the first two bands 
        # must be u and g!
        data = self.get_data("input")
        Nobj = len(data[self.config.redshift_col])

        outData = data.copy()
        for i in range(Nobj):
            # currently this is not efficient
            T_igm = self._igm_transmission(self.wavelen, data[self.config.redshift_col][i])
            
            if self.config.compute_uv_slope == False:
                beta_uv = self.beta_uv_init
            else:
                beta_uv = self._get_uv_slope(data[self.config.bands[0]][i],
                                       data[self.config.bands[1]][i],
                                        self.mean_wavelen_u, self.mean_wavelen_g)

            for band in self.config.bands[:2]:
                # compute this for u and g band only, as
                # other bands have negligible impact
                R_m = self.filters[band]
                R_m_norm = np.sum(self.wavelen**(beta_uv + 1) * R_m * self.dwavelen)
                R_m = self.wavelen**(beta_uv + 1)*R_m / R_m_norm
                flux_ratio_igm = np.sum(T_igm * R_m * self.dwavelen)
                delta_m= -2.5*np.log10(flux_ratio_igm)
                outData[band][i] = data[band][i] - delta_m

        self.add_data('output', outData)