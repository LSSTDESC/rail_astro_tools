"""
Module that implements operations on catalog data such as size and ellipticity.
"""
from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd

from ceci.config import StageParameter as Param
from rail.core.data import PqHandle
from rail.core.stage import RailStage
from rail.core.data import PqHandle
from rail.core.common_params import SHARED_PARAMS

# default pixel size for lsst
ARCSEC_PER_PIX = 0.2

class CatalogManipulator(RailStage, ABC):
    """
    Base class to perform opertations on catalog size columns. A table with input size-related columns is
    processed and transformed into an output table with semi-major and semi-minor axis in arcsec.

    Subclasses must implement the run() and compute() method.
    """

    name = 'CatalogManipulator'
    config_options = RailStage.config_options.copy()
    config_options.update(
        major_columns=Param(
            str, default="major",
            msg="column names for semi-major axes."),
        minor_columns=Param(
            str, default="minor",
            msg="column names for semi-major axes."),
        to_arcsec=Param(
            bool, default=False,
            msg="Whether apply conversion from pixel size to arcsec size."),
        arcsec_per_pix=Param(
            float, default=ARCSEC_PER_PIX,
            msg="Size of the pixel in arcsec."),
    )
    
    inputs = [('input', PqHandle)]
    outputs = [('output', PqHandle)]

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.major_columns = self.config.major_columns
        self.minor_columns = self.config.minor_columns 
        self.arcsec_per_pix = self.config.arcsec_per_pix
        self.to_arcsec = self.config.to_arcsec

    @abstractmethod
    def _get_AB(self, input_data):
        """
        Implmenet conversion code from other size information to semi-major/minor axes.
        """
        pass
    
    def run(self):  # pragma: no cover
        """
        Implements the operation performed on the photometric data.
        """
        input_data = self.get_data('input', allow_missing=True)
        a, b = self._get_AB(input_data)
        output = {}
        output[self.major_columns] = a
        output[self.minor_columns] = b
        output = pd.DataFrame(output, index=input_data.index)
        if self.to_arcsec == True:
            output[self.major_columns] *= self.arcsec_per_pix
            output[self.minor_columns] *= self.arcsec_per_pix
        # attach the semi-major minor axes columns to the original data frame
        output = pd.concat([input_data, output], axis=1)
        self.add_data('output', output)

    def compute(self, data):  # pragma: no cover
        """
        Main method to call.

        Parameters
        ----------
        data : `PqHandle`
           Input tabular data with column names as defined in the configuration.

        Returns
        -------
        output: `PqHandle`
            Output tabular data.
        """
        self.set_data('input', data)
        self.run()
        self.finalize()
        return self.get_handle('output')


class MajorEllipticityToAB(CatalogManipulator):
    """
    Convert semi-major axis (size) and ellipticity columns to semi-major and minor axes
    """
    name = 'MajorEllipticityToAB'
    config_options = CatalogManipulator.config_options.copy()
    config_options.update(
        size_column=Param(
            str, default="size",
            msg="column names for size (here defined to be the same as the major axis)"),
        ellipticity_column=Param(
            str, default="ellipticity",
            msg="column names for ellipticity"),
    )
    inputs = [('input', PqHandle)]
    outputs = [('output', PqHandle)]

    def _get_AB(self, input_data):
        size = input_data[self.config.size_column]
        e = input_data[self.config.ellipticity_column]
        q = (1 - e)/(1 + e)
        a = size
        b = a*q
        return a, b

class SizeEllipticityToAB(CatalogManipulator):
    """
    Convert size and ellipticity columns to semi-major and minor axes
    """

    name = 'SizeEllipticityToAB'
    config_options = CatalogManipulator.config_options.copy()
    config_options.update(
        size_column=Param(
            str, default="size",
            msg="column names for size (here defined to be the geometric average of semi-major and minor axes)"),
        ellipticity_column=Param(
            str, default="ellipticity",
            msg="column names for ellipticity"),
    )
    inputs = [('input', PqHandle)]
    outputs = [('output', PqHandle)]

    def _get_AB(self, input_data):
        size = input_data[self.config.size_column]
        e = input_data[self.config.ellipticity_column]
        q = (1 - e)/(1 + e)
        b = np.sqrt(size**2*q)
        a = size**2/b
        return a, b

class BulgeDiscSizeEllipticityToAB(CatalogManipulator):
    """
    First compute galaxy size from bulge and disk sizes in the simulation,
    then convert size and ellipticity columns to semi-major and minor axes
    """

    name = 'BulgeDiscSizeEllipticityToAB'
    config_options = CatalogManipulator.config_options.copy()
    config_options.update(
        bulge_size_column=Param(
            str, default="size_bulge_true",
            msg="Bulge size"),
        disk_size_column=Param(
            str, default="size_disk_true",
            msg=""),
        bulge_to_total_ratio_colum=Param(
            str, default="bulge_to_total_ratio",
            msg="column names for size (here defined to be the geometric average of semi-major and minor axes)"),
        ellipticity_column=Param(
            str, default="ellipticity_true",
            msg="column names for ellipticity"),
    )
    inputs = [('input', PqHandle)]
    outputs = [('output', PqHandle)]

    def _get_AB(self, input_data):
        f=input_data[self.config.bulge_to_total_ratio_colum]
        size=input_data[self.config.bulge_size_column]*f + input_data[self.config.disk_size_column]*(1-f)
        e = input_data[self.config.ellipticity_column]
        q = (1 - e)/(1 + e)
        b = np.sqrt(size**2*q)
        a = size**2/b
        return a, b

class MomentsToAB(CatalogManipulator):
    """
    Convert image moments to semi-major and minor axes.
    """

    name = 'MomentsToAB'
    config_options = CatalogManipulator.config_options.copy()
    config_options.update(
        xx_column=Param(
            str, default="shape_xx",
            msg="column names for moment xx"),
        xy_column=Param(
            str, default="shape_xy",
            msg="column names for moment xy"),
        yy_column= Param(
            str, default="shape_yy",
            msg="column names for moment yy"),
    )
    inputs = [('input', PqHandle)]
    outputs = [('output', PqHandle)]

    def _get_AB(self, input_data):
        xx = input_data[self.config.xx_column]
        xy = input_data[self.config.xy_column]
        yy = input_data[self.config.yy_column]
        xx_p_yy = xx + yy
        xx_m_yy = xx - yy
        t = np.sqrt(xx_m_yy * xx_m_yy + 4 * xy * xy)
        a = np.sqrt(0.5 * (xx_p_yy + t))
        b = np.sqrt(0.5 * (xx_p_yy - t))
        #theta = 0.5 * np.atan(2.0 * xy/xx_m_yy)
        return a, b



    


    




