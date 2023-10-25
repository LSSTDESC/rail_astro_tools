"""Degrader applied to the magnitude error based on a set of input observing condition maps"""

import os
from dataclasses import fields

import healpy as hp
import numpy as np
import pandas as pd
from ceci.config import StageParameter as Param
from photerr import LsstErrorModel, LsstErrorParams

from rail.creation.degrader import Degrader


class ObsCondition(Degrader):
    """Photometric errors based on observation conditions

    This degrader calculates spatially-varying photometric errors
    using input survey condition maps. The error is based on the
    LSSTErrorModel from the PhotErr python package.

    Parameters
    ----------
    nside: int, optional
        nside used for the HEALPIX maps.
    mask: str, optional
        Path to the mask covering the survey
        footprint in HEALPIX format. Notice that
        all negative values will be set to zero.
    weight: str, optional
        Path to the weights HEALPIX format, used
        to assign sample galaxies to pixels. Default
        is weight="", which uses uniform weighting.
    tot_nVis_flag: bool, optional
        If any map for nVisYr are provided, this flag
        indicates whether the map shows the total number of
        visits in nYrObs (tot_nVis_flag=True), or the average
        number of visits per year (tot_nVis_flag=False). The
        default is set to True.
    random_seed: int, optional
        A random seed for reproducibility.
    map_dict: dict, optional
        A dictionary that contains the paths to the
        survey condition maps in HEALPIX format. This dictionary
        uses the same arguments as LSSTErrorModel (from PhotErr).
        The following arguments, if supplied, may contain either
        a single number (as in the case of LSSTErrorModel), or a path:
        [m5, nVisYr, airmass, gamma, msky, theta, km, tvis, EBV]
        For the following keys:
        [m5, nVisYr, gamma, msky, theta, km]
        numbers/paths for specific bands must be passed.
        Example:
        {"m5": {"u": path, ...}, "theta": {"u": path, ...},}
        Other LSSTErrorModel parameters can also be passed
        in this dictionary (e.g. a necessary one may be [nYrObs]
        for the survey condition maps).
        If any argument is not passed, the default value in
        PhotErr's LsstErrorModel is adopted.

    """

    name = "ObsCondition"
    config_options = Degrader.config_options.copy()
    config_options.update(
        nside=Param(
            int,
            128,
            msg="nside for the input maps in HEALPIX format.",
        ),
        mask=Param(
            str,
            os.path.join(
                os.path.dirname(__file__),
                "../../examples_data/creation_data/data/survey_conditions/DC2-mask-neg-nside-128.fits",
            ),
            msg="mask for the input maps in HEALPIX format.",
        ),
        weight=Param(
            str,
            os.path.join(
                os.path.dirname(__file__),
                "../../examples_data/creation_data/data/survey_conditions/DC2-dr6-galcounts-i20-i25.3-nside-128.fits",
            ),
            msg="weight for assigning pixels to galaxies in HEALPIX format.",
        ),
        tot_nVis_flag=Param(
            bool,
            True,
            msg="flag indicating whether nVisYr is the total or average per year if supplied.",
        ),
        random_seed=Param(int, 42, msg="random seed for reproducibility"),
        map_dict=Param(
            dict,
            {
                "m5": {
                    "i": os.path.join(
                        os.path.dirname(__file__),
                        "../../examples_data/creation_data/data/survey_conditions/minion_1016_dc2_Median_fiveSigmaDepth_i_and_nightlt1825_HEAL.fits",
                    ),
                },
                "nYrObs": 5.0,
            },
            msg="dictionary containing the paths to the survey condition maps and/or additional LSSTErrorModel parameters.",
        ),
    )
    # define constants:
    STANDARD_BANDS = ["u","g","r","i","z","y"]
    # set the A_lamba/E(B-V) values for the six LSST filters 
    BAND_A_EBV = {
            "u":4.81,
            "g":3.64,
            "r":2.70,
            "i":2.06,
            "z":1.58,
            "y":1.31,
        }
    

    def __init__(self, args, comm=None):
        Degrader.__init__(self, args, comm=comm)

        # store a list of keys relevant for
        # survey conditions;
        # a path to the survey condition
        # map or a float number should be
        # provided if these keys are provided
        self.obs_cond_keys = [
            "m5",
            "nVisYr",
            "airmass",
            "gamma",
            "msky",
            "theta",
            "km",
            "tvis",
            "EBV",
        ]

        # validate input parameters
        self._validate_obs_config()

        # initiate self.maps
        self.maps = {}

        # load the maps
        self._get_maps()

    def _validate_obs_config(self):
        """
        Validate the input
        """

        ### Check nside type:
        # check if nside < 0
        if self.config["nside"] < 0:
            raise ValueError("nside must be positive.")

        # check if nside is powers of two
        if not np.log2(self.config["nside"]).is_integer():
            raise ValueError("nside must be powers of two.")

        ### Check mask type:
        # check if mask is provided
        if self.config["mask"] == "":
            raise ValueError("mask needs to be provided for the input maps.")

        # check if the path exists
        if not os.path.exists(self.config["mask"]):
            raise ValueError("The mask file is not found: " + self.config["mask"])

        ### Check weight type:
        if self.config["weight"] != "":
            # check if the path exists
            if not os.path.exists(self.config["weight"]):
                raise ValueError("The weight file is not found: " + self.config["weight"])

        ### Check map_dict:

        # Check if extra keys are passed
        # get lsst_error_model keys
        lsst_error_model_keys = list(LsstErrorParams.__dataclass_fields__.keys())
        if len(set(self.config["map_dict"].keys()) - set(lsst_error_model_keys)) != 0:
            extra_keys = set(self.config["map_dict"].keys()) - set(lsst_error_model_keys)
            # now we added EBV, which is not in LsstErrorParams:
            if extra_keys != {"EBV"}:
                raise ValueError("Extra keywords are passed to the configuration: \n" + str(extra_keys))

        # Check data type for the keys:
        # Note that LSSTErrorModel checks
        # the data type for its parameters,
        # so here we only check the additional
        # parameters and the file paths
        # nYrObs may be used below, so we
        # check its type as well

        if len(self.config["map_dict"]) > 0:

            for key in self.config["map_dict"]:

                if key == "nYrObs":
                    if not isinstance(self.config["map_dict"][key], float):
                        raise TypeError("nYrObs must be a float.")

                elif key in self.obs_cond_keys:

                    # band-independent keys:
                    if key in ["airmass", "tvis", "EBV"]:

                        # check if the input is a string or number
                        if not (
                            isinstance(self.config["map_dict"][key], str)
                            or isinstance(self.config["map_dict"][key], float)
                        ):
                            raise TypeError(f"{key} must be a path (string) or a float.")

                        # check if the paths exist
                        if isinstance(self.config["map_dict"][key], str):
                            if not os.path.exists(self.config["map_dict"][key]):
                                raise ValueError(
                                    "The following file is not found: " + self.config["map_dict"][key]
                                )

                    # band-dependent keys
                    else:

                        # they must be dictionaries:
                        if not isinstance(self.config["map_dict"][key], dict):  # pragma: no cover
                            raise TypeError(f"{key} must be a dictionary.")

                        # the dictionary cannot be empty
                        if len(self.config["map_dict"][key]) == 0:
                            raise ValueError(f"{key} is empty.")

                        for band in self.config["map_dict"][key].keys():

                            # check if the input is a string or float:
                            if not (
                                isinstance(self.config["map_dict"][key][band], str)
                                or isinstance(self.config["map_dict"][key][band], float)
                            ):
                                raise TypeError(f"{key}['{band}'] must be a path (string) or a float.")

                            # check if the paths exist
                            if isinstance(self.config["map_dict"][key][band], str):
                                if not os.path.exists(self.config["map_dict"][key][band]):
                                    raise ValueError(
                                        "The following file is not found: "
                                        + self.config["map_dict"][key][band]
                                    )

    def _get_maps(self):
        """
        Load in the maps from the paths provided by map_dict,
        if it is not empty
        A note on nVisYr: input map usually in terms of
                          total number of exposures, so
                          manually divide the map by nYrObs
        """

        maps = {}

        # Load mask
        mask = hp.read_map(self.config["mask"])
        if (mask < 0).any():
            # set negative values (if any) to zero
            mask[mask < 0] = 0
        pixels = np.arange(int(self.config["nside"] ** 2 * 12))[mask.astype(bool)]
        maps["pixels"] = pixels

        # Load weight if given
        if self.config["weight"] != "":
            maps["weight"] = hp.read_map(self.config["weight"])[pixels]

        # Load all other maps in map_dict
        if len(self.config["map_dict"]) > 0:
            for key in self.config["map_dict"]:
                if key in self.obs_cond_keys:
                    # band-independent keys:
                    if key in ["airmass", "tvis", "EBV"]:
                        if isinstance(self.config["map_dict"][key], str):
                            maps[key] = hp.read_map(self.config["map_dict"][key])[pixels]
                        elif isinstance(self.config["map_dict"][key], float):
                            maps[key] = np.ones(len(pixels)) * self.config["map_dict"][key]
                    # band-dependent keys
                    else:
                        maps[key] = {}
                        for band in self.config["map_dict"][key].keys():
                            if isinstance(self.config["map_dict"][key][band], str):
                                maps[key][band] = hp.read_map(self.config["map_dict"][key][band])[pixels]
                            elif isinstance(self.config["map_dict"][key][band], float):
                                maps[key][band] = np.ones(len(pixels)) * self.config["map_dict"][key][band]
                else:
                    # copy all other lsst_error_model parameters supplied
                    maps[key] = self.config["map_dict"][key]

        if "nVisYr" in list(self.config["map_dict"].keys()):
            if "nYrObs" not in list(maps.keys()):
                # Set to default:
                maps["nYrObs"] = 10.0
            if self.config["tot_nVis_flag"] == True:
                # For each band, compute the average number of visits per year
                for band in maps["nVisYr"].keys():
                    maps["nVisYr"][band] /= float(maps["nYrObs"])

        self.maps = maps

    def get_pixel_conditions(self, pixel: int) -> dict:
        """
        get the map values at given pixel
        output is a dictionary that only
        contains the LSSTErrorModel keys
        """

        allpix = self.maps["pixels"]
        ind = allpix == pixel

        obs_conditions = {}
        for key in self.maps:
            # For keys that may contain the survey condition maps
            if key in self.obs_cond_keys:
                # band-independent keys:
                if key in ["airmass", "tvis", "EBV"]:
                    if key != "EBV":# exclude EBV because it is not in LsstErrorModel
                        obs_conditions[key] = float(self.maps[key][ind])
                # band-dependent keys
                else:
                    obs_conditions[key] = {}
                    for band in (self.maps[key]).keys():
                        obs_conditions[key][band] = float(self.maps[key][band][ind])
            # For other keys in LSSTErrorModel:
            elif key not in ["pixels", "weight"]:
                obs_conditions[key] = self.maps[key]
        # obs_conditions should now only contain the LSSTErrorModel keys
        return obs_conditions

    def assign_pixels(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """
        assign the pixels to the input catalog
        check if catalogue contains position information;
        if so, assign according to ra, dec;
        else, assign randomly.
        """
        pixels = self.maps["pixels"]

        if "renameDict" in self.maps and set(['ra','dec']).issubset(list(self.maps["renameDict"].keys())):
            # if catalog contains ra, dec, but needs renaming
            rakey=self.maps["renameDict"]['ra']
            deckey=self.maps["renameDict"]['dec']
            assigned_pix=hp.ang2pix(self.config["nside"], catalog[rakey].to_numpy(), catalog[deckey].to_numpy(), lonlat=True)            
        elif set(['ra','dec']).issubset(catalog.columns):
            # if catalog contains ra, dec, and no renaming
            assigned_pix=hp.ang2pix(self.config["nside"], catalog['ra'].to_numpy(), catalog['dec'].to_numpy(), lonlat=True)
        else:
            # if catalog doesn't contain position information 
            print("No ra, dec found in catalogue, randomly assign pixels with weights.")
            
            # load weights if specified, otherwise set to uniform weights
            if "weight" in self.maps:
                weights = self.maps["weight"]
                weights = weights / sum(weights)
            else:
                weights = None
            assigned_pix = self.rng.choice(pixels, size=len(catalog), replace=True, p=weights)
            
            # in this case, also attach the ra, dec columns in the data:
            ra, dec=hp.pix2ang(self.config["nside"],assigned_pix,lonlat=True)
            skycoord = pd.DataFrame(np.c_[ra,dec], columns=["ra","decl"])
            catalog = pd.concat([catalog, skycoord], axis=1)
            
        # this is the case where there are objects outside the footprint
        overlap=np.in1d(set(assigned_pix), pixels, assume_unique=True)
        if not (overlap==True).all():
            # flag all those pixels into -99
            print("Warning: objects found outside given mask, pixel assigned=-99. These objects will be assigned with defualt error from LSST error model!")
            ind=np.in1d(assigned_pix, pixels)
            assigned_pix[~ind]=-99
               
        # make it a DataFrame object
        assigned_pix = pd.DataFrame(assigned_pix, columns=["pixel"])
        # attach pixels to the catalogue
        catalog = pd.concat([catalog, assigned_pix], axis=1)

        return catalog
    
    # this is milky way extinction, should be added before other observing conditions is applied
    def apply_galactic_extinction(self, pixel: int, pixel_cat: pd.DataFrame) -> pd.DataFrame:
        """
        MW extinction reddening of the magnitudes
        """
        # find the corresponding ebv for the pixel
        ind = self.maps["pixels"]==pixel
        ebvvec = self.maps["EBV"][ind]
        
        if "renameDict" in self.maps:
            for b in self.STANDARD_BANDS:
                # check which bands are included in renameDict
                if b in self.maps["renameDict"]:
                    key=self.maps["renameDict"][b]
                    # update pixel_cat to the reddened magnitudes
                    pixel_cat[key] = (pixel_cat[key].copy())+ebvvec*self.BAND_A_EBV[b]
        else:
            # go through standard bands
            for b in self.STANDARD_BANDS:
                key=b
                # update pixel_cat to the reddened magnitudes
                pixel_cat[key] = (pixel_cat[key].copy())+ebvvec*self.BAND_A_EBV[b]

        return pixel_cat


    def run(self):
        """
        Run the degrader.
        """
        self.rng = np.random.default_rng(seed=self.config["random_seed"])

        catalog = self.get_data("input", allow_missing=True)

        # if self.map_dict empty, call LsstErrorModel:
        if len(self.config["map_dict"]) == 0:

            print("Empty map_dict, using default parameters from LsstErrorModel.")
            errorModel = LsstErrorModel()
            catalog = errorModel(catalog, random_state=self.rng)
            self.add_data("output", catalog)

        # if maps are provided, compute mag err for each pixel
        elif len(self.config["map_dict"]) > 0:

            # assign each galaxy to a pixel
            print("Assigning pixels.")
            catalog = self.assign_pixels(catalog)
            
            # loop over each pixel
            pixel_cat_list = []
            for pixel, pixel_cat in catalog.groupby("pixel"):
                
                # first, check if pixel is -99 - these objects have default obs_conditions:
                if pixel==-99:
                    # creating the error model for this pixel
                    errorModel = LsstErrorModel()
                
                else:            
                    # get the observing conditions for this pixel
                    obs_conditions = self.get_pixel_conditions(pixel)

                    # apply MW extinction if supplied, 
                    # replace the Mag column with reddened magnitudes:
                    if "EBV" in self.maps.keys():
                        pixel_cat = self.apply_galactic_extinction(pixel, pixel_cat)

                    # creating the error model for this pixel
                    errorModel = LsstErrorModel(**obs_conditions)

                # calculate the error model for this pixel
                obs_cat = errorModel(pixel_cat, random_state=self.rng)

                # add this pixel catalog to the list
                pixel_cat_list.append(obs_cat)

            # recombine all the pixels into a single catalog
            catalog = pd.concat(pixel_cat_list)

            # sort index
            catalog = catalog.sort_index()

            self.add_data("output", catalog)

    def __repr__(self):
        """
        Define how the model is represented and printed.
        """

        # start message
        printMsg = "Loaded observing conditions from configuration file: \n"

        printMsg += f"nside = {self.config['nside']}, \n"

        printMsg += f"mask file:  {self.config['mask']}, \n"

        printMsg += f"weight file:  {self.config['weight']}, \n"

        printMsg += f"tot_nVis_flag = {self.config['tot_nVis_flag']}, \n"

        printMsg += f"random_seed = {self.config['random_seed']}, \n"

        printMsg += "map_dict contains the following items: \n"

        printMsg += str(self.config["map_dict"])

        return printMsg
