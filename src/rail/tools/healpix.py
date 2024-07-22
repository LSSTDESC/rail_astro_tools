from __future__ import annotations

import operator

import healpy
import numpy as np
from numpy.typing import ArrayLike, NDArray

TypeRaDecTuple = tuple[NDArray[np.float64], NDArray[np.float64]]


MAX_ORDER = 29
MAX_NSIDE = 2**MAX_ORDER


def pix_uniform_randoms(
    nside: int, ipix: NDArray[np.int64], n_rand: int
) -> TypeRaDecTuple:
    ipix = np.asarray(ipix)
    if np.any(ipix >= healpy.nside2npix(nside)):
        raise ValueError(f"'ipix' exceeds limit for {nside=}")

    order = healpy.nside2order(nside)
    ipix_scale = 4 ** (MAX_ORDER - order)
    ipix_rand = (ipix * ipix_scale) + np.random.randint(0, ipix_scale, size=n_rand)
    return healpy.pix2ang(nside=MAX_NSIDE, ipix=ipix_rand, nest=True, lonlat=True)


def map_uniform_randoms(
    values: NDArray,
    n_rand: int,
    *,
    values_as_density: bool = True,
    nest: bool = False,
) -> TypeRaDecTuple:
    values = np.asarray(values)
    try:
        assert values.ndim == 1
        nside = healpy.npix2nside(len(values))
    except Exception as err:
        raise ValueError("invalid 'values' array shape") from err

    # determine how often to draw from which pixel
    ipix_nonzero = np.where(values != 0.0)[0]
    if values_as_density:
        density = values / np.sum(values)
        ipix = np.random.choice(ipix_nonzero, size=n_rand, p=density[ipix_nonzero])
    else:
        ipix = np.random.choice(ipix_nonzero, size=n_rand)
    if not nest:
        ipix = healpy.ring2nest(nside, ipix=ipix)

    return pix_uniform_randoms(nside, ipix, n_rand)


class HealpixMap:
    def __init__(self, values: NDArray, *, nest: bool = False) -> None:
        self.values = np.asarray(values)
        if self.values.ndim != 1 or not healpy.isnpixok(self.npix):
            raise ValueError("invalid 'values' array shape")
        self._nside: int = healpy.npix2nside(self.npix)
        self._nest = nest

    def __repr__(self) -> str:
        nside = self.nside
        npix = self.npix
        return f"{self.__class__.__name__}({nside=}, {npix=})"

    @property
    def nside(self) -> int:
        return self._nside

    @property
    def nest(self) -> int:
        return self._nest

    @property
    def ipix(self) -> NDArray[np.int64]:
        return np.arange(0, self.npix)

    def get_ipix_nonzero(self) -> NDArray[np.int64]:
        return np.where(self.values != 0.0)[0]

    @property
    def npix(self) -> int:
        return len(self.values)

    def get_npix_nonzero(self) -> int:
        return np.count_nonzero(self.values)

    @classmethod
    def read_fits(cls, fpath: str, *, nest: bool = False) -> HealpixMap:
        return cls(healpy.read_map(fpath), nest=nest)

    @classmethod
    def empty(cls, nside: int, *, nest: bool = False) -> HealpixMap:
        if not healpy.isnsideok(nside):
            raise ValueError(f"invalid {nside=}")
        npix = healpy.nside2npix(nside)
        values = np.zeros(npix)
        return cls(values, nest=nest)

    @classmethod
    def rectangle(
        cls,
        nside: int,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
        *,
        nest: bool = False,
    ) -> HealpixMap:
        new = cls.empty(nside, nest=nest)
        ra, dec = new.get_centers()
        mask = (ra >= ra_min) & (ra < ra_max) & (dec >= dec_min) & (dec < dec_max)
        return cls(mask, nest=nest)

    @classmethod
    def from_ipix(
        cls,
        nside: int,
        ipix: NDArray[np.int64],
        values: NDArray | float = 1.0,
        *,
        nest: bool = False,
    ) -> HealpixMap:
        new = cls.empty(nside, nest=nest)
        new.values[ipix] = values
        return new

    @classmethod
    def build(
        cls,
        nside: int,
        ra: NDArray[np.float64],
        dec: NDArray[np.float64],
        count: bool = False,
        *,
        nest: bool = False,
    ) -> HealpixMap:
        ipix = healpy.ang2pix(nside, ra, dec, lonlat=True, nest=nest)
        ipix_unique, pix_count = np.unique(ipix, return_counts=True)
        return cls.from_ipix(nside, ipix_unique, values=pix_count if count else 1.0, nest=nest)

    def _operator(self, other, operator_func):
        if isinstance(other, self.__class__):
            for attr in ("nside", "nest"):
                if getattr(self, attr) != getattr(other, attr):
                    raise ValueError(f"'{attr}' does not match")
            return self.__class__(
                operator_func(self.values, other.values),
                nest=self.nest,
            )
        return NotImplemented

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return (
                self.nside == other.nside
                and self.nest == other.nest
                and np.all(self.values, other.values)
            )
        return NotImplemented

    def __add__(self, other) -> HealpixMap:
        return self._operator(other, operator.add)

    def __sub__(self, other) -> HealpixMap:
        return self._operator(other, operator.sub)

    def __mul__(self, other) -> HealpixMap:
        return self._operator(other, operator.mul)

    def __truediv__(self, other) -> HealpixMap:
        return self._operator(other, operator.__truediv__)

    def __getitem__(self, ipix) -> ArrayLike:
        return self.values[ipix]

    def copy(self) -> HealpixMap:
        return self.__class__(self.values.copy(), nest=(self.nest == True))

    def to_resolution(self, nside: int, *, invariant: bool = False) -> HealpixMap:
        return self.__class__(
            healpy.ud_grade(self.values, nside, power=-2 if invariant else None),
            nest=self.nest,
        )

    def clip_values(self, lower: float | None = None, upper: float | None = None) -> HealpixMap:
        new = self.copy()
        if lower is not None:
            new.values = np.where(new.values < lower, lower, new.values)
        if upper is not None:
            new.values = np.where(new.values > upper, upper, new.values)
        return new

    def clip_negative(self) -> HealpixMap:
        return self.clip_values(lower=0.0)

    def as_mask(self) -> HealpixMap:
        new = self.clip_negative()
        new.values = new.values.astype(np.bool_)
        return new

    def area(self, count_nonzero: bool = True) -> float:
        if count_nonzero:
            area_frac = self.get_npix_nonzero() / self.npix
        else:
            area_frac = self.values.sum() / self.npix
        return area_frac * (360.0 * 360.0 / np.pi)  # convert to deg^2

    def get_centers(self) -> TypeRaDecTuple:
        return healpy.pix2ang(
            self.nside, self.get_ipix_nonzero(), nest=self.nest, lonlat=True
        )

    def get_ipix_from_coord(
        self,
        ra: ArrayLike[np.float64],
        dec: ArrayLike[np.float64],
    ) -> ArrayLike[np.int64]:
        return healpy.ang2pix(self.nside, ra, dec, lonlat=True, nest=self.nest)

    def isin_mask(
        self,
        ra: ArrayLike[np.float64],
        dec: ArrayLike[np.float64],
    ) -> ArrayLike[np.bool_]:
        ipix = self.get_ipix_from_coord(ra, dec)
        return self.values[ipix] != 0.0

    def apply(
        self,
        ra: NDArray[np.float64],
        dec: NDArray[np.float64],
    ) -> TypeRaDecTuple:
        mask = self.isin_mask(ra, dec)
        return ra[mask], dec[mask]

    def uniform_randoms(
        self,
        n_rand: int,
        *,
        values_as_density: bool = True,
    ) -> TypeRaDecTuple:
        return map_uniform_randoms(
            self.values,
            n_rand, values_as_density=values_as_density,
            nest=self.nest,
        )
