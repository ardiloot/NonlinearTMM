"""This module contains physical constants and functions to convert between
different units.

Attributes:
    c (float): speed of light (m/s)
    h (float): Planck's constant (J*s)
    hp (float): Reduced Planck's constant (J*s)
    qe (float): Elementary charge (C)
    eps0 (float): Vacuum permittivity (F/m)
    mu0 (float): Vacuum permeability (A/m)

"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from LabPy import Core

# ===============================================================================
# Definition of constants
# ===============================================================================


c = 299792458.0
h = 6.62606957e-34
hp = h / (2.0 * math.pi)
qe = 1.602176565e-19
eps0 = 8.854187817e-12
mu0 = 1.2566370614e-6
pi = math.pi
kB = 1.38064852e-23


# ===============================================================================
# Wavelength to ... conversion functions
# ===============================================================================


def WlToJoule(wl: ArrayLike) -> ArrayLike:
    """Converts wavelength (m) to photon energy (J).

    Args:
        wl (float): wavelength (m)

    Returns:
        float: energy (J)

    """
    E = h * c / wl
    return E


def WlToEv(wl: ArrayLike) -> ArrayLike:
    """Converts wavelength (m) to energy (eV).

    Args:
        wl (float): wavelength

    Returns:
        float: energy (eV)

    """
    ev = h * c / (wl * qe)
    return ev


def WlToFreq(wl: ArrayLike) -> ArrayLike:
    """Converts wavelength to frequency (Hz).

    Args:
    wl (float): wavelength (m)

    Returns:
        float: frequency (Hz)

    """
    freq = c / wl
    return freq


def WlToOmega(wl: ArrayLike) -> ArrayLike:
    """Converts wavelength to angular frequency (rad/s).

    Args:
        wl (float): wavelength (m)

    Returns:
        float: angular frequency (rad/s)

    """
    omega = 2.0 * math.pi * c / wl
    return omega


# ===============================================================================
# ... to wavelength conversion functions
# ===============================================================================


def JouleToWl(E: ArrayLike) -> ArrayLike:
    """Converts energy of a photon (J) to wavelength (m).

    Args:
        E (float): energy (J)

    Returns:
        float: wavelength (nm)

    """
    wl = h * c / E
    return wl


def EvToWl(ev: ArrayLike) -> ArrayLike:
    """Converts energy (eV) to wavelength (m).

    Args:
        ev (float): energy (eV)

    Returns:
        float: wavelength (m)

    """
    wl = h * c / (ev * qe)
    return wl


def FreqToWl(freq: ArrayLike) -> ArrayLike:
    """Converts frequency (1/s) to wavelength (m).

    Args:
        freq (float): frequency (1/s)

    Returns:
        float: wavelength (m)

    """
    wl = c / freq
    return wl


def OmegaToWl(omega: ArrayLike) -> ArrayLike:
    """Converts angular frequency (rad/s) to wavelength (m).

    Args:
        omega (float): angular frequency (rad/s)

    Returns:
        float: wavelength (m)

    """
    wl = 2.0 * math.pi * c / omega
    return wl


# ===============================================================================
# Other conversion functions
# ===============================================================================


def EvToJoule(ev: ArrayLike) -> ArrayLike:
    """Converts eV-s to joules.

    Args:
        ev (float): Energy in eV

    Returns:
        float: Energy in J

    """

    return ev * qe


def JouleToEv(E: ArrayLike) -> ArrayLike:
    """Converts joules to eV-s.

    Args:
        E (float): Energy in J

    Returns:
        float: Energy in eV

    """

    return E / qe


def EvToOmega(ev: ArrayLike) -> ArrayLike:
    """Converts eV-s to angular frequency.

    Args:
        ev (float): Energy in eV

    Returns:
        float: Photon angular frequency (rad/s)

    """
    return 2.0 * math.pi * ev * qe / h


def OmegaToEv(omega: ArrayLike) -> ArrayLike:
    """Converts angular frequency to eV-s.

    Args:
        omega (float): photon angular frequency (rad/s)

    Returns:
        float: Photon energy (eV)

    """
    return omega * h / (2.0 * pi * qe)


# ===============================================================================
# Distributions
# ===============================================================================


class DetlaDistribution(Core.ParamsBaseClass):
    """This class presents delta function distribution.

    Delta function distribution is zero except at x=x0 where it is 1.0.

    Kwargs:
        x0=0.0 (float): The position of delta function

    """

    def __init__(self, **kwargs: Any) -> None:
        self._params = ["x0"]
        self.x0 = 0.0

        super().__init__(**kwargs)

    def __call__(self, x: float) -> float:
        """Returns distribution value at position x.

        Args:
            x (float): The point were to evaluate the distribution

        Returns:
            float: Distribution value at position x

        """
        if x == self.x0:
            return 1.0
        else:
            return 0.0

    def GetDist(self) -> tuple[np.ndarray, np.ndarray]:
        """Function returns (np.array([x0]), np.array([1.0])).

        Returns:
            tuple_of_arrays: Distribution positions and values

        """
        return (np.array([self.x0]), np.array([1.0]))


class NormalDistribution(Core.ParamsBaseClass):
    """This class presents normal distribution.

    Kwargs:
        x0=0.0 (float): mean
        std=1.0 (float): standard deviation
        nPoints=30 (int): number of points in distribution
        rangeB=3.0 (float): beginning of sampling range is x0 - ramgeB * std
        rangeE=3.0 (float): end of sampling range is x0 + ramgeE * std

    """

    def __init__(self, **kwargs: Any) -> None:
        self._params = ["x0", "std", "nPoints", "rangeB", "rangeE"]
        self.x0 = 50e-9
        self.std = 10e-9
        self.nPoints = 30
        self.rangeB = 3.0
        self.rangeE = 3.0

        super().__init__(**kwargs)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Returns distribution value at position x.

        Args:
            x (float): The point were to evaluate the distribution

        Returns:
            float: Distribution value at position x

        """
        res = 1.0 / (self.std * math.sqrt(2.0 * math.pi)) * np.exp(-((x - self.x0) ** 2.0) / (2.0 * self.std**2.0))

        return res

    def GetDist(self) -> tuple[np.ndarray, np.ndarray]:
        """Function returns distribution at nPoints sampling points.

        The sampling range is from max(1e-16, x0 - rangeB * std) to \
        x0 + rangeE * std where nPoints sampling points are taken.

        Returns:
            tuple_of_arrays: Distribution positions and values

        """
        xs = np.linspace(max(1e-12, self.x0 - self.rangeB * self.std), self.x0 + self.rangeE * self.std, self.nPoints)

        ws = self.__call__(xs)
        return xs, ws


class LogNormalDistribution(NormalDistribution):
    """This class presents lognormal distribution.

    Kwargs:
        x0=0.0 (float): mean
        std=1.0 (float): standard deviation
        nPoints=30 (int): number of points in distribution
        rangeB=0.0 (float): beginning of sampling range
        rangeE=10.0 (float): end of sampling range

    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Returns distribution value at position x.

        Args:
            x (float): The point were to evaluate the distribution

        Returns:
            float: Distribution value at position x

        """
        # x02 = math.log(self.x0 ** 2.0 / math.sqrt(self.std + self.x0 ** 2.0))
        # std2 = math.sqrt(math.log(1.0 + self.std / self.x0 ** 2.0))
        x02 = np.log(self.x0)
        std2 = self.std

        # res = 1.0 / (std2 * math.sqrt(2.0 * math.pi) * x) * \
        #    np.exp(- (np.log(x) - x02) ** 2.0 / (2.0 * std2 ** 2.0))

        res = 1.0 / (std2 * math.sqrt(2.0 * math.pi)) * np.exp(-((np.log(x) - x02) ** 2.0) / (2.0 * std2**2.0))

        return res

    def GetDist(self) -> tuple[np.ndarray, np.ndarray]:
        """Function returns distribution at nPoints sampling points.

        The sampling range is from max(1e-16, x0 - rangeB * std) to \
        x0 + rangeE * std where nPoints sampling points are taken.

        Returns:
            tuple_of_arrays: Distribution positions and values

        """
        xs = np.linspace(self.rangeB, self.rangeE, self.nPoints)

        ws = self.__call__(xs)
        return xs, ws


class LogNormalDistributionLocal(Core.ParamsBaseClass):
    """This class presents lognormal distribution with usual local parameters.

    Kwargs:
        xCoef (float): x-scale linear transformation parameter
        xOffset (float): x-offset value (new_x = x + xOffset)
        mu (float): location parameter
        sigma (float): scale parameter
        nPoint (int): number of points in distribution
        rangeB (float): beginning of sampling range
        rangeE (float): end of sampling range


    """

    def __init__(self, **kwargs: Any) -> None:
        self._params = ["xCoef", "xOffset", "mu", "sigma", "nPoints", "distB", "distE"]
        self.xCoef = 1.0
        self.xOffset = 0.0
        self.mu = 0.0
        self.sigma = 1.0
        self.nPoints = 30
        self.distB = 0.0
        self.distE = 3.0

        super().__init__(**kwargs)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Returns distribution value at position x.

        Args:
            x (float): The point were to evaluate the distribution

        Returns:
            float: Distribution value at position x

        """

        x = self.xCoef * (x - self.xOffset)

        res = (
            1.0
            / (self.sigma * math.sqrt(2.0 * math.pi) * x)
            * np.exp(-((np.log(x) - self.mu) ** 2.0) / (2.0 * self.sigma**2.0))
        )

        return res

    def GetDist(self) -> tuple[np.ndarray, np.ndarray]:
        """Function returns distribution at nPoints sampling points.

        The sampling range is from max(1e-16, distB) to \
        distE where nPoints sampling points are taken.

        Returns:
            tuple_of_arrays: Distribution positions and values

        """
        xs = np.linspace(max(self.xOffset + 1e-16, self.distB), self.distE, self.nPoints)

        ws = self.__call__(xs)
        return xs, ws


# ===============================================================================
# Distribution factory
# ===============================================================================

_DISTRIBUTIONS: dict[str, type] = {
    "DetlaDistribution": DetlaDistribution,
    "NormalDistribution": NormalDistribution,
    "LogNormalDistribution": LogNormalDistribution,
    "LogNormalDistributionLocal": LogNormalDistributionLocal,
}


def GetDistributionNames() -> list[str]:
    return list(_DISTRIBUTIONS)


def Distribution(
    name: str, **kwargs: Any
) -> DetlaDistribution | NormalDistribution | LogNormalDistribution | LogNormalDistributionLocal:
    if name not in _DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution name: {name}")
    return _DISTRIBUTIONS[name](**kwargs)
