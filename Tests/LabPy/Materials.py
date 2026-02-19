from __future__ import annotations

import math
import os
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d

from LabPy import Core

materialsDir = Path(__file__).resolve().parent / "../materials"
if not materialsDir.is_dir():
    materialsDir = Path(os.environ["PYLAB_MATERIALS_DIR"])

# ===============================================================================
# Methods
# ===============================================================================


def literal_unicode_representer(dumper: yaml.Dumper, data: str) -> Any:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


def GetMaterialsList() -> list[str]:
    """Generates list of all materials in database.

    Returns:
        list: list of paths to all available materials

    """
    res = ["Static"]
    res.extend(str(p.relative_to(materialsDir).with_suffix("")) for p in sorted(materialsDir.rglob("*.yml")))
    return res


def SaveMaterialFile(
    filename: str | Path,
    wls: NDArray[np.floating],
    n: NDArray[np.complexfloating],
) -> None:
    with open(filename, "w") as stream:
        data: dict[str, Any] = {}
        data["BOOK"] = "a"
        data["BOOK LONG"] = "b"
        data["PAGE"] = "c"
        data["PAGE LONG"] = "d"
        data["REFERENCES"] = "e"
        data["COMMENTS"] = "f"

        nkData = "\n".join(f"{1e6 * wl_val} {n_val.real} {n_val.imag}" for wl_val, n_val in zip(wls, n))

        class literal_unicode(str):
            pass

        yaml.add_representer(literal_unicode, literal_unicode_representer)
        data["DATA"] = {"type": "nk", "data": literal_unicode(nkData)}

        yaml.dump(data, stream, default_flow_style=False)


def MaterialFromConf(conf: tuple[str, dict[str, Any]]) -> Material:
    param, kwargParams = conf
    return Material(param, **kwargParams)


# ===============================================================================
# Classes
# ===============================================================================


class Material(Core.ParamsBaseClass):
    """This class reads and processes the refractiveindex.info database.

    Args:
        materialFile (str): material filename or Static

    Attributes:
        name (str): The name of material
        longName (str): The long name of material
        page (str): The page of material
        longPage (str): The long name of page of material

    """

    def __init__(self, materialFile: str, boundsError: bool = True, **kwargs: Any) -> None:
        self._params = ["kAdditional"]
        self.materialFile = materialFile
        self.wlExp = None
        self.nExp = None
        self.kExp = None
        self.kAdditional = 0.0
        self.boundsError = boundsError
        self.isFormula = False
        if materialFile == "Static":
            self.n = 1.0
            self.k = 0.0
            self.name = "Static"
            self.nFunc = lambda wl: np.ones_like(wl) * self.n
            self.kFunc = lambda wl: np.ones_like(wl) * self.k
            self._params += ["n", "k"]
        else:
            self._LoadFromFile(materialFile)

        super().__init__(**kwargs)

    def _LoadFromFile(self, materialFile: str) -> None:
        self.filename = materialsDir / f"{materialFile}.yml"

        with open(self.filename) as stream:
            self.rawData = yaml.safe_load(stream)

        self.nFunc = None
        self.kFunc = lambda wl: 0.0

        self.name = self.rawData.get("BOOK", None)
        self.longName = self.rawData.get("BOOK LONG", None)
        self.page = self.rawData.get("PAGE", None)
        self.longPage = self.rawData.get("PAGE LONG", None)
        self.comments = self.rawData.get("COMMENTS", None)
        self.references = self.rawData.get("REFERENCES", None)

        self._ReadDatapoints()
        self._ReadFormulas()

    def __call__(self, wl: ArrayLike) -> NDArray[np.complexfloating]:
        """Returns (interpolates if needed) complex refractive index at wavelength wl.

        Args:
            wl (float): wavelength of light

        Returns:
            complex: complex refractive index

        """
        res = self.nFunc(wl) + 1.0j * self.kFunc(wl) + 1.0j * self.kAdditional
        return res

    def GetN(self, wlFr: float = -math.inf, wlTo: float = -math.inf) -> tuple[NDArray, NDArray]:
        """Returns complex refractive index datapoints in range wlFr - wlTo.

        Args:
           wlFr (float, optional):  The minimum wavelength.
           wlTo (float, optional):  The maximum wavelength.

        Returns:
           tuple_of_arrays:  wavelength and complex refractive index arrays

        """
        index1 = bisect_left(self.wlExp, wlFr)
        index2 = bisect_right(self.wlExp, wlTo)

        wls = self.wlExp[index1:index2]
        n = self.nExp[index1:index2] + 1.0j * self.kExp[index1:index2]
        return wls, n

    # Private methods

    def _ReadDatapoints(self) -> None:
        if "DATA" not in self.rawData or "data" not in self.rawData["DATA"]:
            return

        dataType = self.rawData["DATA"]["type"]
        dataStr = self.rawData["DATA"]["data"].strip().split("\n")

        if dataType == "nk":
            data = np.zeros((len(dataStr), 3))
        elif dataType in ("n", "k"):
            data = np.zeros((len(dataStr), 2))
        else:
            raise ValueError("Unknown data type.")

        for i, line in enumerate(dataStr):
            data[i, :] = np.array(list(map(float, line.split())))

        match dataType:
            case "nk":
                self.wlExp, self.nExp, self.kExp = data.T
            case "n":
                self.wlExp, self.nExp = data.T
            case "k":
                self.wlExp, self.kExp = data.T

        # Convert wavelength to SI units
        self.wlExp *= 1e-6

        # Define interpolation functions
        if self.nExp is not None:
            self.nFunc = interp1d(self.wlExp, self.nExp, bounds_error=self.boundsError)

        if self.kExp is not None:
            self.kFunc = interp1d(self.wlExp, self.kExp, bounds_error=self.boundsError)

    def _ReadFormulas(self) -> None:
        if "FORMULA" in self.rawData:
            dataFormula = self.rawData["FORMULA"]
            wlRange = 1e-6 * np.array(dataFormula["range"].split()).astype(float)
            dispType = dataFormula["type"]
            coefs = np.array(dataFormula["coefficients"].split()).astype(float)
            match dispType:
                case 4:
                    self.nFunc = lambda wl: _DispersionFunc4(wl, coefs)
                case 2:
                    self.nFunc = lambda wl: _DispersionFunc2(wl, coefs)
                case 1:
                    self.nFunc = lambda wl: _DispersionFunc1(wl, coefs)
                case _:
                    raise NotImplementedError(f"Unsupported dispersion type: {dispType}")
            self.isFormula = True
            self.wlRange = wlRange

        if "DATA" in self.rawData and isinstance(self.rawData["DATA"], list):
            coefs = np.array(self.rawData["DATA"][0]["coefficients"].split()).astype(float)
            match self.rawData["DATA"][0]["type"]:
                case "formula 1":
                    self.nFunc = lambda wl: _DisperisonFuncFormula1(wl, coefs)
                case "formula 2":
                    self.nFunc = lambda wl: _DisperisonFuncFormula2(wl, coefs)
                case "formula 5":
                    self.nFunc = lambda wl: _DisperisonFuncFormula5(wl, coefs)
                case _ as formula_type:
                    raise NotImplementedError(f"Unsupported formula type: {formula_type}")


# ===============================================================================
# Dispersion functions
# ===============================================================================


def _DispersionFunc4(wl: ArrayLike, coefs: NDArray[np.floating]) -> NDArray[np.floating]:
    wl = wl * 1e6
    n2 = coefs[0]

    for i in range(1, len(coefs), 4):
        if i + 2 < len(coefs):
            n2 += coefs[i] * wl ** coefs[i + 1] / (wl**2.0 - coefs[i + 2] ** coefs[i + 3])
        else:
            n2 += coefs[i] * wl ** coefs[i + 1]
    return np.sqrt(n2)


def _DispersionFunc1(wl: ArrayLike, coefs: NDArray[np.floating]) -> NDArray[np.floating]:
    wl = wl * 1e6

    n2MinOne = coefs[0]
    for i in range(1, len(coefs), 2):
        n2MinOne += coefs[i] * (wl**2.0) / (wl**2.0 - coefs[i + 1] ** 2.0)
    return np.sqrt(n2MinOne + 1.0)


def _DispersionFunc2(wl: ArrayLike, coefs: NDArray[np.floating]) -> NDArray[np.floating]:
    wl = wl * 1e6

    n2MinOne = coefs[0]
    for i in range(1, len(coefs), 2):
        n2MinOne += coefs[i] * (wl**2.0) / (wl**2.0 - coefs[i + 1])
    return np.sqrt(n2MinOne + 1.0)


def _DisperisonFuncFormula1(wl: ArrayLike, coefs: NDArray[np.floating]) -> NDArray[np.floating]:
    wl = wl * 1e6
    n2 = coefs[0] + 1.0
    for i in range(1, len(coefs), 2):
        n2 += coefs[i] * wl**2.0 / (wl**2.0 - coefs[i + 1] ** 2.0)
    return np.sqrt(n2)


def _DisperisonFuncFormula2(wl: ArrayLike, coefs: NDArray[np.floating]) -> NDArray[np.floating]:
    wl = wl * 1e6
    n2 = coefs[0] + 1.0
    for i in range(1, len(coefs), 2):
        n2 += coefs[i] * wl**2.0 / (wl**2.0 - coefs[i + 1])
    return np.sqrt(n2)


def _DisperisonFuncFormula5(wl: ArrayLike, coefs: NDArray[np.floating]) -> NDArray[np.floating]:
    wl = wl * 1e6
    n = coefs[0]
    for i in range(1, len(coefs), 2):
        n += coefs[i] * (wl ** coefs[i + 1])
    return n
