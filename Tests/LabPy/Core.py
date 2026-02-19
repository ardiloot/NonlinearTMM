"""This module contains low-level methods and classes for LabPy package."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from time import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

# ===============================================================================
# Methods
# ===============================================================================


def Norm(vector: NDArray[np.complexfloating]) -> NDArray[np.floating] | np.floating:
    """Calculates vector 2nd order norm

    Args:
        vector (numpy.array): input x-array

    Returns:
        np.array: vector norm

    """
    if len(vector.shape) > 1:
        if vector.shape[1] != 3:
            raise ValueError("Only vectors with length 3 supported.")
        return np.sqrt(
            abs(vector[:, 0]) ** 2.0 + abs(vector[:, 1]) ** 2.0 + abs(vector[:, 2]) ** 2.0, dtype=complex
        ).real
    else:
        if len(vector) != 3:
            raise ValueError("Only vectors with length 3 supported.")
        return np.sqrt(abs(vector[0]) ** 2.0 + abs(vector[1]) ** 2.0 + abs(vector[2]) ** 2.0, dtype=complex).real


def PickPoints(
    x: ArrayLike,
    n: int,
    fr: float = -math.inf,
    to: float = math.inf,
) -> NDArray[np.intp]:
    """Picks n points from array x, where fr <= x <= to

    Args:
        x (numpy.array): input x-array (must be sorted)
        n (int): number of points to pick
        fr (float, optional): requirement fr <= x
        to (float, optional): requirement x >= to

    Returns:
        np.array: array of picked indices

    """

    indexFr = bisect_left(x, fr)
    indexTo = bisect_right(x, to)
    pointsTotal = indexTo - indexFr

    if pointsTotal < n:
        n = pointsTotal

    if n < 1:
        return np.array([])

    res = np.zeros(n, dtype=int)

    res[0] = indexFr
    pointsSel = 1

    pointsDensity = n / pointsTotal

    for i in range(indexFr + 1, indexTo):
        shouldBeSelected = (i - indexFr + 1) * pointsDensity
        # print "cond", shouldBeSelected, pointsSel, int(shouldBeSelected + 1e-10)
        if pointsSel < int(shouldBeSelected + 1e-10):
            res[pointsSel] = i
            pointsSel += 1

    assert pointsSel == n
    return res


def complex_quadrature(
    func: Callable[[float], complex],
    a: float,
    b: float,
    **kwargs: Any,
) -> tuple[complex, tuple[Any, ...], tuple[Any, ...]]:
    """Integrates complex function func from a to b

    Args:
        func (function): complex function to integrate
        a (float): integration parameter
        b (float): integration parameter
        kwargs (dict, optional): optimal parameters for quad

    Returns:
        tuple: (complex integration value, real integration value, imaginary integral value)

    """

    def real_func(x: float) -> float:
        return np.real(func(x))

    def imag_func(x: float) -> float:
        return np.imag(func(x))

    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j * imag_integral[0], real_integral[1:], imag_integral[1:])


def FindZeros(
    func: Callable[[complex], complex],
    dfunc: Callable[[complex], complex],
    a: complex,
    b: complex,
    zerosMaxDepth: int = 5,
    disp: bool = True,
    **kwargs: Any,
) -> list[complex]:
    """Finds all zeros of function (in range a to b) by argument principle method

    Args:
        argPrintcipleFunc (function): function of argument principle
        a (complex): integration parameter
        b (complex): integration parameter
        zerosMaxDepth (int): maximum recursion depth
        disp (bool): display debug data
        kwargs (dict, optional): optimal parameters for integration

    Returns:
        list of complex: the list of zeros

    """

    def _ZerosInBox(ax, bx):
        start = time()

        a1x, a2x = min(ax.real, bx.real), max(ax.real, bx.real)
        b1x, b2x = min(ax.imag, bx.imag), max(ax.imag, bx.imag)

        func1 = lambda x: argPrintcipleFunc(a1x + 1.0j * x)
        func2 = lambda x: argPrintcipleFunc(x + 1.0j * b2x)
        func3 = lambda x: argPrintcipleFunc(a2x + 1.0j * x)
        func4 = lambda x: argPrintcipleFunc(x + 1.0j * b1x)

        integArgs = kwargs
        integral = 1.0j * complex_quadrature(func1, b1x, b2x, **integArgs)[0]
        integral += complex_quadrature(func2, a1x, a2x, **integArgs)[0]
        integral += 1.0j * complex_quadrature(func3, b2x, b1x, **integArgs)[0]
        integral += complex_quadrature(func4, a2x, a1x, **integArgs)[0]

        res = -integral / (2.0j * math.pi)

        if disp:
            print("zeros in box", ax, bx, res)
            print("time for box:", time() - start)

        return res

    def _DoSearch(a, b, depth=0):
        zerosRaw = _ZerosInBox(a, b).real
        print("DoSearch", a, b, depth, "zeros", zerosRaw)
        zeros = int(round(zerosRaw))
        if zeros <= 0:
            return []

        if depth > zerosMaxDepth:
            return [(a + b) / 2.0]

        a1, a3 = min(a.real, b.real), max(a.real, b.real)
        b1, b3 = min(a.imag, b.imag), max(a.imag, b.imag)
        a2, b2 = (a1 + a3) / 2.0, (b1 + b3) / 2.0

        res = []
        res += _DoSearch(complex(a1, b1), complex(a2, b2), depth + 1)
        res += _DoSearch(complex(a2, b1), complex(a3, b2), depth + 1)
        res += _DoSearch(complex(a1, b2), complex(a2, b3), depth + 1)
        res += _DoSearch(complex(a2, b2), complex(a3, b3), depth + 1)
        return res

    argPrintcipleFunc = lambda x: dfunc(x) / func(x)
    res = _DoSearch(a, b, 0)
    return res


def ExtraInterpolate(xs: ArrayLike, ys: ArrayLike, x: float) -> float:
    if len(xs) == 1:
        return ys[0]
    elif len(xs) == 2:
        return (x * ys[0] - x * ys[1] + xs[0] * ys[1] - xs[1] * ys[0]) / (xs[0] - xs[1])
    elif len(xs) == 3:
        return (
            (
                (ys[2] * xs[1] - ys[1] * xs[2] + x * (ys[1] - ys[2])) * xs[0] ** 2
                + (-ys[2] * xs[1] ** 2 + ys[1] * xs[2] ** 2 - x**2 * (ys[1] - ys[2])) * xs[0]
                + (ys[0] * xs[2] - x * (ys[0] - ys[2])) * xs[1] ** 2
                + (-ys[0] * xs[2] ** 2 + x**2 * (ys[0] - ys[2])) * xs[1]
                - xs[2] * x * (ys[0] - ys[1]) * (x - xs[2])
            )
            / (xs[1] - xs[2])
            / (xs[0] - xs[2])
            / (xs[0] - xs[1])
        )
    else:
        raise NotImplementedError(f"len = {len(xs)}")


def RotationMatrixX(phi: float) -> NDArray[np.floating]:
    """Rotation matrix around X-axis

    Args:
        phi (float): roatation angle

    Returns:
        numpy array (3, 3): rotation matrix

    """
    res = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]], dtype=float)
    return res


def RotationMatrixY(phi: float) -> NDArray[np.floating]:
    """Rotation matrix around Y-axis

    Args:
        phi (float): roatation angle

    Returns:
        numpy array (3, 3): rotation matrix

    """
    res = np.array([[np.cos(phi), 0.0, np.sin(phi)], [0.0, 1.0, 0.0], [-np.sin(phi), 0.0, np.cos(phi)]], dtype=float)
    return res


def RotationMatrixZ(phi: float) -> NDArray[np.floating]:
    """Rotation matrix around Z-axis

    Args:
        phi (float): roatation angle

    Returns:
        numpy array (3, 3): rotation matrix

    """
    res = np.array([[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return res


def CartesianProduct(
    arrays: list[ArrayLike],
    out: NDArray | None = None,
) -> NDArray:
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n // arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        CartesianProduct(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m : (j + 1) * m, 1:] = out[0:m, 1:]
    return out


# ===============================================================================
# Classes
# ===============================================================================


class ParamsBaseClass:
    """Inheritance of this class will provide SetParams and GetParams methods.

    Args:
        kwargs: optional arguments, currently unused

    """

    def __init__(self, params: list[str] | None = None, **kwargs: Any) -> None:
        if params is not None:
            self._params = params

        if not hasattr(self, "friendlyNames"):
            self.friendlyNames = self._params[:]

        self.SetParams(**kwargs)

    def SetParams(self, **kwargs: Any) -> None:
        """Sets parameters from kwargs dictionary.

        Args:
            kwargs (dict): dictionary of parameters

        Raises:
            ValueError: if parameter name is not in _params list

        """

        for k, v in kwargs.items():
            if k not in self._params:
                raise ValueError(f"Unknown kwargs: {k}")
            setattr(self, k, v)

    def GetParams(self, asList: bool = False) -> dict[str, Any] | list[tuple[str, str, Any]]:
        """Returns values of parameters in _params.

        Args:
            asList (bool, optional = False): Whether return params as list or as dict.

        Returns:
            dict: dictionary of parameters and values

        """
        if asList:
            res = []
            for param, friendlyName in zip(self._params, self.friendlyNames):
                res.append((param, friendlyName, getattr(self, param)))

        else:
            res = {}
            for param in self._params:
                res[param] = getattr(self, param)
        return res
