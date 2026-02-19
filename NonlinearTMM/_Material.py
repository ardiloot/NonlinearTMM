from __future__ import annotations

import numpy as np

from NonlinearTMM import _SecondOrderNLTMMCython

__all__ = ["Material"]


class Material(_SecondOrderNLTMMCython.Material):
    __doc__ = _SecondOrderNLTMMCython.Material.__doc__

    @staticmethod
    def Static(n: complex | float) -> Material:
        """Helper method to make material with constant refractive index.

        Parameters
        ----------
        n : float or complex
            Constant value for refractive index

        Returns
        -------
        Material

        Examples
        --------
        >>> mat = Material.Static(1.5)
        >>> mat.GetN(532e-9)
        1.5

        """
        wls = np.array([-1.0, 1.0])
        ns = np.array([n, n], dtype=complex)
        return Material(wls, ns)

    @staticmethod
    def FromLabPy(materialLabPy: object) -> Material:
        """Create a Material from a LabPy Material instance.

        Parameters
        ----------
        materialLabPy : LabPy.Material
            Source material object from the LabPy test helpers.

        Returns
        -------
        Material

        """
        if materialLabPy.materialFile == "Static":
            wls = np.array([-1.0, 1.0])
            n = materialLabPy.n + 1.0j * (materialLabPy.k + materialLabPy.kAdditional)
            ns = np.array([n, n], dtype=complex)
        elif materialLabPy.isFormula:
            wls = np.ascontiguousarray(np.linspace(materialLabPy.wlRange[0], materialLabPy.wlRange[1], 500))
            ns = np.ascontiguousarray(materialLabPy(wls))
        else:
            wls = np.ascontiguousarray(materialLabPy.wlExp)
            if materialLabPy.kExp is None:
                ns = np.ascontiguousarray(materialLabPy.nExp, dtype=complex)
            else:
                ns = np.ascontiguousarray(materialLabPy.nExp + 1.0j * materialLabPy.kExp)
            ns += 1.0j * materialLabPy.kAdditional
        res = Material(wls, ns)
        res._materialLabPy = materialLabPy
        return res
