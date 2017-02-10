import numpy as np
from  NonlinearTMM import _SecondOrderNLTMMCython  # @UnresolvedImport

__all__ = ["Material"]

class Material(_SecondOrderNLTMMCython.Material):# @UndefinedVariable
    @staticmethod
    def Static(n):
        wls = np.array([-1.0, 1.0])
        ns = np.array([n, n], dtype = complex)
        res = Material(wls, ns)
        return res
    
    @staticmethod
    def FromLabPy(materialLabPy):
        if materialLabPy.materialFile == "Static":
            wls = np.array([-1.0, 1.0])
            n = materialLabPy.n + 1.0j * materialLabPy.k
            ns = np.array([n, n], dtype = complex)
        else:
            wls = np.ascontiguousarray(materialLabPy.wlExp)
            if materialLabPy.kExp is None:
                ns = np.ascontiguousarray(materialLabPy.nExp, dtype = complex)
            else:
                ns = np.ascontiguousarray(materialLabPy.nExp + 1.0j * materialLabPy.kExp)
        res = Material(wls, ns)
        res._materialLabPy = materialLabPy
        return res
   
if __name__ == "__main__":
    pass