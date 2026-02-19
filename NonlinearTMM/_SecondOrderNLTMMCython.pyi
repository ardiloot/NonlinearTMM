"""Type stubs for the Cython extension module _SecondOrderNLTMMCython.

This module is compiled from ``NonlinearTMM/src/SecondOrderNLTMM.pyx``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# _Chi2Tensor
# ---------------------------------------------------------------------------

class _Chi2Tensor:
    def Update(self, **kwargs: Any) -> None: ...
    def GetChi2Tensor(self) -> NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------

class Material:
    chi2: _Chi2Tensor

    def __init__(self, wls: NDArray[np.float64], ns: NDArray[np.complex128]) -> None: ...
    def __call__(self, wl: float) -> complex: ...
    def GetN(self, wl: float) -> complex: ...
    def IsNonlinear(self) -> bool: ...

# ---------------------------------------------------------------------------
# _Wave
# ---------------------------------------------------------------------------

class _Wave:
    pwr: float
    overrideE0: bool
    E0: float
    w0: float
    Ly: float
    a: float
    nPointsInteg: int
    maxX: float
    dynamicMaxX: bool
    dynamicMaxXCoef: float
    dynamicMaxXAddition: float
    maxPhi: float

    def SetParams(self, **kwargs: Any) -> None: ...
    @property
    def waveType(self) -> str: ...
    @waveType.setter
    def waveType(self, value: str) -> None: ...
    @property
    def xRange(self) -> tuple[float, float]: ...
    @property
    def betas(self) -> NDArray[np.float64]: ...
    @property
    def phis(self) -> NDArray[np.float64]: ...
    @property
    def kxs(self) -> NDArray[np.float64]: ...
    @property
    def kzs(self) -> NDArray[np.complex128]: ...
    @property
    def fieldProfile(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    @property
    def expansionCoefsKx(self) -> NDArray[np.complex128]: ...
    @property
    def beamArea(self) -> float: ...

# ---------------------------------------------------------------------------
# _Intensities
# ---------------------------------------------------------------------------

class _Intensities:
    @property
    def inc(self) -> complex: ...
    @property
    def r(self) -> complex: ...
    @property
    def t(self) -> complex: ...
    @property
    def I(self) -> float: ...  # noqa: E743
    @property
    def R(self) -> float: ...
    @property
    def T(self) -> float: ...

# ---------------------------------------------------------------------------
# _SweepResultNonlinearTMM
# ---------------------------------------------------------------------------

class _SweepResultNonlinearTMM:
    @property
    def inc(self) -> NDArray[np.complex128]: ...
    @property
    def r(self) -> NDArray[np.complex128]: ...
    @property
    def t(self) -> NDArray[np.complex128]: ...
    @property
    def Ii(self) -> NDArray[np.float64]: ...
    @property
    def Ir(self) -> NDArray[np.float64]: ...
    @property
    def It(self) -> NDArray[np.float64]: ...
    @property
    def Ia(self) -> NDArray[np.float64]: ...
    @property
    def enh(self) -> NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# _WaveSweepResultNonlinearTMM
# ---------------------------------------------------------------------------

class _WaveSweepResultNonlinearTMM:
    @property
    def Pi(self) -> NDArray[np.float64]: ...
    @property
    def Pr(self) -> NDArray[np.float64]: ...
    @property
    def Pt(self) -> NDArray[np.float64]: ...
    @property
    def enh(self) -> NDArray[np.float64]: ...
    @property
    def beamArea(self) -> NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# _FieldsZ
# ---------------------------------------------------------------------------

class _FieldsZ:
    @property
    def E(self) -> NDArray[np.complex128]: ...
    @property
    def H(self) -> NDArray[np.complex128]: ...

# ---------------------------------------------------------------------------
# _FieldsZX
# ---------------------------------------------------------------------------

class _FieldsZX:
    @property
    def Ex(self) -> NDArray[np.complex128] | None: ...
    @property
    def Ey(self) -> NDArray[np.complex128] | None: ...
    @property
    def Ez(self) -> NDArray[np.complex128] | None: ...
    @property
    def Hx(self) -> NDArray[np.complex128] | None: ...
    @property
    def Hy(self) -> NDArray[np.complex128] | None: ...
    @property
    def Hz(self) -> NDArray[np.complex128] | None: ...
    @property
    def EN(self) -> NDArray[np.float64]: ...
    @property
    def HN(self) -> NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# _HomogeneousWave
# ---------------------------------------------------------------------------

class _HomogeneousWave:
    def GetMainFields(self, z: float) -> NDArray[np.complex128]: ...
    @property
    def kzF(self) -> complex: ...
    @property
    def kx(self) -> float: ...

# ---------------------------------------------------------------------------
# _NonlinearLayer
# ---------------------------------------------------------------------------

class _NonlinearLayer:
    d: float
    hw: _HomogeneousWave

    def GetIntensity(self, z: float) -> float: ...
    def GetAbsorbedIntensity(self) -> float: ...
    def GetSrcIntensity(self) -> float: ...

# ---------------------------------------------------------------------------
# NonlinearTMM
# ---------------------------------------------------------------------------

class NonlinearTMM:
    wl: float
    beta: float
    pol: str
    I0: float
    overrideE0: bool
    E0: complex
    mode: str

    def __init__(
        self,
        *,
        initStruct: bool = ...,
        parent: object = ...,
        **kwargs: Any,
    ) -> None: ...
    @property
    def materialsCache(self) -> list[Material]: ...
    @property
    def layers(self) -> list[_NonlinearLayer]: ...
    @property
    def wave(self) -> _Wave: ...
    def AddLayer(self, d: float, material: Material) -> None: ...
    def SetParams(self, **kwargs: Any) -> None: ...
    def Solve(self, **kwargs: Any) -> None: ...
    def GetIntensities(self) -> _Intensities: ...
    def Sweep(
        self,
        paramStr: str,
        values: NDArray[np.float64],
        layerNr: int = ...,
        layerZ: float = ...,
        outPwr: bool = ...,
        outAbs: bool = ...,
        outEnh: bool = ...,
    ) -> _SweepResultNonlinearTMM: ...
    def GetFields(
        self,
        zs: NDArray[np.float64],
        dir: str = ...,
    ) -> _FieldsZ: ...
    def GetFields2D(
        self,
        zs: NDArray[np.float64],
        xs: NDArray[np.float64],
        dir: str = ...,
    ) -> _FieldsZX: ...
    def GetAbsorbedIntensity(self) -> float: ...
    def GetEnhancement(self, layerNr: int, z: float = ...) -> float: ...
    def WaveGetPowerFlows(
        self,
        layerNr: int,
        x0: float = ...,
        x1: float = ...,
        z: float = ...,
    ) -> tuple[float, float]: ...
    def WaveGetEnhancement(self, layerNr: int, z: float = ...) -> float: ...
    def WaveSweep(
        self,
        paramStr: str,
        values: NDArray[np.float64],
        layerNr: int = ...,
        layerZ: float = ...,
        outPwr: bool = ...,
        outR: bool = ...,
        outT: bool = ...,
        outEnh: bool = ...,
    ) -> _WaveSweepResultNonlinearTMM: ...
    def WaveGetFields2D(
        self,
        zs: NDArray[np.float64],
        xs: NDArray[np.float64],
        dirStr: str = ...,
    ) -> _FieldsZX: ...

# ---------------------------------------------------------------------------
# _SecondOrderNLIntensities
# ---------------------------------------------------------------------------

class _SecondOrderNLIntensities:
    @property
    def P1(self) -> _Intensities: ...
    @property
    def P2(self) -> _Intensities: ...
    @property
    def Gen(self) -> _Intensities: ...

# ---------------------------------------------------------------------------
# _SweepResultSecondOrderNLTMM
# ---------------------------------------------------------------------------

class _SweepResultSecondOrderNLTMM:
    @property
    def P1(self) -> _SweepResultNonlinearTMM: ...
    @property
    def P2(self) -> _SweepResultNonlinearTMM: ...
    @property
    def Gen(self) -> _SweepResultNonlinearTMM: ...
    @property
    def wlsGen(self) -> NDArray[np.float64]: ...
    @property
    def betasGen(self) -> NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# _WaveSweepResultSecondOrderNLTMM
# ---------------------------------------------------------------------------

class _WaveSweepResultSecondOrderNLTMM:
    @property
    def P1(self) -> _WaveSweepResultNonlinearTMM: ...
    @property
    def P2(self) -> _WaveSweepResultNonlinearTMM: ...
    @property
    def Gen(self) -> _WaveSweepResultNonlinearTMM: ...
    @property
    def wlsGen(self) -> NDArray[np.float64]: ...
    @property
    def betasGen(self) -> NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# SecondOrderNLTMM
# ---------------------------------------------------------------------------

class SecondOrderNLTMM:
    deltaWlSpdc: float
    solidAngleSpdc: float
    deltaThetaSpdc: float

    def __init__(self, mode: str = ..., **kwargs: Any) -> None: ...
    @property
    def P1(self) -> NonlinearTMM: ...
    @property
    def P2(self) -> NonlinearTMM: ...
    @property
    def Gen(self) -> NonlinearTMM: ...
    def SetParams(self, **kwargs: Any) -> None: ...
    def AddLayer(self, d: float, material: Material) -> None: ...
    def Solve(self) -> None: ...
    def UpdateGenParams(self) -> None: ...
    def GetIntensities(self) -> _SecondOrderNLIntensities: ...
    def Sweep(
        self,
        paramStr: str,
        valuesP1: NDArray[np.float64],
        valuesP2: NDArray[np.float64],
        layerNr: int = ...,
        layerZ: float = ...,
        outPwr: bool = ...,
        outAbs: bool = ...,
        outEnh: bool = ...,
        outP1: bool = ...,
        outP2: bool = ...,
        outGen: bool = ...,
    ) -> _SweepResultSecondOrderNLTMM: ...
    def WaveGetPowerFlows(
        self,
        layerNr: int,
        x0: float = ...,
        x1: float = ...,
        z: float = ...,
    ) -> tuple[float, float]: ...
    def WaveSweep(
        self,
        paramStr: str,
        valuesP1: NDArray[np.float64],
        valuesP2: NDArray[np.float64],
        layerNr: int = ...,
        layerZ: float = ...,
        outPwr: bool = ...,
        outR: bool = ...,
        outT: bool = ...,
        outEnh: bool = ...,
        outP1: bool = ...,
        outP2: bool = ...,
        outGen: bool = ...,
    ) -> _WaveSweepResultSecondOrderNLTMM: ...
    def WaveGetFields2D(
        self,
        zs: NDArray[np.float64],
        xs: NDArray[np.float64],
        dirStr: str = ...,
    ) -> _FieldsZX: ...
