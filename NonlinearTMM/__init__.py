from __future__ import annotations

from . import _SecondOrderNLTMMCython
from ._Material import Material

__all__ = [
    "Material",
    "TMM",
    "NonlinearTMM",
    "SecondOrderNLTMM",
]

# Public aliases (previously in _NonlinearTMM.py and _SecondOrderNLTMM.py)
TMM = _SecondOrderNLTMMCython.NonlinearTMM
NonlinearTMM = _SecondOrderNLTMMCython.NonlinearTMM
SecondOrderNLTMM = _SecondOrderNLTMMCython.SecondOrderNLTMM

# Helper classes
_Chi2Tensor = _SecondOrderNLTMMCython._Chi2Tensor
_Wave = _SecondOrderNLTMMCython._Wave
_Intensities = _SecondOrderNLTMMCython._Intensities
_SweepResultNonlinearTMM = _SecondOrderNLTMMCython._SweepResultNonlinearTMM
_WaveSweepResultNonlinearTMM = _SecondOrderNLTMMCython._WaveSweepResultNonlinearTMM
_FieldsZ = _SecondOrderNLTMMCython._FieldsZ
_FieldsZX = _SecondOrderNLTMMCython._FieldsZX
_HomogeneousWave = _SecondOrderNLTMMCython._HomogeneousWave
_NonlinearLayer = _SecondOrderNLTMMCython._NonlinearLayer
_SecondOrderNLIntensities = _SecondOrderNLTMMCython._SecondOrderNLIntensities
_SweepResultSecondOrderNLTMM = _SecondOrderNLTMMCython._SweepResultSecondOrderNLTMM
_WaveSweepResultSecondOrderNLTMM = _SecondOrderNLTMMCython._WaveSweepResultSecondOrderNLTMM
