"""Type stubs for the NonlinearTMM package."""

from __future__ import annotations

from NonlinearTMM._Material import Material as Material
from NonlinearTMM._SecondOrderNLTMMCython import NonlinearTMM as NonlinearTMM
from NonlinearTMM._SecondOrderNLTMMCython import NonlinearTMM as TMM
from NonlinearTMM._SecondOrderNLTMMCython import SecondOrderNLTMM as SecondOrderNLTMM
from NonlinearTMM._SecondOrderNLTMMCython import _Chi2Tensor as _Chi2Tensor
from NonlinearTMM._SecondOrderNLTMMCython import _FieldsZ as _FieldsZ
from NonlinearTMM._SecondOrderNLTMMCython import _FieldsZX as _FieldsZX
from NonlinearTMM._SecondOrderNLTMMCython import _HomogeneousWave as _HomogeneousWave
from NonlinearTMM._SecondOrderNLTMMCython import _Intensities as _Intensities
from NonlinearTMM._SecondOrderNLTMMCython import _NonlinearLayer as _NonlinearLayer
from NonlinearTMM._SecondOrderNLTMMCython import (
    _SecondOrderNLIntensities as _SecondOrderNLIntensities,
)
from NonlinearTMM._SecondOrderNLTMMCython import (
    _SweepResultNonlinearTMM as _SweepResultNonlinearTMM,
)
from NonlinearTMM._SecondOrderNLTMMCython import (
    _SweepResultSecondOrderNLTMM as _SweepResultSecondOrderNLTMM,
)
from NonlinearTMM._SecondOrderNLTMMCython import _Wave as _Wave
from NonlinearTMM._SecondOrderNLTMMCython import (
    _WaveSweepResultNonlinearTMM as _WaveSweepResultNonlinearTMM,
)
from NonlinearTMM._SecondOrderNLTMMCython import (
    _WaveSweepResultSecondOrderNLTMM as _WaveSweepResultSecondOrderNLTMM,
)

__all__ = ["Material", "TMM", "NonlinearTMM", "SecondOrderNLTMM"]
