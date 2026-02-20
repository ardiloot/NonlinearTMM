"""Tests for Material constructors and basic error paths."""

from __future__ import annotations

import numpy as np
import pytest

from NonlinearTMM import TMM, Material

# ---------------------------------------------------------------------------
# Material.Static
# ---------------------------------------------------------------------------


class TestMaterialStatic:
    def test_real_refractive_index(self):
        mat = Material.Static(1.5)
        assert mat.GetN(532e-9) == pytest.approx(1.5)

    def test_complex_refractive_index(self):
        n = 0.054007 + 3.4290j
        mat = Material.Static(n)
        assert mat.GetN(532e-9) == pytest.approx(n)

    def test_wavelength_independent(self):
        mat = Material.Static(2.0)
        n400 = mat.GetN(400e-9)
        n800 = mat.GetN(800e-9)
        assert n400 == pytest.approx(n800)


# ---------------------------------------------------------------------------
# Material(wls, ns) constructor
# ---------------------------------------------------------------------------


class TestMaterialConstructor:
    def test_interpolation(self):
        wls = np.array([400e-9, 600e-9, 800e-9])
        ns = np.array([1.5, 1.4, 1.3], dtype=complex)
        mat = Material(wls, ns)
        # Interior point should be interpolated
        n_mid = mat.GetN(600e-9)
        assert n_mid == pytest.approx(1.4)
        # Value between endpoints
        n_500 = mat.GetN(500e-9)
        assert 1.3 < np.real(n_500) < 1.5

    def test_complex_ns(self):
        wls = np.array([400e-9, 800e-9])
        ns = np.array([1.5 + 0.1j, 1.3 + 0.2j], dtype=complex)
        mat = Material(wls, ns)
        n = mat.GetN(400e-9)
        assert np.real(n) == pytest.approx(1.5)
        assert np.imag(n) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Error path tests
# ---------------------------------------------------------------------------


class TestTMMErrors:
    def test_invalid_polarization(self):
        with pytest.raises((ValueError, RuntimeError)):
            TMM(wl=532e-9, pol="x", I0=1.0)

    def test_sweep_no_layers(self):
        tmm = TMM(wl=532e-9, pol="p", I0=1.0)
        with pytest.raises((ValueError, RuntimeError)):
            betas = np.linspace(0, 1, 10)
            tmm.Sweep("beta", betas)


# ---------------------------------------------------------------------------
# Public API surface test
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_all_exports_importable(self):
        import NonlinearTMM

        for name in NonlinearTMM.__all__:
            assert hasattr(NonlinearTMM, name), f"{name} listed in __all__ but not importable"
