import pytest
import numpy as np
from NonlinearTMM import TMM, Material

@pytest.fixture()
def tmm():
    wl = 532e-9
    metalD = 60e-9

    prism = Material.Static(1.5)
    metal = Material.Static(0.054007 + 3.4290j)
    dielectric = Material.Static(1.0)
    
    res = TMM(wl = wl)
    res.AddLayer(float("inf"), prism)
    res.AddLayer(metalD, metal)
    res.AddLayer(float("inf"), dielectric)
    
    return res

@pytest.fixture(params = ["p", "s"])
def pol(request):
    return request.param

@pytest.fixture(params = ["tukey", "gaussian", "planewave"])
def waveType(request):
    return request.param


def test_WaveSweep(tmm, pol, waveType):
    betas = np.linspace(0.0, 0.5, 15)
    enhLayer = 2
    enhZ = 0.0
    Ly = 1e-3
    pwr = 10e-3
    w0 = 500e-6
    waveParams = {"waveType": waveType, "pwr": pwr, "w0": w0, "Ly": Ly, "nPointsInteg": 300, "dynamicMaxXCoef": 10.0}
    
    tmm.SetParams(pol = pol)
    tmm.wave.SetParams(**waveParams)

    # Ordinary sweep
    tmm.I0 = pwr / (w0 * Ly)
    sr = tmm.Sweep("beta", betas, layerNr = enhLayer, layerZ = enhZ, outEnh = True)
    
    # Waves sweep
    srW = tmm.WaveSweep("beta", betas, layerNr = enhLayer, layerZ = enhZ, outEnh = True)    
    
    
    # Test incident power
    # Convert intensity to power
    pwrIncident = sr.Ii * srW.beamArea
    np.testing.assert_allclose(srW.Pi, pwrIncident, atol = 1e-4)
        
    # Test refl, transmission, enhancment
    np.testing.assert_allclose(srW.Pr / srW.Pi, sr.Ir / sr.Ii, atol = 1e-4)
    np.testing.assert_allclose(srW.Pt / srW.Pi, sr.It / sr.Ii, atol = 1e-4)
    np.testing.assert_allclose(srW.enh, sr.enh, atol = 5e-4)
    