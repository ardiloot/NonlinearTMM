import pytest
import numpy as np
from NonlinearTMM import SecondOrderNLTMM, Material

@pytest.fixture()
def tmm():
    wlP1 = 1000e-9
    wlP2 = 1000e-9
    polP1 = "s"
    polP2 = "s"
    polGen = "s"
    Ly = 2e-3
    w0P1 = 4000e-6
    w0P2 = 4000e-6
    pwrP1 = 1.1
    E0P2 = 12.0
    
    crystalD = 50e-6
    
    # Define materials
    wlsCrystal = np.array([400e-9, 1100e-9])
    nsCrystal = np.array([1.54, 1.53], dtype = complex)
    prism = Material.Static(1.5)
    crystal = Material(wlsCrystal, nsCrystal)
    dielectric = Material.Static(1.6)
    crystal.chi2.Update(d11 = 1e-12, d22 = 1e-12, d33 = 1e-12)
    
    # Init SecondOrderNLTMM
    tmm = SecondOrderNLTMM()
    tmm.P1.SetParams(wl = wlP1, pol = polP1, beta = 0.0, I0 = pwrP1 / (Ly * w0P1))
    tmm.P2.SetParams(wl = wlP2, pol = polP2, beta = 0.0, overrideE0 = True, E0 = E0P2)
    tmm.Gen.SetParams(pol = polGen)
    
    # Add layers
    tmm.AddLayer(float("inf"), prism)
    tmm.AddLayer(crystalD, crystal)
    tmm.AddLayer(float("inf"), dielectric)

    # Init waves
    waveP1Params = {"waveType": "tukey", "pwr": pwrP1, "w0": w0P1, "Ly": Ly, "dynamicMaxXCoef": 5, "nPointsInteg": 200, "maxPhi": np.radians(10.0)}
    waveP2Params = {"waveType": "planewave", "w0": w0P2, "Ly": Ly, "overrideE0": True, "E0": E0P2}
    tmm.P1.wave.SetParams(**waveP1Params)
    tmm.P2.wave.SetParams(**waveP2Params)
    
    return tmm

@pytest.fixture(params = ["tukey", "gaussian", "planewave"])
def waveType(request):
    return request.param

@pytest.fixture(params = ["p", "s"])
def polGen(request):
    return request.param

@pytest.fixture(params = ["p", "s"])
def polP1(request):
    return request.param

@pytest.fixture(params = ["p", "s"])
def polP2(request):
    return request.param

def test_SecondOrderNLTMMWaveSweep(tmm, waveType, polP1, polP2, polGen):
    betas = np.linspace(0.0, 0.99, 10)
    
    tmm.P1.pol = polP1
    tmm.P2.pol = polP2
    tmm.Gen.pol = polGen
    tmm.P1.wave.waveType = waveType
    
    sr = tmm.Sweep("beta", betas, betas, outP1 = False, outP2 = False)
    srW = tmm.WaveSweep("beta", betas, betas, outP1 = False, outP2 = False)

    np.testing.assert_allclose(srW.Gen.Pr, sr.Gen.Ir * srW.Gen.beamArea, rtol = 1e-2)
    np.testing.assert_allclose(srW.Gen.Pt, sr.Gen.It * srW.Gen.beamArea, rtol = 6e-2)
    
    