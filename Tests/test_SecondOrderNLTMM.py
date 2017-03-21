import numpy as np
import pytest
import LabPy
from NonlinearTMM import Material, SecondOrderNLTMM
    
@pytest.fixture(params = ["p", "s"])
def polP1(request):
    return request.param

@pytest.fixture(params = ["p", "s"])
def polP2(request):
    return request.param

@pytest.fixture(params = ["p", "s"])
def polGen(request):
    return request.param

@pytest.fixture(params = ["sfg", "dfg"])
def process(request):
    return request.param
    
@pytest.fixture()
def tmms(polP1, polP2, polGen, process):
    wlP1 = 700e-9
    wlP2 = 1500e-9
    I0P1 = 1.0
    I0P2 = 2.0
    dValues = {"d11": 1e-12, "d22": 2e-12, "d33": 3e-12, "d12": 4e-12, "d23": 5e-12}
    
    chi2 = LabPy.Chi2Tensor(**dValues)
    prism = LabPy.Material("Static", n = 1.5)
    metal = LabPy.Material("main/Ag/Johnson")
    crystal = LabPy.Material("special/gaussian_test")
    dielectric = LabPy.Material("Static", n = 1.0)
    crystalD = 1000e-6
    metalD = 5e-9
    
    tmmPy = LabPy.SecondOrderNLTmm(wlP1 = wlP1, wlP2 = wlP2, \
        polP1 = polP1, polP2 = polP2, polGen = polGen, \
        I0P1 = I0P1, I0P2 = I0P2, process = process)
    
    tmmPy.AddLayer(float("inf"), prism)
    tmmPy.AddLayer(metalD, metal)
    tmmPy.AddLayer(crystalD, crystal, chi2)
    tmmPy.AddLayer(float("inf"), dielectric)
    
    # C++
    prismCpp = Material.FromLabPy(prism)
    metalCpp = Material.FromLabPy(metal)
    crystalCpp = Material.FromLabPy(crystal)
    dielectricCpp = Material.FromLabPy(dielectric)
    crystalCpp.chi2.Update(**dValues)
    
    tmmCpp = SecondOrderNLTMM(process)
    tmmCpp.P1.SetParams(wl = wlP1, pol = polP1, I0 = I0P1)
    tmmCpp.P2.SetParams(wl = wlP2, pol = polP2, I0 = I0P2)
    tmmCpp.Gen.SetParams(pol = polGen)
    
    tmmCpp.AddLayer(float("inf"), prismCpp)
    tmmCpp.AddLayer(metalD, metalCpp)
    tmmCpp.AddLayer(crystalD, crystalCpp)
    tmmCpp.AddLayer(float("inf"), dielectricCpp)
    
    return tmmPy, tmmCpp
    
def test_PowerFlows(tmms):    
    tmmPy, tmmCpp = tmms
    betas = np.linspace(0.0, 0.99, 20)
    enhLayer, enhDist = 2, 20e-9
    srPy = tmmPy.Sweep(["betaP1", "betaP2"], [betas, betas], enhpos = (enhLayer, enhDist))
    srCpp = tmmCpp.Sweep("beta", betas, betas, outEnh = True, outAbs = True, layerNr = enhLayer, layerZ = enhDist)
    
    # P1
    np.testing.assert_allclose(srPy["iP1"], srCpp.P1.inc)
    np.testing.assert_allclose(srPy["rP1"], srCpp.P1.r)
    np.testing.assert_allclose(srPy["tP1"], srCpp.P1.t)
    np.testing.assert_allclose(srPy["IP1"].real, srCpp.P1.Ii)
    np.testing.assert_allclose(srPy["RP1"].real, srCpp.P1.Ir)
    np.testing.assert_allclose(srPy["TP1"].real, srCpp.P1.It)
    np.testing.assert_allclose(srPy["AP1"].real, srCpp.P1.Ia)
    np.testing.assert_allclose(srPy["enhP1"].real, srCpp.P1.enh)
    
    # P2
    np.testing.assert_allclose(srPy["iP2"], srCpp.P2.inc)
    np.testing.assert_allclose(srPy["rP2"], srCpp.P2.r)
    np.testing.assert_allclose(srPy["tP2"], srCpp.P2.t)
    np.testing.assert_allclose(srPy["IP2"].real, srCpp.P2.Ii)
    np.testing.assert_allclose(srPy["RP2"].real, srCpp.P2.Ir)
    np.testing.assert_allclose(srPy["TP2"].real, srCpp.P2.It)
    np.testing.assert_allclose(srPy["AP2"].real, srCpp.P2.Ia)
    np.testing.assert_allclose(srPy["enhP2"].real, srCpp.P2.enh)
    
    # Gen    
    np.testing.assert_allclose(srPy["iGen"], srCpp.Gen.inc)
    np.testing.assert_allclose(srPy["rGen"], srCpp.Gen.r)
    np.testing.assert_allclose(srPy["tGen"], srCpp.Gen.t)
    np.testing.assert_allclose(srPy["IGen"].real, srCpp.Gen.Ii)
    np.testing.assert_allclose(srPy["RGen"].real, srCpp.Gen.Ir)
    np.testing.assert_allclose(srPy["TGen"].real, srCpp.Gen.It)
    np.testing.assert_allclose(srPy["AGen"].real, srCpp.Gen.Ia)
    
    