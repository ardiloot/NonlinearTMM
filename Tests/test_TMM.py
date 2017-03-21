import pytest
import numpy as np
import LabPy
from NonlinearTMM import TMM, Material

@pytest.fixture(params = [400e-9, 800e-9])
def wl(request):
    return request.param

@pytest.fixture(params = [1.5, 2.0])
def nPrism(request):
    return request.param

@pytest.fixture(params = [1.0, 1.3])
def nDielectric(request):
    return request.param

@pytest.fixture(params = ["p", "s"])
def pol(request):
    return request.param

@pytest.fixture(params = [50e-9, 100e-9])
def metalD(request):
    return request.param

@pytest.fixture(params = [0.1, 1.0])
def I0(request):
    return request.param

@pytest.fixture(params = [1.0, 1.0 + 3j, 1j])
def overrideE0(request):
    return request.param

@pytest.fixture(params = [0.0, 0.9, 1.4])    
def betaTest(request):
    return request.param
    
@pytest.fixture()
def tmms1(request, wl, nPrism, nDielectric, pol, metalD, I0):
    prism = LabPy.Material("Static", n = nPrism)
    metal = LabPy.Material("main/Ag/Johnson")
    dielectric = LabPy.Material("Static", n = nDielectric)
    tmmCpp, tmmPy = GetTMMs(wl, pol, I0, prism, metal, dielectric, metalD, None)
   
    return tmmCpp, tmmPy

@pytest.fixture()
def tmms2(request, nPrism, nDielectric, pol, overrideE0):
    wl = 532e-9
    metalD = 50e-9
    I0 = 1.0
    
    prism = LabPy.Material("Static", n = nPrism)
    metal = LabPy.Material("main/Ag/Johnson")
    dielectric = LabPy.Material("Static", n = nDielectric)
    tmmCpp, tmmPy = GetTMMs(wl, pol, I0, prism, metal, dielectric, metalD, overrideE0)
   
    return tmmCpp, tmmPy

def test_Powerflows1(tmms1):
    TestPowerFlows(tmms1)

def test_Powerflows2(tmms2):
    TestPowerFlows(tmms2)

def test_Fields1D(tmms2, betaTest):
    tmmCpp, tmmPy = tmms2
    zs = np.linspace(-200e-9, 200e-9, 20)
    
    tmmCpp.Solve(beta = betaTest)
    tmmPy.Solve(beta = betaTest)
    
    resEPy, resHPy = tmmPy.GetFields(zs)
    resCpp = tmmCpp.GetFields(zs)
    
    np.testing.assert_allclose(resEPy, resCpp.E)
    np.testing.assert_allclose(resHPy, resCpp.H)


def test_Fields2D(tmms2, betaTest):
    tmmCpp, tmmPy = tmms2
    xs = np.linspace(-300e-9, 250e-9, 200)
    zs = np.linspace(-200e-9, 200e-9, 200)
    
    tmmCpp.Solve(beta = betaTest)
    tmmPy.Solve(beta = betaTest)
    
    resEPy, resHPy = tmmPy.GetFields2D(zs, xs)
    resCpp = tmmCpp.GetFields2D(zs, xs)

    if tmmPy.pol == "s":
        np.testing.assert_allclose(resEPy[:, :, 1], resCpp.Ey)
        np.testing.assert_allclose(resHPy[:, :, 0], resCpp.Hx)
        np.testing.assert_allclose(resHPy[:, :, 2], resCpp.Hz)
    else:
        np.testing.assert_allclose(resHPy[:, :, 1], resCpp.Hy)
        np.testing.assert_allclose(resEPy[:, :, 0], resCpp.Ex)
        np.testing.assert_allclose(resEPy[:, :, 2], resCpp.Ez)
        
        
    
#------------------------------------------------------------------------------ 

def GetTMMs(wl, pol, I0, prism, metal, dielectric, metalD, overrideE0):
    # C++ TMM
    tmmCpp = TMM()
    tmmCpp.SetParams(wl = wl, pol = pol, I0 = I0)
    
    if overrideE0 is not None:
        tmmCpp.SetParams(overrideE0 = True, E0 = overrideE0)
    
    tmmCpp.AddLayer(float("inf"), Material.FromLabPy(prism))
    tmmCpp.AddLayer(metalD, Material.FromLabPy(metal))
    tmmCpp.AddLayer(float("inf"), Material.FromLabPy(dielectric))
    

    # Python TMM
    tmmPy = LabPy._Tmm._NonlinearTmm._NonlinearTmm(wl = wl, pol = pol, I0 = I0, mode = "incident", overrideE0 = overrideE0) # @UndefinedVariable
    tmmPy.AddLayer(float("inf"), prism)
    tmmPy.AddLayer(metalD, metal)
    tmmPy.AddLayer(float("inf"), dielectric)
    tmmPy.SetParams(beta = 0.0)
    
    return tmmCpp, tmmPy

def TestPowerFlows(tmms):
    tmmCpp, tmmPy = tmms
    
    betas = np.linspace(0.0, 1.49, 20)
    sr = tmmCpp.Sweep("beta", betas)
    sweepFunc = np.vectorize(lambda beta: tmmPy.Solve(beta = beta))
    inc, r, t, I, R, T, _ = sweepFunc(betas)

    # Comparison
    np.testing.assert_allclose(sr.inc, inc);
    np.testing.assert_allclose(sr.r, r);
    np.testing.assert_allclose(sr.t, t);
    np.testing.assert_allclose(sr.Ii, I);
    np.testing.assert_allclose(sr.Ir, R);    
    np.testing.assert_allclose(sr.It, T);

